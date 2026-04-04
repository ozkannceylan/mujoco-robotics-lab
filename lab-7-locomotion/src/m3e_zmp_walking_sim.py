#!/usr/bin/env python3
"""M3e — ZMP Preview Control Walking Simulation.

Executes the LIPM + ZMP preview control walking plan in MuJoCo:
  1. Generate footstep plan + ZMP reference + CoM trajectory (from planner)
  2. At each control tick, compute swing foot target from swing_trajectory
  3. Solve whole-body IK (Pinocchio) for desired CoM + foot poses
  4. Track IK targets with PD + gravity comp in MuJoCo
  5. Record video + ZMP/CoM plots

Gate criteria:
  - Walk 4+ steps without falling (pelvis > 0.5m)
  - ZMP stays inside support polygon
  - Video: media/m4_walking.mp4
  - Plot: media/m4_zmp.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

from lab7_common import (
    ARM_NEUTRAL_LEFT,
    ARM_NEUTRAL_RIGHT,
    CTRL_LEFT_ARM,
    CTRL_LEFT_LEG,
    CTRL_RIGHT_ARM,
    CTRL_RIGHT_LEG,
    CTRL_STAND,
    CTRL_WAIST,
    DT,
    FOOT_Y_OFFSET,
    MEDIA_DIR,
    PELVIS_MJCF_Z,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    Z_C,
    load_g1_mujoco,
    load_g1_pinocchio,
    mj_qpos_to_pin,
    pin_q_to_mj,
    set_arm_ctrl_neutral,
)
from lipm_preview_control import generate_com_trajectory
from m3c_static_ik import build_standing_q, whole_body_ik
from swing_trajectory import generate_swing_trajectory
from zmp_reference import (
    Footstep,
    WalkingTiming,
    generate_footstep_plan,
    generate_zmp_reference,
    get_phase_at_time,
)

# ---------------------------------------------------------------------------
# Walking parameters
# ---------------------------------------------------------------------------

FOOTSTEP_PLAN: dict = dict(
    n_steps=12,
    stride_x=0.08,       # 8 cm forward per step
    foot_y=FOOT_Y_OFFSET, # lateral half-spacing from lab7_common
)

STEP_TIME: dict = dict(
    t_ss=0.8,    # single support duration [s]
    t_ds=0.2,    # double support duration [s]
)

PRE_SIM_TIME: float = 1.5   # settle in standing pose before walking [s]
POST_SIM_TIME: float = 2.0  # hold final pose after last step [s]

STEP_HEIGHT: float = 0.03   # swing foot clearance [m]
CTRL_DT: float = 0.02       # 50 Hz control rate [s]
KP_SERVO: float = 500.0
KD_SERVO: float = 40.0      # velocity damping

TIMING = WalkingTiming(
    t_ss=STEP_TIME["t_ss"],
    t_ds=STEP_TIME["t_ds"],
    t_init=1.0,
    t_final=1.0,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Footstep plan
    # ------------------------------------------------------------------
    footsteps: list[Footstep] = generate_footstep_plan(
        n_steps=FOOTSTEP_PLAN["n_steps"],
        stride_x=FOOTSTEP_PLAN["stride_x"],
        foot_y=FOOTSTEP_PLAN["foot_y"],
    )
    print(f"Footstep plan: {len(footsteps)} footsteps "
          f"({FOOTSTEP_PLAN['n_steps']} forward steps)")

    # ------------------------------------------------------------------
    # 2. ZMP reference trajectory
    # ------------------------------------------------------------------
    zmp_ref_x, zmp_ref_y, t_arr, total_time = generate_zmp_reference(
        footsteps, TIMING, DT,
    )
    N_plan = len(t_arr)
    print(f"ZMP reference: {N_plan} samples, {total_time:.2f} s total")

    # ------------------------------------------------------------------
    # 3. CoM trajectory via LIPM preview control
    # ------------------------------------------------------------------
    mid_x = (footsteps[0].x + footsteps[1].x) / 2.0
    mid_y = (footsteps[0].y + footsteps[1].y) / 2.0
    com_x, com_y, com_dx, com_dy = generate_com_trajectory(
        zmp_ref_x, zmp_ref_y, Z_C, DT,
        com_x0=mid_x, com_y0=mid_y,
    )
    print(f"CoM trajectory: x [{com_x[0]:.3f} → {com_x[-1]:.3f}], "
          f"y [{com_y[0]:.3f} → {com_y[-1]:.3f}]")

    # ------------------------------------------------------------------
    # 4. Pre-generate swing foot trajectories for each step
    # ------------------------------------------------------------------
    n_steps = len(footsteps) - 2  # exclude initial stance + final alignment
    swing_trajs: list[np.ndarray] = []

    # Ground height for foot positions
    z_foot = 0.0

    for i in range(n_steps):
        # Swing foot = footsteps[i+1] (the foot being placed)
        # Previous position of swing foot: find last time this foot was placed
        swing_foot_id = footsteps[i + 1].foot
        # Search backwards for the previous placement of this foot
        prev_pos = None
        for j in range(i, -1, -1):
            if footsteps[j].foot == swing_foot_id:
                prev_pos = np.array([footsteps[j].x, footsteps[j].y, z_foot])
                break
        if prev_pos is None:
            # First time this foot moves — use initial position
            y_sign = +1.0 if swing_foot_id == "L" else -1.0
            prev_pos = np.array([0.0, y_sign * FOOTSTEP_PLAN["foot_y"], z_foot])

        end_pos = np.array([
            footsteps[i + 1].x, footsteps[i + 1].y, z_foot,
        ])

        traj = generate_swing_trajectory(
            start_pos=prev_pos,
            end_pos=end_pos,
            duration=STEP_TIME["t_ss"],
            dt=DT,
            step_height=STEP_HEIGHT,
        )
        swing_trajs.append(traj)

    # Final alignment step swing trajectory
    last_step = footsteps[-1]
    align_foot_id = last_step.foot
    prev_pos_align = None
    for j in range(len(footsteps) - 2, -1, -1):
        if footsteps[j].foot == align_foot_id:
            prev_pos_align = np.array([footsteps[j].x, footsteps[j].y, z_foot])
            break
    if prev_pos_align is None:
        y_sign = +1.0 if align_foot_id == "L" else -1.0
        prev_pos_align = np.array([0.0, y_sign * FOOTSTEP_PLAN["foot_y"], z_foot])

    align_traj = generate_swing_trajectory(
        start_pos=prev_pos_align,
        end_pos=np.array([last_step.x, last_step.y, z_foot]),
        duration=STEP_TIME["t_ss"],
        dt=DT,
        step_height=STEP_HEIGHT,
    )
    swing_trajs.append(align_traj)

    print(f"Swing trajectories: {len(swing_trajs)} steps, "
          f"{swing_trajs[0].shape[0]} samples each")

    print("planning done")

    # ------------------------------------------------------------------
    # 5. IK pre-computation
    # ------------------------------------------------------------------
    pin_model, pin_data = load_g1_pinocchio()
    q_stand = build_standing_q(pin_model)

    # FK on standing pose to get foot orientation (flat feet)
    pin.forwardKinematics(pin_model, pin_data, q_stand)
    pin.updateFramePlacements(pin_model, pin_data)
    lf_id = pin_model.getFrameId("left_foot")
    rf_id = pin_model.getFrameId("right_foot")
    foot_R = pin_data.oMf[lf_id].rotation.copy()  # flat foot orientation

    # Initial foot positions (on ground, symmetric stance)
    lf_pos_init = np.array([0.0, +FOOTSTEP_PLAN["foot_y"], z_foot])
    rf_pos_init = np.array([0.0, -FOOTSTEP_PLAN["foot_y"], z_foot])

    # Track current planted position per foot
    lf_pos_current = lf_pos_init.copy()
    rf_pos_current = rf_pos_init.copy()

    q_prev = q_stand.copy()
    N_total = len(t_arr)
    q_ik_traj = np.zeros((N_total, pin_model.nq))

    for k in range(N_total):
        t = t_arr[k]
        phase, step_idx, t_in_phase = get_phase_at_time(t, footsteps, TIMING)

        # --- Determine foot positions ---
        lf_pos = lf_pos_current.copy()
        rf_pos = rf_pos_current.copy()

        if phase == "ss":
            swing_foot_id = footsteps[step_idx + 1].foot
            swing_traj = swing_trajs[step_idx]
            # Map t_in_phase to swing trajectory index
            swing_idx = int(round(t_in_phase / DT))
            swing_idx = min(swing_idx, swing_traj.shape[0] - 1)
            swing_pos = swing_traj[swing_idx]

            if swing_foot_id == "L":
                lf_pos = swing_pos
            else:
                rf_pos = swing_pos

            # At end of swing, update planted position
            if swing_idx == swing_traj.shape[0] - 1:
                if swing_foot_id == "L":
                    lf_pos_current = swing_pos.copy()
                else:
                    rf_pos_current = swing_pos.copy()

        elif phase == "ds" and step_idx > 0:
            # After a step completes, ensure foot is updated
            placed = footsteps[step_idx + 1]
            if placed.foot == "L":
                lf_pos_current = np.array([placed.x, placed.y, z_foot])
            else:
                rf_pos_current = np.array([placed.x, placed.y, z_foot])
            lf_pos = lf_pos_current.copy()
            rf_pos = rf_pos_current.copy()

        elif phase == "final_ss":
            swing_traj = swing_trajs[-1]
            swing_idx = int(round(t_in_phase / DT))
            swing_idx = min(swing_idx, swing_traj.shape[0] - 1)
            swing_pos = swing_traj[swing_idx]
            align_foot = footsteps[-1].foot
            if align_foot == "L":
                lf_pos = swing_pos
            else:
                rf_pos = swing_pos

        elif phase == "final_ds":
            last = footsteps[-1]
            if last.foot == "L":
                lf_pos_current = np.array([last.x, last.y, z_foot])
            else:
                rf_pos_current = np.array([last.x, last.y, z_foot])
            lf_pos = lf_pos_current.copy()
            rf_pos = rf_pos_current.copy()

        # --- Build SE3 targets ---
        lf_target = pin.SE3(foot_R.copy(), lf_pos)
        rf_target = pin.SE3(foot_R.copy(), rf_pos)
        com_target = np.array([com_x[k], com_y[k], 0.0])

        # --- Solve IK ---
        q_sol, converged, n_iters, errors = whole_body_ik(
            pin_model, pin_data, q_prev,
            com_target=com_target,
            lf_target=lf_target,
            rf_target=rf_target,
            fix_base=False,
        )
        q_ik_traj[k] = q_sol
        q_prev = q_sol.copy()

        if (k + 1) % 500 == 0:
            print(f"  IK precomp: {k + 1}/{N_total} "
                  f"(conv={converged}, iters={n_iters}, "
                  f"com_err={errors['com_xy']:.4f})")

    print(f"ik precomp done {N_total} waypoints")

    # ------------------------------------------------------------------
    # 6. MuJoCo simulation with PD + gravity comp
    # ------------------------------------------------------------------
    mj_model, mj_data = load_g1_mujoco()
    mj_dof_start = 6  # first actuated dof after free joint

    # Settle in standing pose
    print(f"Settling {PRE_SIM_TIME}s ...")
    n_pre = int(PRE_SIM_TIME / DT)
    for _ in range(n_pre):
        bias = mj_data.qfrc_bias[mj_dof_start:35]
        mj_data.ctrl[:] = CTRL_STAND + bias / KP_SERVO
        mujoco.mj_step(mj_model, mj_data)

    # PD gains per phase
    KP_STANCE = 200.0
    KD_STANCE = 20.0
    KP_SWING = 100.0
    KD_SWING = 10.0
    K_VEL = 40.0

    # Camera for video
    renderer = mujoco.Renderer(mj_model, RENDER_HEIGHT, RENDER_WIDTH)
    cam = mujoco.MjvCamera()
    cam.distance = 2.5
    cam.elevation = -20.0
    cam.azimuth = 135.0
    cam.lookat[:] = [0.2, 0.0, 0.7]

    frames = []
    render_every = max(1, int(1.0 / (RENDER_FPS * DT)))

    pelvis_z_log = []
    com_x_log = []
    com_y_log = []
    times_log = []

    print(f"Simulating {total_time}s ...")
    t_sim_start = mj_data.time

    for k in range(N_total):
        t = k * DT

        # Determine PD gains based on phase
        phase, step_idx, t_in_phase = get_phase_at_time(t, footsteps, TIMING)
        is_swing = phase in ("ss", "final_ss")
        kp = KP_SWING if is_swing else KP_STANCE
        kd = KD_SWING if is_swing else KD_STANCE

        # Get IK target (q_target is nq=36, ctrl is nu=29)
        q_target = q_ik_traj[k]
        ctrl = q_target[7:36].copy()
        # Arms at neutral
        ctrl[15:22] = ARM_NEUTRAL_LEFT
        ctrl[22:29] = ARM_NEUTRAL_RIGHT
        # Gravity feedforward + velocity damping
        bias = mj_data.qfrc_bias[6:35]
        ctrl += bias / kp
        ctrl -= K_VEL * mj_data.qvel[6:35] / kp
        mj_data.ctrl[:] = ctrl
        mujoco.mj_step(mj_model, mj_data)

        # Monitor
        pelvis_z = mj_data.qpos[2]
        com_actual = mj_data.subtree_com[0]
        com_err_x = (com_x[k] - com_actual[0]) * 1000
        com_err_y = (com_y[k] - com_actual[1]) * 1000

        if k % 25 == 0:
            print(
                f"  t={t:.1f}s [{phase:9s}] "
                f"pz={pelvis_z:.3f}  com_err=({com_err_x:.0f},{com_err_y:.0f})mm"
            )

        pelvis_z_log.append(pelvis_z)
        com_x_log.append(com_actual[0])
        com_y_log.append(com_actual[1])
        times_log.append(mj_data.time - t_sim_start)

        if k % render_every == 0:
            renderer.update_scene(mj_data, cam)
            frames.append(renderer.render().copy())

    renderer.close()

    # Post-simulation hold
    print(f"Holding final pose {POST_SIM_TIME}s ...")
    n_post = int(POST_SIM_TIME / DT)
    renderer = mujoco.Renderer(mj_model, RENDER_HEIGHT, RENDER_WIDTH)
    for i in range(n_post):
        bias = mj_data.qfrc_bias[mj_dof_start:35]
        mj_data.ctrl[:] = q_ik_traj[-1][7:36] + bias / 150.0
        mujoco.mj_step(mj_model, mj_data)
        if i % render_every == 0:
            renderer.update_scene(mj_data, cam)
            frames.append(renderer.render().copy())
    renderer.close()

    # Save video
    import imageio
    video_path = MEDIA_DIR / "m3e_zmp_walking.mp4"
    writer = imageio.get_writer(str(video_path), fps=RENDER_FPS,
                                codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"Video: {video_path} ({len(frames)} frames)")

    # Gate results
    min_pz = min(pelvis_z_log)
    forward_disp = com_x_log[-1] - com_x_log[0]
    print(f"\n{'='*60}")
    print(f"  Pelvis min: {min_pz:.3f}m  {'PASS' if min_pz > 0.6 else 'FAIL'}")
    print(f"  Forward disp: {forward_disp*100:.1f}cm  "
          f"{'PASS' if forward_disp > 0.2 else 'FAIL'}")
    print(f"  Video: saved")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
