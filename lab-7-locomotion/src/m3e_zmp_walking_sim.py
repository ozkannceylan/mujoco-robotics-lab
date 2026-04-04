#!/usr/bin/env python3
"""M3e — Single Conservative Step Simulation.

ONE forward step with right foot (no alignment step).

Phases:
  1. init_ds (2.0s): weight shift from center to left foot
  2. SS (0.5s): right foot swings forward 4cm, stance on left
  3. settle (3.0s): both feet on ground, CoM returns to staggered center

Custom ZMP reference (bypasses footstep plan generator):
  - init_ds: ZMP linearly from (0,0) to left foot (0, +foot_y)
  - SS: ZMP held at left foot
  - settle: ZMP linearly from left foot to center of staggered stance

Pre-computed IK + feedforward gravity comp + ankle-strategy feedback.

Gate criteria (M3e):
  - Pelvis > 0.6m throughout
  - Right foot >= 2cm forward
  - Foot lifts >= 1cm during swing
  - Stable 2s after landing
  - Video: media/m3e_single_step.mp4
  - Plot: media/m3e_com_tracking.png
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
    CTRL_STAND,
    DT,
    FOOT_Y_OFFSET,
    MEDIA_DIR,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    mj_qpos_to_pin,
    Z_C,
    load_g1_mujoco,
    load_g1_pinocchio,
)
from lipm_preview_control import generate_com_trajectory
from m3c_static_ik import build_standing_q, whole_body_ik
from swing_trajectory import generate_swing_trajectory

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

STRIDE_X: float = 0.04       # 4 cm forward
STEP_HEIGHT: float = 0.02    # 2 cm swing clearance

# Phase durations
T_INIT_DS: float = 2.0       # weight shift to left foot [s]
T_SS: float = 0.5            # single support (right foot swing) [s]
T_SETTLE: float = 3.0        # settle in staggered stance [s]
SETTLE_RAMP: float = 2.0     # time for ZMP ramp during settle [s]

PRE_SIM_TIME: float = 1.0    # settle in standing [s]
POST_SIM_TIME: float = 0.5   # hold after trajectory [s]

# Ankle feedback gains
KP_COM_X: float = 0.5
KP_COM_Y: float = 1.0

# Leg actuator gains
KP_LEG: float = 800.0
KV_LEG: float = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_phase(t: float) -> str:
    """Return current phase name."""
    if t < T_INIT_DS:
        return "init_ds"
    elif t < T_INIT_DS + T_SS:
        return "ss"
    else:
        return "settle"


def override_leg_gains(mj_model: mujoco.MjModel) -> None:
    """Set all 12 leg actuators to KP_LEG / KV_LEG."""
    for i in range(12):
        mj_model.actuator_gainprm[i, 0] = KP_LEG
        mj_model.actuator_biasprm[i, 1] = -KP_LEG
        mj_model.actuator_biasprm[i, 2] = -KV_LEG
    print(f"  Leg gains: kp={KP_LEG}, kv={KV_LEG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    foot_y = FOOT_Y_OFFSET
    z_foot = 0.0
    total_time = T_INIT_DS + T_SS + T_SETTLE

    # ------------------------------------------------------------------
    # 1. Manual ZMP reference (no footstep plan generator)
    # ------------------------------------------------------------------
    N_plan = int(total_time / DT)
    t_arr = np.arange(N_plan) * DT
    zmp_x = np.zeros(N_plan)
    zmp_y = np.zeros(N_plan)

    # Phase 1: init_ds — center → left foot
    m1 = t_arr < T_INIT_DS
    a1 = t_arr[m1] / T_INIT_DS
    zmp_y[m1] = a1 * foot_y  # 0 → +foot_y

    # Phase 2: SS — hold at left foot
    m2 = (t_arr >= T_INIT_DS) & (t_arr < T_INIT_DS + T_SS)
    zmp_y[m2] = foot_y

    # Phase 3: settle — left foot → staggered center
    t_settle_start = T_INIT_DS + T_SS
    m3 = t_arr >= t_settle_start
    a3 = np.clip((t_arr[m3] - t_settle_start) / SETTLE_RAMP, 0.0, 1.0)
    stag_center_x = STRIDE_X / 2.0  # 0.02
    zmp_x[m3] = a3 * stag_center_x
    zmp_y[m3] = foot_y * (1.0 - a3)  # foot_y → 0

    print(f"ZMP reference: {N_plan} samples, {total_time:.1f}s")
    print(f"  ZMP X: [{zmp_x[0]:.4f} -> {zmp_x[-1]:.4f}]")
    print(f"  ZMP Y: [{zmp_y[0]:.4f} -> {zmp_y[-1]:.4f}]")

    # ------------------------------------------------------------------
    # 2. CoM trajectory (LIPM preview control)
    # ------------------------------------------------------------------
    com_x, com_y, _, _ = generate_com_trajectory(
        zmp_x, zmp_y, Z_C, DT,
        com_x0=0.0, com_y0=0.0,
    )
    print(f"CoM: x=[{com_x[0]:.4f} -> {com_x[-1]:.4f}], "
          f"y=[{com_y[0]:.4f} -> {com_y[-1]:.4f}]")

    # ------------------------------------------------------------------
    # 3. Swing trajectory (right foot only)
    # ------------------------------------------------------------------
    swing_r = generate_swing_trajectory(
        start_pos=np.array([0.0, -foot_y, z_foot]),
        end_pos=np.array([STRIDE_X, -foot_y, z_foot]),
        duration=T_SS, dt=DT, step_height=STEP_HEIGHT,
    )
    print(f"Swing: {swing_r.shape[0]} pts, height={STEP_HEIGHT}m, "
          f"stride={STRIDE_X}m")

    # ------------------------------------------------------------------
    # 4. IK pre-computation
    # ------------------------------------------------------------------
    pin_model, pin_data = load_g1_pinocchio()
    q_stand = build_standing_q(pin_model)

    pin.forwardKinematics(pin_model, pin_data, q_stand)
    pin.updateFramePlacements(pin_model, pin_data)
    lf_id = pin_model.getFrameId("left_foot")
    foot_R = pin_data.oMf[lf_id].rotation.copy()

    # Left foot never moves
    lf_pos = np.array([0.0, +foot_y, z_foot])
    # Right foot: initial position, updated after swing
    rf_pos_current = np.array([0.0, -foot_y, z_foot])

    q_prev = q_stand.copy()
    q_ik_traj = np.zeros((N_plan, pin_model.nq))

    print(f"IK pre-computation ({N_plan} waypoints) ...")
    for k in range(N_plan):
        t = t_arr[k]
        phase = get_phase(t)

        rf_pos = rf_pos_current.copy()

        if phase == "ss":
            t_in_ss = t - T_INIT_DS
            idx = min(int(round(t_in_ss / DT)), swing_r.shape[0] - 1)
            rf_pos = swing_r[idx].copy()
            if idx == swing_r.shape[0] - 1:
                rf_pos_current = rf_pos.copy()

        lf_target = pin.SE3(foot_R.copy(), lf_pos)
        rf_target = pin.SE3(foot_R.copy(), rf_pos)
        com_target = np.array([com_x[k], com_y[k], 0.0])

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
            print(f"  [{k+1}/{N_plan}] {phase:8s} conv={converged} "
                  f"it={n_iters} com={errors['com_xy']*1000:.1f}mm")

    print("IK precomp done (used for warm-starting online IK).")

    # ------------------------------------------------------------------
    # 5. MuJoCo simulation with ONLINE IK
    # ------------------------------------------------------------------
    mj_model, mj_data = load_g1_mujoco()
    override_leg_gains(mj_model)

    # Settle in standing
    n_pre = int(PRE_SIM_TIME / DT)
    for _ in range(n_pre):
        mj_data.ctrl[:] = CTRL_STAND
        mujoco.mj_step(mj_model, mj_data)
    print(f"Settled: pelvis_z={mj_data.qpos[2]:.4f}m")

    # Foot body IDs for tracking
    rf_body_id = mj_model.body("right_ankle_roll_link").id
    rf_z_init = float(mj_data.xpos[rf_body_id][2])
    rf_x_init = float(mj_data.xpos[rf_body_id][0])

    # Renderer
    renderer = mujoco.Renderer(mj_model, RENDER_HEIGHT, RENDER_WIDTH)
    cam = mujoco.MjvCamera()
    cam.distance = 2.0
    cam.elevation = -15.0
    cam.azimuth = 150.0
    cam.lookat[:] = [0.02, 0.0, 0.7]

    frames: list[np.ndarray] = []
    render_every = max(1, int(1.0 / (RENDER_FPS * DT)))

    # Logs
    pelvis_z_log: list[float] = []
    com_x_log: list[float] = []
    com_y_log: list[float] = []
    times_log: list[float] = []
    rf_x_log: list[float] = []
    rf_z_log: list[float] = []

    t_sim_start = mj_data.time
    fell = False
    rf_pos_sim = np.array([0.0, -foot_y, z_foot])

    print(f"Simulating {total_time:.1f}s (online IK at 500Hz) ...")
    for k in range(N_plan):
        t = k * DT
        phase = get_phase(t)

        # --- Compute foot/CoM targets (same logic as precompute) ---
        rf_tgt = rf_pos_sim.copy()
        if phase == "ss":
            t_in_ss = t - T_INIT_DS
            idx = min(int(round(t_in_ss / DT)), swing_r.shape[0] - 1)
            rf_tgt = swing_r[idx].copy()
            if idx == swing_r.shape[0] - 1:
                rf_pos_sim = rf_tgt.copy()

        lf_target = pin.SE3(foot_R.copy(), lf_pos)
        rf_target = pin.SE3(foot_R.copy(), rf_tgt)
        com_target = np.array([com_x[k], com_y[k], 0.0])

        # --- Online IK with MEASURED base ---
        pin_q_meas = mj_qpos_to_pin(mj_data.qpos)
        q_init = q_ik_traj[k].copy()       # pre-computed warm start
        q_init[:7] = pin_q_meas[:7]         # replace base with measured

        q_sol, _, _, _ = whole_body_ik(
            pin_model, pin_data, q_init,
            com_target=com_target,
            lf_target=lf_target,
            rf_target=rf_target,
            fix_base=True,
            max_iter=5,
        )

        ctrl = q_sol[7:36].copy()
        ctrl[15:22] = ARM_NEUTRAL_LEFT
        ctrl[22:29] = ARM_NEUTRAL_RIGHT

        # Feedforward gravity compensation for locomotion joints
        for i in range(15):  # 12 legs + 3 waist
            kp_i = mj_model.actuator_gainprm[i, 0]
            if kp_i > 0:
                ctrl[i] += mj_data.qfrc_bias[i + 6] / kp_i

        # --- Ankle-strategy CoM feedback (extra correction on top of IK) ---
        com_actual = mj_data.subtree_com[0]
        com_err_x = com_x[k] - com_actual[0]
        com_err_y = com_y[k] - com_actual[1]

        pelvis_z = mj_data.qpos[2]
        pelvis_vz = mj_data.qvel[2]
        z_ref = 0.785
        z_err = max(z_ref - pelvis_z, 0.0)

        comp_pitch = KP_COM_X * com_err_x + 3.0 * z_err - 0.5 * pelvis_vz
        comp_pitch = np.clip(comp_pitch, -0.3, 0.4)
        comp_roll = KP_COM_Y * com_err_y
        comp_roll = np.clip(comp_roll, -0.25, 0.25)

        if phase == "ss":
            ctrl[4] += comp_pitch     # left ankle only
            ctrl[5] += comp_roll
        else:
            ctrl[4] += comp_pitch     # both ankles
            ctrl[10] += comp_pitch
            ctrl[5] += comp_roll
            ctrl[11] += comp_roll

        mj_data.ctrl[:] = ctrl
        mujoco.mj_step(mj_model, mj_data)

        # --- Logging ---
        pelvis_z = mj_data.qpos[2]
        com_now = mj_data.subtree_com[0]
        rf_mj = mj_data.xpos[rf_body_id]

        pelvis_z_log.append(pelvis_z)
        com_x_log.append(com_now[0])
        com_y_log.append(com_now[1])
        times_log.append(mj_data.time - t_sim_start)
        rf_x_log.append(float(rf_mj[0]))
        rf_z_log.append(float(rf_mj[2]))

        if abs(t % 0.5) < DT / 2:
            ex = (com_x[k] - com_now[0]) * 1000
            ey = (com_y[k] - com_now[1]) * 1000
            print(f"  t={t:5.1f}s [{phase:8s}] pz={pelvis_z:.3f} "
                  f"err=({ex:+.0f},{ey:+.0f})mm "
                  f"rf=({rf_mj[0]:.3f},{rf_mj[2]:.4f})")

        if k % render_every == 0:
            renderer.update_scene(mj_data, cam)
            frames.append(renderer.render().copy())

        if pelvis_z < 0.4 and not fell:
            fell = True
            print(f"\n  *** FELL at t={t:.2f}s [{phase}] "
                  f"pz={pelvis_z:.3f} ***\n")

    renderer.close()

    # Post-hold
    if POST_SIM_TIME > 0:
        n_post = int(POST_SIM_TIME / DT)
        renderer = mujoco.Renderer(mj_model, RENDER_HEIGHT, RENDER_WIDTH)
        final_ctrl = ctrl.copy()
        for i in range(n_post):
            mj_data.ctrl[:] = final_ctrl
            mujoco.mj_step(mj_model, mj_data)
            pelvis_z_log.append(mj_data.qpos[2])
            rf_mj = mj_data.xpos[rf_body_id]
            rf_x_log.append(float(rf_mj[0]))
            rf_z_log.append(float(rf_mj[2]))
            if i % render_every == 0:
                renderer.update_scene(mj_data, cam)
                frames.append(renderer.render().copy())
        renderer.close()

    # ------------------------------------------------------------------
    # 6. Save video
    # ------------------------------------------------------------------
    import imageio
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    video_path = MEDIA_DIR / "m3e_single_step.mp4"
    writer = imageio.get_writer(str(video_path), fps=RENDER_FPS,
                                codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"Video: {video_path} ({len(frames)} frames)")

    # ------------------------------------------------------------------
    # 7. CoM tracking plot
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor("#08111f")
    t_plot = np.array(times_log)

    for ax in axes:
        ax.set_facecolor("#08111f")
        ax.tick_params(colors="#e0e0e0")
        ax.grid(True, color="#1a2a3f", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color("#1a2a3f")

    axes[0].plot(t_plot, com_x[:len(t_plot)], "--", color="#ff6b6b",
                 linewidth=1.5, label="Planned")
    axes[0].plot(t_plot, com_x_log, "-", color="#4ecdc4",
                 linewidth=2, label="Actual")
    axes[0].set_ylabel("X [m]", color="#e0e0e0")
    axes[0].set_title("CoM X Tracking", color="#e0e0e0",
                       fontsize=14, fontweight="bold")
    axes[0].legend(facecolor="#08111f", edgecolor="#1a2a3f",
                   labelcolor="#e0e0e0")

    axes[1].plot(t_plot, com_y[:len(t_plot)], "--", color="#ff6b6b",
                 linewidth=1.5, label="Planned")
    axes[1].plot(t_plot, com_y_log, "-", color="#4ecdc4",
                 linewidth=2, label="Actual")
    axes[1].set_xlabel("Time [s]", color="#e0e0e0")
    axes[1].set_ylabel("Y [m]", color="#e0e0e0")
    axes[1].set_title("CoM Y Tracking", color="#e0e0e0",
                       fontsize=14, fontweight="bold")
    axes[1].legend(facecolor="#08111f", edgecolor="#1a2a3f",
                   labelcolor="#e0e0e0")

    plot_path = MEDIA_DIR / "m3e_com_tracking.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#08111f")
    plt.close(fig)
    print(f"Plot: {plot_path}")

    # ------------------------------------------------------------------
    # 8. Gate results
    # ------------------------------------------------------------------
    min_pz = min(pelvis_z_log)
    rf_fwd = max(rf_x_log) - rf_x_init
    rf_lift = max(rf_z_log) - rf_z_init

    # Stability: pelvis Z in last 2s
    n_2s = int(2.0 / DT)
    pz_last = pelvis_z_log[-n_2s:] if len(pelvis_z_log) > n_2s else pelvis_z_log
    stable_min = min(pz_last)

    # CoM RMS
    n = min(len(com_x_log), len(com_x))
    rms_x = np.sqrt(np.mean((np.array(com_x_log[:n]) - com_x[:n])**2)) * 1000
    rms_y = np.sqrt(np.mean((np.array(com_y_log[:n]) - com_y[:n])**2)) * 1000

    print(f"\n{'='*60}")
    print(f"  GATE RESULTS — M3e Single Step")
    print(f"{'='*60}")
    print(f"  {'Criterion':<35s} {'Value':<15s} {'Status'}")
    print(f"  {'-'*55}")

    g1 = min_pz > 0.6
    print(f"  {'Pelvis > 0.6m':<35s} {f'{min_pz:.3f}m':<15s} "
          f"{'PASS' if g1 else 'FAIL'}")

    g2 = rf_fwd >= 0.02
    print(f"  {'R foot >= 2cm fwd':<35s} {f'{rf_fwd*100:.1f}cm':<15s} "
          f"{'PASS' if g2 else 'FAIL'}")

    g3 = rf_lift >= 0.01
    print(f"  {'R foot lift >= 1cm':<35s} {f'{rf_lift*100:.1f}cm':<15s} "
          f"{'PASS' if g3 else 'FAIL'}")

    g4 = stable_min > 0.6
    print(f"  {'Stable 2s (pz > 0.6m)':<35s} {f'{stable_min:.3f}m':<15s} "
          f"{'PASS' if g4 else 'FAIL'}")

    print(f"  {'CoM RMS X':<35s} {f'{rms_x:.1f}mm':<15s} (info)")
    print(f"  {'CoM RMS Y':<35s} {f'{rms_y:.1f}mm':<15s} (info)")
    print(f"  {'Video':<35s} {'saved':<15s} PASS")
    print(f"  {'Plot':<35s} {'saved':<15s} PASS")

    all_pass = g1 and g2 and g3 and g4
    print(f"  {'-'*55}")
    print(f"  {'OVERALL':<35s} {'':<15s} "
          f"{'ALL PASS' if all_pass else 'SOME FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
