#!/usr/bin/env python3
"""M5 — Lab 7 Capstone Demo: What Works on the Unitree G1.

Demonstrates the reliable capabilities developed in Lab 7:
  Phase 1: Standing with gravity-compensated PD (2s)
  Phase 2: Push recovery — 5N lateral push on torso (3s)
  Phase 3: Weight shift to left foot via pre-computed IK (3s)
  Phase 4: Weight shift back to center via pre-computed IK (3s)

Also generates an LIPM planner visualization showing the theoretical
CoM + ZMP trajectory for a 6-step walking plan (overlay plot only,
NOT executed in simulation — walking was not achieved).

Outputs:
  media/m5_capstone.mp4   — simulation video (Phases 1–4)
  media/m5_plots.png      — trajectory analysis + LIPM overlay

Gate:
  - Script runs without error
  - Video shows all 4 phases
  - LIPM plot shows CoM/ZMP/footsteps
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
    MEDIA_DIR,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    Z_C,
    load_g1_mujoco,
    load_g1_pinocchio,
    mj_qpos_to_pin,
)
from m3c_static_ik import whole_body_ik

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

KP_SERVO: float = 500.0
K_VEL: float = 40.0         # near-critical damping: ζ ≈ 40 / (2√500) ≈ 0.89

T_STAND: float = 2.0        # Phase 1: quiet standing
T_PUSH_START: float = 2.0   # Push applied at this time
PUSH_DURATION: float = 0.2
PUSH_FORCE_Y: float = 5.0   # lateral push [N]
T_PUSH_PHASE: float = 3.0   # Phase 2: push + recovery
T_SHIFT: float = 1.5        # Phase 3: CoM shift ramp duration
T_SHIFT_HOLD: float = 1.5   # Phase 3: hold shifted position
T_RETURN: float = 1.5       # Phase 4: return ramp duration
T_RETURN_HOLD: float = 1.5  # Phase 4: hold center position

COM_SHIFT_Y: float = 0.05   # 5cm toward left foot (+Y)
N_WAYPOINTS: int = 10


# ---------------------------------------------------------------------------
# IK trajectory precomputation
# ---------------------------------------------------------------------------


def precompute_shift_trajectory(
    model: pin.Model,
    data: pin.Data,
    q_init: np.ndarray,
    com_ref: np.ndarray,
    lf_ref: pin.SE3,
    rf_ref: pin.SE3,
    shift_y: float,
) -> list[np.ndarray]:
    """Pre-compute IK waypoints for CoM Y shift."""
    q_prev = q_init.copy()
    waypoints: list[np.ndarray] = []

    for i in range(N_WAYPOINTS + 1):
        s = i / N_WAYPOINTS
        com_target = com_ref.copy()
        com_target[1] += s * shift_y

        q_sol, converged, n_iters, errors = whole_body_ik(
            model, data, q_prev,
            com_target=com_target,
            lf_target=lf_ref,
            rf_target=rf_ref,
        )
        waypoints.append(q_sol[7:36].copy())
        q_prev = q_sol

    return waypoints


def interp_waypoints(waypoints: list[np.ndarray], s: float) -> np.ndarray:
    """Linearly interpolate joint targets at parameter s in [0, 1]."""
    n = len(waypoints) - 1
    idx = s * n
    i = min(int(idx), n - 1)
    frac = idx - i
    return (1.0 - frac) * waypoints[i] + frac * waypoints[i + 1]


def smooth_s(t: float, t_end: float) -> float:
    """Cosine smooth interpolation 0 -> 1 over [0, t_end]."""
    s = np.clip(t / t_end, 0.0, 1.0)
    return float(0.5 * (1.0 - np.cos(np.pi * s)))


# ---------------------------------------------------------------------------
# Video helper
# ---------------------------------------------------------------------------


def save_mp4(frames: list[np.ndarray], path: Path, fps: int = RENDER_FPS) -> None:
    """Save frames as H.264 MP4."""
    import imageio
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  Video saved: {path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")


# ---------------------------------------------------------------------------
# LIPM planner overlay (theoretical, not executed)
# ---------------------------------------------------------------------------


def generate_lipm_plot(ax_com: object, ax_zmp: object) -> None:
    """Generate LIPM preview control plot on the given axes.

    Shows the theoretical CoM + ZMP trajectory for a 6-step walking plan.
    This is the planner output only — NOT executed in simulation.
    """
    from lipm_preview_control import generate_com_trajectory
    from zmp_reference import (
        Footstep,
        WalkingTiming,
        generate_footstep_plan,
        generate_zmp_reference,
    )

    dt_plan = 0.01
    footsteps = generate_footstep_plan(n_steps=6, stride_x=0.06, foot_y=0.075)
    timing = WalkingTiming(t_ss=0.8, t_ds=0.2, t_init=1.0, t_final=1.0)
    zmp_ref_x, zmp_ref_y, t_arr, total_time = generate_zmp_reference(
        footsteps, timing, dt_plan,
    )
    com_x, com_y, _, _ = generate_com_trajectory(
        zmp_ref_x, zmp_ref_y, z_c=Z_C, dt=dt_plan,
    )

    # -- CoM + ZMP X --
    ax_com.plot(t_arr, zmp_ref_x * 1000, "r--", lw=1, alpha=0.7, label="ZMP ref X")
    ax_com.plot(t_arr, com_x * 1000, "b-", lw=1.5, label="CoM X (LIPM)")
    # Footstep markers
    t_fs = timing.t_init
    for fs in footsteps[1:]:
        marker = "v" if fs.foot == "L" else "^"
        color = "tab:blue" if fs.foot == "L" else "tab:orange"
        ax_com.plot(t_fs, fs.x * 1000, marker, color=color, ms=8, alpha=0.7)
        t_fs += timing.t_ss + timing.t_ds
    ax_com.set_ylabel("X position [mm]")
    ax_com.set_title("LIPM Preview Control — Planner Output (NOT executed)")
    ax_com.legend(fontsize=7, loc="upper left")
    ax_com.grid(True, alpha=0.3)

    # -- CoM + ZMP Y --
    ax_zmp.plot(t_arr, zmp_ref_y * 1000, "r--", lw=1, alpha=0.7, label="ZMP ref Y")
    ax_zmp.plot(t_arr, com_y * 1000, "b-", lw=1.5, label="CoM Y (LIPM)")
    ax_zmp.axhline(0, color="gray", ls=":", alpha=0.3)
    ax_zmp.set_ylabel("Y position [mm]")
    ax_zmp.set_xlabel("Time [s]")
    ax_zmp.legend(fontsize=7, loc="upper left")
    ax_zmp.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────
    mj_model, mj_data = load_g1_mujoco()
    pin_model, pin_data = load_g1_pinocchio()
    dt = mj_model.opt.timestep

    torso_id = mj_model.body("torso_link").id
    lf_site = mj_model.site("left_foot").id
    rf_site = mj_model.site("right_foot").id

    total_time = T_STAND + T_PUSH_PHASE + T_SHIFT + T_SHIFT_HOLD + T_RETURN + T_RETURN_HOLD
    n_steps = int(total_time / dt)
    render_every = max(1, int(1.0 / (RENDER_FPS * dt)))

    print(f"  M5 Capstone Demo")
    print(f"  Total: {total_time:.1f}s, dt={dt}, steps={n_steps}")
    print(f"  Phases: stand {T_STAND}s | push+recovery {T_PUSH_PHASE}s | "
          f"shift {T_SHIFT+T_SHIFT_HOLD}s | return {T_RETURN+T_RETURN_HOLD}s")

    # ── Phase 0: Settle (not counted in demo) ────────────────────────
    print("  Settling 1s ...")
    for _ in range(int(1.0 / dt)):
        bias = mj_data.qfrc_bias[6:35]
        mj_data.ctrl[:] = CTRL_STAND + bias / KP_SERVO
        mujoco.mj_step(mj_model, mj_data)

    # Record initial state
    com_init = mj_data.subtree_com[0].copy()
    pelvis_z_init = mj_data.qpos[2]
    lf_init = mj_data.site_xpos[lf_site].copy()
    rf_init = mj_data.site_xpos[rf_site].copy()

    print(f"  Settled: pelvis_z={pelvis_z_init:.4f}, "
          f"CoM=[{com_init[0]:.4f}, {com_init[1]:.4f}, {com_init[2]:.4f}]")

    # ── Pre-compute IK for weight shift ──────────────────────────────
    print("  Pre-computing IK trajectories ...")
    pin_q = mj_qpos_to_pin(mj_data.qpos.copy())
    pin.forwardKinematics(pin_model, pin_data, pin_q)
    pin.updateFramePlacements(pin_model, pin_data)
    pin.centerOfMass(pin_model, pin_data, pin_q)

    lf_frame = pin_model.getFrameId("left_foot")
    rf_frame = pin_model.getFrameId("right_foot")
    lf_ref_se3 = pin_data.oMf[lf_frame].copy()
    rf_ref_se3 = pin_data.oMf[rf_frame].copy()
    com_ref = pin_data.com[0].copy()

    # Shift left trajectory
    shift_waypoints = precompute_shift_trajectory(
        pin_model, pin_data, pin_q, com_ref, lf_ref_se3, rf_ref_se3,
        shift_y=COM_SHIFT_Y,
    )
    print(f"    Shift trajectory: {len(shift_waypoints)} waypoints")

    # Return trajectory (reverse = shift from shifted back to center)
    # We reverse the waypoints list
    return_waypoints = list(reversed(shift_waypoints))
    print(f"    Return trajectory: {len(return_waypoints)} waypoints (reversed)")

    # ── Simulation ───────────────────────────────────────────────────
    times: list[float] = []
    pelvis_z_log: list[float] = []
    com_x_log: list[float] = []
    com_y_log: list[float] = []
    phase_log: list[str] = []

    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(mj_model, RENDER_HEIGHT, RENDER_WIDTH)
    cam = mujoco.MjvCamera()
    cam.distance = 2.5
    cam.elevation = -15.0
    cam.azimuth = 150.0
    cam.lookat[:] = [0.0, 0.0, 0.7]

    # Phase boundaries
    t1 = T_STAND
    t2 = t1 + T_PUSH_PHASE
    t3 = t2 + T_SHIFT
    t4 = t3 + T_SHIFT_HOLD
    t5 = t4 + T_RETURN
    t6 = t5 + T_RETURN_HOLD

    # Standing joint target
    q_stand_joints = CTRL_STAND.copy()

    print("  Simulating ...")

    for step in range(n_steps):
        t = step * dt

        # ── Determine phase and joint target ──────────────────────
        if t < t1:
            # Phase 1: Standing
            phase = "stand"
            q_target = q_stand_joints.copy()

        elif t < t2:
            # Phase 2: Push + recovery
            phase = "push"
            q_target = q_stand_joints.copy()

            # Apply push
            t_in_phase = t - t1
            if t_in_phase < PUSH_DURATION:
                mj_data.xfrc_applied[torso_id, 1] = PUSH_FORCE_Y
            else:
                mj_data.xfrc_applied[torso_id, :] = 0.0

        elif t < t3:
            # Phase 3a: Shift ramp
            phase = "shift"
            s = smooth_s(t - t2, T_SHIFT)
            q_target = interp_waypoints(shift_waypoints, s)

        elif t < t4:
            # Phase 3b: Hold shifted
            phase = "hold_L"
            q_target = shift_waypoints[-1].copy()

        elif t < t5:
            # Phase 4a: Return ramp
            phase = "return"
            s = smooth_s(t - t4, T_RETURN)
            q_target = interp_waypoints(return_waypoints, s)

        else:
            # Phase 4b: Hold center
            phase = "hold_C"
            q_target = return_waypoints[-1].copy()

        # ── Build control ─────────────────────────────────────────
        ctrl = q_target.copy()

        # Lock arms at neutral
        ctrl[15:22] = ARM_NEUTRAL_LEFT
        ctrl[22:29] = ARM_NEUTRAL_RIGHT

        # Gravity feedforward + velocity damping
        bias = mj_data.qfrc_bias[6:35]
        ctrl += bias / KP_SERVO
        ctrl -= K_VEL * mj_data.qvel[6:35] / KP_SERVO

        mj_data.ctrl[:] = ctrl
        mujoco.mj_step(mj_model, mj_data)

        # ── Log ───────────────────────────────────────────────────
        times.append(t)
        pelvis_z_log.append(mj_data.qpos[2])
        com_x_log.append(mj_data.subtree_com[0, 0])
        com_y_log.append(mj_data.subtree_com[0, 1])
        phase_log.append(phase)

        # Render
        if step % render_every == 0:
            renderer.update_scene(mj_data, cam)
            frames.append(renderer.render().copy())

        # Progress
        if step > 0 and step % int(1.0 / dt) == 0:
            shift_mm = (mj_data.subtree_com[0, 1] - com_init[1]) * 1000
            print(f"    t={t:.1f}s  phase={phase:<8s}  "
                  f"pz={mj_data.qpos[2]:.3f}  shift_y={shift_mm:+.1f}mm")

    renderer.close()

    # ── Save video ────────────────────────────────────────────────────
    save_mp4(frames, MEDIA_DIR / "m5_capstone.mp4")

    # ── Generate plots ────────────────────────────────────────────────
    print("  Generating plots ...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_arr = np.array(times)
    com_y_arr = np.array(com_y_log)
    pelvis_z_arr = np.array(pelvis_z_log)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Lab 7 Capstone: Unitree G1 Locomotion Fundamentals", fontsize=14, fontweight="bold")

    # ── Top-left: Pelvis height over all phases ───────────────────────
    ax = axes[0, 0]
    ax.axvspan(0, t1, alpha=0.08, color="green", label="Stand")
    ax.axvspan(t1, t2, alpha=0.08, color="orange", label="Push+Recovery")
    ax.axvspan(t2, t4, alpha=0.08, color="blue", label="Shift Left")
    ax.axvspan(t4, t6, alpha=0.08, color="purple", label="Return Center")
    ax.plot(t_arr, pelvis_z_arr, "k-", lw=1)
    ax.axhline(pelvis_z_init, color="gray", ls="--", alpha=0.5, label=f"Init ({pelvis_z_init:.3f}m)")
    ax.set_ylabel("Pelvis Z [m]")
    ax.set_title("Pelvis Height — All Phases")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, total_time)

    # ── Top-right: CoM Y shift ────────────────────────────────────────
    ax = axes[0, 1]
    ax.axvspan(0, t1, alpha=0.08, color="green")
    ax.axvspan(t1, t2, alpha=0.08, color="orange")
    ax.axvspan(t2, t4, alpha=0.08, color="blue")
    ax.axvspan(t4, t6, alpha=0.08, color="purple")
    ax.plot(t_arr, (com_y_arr - com_init[1]) * 1000, "b-", lw=1.5)
    ax.axhline(COM_SHIFT_Y * 1000, color="red", ls="--", alpha=0.5,
               label=f"Target ({COM_SHIFT_Y*1000:.0f}mm)")
    ax.axhline(0, color="gray", ls=":", alpha=0.3)
    ax.set_ylabel("CoM Y shift [mm]")
    ax.set_title("CoM Lateral Displacement")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, total_time)

    # ── Bottom-left: LIPM CoM X ──────────────────────────────────────
    ax = axes[1, 0]
    try:
        generate_lipm_plot(ax, axes[1, 1])
    except Exception as e:
        ax.text(0.5, 0.5, f"LIPM plot failed:\n{e}",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        axes[1, 1].text(0.5, 0.5, f"LIPM plot failed:\n{e}",
                        transform=axes[1, 1].transAxes, ha="center", va="center", fontsize=10)

    fig.tight_layout()
    plot_path = MEDIA_DIR / "m5_plots.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"  Plots saved: {plot_path}")

    # ── Summary ───────────────────────────────────────────────────────
    pz_min = pelvis_z_arr.min()
    pz_max = pelvis_z_arr.max()
    com_shift_max = (com_y_arr - com_init[1]).max() * 1000
    com_shift_final = (com_y_arr[-1] - com_init[1]) * 1000

    print(f"\n{'='*80}")
    print("  M5 CAPSTONE RESULTS")
    print(f"{'='*80}")
    print(f"  {'Metric':<45s} {'Value':<25s}")
    print(f"{'-'*80}")
    print(f"  {'Pelvis Z range':<45s} {f'{pz_min:.3f} — {pz_max:.3f}m':<25s}")
    print(f"  {'Max CoM Y shift':<45s} {f'{com_shift_max:.1f}mm':<25s}")
    print(f"  {'Final CoM Y shift (should ≈ 0)':<45s} {f'{com_shift_final:.1f}mm':<25s}")
    print(f"  {'Robot standing throughout':<45s} {'YES' if pz_min > 0.6 else 'NO':<25s}")
    print(f"  {'Video':<45s} {'media/m5_capstone.mp4':<25s}")
    print(f"  {'Plots':<45s} {'media/m5_plots.png':<25s}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
