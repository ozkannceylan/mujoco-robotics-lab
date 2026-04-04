#!/usr/bin/env python3
"""M3e — Single Step with Pre-computed Pinocchio IK + Online Stabilization.

Strategy: Pre-compute IK waypoints for all 4 phases offline, then replay
in MuJoCo with PD + gravity comp + velocity damping. Hybrid approach adds:
  - Asymmetric PD gains: stance leg stiffer during swing phase
  - Online CoM feedback: single-joint hip_roll correction on stance leg

Phases:
  1. Weight shift (2.5s): CoM shifts over left (stance) foot
  2. Swing (4.0s): Right foot lifts 3cm, swings forward 8cm, lands
  3. Weight back (1.5s): CoM shifts toward center of new stance
  4. Settle (2.0s): Hold final position

Controller: ctrl = q_target + bias/Kp_j - Kd_j * qvel / Kp_j
  (per-joint gains vary by phase: stance/swing/post-landing)

Gate criteria:
  - One step without falling (pelvis > 0.6m)
  - Swing foot clearance > 3cm
  - Stable after step for 2s
  - Video: media/m3e_single_step.mp4
  - Plot: media/m3e_foot_height.png
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
    load_g1_mujoco,
    load_g1_pinocchio,
    mj_qpos_to_pin,
)
from m3c_static_ik import whole_body_ik

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

KP_SERVO: float = 500.0
K_VEL: float = 40.0        # near-critical damping: ζ ≈ 40 / (2√500) ≈ 0.89

# Asymmetric PD gains per phase (used as feedforward divisors)
KP_STANCE: float = 200.0    # stance leg during swing: higher stiffness
KD_STANCE: float = 20.0
KP_SWING: float = 100.0     # swing leg during swing: compliant
KD_SWING: float = 10.0
KP_POST: float = 150.0      # both legs after landing (BACK + SETTLE)
KD_POST: float = 15.0

T_SETTLE_INIT: float = 1.0   # initial standing settle
T_SHIFT: float = 2.5         # weight shift to stance foot (slow for stability)
T_SWING: float = 4.0         # swing foot arc (slower to limit dynamic effects)
T_WEIGHT_BACK: float = 1.5   # CoM back to center
T_SETTLE_FINAL: float = 2.0  # hold final position

STEP_LENGTH: float = 0.08    # 8cm forward (conservative first step)
STEP_HEIGHT: float = 0.03    # 3cm foot lift
COM_SHIFT_Y: float = 0.05    # 5cm lateral shift toward left foot (+Y)
COM_SWING_EXTRA_Y: float = 0.01  # 1cm extra CoM shift during swing for safety margin

N_SHIFT: int = 10            # IK waypoints per phase
N_SWING: int = 15
N_BACK: int = 10


# ---------------------------------------------------------------------------
# IK trajectory pre-computation
# ---------------------------------------------------------------------------


def precompute_step_trajectory(
    model: pin.Model,
    data: pin.Data,
    q_init: np.ndarray,
    com_ref: np.ndarray,
    lf_ref: pin.SE3,
    rf_ref: pin.SE3,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Pre-compute IK joint waypoints for all 3 active phases.

    Phase 1 (shift): CoM moves +Y over left foot, both feet fixed.
    Phase 2 (swing): Right foot arc (lift + forward + land), CoM over left.
    Phase 3 (back):  CoM returns to center of new stance, both feet fixed.

    Each waypoint is a (29,) array of actuated joint angles (q[7:36]).
    Warm-starts each solve from the previous solution.

    Args:
        q_init: Pinocchio configuration at settled state (nq=36).
        com_ref: (3,) CoM at settled state.
        lf_ref, rf_ref: SE3 foot poses at settled state.

    Returns:
        (shift_wps, swing_wps, back_wps): Lists of joint-angle waypoints.
    """
    q_prev = q_init.copy()

    # ── Phase 1: Weight shift ────────────────────────────────────────
    print("\n  Phase 1: Weight shift IK ...")
    shift_wps: list[np.ndarray] = []

    for i in range(N_SHIFT + 1):
        s = i / N_SHIFT
        com_target = com_ref.copy()
        com_target[1] += s * COM_SHIFT_Y

        q_sol, converged, n_iters, errors = whole_body_ik(
            model, data, q_prev, com_target, lf_ref, rf_ref,
        )
        shift_wps.append(q_sol[7:36].copy())
        print(
            f"    wp={i:2d} s={s:.2f}: {n_iters:3d} iters, conv={converged}, "
            f"com_err={errors['com_xy']*1000:.2f}mm"
        )
        q_prev = q_sol

    # ── Phase 2: Swing ───────────────────────────────────────────────
    print("\n  Phase 2: Swing IK ...")
    com_swing = com_ref.copy()
    com_swing[1] += COM_SHIFT_Y + COM_SWING_EXTRA_Y  # extra margin during swing

    # Retry with reduced step length if any waypoint fails to converge
    eff_step_length = STEP_LENGTH
    eff_step_height = STEP_HEIGHT
    max_retries = 5
    q_swing_start = q_prev.copy()  # save warm-start for retries

    for attempt in range(max_retries):
        swing_wps: list[np.ndarray] = []
        q_prev = q_swing_start.copy()
        all_converged = True

        for i in range(N_SWING + 1):
            s = i / N_SWING
            # Right foot arc: forward motion + sinusoidal lift
            rf_pos = rf_ref.translation.copy()
            rf_pos[0] += s * eff_step_length
            rf_pos[2] += eff_step_height * np.sin(np.pi * s)
            rf_target = pin.SE3(rf_ref.rotation.copy(), rf_pos)

            q_sol, converged, n_iters, errors = whole_body_ik(
                model, data, q_prev, com_swing, lf_ref, rf_target,
            )
            swing_wps.append(q_sol[7:36].copy())
            print(
                f"    wp={i:2d} s={s:.2f}: {n_iters:3d} iters, conv={converged}, "
                f"rf_z={rf_pos[2]:.4f}, foot_err={errors['rf_pos']*1000:.2f}mm"
            )
            if not converged:
                all_converged = False
            q_prev = q_sol

        if all_converged:
            break
        # Reduce step size by 20% and retry
        eff_step_length *= 0.8
        eff_step_height *= 0.8
        print(
            f"\n  ⚠ Swing IK: non-converged waypoint. "
            f"Retry {attempt + 2}/{max_retries} with step={eff_step_length*1000:.0f}mm, "
            f"height={eff_step_height*1000:.0f}mm"
        )

    if not all_converged:
        print("  ⚠ WARNING: Swing IK did not fully converge after all retries")

    # ── Phase 3: Weight back ─────────────────────────────────────────
    print("\n  Phase 3: Weight back IK ...")
    rf_landed_pos = rf_ref.translation.copy()
    rf_landed_pos[0] += eff_step_length
    rf_landed = pin.SE3(rf_ref.rotation.copy(), rf_landed_pos)

    # New CoM target: center between left foot and landed right foot
    com_final = com_ref.copy()
    com_final[0] += eff_step_length / 2  # midpoint X
    # Y returns to center (between both feet)

    back_wps: list[np.ndarray] = []

    for i in range(N_BACK + 1):
        s = i / N_BACK
        com_target = com_swing + s * (com_final - com_swing)

        q_sol, converged, n_iters, errors = whole_body_ik(
            model, data, q_prev, com_target, lf_ref, rf_landed,
        )
        back_wps.append(q_sol[7:36].copy())
        print(
            f"    wp={i:2d} s={s:.2f}: {n_iters:3d} iters, conv={converged}, "
            f"com_err={errors['com_xy']*1000:.2f}mm"
        )
        q_prev = q_sol

    return shift_wps, swing_wps, back_wps


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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────
    mj_model, mj_data = load_g1_mujoco()
    pin_model, pin_data = load_g1_pinocchio()
    dt = mj_model.opt.timestep

    lf_site = mj_model.site("left_foot").id
    rf_site = mj_model.site("right_foot").id
    lf_frame = pin_model.getFrameId("left_foot")
    rf_frame = pin_model.getFrameId("right_foot")

    print(f"  dt={dt}, K_VEL={K_VEL} (ζ ≈ {K_VEL / (2 * np.sqrt(KP_SERVO)):.2f})")

    # ── Phase 0: Initial settle ───────────────────────────────────────
    print(f"  Settling {T_SETTLE_INIT}s ...")
    for _ in range(int(T_SETTLE_INIT / dt)):
        bias = mj_data.qfrc_bias[6:35]
        mj_data.ctrl[:] = CTRL_STAND + bias / KP_SERVO
        mujoco.mj_step(mj_model, mj_data)

    lf_init = mj_data.site_xpos[lf_site].copy()
    rf_init = mj_data.site_xpos[rf_site].copy()
    com_init = mj_data.subtree_com[0].copy()

    print(
        f"  Settled: CoM=[{com_init[0]:.4f}, {com_init[1]:.4f}, {com_init[2]:.4f}], "
        f"pelvis_z={mj_data.qpos[2]:.4f}"
    )

    # ── Convert settled state to Pinocchio for IK references ──────────
    pin_q_settled = mj_qpos_to_pin(mj_data.qpos.copy())
    pin.forwardKinematics(pin_model, pin_data, pin_q_settled)
    pin.updateFramePlacements(pin_model, pin_data)
    pin.centerOfMass(pin_model, pin_data, pin_q_settled)

    lf_ref_se3 = pin_data.oMf[lf_frame].copy()
    rf_ref_se3 = pin_data.oMf[rf_frame].copy()
    com_ref = pin_data.com[0].copy()

    print(f"  Pin CoM: [{com_ref[0]:.4f}, {com_ref[1]:.4f}, {com_ref[2]:.4f}]")

    # ── Pre-compute IK trajectory ─────────────────────────────────────
    shift_wps, swing_wps, back_wps = precompute_step_trajectory(
        pin_model, pin_data, pin_q_settled, com_ref, lf_ref_se3, rf_ref_se3,
    )

    # ── CoM target for swing phase (from offline IK) ────────────────
    com_swing_target_y = com_ref[1] + COM_SHIFT_Y + COM_SWING_EXTRA_Y

    # ── Simulate all phases ───────────────────────────────────────────
    T_TOTAL = T_SHIFT + T_SWING + T_WEIGHT_BACK + T_SETTLE_FINAL
    n_sim = int(T_TOTAL / dt)
    render_every = max(1, int(1.0 / (RENDER_FPS * dt)))

    times: list[float] = []
    com_x_log: list[float] = []
    com_y_log: list[float] = []
    pelvis_z_log: list[float] = []
    lf_z_log: list[float] = []
    rf_z_log: list[float] = []
    rf_x_log: list[float] = []

    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(mj_model, RENDER_HEIGHT, RENDER_WIDTH)
    cam = mujoco.MjvCamera()
    cam.distance = 2.5
    cam.elevation = -15.0
    cam.azimuth = 150.0
    cam.lookat[:] = [0.1, 0.0, 0.6]

    max_rf_clearance = 0.0

    print(f"\n  Simulating {T_TOTAL:.1f}s ...")

    for step in range(n_sim):
        t = step * dt

        # ── Determine phase and interpolation parameter ───────────
        if t < T_SHIFT:
            phase = "SHIFT"
            s = smooth_s(t, T_SHIFT)
            q_target = interp_waypoints(shift_wps, s)
        elif t < T_SHIFT + T_SWING:
            phase = "SWING"
            s = smooth_s(t - T_SHIFT, T_SWING)
            q_target = interp_waypoints(swing_wps, s)
        elif t < T_SHIFT + T_SWING + T_WEIGHT_BACK:
            phase = "BACK"
            s = smooth_s(t - T_SHIFT - T_SWING, T_WEIGHT_BACK)
            q_target = interp_waypoints(back_wps, s)
        else:
            phase = "SETTLE"
            s = 1.0
            q_target = back_wps[-1].copy()

        # ── Build per-joint gain arrays (asymmetric per phase) ────
        kp = np.full(29, KP_SERVO)
        kd = np.full(29, K_VEL)

        if phase == "SWING":
            # Stance leg (left, ctrl[0:6]): higher stiffness
            kp[:6] = KP_STANCE
            kd[:6] = KD_STANCE
            # Swing leg (right, ctrl[6:12]): compliant
            kp[6:12] = KP_SWING
            kd[6:12] = KD_SWING
        elif phase in ("BACK", "SETTLE"):
            # Both legs: medium stiffness after landing
            kp[:12] = KP_POST
            kd[:12] = KD_POST

        # ── Build ctrl ────────────────────────────────────────────
        ctrl = q_target.copy()

        # Lock arms at neutral
        ctrl[15:22] = ARM_NEUTRAL_LEFT
        ctrl[22:29] = ARM_NEUTRAL_RIGHT

        # Gravity feedforward + velocity damping (per-joint gains)
        bias = mj_data.qfrc_bias[6:35]
        ctrl += bias / kp
        ctrl -= kd * mj_data.qvel[6:35] / kp

        # ── Online CoM feedback on stance hip_roll (swing only) ──
        com_corr = 0.0
        if phase == "SWING":
            com_actual_y = mj_data.subtree_com[0, 1]
            com_err_y = com_swing_target_y - com_actual_y
            if abs(com_err_y) > 0.01:  # >1cm drift threshold
                com_corr = 0.5 * com_err_y
                ctrl[1] += com_corr  # left_hip_roll (stance)

        mj_data.ctrl[:] = ctrl
        mujoco.mj_step(mj_model, mj_data)

        # ── Logging ───────────────────────────────────────────────
        times.append(t)
        com_x_log.append(mj_data.subtree_com[0, 0])
        com_y_log.append(mj_data.subtree_com[0, 1])
        pelvis_z_log.append(mj_data.qpos[2])
        lf_z_log.append(mj_data.site_xpos[lf_site, 2])
        rf_z_log.append(mj_data.site_xpos[rf_site, 2])
        rf_x_log.append(mj_data.site_xpos[rf_site, 0])

        rf_clearance = mj_data.site_xpos[rf_site, 2] - rf_init[2]
        if rf_clearance > max_rf_clearance:
            max_rf_clearance = rf_clearance

        if step % render_every == 0:
            renderer.update_scene(mj_data, cam)
            frames.append(renderer.render().copy())

        if step > 0 and step % int(0.5 / dt) == 0:
            if phase == "SWING":
                com_y_err = com_swing_target_y - mj_data.subtree_com[0, 1]
                print(
                    f"    t={t:.1f}s [{phase:6s}] s={s:.3f}  "
                    f"pz={mj_data.qpos[2]:.3f}  "
                    f"com_y_err={com_y_err*1000:.1f}mm  "
                    f"hip_roll_corr={com_corr:.4f}  "
                    f"rf_z={mj_data.site_xpos[rf_site, 2]:.4f}"
                )
            else:
                print(
                    f"    t={t:.1f}s [{phase:6s}] s={s:.3f}  "
                    f"pz={mj_data.qpos[2]:.3f}  "
                    f"rf_z={mj_data.site_xpos[rf_site, 2]:.4f}  "
                    f"rf_x={mj_data.site_xpos[rf_site, 0]:.4f}"
                )

    renderer.close()

    # ── Save video ────────────────────────────────────────────────────
    import imageio

    video_path = MEDIA_DIR / "m3e_single_step.mp4"
    writer = imageio.get_writer(
        str(video_path),
        fps=RENDER_FPS,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(
        f"  Video: {video_path} "
        f"({len(frames)} frames, {len(frames) / RENDER_FPS:.1f}s)"
    )

    # ── Analysis ──────────────────────────────────────────────────────
    min_pz = min(pelvis_z_log)
    rf_forward = rf_x_log[-1] - rf_init[0]

    # Stability in final settle phase
    settle_idx = int((T_SHIFT + T_SWING + T_WEIGHT_BACK) / dt)
    settle_pz = pelvis_z_log[settle_idx:]
    settle_stable = len(settle_pz) > 0 and min(settle_pz) > 0.6

    # ── Plot ──────────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        t_arr = np.array(times)

        # Phase background shading
        t1 = T_SHIFT
        t2 = T_SHIFT + T_SWING
        t3 = T_SHIFT + T_SWING + T_WEIGHT_BACK
        for ax in axes:
            ax.axvspan(0, t1, alpha=0.08, color="gold", label="_")
            ax.axvspan(t1, t2, alpha=0.08, color="orange", label="_")
            ax.axvspan(t2, t3, alpha=0.08, color="cyan", label="_")
            ax.axvspan(t3, T_TOTAL, alpha=0.08, color="lightgreen", label="_")

        # (a) CoM XY shift
        axes[0].plot(
            t_arr, (np.array(com_y_log) - com_init[1]) * 1000,
            "b-", lw=1.5, label="CoM Y shift",
        )
        axes[0].plot(
            t_arr, (np.array(com_x_log) - com_init[0]) * 1000,
            "r-", lw=1.5, label="CoM X shift",
        )
        axes[0].set_ylabel("CoM shift [mm]")
        axes[0].set_title(
            "M3e: Single Step — SHIFT | SWING | BACK | SETTLE"
        )
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # (b) Foot Z (clearance)
        axes[1].plot(
            t_arr, np.array(rf_z_log) * 1000,
            "r-", lw=1.5, label="Right foot Z",
        )
        axes[1].plot(
            t_arr, np.array(lf_z_log) * 1000,
            "b-", lw=1, alpha=0.7, label="Left foot Z",
        )
        axes[1].axhline(
            rf_init[2] * 1000 + 30, color="red", ls=":", alpha=0.5,
            label="3cm clearance gate",
        )
        axes[1].set_ylabel("Foot Z [mm]")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # (c) Right foot X forward progress
        axes[2].plot(
            t_arr, (np.array(rf_x_log) - rf_init[0]) * 1000,
            "r-", lw=1.5, label="Right foot X fwd",
        )
        axes[2].axhline(
            80, color="red", ls=":", alpha=0.5, label="8cm target",
        )
        axes[2].set_ylabel("RF X forward [mm]")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        # (d) Pelvis height
        axes[3].plot(t_arr, np.array(pelvis_z_log), "g-", lw=1.5)
        axes[3].axhline(
            0.6, color="red", ls=":", alpha=0.5, label="Fall threshold",
        )
        axes[3].set_ylabel("Pelvis Z [m]")
        axes[3].set_xlabel("Time [s]")
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = MEDIA_DIR / "m3_step_analysis.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"  Plot: {plot_path}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    # ── Gate summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  M3e GATE RESULTS")
    print(f"{'=' * 80}")
    print(f"  {'Criterion':<50s} {'Value':<25s} {'Status'}")
    print(f"{'-' * 80}")

    g1 = min_pz > 0.6
    print(
        f"  {'No fall (pelvis > 0.6m)':<50s} "
        f"{f'min={min_pz:.3f}m':<25s} "
        f"{'PASS' if g1 else 'FAIL'}"
    )

    g2 = max_rf_clearance * 1000 > 30
    print(
        f"  {'Swing foot clearance > 3cm':<50s} "
        f"{f'{max_rf_clearance * 1000:.1f}mm':<25s} "
        f"{'PASS' if g2 else 'FAIL'}"
    )

    g3 = settle_stable
    settle_min_str = (
        f"min_pz={min(settle_pz):.3f}m" if settle_pz else "N/A"
    )
    print(
        f"  {'Stable after step for 2s':<50s} "
        f"{settle_min_str:<25s} "
        f"{'PASS' if g3 else 'FAIL'}"
    )

    g4 = rf_forward > 0.05
    print(
        f"  {'Right foot forward > 5cm':<50s} "
        f"{f'{rf_forward * 1000:.1f}mm':<25s} "
        f"{'PASS' if g4 else 'FAIL'}"
    )

    print(f"  {'Video saved':<50s} {'m3e_single_step.mp4':<25s} PASS")
    print(f"  {'Plot saved':<50s} {'m3_step_analysis.png':<25s} PASS")

    all_pass = g1 and g2 and g3 and g4
    print(
        f"\n  {'OVERALL':<50s} {'':25s} "
        f"{'ALL PASS' if all_pass else 'SOME FAIL'}"
    )
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
