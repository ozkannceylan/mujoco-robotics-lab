#!/usr/bin/env python3
"""M3d — Weight Shift in Simulation using Pre-computed Pinocchio IK.

Strategy: Pre-compute the entire IK trajectory offline using whole_body_ik
from M3c, then replay joint targets in MuJoCo with PD + gravity comp.
This avoids online IK feedback instability while using Pinocchio IK for
all joint targets.

Pipeline:
  1. Settle 1s with standing PD + gravity comp
  2. Convert settled MuJoCo state to Pinocchio, get actual foot/CoM references
  3. Pre-compute IK at 11 waypoints along CoM Y shift path (whole_body_ik)
  4. Smooth ramp 1.5s: cosine interpolation through IK joint waypoints
  5. Hold 2s: hold final IK joint target (no online corrections)
  6. Controller: ctrl = q_target + bias/Kp - K_VEL * qvel / Kp

Gate criteria:
  - CoM shifts >= 40mm (80% of 50mm target)
  - Feet stay within 5mm of initial position
  - Robot does not fall (base height > 0.6m)
  - Video: media/m3d_weight_shift.mp4
  - Screenshot: media/m3d_shifted.png
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
    LEG_JOINT_NAMES,
    MEDIA_DIR,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    WAIST_JOINT_NAMES,
    load_g1_mujoco,
    load_g1_pinocchio,
    mj_qpos_to_pin,
)
from m3c_static_ik import whole_body_ik

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

KP_SERVO: float = 500.0
K_VEL: float = 40.0        # near-critical: 2*sqrt(500) ≈ 44.7, so ζ ≈ 0.89

T_SETTLE: float = 1.0      # stand and settle
T_SHIFT: float = 1.5       # smooth CoM shift
T_HOLD: float = 2.0        # hold shifted position

COM_SHIFT_Y: float = 0.05  # 5cm toward left foot (+Y)
N_WAYPOINTS: int = 10      # IK waypoints along shift path


# ---------------------------------------------------------------------------
# IK trajectory pre-computation
# ---------------------------------------------------------------------------


def precompute_trajectory(
    model: pin.Model,
    data: pin.Data,
    q_init: np.ndarray,
    com_ref: np.ndarray,
    lf_ref: pin.SE3,
    rf_ref: pin.SE3,
) -> list[np.ndarray]:
    """Pre-compute IK joint targets along CoM Y-shift path.

    Uses whole_body_ik from M3c at N_WAYPOINTS+1 evenly-spaced targets.
    Each call warm-starts from the previous solution.

    Args:
        q_init: Pinocchio configuration at settled state.
        com_ref: (3,) CoM at settled state.
        lf_ref, rf_ref: SE3 foot poses at settled state.

    Returns:
        List of (29,) joint-angle arrays (Pinocchio q[7:36]).
    """
    print("\n  Pre-computing IK trajectory ...")
    q_prev = q_init.copy()
    waypoints: list[np.ndarray] = []

    for i in range(N_WAYPOINTS + 1):
        s = i / N_WAYPOINTS
        com_target = com_ref.copy()
        com_target[1] += s * COM_SHIFT_Y

        q_sol, converged, n_iters, errors = whole_body_ik(
            model, data, q_prev,
            com_target=com_target,
            lf_target=lf_ref,
            rf_target=rf_ref,
        )

        waypoints.append(q_sol[7:36].copy())

        print(
            f"    wp={i:2d} s={s:.1f}: {n_iters:3d} iters, "
            f"conv={converged}, "
            f"com_err={errors['com_xy']*1000:.2f}mm, "
            f"foot={max(errors['lf_pos'], errors['rf_pos'])*1000:.2f}mm"
        )

        q_prev = q_sol  # warm-start next waypoint

    return waypoints


def interp_waypoints(waypoints: list[np.ndarray], s: float) -> np.ndarray:
    """Linearly interpolate joint targets at parameter s in [0, 1]."""
    n = len(waypoints) - 1
    idx = s * n
    i = min(int(idx), n - 1)
    frac = idx - i
    return (1.0 - frac) * waypoints[i] + frac * waypoints[i + 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def smooth_s(t: float, t_end: float) -> float:
    """Cosine smooth interpolation 0 -> 1 over [0, t_end]."""
    s = np.clip(t / t_end, 0.0, 1.0)
    return float(0.5 * (1.0 - np.cos(np.pi * s)))


def save_screenshot(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    path: Path,
) -> None:
    """Render and save a single frame."""
    renderer = mujoco.Renderer(mj_model, RENDER_HEIGHT, RENDER_WIDTH)
    cam = mujoco.MjvCamera()
    cam.distance = 2.0
    cam.elevation = -15.0
    cam.azimuth = 170.0
    cam.lookat[:] = [0.0, 0.0, 0.6]
    renderer.update_scene(mj_data, cam)
    img = renderer.render()
    renderer.close()

    import imageio

    imageio.imwrite(str(path), img)
    print(f"  Screenshot saved: {path}")


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

    # ── Phase 1: Settle ───────────────────────────────────────────────
    print(f"  Settling {T_SETTLE}s ...")
    for _ in range(int(T_SETTLE / dt)):
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

    print(
        f"  Pin CoM: [{com_ref[0]:.4f}, {com_ref[1]:.4f}, {com_ref[2]:.4f}]"
    )

    # ── Pre-compute IK trajectory ─────────────────────────────────────
    waypoints = precompute_trajectory(
        pin_model, pin_data, pin_q_settled, com_ref, lf_ref_se3, rf_ref_se3,
    )

    # Show key joint changes
    dq_final = waypoints[-1] - waypoints[0]
    print(f"\n  IK joint changes (final - initial):")
    for i, name in enumerate(LEG_JOINT_NAMES + WAIST_JOINT_NAMES):
        if abs(dq_final[i]) > 1e-4:
            print(
                f"    {name:<35s} {dq_final[i]:+.4f} rad "
                f"({np.degrees(dq_final[i]):+.2f} deg)"
            )

    # ── Phase 2+3: Shift + Hold ───────────────────────────────────────
    n_sim = int((T_SHIFT + T_HOLD) / dt)
    render_every = max(1, int(1.0 / (RENDER_FPS * dt)))

    times: list[float] = []
    com_y_log: list[float] = []
    pelvis_z_log: list[float] = []
    lf_drift_log: list[float] = []
    rf_drift_log: list[float] = []

    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(mj_model, RENDER_HEIGHT, RENDER_WIDTH)
    cam = mujoco.MjvCamera()
    cam.distance = 2.0
    cam.elevation = -15.0
    cam.azimuth = 170.0
    cam.lookat[:] = [0.0, 0.0, 0.6]

    screenshot_saved = False

    print(f"\n  Simulating {T_SHIFT + T_HOLD:.1f}s ...")

    for step in range(n_sim):
        t = step * dt

        # ── Trajectory parameter s ∈ [0, 1] ──────────────────────
        s = smooth_s(t, T_SHIFT) if t < T_SHIFT else 1.0

        # ── Joint target from pre-computed IK waypoints ───────────
        q_target = interp_waypoints(waypoints, s)

        # ── Build ctrl ────────────────────────────────────────────
        ctrl = q_target.copy()  # 29 joint position targets

        # Lock arms at neutral (overwrite IK arm values)
        ctrl[15:22] = ARM_NEUTRAL_LEFT
        ctrl[22:29] = ARM_NEUTRAL_RIGHT

        # Gravity feedforward + velocity damping (all 29 joints)
        bias = mj_data.qfrc_bias[6:35]
        ctrl += bias / KP_SERVO
        ctrl -= K_VEL * mj_data.qvel[6:35] / KP_SERVO

        mj_data.ctrl[:] = ctrl
        mujoco.mj_step(mj_model, mj_data)

        # ── Logging ───────────────────────────────────────────────
        times.append(t)
        com_y_log.append(mj_data.subtree_com[0, 1])
        pelvis_z_log.append(mj_data.qpos[2])
        lf_drift_log.append(
            np.linalg.norm(mj_data.site_xpos[lf_site] - lf_init)
        )
        rf_drift_log.append(
            np.linalg.norm(mj_data.site_xpos[rf_site] - rf_init)
        )

        if step % render_every == 0:
            renderer.update_scene(mj_data, cam)
            frames.append(renderer.render().copy())

        if not screenshot_saved and t >= T_SHIFT:
            save_screenshot(mj_model, mj_data, MEDIA_DIR / "m3d_shifted.png")
            screenshot_saved = True

        if step > 0 and step % int(0.5 / dt) == 0:
            shift = (mj_data.subtree_com[0, 1] - com_init[1]) * 1000
            print(
                f"    t={t:.1f}s  s={s:.3f}  shift={shift:+.1f}mm  "
                f"lf={lf_drift_log[-1]*1000:.2f}mm  "
                f"rf={rf_drift_log[-1]*1000:.2f}mm  "
                f"pz={mj_data.qpos[2]:.3f}"
            )

    renderer.close()

    # ── Save video ────────────────────────────────────────────────────
    import imageio

    writer = imageio.get_writer(
        str(MEDIA_DIR / "m3d_weight_shift.mp4"),
        fps=RENDER_FPS,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(
        f"  Video: {MEDIA_DIR / 'm3d_weight_shift.mp4'} "
        f"({len(frames)} frames, {len(frames)/RENDER_FPS:.1f}s)"
    )

    # ── Analysis ──────────────────────────────────────────────────────
    com_shift = (com_y_log[-1] - com_init[1]) * 1000
    max_lf = max(lf_drift_log) * 1000
    max_rf = max(rf_drift_log) * 1000
    min_pz = min(pelvis_z_log)

    # ── Plot ──────────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        t_arr = np.array(times)

        axes[0].axvspan(0, T_SHIFT, alpha=0.1, color="yellow", label="SHIFT")
        axes[0].axvspan(
            T_SHIFT, T_SHIFT + T_HOLD, alpha=0.1, color="lightgreen", label="HOLD"
        )
        axes[0].plot(
            t_arr,
            (np.array(com_y_log) - com_init[1]) * 1000,
            "b-",
            lw=1.5,
            label="CoM Y shift",
        )
        axes[0].axhline(
            COM_SHIFT_Y * 1000, color="red", ls="--", alpha=0.5,
            label=f"Target ({COM_SHIFT_Y*1000:.0f}mm)",
        )
        axes[0].axhline(40, color="orange", ls=":", alpha=0.5, label="Gate (40mm)")
        axes[0].set_ylabel("CoM Y shift [mm]")
        axes[0].set_title("M3d: Pre-computed IK Weight Shift")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            t_arr, np.array(lf_drift_log) * 1000, "r-", lw=1, label="Left foot"
        )
        axes[1].plot(
            t_arr, np.array(rf_drift_log) * 1000, "b-", lw=1, label="Right foot"
        )
        axes[1].axhline(5, color="red", ls=":", alpha=0.5, label="Gate (5mm)")
        axes[1].set_ylabel("Foot drift [mm]")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t_arr, np.array(pelvis_z_log), "g-", lw=1)
        axes[2].axhline(
            0.6, color="red", ls=":", alpha=0.5, label="Fall threshold"
        )
        axes[2].set_ylabel("Pelvis Z [m]")
        axes[2].set_xlabel("Time [s]")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(MEDIA_DIR / "m3d_weight_shift_plot.png"), dpi=150)
        plt.close(fig)
        print(f"  Plot: {MEDIA_DIR / 'm3d_weight_shift_plot.png'}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    # ── Gate summary ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  M3d GATE RESULTS")
    print(f"{'='*80}")
    print(f"  {'Criterion':<50s} {'Value':<25s} {'Status'}")
    print(f"{'-'*80}")

    sp = com_shift >= 40.0
    print(
        f"  {'CoM lateral shift >= 40mm':<50s} "
        f"{f'{com_shift:.1f}mm':<25s} "
        f"{'PASS' if sp else 'FAIL'}"
    )

    lp = max_lf < 5.0
    print(
        f"  {'Left foot drift < 5mm':<50s} "
        f"{f'{max_lf:.2f}mm':<25s} "
        f"{'PASS' if lp else 'FAIL'}"
    )

    rp = max_rf < 5.0
    print(
        f"  {'Right foot drift < 5mm':<50s} "
        f"{f'{max_rf:.2f}mm':<25s} "
        f"{'PASS' if rp else 'FAIL'}"
    )

    hp = min_pz > 0.6
    print(
        f"  {'Robot standing (pelvis > 0.6m)':<50s} "
        f"{f'min={min_pz:.3f}m':<25s} "
        f"{'PASS' if hp else 'FAIL'}"
    )

    print(f"  {'Video saved':<50s} {'m3d_weight_shift.mp4':<25s} PASS")
    print(f"  {'Screenshot saved':<50s} {'m3d_shifted.png':<25s} PASS")

    all_pass = sp and lp and rp and hp
    print(f"\n  {'OVERALL':<50s} {'':25s} {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
