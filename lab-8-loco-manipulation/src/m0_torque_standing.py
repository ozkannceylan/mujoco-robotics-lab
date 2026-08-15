"""Lab 8 — M0: Torque-actuated G1 bring-up (gate demo).

Runs the M0 gate end to end and writes its evidence:

1. Model cross-validation — Pinocchio vs MuJoCo gravity, mass matrix, CoM.
2. Gravity-mode ablation — three controller variants over the full 10 s hold,
   reported as a table (this is the milestone's actual finding, not a detour).
3. Gate run — 10 s stand with the selected controller, video + plots.

Usage:
    MUJOCO_GL=egl python3 lab-8-loco-manipulation/src/m0_torque_standing.py

Gate criteria (tasks/PLAN.md M0):
    * stands 10 s without falling
    * CoM horizontal drift < 30 mm
    * g(q) cross-validation vs MuJoCo qfrc_bias < 1e-6 (relative)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab8_common import (  # noqa: E402
    DT,
    MEDIA_DIR,
    NU,
    PELVIS_MJCF_Z,
    Q_STAND_JOINTS,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    TOTAL_MASS,
    com_position,
    dense_mass_matrix,
    foot_contact_state,
    joint_torques_to_ctrl,
    load_g1_pinocchio,
    load_g1_torque_mujoco,
    mj_state_to_pin,
    support_polygon_margin,
)
from g1_torque_model import torque_limits  # noqa: E402
from standing_controller import GravityMode, StandingController  # noqa: E402

HOLD_SECONDS = 10.0
COM_DRIFT_LIMIT_MM = 30.0
PELVIS_FALL_THRESHOLD = 0.50  # m — below this the robot has collapsed
VIDEO_PATH = MEDIA_DIR / "m0_torque_standing.mp4"
PLOT_PATH = MEDIA_DIR / "m0_standing_metrics.png"
RENDER_EVERY = int(round(1.0 / (RENDER_FPS * DT)))  # 1 kHz sim → 60 fps video


# ---------------------------------------------------------------------------
# 1. Model cross-validation
# ---------------------------------------------------------------------------


def cross_validate(mj_model, mj_data, pin_model, pin_data, n_random: int = 5) -> dict:
    """Check that the analytical model matches the simulated body.

    Lab 5's L-6.1c: gravity compensation computed from a model of a *different*
    robot is silently wrong. So M0 verifies g(q), M(q) and the CoM before any
    controller is trusted.
    """
    rng = np.random.default_rng(0)
    g_errors: list[float] = []
    m_errors: list[float] = []
    com_errors: list[float] = []

    for trial in range(n_random + 1):
        if trial == 0:
            mj_data.qpos[7:] = Q_STAND_JOINTS
        else:
            mj_data.qpos[7:] = rng.uniform(-0.4, 0.4, NU)
            mj_data.qpos[2] = 1.2  # lift clear of the floor: free-space check
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        q, _ = mj_state_to_pin(mj_data)

        pin.computeGeneralizedGravity(pin_model, pin_data, q)
        g_pin = pin_data.g.copy()
        g_mj = mj_data.qfrc_bias.copy()  # qvel = 0 → bias is the gravity term
        scale = max(np.abs(g_mj).max(), 1.0)
        g_errors.append(float(np.abs(g_pin - g_mj).max() / scale))

        m_pin = pin.crba(pin_model, pin_data, q)
        m_pin = np.triu(m_pin) + np.triu(m_pin, 1).T
        m_mj = dense_mass_matrix(mj_model, mj_data)
        m_errors.append(float(np.abs(m_pin - m_mj).max() / max(np.abs(m_mj).max(), 1.0)))

        com_pin = pin.centerOfMass(pin_model, pin_data, q).copy()
        com_pin[2] += PELVIS_MJCF_Z  # Pinocchio base frame → world (Lab 7 offset)
        com_errors.append(float(np.linalg.norm(com_pin - com_position(mj_model, mj_data))))

    mass_pin = sum(inertia.mass for inertia in pin_model.inertias[1:])
    return {
        "gravity_rel_err": max(g_errors),
        "mass_matrix_rel_err": max(m_errors),
        "com_err_mm": max(com_errors) * 1000.0,
        "mass_pin": mass_pin,
        "mass_mj": float(mj_model.body_subtreemass[0]),
        "n_configs": n_random + 1,
    }


# ---------------------------------------------------------------------------
# 2 & 3. Standing runs
# ---------------------------------------------------------------------------


def run_stand(
    gravity_mode: GravityMode,
    kp: float = 500.0,
    kd: float = 50.0,
    seconds: float = HOLD_SECONDS,
    record: bool = False,
) -> dict:
    """Hold the standing pose under torque control; return metrics (+ video)."""
    mj_model, mj_data = load_g1_torque_mujoco(timestep=DT)
    pin_model, pin_data = load_g1_pinocchio()
    controller = StandingController(
        mj_model, pin_model, pin_data, kp=kp, kd=kd, gravity_mode=gravity_mode
    )

    n_steps = int(seconds / DT)
    com0 = com_position(mj_model, mj_data)

    log = {
        "t": np.zeros(n_steps),
        "pelvis_z": np.zeros(n_steps),
        "com_drift_mm": np.zeros(n_steps),
        "joint_err_mrad": np.zeros(n_steps),
        "tau_max": np.zeros(n_steps),
        "margin_mm": np.zeros(n_steps),
    }

    writer = renderer = camera = None
    if record:
        import imageio

        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        # Stream frames to the encoder — buffering a 10 s 720p60 run in RAM is
        # tens of GB and gets OOM-killed (Lab 5 recording lesson).
        writer = imageio.get_writer(
            str(VIDEO_PATH), fps=RENDER_FPS, codec="libx264", quality=8,
            macro_block_size=1,
        )
        renderer = mujoco.Renderer(mj_model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, 0.0, 0.8]
        camera.distance = 3.2
        camera.azimuth = 135.0
        camera.elevation = -12.0

    fell_at: float | None = None
    try:
        for step in range(n_steps):
            tau = controller.step(mj_data)

            com = com_position(mj_model, mj_data)
            log["t"][step] = step * DT
            log["pelvis_z"][step] = mj_data.qpos[2]
            log["com_drift_mm"][step] = np.linalg.norm(com[:2] - com0[:2]) * 1000.0
            log["joint_err_mrad"][step] = (
                np.abs(Q_STAND_JOINTS - mj_data.qpos[7:]).max() * 1000.0
            )
            log["tau_max"][step] = np.abs(tau).max()
            log["margin_mm"][step] = support_polygon_margin(mj_model, mj_data) * 1000.0

            if fell_at is None and mj_data.qpos[2] < PELVIS_FALL_THRESHOLD:
                fell_at = step * DT

            if writer is not None and step % RENDER_EVERY == 0:
                renderer.update_scene(mj_data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        if writer is not None:
            writer.close()

    left, right = foot_contact_state(mj_model, mj_data)
    return {
        "mode": gravity_mode.value,
        "fell": fell_at is not None,
        "fell_at": fell_at,
        "final_pelvis_z": float(mj_data.qpos[2]),
        "com_drift_mm": float(log["com_drift_mm"][-1]),
        "com_drift_max_mm": float(log["com_drift_mm"].max()),
        "joint_err_mrad": float(log["joint_err_mrad"][-1]),
        "tau_max": float(log["tau_max"].max()),
        "margin_mm": float(log["margin_mm"][-1]),
        "both_feet_down": bool(left and right),
        "log": log,
    }


def plot_metrics(log: dict, path: Path) -> None:
    """Plot the gate run: pelvis height, CoM drift, joint error, torque."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    t = log["t"]

    axes[0].plot(t, log["pelvis_z"], "C0")
    axes[0].axhline(PELVIS_FALL_THRESHOLD, color="C3", ls="--", label="fall threshold")
    axes[0].set_ylabel("pelvis z (m)")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, log["com_drift_mm"], "C1")
    axes[1].axhline(COM_DRIFT_LIMIT_MM, color="C3", ls="--", label="gate 30 mm")
    axes[1].set_ylabel("CoM drift (mm)")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, log["joint_err_mrad"], "C2")
    axes[2].set_ylabel("max |q err| (mrad)")

    axes[3].plot(t, log["tau_max"], "C4")
    axes[3].set_ylabel("max |τ| (N·m)")
    axes[3].set_xlabel("time (s)")

    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title("Lab 8 M0 — Torque-Actuated G1 Standing (contact-consistent gravity + joint PD)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full M0 gate and write evidence."""
    print("=" * 72)
    print(" Lab 8 — M0: Torque-Actuated G1 Bring-Up")
    print("=" * 72)

    mj_model, mj_data = load_g1_torque_mujoco(timestep=DT)
    pin_model, pin_data = load_g1_pinocchio()
    limits = torque_limits(mj_model)

    print("\n[1/4] Model")
    print(f"  nq={mj_model.nq}  nv={mj_model.nv}  nu={mj_model.nu}  dt={mj_model.opt.timestep*1000:.1f} ms")
    print(f"  actuator type: {'motor (torque)' if mj_model.actuator_gaintype[0] == 0 and mj_model.actuator_biastype[0] == 0 else 'NOT torque'}")
    print(f"  torque limits: {limits[:, 1].min():.0f} … {limits[:, 1].max():.0f} N·m")
    print(f"  total mass:    {TOTAL_MASS:.2f} kg")

    print("\n[2/4] Pinocchio ↔ MuJoCo cross-validation")
    cv = cross_validate(mj_model, mj_data, pin_model, pin_data)
    print(f"  configurations checked : {cv['n_configs']}")
    print(f"  max rel err g(q)       : {cv['gravity_rel_err']:.3e}")
    print(f"  max rel err M(q)       : {cv['mass_matrix_rel_err']:.3e}")
    print(f"  max CoM error          : {cv['com_err_mm']:.6f} mm")
    print(f"  mass  pin / mj         : {cv['mass_pin']:.4f} / {cv['mass_mj']:.4f} kg")

    print("\n[3/4] Gravity-mode ablation (10 s hold each)")
    ablation = [run_stand(mode) for mode in GravityMode]
    header = f"  {'mode':22s} {'result':7s} {'pelvis z':>9s} {'CoM drift':>11s} {'q err':>10s} {'|τ|max':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for result in ablation:
        verdict = "FELL" if result["fell"] else "STAND"
        print(
            f"  {result['mode']:22s} {verdict:7s} {result['final_pelvis_z']:8.3f}m "
            f"{result['com_drift_mm']:9.2f}mm {result['joint_err_mrad']:8.2f}mrad "
            f"{result['tau_max']:7.1f}Nm"
        )

    print("\n[4/4] Gate run + recording (contact-consistent gravity)")
    gate = run_stand(GravityMode.CONTACT_CONSISTENT, record=True)
    plot_metrics(gate["log"], PLOT_PATH)
    print(f"  video: {VIDEO_PATH}  ({VIDEO_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"  plot : {PLOT_PATH}")

    checks = [
        ("Stands 10 s without falling", not gate["fell"], "no fall" if not gate["fell"] else f"fell at {gate['fell_at']:.2f}s"),
        ("CoM horizontal drift < 30 mm", gate["com_drift_max_mm"] < COM_DRIFT_LIMIT_MM, f"{gate['com_drift_max_mm']:.2f} mm"),
        ("Both feet in contact at end", gate["both_feet_down"], "yes" if gate["both_feet_down"] else "no"),
        ("CoM inside support polygon", gate["margin_mm"] > 0, f"{gate['margin_mm']:.1f} mm margin"),
        ("g(q) parity < 1e-6 (relative)", cv["gravity_rel_err"] < 1e-6, f"{cv['gravity_rel_err']:.2e}"),
        ("M(q) parity < 1e-6 (relative)", cv["mass_matrix_rel_err"] < 1e-6, f"{cv['mass_matrix_rel_err']:.2e}"),
        ("Torque command authority", int(mj_model.actuator_gaintype[0]) == 0, "motor actuators"),
    ]

    print("\n" + "=" * 72)
    print(" M0 GATE")
    print("=" * 72)
    print(f" {'criterion':34s} {'result':>8s}   measured")
    print(" " + "-" * 69)
    for name, passed, detail in checks:
        print(f" {name:34s} {'PASS' if passed else 'FAIL':>8s}   {detail}")
    all_passed = all(passed for _, passed, _ in checks)
    print("=" * 72)
    print(f" M0: {'PASS — torque authority established' if all_passed else 'FAIL'}")
    print("=" * 72)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
