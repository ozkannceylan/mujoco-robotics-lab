"""Lab 8 — M1: whole-body QP standing reach (gate demo).

The G1 stands on both feet under torque control while its right hand traces a
circle in the air. Balance, stance feet, hand and posture are resolved together
by one inverse-dynamics QP each tick; the QP's contact wrenches make "keep the
CoM over the feet" a constraint the solver can enforce rather than a wish.

Usage:
    MUJOCO_GL=egl python3 lab-8-loco-manipulation/src/m1_standing_reach.py

Gate criteria (tasks/PLAN.md M1):
    * hand tracking RMS < 20 mm over the trajectory
    * CoM stays ≥ 20 mm inside the support polygon
    * stance feet move < 5 mm
    * no fall
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from g1_torque_model import torque_limits  # noqa: E402
from lab8_common import (  # noqa: E402
    DT,
    MEDIA_DIR,
    Q_STAND_JOINTS,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    com_position,
    load_g1_pinocchio,
    load_g1_torque_mujoco,
    mj_state_to_pin,
    support_polygon_margin,
)
from wb_id_qp import ContactSpec, WholeBodyIDQP  # noqa: E402
from wb_tasks import CoMTask, FramePositionTask, PostureTask, TaskStack  # noqa: E402

HAND_FRAME = "right_wrist_yaw_link"
LEFT_FOOT = "left_ankle_roll_link"
RIGHT_FOOT = "right_ankle_roll_link"

SETTLE_SECONDS = 1.0     # hold the home pose before moving
REACH_SECONDS = 2.0      # blend out to the circle centre
CIRCLE_SECONDS = 8.0     # two laps
TOTAL_SECONDS = SETTLE_SECONDS + REACH_SECONDS + CIRCLE_SECONDS

REACH_OFFSET = np.array([0.18, 0.0, 0.08])   # from home hand pose [m]
CIRCLE_RADIUS = 0.10                          # m
CIRCLE_LAPS = 2.0

HAND_RMS_LIMIT_MM = 20.0
COM_MARGIN_LIMIT_MM = 20.0
FOOT_MOTION_LIMIT_MM = 5.0
PELVIS_FALL_THRESHOLD = 0.50

VIDEO_PATH = MEDIA_DIR / "m1_standing_reach.mp4"
PLOT_PATH = MEDIA_DIR / "m1_reach_metrics.png"
RENDER_EVERY = int(round(1.0 / (RENDER_FPS * DT)))


def hand_target(t: float, home: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trajectory: settle at home, blend to the circle centre, then two laps.

    Returns (position, velocity, acceleration) — the derivatives are handed to
    the task as feedforward, which is what keeps a moving target from being
    tracked with a pure lag error.
    """
    zero = np.zeros(3)
    if t < SETTLE_SECONDS:
        return home.copy(), zero, zero

    if t < SETTLE_SECONDS + REACH_SECONDS:
        alpha = (t - SETTLE_SECONDS) / REACH_SECONDS
        # C¹ raised-cosine blend: s(0)=0, s(1)=1, ṡ(0)=ṡ(1)=0.
        smooth = 0.5 * (1.0 - np.cos(np.pi * alpha))
        d_smooth = 0.5 * np.pi * np.sin(np.pi * alpha) / REACH_SECONDS
        dd_smooth = 0.5 * (np.pi / REACH_SECONDS) ** 2 * np.cos(np.pi * alpha)
        return (
            home + smooth * REACH_OFFSET,
            d_smooth * REACH_OFFSET,
            dd_smooth * REACH_OFFSET,
        )

    centre = home + REACH_OFFSET
    omega = 2.0 * np.pi * CIRCLE_LAPS / CIRCLE_SECONDS
    phase = omega * (t - SETTLE_SECONDS - REACH_SECONDS)
    position = centre + CIRCLE_RADIUS * np.array([0.0, np.sin(phase), 1.0 - np.cos(phase)])
    velocity = CIRCLE_RADIUS * omega * np.array([0.0, np.cos(phase), np.sin(phase)])
    acceleration = CIRCLE_RADIUS * omega**2 * np.array([0.0, -np.sin(phase), np.cos(phase)])
    return position, velocity, acceleration


def run(record: bool = True) -> dict:
    """Run the reach and return metrics + logs."""
    mj_model, mj_data = load_g1_torque_mujoco(timestep=DT)
    pin_model, pin_data = load_g1_pinocchio()

    stack = TaskStack(pin_model, pin_data)
    com_task = stack.add(CoMTask(pin_model, weight=1e4, gain=100.0))
    hand_task = stack.add(
        FramePositionTask(HAND_FRAME, pin_model, weight=1e3, gain=400.0)
    )
    posture_task = stack.add(PostureTask(Q_STAND_JOINTS, weight=1.0, gain=50.0))

    q, v = mj_state_to_pin(mj_data)
    stack.update_dynamics(q, v)
    com_task.set_target(com_task.current_com(pin_data))
    hand_home = hand_task.current_position(pin_data)

    left_id = pin_model.getFrameId(LEFT_FOOT)
    right_id = pin_model.getFrameId(RIGHT_FOOT)
    left_home = pin_data.oMf[left_id].translation.copy()
    right_home = pin_data.oMf[right_id].translation.copy()

    qp = WholeBodyIDQP(
        pin_model,
        pin_data,
        [ContactSpec(LEFT_FOOT), ContactSpec(RIGHT_FOOT)],
        torque_limits(mj_model),
    )

    n_steps = int(TOTAL_SECONDS / DT)
    log = {
        key: np.zeros(n_steps)
        for key in ("t", "hand_err_mm", "com_margin_mm", "foot_move_mm", "tau_max", "solve_ms")
    }
    log["hand_pos"] = np.zeros((n_steps, 3))
    log["hand_ref"] = np.zeros((n_steps, 3))

    writer = renderer = camera = None
    if record:
        import imageio

        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(VIDEO_PATH), fps=RENDER_FPS, codec="libx264", quality=8,
            macro_block_size=1,
        )
        renderer = mujoco.Renderer(mj_model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, 0.0, 0.85]
        camera.distance = 2.6
        camera.azimuth = 145.0
        camera.elevation = -10.0

    fell_at: float | None = None
    wall_start = time.time()
    try:
        for step in range(n_steps):
            t = step * DT
            target, target_vel, target_acc = hand_target(t, hand_home)
            hand_task.set_target(target, target_vel, target_acc)

            q, v = mj_state_to_pin(mj_data)
            stack.update_dynamics(q, v)
            result = qp.solve(stack, q, v)
            mj_data.ctrl[:] = result.tau
            mujoco.mj_step(mj_model, mj_data)

            hand_now = hand_task.current_position(pin_data)
            foot_move = max(
                np.linalg.norm(pin_data.oMf[left_id].translation - left_home),
                np.linalg.norm(pin_data.oMf[right_id].translation - right_home),
            )
            log["t"][step] = t
            log["hand_pos"][step] = hand_now
            log["hand_ref"][step] = target
            log["hand_err_mm"][step] = np.linalg.norm(target - hand_now) * 1000.0
            log["com_margin_mm"][step] = support_polygon_margin(mj_model, mj_data) * 1000.0
            log["foot_move_mm"][step] = foot_move * 1000.0
            log["tau_max"][step] = np.abs(result.tau).max()
            log["solve_ms"][step] = result.solve_time_ms

            if fell_at is None and mj_data.qpos[2] < PELVIS_FALL_THRESHOLD:
                fell_at = t

            if writer is not None and step % RENDER_EVERY == 0:
                renderer.update_scene(mj_data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        if writer is not None:
            writer.close()

    # Tracking is scored once the hand is actually following the circle.
    moving = log["t"] >= SETTLE_SECONDS + REACH_SECONDS
    return {
        "fell": fell_at is not None,
        "fell_at": fell_at,
        "hand_rms_mm": float(np.sqrt(np.mean(log["hand_err_mm"][moving] ** 2))),
        "hand_max_mm": float(log["hand_err_mm"][moving].max()),
        "com_margin_min_mm": float(log["com_margin_mm"].min()),
        "foot_move_max_mm": float(log["foot_move_mm"].max()),
        "tau_max": float(log["tau_max"].max()),
        "solve_mean_ms": float(log["solve_ms"].mean()),
        "wall_seconds": time.time() - wall_start,
        "log": log,
    }


def plot_metrics(log: dict, path: Path) -> None:
    """Hand tracking, CoM margin, foot motion, torque."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    t = log["t"]

    axes[0].plot(t, log["hand_ref"][:, 1] * 1000, "C7--", lw=1, label="ref y")
    axes[0].plot(t, log["hand_pos"][:, 1] * 1000, "C0", lw=1, label="actual y")
    axes[0].plot(t, log["hand_ref"][:, 2] * 1000, "C8--", lw=1, label="ref z")
    axes[0].plot(t, log["hand_pos"][:, 2] * 1000, "C1", lw=1, label="actual z")
    axes[0].set_ylabel("hand (mm)")
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].plot(t, log["hand_err_mm"], "C3")
    axes[1].axhline(HAND_RMS_LIMIT_MM, color="k", ls="--", label="gate 20 mm (RMS)")
    axes[1].set_ylabel("hand error (mm)")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, log["com_margin_mm"], "C2", label="CoM margin")
    axes[2].plot(t, log["foot_move_mm"], "C4", label="foot motion")
    axes[2].axhline(COM_MARGIN_LIMIT_MM, color="k", ls="--", lw=0.8)
    axes[2].set_ylabel("mm")
    axes[2].legend(fontsize=8)

    axes[3].plot(t, log["tau_max"], "C5")
    axes[3].set_ylabel("max |τ| (N·m)")
    axes[3].set_xlabel("time (s)")

    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title("Lab 8 M1 — Whole-Body QP: standing reach (hand circle, both feet planted)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the M1 gate and write evidence."""
    print("=" * 72)
    print(" Lab 8 — M1: Whole-Body QP, Standing Reach")
    print("=" * 72)
    print(f"\n  trajectory: settle {SETTLE_SECONDS}s → reach {REACH_SECONDS}s → "
          f"{CIRCLE_LAPS:.0f} laps of a {CIRCLE_RADIUS*100:.0f} cm circle in {CIRCLE_SECONDS}s")
    print("  tasks: CoM (1e4) > hand (1e3) > posture (1); both feet in contact")

    result = run(record=True)
    plot_metrics(result["log"], PLOT_PATH)

    print(f"\n  QP mean solve: {result['solve_mean_ms']:.2f} ms   "
          f"({result['wall_seconds']:.0f}s wall for {TOTAL_SECONDS:.0f}s sim)")
    print(f"  video: {VIDEO_PATH}  ({VIDEO_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"  plot : {PLOT_PATH}")

    checks = [
        ("No fall", not result["fell"],
         "stood" if not result["fell"] else f"fell at {result['fell_at']:.2f}s"),
        ("Hand tracking RMS < 20 mm", result["hand_rms_mm"] < HAND_RMS_LIMIT_MM,
         f"{result['hand_rms_mm']:.2f} mm"),
        ("CoM margin ≥ 20 mm", result["com_margin_min_mm"] >= COM_MARGIN_LIMIT_MM,
         f"{result['com_margin_min_mm']:.1f} mm min"),
        ("Stance feet move < 5 mm", result["foot_move_max_mm"] < FOOT_MOTION_LIMIT_MM,
         f"{result['foot_move_max_mm']:.2f} mm"),
        ("Torques within limits", result["tau_max"] <= 139.0,
         f"{result['tau_max']:.1f} N·m peak"),
    ]

    print("\n" + "=" * 72)
    print(" M1 GATE")
    print("=" * 72)
    print(f" {'criterion':32s} {'result':>8s}   measured")
    print(" " + "-" * 69)
    for name, passed, detail in checks:
        print(f" {name:32s} {'PASS' if passed else 'FAIL':>8s}   {detail}")
    all_passed = all(passed for _, passed, _ in checks)
    print("=" * 72)
    print(f" M1: {'PASS — whole-body QP resolves reach and balance together' if all_passed else 'FAIL'}")
    print("=" * 72)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
