"""Lab 8 — M2: torque-level stepping in place (gate demo).

Drives `GaitSchedule` through the M1 whole-body inverse-dynamics QP, switching
the contact set as feet leave and rejoin the ground.

Usage:
    MUJOCO_GL=egl python3 lab-8-loco-manipulation/src/m2_stepping.py

Gate criteria (tasks/PLAN.md M2):
    * 4 consecutive in-place steps without falling
    * ZMP inside the support polygon for > 95 % of loaded ticks

The decisive tuning result (LESSONS L-M2-b): the CoM task must control
**horizontal position only**. Adding the vertical axis makes the QP hold the
pelvis at a fixed height through the whole weight transfer, which it can only
do by fighting the natural drop with waist and hip torque — the waist roll
actuator (50 N·m) saturated for 583 ticks and the robot fell on the fourth
step. Releasing z drops peak torque from 139 N·m (saturated) to 50 N·m with
zero saturated ticks, and the gait completes.
"""

from __future__ import annotations

import sys
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
from gait_planner import GaitSchedule  # noqa: E402
from lab8_common import (  # noqa: E402
    DT,
    MEDIA_DIR,
    Q_STAND_JOINTS,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    load_g1_pinocchio,
    load_g1_torque_mujoco,
    mj_state_to_pin,
    pin_point_to_world,
)
from locomotion_controller import SteppingController  # noqa: E402
from standing_controller import GravityMode, StandingController  # noqa: E402
from wb_id_qp import ContactSpec, WholeBodyIDQP  # noqa: E402
from wb_tasks import (  # noqa: E402
    CoMTask,
    FrameOrientationTask,
    FramePositionTask,
    PostureTask,
    TaskStack,
)

LEFT_FOOT = "left_ankle_roll_link"
RIGHT_FOOT = "right_ankle_roll_link"

N_STEPS = 4
T_DOUBLE = 2.0          # weight-transfer duration [s]
T_SINGLE = 0.5          # swing duration [s]
STEP_HEIGHT = 0.015     # swing clearance [m]
COM_SHIFT = 0.9         # fraction of the way to the stance foot
COM_AXES = (0, 1)       # horizontal only — see module docstring
PELVIS_ORI_WEIGHT = 1e2
PELVIS_ORI_GAIN = 20.0
SETTLE_SECONDS = 1.0

PELVIS_FALL_THRESHOLD = 0.50
ZMP_INSIDE_TARGET = 0.95

VIDEO_PATH = MEDIA_DIR / "m2_stepping.mp4"
PLOT_PATH = MEDIA_DIR / "m2_stepping_metrics.png"
RENDER_EVERY = int(round(1.0 / (RENDER_FPS * DT)))


def build(mj_model, mj_data, pin_model, pin_data):
    """Settle the robot, then build the task stack, schedule and controller."""
    # Settle under the M0 standing controller first: the gait schedule's foot
    # and CoM homes must be measured on the *resting* robot. Taking them at
    # t=0 puts the swing reference ~10 mm below where the foot actually ends
    # up, so every touchdown fights a target buried in the floor.
    settle = StandingController(
        mj_model, pin_model, pin_data, gravity_mode=GravityMode.CONTACT_CONSISTENT
    )
    for _ in range(int(SETTLE_SECONDS / DT)):
        settle.step(mj_data)

    stack = TaskStack(pin_model, pin_data)
    com_task = stack.add(CoMTask(pin_model, weight=1e4, gain=100.0, axes=COM_AXES))
    pelvis_task = stack.add(
        FrameOrientationTask(
            "pelvis", pin_model, weight=PELVIS_ORI_WEIGHT, gain=PELVIS_ORI_GAIN
        )
    )
    swing_task = stack.add(
        FramePositionTask(LEFT_FOOT, pin_model, weight=1e3, gain=400.0, name="swing")
    )
    stack.add(PostureTask(Q_STAND_JOINTS, weight=10.0, gain=50.0))

    q, v = mj_state_to_pin(mj_data)
    stack.update_dynamics(q, v)
    pelvis_task.capture_current(pin_data)

    left_home = pin_point_to_world(pin_data.oMf[pin_model.getFrameId(LEFT_FOOT)].translation)
    right_home = pin_point_to_world(pin_data.oMf[pin_model.getFrameId(RIGHT_FOOT)].translation)

    schedule = GaitSchedule(
        LEFT_FOOT, RIGHT_FOOT, left_home, right_home, com_task.current_com(pin_data),
        n_steps=N_STEPS, t_double=T_DOUBLE, t_single=T_SINGLE,
        step_length=0.0, step_height=STEP_HEIGHT, com_shift_ratio=COM_SHIFT,
    )
    qp = WholeBodyIDQP(
        pin_model, pin_data,
        [ContactSpec(LEFT_FOOT), ContactSpec(RIGHT_FOOT)],
        torque_limits(mj_model),
    )
    controller = SteppingController(
        mj_model, mj_data, pin_model, pin_data, schedule, qp, stack, com_task, swing_task
    )
    return stack, schedule, qp, controller


def run(record: bool = True) -> dict:
    """Execute the stepping sequence; return metrics and the telemetry log."""
    mj_model, mj_data = load_g1_torque_mujoco(timestep=DT)
    pin_model, pin_data = load_g1_pinocchio()
    stack, schedule, qp, controller = build(mj_model, mj_data, pin_model, pin_data)

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
        camera.lookat[:] = [0.0, 0.0, 0.75]
        camera.distance = 2.8
        camera.azimuth = 120.0
        camera.elevation = -8.0

    fell_at: float | None = None
    steps_completed = 0
    n_ticks = int(schedule.total_duration / DT)
    try:
        for step in range(n_ticks):
            t = step * DT
            reference = controller.update_targets(t)

            q, v = mj_state_to_pin(mj_data)
            stack.update_dynamics(q, v)
            result = qp.solve(stack, q, v)
            mj_data.ctrl[:] = result.tau
            mujoco.mj_step(mj_model, mj_data)
            controller.record(t, reference, result.tau)

            steps_completed = max(steps_completed, reference.step_index)

            if writer is not None and step % RENDER_EVERY == 0:
                renderer.update_scene(mj_data, camera=camera)
                writer.append_data(renderer.render())

            if mj_data.qpos[2] < PELVIS_FALL_THRESHOLD:
                fell_at = t
                # A step only counts if the robot was still standing when it
                # finished; the one in progress at the fall does not.
                steps_completed = max(reference.step_index - 1, 0)
                break
    finally:
        if writer is not None:
            writer.close()

    log = controller.log
    return {
        "fell": fell_at is not None,
        "fell_at": fell_at,
        "steps_completed": steps_completed,
        "steps_planned": N_STEPS,
        "zmp_inside": controller.stance_fraction_zmp_inside(),
        "contact_switches": controller.contact_switches,
        "flight_ticks": controller.flight_ticks,
        "swing_err_max_mm": max(log.swing_err_mm) if log.swing_err_mm else 0.0,
        "com_err_max_mm": max(log.com_err_mm) if log.com_err_mm else 0.0,
        "tau_max": max(log.tau_max) if log.tau_max else 0.0,
        "duration": schedule.total_duration,
        "log": log,
    }


def plot_metrics(log, path: Path) -> None:
    """Phase timeline, swing tracking, CoM error, ZMP margin, torque."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    t = np.array(log.t)

    axes[0].plot(t, log.n_stance, "C0", drawstyle="steps-post")
    axes[0].set_ylabel("feet in contact")
    axes[0].set_yticks([0, 1, 2])

    axes[1].plot(t, log.swing_height_mm, "C1", label="swing foot height")
    axes[1].plot(t, log.swing_err_mm, "C3", lw=0.8, label="swing tracking error")
    axes[1].set_ylabel("mm")
    axes[1].legend(fontsize=8)

    margins = np.array(log.zmp_margin_mm, dtype=float)
    finite = np.isfinite(margins)
    axes[2].plot(t[finite], margins[finite], "C2", lw=0.8, label="ZMP margin")
    axes[2].plot(t, log.com_err_mm, "C4", lw=0.8, label="CoM tracking error")
    axes[2].axhline(0.0, color="k", ls="--", lw=0.8)
    axes[2].set_ylabel("mm")
    axes[2].legend(fontsize=8)

    axes[3].plot(t, log.tau_max, "C5")
    axes[3].axhline(139.0, color="C3", ls="--", lw=0.8, label="actuator limit")
    axes[3].set_ylabel("max |τ| (N·m)")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title("Lab 8 M2 — Torque-level stepping in place (whole-body ID QP)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the M2 gate and write evidence."""
    print("=" * 72)
    print(" Lab 8 — M2: Torque-Level Stepping")
    print("=" * 72)
    print(f"\n  {N_STEPS} in-place steps · double support {T_DOUBLE}s · swing {T_SINGLE}s")
    print(f"  clearance {STEP_HEIGHT*1000:.0f} mm · CoM shift {COM_SHIFT:.0%} toward stance foot")

    result = run(record=True)
    plot_metrics(result["log"], PLOT_PATH)

    print(f"\n  steps completed : {result['steps_completed']} / {result['steps_planned']}")
    print(f"  contact switches: {result['contact_switches']}")
    print(f"  ZMP inside      : {result['zmp_inside']*100:.1f} % of loaded ticks")
    print(f"  max swing error : {result['swing_err_max_mm']:.1f} mm")
    print(f"  max CoM error   : {result['com_err_max_mm']:.1f} mm")
    print(f"  peak torque     : {result['tau_max']:.1f} N·m")
    if result["fell"]:
        print(f"  FALL at {result['fell_at']:.2f} s of {result['duration']:.1f} s")
    print(f"\n  video: {VIDEO_PATH}  ({VIDEO_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"  plot : {PLOT_PATH}")

    checks = [
        (f"{N_STEPS} steps without falling", not result["fell"],
         f"{result['steps_completed']}/{result['steps_planned']} steps"
         + ("" if not result["fell"] else f", fell at {result['fell_at']:.2f}s")),
        ("ZMP inside support > 95 %", result["zmp_inside"] > ZMP_INSIDE_TARGET,
         f"{result['zmp_inside']*100:.1f} %"),
        ("Torques within limits", result["tau_max"] <= 139.0,
         f"{result['tau_max']:.1f} N·m peak"),
    ]

    print("\n" + "=" * 72)
    print(" M2 GATE")
    print("=" * 72)
    print(f" {'criterion':34s} {'result':>8s}   measured")
    print(" " + "-" * 69)
    for name, passed, detail in checks:
        print(f" {name:34s} {'PASS' if passed else 'FAIL':>8s}   {detail}")
    all_passed = all(passed for _, passed, _ in checks)
    print("=" * 72)
    if all_passed:
        print(" M2: PASS — torque-level stepping established")
    else:
        print(" M2: FAIL — milestone still open, see tasks/LESSONS.md § M2")
    print("=" * 72)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
