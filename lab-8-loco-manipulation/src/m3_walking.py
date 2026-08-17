"""Lab 8 — M3: forward walking on DCM (capture-point) tracking.

Retires the capstone Lab 7 deferred: Lab 7 could not walk because Menagerie's
position servos cannot track a dynamic reference, and M2 here could step in
place under torque control but not travel, because its CoM reference was
quasi-static — "put the CoM over the foot that is about to take the load"
needs a moment of rest over each foot that forward walking never grants.

M3 replaces that reference with the divergent component of motion. See
`dcm_planner.py` for the planning side and `wb_tasks.DCMTask` for the control
law; the short version is that the controller no longer says where the CoM
should be, only where its unstable part should be heading, and lets the ZMP
(which the QP places inside the feet) do the steering.

Usage:
    MUJOCO_GL=egl python3 lab-8-loco-manipulation/src/m3_walking.py
    MUJOCO_GL=egl python3 lab-8-loco-manipulation/src/m3_walking.py --in-place

Gate criteria (tasks/PLAN.md M3):
    * ≥ 10 consecutive steps without falling
    * ≥ 1.0 m travelled
    * ZMP inside the support polygon for > 90 % of loaded ticks
    * torques within actuator limits

`--in-place` re-runs M2's gate (4 steps, `step_length = 0`) through the *same*
DCM controller. A regression there would mean the new reference is wrong
rather than merely differently tuned, so it is checked every time this file's
tuning changes.
"""

from __future__ import annotations

import argparse
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

from dcm_planner import DCMPlan  # noqa: E402
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
    DCMTask,
    FrameOrientationTask,
    FramePositionTask,
    PostureTask,
    TaskStack,
)

LEFT_FOOT = "left_ankle_roll_link"
RIGHT_FOOT = "right_ankle_roll_link"

# -- gait ---------------------------------------------------------------
N_STEPS = 12
STEP_LENGTH = 0.10       # stride [m]; body advance per step is a full stride
STEP_WIDTH = 0.18        # foot separation while walking [m]; rest stance 0.237
T_DOUBLE = 0.25          # weight transfer [s] — dynamic, not quasi-static
T_SINGLE = 0.65          # swing [s]
T_INITIAL = 1.5          # settle before the first step [s]
STEP_HEIGHT = 0.03       # swing clearance [m]

# -- control ------------------------------------------------------------
DCM_GAIN = 3.0           # k_ξ [1/s]; ė = −k·e on the divergent component
DCM_INTEGRAL_GAIN = 0.0  # k_i on ∫(ξ − ξ_ref) — rejects the model-error bias
DCM_INTEGRAL_LEAK = 0.5
DCM_WEIGHT = 1e4
SWING_WEIGHT = 1e4
SWING_GAIN = 400.0
PELVIS_ORI_WEIGHT = 1e2
PELVIS_ORI_GAIN = 20.0
POSTURE_WEIGHT = 10.0
POSTURE_GAIN = 50.0
VRP_SHRINK = 0.7
SETTLE_SWEEP = 1.0       # fraction of the settle spent moving the ZMP onto foot 1
SETTLE_SECONDS = 1.0

# -- gate ---------------------------------------------------------------
PELVIS_FALL_THRESHOLD = 0.50
STEPS_REQUIRED = 10
DISTANCE_REQUIRED = 1.0
ZMP_INSIDE_TARGET = 0.90

VIDEO_PATH = MEDIA_DIR / "m3_walking.mp4"
PLOT_PATH = MEDIA_DIR / "m3_walking_metrics.png"
INPLACE_PLOT_PATH = MEDIA_DIR / "m3_inplace_metrics.png"
RENDER_EVERY = int(round(1.0 / (RENDER_FPS * DT)))


def make_plan(pin_model, pin_data, *, step_length: float, n_steps: int,
              close_stance: bool = False):
    """Gait schedule + DCM plan for the robot's **current** configuration.

    Split out of `build` because a caller may need to change the robot's pose
    after the initial settle — M4 brings both arms into a carry position, which
    moves the CoM some 80 mm forward — and a plan built before that change
    describes a robot that no longer exists. `pin_data` must already hold the
    current kinematics.
    """
    left_home = pin_point_to_world(pin_data.oMf[pin_model.getFrameId(LEFT_FOOT)].translation)
    right_home = pin_point_to_world(pin_data.oMf[pin_model.getFrameId(RIGHT_FOOT)].translation)
    com_home = pin_point_to_world(pin_data.com[0])

    # ω is measured, not assumed: it is the one number that couples the plan to
    # this particular robot's stance height, and the G1 settles ~15 mm lower
    # than its keyframe.
    com_height = float(com_home[2] - 0.5 * (left_home[2] + right_home[2]))

    # Step with the **trailing** foot first. From a level stance the two are
    # equivalent and this reduces to M3's original left-first gait; resuming a
    # gait after a stop, it is the difference between walking and falling
    # (L-M5-e).
    first_swing = LEFT_FOOT if left_home[0] <= right_home[0] + 1e-4 else RIGHT_FOOT
    schedule = GaitSchedule(
        LEFT_FOOT, RIGHT_FOOT, left_home, right_home, com_home,
        n_steps=n_steps, t_initial=T_INITIAL, t_double=T_DOUBLE, t_single=T_SINGLE,
        step_length=step_length, step_height=STEP_HEIGHT,
        step_width=STEP_WIDTH if step_length > 0.0 else None,
        first_swing=first_swing,
        close_stance=close_stance,
    )
    plan = DCMPlan(schedule, com_height, com_home, settle_sweep=SETTLE_SWEEP)
    return schedule, plan


def build(
    mj_model, mj_data, pin_model, pin_data, *,
    step_length: float,
    n_steps: int,
    q_nominal: np.ndarray | None = None,
):
    """Settle the robot, then build the DCM plan, task stack and controller.

    `q_nominal` overrides the joint pose the robot settles into and that the
    posture task pulls toward. Everything downstream — foot homes, CoM home, ω,
    the whole DCM plan — is measured **after** that settle, so a caller that
    wants to walk in a different upper-body posture (M4's carry pose) gets a
    gait plan built on the configuration the robot will actually walk in
    rather than on a pose it is about to leave.
    """
    # Homes must be measured on the *settled* robot (L-M2-d): taken at t=0 the
    # swing target sits ~10 mm below where the foot actually rests, so every
    # touchdown fights a reference buried in the floor.
    q_nominal = Q_STAND_JOINTS if q_nominal is None else np.asarray(q_nominal, dtype=float)
    settle = StandingController(
        mj_model, pin_model, pin_data,
        gravity_mode=GravityMode.CONTACT_CONSISTENT, q_nom=q_nominal,
    )
    for _ in range(int(SETTLE_SECONDS / DT)):
        settle.step(mj_data)

    q, v = mj_state_to_pin(mj_data)
    stack = TaskStack(pin_model, pin_data)
    stack.update_dynamics(q, v)

    schedule, plan = make_plan(pin_model, pin_data, step_length=step_length, n_steps=n_steps)

    dcm_task = stack.add(
        DCMTask(
            pin_model, plan.omega, weight=DCM_WEIGHT, gain=DCM_GAIN,
            integral_gain=DCM_INTEGRAL_GAIN, integral_leak=DCM_INTEGRAL_LEAK,
        )
    )
    pelvis_task = stack.add(
        FrameOrientationTask(
            "pelvis", pin_model, weight=PELVIS_ORI_WEIGHT, gain=PELVIS_ORI_GAIN
        )
    )
    swing_task = stack.add(
        FramePositionTask(
            LEFT_FOOT, pin_model, weight=SWING_WEIGHT, gain=SWING_GAIN, name="swing"
        )
    )
    stack.add(PostureTask(q_nominal, weight=POSTURE_WEIGHT, gain=POSTURE_GAIN))
    pelvis_task.capture_current(pin_data)

    qp = WholeBodyIDQP(
        pin_model, pin_data,
        [ContactSpec(LEFT_FOOT), ContactSpec(RIGHT_FOOT)],
        torque_limits(mj_model),
    )
    controller = SteppingController(
        mj_model, mj_data, pin_model, pin_data, schedule, qp, stack, dcm_task, swing_task,
        dcm_plan=plan, vrp_shrink=VRP_SHRINK,
    )
    return stack, schedule, plan, qp, controller


def run(
    record: bool = True,
    *,
    step_length: float = STEP_LENGTH,
    n_steps: int = N_STEPS,
    video_path: Path = VIDEO_PATH,
) -> dict:
    """Execute the walk; return gate metrics and telemetry."""
    mj_model, mj_data = load_g1_torque_mujoco(timestep=DT)
    pin_model, pin_data = load_g1_pinocchio()
    stack, schedule, plan, qp, controller = build(
        mj_model, mj_data, pin_model, pin_data, step_length=step_length, n_steps=n_steps
    )

    start_x = float(mj_data.subtree_com[0][0])

    writer = renderer = camera = None
    if record:
        import imageio

        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(video_path), fps=RENDER_FPS, codec="libx264", quality=8,
            macro_block_size=1,
        )
        renderer = mujoco.Renderer(mj_model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        camera = mujoco.MjvCamera()
        camera.distance = 3.4
        camera.azimuth = 130.0
        camera.elevation = -10.0

    fell_at: float | None = None
    steps_completed = 0
    distance = 0.0
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
            distance = float(mj_data.subtree_com[0][0]) - start_x

            if writer is not None and step % RENDER_EVERY == 0:
                # Track the robot: a fixed camera loses it after half a metre.
                camera.lookat[:] = [mj_data.qpos[0], mj_data.qpos[1], 0.75]
                renderer.update_scene(mj_data, camera=camera)
                writer.append_data(renderer.render())

            if mj_data.qpos[2] < PELVIS_FALL_THRESHOLD:
                fell_at = t
                steps_completed = max(reference.step_index - 1, 0)
                break
    finally:
        if writer is not None:
            writer.close()

    log = controller.log
    saturated = sum(1 for flag in log.vrp_saturated if flag)
    return {
        "fell": fell_at is not None,
        "fell_at": fell_at,
        "steps_completed": steps_completed,
        "steps_planned": n_steps,
        "distance": distance,
        "planned_distance": schedule.total_advance,
        "zmp_inside": controller.stance_fraction_zmp_inside(),
        "contact_switches": controller.contact_switches,
        "flight_ticks": controller.flight_ticks,
        "vrp_saturated_ticks": saturated,
        "vrp_saturated_frac": saturated / max(len(log.vrp_saturated), 1),
        "swing_err_max_mm": max(log.swing_err_mm) if log.swing_err_mm else 0.0,
        "dcm_err_max_mm": max(log.dcm_err_mm) if log.dcm_err_mm else 0.0,
        "dcm_err_rms_mm": (
            float(np.sqrt(np.mean(np.square(log.dcm_err_mm)))) if log.dcm_err_mm else 0.0
        ),
        "tau_max": max(log.tau_max) if log.tau_max else 0.0,
        "duration": schedule.total_duration,
        "omega": plan.omega,
        "log": log,
    }


def plot_metrics(log, path: Path) -> None:
    """Contact timeline, forward travel, DCM tracking, ZMP margin, torque."""
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    t = np.array(log.t)

    axes[0].plot(t, log.n_stance, "C0", drawstyle="steps-post")
    axes[0].set_ylabel("feet in contact")
    axes[0].set_yticks([0, 1, 2])

    axes[1].plot(t, np.array(log.com_x) - log.com_x[0], "C0", label="CoM travel (x)")
    axes[1].axhline(DISTANCE_REQUIRED, color="C3", ls="--", lw=0.8, label="gate 1.0 m")
    axes[1].set_ylabel("m")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, log.dcm_err_mm, "C4", lw=0.8, label="DCM tracking error")
    axes[2].plot(t, log.swing_err_mm, "C1", lw=0.8, label="swing tracking error")
    axes[2].set_ylabel("mm")
    axes[2].legend(fontsize=8)

    margins = np.array(log.zmp_margin_mm, dtype=float)
    finite = np.isfinite(margins)
    axes[3].plot(t[finite], margins[finite], "C2", lw=0.8, label="ZMP margin")
    axes[3].axhline(0.0, color="k", ls="--", lw=0.8)
    axes[3].set_ylabel("mm")
    axes[3].legend(fontsize=8)

    axes[4].plot(t, log.tau_max, "C5")
    axes[4].axhline(139.0, color="C3", ls="--", lw=0.8, label="actuator limit")
    axes[4].set_ylabel("max |τ| (N·m)")
    axes[4].set_xlabel("time (s)")
    axes[4].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title("Lab 8 M3 — Forward walking on DCM tracking (whole-body ID QP)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the M3 gate (or the in-place regression) and write evidence."""
    parser = argparse.ArgumentParser(description="Lab 8 M3 — forward walking")
    parser.add_argument(
        "--in-place", action="store_true",
        help="re-run M2's 4-step in-place gate through the DCM controller",
    )
    parser.add_argument("--no-video", action="store_true", help="skip recording")
    args = parser.parse_args()

    if args.in_place:
        print("=" * 72)
        print(" Lab 8 — M3 regression: M2's in-place gate under DCM control")
        print("=" * 72)
        result = run(
            record=not args.no_video, step_length=0.0, n_steps=4,
            video_path=MEDIA_DIR / "m3_inplace_regression.mp4",
        )
        checks = [
            ("4 in-place steps without falling", not result["fell"],
             f"{result['steps_completed']}/4 steps"),
            ("ZMP inside support > 95 %", result["zmp_inside"] > 0.95,
             f"{result['zmp_inside']*100:.1f} %"),
            ("Torques within limits", result["tau_max"] <= 139.0,
             f"{result['tau_max']:.1f} N·m peak"),
        ]
        title = "M2 IN-PLACE REGRESSION (under M3's DCM controller)"
        plot_path = INPLACE_PLOT_PATH
    else:
        print("=" * 72)
        print(" Lab 8 — M3: Forward Walking (DCM tracking)")
        print("=" * 72)
        print(f"\n  {N_STEPS} steps × {STEP_LENGTH*100:.0f} cm · swing {T_SINGLE}s "
              f"· double support {T_DOUBLE}s")
        print(f"  DCM gain k = {DCM_GAIN} 1/s · ZMP confined to {VRP_SHRINK:.0%} "
              f"of each contact patch")
        result = run(record=not args.no_video)
        print(f"\n  omega           : {result['omega']:.3f} 1/s")
        print(f"  steps completed : {result['steps_completed']} / {result['steps_planned']}")
        print(f"  distance        : {result['distance']:.3f} m "
              f"(planned {result['planned_distance']:.3f} m)")
        print(f"  DCM error       : {result['dcm_err_rms_mm']:.1f} mm RMS, "
              f"{result['dcm_err_max_mm']:.1f} mm max")
        print(f"  ZMP inside      : {result['zmp_inside']*100:.1f} % of loaded ticks")
        print(f"  ZMP clamped     : {result['vrp_saturated_frac']*100:.1f} % of ticks")
        print(f"  max swing error : {result['swing_err_max_mm']:.1f} mm")
        print(f"  peak torque     : {result['tau_max']:.1f} N·m")
        if result["fell"]:
            print(f"  FALL at {result['fell_at']:.2f} s of {result['duration']:.1f} s")
        checks = [
            (f"{STEPS_REQUIRED} steps without falling",
             not result["fell"] and result["steps_completed"] >= STEPS_REQUIRED,
             f"{result['steps_completed']}/{result['steps_planned']} steps"
             + ("" if not result["fell"] else f", fell at {result['fell_at']:.2f}s")),
            ("Travelled ≥ 1.0 m", result["distance"] >= DISTANCE_REQUIRED,
             f"{result['distance']:.3f} m"),
            ("ZMP inside support > 90 %", result["zmp_inside"] > ZMP_INSIDE_TARGET,
             f"{result['zmp_inside']*100:.1f} %"),
            ("Torques within limits", result["tau_max"] <= 139.0,
             f"{result['tau_max']:.1f} N·m peak"),
        ]
        title = "M3 GATE"
        plot_path = PLOT_PATH

    plot_metrics(result["log"], plot_path)
    print(f"\n  plot : {plot_path}")

    print("\n" + "=" * 72)
    print(f" {title}")
    print("=" * 72)
    print(f" {'criterion':34s} {'result':>8s}   measured")
    print(" " + "-" * 69)
    for name, passed, detail in checks:
        print(f" {name:34s} {'PASS' if passed else 'FAIL':>8s}   {detail}")
    all_passed = all(passed for _, passed, _ in checks)
    print("=" * 72)
    print(" PASS" if all_passed else " FAIL — milestone still open, see tasks/LESSONS.md § M3")
    print("=" * 72)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
