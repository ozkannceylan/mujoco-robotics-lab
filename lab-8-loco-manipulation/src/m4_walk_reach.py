"""Lab 8 — M4: walking while the hands do work.

M3 established that the G1 can walk under torque control. M4 asks it to walk
*and* hold a Cartesian hand task at the same time, which is the operating mode
Lab 9's VLA policy will have to produce — and the first milestone where the two
halves of this lab actually collide.

The collision is specific. Locomotion is a **centroidal** problem: the DCM task
regulates the divergent component of the CoM, and the CoM Jacobian includes the
arms. An arm task is therefore not an independent addition — every kilogram the
QP accelerates sideways to satisfy a hand target shows up as a disturbance in
the very quantity keeping the robot upright. M1 tracked a moving hand to
7.08 mm while standing, so nothing here is new *kinematically*; what is new is
that the balance controller now has to reject its own manipulation.

Two tasks, both from `tasks/PLAN.md` M4:

* **carry** — both hands hold a pose that is fixed in the world except for
  forward travel. The body sways ±90 mm laterally every step underneath them,
  so the arms must actively counter the gait rather than ride along with it.
  This is the harder reading of "fixed Cartesian pose" and the better test.
* **reach** — the right hand traces a 100 mm circle; the **left arm is free**,
  held only by the posture task. Locking the left hand to a Cartesian pose as
  well was measured to be the failure: with every upper-body DOF spoken for,
  the momentum task's demands were ground into the 25 N·m shoulders (143
  saturated ticks) and the robot fell at step 6. A task arm and a free arm is
  also simply what humans do — the free arm is the reaction wheel (L-M4-d).

Usage:
    MUJOCO_GL=egl python3 lab-8-loco-manipulation/src/m4_walk_reach.py
    MUJOCO_GL=egl python3 lab-8-loco-manipulation/src/m4_walk_reach.py --mode carry

Gate criteria (tasks/PLAN.md M4):
    * the M3 walking gate still passes (≥10 steps, ≥1.0 m, ZMP > 90 %, torques)
    * hand error < 50 mm while walking, for both tasks
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

import m3_walking as m3  # noqa: E402
from lab8_common import (  # noqa: E402
    CTRL_LEFT_ARM,
    CTRL_RIGHT_ARM,
    DT,
    V_RIGHT_ARM,
    MEDIA_DIR,
    Q_STAND_JOINTS,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    load_g1_pinocchio,
    load_g1_torque_mujoco,
    mj_state_to_pin,
)
from wb_tasks import CentroidalAngularMomentumTask, FramePositionTask  # noqa: E402

LEFT_HAND = "left_wrist_yaw_link"
RIGHT_HAND = "right_wrist_yaw_link"

# -- carry posture ------------------------------------------------------
# Reached **before** the gait plan is built, and reached **under the QP**, not
# under joint PD. Both halves were learned the hard way (LESSONS L-M4-a):
#   * Bringing both arms forward moves the CoM ~85 mm ahead of where it rests.
#     A DCM plan built on the arms-down pose then spends the whole walk asking
#     the robot to pull its CoM back to a place the posture will not allow.
#   * Joint PD cannot get there. It holds joint *angles*, so with the arms out
#     front the robot simply leans, and over 6 s it drifts and yaws off its
#     footprint (CoM y 0.027 → 0.24 m) — a posture the whole-body QP holds
#     without difficulty, because it can bend the hips to keep the CoM planted.
CARRY_OFFSET = np.array([0.20, 0.0, 0.05])   # from each hand's rest pose [m]
CARRY_SECONDS = 2.0      # QP-driven pre-pose before the gait plan is made

# -- hand task ----------------------------------------------------------
HAND_WEIGHT = 1e2        # below DCM and swing (1e4): balance and footfall win
HAND_GAIN = 400.0
REACH_RADIUS = 0.10      # right-hand circle [m]
REACH_PERIOD = 2.0       # one lap [s]
# Centroidal angular-momentum damping. This is what makes M4 possible at all
# (L-M4-c): with it off, a hand task strong enough to track (weight 1e2) falls
# on the 7th step, and the only weight that walks (1e1) leaves a 46 mm droop.
# With it on at weight 1e1, the *same* 1e2 hand task walks all 12 steps and
# tracks to 14.5 mm. Too much is as bad as none — 1e2 falls at step 5.
MOMENTUM_WEIGHT = 1e1
MOMENTUM_GAIN = 10.0
# Reach only: feed the momentum task the reference the commanded circle
# implies (resolved momentum control, Kajita et al. 2003) instead of zero.
# With L→0 the task fights the very trajectory the hand task feeds forward.
MOMENTUM_REFERENCE = True
# Ablation flag: lock the left hand to a Cartesian pose during reach as well
# (the configuration that saturated the shoulders and fell). Kept toggleable
# so the free-arm claim stays measurable, not narrative.
REACH_FREE_LEFT = True
HAND_ANCHOR = "body"     # "body": pose fixed in the walking frame (the brief's
                         #   carry task — the load travels with the robot)
                         # "world": y and z pinned in world, forward travel only

# -- gate ---------------------------------------------------------------
HAND_ERROR_LIMIT_MM = 50.0

VIDEO_PATH = MEDIA_DIR / "m4_walk_reach.mp4"
PLOT_PATH = MEDIA_DIR / "m4_hand_error.png"
RENDER_EVERY = int(round(1.0 / (RENDER_FPS * DT)))


def _smoothstep(alpha: float) -> tuple[float, float]:
    """Raised cosine and its derivative w.r.t. alpha, clipped to [0, 1]."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * alpha)), 0.5 * np.pi * np.sin(np.pi * alpha)


class HandReference:
    """Where each hand should be at time `t`, with feedforward.

    The pose is anchored to the **planned** forward travel, not the measured
    one. Two reasons: the plan supplies an analytic ẋ (the nominal CoM velocity)
    where a measured anchor would need differencing, and — more importantly —
    anchoring a task to the state the balance controller is regulating closes a
    loop the QP cannot see. The hand would chase the CoM while the CoM reacts to
    the hand.

    Lateral and vertical components stay fixed in the world. The body sways
    ±90 mm sideways every step underneath the hands, so holding still is real
    work for the arms; letting the hands ride the pelvis would test almost
    nothing.
    """

    def __init__(self, plan, homes: dict[str, np.ndarray], t_initial: float, mode: str):
        self.plan = plan
        self.homes = {k: v.copy() for k, v in homes.items()}
        self.t_initial = float(t_initial)
        self.mode = mode
        self.com_home = plan.nominal_com(0.0)[0].copy()

    def travel(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Displacement of the anchor frame, its rate, and its acceleration (3,).

        The acceleration comes from the LIPM the plan is built on,
        `c̈ = ω²(c − p_zmp)`, rather than being left at zero. Handing the task
        the trajectory's own ẍ is what turned M1's hand tracking from 18.63 mm
        to 7.08 mm (L-M1-b), and it matters more here than it did there: an
        acceleration the feedforward does not supply has to be produced by the
        task's PD, and on this robot every newton-metre of upper-body effort is
        drawn from a 50 N·m waist.
        """
        reference = self.plan.reference(t)
        com, com_velocity = reference.com, reference.com_velocity
        acceleration = self.plan.omega**2 * (com - reference.vrp)
        lateral = 1.0 if HAND_ANCHOR == "body" else 0.0
        return (
            np.array([com[0] - self.com_home[0], lateral * (com[1] - self.com_home[1]), 0.0]),
            np.array([com_velocity[0], lateral * com_velocity[1], 0.0]),
            np.array([acceleration[0], lateral * acceleration[1], 0.0]),
        )

    def __call__(self, frame: str, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(position, velocity, acceleration) for `frame` in world coordinates."""
        # `homes` were measured with the robot already settled in the carry
        # posture, so there is nothing to ramp into — the hand task's whole job
        # is to *hold* that pose against the gait.
        travel, travel_velocity, travel_acceleration = self.travel(t)
        position = self.homes[frame] + travel
        velocity = travel_velocity.copy()
        acceleration = travel_acceleration.copy()

        if self.mode == "reach" and frame == RIGHT_HAND and t > self.t_initial:
            # Circle in the **sagittal** (x–z) plane, faded in so the arm does
            # not snap into motion at the same instant the first step starts.
            #
            # The plane is not cosmetic. A lateral (y–z) circle of the same
            # size and speed falls every time, at 3–4 steps, at every period
            # from 2 s to 4 s — because lateral balance is the binding
            # constraint on this robot (L-M3-f: the foot is 170 mm long and
            # 50 mm wide, and stance width alone decided M3). Swinging a hand
            # sideways spends the axis that has nothing to spare; swinging it
            # fore-aft spends the axis that does.
            phase = 2.0 * np.pi * (t - self.t_initial) / REACH_PERIOD
            w = 2.0 * np.pi / REACH_PERIOD
            fade, d_fade = _smoothstep((t - self.t_initial) / REACH_PERIOD)
            d_fade /= REACH_PERIOD
            circle = REACH_RADIUS * np.array([np.cos(phase) - 1.0, 0.0, np.sin(phase)])
            circle_velocity = REACH_RADIUS * w * np.array([-np.sin(phase), 0.0, np.cos(phase)])
            circle_acceleration = REACH_RADIUS * w * w * np.array(
                [-np.cos(phase), 0.0, -np.sin(phase)]
            )
            position = position + fade * circle
            velocity = velocity + fade * circle_velocity + d_fade * circle
            acceleration = acceleration + fade * circle_acceleration + 2.0 * d_fade * circle_velocity

        return position, velocity, acceleration


def pre_pose(mj_model, mj_data, pin_model, pin_data, stack, qp, controller, hand_tasks, rest):
    """Drive both hands into the carry pose under the whole-body QP.

    The DCM reference is frozen at the CoM measured right now, so while the
    arms travel forward the QP has an explicit instruction to keep the capture
    point where it is — which it satisfies by leaning the hips back. That is
    the whole reason this runs through the QP instead of the standing
    controller: joint PD would hold the arm angles and let the robot topple
    forward out of its footprint.

    Returns the hand positions actually achieved, which is what the walking
    phase then holds. Whatever the QP settled on is by definition a pose this
    robot can balance in; a nominal FK target is not.
    """
    frozen = controller.com_task.current_dcm(pin_data).copy()
    controller.com_task.set_reference(frozen, np.zeros(2))
    controller.com_task.set_vrp_bounds(None, None)
    targets = {frame: rest[frame] + CARRY_OFFSET for frame in hand_tasks}

    for step in range(int(CARRY_SECONDS / DT)):
        blend, d_blend = _smoothstep(step * DT / CARRY_SECONDS)
        for frame, task in hand_tasks.items():
            task.set_target(
                rest[frame] + CARRY_OFFSET * blend,
                CARRY_OFFSET * d_blend / CARRY_SECONDS,
            )
        q, v = mj_state_to_pin(mj_data)
        stack.update_dynamics(q, v)
        mj_data.ctrl[:] = qp.solve(stack, q, v).tau
        mujoco.mj_step(mj_model, mj_data)

    q, v = mj_state_to_pin(mj_data)
    stack.update_dynamics(q, v)
    del targets

    # NOT re-nominalised on the carry pose, though the arithmetic argues for it:
    # the posture task still pulls the arms toward `Q_STAND_JOINTS` at weight 10
    # while the hand task holds them forward at weight 10, and the QP settles
    # that by parking the hands ~40 mm short with almost no variance. That
    # offset is the *price of a stabilising term*, not a bug — see L-M4-b, where
    # removing the contradiction (whole configuration, then arms only) took the
    # walk from 12/12 steps to 2/12 and 6/12 respectively. The pull toward rest
    # damps the arms, and undamped arms are a centroidal disturbance.
    return {frame: task.current_position(pin_data) for frame, task in hand_tasks.items()}


def _planned_arm_momentum(pin_model, pin_data, hand_task, reference, t) -> np.ndarray:
    """Angular momentum the commanded right-hand motion implies (3,).

    Resolved-momentum-control style: map the circle's Cartesian velocity
    (hand ẋ_ref minus the walking frame's own travel) into right-arm joint
    velocities through the arm block of the hand Jacobian, then through the
    arm block of the centroidal momentum matrix `A_g`. The base's contribution
    is deliberately excluded — the gait owns the base, and its momentum is not
    something the arm plan should claim.
    """
    del t  # the reference is already evaluated into the task's targets
    _, travel_velocity, _ = reference.travel(reference._last_t)
    relative = hand_task.target_velocity - travel_velocity
    arm_jacobian = hand_task.jacobian(pin_model, pin_data, None)[:, V_RIGHT_ARM]
    arm_velocity = np.linalg.lstsq(arm_jacobian, relative, rcond=1e-4)[0]
    return np.asarray(pin_data.Ag)[3:6, V_RIGHT_ARM] @ arm_velocity


def run(mode: str, record: bool = False, video_path: Path = VIDEO_PATH) -> dict:
    """Walk the M3 gait while both hands hold a task. Returns gate metrics."""
    mj_model, mj_data = load_g1_torque_mujoco(timestep=DT)
    pin_model, pin_data = load_g1_pinocchio()
    stack, schedule, plan, qp, controller = m3.build(
        mj_model, mj_data, pin_model, pin_data,
        step_length=m3.STEP_LENGTH, n_steps=m3.N_STEPS,
    )

    momentum_task = None
    if MOMENTUM_WEIGHT > 0.0:
        momentum_task = stack.add(
            CentroidalAngularMomentumTask(weight=MOMENTUM_WEIGHT, gain=MOMENTUM_GAIN)
        )
    # carry: both hands are task hands. reach: only the right — the left arm
    # stays free under the posture task, as the momentum task's actuation.
    task_frames = (
        (RIGHT_HAND,) if (mode == "reach" and REACH_FREE_LEFT)
        else (LEFT_HAND, RIGHT_HAND)
    )
    hand_tasks = {
        frame: stack.add(
            FramePositionTask(
                frame, pin_model, weight=HAND_WEIGHT, gain=HAND_GAIN, name=f"hand:{frame}"
            )
        )
        for frame in task_frames
    }
    rest = {frame: task.current_position(pin_data) for frame, task in hand_tasks.items()}
    homes = pre_pose(mj_model, mj_data, pin_model, pin_data, stack, qp, controller, hand_tasks, rest)

    # The robot is a different machine now — CoM 85 mm forward, feet nudged.
    # Rebuild the gait on the configuration it will actually walk in.
    schedule, plan = m3.make_plan(
        pin_model, pin_data, step_length=m3.STEP_LENGTH, n_steps=m3.N_STEPS
    )
    controller.schedule = schedule
    controller.dcm_plan = plan
    controller.com_task.omega = plan.omega
    reference = HandReference(plan, homes, schedule.t_initial, mode)

    start_x = float(mj_data.subtree_com[0][0])
    walk_start, walk_end = schedule.t_initial, schedule.total_duration - schedule.t_initial

    writer = renderer = camera = None
    if record:
        import imageio

        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(video_path), fps=RENDER_FPS, codec="libx264", quality=8, macro_block_size=1
        )
        renderer = mujoco.Renderer(mj_model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        camera = mujoco.MjvCamera()
        camera.distance = 3.4
        camera.azimuth = 130.0
        camera.elevation = -10.0

    log = {
        "t": [], "hand_err_mm": {frame: [] for frame in task_frames},
        "hand_ref": {frame: [] for frame in task_frames},
        "hand_pos": {frame: [] for frame in task_frames},
    }
    fell_at: float | None = None
    steps_completed = 0
    distance = 0.0
    n_ticks = int(schedule.total_duration / DT)
    try:
        for step in range(n_ticks):
            t = step * DT
            gait_reference = controller.update_targets(t)
            for frame, task in hand_tasks.items():
                task.set_target(*reference(frame, t))

            q, v = mj_state_to_pin(mj_data)
            stack.update_dynamics(q, v)
            if momentum_task is not None and mode == "reach" and MOMENTUM_REFERENCE:
                reference._last_t = t
                momentum_task.set_reference(
                    _planned_arm_momentum(
                        pin_model, pin_data, hand_tasks[RIGHT_HAND], reference, t
                    )
                )
            result = qp.solve(stack, q, v)
            mj_data.ctrl[:] = result.tau
            mujoco.mj_step(mj_model, mj_data)
            controller.record(t, gait_reference, result.tau)

            log["t"].append(t)
            for frame, task in hand_tasks.items():
                target, _, _ = reference(frame, t)
                actual = task.current_position(pin_data)
                log["hand_err_mm"][frame].append(float(np.linalg.norm(target - actual)) * 1000.0)
                log["hand_ref"][frame].append(target.copy())
                log["hand_pos"][frame].append(actual.copy())

            steps_completed = max(steps_completed, gait_reference.step_index)
            distance = float(mj_data.subtree_com[0][0]) - start_x

            if writer is not None and step % RENDER_EVERY == 0:
                camera.lookat[:] = [mj_data.qpos[0], mj_data.qpos[1], 0.75]
                renderer.update_scene(mj_data, camera=camera)
                writer.append_data(renderer.render())

            if mj_data.qpos[2] < m3.PELVIS_FALL_THRESHOLD:
                fell_at = t
                steps_completed = max(gait_reference.step_index - 1, 0)
                break
    finally:
        if writer is not None:
            writer.close()

    times = np.array(log["t"])
    walking = (times >= walk_start) & (times <= walk_end)
    errors = {
        frame: np.array(values)[walking] if walking.any() else np.array(values)
        for frame, values in log["hand_err_mm"].items()
    }
    combined = np.concatenate(list(errors.values())) if errors else np.zeros(1)

    return {
        "mode": mode,
        "fell": fell_at is not None,
        "fell_at": fell_at,
        "steps_completed": steps_completed,
        "steps_planned": m3.N_STEPS,
        "distance": distance,
        "zmp_inside": controller.stance_fraction_zmp_inside(),
        "tau_max": max(controller.log.tau_max) if controller.log.tau_max else 0.0,
        "dcm_err_rms_mm": float(np.sqrt(np.mean(np.square(controller.log.dcm_err_mm)))),
        "hand_rms_mm": float(np.sqrt(np.mean(np.square(combined)))),
        "hand_max_mm": float(combined.max()),
        "hand_rms_per_frame": {f: float(np.sqrt(np.mean(np.square(e)))) for f, e in errors.items()},
        "duration": schedule.total_duration,
        "walk_window": (walk_start, walk_end),
        "log": log,
        "gait_log": controller.log,
    }


def plot_metrics(results: list[dict], path: Path) -> None:
    """Hand error per task, the right hand's traced circle, and gait health."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    for result, colour in zip(results, ("C0", "C3")):
        times = np.array(result["log"]["t"])
        for frame, style in ((LEFT_HAND, "--"), (RIGHT_HAND, "-")):
            if frame not in result["log"]["hand_err_mm"]:
                continue  # reach mode has no left-hand task — the arm is free
            axes[0].plot(
                times, result["log"]["hand_err_mm"][frame], style, color=colour, lw=0.9,
                label=f"{result['mode']} · {'left' if frame == LEFT_HAND else 'right'}",
            )
    axes[0].axhline(HAND_ERROR_LIMIT_MM, color="k", ls=":", lw=1.0, label="gate 50 mm")
    start, end = results[0]["walk_window"]
    axes[0].axvspan(start, end, color="0.9", zorder=0)
    axes[0].set_ylabel("hand error (mm)")
    axes[0].set_title("Lab 8 M4 — Hand tracking while walking (shaded = walking interval)")
    axes[0].legend(fontsize=8, ncol=3)

    reach = next((r for r in results if r["mode"] == "reach"), results[-1])
    reference = np.array(reach["log"]["hand_ref"][RIGHT_HAND])
    actual = np.array(reach["log"]["hand_pos"][RIGHT_HAND])
    mask = np.array(reach["log"]["t"]) >= reach["walk_window"][0]
    # Subtract forward travel: the circle is drawn in the walking frame, and
    # plotting it in world coordinates would just show a 1.2 m smear.
    axes[1].plot(reference[mask, 0] - reference[mask, 0].mean(), reference[mask, 2],
                 "k--", lw=1.0, label="reference")
    axes[1].plot(actual[mask, 0] - reference[mask, 0].mean(), actual[mask, 2],
                 "C3", lw=0.9, label="measured")
    axes[1].set_xlabel("x − travel (m)")
    axes[1].set_ylabel("z (m)")
    axes[1].set_title("Right hand, walking frame")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].legend(fontsize=8)

    for result, colour in zip(results, ("C0", "C3")):
        gait = result["gait_log"]
        axes[2].plot(gait.t, gait.dcm_err_mm, color=colour, lw=0.9,
                     label=f"{result['mode']} · DCM error")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("DCM error (mm)")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _checks(result: dict) -> list[tuple[str, bool, str]]:
    mode = result["mode"]
    return [
        (f"[{mode}] {m3.STEPS_REQUIRED} steps, no fall",
         not result["fell"] and result["steps_completed"] >= m3.STEPS_REQUIRED,
         f"{result['steps_completed']}/{result['steps_planned']} steps"
         + ("" if not result["fell"] else f", fell at {result['fell_at']:.2f}s")),
        (f"[{mode}] Travelled ≥ 1.0 m", result["distance"] >= m3.DISTANCE_REQUIRED,
         f"{result['distance']:.3f} m"),
        (f"[{mode}] ZMP inside support > 90 %", result["zmp_inside"] > m3.ZMP_INSIDE_TARGET,
         f"{result['zmp_inside']*100:.1f} %"),
        (f"[{mode}] Hand error < 50 mm walking", result["hand_max_mm"] < HAND_ERROR_LIMIT_MM,
         f"{result['hand_rms_mm']:.1f} mm RMS, {result['hand_max_mm']:.1f} mm max"),
        (f"[{mode}] Torques within limits", result["tau_max"] <= 139.0,
         f"{result['tau_max']:.1f} N·m peak"),
    ]


def main() -> None:
    """Run the M4 gate and write evidence."""
    parser = argparse.ArgumentParser(description="Lab 8 M4 — walk + arm task")
    parser.add_argument("--mode", choices=("carry", "reach", "both"), default="both")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    modes = ("carry", "reach") if args.mode == "both" else (args.mode,)
    print("=" * 72)
    print(" Lab 8 — M4: Walk + Arm Task")
    print("=" * 72)
    print(f"\n  M3 gait ({m3.N_STEPS} × {m3.STEP_LENGTH*100:.0f} cm) + hand tasks at "
          f"weight {HAND_WEIGHT:.0e} (DCM and swing are {m3.DCM_WEIGHT:.0e})")

    results = []
    for mode in modes:
        # Only the reach run is recorded: it is the milestone's evidence, and a
        # second video of the same gait with stiller arms adds nothing.
        record = (not args.no_video) and mode == modes[-1]
        print(f"\n  running '{mode}'{' (recording)' if record else ''} …")
        result = run(mode, record=record)
        results.append(result)
        print(f"    steps {result['steps_completed']}/{result['steps_planned']} · "
              f"{result['distance']:.3f} m · ZMP {result['zmp_inside']*100:.1f} % · "
              f"τ {result['tau_max']:.1f} N·m")
        print(f"    hand  {result['hand_rms_mm']:.1f} mm RMS, "
              f"{result['hand_max_mm']:.1f} mm max  "
              + "  (" + ", ".join(
                  f"{'left' if f == LEFT_HAND else 'right'} {rms:.1f}"
                  for f, rms in result["hand_rms_per_frame"].items()) + ")")
        print(f"    DCM   {result['dcm_err_rms_mm']:.1f} mm RMS "
              f"(M3 without arms: 6.2 mm)")
        if result["fell"]:
            print(f"    FALL at {result['fell_at']:.2f} s of {result['duration']:.1f} s")

    plot_metrics(results, PLOT_PATH)
    print(f"\n  plot : {PLOT_PATH}")
    if not args.no_video and VIDEO_PATH.exists():
        print(f"  video: {VIDEO_PATH}  ({VIDEO_PATH.stat().st_size/1e6:.1f} MB)")

    checks = [check for result in results for check in _checks(result)]
    print("\n" + "=" * 72)
    print(" M4 GATE")
    print("=" * 72)
    print(f" {'criterion':40s} {'result':>8s}   measured")
    print(" " + "-" * 75)
    for name, passed, detail in checks:
        print(f" {name:40s} {'PASS' if passed else 'FAIL':>8s}   {detail}")
    all_passed = all(passed for _, passed, _ in checks)
    complete = set(modes) == {"carry", "reach"}
    print("=" * 72)
    if all_passed and complete:
        print(" M4: PASS — the robot walks and works at the same time")
    elif all_passed:
        # A single-mode run is a partial result. Saying "M4: PASS" here would
        # let the easier half stand in for the milestone.
        print(f" '{modes[0]}' passes — M4 needs both sub-tasks; run without --mode")
    else:
        print(" M4: FAIL — milestone still open, see tasks/LESSONS.md § M4")
    print("=" * 72)

    if not complete:
        return

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
