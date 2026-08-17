"""Lab 9 — the programmatic expert: Lab 8's capstone over the two-object scene.

The demonstrations this lab learns from are produced by Lab 8's whole-body
controller, unmodified. That is the point of the lab and the reason the master
plan called Lab 8 the critical path: `humanoid_vla` already learns tabletop
manipulation from an IK expert, and what has never existed here is a policy
trained on a **walking** humanoid, where every action it emits is taken while a
balance controller is actively keeping the robot up.

How Lab 8 is reused
-------------------
`VLAExpert` subclasses Lab 8's `Capstone`. Every phase method — `stand`,
`walk`, `carry_targets`, `_attach_payload`, `payload_goal_to_hand`,
`_freeze_balance`, `_step` — is inherited verbatim. Only three things change:

1. the scene is Lab 9's (two objects, cameras, per-seed randomisation);
2. which object the manipulation phases act on is chosen by the instruction;
3. `_step` additionally captures an observation/action pair every 100 control
   ticks (1 kHz control, 10 Hz policy).

Lab 8's own source is not edited. `Capstone.__init__` hardcodes its scene, so
this class replaces `__init__` and calls the inherited `_settle` and
`_build_stack`; that is ~20 duplicated lines against a Lab 8 regression re-run,
and the ground rules pick the duplication.

What a demonstration records
----------------------------
At each captured tick: both camera images, the proprioception vector, and the
*expert's own command* — the hand targets it is driving, whether it is walking,
and the weld states. Not the achieved state: behaviour cloning has to imitate
what the expert did, and on a compliant, disturbed system the two differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from lab9_common import (
    DT,
    POLICY_DECIMATION,
    TASK_NAMES,
    instruction_label,
)
from observations import (
    ObservationRenderer,
    build_state,
    encode_joint_action,
    encode_task_action,
    pelvis_frame,
)
from vla_scene import Randomisation, build_vla_scene

# Lab 8.
from capstone_scene import PAYLOAD_HALF  # noqa: E402
from m5_capstone import (  # noqa: E402
    GRASP_OFFSET,
    GRASP_TOLERANCE_MM,
    HAND_WEIGHT_STAND,
    LEFT_HAND,
    RIGHT_HAND,
    T_REACH,
    Capstone,
    Fell,
)

__all__ = [
    "EpisodeRecord",
    "VLAExpert",
    "run_episode",
    "PHASE_TO_TASK",
    "REACH_STANDOFF",
    "approach_steps_for",
]

# How far in front of the pelvis the target object should sit when the robot
# stops. Reach *accuracy* is flat at 7-11 mm anywhere between a standoff of
# -0.01 and 0.37 m, which is what an early sweep measured and why 0.22 m looked
# fine. Reach accuracy is the wrong quantity: at 0.22 m the arm is extended
# ~0.43 m from the pelvis, and *lifting* half a kilogram out there saturates the
# waist and takes the robot down at the end of the lift, not during the reach
# (tasks/LESSONS.md § L-M0-b). Lab 8's own capstone stood 0.06 m from its
# payload — almost entirely a lateral reach, with the arm folded rather than
# extended. This matches it.
REACH_STANDOFF: float = 0.07

# Pelvis x after n approach steps, fitted to measurement: 0.137, 0.238, 0.339
# for n = 1, 2, 3 (Lab 8's gait, STEP_LENGTH 0.10 plus the closing step).
_PELVIS_X0: float = 0.036
_PELVIS_X_PER_STEP: float = 0.101
MIN_APPROACH_STEPS: int = 1
MAX_APPROACH_STEPS: int = 4

# The reach runs at Lab 8's *standing* hand weight, not its walking one. Lab 8's
# capstone used the walking weight (1e2) throughout and reached to 59.8 mm
# against a 70 mm grasp tolerance — inside its own gate, but with no margin for
# an object that has moved. At M1's standing weight the same reach converges to
# 14 mm, and a one-second hold at the final target takes it to ~7 mm, which is
# M1's validated regime (7.08 mm). L-M5-i's warning is about raising the weight
# in *every* phase, which destabilises the tuck; this raises it only in the
# phase that is standing still, which is the phase M1 measured.
T_REACH_SETTLE: float = 1.0

# ---------------------------------------------------------------------------
# Phase durations — Lab 9's own, because Lab 8's do not fit here
# ---------------------------------------------------------------------------
# Lab 8's sequence never stands still for more than ~7 s: it stops, picks,
# tucks, and then *walks*, which replans the DCM and re-establishes the balance
# reference from scratch. Lab 9 has no carry-walk (see L-M0-c), so its whole
# manipulation happens in one continuous stand — and at Lab 8's timings that is
# 11.5 s, which this controller does not survive.
#
# The failure is unambiguous and is not a saturation: the DCM error grows
# exponentially at the LIPM rate (doubling every ~0.15 s) from 4.5 mm while the
# hand still tracks to 5 mm and peak torque sits at 21 N.m. `_freeze_balance`
# pins the divergent-component target at the value it had when the phase began,
# and an arm motion moves the centre of mass out from under it.
#
# Measured standing budget (4 configurations, no jitter):
#   11.5 s -> 0/4 complete, all fall between t = 13.7 s and 17.0 s
#    6.9 s -> 1/4 fall
#    5.2 s -> 4/4 complete, and what is left is accuracy, not balance
# See tasks/LESSONS.md § L-M0-d.
T_STOP_L9: float = 0.55
T_REACH_L9: float = 1.30
T_REACH_SETTLE: float = 0.45
T_GRASP_L9: float = 0.30
T_LIFT_L9: float = 0.60
T_HOLD_L9: float = 0.40

#: Total continuous standing time, asserted against the measured budget.
STAND_BUDGET_S: float = T_STOP_L9 + T_REACH_L9 + T_REACH_SETTLE + T_GRASP_L9 \
    + T_LIFT_L9 + T_HOLD_L9

#: How high the object must actually rise for the pick to count.
MIN_LIFT_M: float = 0.04

#: Lab 8 lifted 0.12 m to clear a pedestal before travelling to a *different*
#: one. Here the pick and the place share a surface, so the lift only has to
#: unseat the object; every extra centimetre is mass raised at arm's length,
#: which is what the balance controller pays for.
LIFT_CLEARANCE: float = 0.08


# Success thresholds, all asserted on the *simulated* object pose.
#: How far the *other* object is allowed to shift. A policy that sweeps the
#: distractor off the pedestal on its way has not followed its instruction.
DISTRACTOR_TOLERANCE_M: float = 0.05


def approach_steps_for(
    object_x: float, marker_x: float | None = None, pelvis_x: float = 0.0
) -> int:
    """How many Lab 8 walk steps put the episode's workspace within reach.

    The robot stops once and does everything from there — pick *and* place — so
    the stopping point is chosen for the midpoint of the two, not for the object
    alone. Aiming at the object alone costs the other end of the task: the near
    object is a one-step approach, which then leaves the marker 0.26 m forward,
    and the place either misses or throws the object (measured: 58-4699 mm).

    Args:
        object_x: Forward position of the target object [m].
        marker_x: Forward position of the drop marker [m]; the object's own
            position if omitted.
        pelvis_x: Current pelvis forward position [m].

    Returns:
        Step count, clamped to the range the gait was validated over.
    """
    reference = object_x if marker_x is None else 0.5 * (object_x + marker_x)
    wanted = reference - REACH_STANDOFF - pelvis_x
    steps = int(round((wanted - _PELVIS_X0) / _PELVIS_X_PER_STEP))
    return int(np.clip(steps, MIN_APPROACH_STEPS, MAX_APPROACH_STEPS))

#: Which Lab 8 phase belongs to which Lab 9 task label. Phases not listed here
#: (the settle, the release tail) are recorded but belong to no task and are
#: dropped when the episode is sliced.
PHASE_TO_TASK: dict[str, str] = {
    "walk_to_pick": "walk",
    "stop_at_pick": "pick",
    "reach": "pick",
    "grasp": "pick",
    "lift": "pick",
    "hold": "pick",
}


@dataclass
class EpisodeRecord:
    """One expert rollout, sampled at the policy rate."""

    seed: int
    target: str
    wide: bool
    near_object: str
    success: bool = False
    reason: str = ""
    time: list[float] = field(default_factory=list)
    phase: list[str] = field(default_factory=list)
    head: list[np.ndarray] = field(default_factory=list)
    wrist: list[np.ndarray] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    task_action: list[np.ndarray] = field(default_factory=list)
    joint_action: list[np.ndarray] = field(default_factory=list)
    approach_steps: int = 0
    metrics: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.time)


class VLAExpert(Capstone):
    """Lab 8's capstone sequence, seeded, over a two-object scene."""

    def __init__(
        self,
        seed: int,
        target: str = "cup",
        wide: bool = False,
        capture: bool = True,
        image_size: int | None = None,
    ):
        """Build the scene and controller for one demonstration.

        Args:
            seed: Scene randomisation seed.
            target: Which object to manipulate, ``"cup"`` or ``"box"``.
            wide: Draw the object placement from the wider evaluation range.
            capture: Record observations (off for pure success screening, which
                is ~3x faster because rendering dominates).
            image_size: Override the camera resolution.
        """
        from lab9_common import IMAGE_SIZE
        from lab8_common import load_g1_pinocchio

        self.randomisation = Randomisation.sample(seed, wide=wide)
        self.scene = build_vla_scene(DT, self.randomisation, target=target)
        self.mj_model, self.mj_data = self.scene.model, self.scene.data
        self.pin_model, self.pin_data = load_g1_pinocchio()

        # Fields the inherited phase methods expect.
        from m5_capstone import CapstoneLog

        self.log = CapstoneLog()
        self.t = 0.0
        self.phase = "settle"
        self.grasped = False
        self.two_handed = False
        self.hand_target: np.ndarray | None = None
        self.payload_in_hand: np.ndarray | None = None
        self.writer = self.renderer = self.camera = None

        self._settle()
        self._build_stack()

        self.record = EpisodeRecord(
            seed=seed,
            target=target,
            wide=wide,
            near_object=self.randomisation.near_object,
        )
        self._capture = capture
        self._obs = (
            ObservationRenderer(self.mj_model, size=image_size or IMAGE_SIZE)
            if capture
            else None
        )
        self._ticks = 0

    # -- capture --------------------------------------------------------

    def _step(self, controller=None) -> None:
        """One control tick, plus an observation every policy period."""
        super()._step(controller)
        self._ticks += 1
        if self._capture and self._ticks % POLICY_DECIMATION == 0:
            self._capture_frame()

    def _effective_hand_target(self, frame: str) -> np.ndarray:
        """What the expert is currently commanding this hand to do.

        A hand whose task is disabled is not being commanded anywhere, and the
        honest action for the policy to imitate is "leave it where it is" —
        recording a stale target would teach a step the expert never made.

        Args:
            frame: Pinocchio frame name of the hand.

        Returns:
            ``(3,)`` world target.
        """
        task = self.hand_tasks[frame]
        if task.enabled:
            return np.asarray(task.target, dtype=float).copy()
        return task.current_position(self.pin_data)

    def _capture_frame(self) -> None:
        """Append one observation/action pair to the record."""
        position, yaw = pelvis_frame(self.mj_data)
        right_weld = float(self.mj_data.eq_active[self.scene.weld_ids[(self.scene.target, "right")]])
        left_weld = float(self.mj_data.eq_active[self.scene.weld_ids[(self.scene.target, "left")]])

        self.record.time.append(self.t)
        self.record.phase.append(self.phase)
        self.record.head.append(self._obs.render(self.mj_data, "head"))
        self.record.wrist.append(self._obs.render(self.mj_data, "wrist"))
        self.record.state.append(build_state(self.mj_data, self.scene.any_weld_active()))
        self.record.task_action.append(
            encode_task_action(
                right_hand=self._effective_hand_target(RIGHT_HAND),
                left_hand=self._effective_hand_target(LEFT_HAND),
                gait=1.0 if self.phase.startswith("walk") else 0.0,
                grasp_right=right_weld,
                grasp_left=left_weld,
                pelvis_position=position,
                pelvis_yaw=yaw,
            )
        )
        self.record.joint_action.append(encode_joint_action(self.mj_data))

    def _attach_payload(self) -> None:
        """Lab 8's grasp, with this lab's object mass in the Pinocchio model.

        Lab 8 folds `PAYLOAD_MASS` (0.5 kg) into the wrist's parent joint; the
        objects here are lighter, and telling the controller about a mass it is
        not carrying is exactly the modelling error that makes a QP plan
        wrenches the simulator will not produce.
        """
        from lab8_common import attach_payload_to_pinocchio
        from m5_capstone import RIGHT_HAND as _RIGHT
        from lab9_common import pin_point_to_world as _to_world
        from vla_scene import OBJECT_MASS

        self.scene.set_weld(True)
        self.grasped = True
        self._sync_kinematics()

        hand = self.pin_data.oMf[self.pin_model.getFrameId(_RIGHT)]
        self.payload_in_hand = hand.rotation.T @ (
            self.scene.payload_position() - _to_world(hand.translation)
        )
        self.pin_data = attach_payload_to_pinocchio(
            self.pin_model, OBJECT_MASS, PAYLOAD_HALF, _RIGHT, self.payload_in_hand
        )
        self.stack.data = self.pin_data
        self.qp.data = self.pin_data
        self._sync_kinematics()

    def stand_segmented(
        self,
        duration: float,
        phase: str,
        segments: int,
        hand_goals: dict[str, np.ndarray] | None = None,
        payload_goal: np.ndarray | None = None,
        hand_weight: float | None = None,
    ) -> None:
        """A standing motion split into several short `stand` calls.

        Lab 8's `stand` freezes the balance reference **once**, at the start:
        the DCM target is set to whatever the divergent component is at that
        instant and held there for the whole phase. That is correct for a short
        motion and wrong for a long one, because moving an arm — with or without
        a load — shifts the centre of mass, and a frozen target then commands
        the robot back toward a snapshot that no longer describes a resting
        configuration. The controller ends up fighting its own arm.

        Measured: a single 2.5 s place diverges from 4.5 mm of DCM error to
        340 mm at the LIPM rate, with the hand still tracking to 5 mm and peak
        torque at 21 N.m — an instability, not a saturation. Splitting the same
        motion into short segments re-anchors the reference between them and
        the error never accumulates (tasks/LESSONS.md § L-M0-d).

        Args:
            duration: Total motion time [s].
            phase: Phase label written into the log.
            segments: How many `stand` calls to split it into.
            hand_goals: Final hand targets, world coordinates.
            payload_goal: Final payload target; servoed, mutually exclusive
                with `hand_goals`.
            hand_weight: Hand task weight; Lab 8's default if omitted.
        """
        from m5_capstone import HAND_WEIGHT

        weight = HAND_WEIGHT if hand_weight is None else hand_weight
        span = duration / segments
        starts = (
            {frame: self.hand_position(frame) for frame in hand_goals}
            if hand_goals else {}
        )
        payload_start = (
            self.scene.payload_position() if payload_goal is not None else None
        )
        for index in range(segments):
            alpha = (index + 1) / segments
            if payload_goal is not None:
                waypoint = payload_start + alpha * (payload_goal - payload_start)
                self.stand(span, phase, payload_goal=waypoint, hand_weight=weight)
            else:
                goals = {
                    frame: starts[frame] + alpha * (goal - starts[frame])
                    for frame, goal in hand_goals.items()
                }
                self.stand(span, phase, hand_goals=goals, hand_weight=weight)

    def grasp_offset(self) -> np.ndarray:
        """Where the wrist should sit relative to the target object's centre.

        Lab 8 used a fixed -0.060 m in x, which is its payload's half-extent
        plus a wrist clearance. Applied to a wider object that puts the wrist
        *inside* the object's footprint: measured, the 0.040 m-radius cup
        reached to 29-30 mm where the 0.030 m box reached to 7-11 mm, and every
        one of M0's failures was a near cup. Scaling the offset by the object's
        own half-extent gives both the same surface clearance.

        Returns:
            ``(3,)`` world-frame offset from the object centre.
        """
        clearance = GRASP_OFFSET[0] + PAYLOAD_HALF   # Lab 8's wrist clearance
        return np.array([
            clearance - self.scene.object_half_x(self.scene.target),
            GRASP_OFFSET[1],
            GRASP_OFFSET[2],
        ])

    def object_speed(self) -> float:
        """Linear speed of the target object [m/s], from the simulator."""
        address = self.scene.object_qpos[self.scene.target]
        # A freejoint's qvel block starts where its qpos block does, minus the
        # one extra slot the quaternion takes over the tangent space, counted
        # across every joint before it. Read it off the body instead.
        body = self.scene.object_bodies[self.scene.target]
        del address
        return float(np.linalg.norm(self.mj_data.cvel[body][3:6]))

    def close(self) -> None:
        super().close()
        if self._obs is not None:
            self._obs.close()
            self._obs = None


def run_episode(
    seed: int,
    target: str = "cup",
    wide: bool = False,
    capture: bool = True,
    image_size: int | None = None,
    verbose: bool = False,
) -> EpisodeRecord:
    """Run one expert demonstration end to end.

    The sequence is Lab 8's M5 with the target object substituted: walk to the
    pedestal, stop, reach, grasp, lift, tuck to the chest and close the second
    weld, walk carrying, stop, place, release.

    Args:
        seed: Scene randomisation seed.
        target: ``"cup"`` or ``"box"``.
        wide: Wider object-placement range.
        capture: Record observations.
        image_size: Camera resolution override.
        verbose: Print phase progress.

    Returns:
        An :class:`EpisodeRecord`; ``success`` is decided on the **simulated**
        object pose, never on a commanded value.
    """
    expert = VLAExpert(seed, target=target, wide=wide, capture=capture,
                       image_size=image_size)
    scene = expert.scene
    reached_mm = float("nan")
    lift_m = 0.0
    start_z = 0.0

    try:
        # How far to walk is decided by *which object was named* — the near one
        # is a one-step approach, the far one two or three. That makes the walk
        # phase itself instruction-dependent rather than a fixed preamble.
        steps = approach_steps_for(
            float(scene.payload_position()[0]), float(scene.place_target[0])
        )
        expert.record.approach_steps = steps
        expert.walk(steps, "walk_to_pick")
        expert.stand(T_STOP_L9, "stop_at_pick")

        start_z = float(scene.payload_position()[2])
        grasp_point = scene.payload_position() + expert.grasp_offset()
        expert.stand(
            T_REACH_L9, "reach", hand_goals={RIGHT_HAND: grasp_point},
            hand_weight=HAND_WEIGHT_STAND,
        )
        # Hold the final target so the task converges instead of stopping
        # wherever the ramp ended. Re-read the object: a 2.5 s reach can nudge
        # it, and the grasp tolerance is what pays for a stale target.
        grasp_point = scene.payload_position() + expert.grasp_offset()
        expert.stand(
            T_REACH_SETTLE, "reach", hand_goals={RIGHT_HAND: grasp_point},
            hand_weight=HAND_WEIGHT_STAND,
        )
        expert._sync_kinematics()
        reached_mm = float(np.linalg.norm(expert.hand_position() - grasp_point)) * 1000.0

        expert.phase = "grasp"
        if reached_mm > GRASP_TOLERANCE_MM:
            # Lab 8 L-M5-b: a weld closed across a gap teleports the object
            # rather than picking it up, so the expert refuses.
            raise Fell(
                f"grasp refused: hand {reached_mm:.1f} mm from the grasp point"
            )
        expert._attach_payload()
        for _ in range(int(T_GRASP_L9 / DT)):
            expert._freeze_balance()
            expert._step()

        lift_goal = scene.payload_position() + np.array([0.0, 0.0, LIFT_CLEARANCE])
        expert.stand(
            T_LIFT_L9, "lift",
            hand_goals={RIGHT_HAND: expert.payload_goal_to_hand(lift_goal)},
        )

        # HOLD — keep the object up long enough that the lift is a state the
        # simulator confirms, not a transient the record caught mid-flight.
        expert.phase = "hold"
        for _ in range(int(T_HOLD_L9 / DT)):
            expert._freeze_balance()
            expert._step()
        expert._sync_kinematics()
        lift_m = float(scene.payload_position()[2]) - start_z

    except Fell as exc:
        expert.record.reason = str(exc)
        if verbose:
            print(f"    seed {seed} {target}: FAILED — {exc}")
    finally:
        expert.close()

    final = scene.payload_position()
    other = [n for n in scene.object_bodies if n != scene.target][0]
    disturbed = float(
        np.linalg.norm(
            scene.object_position(other)[:2] - expert.randomisation.object_xy(other)
        )
    )

    expert.record.metrics = {
        "lift_m": lift_m,
        "reach_error_mm": reached_mm,
        "distractor_moved_m": disturbed,
        "final_height_m": float(final[2]),
        "duration_s": expert.t,
        "tau_max": max(expert.log.tau_max) if expert.log.tau_max else 0.0,
        "approach_steps": expert.record.approach_steps,
        "phases": [
            p for i, p in enumerate(expert.log.phase)
            if i == 0 or expert.log.phase[i - 1] != p
        ],
    }
    # Success is a statement about the simulated world, never about a commanded
    # value (Lab 5's lesson): the named object actually rose, and the other one
    # was left where it stood.
    expert.record.success = bool(
        not expert.record.reason
        and lift_m > MIN_LIFT_M
        and disturbed < DISTRACTOR_TOLERANCE_M
    )
    if not expert.record.success and not expert.record.reason:
        expert.record.reason = (
            f"lifted {lift_m * 1000:.0f} mm, distractor moved {disturbed * 1000:.0f} mm"
        )
    if verbose and expert.record.success:
        print(
            f"    seed {seed} {target}: OK — lifted {lift_m * 1000:.0f} mm, "
            f"{len(expert.record)} frames"
        )
    return expert.record


def task_segments(record: EpisodeRecord) -> dict[str, tuple[int, int]]:
    """Index ranges for each of the four task labels within one episode.

    Args:
        record: A captured episode.

    Returns:
        ``{task: (start, stop)}`` half-open index ranges over the record's
        frames. Tasks with no frames are omitted.
    """
    bounds: dict[str, list[int]] = {}
    for index, phase in enumerate(record.phase):
        task = PHASE_TO_TASK.get(phase)
        if task is None:
            continue
        if task not in bounds:
            bounds[task] = [index, index + 1]
        else:
            bounds[task][1] = index + 1
    return {
        task: (start, stop)
        for task, (start, stop) in bounds.items()
        if task in TASK_NAMES and stop > start
    }


def instruction_for(record: EpisodeRecord, task: str, variant: int = 0) -> str:
    """The instruction sentence for one sliced task segment.

    Args:
        record: The episode the segment came from.
        task: Task label.
        variant: Paraphrase index.

    Returns:
        The instruction string.
    """
    return instruction_label(task, record.target, variant)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one expert demonstration.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", choices=("cup", "box"), default="cup")
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--no-capture", action="store_true")
    args = parser.parse_args()

    record = run_episode(
        args.seed, target=args.target, wide=args.wide,
        capture=not args.no_capture, verbose=True,
    )
    print(f"seed={record.seed} target={record.target} success={record.success}")
    print(f"  reason: {record.reason or '-'}")
    for key, value in record.metrics.items():
        if key != "phases":
            print(f"  {key}: {value}")
    print(f"  frames: {len(record)}  segments: {task_segments(record)}")


if __name__ == "__main__":
    _main()
