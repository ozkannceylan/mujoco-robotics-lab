"""Lab 8 — M5: the loco-manipulation capstone.

WALK → STOP → REACH → GRASP → LIFT → WALK-CARRY → STOP → PLACE, on the
torque-actuated G1, with every phase driven by the same whole-body
inverse-dynamics QP that M1–M4 built up.

Everything the earlier milestones established is load-bearing here:

* **M3's DCM tracking** drives both walking phases — the capstone walks
  twice, once empty-handed and once carrying.
* **M4's centroidal angular-momentum task** is what lets the arms work while
  the legs walk (L-M4-c) — but it is enabled *only* alongside an arm task
  (L-M5-a). Walking generates angular momentum on purpose, and a term
  commanding `L → 0` across a bare walk cancels the gait itself: measured, it
  took the approach walk from a clean 12.5 mm DCM error to 226.7 mm and a fall
  on the second step.

Each phase is deliberately a configuration an earlier milestone already
validated — approach walk = M3, standing reach = M1, carry walk = M4's carry.
The capstone's job is to sequence proven regimes and survive the *transitions*
between them, not to invent a fourth one.
* **M4's replanning rule** (L-M4-a) is the reason GRASP is a hard boundary:
  picking up 1.5 kg moves the CoM, and a gait plan built before that describes
  a robot that no longer exists. The plan is rebuilt after the weld closes,
  on a Pinocchio model that now carries the payload's inertia.
* **M4's deferral** (L-M4-f) is why REACH happens *stopped*. Reaching while
  walking was measured to be marginal on this robot; standing reach is M1's
  validated regime, and the capstone has no reason to prefer the hard version.

Usage:
    MUJOCO_GL=egl python3 lab-8-loco-manipulation/src/m5_capstone.py

Gate criteria (tasks/PLAN.md M5):
    * the full sequence completes without a fall
    * the payload ends within 50 mm of the place target
    * a post-condition assert on the payload's **simulated** pose — Lab 5's
      lesson, that DONE must verify the object actually moved, not that the
      controller believed it did
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
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

import m3_walking as m3  # noqa: E402
from capstone_scene import (  # noqa: E402
    PAYLOAD_HALF,
    PAYLOAD_MASS,
    build_capstone_scene,
)
from g1_torque_model import torque_limits  # noqa: E402
from lab8_common import (  # noqa: E402
    DT,
    MEDIA_DIR,
    Q_STAND_JOINTS,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    attach_payload_to_pinocchio,
    load_g1_pinocchio,
    mj_state_to_pin,
    pin_point_to_world,
    robot_com,
)
from locomotion_controller import SteppingController  # noqa: E402
from standing_controller import GravityMode, StandingController  # noqa: E402
from wb_id_qp import ContactSpec, WholeBodyIDQP  # noqa: E402
from wb_tasks import (  # noqa: E402
    CentroidalAngularMomentumTask,
    DCMTask,
    FrameOrientationTask,
    FramePositionTask,
    PostureTask,
    TaskStack,
)

LEFT_FOOT = "left_ankle_roll_link"
RIGHT_FOOT = "right_ankle_roll_link"
LEFT_HAND = "left_wrist_yaw_link"
RIGHT_HAND = "right_wrist_yaw_link"

# -- scene --------------------------------------------------------------
PICK_X = 0.40
PLACE_X = 0.75
PLACE_Y = -0.28    # closer in than the pick pedestal — never walked past
APPROACH_STEPS = 3        # ≈0.25 m — puts the payload a comfortable reach away
# The transport leg, sized to what the loaded controller reliably delivers:
# six steps ran out of torque on its closing step, four fell in the walk's
# final settle. The gate asks for a transport, not a marathon — the walking
# *range* is M3's result (12 steps, 1.18 m) and re-proving it while carrying is
# not what this milestone is for. What M5 has to show is that the whole-body
# controller survives acquiring, carrying and releasing mass.
CARRY_STEPS = 3           # ≈0.25 m

# -- phase durations [s] ------------------------------------------------
T_SETTLE = 1.0
T_STOP = 1.2
T_REACH = 2.5
T_GRASP = 0.6
T_LIFT = 1.2
T_TUCK = 1.8
T_PLACE = 2.5
T_RELEASE = 1.5

# -- manipulation -------------------------------------------------------
LIFT_HEIGHT = 0.12        # clearance above the pedestal before moving [m]
# Where the **payload** rides during the carry, in the pelvis frame: in front
# of the chest, on the mid-line. The hands follow from it rather than the other
# way round, because what the balance controller cares about is where the mass
# is (L-M5-g).
#
# TUCK therefore does two things: it brings the load from the pedestal-side
# pose to the body's centre, and it brings the left hand onto the load's other
# side so the second weld can close. Carrying on one arm — even with both arms
# held in symmetric *poses* — leaves the mass one-sided, and that is the
# configuration M4 measured as marginal (L-M4-f). Making the arms symmetric
# bought 0.54 → 0.64 m of transport; making the *load* symmetric is the point
# of this phase.
CARRY_PAYLOAD_LOCAL = np.array([0.255, 0.0, -0.030])
# Two hand weights, because the two regimes have different validated values.
# Walking, M4 measured 1e2 as the ceiling — above it the arm task starts
# trading against balance (L-M4-c). Standing, M1 ran 1e3 and tracked a moving
# hand to 7.08 mm, and standing is not the fragile regime. Using the walking
# value everywhere left a ~60 mm droop on every stationary reach, which lands
# directly on the placement accuracy the gate measures (L-M5-i).
HAND_WEIGHT = 1e2         # while walking — M4's ceiling
HAND_WEIGHT_STAND = 1e3   # while standing — M1's value
HAND_GAIN = 400.0
MOMENTUM_WEIGHT = 1e1     # M4's value — the term that makes carrying possible
MOMENTUM_GAIN = 10.0
# The wrist cannot occupy the payload's centre — it is a solid 90 mm box, and
# commanding the hand there just presses the two geoms together. The grasp
# point sits on the robot's side of the box, where a hand actually goes; the
# weld then holds whatever relative pose was achieved.
GRASP_OFFSET = np.array([-0.060, 0.0, 0.015])
GRASP_TOLERANCE_MM = 70.0  # hand must be this close to the grasp point to weld

# -- gate ---------------------------------------------------------------
PELVIS_FALL_THRESHOLD = 0.50
PLACE_TOLERANCE_M = 0.050

VIDEO_PATH = MEDIA_DIR / "m5_capstone.mp4"
PLOT_PATH = MEDIA_DIR / "m5_capstone_metrics.png"
RENDER_EVERY = int(round(1.0 / (RENDER_FPS * DT)))


@dataclass
class CapstoneLog:
    """Per-tick telemetry for the M5 gate."""

    t: list[float] = field(default_factory=list)
    phase: list[str] = field(default_factory=list)
    pelvis_z: list[float] = field(default_factory=list)
    com_x: list[float] = field(default_factory=list)
    payload: list[np.ndarray] = field(default_factory=list)
    hand_err_mm: list[float] = field(default_factory=list)
    dcm_err_mm: list[float] = field(default_factory=list)
    tau_max: list[float] = field(default_factory=list)
    grasped: list[bool] = field(default_factory=list)


class Fell(RuntimeError):
    """The robot fell; the sequence cannot continue."""


class Capstone:
    """Owns the scene, one task stack, and the phase sequence.

    A single `TaskStack` and `WholeBodyIDQP` live for the whole episode. Phases
    change *what the tasks are asked for*, not which tasks exist — the stance
    controller and the standing controller are two ways of driving the same
    stack. The one thing that genuinely changes mid-episode is the Pinocchio
    model, at the grasp, and that swap is explicit (`_attach_payload`).
    """

    def __init__(self, record: bool = False):
        self.scene = build_capstone_scene(
            DT, pick_x=PICK_X, place_x=PLACE_X, place_pedestal_y=PLACE_Y
        )
        self.mj_model, self.mj_data = self.scene.model, self.scene.data
        self.pin_model, self.pin_data = load_g1_pinocchio()
        self.log = CapstoneLog()
        self.t = 0.0
        self.phase = "settle"
        self.grasped = False
        self.two_handed = False
        self.hand_target: np.ndarray | None = None
        self.payload_in_hand: np.ndarray | None = None  # payload offset, hand frame

        self._settle()
        self._build_stack()

        self.writer = self.renderer = self.camera = None
        if record:
            import imageio

            MEDIA_DIR.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(
                str(VIDEO_PATH), fps=RENDER_FPS, codec="libx264", quality=8,
                macro_block_size=1,
            )
            self.renderer = mujoco.Renderer(
                self.mj_model, height=RENDER_HEIGHT, width=RENDER_WIDTH
            )
            self.camera = mujoco.MjvCamera()
            self.camera.distance = 3.2
            self.camera.azimuth = 140.0
            self.camera.elevation = -12.0

    # -- setup ----------------------------------------------------------

    def _settle(self) -> None:
        settle = StandingController(
            self.mj_model, self.pin_model, self.pin_data,
            gravity_mode=GravityMode.CONTACT_CONSISTENT,
        )
        for _ in range(int(T_SETTLE / DT)):
            settle.step(self.mj_data)

    def _build_stack(self) -> None:
        self.stack = TaskStack(self.pin_model, self.pin_data)
        self._sync_kinematics()

        omega = self._omega()
        self.dcm_task = self.stack.add(
            DCMTask(self.pin_model, omega, weight=m3.DCM_WEIGHT, gain=m3.DCM_GAIN)
        )
        self.pelvis_task = self.stack.add(
            FrameOrientationTask(
                "pelvis", self.pin_model,
                weight=m3.PELVIS_ORI_WEIGHT, gain=m3.PELVIS_ORI_GAIN,
            )
        )
        self.swing_task = self.stack.add(
            FramePositionTask(
                LEFT_FOOT, self.pin_model,
                weight=m3.SWING_WEIGHT, gain=m3.SWING_GAIN, name="swing",
            )
        )
        self.momentum_task = self.stack.add(
            CentroidalAngularMomentumTask(weight=MOMENTUM_WEIGHT, gain=MOMENTUM_GAIN)
        )
        # Off unless an arm task is running — see L-M5-a and the module docstring.
        self.momentum_task.enabled = False
        self.hand_tasks = {
            frame: self.stack.add(
                FramePositionTask(
                    frame, self.pin_model, weight=HAND_WEIGHT, gain=HAND_GAIN,
                    name=f"hand:{frame}",
                )
            )
            for frame in (LEFT_HAND, RIGHT_HAND)
        }
        for task in self.hand_tasks.values():
            task.enabled = False
        self.posture_task = self.stack.add(
            PostureTask(Q_STAND_JOINTS, weight=m3.POSTURE_WEIGHT, gain=m3.POSTURE_GAIN)
        )
        self.pelvis_task.capture_current(self.pin_data)

        self.qp = WholeBodyIDQP(
            self.pin_model, self.pin_data,
            [ContactSpec(LEFT_FOOT), ContactSpec(RIGHT_FOOT)],
            torque_limits(self.mj_model),
        )

    def _sync_kinematics(self) -> None:
        q, v = mj_state_to_pin(self.mj_data)
        self.stack.update_dynamics(q, v)
        return q, v

    def _omega(self) -> float:
        from lab8_common import lipm_omega

        com = pin_point_to_world(self.pin_data.com[0])
        feet = [
            pin_point_to_world(self.pin_data.oMf[self.pin_model.getFrameId(f)].translation)
            for f in (LEFT_FOOT, RIGHT_FOOT)
        ]
        return lipm_omega(float(com[2] - 0.5 * (feet[0][2] + feet[1][2])))

    def hand_position(self, frame: str = RIGHT_HAND) -> np.ndarray:
        return self.hand_tasks[frame].current_position(self.pin_data)

    # -- the tick -------------------------------------------------------

    def _step(self, controller: SteppingController | None = None) -> None:
        """One control tick: solve, apply, integrate, record."""
        q, v = mj_state_to_pin(self.mj_data)
        self.stack.update_dynamics(q, v)
        result = self.qp.solve(self.stack, q, v)
        self.mj_data.ctrl[:] = result.tau
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.t += DT
        self._record(result.tau)
        del controller

        if self.writer is not None and len(self.log.t) % RENDER_EVERY == 0:
            self.camera.lookat[:] = [self.mj_data.qpos[0], self.mj_data.qpos[1], 0.8]
            self.renderer.update_scene(self.mj_data, camera=self.camera)
            self.writer.append_data(self.renderer.render())

        if self.mj_data.qpos[2] < PELVIS_FALL_THRESHOLD:
            raise Fell(f"pelvis dropped to {self.mj_data.qpos[2]:.3f} m at t={self.t:.2f}s")

    def _record(self, tau: np.ndarray) -> None:
        self.log.t.append(self.t)
        self.log.phase.append(self.phase)
        self.log.pelvis_z.append(float(self.mj_data.qpos[2]))
        self.log.com_x.append(float(robot_com(self.mj_model, self.mj_data)[0]))
        self.log.payload.append(self.scene.payload_position())
        self.log.tau_max.append(float(np.abs(tau).max()))
        self.log.grasped.append(self.grasped)
        target = self.hand_target
        self.log.hand_err_mm.append(
            0.0 if target is None
            else float(np.linalg.norm(target - self.hand_position())) * 1000.0
        )
        self.log.dcm_err_mm.append(
            float(np.linalg.norm(
                self.dcm_task.xi_target - self.dcm_task.current_dcm(self.pin_data)
            )) * 1000.0
        )

    # -- phases ---------------------------------------------------------

    def _freeze_balance(self) -> None:
        """Hold the capture point where it is and stand on both feet."""
        self.qp.set_contacts([ContactSpec(LEFT_FOOT), ContactSpec(RIGHT_FOOT)])
        self.dcm_task.set_reference(self.dcm_task.current_dcm(self.pin_data), np.zeros(2))
        lower = np.full(2, np.inf)
        upper = np.full(2, -np.inf)
        patch = ContactSpec("")
        offset = np.array([patch.center_x, patch.center_y])
        half = np.array([patch.half_length, patch.half_width]) * m3.VRP_SHRINK
        for frame in (LEFT_FOOT, RIGHT_FOOT):
            centre = np.asarray(
                self.pin_data.oMf[self.pin_model.getFrameId(frame)].translation
            )[:2] + offset
            lower = np.minimum(lower, centre - half)
            upper = np.maximum(upper, centre + half)
        self.dcm_task.set_vrp_bounds(lower, upper)
        self.swing_task.enabled = False

    def stand(self, duration: float, phase: str,
              hand_goals: dict[str, np.ndarray] | None = None,
              hand_weight: float = HAND_WEIGHT,
              payload_goal: np.ndarray | None = None) -> None:
        """Stand still; optionally move hands to `hand_goals` over `duration`.

        Hands are driven along a raised-cosine so they start and end at rest —
        a step target on a 1e2-weight task against a frozen capture point is a
        disturbance the balance controller has to absorb for no reason.

        `payload_goal` servos the **payload** rather than the right hand: the
        hand target is recomputed every tick from the live hand→payload offset.
        The grip is compliant by design, so the load shifts a little during a
        2.5 s motion, and a hand target derived from a single pre-motion
        measurement bakes that shift into the result — a systematic 55 mm
        outboard placement error, on the one quantity the gate measures
        (L-M5-i). The object is what the task is about, so the object is what
        gets closed around.
        """
        self.phase = phase
        self._sync_kinematics()
        self._freeze_balance()

        goals = hand_goals or {}
        starts = {frame: self.hand_position(frame) for frame in goals}
        for frame, task in self.hand_tasks.items():
            task.enabled = frame in goals
            task.weight = hand_weight
        # Standing reach is M1's regime, which had no momentum term and tracked
        # to 7.08 mm. Nothing here needs one, and L-M5-a is the cost of adding
        # it where it is not earning anything.
        self.momentum_task.enabled = False

        payload_start = self.scene.payload_position() if payload_goal is not None else None
        if payload_goal is not None:
            self.hand_tasks[RIGHT_HAND].enabled = True
            self.hand_tasks[RIGHT_HAND].weight = hand_weight

        ticks = int(duration / DT)
        for i in range(ticks):
            alpha = (i + 1) / ticks
            blend = 0.5 * (1.0 - np.cos(np.pi * alpha))
            rate = 0.5 * np.pi * np.sin(np.pi * alpha) / duration
            for frame, goal in goals.items():
                delta = goal - starts[frame]
                target = starts[frame] + blend * delta
                self.hand_tasks[frame].set_target(target, rate * delta)
                if frame == RIGHT_HAND:
                    self.hand_target = target
            if payload_goal is not None:
                delta = payload_goal - payload_start
                waypoint = payload_start + blend * delta
                target = self.payload_goal_to_hand(waypoint)
                self.hand_tasks[RIGHT_HAND].set_target(target, rate * delta)
                self.hand_target = target
            self._step()

    def carry_targets(self) -> dict[str, np.ndarray]:
        """Both hands' carry poses in world coordinates.

        Derived from where the **payload** should ride, not from nominal hand
        poses: the right hand's position is whatever puts the welded load on
        the chest mid-line, and the left mirrors it about the load.
        """
        pelvis = self.pin_data.oMf[self.pin_model.getFrameId("pelvis")]
        origin = pin_point_to_world(pelvis.translation)
        payload_goal = origin + pelvis.rotation @ CARRY_PAYLOAD_LOCAL
        right = self.payload_goal_to_hand(payload_goal)
        # Mirror the right hand's grip about the payload's sagittal plane, so
        # the two hands sit symmetrically on the load.
        offset = right - payload_goal
        left = payload_goal + np.array([offset[0], -offset[1], offset[2]])
        return {RIGHT_HAND: right, LEFT_HAND: left}

    def walk(self, n_steps: int, phase: str, hold_hands: bool = False) -> None:
        """Walk `n_steps` under M3's DCM controller.

        `hold_hands` keeps both hands on a Cartesian pose that travels with the
        planned CoM — M4's carry configuration, which was the *robust* one
        (12/12 across perturbations) where a single held hand was not (L-M4-f).
        The payload rides on the right hand, so the left is held purely to keep
        the upper body symmetric.
        """
        self.phase = phase
        q, v = mj_state_to_pin(self.mj_data)
        self.stack.update_dynamics(q, v)

        # Close the stance at the end of every capstone walk. The sequence
        # stops to manipulate and then walks again, and a gait that ends
        # mid-stride hands the next one a staggered start it cannot survive
        # (L-M5-e).
        schedule, plan = m3.make_plan(
            self.pin_model, self.pin_data,
            step_length=m3.STEP_LENGTH, n_steps=n_steps, close_stance=True,
        )
        self.dcm_task.omega = plan.omega
        controller = SteppingController(
            self.mj_model, self.mj_data, self.pin_model, self.pin_data,
            schedule, self.qp, self.stack, self.dcm_task, self.swing_task,
            dcm_plan=plan, vrp_shrink=m3.VRP_SHRINK,
        )

        # Hold whatever pose the hands are in now (after TUCK, both are on the
        # load) and let it travel with the planned CoM.
        homes = {f: self.hand_position(f) for f in self.hand_tasks}
        com_home = plan.nominal_com(0.0)[0].copy()
        for frame, task in self.hand_tasks.items():
            task.enabled = hold_hands
            task.weight = HAND_WEIGHT
        # The momentum task rides with the hand tasks: together they are M4's
        # validated carry configuration; alone the momentum term fights the
        # gait's own angular momentum (L-M5-a).
        self.momentum_task.enabled = hold_hands

        for step in range(int(schedule.total_duration / DT)):
            t_local = step * DT
            controller.update_targets(t_local)
            if hold_hands:
                reference = plan.reference(t_local)
                travel = np.array([
                    reference.com[0] - com_home[0], reference.com[1] - com_home[1], 0.0
                ])
                velocity = np.array([
                    reference.com_velocity[0], reference.com_velocity[1], 0.0
                ])
                for frame, task in self.hand_tasks.items():
                    target = homes[frame] + travel
                    task.set_target(target, velocity)
                    if frame == RIGHT_HAND:
                        self.hand_target = target
            self._step(controller)

        for task in self.hand_tasks.values():
            task.enabled = False
        self.momentum_task.enabled = False
        self.hand_target = None

    def _attach_payload(self) -> None:
        """Close the weld and tell the controller about the mass it just took.

        Order matters. The weld closes first so the payload's pose is settled
        by the simulator, then the offset is captured *in the hand frame* (the
        wrist rotates over the sequence; a world-frame offset would slowly
        become wrong), then the Pinocchio model gains the inertia and every
        holder of the old `pin.Data` is re-pointed at the new one.
        """
        self.scene.set_weld(True)
        self.grasped = True
        self._sync_kinematics()

        hand = self.pin_data.oMf[self.pin_model.getFrameId(RIGHT_HAND)]
        hand_world = pin_point_to_world(hand.translation)
        self.payload_in_hand = hand.rotation.T @ (
            self.scene.payload_position() - hand_world
        )

        self.pin_data = attach_payload_to_pinocchio(
            self.pin_model, PAYLOAD_MASS, PAYLOAD_HALF, RIGHT_HAND, self.payload_in_hand
        )
        # Everything that cached the old data object must follow it.
        self.stack.data = self.pin_data
        self.qp.data = self.pin_data
        self._sync_kinematics()

    def payload_goal_to_hand(self, payload_goal: np.ndarray) -> np.ndarray:
        """Hand target that puts the held payload at `payload_goal` (world).

        The hand→payload offset is re-measured from the **live** state rather
        than reused from the grasp. The weld is deliberately compliant
        (`solref` 0.02) so it behaves like a firm hand rather than a rigid bar,
        which means the load settles a little in the grip over a 25 s sequence;
        a grasp-time offset is stale by the time the place is commanded, and
        the error lands directly on the placement.
        """
        frame = self.pin_data.oMf[self.pin_model.getFrameId(RIGHT_HAND)]
        hand_world = pin_point_to_world(frame.translation)
        offset = frame.rotation.T @ (self.scene.payload_position() - hand_world)
        return payload_goal - frame.rotation @ offset

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def run(record: bool = True) -> dict:
    """Execute the capstone sequence; return gate metrics."""
    capstone = Capstone(record=record)
    scene = capstone.scene
    result: dict = {"fell": False, "fell_at": None, "reason": ""}
    reached_mm = grasp_gap_mm = release_error_mm = float("nan")

    try:
        # 1. WALK to the pick pedestal.
        capstone.walk(APPROACH_STEPS, "walk_to_pick")

        # 2. STOP — settle before manipulating.
        capstone.stand(T_STOP, "stop_at_pick")

        # 3. REACH — stopped, per M4's deferral (L-M4-f).
        grasp_point = scene.payload_position() + GRASP_OFFSET
        capstone.stand(T_REACH, "reach", hand_goals={RIGHT_HAND: grasp_point})
        capstone._sync_kinematics()
        reached_mm = float(
            np.linalg.norm(capstone.hand_position() - grasp_point)
        ) * 1000.0

        # 4. GRASP — weld, then fold the payload into the controller's model.
        capstone.phase = "grasp"
        grasp_gap_mm = reached_mm
        if reached_mm > GRASP_TOLERANCE_MM:
            # Refusing to weld across a gap is the point: a weld is a rigid
            # constraint, and closing one the hand has not actually reached
            # teleports the payload instead of picking it up.
            raise Fell(
                f"grasp refused: hand {reached_mm:.1f} mm from the grasp point "
                f"(tolerance {GRASP_TOLERANCE_MM:.0f} mm)"
            )
        capstone._attach_payload()
        for _ in range(int(T_GRASP / DT)):
            capstone._freeze_balance()
            capstone._step()

        # 5. LIFT — straight up, clear of the pedestal.
        lift_goal = scene.payload_position() + np.array([0.0, 0.0, LIFT_HEIGHT])
        capstone.stand(
            T_LIFT, "lift", hand_goals={RIGHT_HAND: capstone.payload_goal_to_hand(lift_goal)}
        )

        # 6. TUCK — bring the load to the chest mid-line and put the left hand
        #    on its far side, then close the second weld. Walking with the load
        #    out on one arm is the asymmetric configuration M4 measured to be
        #    marginal (L-M4-f); this phase is what makes the *mass* symmetric
        #    rather than just the arms holding it (L-M5-g).
        capstone.stand(T_TUCK, "tuck", hand_goals=capstone.carry_targets())
        capstone._sync_kinematics()
        left_gap_mm = float(np.linalg.norm(
            capstone.hand_position(LEFT_HAND)
            - capstone.carry_targets()[LEFT_HAND]
        )) * 1000.0
        if left_gap_mm > GRASP_TOLERANCE_MM:
            raise Fell(
                f"second grasp refused: left hand {left_gap_mm:.1f} mm from the "
                f"load (tolerance {GRASP_TOLERANCE_MM:.0f} mm)"
            )
        scene.set_weld(True, which="left")
        capstone.two_handed = True

        # 7. WALK-CARRY — M4's carry configuration, now with real mass.
        capstone.walk(CARRY_STEPS, "walk_carry", hold_hands=True)

        # 8. STOP before placing.
        capstone.stand(T_STOP, "stop_at_place")

        # 9. PLACE — one-handed. The left weld opens *first*: with both wrists
        #    welded the arms and the payload form a closed kinematic chain, and
        #    the left arm (its task now off, but still rigidly attached) drags
        #    against the placing motion. Measured: the payload reached only
        #    halfway to the target and was released in mid-air (L-M5-h).
        scene.set_weld(False, which="left")
        capstone.two_handed = False
        capstone._sync_kinematics()
        place_goal = scene.place_target + np.array([0.0, 0.0, 0.004])
        # The one phase whose accuracy the gate measures, and the only one run
        # at M1's standing weight. Raising it in the earlier phases too was
        # measured to destabilise the tuck (fall at t=13.9 s): every phase gets
        # the weight its own milestone validated, and precision is bought only
        # where precision is what is being asked for (L-M5-i).
        capstone.stand(
            T_PLACE, "place", payload_goal=place_goal,
            hand_weight=HAND_WEIGHT_STAND,
        )
        capstone._sync_kinematics()
        at_release = scene.payload_position().copy()
        release_error_mm = float(np.linalg.norm(at_release - scene.place_target)) * 1000.0
        print(f"    [place] payload at release {np.round(at_release, 3)} "
              f"→ {release_error_mm:.1f} mm from target")
        scene.set_weld(False, which="right")
        capstone.grasped = False
        capstone.hand_target = None
        for task in capstone.hand_tasks.values():
            task.enabled = False
        capstone.phase = "release"
        capstone._sync_kinematics()
        capstone._freeze_balance()
        for _ in range(int(T_RELEASE / DT)):
            capstone._step()

    except Fell as exc:
        result["fell"] = True
        result["fell_at"] = capstone.t
        result["reason"] = str(exc)
    finally:
        capstone.close()

    final = scene.payload_position()
    error = float(np.linalg.norm(final - scene.place_target))
    log = capstone.log
    result.update({
        "payload_start": scene.pick_position,
        "payload_final": final,
        "place_target": scene.place_target,
        "place_error_m": error,
        "payload_travel_m": float(np.linalg.norm(final - scene.pick_position)),
        "reach_error_mm": reached_mm,
        "release_error_mm": release_error_mm,
        "grasp_gap_mm": grasp_gap_mm,
        "duration": capstone.t,
        "tau_max": max(log.tau_max) if log.tau_max else 0.0,
        "com_travel_m": (log.com_x[-1] - log.com_x[0]) if log.com_x else 0.0,
        "phases": [p for i, p in enumerate(log.phase)
                   if i == 0 or log.phase[i - 1] != p],
        "log": log,
    })
    return result


def plot_metrics(log: CapstoneLog, result: dict, path: Path) -> None:
    """Payload trajectory, hand tracking, balance and torque across phases."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    t = np.array(log.t)
    payload = np.array(log.payload)

    # Phase bands make the sequence readable at a glance.
    boundaries, labels = [], []
    for i, phase in enumerate(log.phase):
        if i == 0 or log.phase[i - 1] != phase:
            boundaries.append(t[i])
            labels.append(phase)
    boundaries.append(t[-1])
    for ax in axes:
        for k in range(len(labels)):
            if k % 2 == 0:
                ax.axvspan(boundaries[k], boundaries[k + 1], color="0.93", zorder=0)
    for k, label in enumerate(labels):
        axes[0].text(
            0.5 * (boundaries[k] + boundaries[k + 1]), 1.02, label,
            transform=axes[0].get_xaxis_transform(), ha="center", fontsize=7, rotation=30,
        )

    axes[0].plot(t, payload[:, 0], "C1", label="payload x")
    axes[0].plot(t, payload[:, 2], "C2", label="payload z")
    axes[0].axhline(result["place_target"][0], color="C1", ls="--", lw=0.8)
    axes[0].axhline(result["place_target"][2], color="C2", ls="--", lw=0.8)
    axes[0].set_ylabel("m")
    axes[0].legend(fontsize=8, loc="center left")

    axes[1].plot(t, log.hand_err_mm, "C3", lw=0.9, label="hand → target")
    axes[1].plot(t, log.dcm_err_mm, "C4", lw=0.9, label="DCM error")
    axes[1].set_ylabel("mm")
    axes[1].set_ylim(0, 200)
    axes[1].legend(fontsize=8)

    axes[2].plot(t, log.pelvis_z, "C0", label="pelvis height")
    axes[2].axhline(PELVIS_FALL_THRESHOLD, color="C3", ls="--", lw=0.8, label="fall threshold")
    axes[2].plot(t, log.com_x, "C5", label="robot CoM x")
    axes[2].set_ylabel("m")
    axes[2].legend(fontsize=8)

    axes[3].plot(t, log.tau_max, "C5")
    axes[3].axhline(139.0, color="C3", ls="--", lw=0.8, label="actuator limit")
    axes[3].set_ylabel("max |τ| (N·m)")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title("Lab 8 M5 — Loco-manipulation capstone", pad=28)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the capstone and write evidence."""
    parser = argparse.ArgumentParser(description="Lab 8 M5 — loco-manipulation capstone")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print(" Lab 8 — M5: Loco-Manipulation Capstone")
    print("=" * 72)
    print(f"\n  WALK({APPROACH_STEPS}) → STOP → REACH → GRASP → LIFT → TUCK → "
          f"WALK-CARRY({CARRY_STEPS}) → STOP → PLACE")
    print(f"  payload {PAYLOAD_MASS} kg ({PAYLOAD_MASS / 33.34:.1%} of body mass) · "
          f"pick x={PICK_X} → place x={PLACE_X}")

    result = run(record=not args.no_video)
    plot_metrics(result["log"], result, PLOT_PATH)

    print(f"\n  phases run     : {' → '.join(result['phases'])}")
    print(f"  duration       : {result['duration']:.1f} s")
    print(f"  robot travelled: {result['com_travel_m']:.3f} m")
    print(f"  reach error    : {result['reach_error_mm']:.1f} mm (hand → payload at grasp)")
    print(f"  payload moved  : {result['payload_travel_m']:.3f} m")
    print(f"  payload final  : {np.round(result['payload_final'], 3)}")
    print(f"  place target   : {np.round(result['place_target'], 3)}")
    print(f"  place error    : {result['place_error_m'] * 1000:.1f} mm")
    print(f"  peak torque    : {result['tau_max']:.1f} N·m")
    if result["fell"]:
        print(f"  FALL: {result['reason']}")
    print(f"\n  plot : {PLOT_PATH}")
    if not args.no_video and VIDEO_PATH.exists():
        print(f"  video: {VIDEO_PATH}  ({VIDEO_PATH.stat().st_size / 1e6:.1f} MB)")

    completed = not result["fell"] and result["phases"][-1] == "release"
    checks = [
        ("Full sequence, no fall", completed,
         f"{len(result['phases'])} phases"
         + ("" if not result["fell"] else f", fell at {result['fell_at']:.2f}s")),
        ("Payload within 50 mm of target",
         result["place_error_m"] <= PLACE_TOLERANCE_M,
         f"{result['place_error_m'] * 1000:.1f} mm"),
        ("Payload actually transported", result["payload_travel_m"] >= 0.30,
         f"{result['payload_travel_m']:.3f} m"),
        ("Torques within limits", result["tau_max"] <= 139.0,
         f"{result['tau_max']:.1f} N·m peak"),
    ]

    print("\n" + "=" * 72)
    print(" M5 GATE")
    print("=" * 72)
    print(f" {'criterion':34s} {'result':>8s}   measured")
    print(" " + "-" * 69)
    for name, passed, detail in checks:
        print(f" {name:34s} {'PASS' if passed else 'FAIL':>8s}   {detail}")
    all_passed = all(passed for _, passed, _ in checks)
    print("=" * 72)
    print(" M5: PASS — walk, pick, carry, place" if all_passed
          else " M5: FAIL — milestone still open, see tasks/LESSONS.md § M5")
    print("=" * 72)

    if all_passed:
        # Post-condition on the SIMULATED pose (Lab 5's lesson): a capstone
        # that only checks its own commands can report success while the
        # object never moved.
        final = np.asarray(result["payload_final"])
        assert np.linalg.norm(final - result["place_target"]) <= PLACE_TOLERANCE_M, (
            f"post-condition: payload at {final}, target {result['place_target']}"
        )
        assert np.linalg.norm(final - result["payload_start"]) >= 0.30, (
            "post-condition: payload never left its pedestal"
        )
        print(" post-condition asserts on the simulated payload pose: OK")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
