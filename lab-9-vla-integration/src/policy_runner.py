"""Lab 9 — closed-loop execution: the policy proposes, Lab 8's QP disposes.

The policy runs at 10 Hz and emits task-space commands; Lab 8's whole-body
inverse-dynamics QP runs at 1 kHz and keeps the robot upright. Balance is never
a learned quantity — the reason is `tasks/PLAN.md` deviation 3 and, before that,
Lab 7's finding that a joint-position reference cannot stabilise this robot.

The executor is a small state machine, and its shape is forced by the gait:

* **STAND** — the balance reference is frozen and the hands track whatever the
  policy last asked for. The policy is re-polled every 100 control ticks.
* **WALK** — a Lab 8 gait unit is running. A biped cannot be told to stop in the
  middle of a step, so the gait command is only acted on at unit boundaries; a
  unit is one step plus its closing step, which is the configuration Lab 8
  validated (L-M5-e: a walk that ends mid-stride hands the next one a stance it
  cannot survive).

The grasp is a weld, as in Lab 8, and it closes only when the policy asks *and*
the hand has actually reached an object — a weld closed across a gap teleports
the object instead of picking it up (Lab 8 L-M5-b).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from lab9_common import (
    DT,
    POLICY_DECIMATION,
    load_g1_pinocchio,
    mj_state_to_pin,
)
from observations import (
    ObservationRenderer,
    build_state,
    decode_task_action,
    pelvis_frame,
)
from vla_scene import Randomisation, build_vla_scene

# Lab 8.
from m5_capstone import (  # noqa: E402
    GRASP_TOLERANCE_MM,
    HAND_WEIGHT_STAND,
    LEFT_HAND,
    RIGHT_HAND,
    Capstone,
    Fell,
)

__all__ = ["RolloutResult", "PolicyRunner"]

#: Steps in one walk unit. One step plus the closing step, so every unit starts
#: and ends from a stance the gait was validated on.
WALK_UNIT_STEPS: int = 1

#: A rollout is stopped here whatever the policy is doing. Long enough for four
#: walk units and a full manipulation, short enough that a policy which has
#: decided to stand still for ever does not cost a minute of wall clock.
MAX_EPISODE_S: float = 22.0

#: How close the hand must be to the object before a grasp request is honoured.
GRASP_GATE_MM: float = GRASP_TOLERANCE_MM


@dataclass
class RolloutResult:
    """What one closed-loop episode did."""

    seed: int
    target: str
    instruction: str
    task: str
    success: bool = False
    reason: str = ""
    fell: bool = False
    walk_units: int = 0
    grasped: bool = False
    lift_m: float = 0.0
    distractor_moved_m: float = 0.0
    final_pelvis_x: float = 0.0
    expert_pelvis_x: float = 0.0
    stop_error_m: float = float("nan")
    stop_error_other_m: float = float("nan")
    walk_units_expert: int = 0
    grasped_object: str = ""
    hand_error_mm: float = float("nan")
    duration_s: float = 0.0
    tau_max: float = 0.0
    inferences: int = 0
    gait_commands: list[float] = field(default_factory=list)


class PolicyRunner(Capstone):
    """Runs a trained policy in closed loop over the Lab 9 scene.

    Args:
        seed: Scene seed.
        target: Object the instruction names.
        model: A trained `act_policy.ACTPolicy`.
        bank: Its instruction bank.
        wide: Use the wider object-placement range.
        image_size: Camera resolution.
    """

    def __init__(
        self,
        seed: int,
        target: str,
        model,
        bank,
        wide: bool = False,
        image_size: int | None = None,
    ):
        from lab9_common import IMAGE_SIZE
        from m5_capstone import CapstoneLog

        self.randomisation = Randomisation.sample(seed, wide=wide)
        self.scene = build_vla_scene(DT, self.randomisation, target=target)
        self.mj_model, self.mj_data = self.scene.model, self.scene.data
        self.pin_model, self.pin_data = load_g1_pinocchio()

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

        self.model = model
        self.bank = bank
        self.obs = ObservationRenderer(self.mj_model, size=image_size or IMAGE_SIZE)
        self.frame_hook = None      # set by the recorder
        self._inferences = 0
        self._gait_commands: list[float] = []

    # -- the policy ------------------------------------------------------

    def _observe(self) -> dict:
        """Build one observation from the live simulator state."""
        return {
            "head": self.obs.render(self.mj_data, "head"),
            "wrist": self.obs.render(self.mj_data, "wrist"),
            "state": build_state(self.mj_data, self.scene.any_weld_active()),
        }

    def infer(self, instruction: str):
        """Run the policy once and decode the first action of the chunk.

        Args:
            instruction: The natural-language command.

        Returns:
            A decoded `observations.TaskAction` for the `task` head, or the raw
            chunk for the `joint` head.
        """
        observation = self._observe()
        images = {
            camera: torch.from_numpy(
                observation[camera].astype(np.float32).transpose(2, 0, 1) / 255.0
            ).unsqueeze(0)
            for camera in self.model.cameras
        }
        state = torch.from_numpy(observation["state"]).unsqueeze(0)
        if self.model.conditioning == "text":
            conditioning = torch.from_numpy(
                self.bank.get(instruction)
            ).unsqueeze(0)
        else:
            conditioning = torch.tensor([self._task_id(instruction)], dtype=torch.long)

        chunk = self.model.predict(images, state, conditioning)[0].numpy()
        self._inferences += 1
        if self.model.action_head == "joint":
            return chunk
        position, yaw = pelvis_frame(self.mj_data)
        return decode_task_action(chunk[0], position, yaw)

    @staticmethod
    def _task_id(instruction: str) -> int:
        """Integer conditioning id, for the `task_id` ablation only."""
        from lab9_common import OBJECT_NAMES, TASK_NAMES

        task = "walk" if instruction.split()[0] in ("walk", "go", "approach") else "pick"
        obj = "cup" if "cup" in instruction else "box"
        return TASK_NAMES.index(task) * 2 + OBJECT_NAMES.index(obj)

    # -- execution -------------------------------------------------------

    def _step(self, controller=None) -> None:
        super()._step(controller)
        if self.frame_hook is not None:
            self.frame_hook(self)

    def _apply_hands(self, action, weight: float = HAND_WEIGHT_STAND) -> None:
        """Point the hand tasks at the policy's targets."""
        for frame, target in (
            (RIGHT_HAND, action.right_hand), (LEFT_HAND, action.left_hand)
        ):
            task = self.hand_tasks[frame]
            task.enabled = True
            task.weight = weight
            task.set_target(np.asarray(target, dtype=float))
        self.hand_target = np.asarray(action.right_hand, dtype=float)

    def stand_tick(self, action, ticks: int = POLICY_DECIMATION) -> None:
        """Hold balance and track the policy's hand targets for one policy period.

        Args:
            action: The decoded action.
            ticks: Control ticks to run.
        """
        self.phase = "stand"
        self._sync_kinematics()
        self._freeze_balance()
        self._apply_hands(action)
        self.momentum_task.enabled = False
        for _ in range(ticks):
            self._step()

    def walk_unit(self) -> None:
        """Execute one Lab 8 gait unit: a step plus its closing step."""
        self.phase = "walk"
        for task in self.hand_tasks.values():
            task.enabled = False
        self.hand_target = None
        self.momentum_task.enabled = False
        # `walk` is Lab 8's own method: it plans a DCM trajectory through the
        # footsteps, builds a SteppingController, and runs it to completion.
        self.walk(WALK_UNIT_STEPS, "walk")

    def joint_tick(self, chunk: np.ndarray, ticks: int = POLICY_DECIMATION) -> None:
        """Execute the brief's literal action space: joint targets under PD.

        This is the ablation, and it is here to be measured rather than
        believed. Lab 7's finding is that a joint-position reference tracked by
        PD cannot stabilise this robot — that is why its ZMP walking failed and
        why Lab 8 rebuilt the stack around a whole-body QP. If it holds, this
        runner falls; if it does not, the task head was unnecessary.

        Gains are Lab 8's standing controller values, and gravity compensation
        is supplied, so the ablation is the *action space* rather than a
        strawman controller.

        Args:
            chunk: ``(chunk_size, 29)`` predicted joint targets.
            ticks: Control ticks to run.
        """
        from standing_controller import GravityMode, StandingController

        if getattr(self, "_joint_controller", None) is None:
            self._joint_controller = StandingController(
                self.mj_model, self.pin_model, self.pin_data,
                gravity_mode=GravityMode.CONTACT_CONSISTENT,
            )
        self.phase = "joint"
        controller = self._joint_controller
        for index in range(ticks):
            # Walk along the chunk at the policy rate the chunk was trained at.
            step = min(index * len(chunk) // max(ticks, 1), len(chunk) - 1)
            controller.q_nom = np.asarray(chunk[step], dtype=float)
            controller.step(self.mj_data)
            self.t += DT
            self._record(np.asarray(self.mj_data.ctrl, dtype=float))
            if self.frame_hook is not None:
                self.frame_hook(self)
            if self.mj_data.qpos[2] < 0.50:
                raise Fell(f"pelvis dropped to {self.mj_data.qpos[2]:.3f} m "
                           f"at t={self.t:.2f}s")

    def try_grasp(self, action) -> bool:
        """Close the weld if the policy asks and the hand has arrived.

        Args:
            action: The decoded action.

        Returns:
            Whether a grasp was made.
        """
        if self.grasped or action.grasp_right <= 0.5:
            return False
        self._sync_kinematics()
        hand = self.hand_position(RIGHT_HAND)
        distances = {
            name: float(np.linalg.norm(hand - self.scene.object_position(name)))
            for name in self.scene.object_bodies
        }
        nearest = min(distances, key=distances.get)
        if distances[nearest] * 1000.0 > GRASP_GATE_MM:
            return False
        # Grasp whatever the hand actually reached, not what the instruction
        # named. Picking the wrong object is a result the evaluation has to be
        # able to see.
        self.scene.set_target(nearest)
        self._attach_payload()
        return True

    def _attach_payload(self) -> None:
        """Lab 8's grasp with this lab's object mass (see `expert.VLAExpert`)."""
        from capstone_scene import PAYLOAD_HALF
        from lab8_common import attach_payload_to_pinocchio
        from lab9_common import pin_point_to_world
        from vla_scene import OBJECT_MASS

        self.scene.set_weld(True)
        self.grasped = True
        self._sync_kinematics()
        hand = self.pin_data.oMf[self.pin_model.getFrameId(RIGHT_HAND)]
        self.payload_in_hand = hand.rotation.T @ (
            self.scene.payload_position() - pin_point_to_world(hand.translation)
        )
        self.pin_data = attach_payload_to_pinocchio(
            self.pin_model, OBJECT_MASS, PAYLOAD_HALF, RIGHT_HAND, self.payload_in_hand
        )
        self.stack.data = self.pin_data
        self.qp.data = self.pin_data
        self._sync_kinematics()

    def close(self) -> None:
        super().close()
        if getattr(self, "obs", None) is not None:
            self.obs.close()
            self.obs = None
