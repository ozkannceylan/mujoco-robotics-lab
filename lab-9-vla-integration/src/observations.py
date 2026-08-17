"""Lab 9 — the observation and action contract.

The single place that knows the layout of what the policy sees and what it
emits. Everything else — the expert, the dataset, the model, the evaluator —
goes through these functions, so a layout change is one edit rather than six.

What the policy is allowed to know
----------------------------------
`state` carries joint positions, joint velocities, pelvis height, pelvis roll
and pitch, and the grasp bit. It deliberately **excludes the pelvis's world x,
y and yaw**.

That exclusion is the difference between an evaluation that measures something
and one that does not. A policy handed its own world coordinates can solve
every task in this lab by dead reckoning — walk until x > 0.25, reach to a
fixed offset — without ever looking at an image or reading its instruction, and
would post a high success rate while having learned nothing about either. The
remaining base quantities (height, roll, pitch) are all things a real IMU and
leg kinematics observe, so the restriction is physical rather than arbitrary.

Actions
-------
Two heads, and the reason there are two is Lab 7. See ``tasks/PLAN.md``
deviation 3: the brief specifies joint-position targets for all actuated DOFs,
which on a fixed-base arm is right and on a floating base runs into the finding
that ended Lab 7. The ``task`` head emits what Lab 8's whole-body QP consumes;
the ``joint`` head is the brief's literal version, kept as the ablation that
tests the prediction.

Hand targets are expressed **relative to the pelvis, in its yaw-only frame**,
so the same reach is the same action wherever along the walk it happens.
Yaw-only, not the full pelvis rotation: the pelvis pitches and rolls
continuously while walking, and folding that into the target would inject gait
oscillation into a quantity the policy is supposed to hold still.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from lab9_common import IMAGE_SIZE, NU, STATE_DIM

__all__ = [
    "TASK_ACTION_DIM",
    "JOINT_ACTION_DIM",
    "ACTION_DIMS",
    "TaskAction",
    "ObservationRenderer",
    "build_state",
    "pelvis_frame",
    "world_to_pelvis",
    "pelvis_to_world",
    "encode_task_action",
    "decode_task_action",
    "encode_joint_action",
]

#: right hand (3) + left hand (3) + gait (1) + grasp right (1) + grasp left (1)
TASK_ACTION_DIM: int = 9
#: the brief's literal action space: one joint target per actuated DOF
JOINT_ACTION_DIM: int = NU

ACTION_DIMS: dict[str, int] = {"task": TASK_ACTION_DIM, "joint": JOINT_ACTION_DIM}

_GAIT = 6
_GRASP_RIGHT = 7
_GRASP_LEFT = 8


@dataclass
class TaskAction:
    """A decoded `task`-head action in world coordinates."""

    right_hand: np.ndarray   # (3,) world
    left_hand: np.ndarray    # (3,) world
    gait: float              # >0.5 → take a walk unit
    grasp_right: float       # >0.5 → right weld closed
    grasp_left: float        # >0.5 → left weld closed

    @property
    def walking(self) -> bool:
        return self.gait > 0.5


# ---------------------------------------------------------------------------
# Pelvis frame
# ---------------------------------------------------------------------------


def pelvis_frame(mj_data: mujoco.MjData) -> tuple[np.ndarray, float]:
    """Pelvis position and yaw from the floating-base freejoint.

    Args:
        mj_data: Simulation state.

    Returns:
        ``(position (3,), yaw [rad])``. Yaw is extracted from the pelvis
        quaternion, which MuJoCo stores as ``(w, x, y, z)``.
    """
    position = np.asarray(mj_data.qpos[0:3], dtype=float).copy()
    w, x, y, z = (float(v) for v in mj_data.qpos[3:7])
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return position, yaw


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def world_to_pelvis(point: np.ndarray, position: np.ndarray, yaw: float) -> np.ndarray:
    """Express a world point in the pelvis's yaw-only frame.

    Args:
        point: ``(3,)`` world point.
        position: Pelvis world position.
        yaw: Pelvis yaw [rad].

    Returns:
        ``(3,)`` pelvis-relative point.
    """
    return _yaw_matrix(yaw).T @ (np.asarray(point, dtype=float) - position)


def pelvis_to_world(point: np.ndarray, position: np.ndarray, yaw: float) -> np.ndarray:
    """Inverse of :func:`world_to_pelvis`.

    Args:
        point: ``(3,)`` pelvis-relative point.
        position: Pelvis world position.
        yaw: Pelvis yaw [rad].

    Returns:
        ``(3,)`` world point.
    """
    return _yaw_matrix(yaw) @ np.asarray(point, dtype=float) + position


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def build_state(mj_data: mujoco.MjData, grasped: bool) -> np.ndarray:
    """Assemble the proprioception vector.

    Args:
        mj_data: Simulation state. The robot occupies ``qpos[:36]`` /
            ``qvel[:35]``; a scene may append free bodies after it.
        grasped: Whether any grasp weld is currently closed.

    Returns:
        ``(STATE_DIM,)`` float32: 29 joint positions, 29 joint velocities,
        pelvis height, pelvis roll, pelvis pitch, grasp bit.
    """
    joints = np.asarray(mj_data.qpos[7 : 7 + NU], dtype=np.float32)
    velocities = np.asarray(mj_data.qvel[6 : 6 + NU], dtype=np.float32)

    w, x, y, z = (float(v) for v in mj_data.qpos[3:7])
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    # asin's argument leaves [-1, 1] only on a denormalised quaternion, but a
    # NaN here would poison a whole demonstration silently.
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))

    return np.concatenate(
        [
            joints,
            velocities,
            np.array(
                [mj_data.qpos[2], roll, pitch, 1.0 if grasped else 0.0],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


class ObservationRenderer:
    """Renders the egocentric camera views for one scene.

    One `mujoco.Renderer` is kept alive for the scene's lifetime; constructing
    one per frame costs an EGL context each time. Shadows, reflections and the
    skybox are disabled: measured on this machine they cost 4x per frame
    (380 ms vs 94 ms) and contribute nothing a 128 px policy can use.
    """

    def __init__(self, model: mujoco.MjModel, size: int = IMAGE_SIZE):
        self._renderer = mujoco.Renderer(model, height=size, width=size)
        flags = self._renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.size = size

    def render(self, mj_data: mujoco.MjData, camera: str) -> np.ndarray:
        """Render one camera.

        Args:
            mj_data: Simulation state.
            camera: Camera name, ``"head"`` or ``"wrist"``.

        Returns:
            ``(size, size, 3)`` uint8 RGB.
        """
        self._renderer.update_scene(mj_data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "ObservationRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def encode_task_action(
    right_hand: np.ndarray,
    left_hand: np.ndarray,
    gait: float,
    grasp_right: float,
    grasp_left: float,
    pelvis_position: np.ndarray,
    pelvis_yaw: float,
) -> np.ndarray:
    """Pack a task-space command into the policy's action vector.

    Args:
        right_hand: Right-hand target, world coordinates.
        left_hand: Left-hand target, world coordinates.
        gait: 1 while the expert is walking, 0 while standing.
        grasp_right: 1 while the right weld is closed.
        grasp_left: 1 while the left weld is closed.
        pelvis_position: Pelvis world position.
        pelvis_yaw: Pelvis yaw [rad].

    Returns:
        ``(TASK_ACTION_DIM,)`` float32.
    """
    action = np.zeros(TASK_ACTION_DIM, dtype=np.float32)
    action[0:3] = world_to_pelvis(right_hand, pelvis_position, pelvis_yaw)
    action[3:6] = world_to_pelvis(left_hand, pelvis_position, pelvis_yaw)
    action[_GAIT] = float(gait)
    action[_GRASP_RIGHT] = float(grasp_right)
    action[_GRASP_LEFT] = float(grasp_left)
    return action


def decode_task_action(
    action: np.ndarray, pelvis_position: np.ndarray, pelvis_yaw: float
) -> TaskAction:
    """Unpack an action vector into world-frame commands.

    Args:
        action: ``(TASK_ACTION_DIM,)`` as produced by the policy.
        pelvis_position: Pelvis world position *now*.
        pelvis_yaw: Pelvis yaw *now* [rad].

    Returns:
        A :class:`TaskAction`.
    """
    action = np.asarray(action, dtype=float).reshape(TASK_ACTION_DIM)
    return TaskAction(
        right_hand=pelvis_to_world(action[0:3], pelvis_position, pelvis_yaw),
        left_hand=pelvis_to_world(action[3:6], pelvis_position, pelvis_yaw),
        gait=float(action[_GAIT]),
        grasp_right=float(action[_GRASP_RIGHT]),
        grasp_left=float(action[_GRASP_LEFT]),
    )


def encode_joint_action(mj_data: mujoco.MjData) -> np.ndarray:
    """The brief's literal action space: the actuated joint configuration.

    Args:
        mj_data: Simulation state to read the configuration from.

    Returns:
        ``(JOINT_ACTION_DIM,)`` float32.
    """
    return np.asarray(mj_data.qpos[7 : 7 + NU], dtype=np.float32).copy()


def state_dim_check() -> None:
    """Assert the declared state dimension matches the builder's output.

    Raises:
        AssertionError: If ``lab9_common.STATE_DIM`` and the layout disagree.
    """
    assert STATE_DIM == 2 * NU + 4, (
        f"STATE_DIM={STATE_DIM} but the layout builds {2 * NU + 4}"
    )
