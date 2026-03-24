"""Lab 9 — VLA Integration: Common constants, paths, and helper functions.

Provides model loading, camera rendering, proprioception extraction,
and task definitions shared across all Lab 9 modules.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LAB_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = LAB_DIR / "models"
MEDIA_DIR = LAB_DIR / "media"
DATA_DIR = LAB_DIR / "data"
DEMOS_DIR = DATA_DIR / "demos"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

# Ensure output directories exist
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
DEMOS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
ACTION_CHUNK_SIZE: int = 10
POLICY_DT: float = 1.0 / 30.0
SIM_DT: float = 0.005  # MuJoCo timestep
STEPS_PER_POLICY: int = int(POLICY_DT / SIM_DT)  # sim steps between policy queries

# Robot structure — 29 actuated joints total
NUM_ROBOT_JOINTS: int = 29
# Proprioception = joint positions (29) + joint velocities (29) = 58
PROPRIO_DIM: int = NUM_ROBOT_JOINTS * 2
# Action = joint position targets for all actuated joints
ACTION_DIM: int = NUM_ROBOT_JOINTS

# Free-joint qpos offset: root(7) then robot joints start at index 7
ROBOT_QPOS_START: int = 7
ROBOT_QPOS_END: int = ROBOT_QPOS_START + NUM_ROBOT_JOINTS  # 36

# Object body names and their free-joint qpos addresses
OBJECT_NAMES: list[str] = ["cup", "box", "bottle"]

# Right arm joint indices within the robot joint array (0-indexed from robot start)
RIGHT_ARM_JOINT_INDICES: list[int] = [10, 11, 12, 13, 14, 15, 16]  # shoulder_roll..gripper
RIGHT_ARM_JOINT_NAMES: list[str] = [
    "right_shoulder_roll", "right_shoulder_pitch", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_gripper",
]

# Camera names
WRIST_CAM: str = "right_wrist_cam"
HEAD_CAM: str = "head_cam"


# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class TaskDefinition:
    """Definition of a manipulation task for the VLA pipeline."""

    name: str
    language: str
    object_name: str
    target_pos: np.ndarray  # (3,) target position on table surface
    grasp_height: float = 0.15  # height above table to lift
    success_threshold: float = 0.05  # metres

    def check_success(self, object_pos: np.ndarray) -> bool:
        """Check if the object is within threshold of the target."""
        return float(np.linalg.norm(object_pos[:2] - self.target_pos[:2])) < self.success_threshold


# Table surface height in world frame
TABLE_SURFACE_Z: float = 0.76

TASKS: dict[str, TaskDefinition] = {
    "pick_red_cup": TaskDefinition(
        name="pick_red_cup",
        language="pick up the red cup",
        object_name="cup",
        target_pos=np.array([0.65, 0.1, TABLE_SURFACE_Z + 0.15]),
    ),
    "pick_blue_box": TaskDefinition(
        name="pick_blue_box",
        language="pick up the blue box",
        object_name="box",
        target_pos=np.array([0.7, -0.1, TABLE_SURFACE_Z + 0.15]),
    ),
    "pick_green_bottle": TaskDefinition(
        name="pick_green_bottle",
        language="pick up the green bottle",
        object_name="bottle",
        target_pos=np.array([0.75, 0.0, TABLE_SURFACE_Z + 0.15]),
    ),
    "move_cup_left": TaskDefinition(
        name="move_cup_left",
        language="move the red cup to the left",
        object_name="cup",
        target_pos=np.array([0.65, 0.25, TABLE_SURFACE_Z + 0.06]),
        grasp_height=0.15,
    ),
    "move_box_right": TaskDefinition(
        name="move_box_right",
        language="move the blue box to the right",
        object_name="box",
        target_pos=np.array([0.65, -0.25, TABLE_SURFACE_Z + 0.06]),
        grasp_height=0.15,
    ),
}


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_mujoco_model(
    scene_xml: str = "scene_vla.xml",
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the MuJoCo model and create data.

    Args:
        scene_xml: Filename of the scene XML inside MODELS_DIR.

    Returns:
        (mj_model, mj_data) tuple.
    """
    xml_path = str(MODELS_DIR / scene_xml)
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


# ---------------------------------------------------------------------------
# Camera Rendering
# ---------------------------------------------------------------------------
def render_camera(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    camera_name: str,
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
) -> np.ndarray:
    """Render an RGB image from the named camera using offscreen rendering.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data (state must be up-to-date via mj_forward).
        camera_name: Name of the camera in the MJCF.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        RGB image as uint8 ndarray of shape (height, width, 3).
    """
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    renderer.update_scene(mj_data, camera=camera_name)
    img = renderer.render()
    renderer.close()
    return img.copy()


class CameraRenderer:
    """Persistent renderer to avoid creating/destroying on every frame."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
    ) -> None:
        self._model = mj_model
        self._renderer = mujoco.Renderer(mj_model, height=height, width=width)

    def render(self, mj_data: mujoco.MjData, camera_name: str) -> np.ndarray:
        """Render and return an RGB image from the given camera."""
        self._renderer.update_scene(mj_data, camera=camera_name)
        return self._renderer.render().copy()

    def close(self) -> None:
        """Release rendering resources."""
        self._renderer.close()


# ---------------------------------------------------------------------------
# Proprioception Extraction
# ---------------------------------------------------------------------------
def get_proprioception(mj_data: mujoco.MjData) -> np.ndarray:
    """Extract proprioception vector from MuJoCo data.

    Returns concatenation of robot joint positions and velocities.

    Args:
        mj_data: MuJoCo data with current state.

    Returns:
        1-D float64 array of shape (PROPRIO_DIM,) = (58,).
    """
    qpos = mj_data.qpos[ROBOT_QPOS_START:ROBOT_QPOS_END].copy()
    qvel = mj_data.qvel[6:6 + NUM_ROBOT_JOINTS].copy()  # root has 6 dof in vel
    return np.concatenate([qpos, qvel])


def get_object_position(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    object_name: str,
) -> np.ndarray:
    """Get the 3D world position of a named object body.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        object_name: Body name (e.g. 'cup', 'box', 'bottle').

    Returns:
        Position as (3,) float64 array.
    """
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    return mj_data.xpos[body_id].copy()


def get_ee_position(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    site_name: str = "right_ee",
) -> np.ndarray:
    """Get end-effector position from a named site.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        site_name: Site name.

    Returns:
        Position as (3,) float64 array.
    """
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return mj_data.site_xpos[site_id].copy()


def set_grasp_weld(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    object_name: str,
    active: bool,
) -> None:
    """Enable or disable the weld constraint for grasping an object.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        object_name: One of 'cup', 'box', 'bottle'.
        active: True to activate the weld (grasp), False to release.
    """
    eq_name = f"grasp_{object_name}"
    eq_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    if eq_id >= 0:
        mj_model.eq_active[eq_id] = int(active)
