"""Lab 4 — Common constants, paths, and model loading utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_LAB_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = _LAB_DIR / "models"
MEDIA_DIR = _LAB_DIR / "media"
URDF_PATH = MODELS_DIR / "ur5e_collision.urdf"
SCENE_PATH = MODELS_DIR / "scene_obstacles.xml"
UR5E_XML_PATH = MODELS_DIR / "ur5e.xml"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_JOINTS = 6
DT = 0.001

TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
VEL_LIMITS = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28])
ACC_LIMITS = np.array([8.0, 8.0, 8.0, 16.0, 16.0, 16.0])

JOINT_LOWER = np.full(NUM_JOINTS, -2 * math.pi)
JOINT_UPPER = np.full(NUM_JOINTS, 2 * math.pi)

Q_HOME = np.array([
    -math.pi / 2, -math.pi / 2, math.pi / 2,
    -math.pi / 2, -math.pi / 2, 0.0,
])


# ---------------------------------------------------------------------------
# Obstacle specifications (must match scene_obstacles.xml)
# ---------------------------------------------------------------------------

@dataclass
class ObstacleSpec:
    """Obstacle box specification for collision checking."""
    name: str
    position: np.ndarray   # (3,) world position
    half_extents: np.ndarray  # (3,) box half-sizes


OBSTACLES = [
    ObstacleSpec("obs1", np.array([0.30, 0.15, 0.415]), np.array([0.06, 0.06, 0.10])),
    ObstacleSpec("obs2", np.array([0.30, -0.15, 0.415]), np.array([0.06, 0.06, 0.10])),
    ObstacleSpec("obs3", np.array([0.45, 0.0, 0.375]), np.array([0.04, 0.12, 0.06])),
    ObstacleSpec("obs4", np.array([0.55, 0.12, 0.395]), np.array([0.05, 0.05, 0.08])),
]

# Table as collision object
TABLE_SPEC = ObstacleSpec("table", np.array([0.45, 0.0, 0.3]), np.array([0.30, 0.45, 0.015]))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_mujoco_model(scene_path: Path | None = None):
    """Load MuJoCo model and data.

    Args:
        scene_path: Path to MJCF scene. Defaults to SCENE_PATH.

    Returns:
        Tuple of (MjModel, MjData).
    """
    if scene_path is None:
        scene_path = SCENE_PATH
    mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def load_pinocchio_model(urdf_path: Path | None = None):
    """Load Pinocchio model, data, and EE frame ID.

    Args:
        urdf_path: Path to URDF. Defaults to URDF_PATH.

    Returns:
        Tuple of (model, data, ee_frame_id).
    """
    if urdf_path is None:
        urdf_path = URDF_PATH
    model = pin.buildModelFromUrdf(str(urdf_path))
    model.armature[:] = 0.01
    data = model.createData()
    ee_fid = model.getFrameId("ee_link")
    return model, data, ee_fid


def get_ee_pos(pin_model, pin_data, ee_fid: int, q: np.ndarray) -> np.ndarray:
    """Compute EE position for configuration q.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
        q: Joint angles (6,).

    Returns:
        EE position (3,).
    """
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    return pin_data.oMf[ee_fid].translation.copy()


def clip_torques(tau: np.ndarray) -> np.ndarray:
    """Clip torques to joint limits.

    Args:
        tau: Joint torques (6,).

    Returns:
        Clipped torques (6,).
    """
    return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
