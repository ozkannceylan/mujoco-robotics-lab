"""Lab 6 — Shared constants, paths, and utilities for dual-arm coordination."""

from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pinocchio as pin

# Paths
LAB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = LAB_DIR.parent
MODELS_DIR = LAB_DIR / "models"
MEDIA_DIR = LAB_DIR / "media"
DOCS_DIR = LAB_DIR / "docs"
DOCS_TURKISH_DIR = LAB_DIR / "docs-turkish"

SCENE_DUAL_PATH = MODELS_DIR / "scene_dual.xml"
URDF_PATH = MODELS_DIR / "ur5e.urdf"

# Constants
NUM_JOINTS_PER_ARM = 6
NUM_JOINTS_TOTAL = 12
DT = 0.001
GRAVITY = np.array([0.0, 0.0, -9.81])

# Index slicing for MuJoCo arrays
LEFT_JOINT_SLICE = slice(0, 6)
RIGHT_JOINT_SLICE = slice(6, 12)
LEFT_CTRL_SLICE = slice(0, 6)
RIGHT_CTRL_SLICE = slice(6, 12)

# Joint limits (per arm)
JOINT_LIMITS_LOWER = np.full(NUM_JOINTS_PER_ARM, -2 * math.pi)
JOINT_LIMITS_UPPER = np.full(NUM_JOINTS_PER_ARM, 2 * math.pi)

# Torque limits (per arm)
TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

# Velocity limits (per arm)
VEL_LIMITS = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28])

# Home configurations
# Left arm: standard UR5e home (pointing forward/up)
Q_HOME_LEFT = np.array([
    -math.pi / 2, -math.pi / 2, math.pi / 2,
    -math.pi / 2, -math.pi / 2, 0.0,
])

# Right arm: mirrored home (accounting for 180° base rotation)
# The right arm is rotated 180° about Z, so its shoulder_pan is offset by pi
Q_HOME_RIGHT = np.array([
    math.pi / 2, -math.pi / 2, math.pi / 2,
    -math.pi / 2, math.pi / 2, 0.0,
])

# Base transforms for each arm in world frame
# Left arm: at origin, no rotation
LEFT_BASE_POS = np.array([0.0, 0.0, 0.0])
LEFT_BASE_SE3 = pin.SE3(np.eye(3), LEFT_BASE_POS)

# Right arm: at (1.0, 0, 0), rotated 180° about Z
RIGHT_BASE_POS = np.array([1.0, 0.0, 0.0])
_R_right = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
])  # 180° rotation about Z
RIGHT_BASE_SE3 = pin.SE3(_R_right, RIGHT_BASE_POS)

# Table parameters
TABLE_POS = np.array([0.5, 0.0, 0.15])
TABLE_HALF_EXTENTS = np.array([0.30, 0.40, 0.02])
TABLE_SURFACE_Z = TABLE_POS[2] + TABLE_HALF_EXTENTS[2]  # 0.17

# Box object parameters
BOX_HALF_EXTENTS = np.array([0.15, 0.075, 0.075])
BOX_INIT_POS = np.array([0.5, 0.0, TABLE_SURFACE_Z + BOX_HALF_EXTENTS[2]])


@dataclass
class ObstacleSpec:
    """Obstacle box specification for collision checking."""
    name: str
    position: np.ndarray
    half_extents: np.ndarray


TABLE_SPEC = ObstacleSpec("table", TABLE_POS, TABLE_HALF_EXTENTS)


def mj_quat_to_pin(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion (w,x,y,z) to Pinocchio quaternion (x,y,z,w)."""
    w, x, y, z = quat_wxyz
    return np.array([x, y, z, w])


def pin_quat_to_mj(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert Pinocchio quaternion (x,y,z,w) to MuJoCo quaternion (w,x,y,z)."""
    x, y, z, w = quat_xyzw
    return np.array([w, x, y, z])


def load_mujoco_model(scene_path: Path | None = None):
    """Load MuJoCo model and data.

    Args:
        scene_path: Path to MJCF scene. Defaults to SCENE_DUAL_PATH.

    Returns:
        Tuple of (MjModel, MjData).
    """
    import mujoco
    path = scene_path or SCENE_DUAL_PATH
    mj_model = mujoco.MjModel.from_xml_path(str(path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def load_pinocchio_model(urdf_path: Path | None = None):
    """Load a single Pinocchio model with standard UR5e settings.

    Sets armature to 0.01 to match MuJoCo.

    Args:
        urdf_path: Path to URDF. Defaults to URDF_PATH.

    Returns:
        Tuple of (model, data, ee_frame_id).
    """
    path = urdf_path or URDF_PATH
    model = pin.buildModelFromUrdf(str(path))
    model.armature[:] = 0.01
    data = model.createData()
    ee_fid = model.getFrameId("ee_link")
    return model, data, ee_fid


def get_ee_pose(pin_model, pin_data, ee_fid: int, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute EE position and rotation via Pinocchio FK.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
        q: Joint angles (6,).

    Returns:
        Tuple of (position [3], rotation_matrix [3x3]).
    """
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    oMf = pin_data.oMf[ee_fid]
    return oMf.translation.copy(), oMf.rotation.copy()


def clip_torques(tau: np.ndarray) -> np.ndarray:
    """Clip joint torques to actuator limits.

    Args:
        tau: Joint torques (6,).

    Returns:
        Clipped torques within ±TORQUE_LIMITS.
    """
    return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)


def get_mj_body_id(mj_model, name: str) -> int:
    """Get MuJoCo body ID by name."""
    import mujoco
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)


def get_mj_site_id(mj_model, name: str) -> int:
    """Get MuJoCo site ID by name."""
    import mujoco
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
