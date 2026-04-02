"""Lab 6 — Shared constants, paths, and utilities for dual-arm coordination."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

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
# Left arm: standard UR5e home (EE pointing forward/down toward table)
Q_HOME_LEFT = np.array([
    -math.pi / 2, -math.pi / 2, math.pi / 2,
    -math.pi / 2, -math.pi / 2, 0.0,
])

# Right arm: identical joint angles — both bases have the same orientation
# (Menagerie quat only, no yaw mount). Arms face the same direction at home.
# Facing each other for grasping is handled by IK targets in M3.
Q_HOME_RIGHT = Q_HOME_LEFT.copy()

# Table parameters
TABLE_POS = np.array([0.5, 0.0, 0.15])
TABLE_HALF_EXTENTS = np.array([0.20, 0.30, 0.02])
TABLE_SURFACE_Z = TABLE_POS[2] + TABLE_HALF_EXTENTS[2]  # 0.17

# Box object parameters
BOX_HALF_EXTENTS = np.array([0.15, 0.075, 0.075])
BOX_INIT_POS = np.array([0.5, 0.0, TABLE_SURFACE_Z + BOX_HALF_EXTENTS[2]])


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


def clip_torques(tau: np.ndarray) -> np.ndarray:
    """Clip joint torques to actuator limits.

    Args:
        tau: Joint torques (6,).

    Returns:
        Clipped torques within +/-TORQUE_LIMITS.
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
