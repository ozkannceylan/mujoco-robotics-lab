"""Shared constants, model paths, and utility functions for the UR5e lab.

This module provides:
  - Project directory paths
  - UR5e physical constants (DH parameters, joint limits, link masses)
  - Model loading helpers for MuJoCo and Pinocchio
  - Quaternion conversion utilities (MuJoCo ↔ Pinocchio)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
LAB_NAME = "lab-2-Ur5e-robotics-lab"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LAB_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(__file__).resolve().parent
MODELS_DIR = _LAB_DIR / "models"
DOCS_DIR = _LAB_DIR / "docs"
DOCS_TURKISH_DIR = _LAB_DIR / "docs-turkish"
MEDIA_DIR = _LAB_DIR / "media"

# ---------------------------------------------------------------------------
# Model file paths
# ---------------------------------------------------------------------------
MENAGERIE_DIR = MODELS_DIR / "mujoco_menagerie" / "universal_robots_ur5e"
MJCF_SCENE_PATH = MENAGERIE_DIR / "lab_scene.xml"
MJCF_ROBOT_PATH = MENAGERIE_DIR / "ur5e.xml"
URDF_PATH = MODELS_DIR / "ur5e.urdf"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
NUM_JOINTS = 6

JOINT_NAMES: Tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Joint limits (rad) — UR5e spec: ±2π for all joints except elbow (±π)
JOINT_LIMITS_LOWER = np.array([-2 * math.pi, -2 * math.pi, -math.pi,
                                -2 * math.pi, -2 * math.pi, -2 * math.pi])
JOINT_LIMITS_UPPER = np.array([2 * math.pi, 2 * math.pi, math.pi,
                                2 * math.pi, 2 * math.pi, 2 * math.pi])

# Velocity limits (rad/s)
VELOCITY_LIMITS = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28])

# Torque limits (Nm)
TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

# ---------------------------------------------------------------------------
# DH Parameters (Modified DH, from official UR5e documentation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DHRow:
    """One row of the Modified DH parameter table."""
    alpha: float  # twist angle (rad)
    a: float      # link length (m)
    d: float      # link offset (m)


# Standard UR5e DH parameters
DH_PARAMS: Tuple[DHRow, ...] = (
    DHRow(alpha=0.0,          a=0.0,     d=0.1625),
    DHRow(alpha=math.pi / 2,  a=0.0,     d=0.0),
    DHRow(alpha=0.0,          a=-0.425,  d=0.0),
    DHRow(alpha=0.0,          a=-0.3922, d=0.1333),
    DHRow(alpha=math.pi / 2,  a=0.0,     d=0.0997),
    DHRow(alpha=-math.pi / 2, a=0.0,     d=0.0996),
)

# ---------------------------------------------------------------------------
# Common joint configurations (rad)
# ---------------------------------------------------------------------------
Q_ZEROS = np.zeros(NUM_JOINTS)
Q_HOME = np.array([-math.pi / 2, -math.pi / 2, math.pi / 2,
                    -math.pi / 2, -math.pi / 2, 0.0])

# ---------------------------------------------------------------------------
# Link masses (kg) — from MuJoCo Menagerie UR5e
# ---------------------------------------------------------------------------
LINK_MASSES = np.array([4.0, 3.7, 8.393, 2.275, 1.219, 1.219, 0.1889])

# ---------------------------------------------------------------------------
# Quaternion conversion utilities
# ---------------------------------------------------------------------------


def mj_quat_to_pin(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion (w,x,y,z) to Pinocchio quaternion (x,y,z,w)."""
    w, x, y, z = quat_wxyz
    return np.array([x, y, z, w])


def pin_quat_to_mj(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert Pinocchio quaternion (x,y,z,w) to MuJoCo quaternion (w,x,y,z)."""
    x, y, z, w = quat_xyzw
    return np.array([w, x, y, z])


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def load_mujoco_model(scene_path: Path | None = None):
    """Load MuJoCo model and data from the lab scene XML.

    Args:
        scene_path: Path to MJCF scene file. Defaults to MJCF_SCENE_PATH.

    Returns:
        Tuple of (mj_model, mj_data).
    """
    import mujoco

    path = scene_path or MJCF_SCENE_PATH
    mj_model = mujoco.MjModel.from_xml_path(str(path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def load_pinocchio_model(urdf_path: Path | None = None):
    """Load Pinocchio model and data from URDF.

    Args:
        urdf_path: Path to URDF file. Defaults to URDF_PATH.

    Returns:
        Tuple of (pin_model, pin_data, ee_frame_id).
    """
    import pinocchio as pin

    path = urdf_path or URDF_PATH
    model = pin.buildModelFromUrdf(str(path))
    data = model.createData()
    ee_frame_id = model.getFrameId("ee_link")
    return model, data, ee_frame_id


def get_mj_ee_site_id(mj_model) -> int:
    """Get the MuJoCo site ID for the end-effector (attachment_site)."""
    import mujoco
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
