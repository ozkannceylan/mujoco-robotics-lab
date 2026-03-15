"""Shared constants, model paths, and utility functions for Lab 3.

This module provides:
  - Project directory paths
  - UR5e physical constants (joint limits, torque limits)
  - Model loading helpers for MuJoCo and Pinocchio
  - Quaternion conversion utilities (MuJoCo ↔ Pinocchio)
  - EE pose/velocity helpers
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
LAB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = LAB_DIR.parent
MODELS_DIR = LAB_DIR / "models"
DOCS_DIR = LAB_DIR / "docs"
DOCS_TURKISH_DIR = LAB_DIR / "docs-turkish"
MEDIA_DIR = LAB_DIR / "media"

# ---------------------------------------------------------------------------
# Model file paths
# ---------------------------------------------------------------------------
SCENE_TORQUE_PATH = MODELS_DIR / "scene_torque.xml"
SCENE_TABLE_PATH = MODELS_DIR / "scene_table.xml"
URDF_PATH = MODELS_DIR / "ur5e.urdf"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
NUM_JOINTS = 6
DT = 0.001  # 1 kHz simulation timestep

JOINT_NAMES: Tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Joint limits (rad)
JOINT_LIMITS_LOWER = np.full(NUM_JOINTS, -2 * math.pi)
JOINT_LIMITS_UPPER = np.full(NUM_JOINTS, 2 * math.pi)

# Velocity limits (rad/s)
VELOCITY_LIMITS = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28])

# Torque limits (Nm) — from MJCF ctrlrange ±180 for all motors
TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

# ---------------------------------------------------------------------------
# Common joint configurations (rad)
# ---------------------------------------------------------------------------
Q_ZEROS = np.zeros(NUM_JOINTS)
Q_HOME = np.array([
    -math.pi / 2, -math.pi / 2, math.pi / 2,
    -math.pi / 2, -math.pi / 2, 0.0,
])


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
    """Load MuJoCo model and data from a scene XML.

    Args:
        scene_path: Path to MJCF scene file. Defaults to SCENE_TORQUE_PATH.

    Returns:
        Tuple of (mj_model, mj_data).
    """
    import mujoco

    path = scene_path or SCENE_TORQUE_PATH
    mj_model = mujoco.MjModel.from_xml_path(str(path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def load_pinocchio_model(urdf_path: Path | None = None):
    """Load Pinocchio model and data from URDF.

    Sets rotor inertia to match MuJoCo's joint armature (0.01 kg·m²)
    so that mass matrices agree between the two engines.

    Args:
        urdf_path: Path to URDF file. Defaults to URDF_PATH.

    Returns:
        Tuple of (pin_model, pin_data, ee_frame_id).
    """
    import pinocchio as pin

    path = urdf_path or URDF_PATH
    model = pin.buildModelFromUrdf(str(path))

    # Match MuJoCo armature: MuJoCo adds armature directly to M diagonal.
    # In Pinocchio, model.armature is the equivalent — added to CRBA diagonal.
    # (Note: rotorInertia does NOT affect crba(); armature does.)
    ARMATURE = 0.01
    model.armature[:] = ARMATURE

    data = model.createData()
    ee_frame_id = model.getFrameId("ee_link")
    return model, data, ee_frame_id


def get_mj_ee_site_id(mj_model) -> int:
    """Get the MuJoCo site ID for the end-effector (tool_site)."""
    import mujoco
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")


# ---------------------------------------------------------------------------
# EE pose and velocity helpers
# ---------------------------------------------------------------------------

def get_ee_pose(
    pin_model, pin_data, ee_fid: int, q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute end-effector position and rotation matrix via Pinocchio FK.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: End-effector frame ID.
        q: Joint angles (6,).

    Returns:
        Tuple of (position [3], rotation_matrix [3x3]).
    """
    import pinocchio as pin

    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    oMf = pin_data.oMf[ee_fid]
    return oMf.translation.copy(), oMf.rotation.copy()


def get_ee_velocity(
    pin_model, pin_data, ee_fid: int, q: np.ndarray, qd: np.ndarray
) -> np.ndarray:
    """Compute 6D end-effector velocity (linear, angular) in world frame.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: End-effector frame ID.
        q: Joint angles (6,).
        qd: Joint velocities (6,).

    Returns:
        6D velocity vector [vx, vy, vz, wx, wy, wz].
    """
    import pinocchio as pin

    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    pin.computeJointJacobians(pin_model, pin_data, q)
    J = pin.getFrameJacobian(
        pin_model, pin_data, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    return J @ qd


def clip_torques(tau: np.ndarray) -> np.ndarray:
    """Clip joint torques to actuator limits.

    Args:
        tau: Joint torques (6,).

    Returns:
        Clipped torques within ±TORQUE_LIMITS.
    """
    return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
