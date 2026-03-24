"""Lab 6 — Common constants, paths, and model loading utilities.

Dual-arm scene layout:
  Left arm base:  (0, -0.35, 0)  — grips bar -Y end at ~(0.45, -0.10, bar_z)
  Right arm base: (0, +0.35, 0)  — grips bar +Y end at ~(0.45, +0.10, bar_z)
  Carry bar centre: (0.45, 0, 0.335), half_extents (0.025, 0.12, 0.020)

MuJoCo model indices (verified against dual_arm_scene.xml):
  qpos[0:6]   — left arm joints
  qpos[6]     — left_left_finger_joint
  qpos[7]     — left_right_finger_joint  (equality-mirrored)
  qpos[8:14]  — right arm joints
  qpos[14]    — right_left_finger_joint
  qpos[15]    — right_right_finger_joint (equality-mirrored)
  qpos[16:23] — carry_bar freejoint (3 pos + 4 quat)

  ctrl[0:6]  — left arm torque motors
  ctrl[6]    — left gripper (position actuator)
  ctrl[7:13] — right arm torque motors
  ctrl[13]   — right gripper (position actuator)
"""

from __future__ import annotations

import math
import sys
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
SCENE_PATH = MODELS_DIR / "dual_arm_scene.xml"

# Pinocchio: use arm-only URDF from Lab 3 (no gripper needed for FK/IK)
_LAB3_DIR = _LAB_DIR.parent / "lab-3-dynamics-force-control"
URDF_PATH = _LAB3_DIR / "models" / "ur5e.urdf"

# ---------------------------------------------------------------------------
# qpos / ctrl index layout
# ---------------------------------------------------------------------------

LEFT_QPOS   = slice(0, 6)      # left arm joint positions in qpos
LEFT_CTRL   = slice(0, 6)      # left arm motor ctrls
LEFT_GRIP_CTRL = 6             # ctrl index for left gripper position actuator

RIGHT_QPOS  = slice(8, 14)     # right arm joint positions in qpos
RIGHT_CTRL  = slice(7, 13)     # right arm motor ctrls
RIGHT_GRIP_CTRL = 13           # ctrl index for right gripper position actuator

# Carry bar freejoint: qpos[16:19] = position, qpos[19:23] = quaternion (w,x,y,z)
BAR_QPOS_POS  = slice(16, 19)
BAR_QPOS_QUAT = slice(19, 23)

NUM_JOINTS = 6          # UR5e arm DOF (per arm)
DT = 0.001              # simulation timestep (1 kHz)

# ---------------------------------------------------------------------------
# Gripper setpoints
# ---------------------------------------------------------------------------

GRIPPER_OPEN   = 0.030   # metres — fingers spread (60 mm total gap)
GRIPPER_CLOSED = 0.000   # metres — fingers closed

# Distance from tool0 origin to fingertip pad centre (along tool0 local Z)
GRIPPER_TIP_OFFSET = 0.105   # metres  (same gripper geometry as Lab 5)

# ---------------------------------------------------------------------------
# Joint / dynamic limits (same for both arms — identical UR5e)
# ---------------------------------------------------------------------------

TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
VEL_LIMITS    = np.array([3.14,  3.14,  3.14,  6.28, 6.28, 6.28])
ACC_LIMITS    = np.array([8.0,   8.0,   8.0,  16.0, 16.0, 16.0])
JOINT_LOWER   = np.full(NUM_JOINTS, -2 * math.pi)
JOINT_UPPER   = np.full(NUM_JOINTS,  2 * math.pi)

# ---------------------------------------------------------------------------
# Home configurations
# ---------------------------------------------------------------------------

# Left arm: same as Lab 5 Q_HOME (reaches toward +X, +Y relative to base)
Q_HOME_LEFT = np.array([
    -math.pi / 2, -math.pi / 2,  math.pi / 2,
    -math.pi / 2, -math.pi / 2,  0.0,
])

# Right arm: mirror shoulder_pan for symmetric reach toward +X, -Y relative to base
Q_HOME_RIGHT = np.array([
     math.pi / 2, -math.pi / 2,  math.pi / 2,
    -math.pi / 2, -math.pi / 2,  0.0,
])

# ---------------------------------------------------------------------------
# Arm base positions in world frame (must match dual_arm_scene.xml)
# ---------------------------------------------------------------------------

# IMPORTANT: Pinocchio computes FK with the arm base at the LOCAL origin.
# When the MuJoCo arm base is offset in world, you must subtract the arm base
# position from any WORLD-FRAME IK target before passing it to Pinocchio:
#   arm_local_target = world_target - ARM_BASE_POS_xxx
LEFT_ARM_BASE_POS  = np.array([0.0, -0.35, 0.0])
RIGHT_ARM_BASE_POS = np.array([0.0,  0.35, 0.0])

# ---------------------------------------------------------------------------
# Scene geometry (must match dual_arm_scene.xml)
# ---------------------------------------------------------------------------

TABLE_TOP_Z = 0.315          # world Z of table surface (centre 0.300, half 0.015)

# Carry bar — pick position
BAR_HALF_X = 0.025           # bar half-width in X (50 mm)  — bar is Y-oriented
BAR_HALF_Y = 0.12            # bar half-length in Y (240 mm)
BAR_HALF_Z = 0.020           # bar half-height in Z (40 mm)
BAR_PICK_POS = np.array([0.45, 0.0, TABLE_TOP_Z + BAR_HALF_Z])   # (0.45, 0, 0.335)

# Carry bar — place position (moved +0.20 m in Y)
BAR_PLACE_POS = np.array([0.45, 0.20, TABLE_TOP_Z + BAR_HALF_Z])  # (0.45, 0.20, 0.335)

# Grip Y-offsets from bar centre (each arm grabs one Y-end from its own side)
LEFT_GRIP_Y_OFFSET  = -0.10   # left arm grips at bar_y - 0.10 (bar -Y end)
RIGHT_GRIP_Y_OFFSET =  0.10   # right arm grips at bar_y + 0.10 (bar +Y end)

# Pre-grasp clearance (metres above grasp height)
PREGRASP_CLEARANCE = 0.15


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_mujoco_model(scene_path: Path | None = None) -> tuple:
    """Load MuJoCo model and data for the Lab 6 dual-arm scene.

    Args:
        scene_path: Path to MJCF. Defaults to SCENE_PATH.

    Returns:
        Tuple of (MjModel, MjData).
    """
    path = scene_path or SCENE_PATH
    mj_model = mujoco.MjModel.from_xml_path(str(path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def load_pinocchio_model(urdf_path: Path | None = None) -> tuple:
    """Load a single Pinocchio UR5e arm model.

    Both arms reuse the same URDF (they are kinematically identical).
    Call twice to get independent (model, data) pairs.

    Args:
        urdf_path: Path to arm URDF. Defaults to Lab 3 ur5e.urdf.

    Returns:
        Tuple of (pin_model, pin_data, ee_frame_id).
    """
    path = urdf_path or URDF_PATH
    model = pin.buildModelFromUrdf(str(path))
    model.armature[:] = 0.01   # match MuJoCo joint armature
    data = model.createData()
    ee_fid = model.getFrameId("ee_link")
    return model, data, ee_fid


def load_both_pinocchio_models(urdf_path: Path | None = None) -> tuple:
    """Load two independent Pinocchio models for left and right arms.

    Args:
        urdf_path: Path to arm URDF.

    Returns:
        Tuple of ((pin_L, data_L, ee_fid_L), (pin_R, data_R, ee_fid_R)).
    """
    left  = load_pinocchio_model(urdf_path)
    right = load_pinocchio_model(urdf_path)
    return left, right


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def clip_torques(tau: np.ndarray) -> np.ndarray:
    """Clip arm torques to joint limits.

    Args:
        tau: Joint torques (6,).

    Returns:
        Clipped torques within ±TORQUE_LIMITS.
    """
    return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)


def get_ee_pose(pin_model, pin_data, ee_fid: int, q: np.ndarray) -> tuple:
    """Compute EE position and rotation via Pinocchio FK.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
        q: Joint angles (6,).

    Returns:
        Tuple of (position [3], rotation [3×3]).
    """
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    oMf = pin_data.oMf[ee_fid]
    return oMf.translation.copy(), oMf.rotation.copy()


def get_topdown_rotation(pin_model, pin_data, ee_fid: int, q_home: np.ndarray) -> np.ndarray:
    """Return the EE rotation matrix at Q_HOME (top-down approach orientation).

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
        q_home: Home configuration for this arm.

    Returns:
        3×3 rotation matrix R_topdown.
    """
    pin.forwardKinematics(pin_model, pin_data, q_home)
    pin.updateFramePlacements(pin_model, pin_data)
    return pin_data.oMf[ee_fid].rotation.copy()


def get_mj_body_id(mj_model, name: str) -> int:
    """Return MuJoCo body ID by name."""
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)


def get_mj_site_id(mj_model, name: str) -> int:
    """Return MuJoCo site ID by name."""
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)


def add_lab_src_to_path(lab_name: str) -> None:
    """Add another lab's src/ directory to sys.path for cross-lab imports.

    Args:
        lab_name: Directory name e.g. 'lab-3-dynamics-force-control'.
    """
    src = _LAB_DIR.parent / lab_name / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
