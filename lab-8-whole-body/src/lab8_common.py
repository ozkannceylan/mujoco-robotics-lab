"""Shared constants, model paths, and utility functions for Lab 8.

This module provides:
  - Project directory paths
  - G1 humanoid physical constants (joint groups, limits)
  - Model loading helpers for MuJoCo and Pinocchio
  - Quaternion conversion utilities (MuJoCo <-> Pinocchio)
  - State extraction from MuJoCo (freeflyer quaternion reordering)
  - Actuator Kp lookup for feedforward control
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

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
SCENE_PATH = MODELS_DIR / "scene_loco_manip.xml"
G1_MJCF_PATH = MODELS_DIR / "g1_full.xml"
URDF_PATH = MODELS_DIR / "g1_full.urdf"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
DT = 0.002  # 2 ms simulation timestep (500 Hz)
NUM_ACTUATED = 25  # total actuated DOFs
NQ_FREEFLYER = 7  # pos(3) + quat(4)
NV_FREEFLYER = 6  # lin_vel(3) + ang_vel(3)
NQ = NQ_FREEFLYER + NUM_ACTUATED  # 32
NV = NV_FREEFLYER + NUM_ACTUATED  # 31

# ---------------------------------------------------------------------------
# Joint group indices (0-indexed into the 25 actuated joints)
# ---------------------------------------------------------------------------
WAIST_IDX = np.array([0])

LEFT_LEG_IDX = np.array([1, 2, 3, 4, 5, 6])
RIGHT_LEG_IDX = np.array([7, 8, 9, 10, 11, 12])
LEGS_IDX = np.concatenate([LEFT_LEG_IDX, RIGHT_LEG_IDX])

LEFT_ARM_IDX = np.array([13, 14, 15, 16, 17, 18])
RIGHT_ARM_IDX = np.array([19, 20, 21, 22, 23, 24])
ARMS_IDX = np.concatenate([LEFT_ARM_IDX, RIGHT_ARM_IDX])

# Joint names in actuator order
JOINT_NAMES: Tuple[str, ...] = (
    "waist_yaw",
    # Left leg
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    # Right leg
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    # Left arm
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch",
    # Right arm
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch",
)

# Frame names for key end-effectors
LEFT_FOOT_FRAME = "left_ankle_roll_link"
RIGHT_FOOT_FRAME = "right_ankle_roll_link"
LEFT_HAND_FRAME = "left_hand"
RIGHT_HAND_FRAME = "right_hand"

# ---------------------------------------------------------------------------
# Standing configuration (slight knee bend, arms at sides)
# ---------------------------------------------------------------------------
Q_STAND = np.zeros(NUM_ACTUATED)
# Waist: 0
# Left leg: hip_yaw=0, hip_roll=0, hip_pitch=-0.2, knee=0.4, ankle_pitch=-0.2, ankle_roll=0
Q_STAND[3] = -0.2    # left_hip_pitch
Q_STAND[4] = 0.4     # left_knee
Q_STAND[5] = -0.2    # left_ankle_pitch
# Right leg (same as left)
Q_STAND[9] = -0.2    # right_hip_pitch
Q_STAND[10] = 0.4    # right_knee
Q_STAND[11] = -0.2   # right_ankle_pitch
# Left arm: shoulder_pitch=0.3, shoulder_roll=0, shoulder_yaw=0, elbow=-0.5
Q_STAND[13] = 0.3    # left_shoulder_pitch
Q_STAND[16] = -0.5   # left_elbow
# Right arm (same as left)
Q_STAND[19] = 0.3    # right_shoulder_pitch
Q_STAND[22] = -0.5   # right_elbow

# Torso nominal height when standing
TORSO_HEIGHT = 0.78


# ---------------------------------------------------------------------------
# Quaternion conversion utilities
# ---------------------------------------------------------------------------

def mj_quat_to_pin(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion (w,x,y,z) to Pinocchio quaternion (x,y,z,w).

    Args:
        quat_wxyz: Quaternion in MuJoCo convention [w, x, y, z].

    Returns:
        Quaternion in Pinocchio convention [x, y, z, w].
    """
    w, x, y, z = quat_wxyz
    return np.array([x, y, z, w])


def pin_quat_to_mj(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert Pinocchio quaternion (x,y,z,w) to MuJoCo quaternion (w,x,y,z).

    Args:
        quat_xyzw: Quaternion in Pinocchio convention [x, y, z, w].

    Returns:
        Quaternion in MuJoCo convention [w, x, y, z].
    """
    x, y, z, w = quat_xyzw
    return np.array([w, x, y, z])


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_mujoco_model(scene_path: Path | None = None):
    """Load MuJoCo model and data from a scene XML.

    Args:
        scene_path: Path to MJCF scene file. Defaults to SCENE_PATH.

    Returns:
        Tuple of (mj_model, mj_data).
    """
    import mujoco

    path = scene_path or SCENE_PATH
    mj_model = mujoco.MjModel.from_xml_path(str(path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def load_mujoco_model_standalone(mjcf_path: Path | None = None):
    """Load the standalone G1 model (no scene).

    Args:
        mjcf_path: Path to MJCF file. Defaults to G1_MJCF_PATH.

    Returns:
        Tuple of (mj_model, mj_data).
    """
    import mujoco

    path = mjcf_path or G1_MJCF_PATH
    mj_model = mujoco.MjModel.from_xml_path(str(path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def load_pinocchio_model(urdf_path: Path | None = None):
    """Load Pinocchio model with freeflyer root joint.

    Args:
        urdf_path: Path to URDF file. Defaults to URDF_PATH.

    Returns:
        Tuple of (pin_model, pin_data).
    """
    import pinocchio as pin

    path = urdf_path or URDF_PATH
    model = pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())

    # Match MuJoCo armature
    ARMATURE = 0.01
    model.armature[:] = ARMATURE

    data = model.createData()
    return model, data


# ---------------------------------------------------------------------------
# State conversion: MuJoCo -> Pinocchio
# ---------------------------------------------------------------------------

def get_state_from_mujoco(mj_model, mj_data) -> Tuple[np.ndarray, np.ndarray]:
    """Extract generalized position and velocity from MuJoCo data.

    Handles freeflyer quaternion reordering:
      MuJoCo qpos: [pos(3), quat_wxyz(4), joints(25)]
      Pinocchio q:  [pos(3), quat_xyzw(4), joints(25)]

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.

    Returns:
        Tuple of (q_pin [32], v_pin [31]).
    """
    # Position: reorder quaternion
    q_pin = np.zeros(NQ)
    q_pin[:3] = mj_data.qpos[:3]  # position
    q_pin[3:7] = mj_quat_to_pin(mj_data.qpos[3:7])  # quaternion
    q_pin[7:] = mj_data.qpos[7:7 + NUM_ACTUATED]  # joint angles

    # Velocity: MuJoCo and Pinocchio use the same convention for freeflyer
    # velocity (linear, angular in world frame + joint velocities)
    v_pin = np.zeros(NV)
    v_pin[:6] = mj_data.qvel[:6]  # freeflyer velocity
    v_pin[6:] = mj_data.qvel[6:6 + NUM_ACTUATED]  # joint velocities

    return q_pin, v_pin


def set_mujoco_state(mj_model, mj_data, q_pin: np.ndarray, v_pin: np.ndarray) -> None:
    """Set MuJoCo state from Pinocchio generalized coordinates.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        q_pin: Pinocchio generalized positions [32].
        v_pin: Pinocchio generalized velocities [31].
    """
    mj_data.qpos[:3] = q_pin[:3]
    mj_data.qpos[3:7] = pin_quat_to_mj(q_pin[3:7])
    mj_data.qpos[7:7 + NUM_ACTUATED] = q_pin[7:]

    mj_data.qvel[:6] = v_pin[:6]
    mj_data.qvel[6:6 + NUM_ACTUATED] = v_pin[6:]


# ---------------------------------------------------------------------------
# Actuator Kp lookup
# ---------------------------------------------------------------------------

# Kp values per actuated joint (from MJCF)
ACTUATOR_KP = np.array([
    200.0,   # waist_yaw
    # Left leg
    300.0, 300.0, 400.0, 400.0, 200.0, 150.0,
    # Right leg
    300.0, 300.0, 400.0, 400.0, 200.0, 150.0,
    # Left arm
    100.0, 100.0, 80.0, 80.0, 40.0, 40.0,
    # Right arm
    100.0, 100.0, 80.0, 80.0, 40.0, 40.0,
])

ACTUATOR_KV = np.array([
    15.0,    # waist_yaw
    # Left leg
    20.0, 20.0, 25.0, 25.0, 15.0, 10.0,
    # Right leg
    20.0, 20.0, 25.0, 25.0, 15.0, 10.0,
    # Left arm
    8.0, 8.0, 6.0, 6.0, 4.0, 4.0,
    # Right arm
    8.0, 8.0, 6.0, 6.0, 4.0, 4.0,
])


def compute_feedforward_ctrl(
    q_des: np.ndarray,
    qd_des: np.ndarray,
    qfrc_bias: np.ndarray,
) -> np.ndarray:
    """Compute position actuator control with gravity + velocity feedforward.

    For MuJoCo position actuators: tau = Kp*(ctrl - qpos) - Kv*qvel
    Feedforward: ctrl = q_des + qfrc_bias/Kp + Kv*qd_des/Kp

    Args:
        q_des: Desired joint positions [25].
        qd_des: Desired joint velocities [25].
        qfrc_bias: Gravity + Coriolis bias forces for actuated joints [25].

    Returns:
        Control signal [25].
    """
    ctrl = q_des + qfrc_bias / ACTUATOR_KP + ACTUATOR_KV * qd_des / ACTUATOR_KP
    return ctrl


def get_actuated_qfrc_bias(mj_data) -> np.ndarray:
    """Extract bias forces for actuated joints from MuJoCo.

    Args:
        mj_data: MuJoCo data (after mj_step or mj_forward).

    Returns:
        Bias forces for actuated joints [25].
    """
    return mj_data.qfrc_bias[6:6 + NUM_ACTUATED].copy()


def build_initial_qpos(q_act: np.ndarray | None = None) -> np.ndarray:
    """Build full MuJoCo qpos from actuated joint config.

    Args:
        q_act: Actuated joint angles [25]. Defaults to Q_STAND.

    Returns:
        Full qpos for MuJoCo [32+] (freeflyer + actuated + any freejoint objects).
    """
    if q_act is None:
        q_act = Q_STAND.copy()
    # Freeflyer: pos=[0,0,TORSO_HEIGHT], quat=[1,0,0,0] (identity in wxyz)
    qpos_base = np.array([0.0, 0.0, TORSO_HEIGHT, 1.0, 0.0, 0.0, 0.0])
    return np.concatenate([qpos_base, q_act])
