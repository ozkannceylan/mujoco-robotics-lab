"""Shared constants, model paths, and utility functions for Lab 7.

This module provides:
  - Project directory paths
  - G1 humanoid physical constants (joint names, Kp values, home config)
  - Model loading helpers for MuJoCo and Pinocchio
  - Quaternion conversion utilities (MuJoCo <-> Pinocchio)
  - State conversion between MuJoCo and Pinocchio representations
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

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
SRC_DIR = LAB_DIR / "src"

# ---------------------------------------------------------------------------
# Model file paths
# ---------------------------------------------------------------------------
SCENE_FLAT_PATH = MODELS_DIR / "scene_flat.xml"
URDF_PATH = MODELS_DIR / "g1_humanoid.urdf"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
GRAVITY_MAGNITUDE = 9.81
DT = 0.002  # 2 ms simulation timestep (matches MJCF)
NUM_ACTUATED = 13

# Joint names in MJCF actuator order
JOINT_NAMES: Tuple[str, ...] = (
    "waist_yaw",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
)

# Actuator Kp values (from MJCF gainprm)
ACTUATOR_KP = np.array([
    600.0,   # waist_yaw
    600.0,   # left_hip_yaw
    600.0,   # left_hip_roll
    900.0,   # left_hip_pitch
    900.0,   # left_knee_pitch
    450.0,   # left_ankle_pitch
    240.0,   # left_ankle_roll
    600.0,   # right_hip_yaw
    600.0,   # right_hip_roll
    900.0,   # right_hip_pitch
    900.0,   # right_knee_pitch
    450.0,   # right_ankle_pitch
    240.0,   # right_ankle_roll
])

# Actuator Kd values (from MJCF biasprm third element, negated)
ACTUATOR_KD = np.array([
    30.0,    # waist_yaw
    30.0,    # left_hip_yaw
    30.0,    # left_hip_roll
    45.0,    # left_hip_pitch
    45.0,    # left_knee_pitch
    22.0,    # left_ankle_pitch
    12.0,    # left_ankle_roll
    30.0,    # right_hip_yaw
    30.0,    # right_hip_roll
    45.0,    # right_hip_pitch
    45.0,    # right_knee_pitch
    22.0,    # right_ankle_pitch
    12.0,    # right_ankle_roll
])

# ---------------------------------------------------------------------------
# Common joint configurations (rad) - actuated joints only
# ---------------------------------------------------------------------------
# Home position: slight knee bend for stability
# hip_pitch negative = lean forward slightly, knee_pitch positive = bend
Q_HOME = np.array([
    0.0,     # waist_yaw
    0.0,     # left_hip_yaw
    0.0,     # left_hip_roll
    -0.1,    # left_hip_pitch (slight forward lean)
    0.2,     # left_knee_pitch (slight bend)
    -0.1,    # left_ankle_pitch (compensate)
    0.0,     # left_ankle_roll
    0.0,     # right_hip_yaw
    0.0,     # right_hip_roll
    -0.1,    # right_hip_pitch
    0.2,     # right_knee_pitch
    -0.1,    # right_ankle_pitch
    0.0,     # right_ankle_roll
])

# Freeflyer configuration dimensions
NQ_FREEFLYER = 7   # pos(3) + quat(4)
NV_FREEFLYER = 6   # linear_vel(3) + angular_vel(3)

# Total config/velocity dimensions for Pinocchio
NQ_PIN = NQ_FREEFLYER + NUM_ACTUATED  # 20
NV_PIN = NV_FREEFLYER + NUM_ACTUATED  # 19

# G1 geometry
TORSO_HEIGHT = 0.95      # initial torso z (feet just above ground at home)
HIP_WIDTH = 0.105        # lateral offset to hip
THIGH_LENGTH = 0.35
SHIN_LENGTH = 0.35
FOOT_HEIGHT = 0.03       # approx bottom of foot below ankle

# Total robot mass (approximate)
TOTAL_MASS = 35.0

# CoM height at home pose (approximate)
COM_HEIGHT_HOME = 0.70


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
        scene_path: Path to MJCF scene file. Defaults to SCENE_FLAT_PATH.

    Returns:
        Tuple of (mj_model, mj_data).
    """
    import mujoco

    path = scene_path or SCENE_FLAT_PATH
    mj_model = mujoco.MjModel.from_xml_path(str(path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def load_pinocchio_model(urdf_path: Path | None = None):
    """Load Pinocchio model with freeflyer root joint from URDF.

    The URDF has a floating joint which Pinocchio handles via
    JointModelFreeFlyer as root_joint.

    Args:
        urdf_path: Path to URDF file. Defaults to URDF_PATH.

    Returns:
        Tuple of (pin_model, pin_data).
    """
    import pinocchio as pin

    path = urdf_path or URDF_PATH
    model = pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())

    # Match MuJoCo armature for joints after freeflyer
    ARMATURE = 0.01
    for i in range(1, model.njoints):
        jidx = model.joints[i].idx_v
        jnv = model.joints[i].nv
        model.armature[jidx:jidx + jnv] = ARMATURE

    data = model.createData()
    return model, data


# ---------------------------------------------------------------------------
# State conversion helpers
# ---------------------------------------------------------------------------

def get_mj_joint_qpos_adr(mj_model) -> np.ndarray:
    """Get qpos addresses for each actuated joint in MuJoCo.

    Returns:
        Array of qpos indices for the 13 actuated joints.
    """
    import mujoco

    addrs = np.zeros(NUM_ACTUATED, dtype=int)
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addrs[i] = mj_model.jnt_qposadr[jid]
    return addrs


def get_mj_joint_dofadr(mj_model) -> np.ndarray:
    """Get dof addresses for each actuated joint in MuJoCo.

    Returns:
        Array of dof (velocity) indices for the 13 actuated joints.
    """
    import mujoco

    addrs = np.zeros(NUM_ACTUATED, dtype=int)
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addrs[i] = mj_model.jnt_dofadr[jid]
    return addrs


def get_state_from_mujoco(
    mj_model, mj_data
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract full state from MuJoCo and convert to Pinocchio convention.

    MuJoCo freeflyer qpos: [x, y, z, qw, qx, qy, qz, j1..j13]
    Pinocchio freeflyer q: [x, y, z, qx, qy, qz, qw, j1..j13]

    MuJoCo qvel: [vx, vy, vz, wx, wy, wz, dj1..dj13]
    Pinocchio v:  [vx, vy, vz, wx, wy, wz, dj1..dj13]  (same ordering)

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.

    Returns:
        Tuple of (q_pin [NQ_PIN], v_pin [NV_PIN]).
    """
    qpos_addrs = get_mj_joint_qpos_adr(mj_model)
    dof_addrs = get_mj_joint_dofadr(mj_model)

    q_pin = np.zeros(NQ_PIN)
    v_pin = np.zeros(NV_PIN)

    # Freeflyer position
    q_pin[0:3] = mj_data.qpos[0:3]  # xyz position

    # Quaternion: MuJoCo (w,x,y,z) -> Pinocchio (x,y,z,w)
    q_pin[3:7] = mj_quat_to_pin(mj_data.qpos[3:7])

    # Actuated joints
    for i in range(NUM_ACTUATED):
        q_pin[NQ_FREEFLYER + i] = mj_data.qpos[qpos_addrs[i]]

    # Freeflyer velocity (same convention: linear, angular)
    v_pin[0:6] = mj_data.qvel[0:6]

    # Actuated joint velocities
    for i in range(NUM_ACTUATED):
        v_pin[NV_FREEFLYER + i] = mj_data.qvel[dof_addrs[i]]

    return q_pin, v_pin


def set_mujoco_state(
    mj_model, mj_data, q_pin: np.ndarray, v_pin: np.ndarray
) -> None:
    """Set MuJoCo state from Pinocchio convention vectors.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        q_pin: Configuration in Pinocchio convention [NQ_PIN].
        v_pin: Velocity in Pinocchio convention [NV_PIN].
    """
    qpos_addrs = get_mj_joint_qpos_adr(mj_model)
    dof_addrs = get_mj_joint_dofadr(mj_model)

    # Freeflyer position
    mj_data.qpos[0:3] = q_pin[0:3]

    # Quaternion: Pinocchio (x,y,z,w) -> MuJoCo (w,x,y,z)
    mj_data.qpos[3:7] = pin_quat_to_mj(q_pin[3:7])

    # Actuated joints
    for i in range(NUM_ACTUATED):
        mj_data.qpos[qpos_addrs[i]] = q_pin[NQ_FREEFLYER + i]

    # Velocities
    mj_data.qvel[0:6] = v_pin[0:6]
    for i in range(NUM_ACTUATED):
        mj_data.qvel[dof_addrs[i]] = v_pin[NV_FREEFLYER + i]


def build_q_pin(
    base_pos: np.ndarray,
    base_quat_xyzw: np.ndarray,
    q_joints: np.ndarray,
) -> np.ndarray:
    """Build full Pinocchio configuration vector.

    Args:
        base_pos: Base position [3].
        base_quat_xyzw: Base quaternion in Pinocchio convention [4].
        q_joints: Actuated joint angles [NUM_ACTUATED].

    Returns:
        Full configuration vector [NQ_PIN].
    """
    q = np.zeros(NQ_PIN)
    q[0:3] = base_pos
    q[3:7] = base_quat_xyzw
    q[NQ_FREEFLYER:] = q_joints
    return q


def build_home_q_pin() -> np.ndarray:
    """Build Pinocchio configuration vector at home position.

    Returns:
        Home configuration [NQ_PIN] with identity quaternion and Q_HOME joints.
    """
    return build_q_pin(
        base_pos=np.array([0.0, 0.0, TORSO_HEIGHT]),
        base_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),  # identity
        q_joints=Q_HOME.copy(),
    )


def ensure_media_dir() -> Path:
    """Create and return the media directory path.

    Returns:
        Path to the media directory.
    """
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    return MEDIA_DIR
