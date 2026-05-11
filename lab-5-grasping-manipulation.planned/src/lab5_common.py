"""Lab 5 — Common constants, paths, and model loading utilities."""

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
SCENE_PATH = MODELS_DIR / "scene_grasp.xml"

# Pinocchio model: arm-only URDF from Lab 3 (no gripper needed for FK/IK)
_LAB3_DIR = _LAB_DIR.parent / "lab-3-dynamics-force-control"
URDF_PATH = _LAB3_DIR / "models" / "ur5e.urdf"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_JOINTS = 6          # UR5e arm DOF
GRIPPER_IDX = 6         # ctrl index for gripper position actuator
DT = 0.001              # simulation timestep (1 kHz)

# ctrl[GRIPPER_IDX] setpoints
GRIPPER_OPEN = 0.030    # metres — fingers spread 60 mm gap
GRIPPER_CLOSED = 0.0    # metres — fingers at minimum gap (squeeze object)

# Offset from tool0 origin to fingertip pad centre (along tool0 local Z, pointing down)
# = gripper_base z (0.020) + finger_body z (0.060) + pad z in finger_body (0.025) = 0.105 m
# NOTE: the pad z within finger_body is 0.025 m (pos="0 0.009 0.025" in ur5e_gripper.xml),
#       NOT the pad half-height (0.008). Using 0.090 placed pads 15 mm too low, into table.
GRIPPER_TIP_OFFSET = 0.105  # metres

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
# Scene geometry (must match scene_grasp.xml)
# ---------------------------------------------------------------------------

TABLE_TOP_Z = 0.315       # world Z of table surface (table centre 0.300, half 0.015)
BOX_HALF = 0.020          # half-side of graspable box (40 mm cube)

# Graspable box centre (world frame)
BOX_A_POS = np.array([0.35, 0.20, TABLE_TOP_Z + BOX_HALF])   # pick location
BOX_B_POS = np.array([0.35, -0.20, TABLE_TOP_Z + BOX_HALF])  # place location

# Pre-grasp / pre-place height above the grasp target (tool0 frame Z offset)
PREGRASP_CLEARANCE = 0.15  # metres above grasp level


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_mujoco_model(scene_path: Path | None = None) -> tuple:
    """Load MuJoCo model and data for the Lab 5 scene.

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
    """Load Pinocchio arm model (no gripper) and return (model, data, ee_fid).

    Args:
        urdf_path: Path to arm URDF. Defaults to Lab 3 URDF.

    Returns:
        Tuple of (pin_model, pin_data, ee_frame_id).
    """
    path = urdf_path or URDF_PATH
    model = pin.buildModelFromUrdf(str(path))
    model.armature[:] = 0.01   # match MuJoCo joint armature
    data = model.createData()
    ee_fid = model.getFrameId("ee_link")
    return model, data, ee_fid


# ---------------------------------------------------------------------------
# Top-down orientation
# ---------------------------------------------------------------------------

def get_topdown_rotation(pin_model, pin_data, ee_fid: int) -> np.ndarray:
    """Return the EE rotation matrix at Q_HOME (top-down approach orientation).

    For tabletop grasping we keep this orientation fixed throughout the
    pick-and-place cycle. Computing it from FK ensures consistency with
    the actual kinematic chain.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.

    Returns:
        3×3 rotation matrix R_topdown.
    """
    pin.forwardKinematics(pin_model, pin_data, Q_HOME)
    pin.updateFramePlacements(pin_model, pin_data)
    return pin_data.oMf[ee_fid].rotation.copy()


# ---------------------------------------------------------------------------
# Helpers
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


def get_mj_body_id(mj_model, name: str) -> int:
    """Return MuJoCo body ID by name."""
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)


def get_mj_geom_id(mj_model, name: str) -> int:
    """Return MuJoCo geom ID by name."""
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)


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
