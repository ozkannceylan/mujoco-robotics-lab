"""Lab 7 — Common constants, paths, and model loading utilities for the Unitree G1.

Uses the real Unitree G1 from MuJoCo Menagerie (29 actuated DOFs).
For locomotion, only legs (12) + waist (3) = 15 DOFs are actively controlled.
Arms are locked at their "stand" keyframe pose.

Model dimensions: nq=36, nv=35, nu=29, mass=33.34 kg
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LAB_DIR: Path = Path(__file__).resolve().parent.parent
PROJECT_ROOT: Path = LAB_DIR.parent
MODELS_DIR: Path = LAB_DIR / "models"
MEDIA_DIR: Path = LAB_DIR / "media"
DOCS_DIR: Path = LAB_DIR / "docs"

# Menagerie G1 — the real model with mesh geometry and calibrated inertias
_MENAGERIE_G1_DIR: Path = (
    PROJECT_ROOT.parent / "vla_zero_to_hero"
    / "third_party" / "mujoco_menagerie" / "unitree_g1"
)
G1_MJCF_PATH: Path = _MENAGERIE_G1_DIR / "g1.xml"
G1_SCENE_PATH: Path = _MENAGERIE_G1_DIR / "scene.xml"

# ---------------------------------------------------------------------------
# G1 Menagerie model constants
# ---------------------------------------------------------------------------

NQ: int = 36   # 7 (freejoint) + 29 (hinge)
NV: int = 35   # 6 (base vel) + 29 (joint vel)
NU: int = 29   # all actuated joints
DT: float = 0.002  # default timestep [s] — set via option in scene
GRAVITY: float = 9.81

TOTAL_MASS: float = 33.34  # kg

# ---------------------------------------------------------------------------
# qpos slices (MuJoCo convention, nq=36)
# ---------------------------------------------------------------------------
# Freejoint: [x, y, z, qw, qx, qy, qz]           → qpos[0:7]
# Left leg:  hip_pitch, hip_roll, hip_yaw,
#            knee, ankle_pitch, ankle_roll          → qpos[7:13]
# Right leg: (same pattern)                         → qpos[13:19]
# Waist:     yaw, roll, pitch                       → qpos[19:22]
# Left arm:  shoulder_pitch/roll/yaw, elbow,
#            wrist_roll/pitch/yaw                   → qpos[22:29]
# Right arm: (same pattern)                         → qpos[29:36]

Q_BASE = slice(0, 7)
Q_LEFT_LEG = slice(7, 13)
Q_RIGHT_LEG = slice(13, 19)
Q_WAIST = slice(19, 22)
Q_LEFT_ARM = slice(22, 29)
Q_RIGHT_ARM = slice(29, 36)

# ---------------------------------------------------------------------------
# qvel slices (nv=35)
# ---------------------------------------------------------------------------

V_BASE = slice(0, 6)
V_LEFT_LEG = slice(6, 12)
V_RIGHT_LEG = slice(12, 18)
V_WAIST = slice(18, 21)
V_LEFT_ARM = slice(21, 28)
V_RIGHT_ARM = slice(28, 35)

# ---------------------------------------------------------------------------
# ctrl slices (nu=29)
# ---------------------------------------------------------------------------
# Actuator order matches XML actuator declaration:
#   left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)

CTRL_LEFT_LEG = slice(0, 6)
CTRL_RIGHT_LEG = slice(6, 12)
CTRL_WAIST = slice(12, 15)
CTRL_LEFT_ARM = slice(15, 22)
CTRL_RIGHT_ARM = slice(22, 29)

# Locomotion control: legs + waist = 15 actuators
CTRL_LOCOMOTION = slice(0, 15)
N_LOCOMOTION: int = 15

# ---------------------------------------------------------------------------
# Joint name lists
# ---------------------------------------------------------------------------

LEG_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

WAIST_JOINT_NAMES: list[str] = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

ARM_JOINT_NAMES: list[str] = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# ---------------------------------------------------------------------------
# Standing pose (from Menagerie keyframe "stand")
# ---------------------------------------------------------------------------

# Arm neutral pose from keyframe — held fixed during locomotion
ARM_NEUTRAL_LEFT: np.ndarray = np.array([0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0])
ARM_NEUTRAL_RIGHT: np.ndarray = np.array([0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0])

# Full ctrl for stand keyframe (nu=29)
CTRL_STAND: np.ndarray = np.array([
    # left leg (6)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # right leg (6)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # waist (3)
    0.0, 0.0, 0.0,
    # left arm (7)
    0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
    # right arm (7)
    0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
])

# Pelvis MJCF offset: <body name="pelvis" pos="0 0 0.793"> in g1.xml.
# Pinocchio FreeFlyer adds this to q[2], so: pelvis_world_z = pin_q[2] + 0.793.
# Conversely: pin_q[2] = mj_qpos[2] - 0.793.
PELVIS_MJCF_Z: float = 0.793

# Pelvis height from keyframe
PELVIS_Z_KEYFRAME: float = 0.79   # from keyframe qpos
PELVIS_Z_STAND: float = 0.757     # after settling under gravity

# CoM height above ground during upright standing (LIPM z_c)
Z_C: float = 0.66

# Foot Y offset from center (hip pos in MJCF: ±0.064452)
FOOT_Y_OFFSET: float = 0.064452

# ---------------------------------------------------------------------------
# Pinocchio frame names
# ---------------------------------------------------------------------------

LEFT_FOOT_FRAME: str = "left_ankle_roll_link"
RIGHT_FOOT_FRAME: str = "right_ankle_roll_link"

# ---------------------------------------------------------------------------
# Video / rendering constants
# ---------------------------------------------------------------------------

RENDER_WIDTH: int = 1920
RENDER_HEIGHT: int = 1080
RENDER_FPS: int = 30


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_g1_mujoco(
    scene_path: Path | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the Unitree G1 MuJoCo model from Menagerie.

    Resets to the "stand" keyframe and runs mj_forward.
    Sets offscreen framebuffer to RENDER_WIDTH x RENDER_HEIGHT.

    Args:
        scene_path: Path to scene MJCF. Defaults to G1_SCENE_PATH.

    Returns:
        (mj_model, mj_data) initialised to the "stand" keyframe.
    """
    path = scene_path or G1_SCENE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"G1 scene not found at {path}.\n"
            "Clone MuJoCo Menagerie:\n"
            "  git clone https://github.com/google-deepmind/mujoco_menagerie.git"
        )
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)

    # Set offscreen buffer for rendering
    m.vis.global_.offwidth = RENDER_WIDTH
    m.vis.global_.offheight = RENDER_HEIGHT

    # Reset to "stand" keyframe
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def load_g1_pinocchio(
    mjcf_path: Path | None = None,
) -> tuple[pin.Model, pin.Data]:
    """Load the Unitree G1 into Pinocchio using the MJCF file.

    Uses a FreeFlyer root joint so that nq/nv matches MuJoCo.

    Returns:
        (model, data) ready for kinematics/dynamics computations.
    """
    path = mjcf_path or G1_MJCF_PATH
    if not path.exists():
        raise FileNotFoundError(f"G1 MJCF not found at {path}.")
    model = pin.buildModelFromMJCF(str(path), pin.JointModelFreeFlyer())
    data = model.createData()
    return model, data


# ---------------------------------------------------------------------------
# Quaternion convention utilities
# ---------------------------------------------------------------------------


def mj_quat_to_pin(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion (w,x,y,z) to Pinocchio (x,y,z,w)."""
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def pin_quat_to_mj(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert Pinocchio quaternion (x,y,z,w) to MuJoCo (w,x,y,z)."""
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def mj_qpos_to_pin(mj_qpos: np.ndarray) -> np.ndarray:
    """Convert MuJoCo full qpos (nq=36) to Pinocchio convention.

    Two corrections:
      1. Base Z: pin_q[2] = mj_qpos[2] - PELVIS_MJCF_Z  (remove MJCF offset)
      2. Quaternion: MuJoCo (w,x,y,z) → Pinocchio (x,y,z,w)
    """
    q = mj_qpos.copy()
    q[2] = mj_qpos[2] - PELVIS_MJCF_Z
    q[3:7] = mj_quat_to_pin(mj_qpos[3:7])
    return q


def pin_q_to_mj(pin_q: np.ndarray) -> np.ndarray:
    """Convert Pinocchio full q (nq=36) to MuJoCo qpos convention.

    Inverse of mj_qpos_to_pin.
    """
    q = pin_q.copy()
    q[2] = pin_q[2] + PELVIS_MJCF_Z
    q[3:7] = pin_quat_to_mj(pin_q[3:7])
    return q


def mj_state_to_pin(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract current state from MuJoCo and return in Pinocchio format.

    Returns:
        (q, v): Pinocchio-convention configuration and velocity vectors.
    """
    q = mj_qpos_to_pin(mj_data.qpos.copy())
    v = mj_data.qvel.copy()
    return q, v


# ---------------------------------------------------------------------------
# Arm locking helper
# ---------------------------------------------------------------------------


def set_arm_ctrl_neutral(ctrl: np.ndarray) -> np.ndarray:
    """Set arm actuator commands to neutral (stand keyframe) pose.

    Modifies ctrl in-place and returns it.
    """
    ctrl[CTRL_LEFT_ARM] = ARM_NEUTRAL_LEFT
    ctrl[CTRL_RIGHT_ARM] = ARM_NEUTRAL_RIGHT
    return ctrl


# ---------------------------------------------------------------------------
# Quick model info printout
# ---------------------------------------------------------------------------


def print_g1_info(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """Print a summary of the G1 model dimensions and key positions."""
    print(f"G1 MuJoCo model: nq={m.nq}, nv={m.nv}, nu={m.nu}")
    print(f"Bodies: {m.nbody}, Joints: {m.njnt}, Geoms: {m.ngeom}")
    total_mass = sum(m.body(i).mass[0] for i in range(m.nbody))
    print(f"Total mass: {total_mass:.2f} kg")
    pelvis_id = m.body("pelvis").id
    print(f"Pelvis pos: {d.xpos[pelvis_id]}")
    print(f"World CoM: {d.subtree_com[0]}")
