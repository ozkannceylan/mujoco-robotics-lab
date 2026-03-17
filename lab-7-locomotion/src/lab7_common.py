"""Lab 7 — Common constants, paths, and model loading utilities for the Unitree G1."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_LAB_DIR = Path(__file__).resolve().parent.parent

# G1 model lives in the vla_zero_to_hero project's menagerie copy.
_MENAGERIE_DIR = (
    Path(__file__).resolve().parents[3]  # ~/projects
    / "vla_zero_to_hero"
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
)
G1_MJCF_PATH: Path = _MENAGERIE_DIR / "g1.xml"
G1_SCENE_PATH: Path = _MENAGERIE_DIR / "scene.xml"

MEDIA_DIR: Path = _LAB_DIR / "media"

# ---------------------------------------------------------------------------
# G1 model constants
# ---------------------------------------------------------------------------

# G1 has 29 actuated joints + 1 freejoint → nq=36, nv=35, nu=29.
NQ: int = 36
NV: int = 35
NU: int = 29

# ---------- qpos slices (MuJoCo convention) ----------
# Freejoint: [x, y, z, qw, qx, qy, qz]
Q_BASE = slice(0, 7)
Q_LEFT_LEG = slice(7, 13)   # hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
Q_RIGHT_LEG = slice(13, 19)
Q_WAIST = slice(19, 22)     # waist_yaw, waist_roll, waist_pitch
Q_LEFT_ARM = slice(22, 29)  # shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist×3
Q_RIGHT_ARM = slice(29, 36)

# ---------- qvel slices (velocity DOF, nv=35) ----------
# Freejoint velocity: [vx, vy, vz, wx, wy, wz]
V_BASE = slice(0, 6)
V_LEFT_LEG = slice(6, 12)
V_RIGHT_LEG = slice(12, 18)
V_WAIST = slice(18, 21)
V_LEFT_ARM = slice(21, 28)
V_RIGHT_ARM = slice(28, 35)

# ---------- ctrl slices (nu=29 position actuators) ----------
CTRL_LEFT_LEG = slice(0, 6)
CTRL_RIGHT_LEG = slice(6, 12)
CTRL_WAIST = slice(12, 15)
CTRL_LEFT_ARM = slice(15, 22)
CTRL_RIGHT_ARM = slice(22, 29)

# ---------------------------------------------------------------------------
# Standing pose (from G1 keyframe "stand")
# ---------------------------------------------------------------------------

# qpos of the 29 actuated joints in standing keyframe order:
#   left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)
Q_STAND_JOINTS = np.array([
    # left leg
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # right leg
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # waist
    0.0, 0.0, 0.0,
    # left arm
    0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
    # right arm
    0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
])

# Pelvis height after robot settles (keyframe = 0.79m, feet drop ~33mm to ground)
PELVIS_Z_STAND: float = 0.757   # metres above ground

# When Pinocchio loads the G1 MJCF with a FreeFlyer root joint, the pelvis body
# sits at its MJCF-defined position RELATIVE to the FreeFlyer frame.
# The MJCF places the pelvis at z=0.793 relative to the world root.
# Therefore: pelvis_world_z = pin_q[2] + PELVIS_MJCF_Z
#
# Conversely, for a desired pelvis world position z:
#   pin_q[2] = z - PELVIS_MJCF_Z
#
# MuJoCo freejoint directly stores the pelvis world position (qpos[2] = pelvis_z).
PELVIS_MJCF_Z: float = 0.793    # pelvis z offset in MJCF from world root [m]

# CoM height above ground during upright standing (used as LIPM z_c)
Z_C: float = 0.66               # metres

# Initial foot positions in world frame (symmetrical stance, x=0)
FOOT_Y_OFFSET: float = 0.118    # metres, ±y offset of each foot from centre

# ---------------------------------------------------------------------------
# Pinocchio frame names for the G1
# ---------------------------------------------------------------------------

LEFT_FOOT_FRAME: str = "left_foot"
RIGHT_FOOT_FRAME: str = "right_foot"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_g1_mujoco(
    scene_path: Path | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the Unitree G1 MuJoCo model.

    Args:
        scene_path: Path to scene MJCF. Defaults to G1_SCENE_PATH.

    Returns:
        (mj_model, mj_data) initialised to the "stand" keyframe.
    """
    path = scene_path or G1_SCENE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"G1 scene not found at {path}. "
            "Ensure the vla_zero_to_hero menagerie is cloned at "
            "~/projects/vla_zero_to_hero/third_party/mujoco_menagerie/."
        )
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)  # "stand" keyframe
    mujoco.mj_forward(m, d)
    return m, d


def load_g1_pinocchio(
    mjcf_path: Path | None = None,
) -> tuple[pin.Model, pin.Data]:
    """Load the Unitree G1 into Pinocchio using the MJCF file.

    Uses a FreeFlyer root joint so that nq/nv matches MuJoCo (nq=36, nv=35).

    Args:
        mjcf_path: Path to g1.xml. Defaults to G1_MJCF_PATH.

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


def mj_qpos_to_pin(mj_qpos: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo full qpos (nq=36) to Pinocchio convention.

    Two differences between MuJoCo and Pinocchio for the free joint:
      1. Quaternion order: MuJoCo (w,x,y,z) → Pinocchio (x,y,z,w)
      2. Base position: MuJoCo qpos[0:3] is the ABSOLUTE pelvis world position.
         Pinocchio q[0:3] is the FreeFlyer TRANSLATION (pelvis relative to
         its MJCF origin PELVIS_MJCF_Z).
         → pin_q[2] = mj_qpos[2] - PELVIS_MJCF_Z
    """
    q = mj_qpos.copy()
    # Base position: subtract MJCF pelvis z offset
    q[2] = mj_qpos[2] - PELVIS_MJCF_Z
    # Quaternion: MuJoCo (w,x,y,z) → Pinocchio (x,y,z,w)
    q[3] = mj_qpos[4]  # x
    q[4] = mj_qpos[5]  # y
    q[5] = mj_qpos[6]  # z
    q[6] = mj_qpos[3]  # w
    return q


def pin_q_to_mj(pin_q: np.ndarray) -> np.ndarray:
    """Convert a Pinocchio full q (nq=36) to MuJoCo qpos convention.

    Inverse of mj_qpos_to_pin.
    """
    q = pin_q.copy()
    # Base position: add MJCF pelvis z offset
    q[2] = pin_q[2] + PELVIS_MJCF_Z
    # Quaternion: Pinocchio (x,y,z,w) → MuJoCo (w,x,y,z)
    q[3] = pin_q[6]   # w
    q[4] = pin_q[3]   # x
    q[5] = pin_q[4]   # y
    q[6] = pin_q[5]   # z
    return q


def pelvis_world_to_pin_base(pelvis_world: np.ndarray) -> np.ndarray:
    """Convert desired pelvis world position to Pinocchio FreeFlyer base position.

    Args:
        pelvis_world: (3,) desired pelvis position in world frame [m].

    Returns:
        (3,) Pinocchio q[0:3] = FreeFlyer translation.
    """
    base = pelvis_world.copy()
    base[2] -= PELVIS_MJCF_Z
    return base


def mj_state_to_pin(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray]:
    """Extract current state from MuJoCo and return in Pinocchio format.

    Returns:
        (q, v): Pinocchio-convention configuration and velocity vectors.
    """
    q = mj_qpos_to_pin(mj_data.qpos.copy())
    v = mj_data.qvel.copy()  # velocity layout is identical (both use (vx,vy,vz,wx,wy,wz))
    return q, v


def build_pin_q_standing(pelvis_world_pos: np.ndarray | None = None) -> np.ndarray:
    """Build the full Pinocchio q vector for the standing pose.

    The FreeFlyer base translation is set such that the pelvis ends up at
    the desired world position (accounts for PELVIS_MJCF_Z offset).

    Args:
        pelvis_world_pos: (3,) desired pelvis WORLD position.
                          Defaults to [0, 0, PELVIS_Z_STAND].

    Returns:
        q (nq=36) in Pinocchio convention with correct FreeFlyer translation.
    """
    q = np.zeros(NQ)
    if pelvis_world_pos is None:
        world_pos = np.array([0.0, 0.0, PELVIS_Z_STAND])
    else:
        world_pos = np.asarray(pelvis_world_pos, dtype=float)
    # Pinocchio FreeFlyer translation = pelvis_world - MJCF_pelvis_offset
    q[0:3] = pelvis_world_to_pin_base(world_pos)
    # Upright: Pinocchio quaternion (x,y,z,w) = (0,0,0,1)
    q[3:7] = [0.0, 0.0, 0.0, 1.0]
    q[7:36] = Q_STAND_JOINTS
    return q


# ---------------------------------------------------------------------------
# Quick model info printout (useful for debugging)
# ---------------------------------------------------------------------------


def print_g1_info(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """Print a summary of G1 model dimensions and joint names."""
    print(f"G1 MuJoCo model: nq={m.nq}, nv={m.nv}, nu={m.nu}")
    print("Joints:", [m.joint(i).name for i in range(m.njnt)])
    print("Actuators:", [m.actuator(i).name for i in range(m.nu)])
    mujoco.mj_forward(m, d)
    print(f"Pelvis pos: {d.xpos[m.body('pelvis').id]}")
    print(f"Left foot site: {d.site_xpos[m.site('left_foot').id]}")
    print(f"Right foot site: {d.site_xpos[m.site('right_foot').id]}")
    print(f"subtree_com[pelvis]: {d.subtree_com[m.body('pelvis').id]}")
