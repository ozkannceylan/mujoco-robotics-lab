"""Shared constants, real-stack model loading, and controller helpers for Lab 3."""

from __future__ import annotations

import math
from functools import lru_cache
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

LAB2_MODELS_DIR = PROJECT_ROOT / "lab-2-Ur5e-robotics-lab" / "models"
MENAGERIE_DIR = LAB2_MODELS_DIR / "mujoco_menagerie"
UR5E_XML_PATH = MENAGERIE_DIR / "universal_robots_ur5e" / "ur5e.xml"
ROBOTIQ_XML_PATH = MENAGERIE_DIR / "robotiq_2f85" / "2f85.xml"

# Legacy scene paths remain as public selectors for load_mujoco_model().
SCENE_TORQUE_PATH = MODELS_DIR / "scene_torque.xml"
SCENE_TABLE_PATH = MODELS_DIR / "scene_table.xml"
URDF_PATH = MODELS_DIR / "ur5e.urdf"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
NUM_JOINTS = 6
DT = 0.001
ARMATURE = 0.1

JOINT_NAMES: Tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

JOINT_LIMITS_LOWER = np.array([
    -2 * math.pi,
    -2 * math.pi,
    -math.pi,
    -2 * math.pi,
    -2 * math.pi,
    -2 * math.pi,
])
JOINT_LIMITS_UPPER = np.array([
    2 * math.pi,
    2 * math.pi,
    math.pi,
    2 * math.pi,
    2 * math.pi,
    2 * math.pi,
])

VELOCITY_LIMITS = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28])
TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

# ---------------------------------------------------------------------------
# Common joint configurations and scene constants
# ---------------------------------------------------------------------------
Q_ZEROS = np.zeros(NUM_JOINTS)
Q_HOME = np.array([
    -math.pi / 2, -math.pi / 2, math.pi / 2,
    -math.pi / 2, -math.pi / 2, 0.0,
])

ATTACHMENT_SITE_NAME = "attachment_site"
EE_SITE_NAME = "2f85_pinch"
GRIPPER_ACTUATOR_NAME = "2f85_fingers_actuator"
GRIPPER_OPEN_CTRL = 0.0

TABLE_POS = np.array([0.45, 0.0, 0.30])
TABLE_SIZE = np.array([0.30, 0.40, 0.015])
TABLE_SURFACE_Z = float(TABLE_POS[2] + TABLE_SIZE[2])

GRIPPER_TIP_OFFSET = 0.1558


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
# MuJoCo scene construction
# ---------------------------------------------------------------------------

def _configure_scene_visuals(spec) -> None:
    """Apply basic visual settings to the generated lab scene."""
    import mujoco

    spec.option.timestep = DT
    spec.stat.center = [0.35, 0.0, 0.42]
    spec.stat.extent = 1.2

    spec.visual.headlight.active = True
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    floor_tex = spec.add_texture()
    floor_tex.name = "lab3_floor_tex"
    floor_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_tex.rgb1 = [0.88, 0.88, 0.88]
    floor_tex.rgb2 = [0.68, 0.68, 0.68]
    floor_tex.width = 512
    floor_tex.height = 512

    floor_mat = spec.add_material()
    floor_mat.name = "lab3_floor_mat"
    floor_mat.textures = ["lab3_floor_tex"] + [""] * 9
    floor_mat.texrepeat = [6.0, 6.0]
    floor_mat.specular = 0.08
    floor_mat.shininess = 0.1

    table_mat = spec.add_material()
    table_mat.name = "lab3_table_mat"
    table_mat.rgba = [0.55, 0.38, 0.20, 1.0]
    table_mat.specular = 0.12
    table_mat.shininess = 0.2


def _add_floor(spec) -> None:
    """Add the floor to the scene."""
    import mujoco

    wb = spec.worldbody

    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.material = "lab3_floor_mat"


def _add_table(spec) -> None:
    """Add the tabletop used by the force-control phases."""
    import mujoco

    wb = spec.worldbody
    table = wb.add_body()
    table.name = "table"
    table.pos = TABLE_POS.tolist()

    top = table.add_geom()
    top.name = "table_top"
    top.type = mujoco.mjtGeom.mjGEOM_BOX
    top.size = TABLE_SIZE.tolist()
    top.material = "lab3_table_mat"

    leg_half = [0.02, 0.02, TABLE_POS[2] / 2]
    leg_offsets = [
        (0.25, 0.35, -TABLE_POS[2] / 2 - TABLE_SIZE[2]),
        (-0.25, 0.35, -TABLE_POS[2] / 2 - TABLE_SIZE[2]),
        (0.25, -0.35, -TABLE_POS[2] / 2 - TABLE_SIZE[2]),
        (-0.25, -0.35, -TABLE_POS[2] / 2 - TABLE_SIZE[2]),
    ]
    for i, (lx, ly, lz) in enumerate(leg_offsets):
        leg = table.add_geom()
        leg.name = f"table_leg_{i}"
        leg.type = mujoco.mjtGeom.mjGEOM_BOX
        leg.size = leg_half
        leg.pos = [lx, ly, lz]
        leg.rgba = [0.35, 0.35, 0.35, 1.0]


def build_mujoco_scene_spec(with_table: bool = False):
    """Build a UR5e + Robotiq scene using the MuJoCo Menagerie assets."""
    import mujoco

    robot = mujoco.MjSpec.from_file(str(UR5E_XML_PATH))
    gripper = mujoco.MjSpec.from_file(str(ROBOTIQ_XML_PATH))

    attach_site = next(site for site in robot.sites if site.name == ATTACHMENT_SITE_NAME)
    robot.attach(gripper, prefix="2f85_", site=attach_site)

    _configure_scene_visuals(robot)
    _add_floor(robot)
    if with_table:
        _add_table(robot)

    return robot


@lru_cache(maxsize=2)
def _compile_scene_model(with_table: bool):
    """Compile and cache the generated MuJoCo scene."""
    spec = build_mujoco_scene_spec(with_table=with_table)
    return spec.compile()


def load_mujoco_model(scene_path: Path | None = None):
    """Load MuJoCo model/data for the requested Lab 3 scene selector."""
    import mujoco

    with_table = Path(scene_path) == SCENE_TABLE_PATH if scene_path is not None else False
    mj_model = _compile_scene_model(with_table)
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


# ---------------------------------------------------------------------------
# Pinocchio model loading
# ---------------------------------------------------------------------------

def load_pinocchio_model(urdf_path: Path | None = None):
    """Load the real-stack UR5e + fixed Robotiq payload Pinocchio model."""
    import pinocchio as pin

    path = urdf_path or URDF_PATH
    model = pin.buildModelFromUrdf(str(path))
    model.armature[:] = ARMATURE

    data = model.createData()
    try:
        model.getFrameId("pinch_link")
        ee_frame_name = "pinch_link"
    except Exception:
        ee_frame_name = "ee_link"
    ee_frame_id = model.getFrameId(ee_frame_name)
    return model, data, ee_frame_id


# ---------------------------------------------------------------------------
# IDs and scene helpers
# ---------------------------------------------------------------------------

def get_mj_body_id(mj_model, name: str) -> int:
    """Return the MuJoCo body ID for a named body."""
    import mujoco
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)


def get_mj_geom_id(mj_model, name: str) -> int:
    """Return the MuJoCo geom ID for a named geom."""
    import mujoco
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)


def get_mj_site_id(mj_model, name: str) -> int:
    """Return the MuJoCo site ID for a named site."""
    import mujoco
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)


def get_mj_ee_site_id(mj_model) -> int:
    """Get the MuJoCo site ID for the Robotiq pinch site."""
    return get_mj_site_id(mj_model, EE_SITE_NAME)


def get_mj_attachment_site_id(mj_model) -> int:
    """Get the MuJoCo site ID for the UR5e attachment site."""
    return get_mj_site_id(mj_model, ATTACHMENT_SITE_NAME)


def get_gripper_actuator_id(mj_model) -> int:
    """Get the MuJoCo actuator ID for the Robotiq finger actuator."""
    import mujoco
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME)


def get_table_surface_z(mj_model) -> float:
    """Return the world-frame Z of the table surface."""
    table_geom_id = get_mj_geom_id(mj_model, "table_top")
    table_body_id = mj_model.geom_bodyid[table_geom_id]
    table_body_pos = mj_model.body_pos[table_body_id]
    table_geom_pos = mj_model.geom_pos[table_geom_id]
    table_half_height = mj_model.geom_size[table_geom_id][2]
    return float(table_body_pos[2] + table_geom_pos[2] + table_half_height)


# ---------------------------------------------------------------------------
# IK and control helpers
# ---------------------------------------------------------------------------

def get_topdown_rotation(pin_model, pin_data, ee_fid: int) -> np.ndarray:
    """Return the end-effector orientation at Q_HOME."""
    import pinocchio as pin

    pin.forwardKinematics(pin_model, pin_data, Q_HOME)
    pin.updateFramePlacements(pin_model, pin_data)
    return pin_data.oMf[ee_fid].rotation.copy()


def solve_dls_ik(
    pin_model,
    pin_data,
    ee_fid: int,
    x_target: np.ndarray,
    R_target: np.ndarray,
    q_init: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-4,
    alpha: float = 0.5,
    lambda_sq: float = 1e-4,
) -> np.ndarray | None:
    """Solve full 6D IK with damped least squares."""
    import pinocchio as pin

    q = q_init.copy()

    for _ in range(max_iter):
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        oMf = pin_data.oMf[ee_fid]

        pos_err = x_target - oMf.translation
        ori_err = pin.log3(R_target @ oMf.rotation.T)
        err = np.concatenate([pos_err, ori_err])
        if np.linalg.norm(err) < tol:
            return q

        pin.computeJointJacobians(pin_model, pin_data, q)
        J = pin.getFrameJacobian(
            pin_model, pin_data, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        JJt = J @ J.T + lambda_sq * np.eye(6)
        dq = alpha * J.T @ np.linalg.solve(JJt, err)
        q = pin.integrate(pin_model, q, dq)
        q = np.clip(q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    return None


def arm_torques_to_ctrl(
    mj_model,
    mj_data,
    tau: np.ndarray,
) -> np.ndarray:
    """Map desired joint torques to Menagerie arm actuator position setpoints."""
    tau = np.asarray(tau, dtype=float)
    ctrl = np.zeros(NUM_JOINTS, dtype=float)

    for act_id in range(NUM_JOINTS):
        gain = float(mj_model.actuator_gainprm[act_id, 0])
        bias0 = float(mj_model.actuator_biasprm[act_id, 0])
        bias1 = float(mj_model.actuator_biasprm[act_id, 1])
        bias2 = float(mj_model.actuator_biasprm[act_id, 2])
        length = float(mj_data.actuator_length[act_id])
        velocity = float(mj_data.actuator_velocity[act_id])

        ctrl_i = (tau[act_id] - (bias0 + bias1 * length + bias2 * velocity)) / gain
        if mj_model.actuator_ctrllimited[act_id]:
            lo, hi = mj_model.actuator_ctrlrange[act_id]
            ctrl_i = float(np.clip(ctrl_i, lo, hi))
        ctrl[act_id] = ctrl_i

    return ctrl


def apply_arm_torques(
    mj_model,
    mj_data,
    tau: np.ndarray,
    gripper_ctrl: float = GRIPPER_OPEN_CTRL,
) -> None:
    """Apply desired arm torques through the Menagerie position-servo actuators."""
    ctrl = mj_data.ctrl.copy()
    ctrl[:NUM_JOINTS] = arm_torques_to_ctrl(mj_model, mj_data, tau)
    if mj_model.nu > NUM_JOINTS:
        try:
            ctrl[get_gripper_actuator_id(mj_model)] = gripper_ctrl
        except Exception:
            ctrl[NUM_JOINTS] = gripper_ctrl
    mj_data.ctrl[:] = ctrl


# ---------------------------------------------------------------------------
# EE pose and velocity helpers
# ---------------------------------------------------------------------------

def get_ee_pose(
    pin_model, pin_data, ee_fid: int, q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute end-effector position and rotation matrix via Pinocchio FK."""
    import pinocchio as pin

    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    oMf = pin_data.oMf[ee_fid]
    return oMf.translation.copy(), oMf.rotation.copy()


def get_ee_velocity(
    pin_model, pin_data, ee_fid: int, q: np.ndarray, qd: np.ndarray
) -> np.ndarray:
    """Compute 6D end-effector velocity (linear, angular) in world frame."""
    import pinocchio as pin

    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    pin.computeJointJacobians(pin_model, pin_data, q)
    J = pin.getFrameJacobian(
        pin_model, pin_data, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    return J @ qd


def clip_torques(tau: np.ndarray) -> np.ndarray:
    """Clip joint torques to the UR5e torque limits."""
    return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
