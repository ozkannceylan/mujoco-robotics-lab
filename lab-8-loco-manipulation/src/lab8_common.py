"""Lab 8 — Common constants, paths, and model loaders (whole-body loco-manipulation).

Platform: Unitree G1 (Menagerie, 29 actuated DOF) driven by **torque** actuators.

Lab 8 reuses Lab 7's G1 conventions wholesale — joint ordering, qpos/qvel
slices, the pelvis MJCF z-offset, quaternion conversions — and re-exports them
so downstream modules import one namespace. What Lab 8 adds is the torque
model, its limits, and the control-side constants the QP/ID stack needs.

Model dimensions: nq=36, nv=35, nu=29, total mass 33.34 kg.
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
SRC_DIR: Path = LAB_DIR / "src"
MODELS_DIR: Path = LAB_DIR / "models"
MEDIA_DIR: Path = LAB_DIR / "media"
DOCS_DIR: Path = LAB_DIR / "docs"


def add_lab_src_to_path(lab_name: str) -> None:
    """Add another lab's ``src/`` to sys.path for cross-lab imports.

    Appended, never inserted at position 0. Labs share module names — Lab 7
    also has a ``standing_controller.py`` — so a foreign ``src/`` placed ahead
    of this lab's own directory silently shadows local modules with the wrong
    implementation. Appending keeps local modules winning.
    """
    src = PROJECT_ROOT / lab_name / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.append(str(src))


# This lab's own src/ must precede any foreign lab on the path.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


add_lab_src_to_path("lab-7-locomotion")

# Re-exported Lab 7 G1 conventions — single source of truth, no duplication.
from lab7_common import (  # noqa: E402
    ARM_JOINT_NAMES,
    ARM_NEUTRAL_LEFT,
    ARM_NEUTRAL_RIGHT,
    CTRL_LEFT_ARM,
    CTRL_LEFT_LEG,
    CTRL_RIGHT_ARM,
    CTRL_RIGHT_LEG,
    CTRL_STAND,
    CTRL_WAIST,
    FOOT_Y_OFFSET,
    G1_MJCF_PATH,
    LEFT_FOOT_FRAME,
    LEG_JOINT_NAMES,
    NQ,
    NU,
    NV,
    PELVIS_MJCF_Z,
    PELVIS_Z_STAND,
    Q_BASE,
    Q_LEFT_ARM,
    Q_LEFT_LEG,
    Q_RIGHT_ARM,
    Q_RIGHT_LEG,
    Q_STAND_JOINTS,
    Q_WAIST,
    RIGHT_FOOT_FRAME,
    TOTAL_MASS,
    V_BASE,
    V_LEFT_ARM,
    V_LEFT_LEG,
    V_RIGHT_ARM,
    V_RIGHT_LEG,
    V_WAIST,
    WAIST_JOINT_NAMES,
    Z_C,
    mj_quat_to_pin,
    mj_qpos_to_pin,
    pin_q_to_mj,
    pin_quat_to_mj,
    pelvis_world_to_pin_base,
)
from g1_torque_model import compile_g1_torque_model, torque_limits  # noqa: E402

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

DT: float = 0.001          # 1 kHz torque control (Lab 7 ran 0.002 position control)
GRAVITY: float = 9.81

# Index of the actuated joints inside the nv=35 velocity vector: the first 6
# entries are the floating base (unactuated), the remaining 29 are the joints.
V_JOINTS = slice(6, NV)
N_JOINTS: int = NU         # 29

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

RENDER_WIDTH: int = 1280
RENDER_HEIGHT: int = 720
RENDER_FPS: int = 60

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_g1_torque_mujoco(
    timestep: float = DT,
    keyframe: bool = True,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the torque-actuated G1 (Menagerie kinematics, `<motor>` actuators).

    Args:
        timestep: Simulation timestep [s].
        keyframe: Reset to the Menagerie "stand" pose (its ctrl vector is
            zeroed at build time — those were position targets).

    Returns:
        (mj_model, mj_data), forward-kinematics evaluated.
    """
    mj_model = compile_g1_torque_model(G1_MJCF_PATH, timestep=timestep)
    mj_model.vis.global_.offwidth = RENDER_WIDTH
    mj_model.vis.global_.offheight = RENDER_HEIGHT
    mj_data = mujoco.MjData(mj_model)
    if keyframe and mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    return mj_model, mj_data


def load_g1_pinocchio() -> tuple[pin.Model, pin.Data]:
    """Load the G1 into Pinocchio with a FreeFlyer root (nq=36, nv=35).

    Built from the same Menagerie MJCF the simulation uses. The torque variant
    differs only in actuator transmission and an added floor geom, neither of
    which enters the multibody dynamics — a claim the M0 gate verifies
    numerically (g(q) vs `qfrc_bias`, M(q) vs `mj_fullM`) rather than assuming.
    """
    if not G1_MJCF_PATH.exists():
        raise FileNotFoundError(
            f"G1 MJCF not found at {G1_MJCF_PATH}. Run ./tools/setup_env.sh"
        )
    # Pinocchio >= 4.1 wants an explicit root joint name and returns a tuple.
    try:
        built = pin.buildModelFromMJCF(
            str(G1_MJCF_PATH), pin.JointModelFreeFlyer(), "root_joint"
        )
    except TypeError:  # pragma: no cover - Pinocchio < 4.1
        built = pin.buildModelFromMJCF(str(G1_MJCF_PATH), pin.JointModelFreeFlyer())
    model = built[0] if isinstance(built, tuple) else built
    return model, model.createData()


# ---------------------------------------------------------------------------
# State conversion
# ---------------------------------------------------------------------------


def mj_state_to_pin(
    mj_data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the MuJoCo state to Pinocchio (q, v).

    Position: pelvis z-offset removed, quaternion reordered to (x,y,z,w).

    Velocity: MuJoCo reports the floating-base linear velocity in the **world**
    frame while Pinocchio's FreeFlyer expects it in the **local body** frame,
    so the base twist is rotated by Rᵀ. Angular velocity is already body-local
    in both. Getting this wrong is silent — it only shows up as wrong Coriolis
    terms once the base actually moves, which is why it is done here once
    rather than at each call site.
    """
    q = mj_qpos_to_pin(mj_data.qpos)
    v = mj_data.qvel.copy()
    quat_wxyz = mj_data.qpos[3:7]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat_wxyz)
    R = rot.reshape(3, 3)
    v[0:3] = R.T @ mj_data.qvel[0:3]
    return q, v


def pin_point_to_world(point: np.ndarray) -> np.ndarray:
    """Convert a Pinocchio-frame point to MuJoCo world coordinates.

    `mj_qpos_to_pin` subtracts `PELVIS_MJCF_Z` from the base translation, so
    Pinocchio's world sits 0.793 m below MuJoCo's. Every position Pinocchio
    reports (frame placements, CoM) carries that offset; verified to 0.000000
    mm by the M0 parity tests.
    """
    out = np.asarray(point, dtype=float).copy()
    out[2] += PELVIS_MJCF_Z
    return out


def world_point_to_pin(point: np.ndarray) -> np.ndarray:
    """Inverse of `pin_point_to_world`."""
    out = np.asarray(point, dtype=float).copy()
    out[2] -= PELVIS_MJCF_Z
    return out


def joint_torques_to_ctrl(tau_full: np.ndarray) -> np.ndarray:
    """Extract the 29 actuated-joint torques from an nv=35 generalized force."""
    tau_full = np.asarray(tau_full, dtype=float)
    if tau_full.shape[0] == NU:
        return tau_full.copy()
    if tau_full.shape[0] != NV:
        raise ValueError(f"expected nv={NV} or nu={NU} torque vector, got {tau_full.shape}")
    return tau_full[V_JOINTS].copy()


def clip_torques(tau: np.ndarray, mj_model: mujoco.MjModel) -> np.ndarray:
    """Clip joint torques to the model's actuator limits [N·m]."""
    limits = torque_limits(mj_model)
    return np.clip(np.asarray(tau, dtype=float), limits[:, 0], limits[:, 1])


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def com_position(mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> np.ndarray:
    """Whole-body centre of mass in world coordinates [m]."""
    del mj_model
    return mj_data.subtree_com[0].copy()


def lipm_omega(com_height: float) -> float:
    """Natural frequency of the linear inverted pendulum, ω = √(g/z_c) [1/s].

    `z_c` is the CoM height **above the contact plane**, not above the world
    origin — on flat ground with the feet at z≈0 the two coincide, but the
    distinction matters the moment a step goes onto a different level.
    """
    if com_height <= 0.0:
        raise ValueError(f"CoM height must be positive, got {com_height}")
    return float(np.sqrt(GRAVITY / com_height))


def divergent_component(
    com: np.ndarray, com_velocity: np.ndarray, omega: float
) -> np.ndarray:
    """Divergent component of motion (capture point), ξ = c + ċ/ω.

    The LIPM's CoM dynamics `c̈ = ω²(c − p_zmp)` split into a stable part that
    needs no control and an unstable part `ξ` obeying `ξ̇ = ω(ξ − p_zmp)`.
    Only `ξ` can run away, so a walking controller only has to steer `ξ` — and
    the ZMP it can produce is bounded by the support polygon, which is exactly
    the constraint the whole-body QP already enforces.

    Accepts 2D or 3D input and returns the same shape; only the horizontal
    components are meaningful under the constant-height assumption.
    """
    com = np.asarray(com, dtype=float)
    com_velocity = np.asarray(com_velocity, dtype=float)
    return com + com_velocity / float(omega)


def foot_contact_state(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData
) -> tuple[bool, bool]:
    """Return (left_in_contact, right_in_contact) from live MuJoCo contacts."""
    left = right = False
    for cid in range(mj_data.ncon):
        contact = mj_data.contact[cid]
        for gid in (contact.geom1, contact.geom2):
            body = mj_model.geom_bodyid[gid]
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if name.startswith("left_ankle"):
                left = True
            elif name.startswith("right_ankle"):
                right = True
    return left, right


def support_polygon_margin(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData
) -> float:
    """Signed distance from the ground-projected CoM to the support polygon [m].

    Positive = inside. Uses the convex hull of active foot contact points, so
    it reflects the polygon the simulator is actually standing on rather than a
    nominal footprint. Returns -inf when there is no contact (airborne).
    """
    points = [
        mj_data.contact[cid].pos[:2].copy()
        for cid in range(mj_data.ncon)
        if _is_foot_contact(mj_model, mj_data.contact[cid])
    ]
    if len(points) < 3:
        return float("-inf")

    from scipy.spatial import ConvexHull  # local import: optional dependency

    pts = np.array(points)
    com_xy = mj_data.subtree_com[0][:2]
    try:
        hull = ConvexHull(pts)
    except Exception:
        return float("-inf")

    # Hull equations: A·x + b <= 0 inside. Margin = -max(A·x + b).
    margins = hull.equations[:, :2] @ com_xy + hull.equations[:, 2]
    return float(-np.max(margins))


def measured_zmp(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData
) -> np.ndarray | None:
    """Zero-moment point on flat ground from live contact forces [m].

    For point contacts on a horizontal plane the ZMP reduces to the normal
    -force-weighted mean of the contact positions::

        p_zmp = Σ f_z,i · p_i / Σ f_z,i

    Returns None when no foot contact carries load (flight phase). Uses
    MuJoCo's own contact forces rather than the QP's predicted wrenches on
    purpose: the gate must measure what the simulator did, not what the
    controller intended.
    """
    force = np.zeros(6)
    total = 0.0
    weighted = np.zeros(2)
    for cid in range(mj_data.ncon):
        contact = mj_data.contact[cid]
        if not _is_foot_contact(mj_model, contact):
            continue
        mujoco.mj_contactForce(mj_model, mj_data, cid, force)
        # contact.frame is the contact frame rotation, row-major; the first row
        # is the normal in world coordinates.
        normal = contact.frame[:3]
        f_z = abs(float(force[0] * normal[2]))
        if f_z <= 1e-9:
            continue
        total += f_z
        weighted += f_z * contact.pos[:2]
    if total <= 1e-6:
        return None
    return weighted / total


def point_in_support_polygon(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData, point_xy: np.ndarray
) -> float:
    """Signed distance from `point_xy` to the foot support polygon [m].

    Positive inside. Mirrors `support_polygon_margin` but for an arbitrary
    point (the ZMP), so the gate can score both.
    """
    points = [
        mj_data.contact[cid].pos[:2].copy()
        for cid in range(mj_data.ncon)
        if _is_foot_contact(mj_model, mj_data.contact[cid])
    ]
    if len(points) < 3:
        return float("-inf")

    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(np.array(points))
    except Exception:
        return float("-inf")
    margins = hull.equations[:, :2] @ np.asarray(point_xy, dtype=float) + hull.equations[:, 2]
    return float(-np.max(margins))


def _is_foot_contact(mj_model: mujoco.MjModel, contact) -> bool:
    """True if a contact involves an ankle/foot body."""
    for gid in (contact.geom1, contact.geom2):
        body = mj_model.geom_bodyid[gid]
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if "ankle" in name:
            return True
    return False


def dense_mass_matrix(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData
) -> np.ndarray:
    """Dense MuJoCo mass matrix (nv×nv), MuJoCo-version agnostic.

    MuJoCo >= 3.11 removed ``MjData.qM`` and re-signatured ``mj_fullM``.
    """
    dense = np.zeros((mj_model.nv, mj_model.nv))
    qM = getattr(mj_data, "qM", None)
    if qM is None:
        mujoco.mj_fullM(mj_model, mj_data, dense)
    else:
        mujoco.mj_fullM(mj_model, dense, qM)
    return dense
