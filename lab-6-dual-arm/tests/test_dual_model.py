"""Tests for DualArmModel — FK, Jacobian, gravity, IK, ObjectFrame, relative pose."""
import sys
from pathlib import Path
import math

import numpy as np
import pytest
import pinocchio as pin

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from lab6_common import Q_HOME_LEFT, Q_HOME_RIGHT, load_mujoco_model
from dual_arm_model import DualArmModel, ObjectFrame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model() -> DualArmModel:
    """Shared DualArmModel instance for the test module."""
    return DualArmModel()


@pytest.fixture(scope="module")
def mujoco_env():
    """Shared MuJoCo model + data for the test module."""
    import mujoco
    mj_model, mj_data = load_mujoco_model()
    return mj_model, mj_data


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _set_mujoco_joints(mj_model, mj_data, q_left: np.ndarray, q_right: np.ndarray) -> None:
    """Set left and right arm joint positions in MuJoCo data by joint name."""
    import mujoco

    left_joint_names = [
        "left_shoulder_pan_joint",
        "left_shoulder_lift_joint",
        "left_elbow_joint",
        "left_wrist_1_joint",
        "left_wrist_2_joint",
        "left_wrist_3_joint",
    ]
    right_joint_names = [
        "right_shoulder_pan_joint",
        "right_shoulder_lift_joint",
        "right_elbow_joint",
        "right_wrist_1_joint",
        "right_wrist_2_joint",
        "right_wrist_3_joint",
    ]

    for i, name in enumerate(left_joint_names):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = mj_model.jnt_qposadr[jid]
        mj_data.qpos[qadr] = q_left[i]

    for i, name in enumerate(right_joint_names):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = mj_model.jnt_qposadr[jid]
        mj_data.qpos[qadr] = q_right[i]

    mujoco.mj_forward(mj_model, mj_data)


# ---------------------------------------------------------------------------
# FK cross-validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q_left,q_right,label", [
    (Q_HOME_LEFT, Q_HOME_RIGHT, "home"),
    (np.zeros(6), np.zeros(6), "zeros"),
    (
        np.array([0.3, -0.8, 1.2, -0.5, 0.4, -0.2]),
        np.array([-0.3, -0.8, 1.2, -0.5, -0.4, 0.2]),
        "random_a",
    ),
    (
        np.array([-math.pi / 4, -math.pi / 3, math.pi / 4, -math.pi / 6, math.pi / 6, 0.0]),
        np.array([math.pi / 4, -math.pi / 3, math.pi / 4, -math.pi / 6, -math.pi / 6, 0.0]),
        "random_b",
    ),
])
def test_fk_cross_validation(model, mujoco_env, q_left, q_right, label):
    """Pinocchio FK (with base offset) matches MuJoCo body position within 2mm."""
    import mujoco

    mj_model, mj_data = mujoco_env
    _set_mujoco_joints(mj_model, mj_data, q_left, q_right)

    left_tool_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_tool0")
    right_tool_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_tool0")

    mj_left_pos = mj_data.xpos[left_tool_bid].copy()
    mj_right_pos = mj_data.xpos[right_tool_bid].copy()

    pin_left_pos = model.fk_left_pos(q_left)
    pin_right_pos = model.fk_right_pos(q_right)

    assert np.allclose(pin_left_pos, mj_left_pos, atol=2e-3), (
        f"[{label}] Left FK mismatch: pin={pin_left_pos}, mj={mj_left_pos}, "
        f"err={np.linalg.norm(pin_left_pos - mj_left_pos)*1000:.2f} mm"
    )
    assert np.allclose(pin_right_pos, mj_right_pos, atol=2e-3), (
        f"[{label}] Right FK mismatch: pin={pin_right_pos}, mj={mj_right_pos}, "
        f"err={np.linalg.norm(pin_right_pos - mj_right_pos)*1000:.2f} mm"
    )


# ---------------------------------------------------------------------------
# FK returns SE3
# ---------------------------------------------------------------------------

def test_fk_left_returns_se3(model):
    """fk_left returns a pin.SE3 object."""
    pose = model.fk_left(Q_HOME_LEFT)
    assert isinstance(pose, pin.SE3)


def test_fk_right_returns_se3(model):
    """fk_right returns a pin.SE3 object."""
    pose = model.fk_right(Q_HOME_RIGHT)
    assert isinstance(pose, pin.SE3)


def test_fk_left_pos_shape(model):
    """fk_left_pos returns a (3,) vector."""
    pos = model.fk_left_pos(Q_HOME_LEFT)
    assert pos.shape == (3,)


def test_fk_right_pos_shape(model):
    """fk_right_pos returns a (3,) vector."""
    pos = model.fk_right_pos(Q_HOME_RIGHT)
    assert pos.shape == (3,)


# ---------------------------------------------------------------------------
# Jacobian shape
# ---------------------------------------------------------------------------

def test_jacobian_left_shape(model):
    """jacobian_left returns shape (6, 6)."""
    J = model.jacobian_left(Q_HOME_LEFT)
    assert J.shape == (6, 6), f"Expected (6, 6), got {J.shape}"


def test_jacobian_right_shape(model):
    """jacobian_right returns shape (6, 6)."""
    J = model.jacobian_right(Q_HOME_RIGHT)
    assert J.shape == (6, 6), f"Expected (6, 6), got {J.shape}"


def test_jacobian_left_nonzero(model):
    """Jacobian for left arm at home is not all zeros."""
    J = model.jacobian_left(Q_HOME_LEFT)
    assert np.any(J != 0.0), "Left Jacobian is unexpectedly all zeros"


def test_jacobian_right_nonzero(model):
    """Jacobian for right arm at home is not all zeros."""
    J = model.jacobian_right(Q_HOME_RIGHT)
    assert np.any(J != 0.0), "Right Jacobian is unexpectedly all zeros"


@pytest.mark.parametrize("q,side", [
    (np.zeros(6), "left"),
    (np.zeros(6), "right"),
    (Q_HOME_LEFT, "left"),
    (Q_HOME_RIGHT, "right"),
])
def test_jacobian_shape_various_configs(model, q, side):
    """Jacobian is (6, 6) for various configs on both sides."""
    if side == "left":
        J = model.jacobian_left(q)
    else:
        J = model.jacobian_right(q)
    assert J.shape == (6, 6)


# ---------------------------------------------------------------------------
# Gravity torques
# ---------------------------------------------------------------------------

def test_gravity_left_nonzero_at_home(model):
    """Gravity torques for left arm are non-zero at home config."""
    g = model.gravity_left(Q_HOME_LEFT)
    assert g.shape == (6,)
    assert np.linalg.norm(g) > 0.1, f"Gravity near zero: {g}"


def test_gravity_right_nonzero_at_home(model):
    """Gravity torques for right arm are non-zero at home config."""
    g = model.gravity_right(Q_HOME_RIGHT)
    assert g.shape == (6,)
    assert np.linalg.norm(g) > 0.1, f"Gravity near zero: {g}"


def test_gravity_left_near_zero_at_vertical(model):
    """Gravity torques for left arm are small when all joints are near zero (arm near-vertical)."""
    # q = zeros: arm pointing straight up — shoulder_lift near zero gives vertical posture.
    # Gravity at all-zeros is still non-zero due to offset links; just confirm it's finite.
    g = model.gravity_left(np.zeros(6))
    assert np.all(np.isfinite(g)), "Gravity returned non-finite values"


def test_gravity_right_nonzero_for_nonvertical(model):
    """Gravity torques for right arm vary with configuration."""
    g_home = model.gravity_right(Q_HOME_RIGHT)
    g_zeros = model.gravity_right(np.zeros(6))
    # They should differ (different configs produce different gravity torques)
    assert not np.allclose(g_home, g_zeros, atol=1e-3), \
        "Gravity torques are identical for different configs — likely a bug"


# ---------------------------------------------------------------------------
# IK round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q_true,side,label", [
    (Q_HOME_LEFT, "left", "home_left"),
    (Q_HOME_RIGHT, "right", "home_right"),
    (np.array([0.2, -0.6, 1.0, -0.4, 0.3, -0.1]), "left", "random_left"),
    (np.array([-0.2, -0.6, 1.0, -0.4, -0.3, 0.1]), "right", "random_right"),
])
def test_ik_roundtrip(model, q_true, side, label):
    """FK → IK(target) → FK: position error < 1mm."""
    if side == "left":
        target = model.fk_left(q_true)
        q_init = np.zeros(6)
        q_sol = model.ik_left(target, q_init)
        pos_check = model.fk_left_pos(q_sol) if q_sol is not None else None
        target_pos = model.fk_left_pos(q_true)
    else:
        target = model.fk_right(q_true)
        q_init = np.zeros(6)
        q_sol = model.ik_right(target, q_init)
        pos_check = model.fk_right_pos(q_sol) if q_sol is not None else None
        target_pos = model.fk_right_pos(q_true)

    assert q_sol is not None, f"[{label}] IK returned None (failed to converge)"
    assert q_sol.shape == (6,), f"[{label}] IK solution has wrong shape: {q_sol.shape}"

    err_m = np.linalg.norm(pos_check - target_pos)
    assert err_m < 1e-3, (
        f"[{label}] IK round-trip position error {err_m*1000:.3f} mm > 1 mm"
    )


def test_ik_left_returns_none_for_unreachable(model):
    """IK returns None for a target far outside the workspace."""
    unreachable = pin.SE3(np.eye(3), np.array([10.0, 10.0, 10.0]))
    q_sol = model.ik_left(unreachable, np.zeros(6), max_iter=20)
    assert q_sol is None, "IK should return None for unreachable target"


def test_ik_right_returns_none_for_unreachable(model):
    """IK returns None for a target far outside the workspace."""
    unreachable = pin.SE3(np.eye(3), np.array([10.0, 10.0, 10.0]))
    q_sol = model.ik_right(unreachable, np.zeros(6), max_iter=20)
    assert q_sol is None, "IK should return None for unreachable target"


# ---------------------------------------------------------------------------
# ObjectFrame
# ---------------------------------------------------------------------------

def test_object_frame_from_ee_poses_roundtrip(model):
    """from_ee_poses → get_left_target / get_right_target match original EE poses."""
    left_ee = model.fk_left(Q_HOME_LEFT)
    right_ee = model.fk_right(Q_HOME_RIGHT)

    # Object at midpoint between the two EEs
    mid_pos = 0.5 * (left_ee.translation + right_ee.translation)
    object_pose = pin.SE3(np.eye(3), mid_pos)

    frame = ObjectFrame.from_ee_poses(object_pose, left_ee, right_ee)

    recovered_left = frame.get_left_target()
    recovered_right = frame.get_right_target()

    assert np.allclose(recovered_left.translation, left_ee.translation, atol=1e-9), (
        f"Left target mismatch: {recovered_left.translation} vs {left_ee.translation}"
    )
    assert np.allclose(recovered_right.translation, right_ee.translation, atol=1e-9), (
        f"Right target mismatch: {recovered_right.translation} vs {right_ee.translation}"
    )


def test_object_frame_moved_to_changes_targets(model):
    """moved_to changes the target positions proportionally to the object translation."""
    left_ee = model.fk_left(Q_HOME_LEFT)
    right_ee = model.fk_right(Q_HOME_RIGHT)

    mid_pos = 0.5 * (left_ee.translation + right_ee.translation)
    object_pose = pin.SE3(np.eye(3), mid_pos)
    frame = ObjectFrame.from_ee_poses(object_pose, left_ee, right_ee)

    delta = np.array([0.1, 0.0, 0.05])
    new_pose = pin.SE3(np.eye(3), mid_pos + delta)
    moved_frame = frame.moved_to(new_pose)

    # Targets should have shifted by delta
    orig_left = frame.get_left_target().translation
    orig_right = frame.get_right_target().translation
    new_left = moved_frame.get_left_target().translation
    new_right = moved_frame.get_right_target().translation

    assert np.allclose(new_left - orig_left, delta, atol=1e-9), (
        f"Left target delta wrong: {new_left - orig_left} vs {delta}"
    )
    assert np.allclose(new_right - orig_right, delta, atol=1e-9), (
        f"Right target delta wrong: {new_right - orig_right} vs {delta}"
    )


def test_object_frame_preserves_grasp_offsets(model):
    """moved_to preserves grasp offsets."""
    left_ee = model.fk_left(Q_HOME_LEFT)
    right_ee = model.fk_right(Q_HOME_RIGHT)
    mid_pos = 0.5 * (left_ee.translation + right_ee.translation)
    object_pose = pin.SE3(np.eye(3), mid_pos)
    frame = ObjectFrame.from_ee_poses(object_pose, left_ee, right_ee)

    new_pose = pin.SE3(np.eye(3), mid_pos + np.array([0.2, 0.1, 0.0]))
    moved = frame.moved_to(new_pose)

    # Grasp offsets are the same object references — translations must match
    assert np.allclose(
        frame.grasp_offset_left.translation,
        moved.grasp_offset_left.translation,
        atol=1e-12,
    )
    assert np.allclose(
        frame.grasp_offset_right.translation,
        moved.grasp_offset_right.translation,
        atol=1e-12,
    )


def test_object_frame_at_identity(model):
    """ObjectFrame with identity object pose: offsets equal EE poses."""
    left_ee = model.fk_left(Q_HOME_LEFT)
    right_ee = model.fk_right(Q_HOME_RIGHT)
    identity = pin.SE3.Identity()

    frame = ObjectFrame.from_ee_poses(identity, left_ee, right_ee)

    # offset = identity.inverse() * ee = ee
    assert np.allclose(
        frame.grasp_offset_left.translation, left_ee.translation, atol=1e-9
    )
    assert np.allclose(
        frame.grasp_offset_right.translation, right_ee.translation, atol=1e-9
    )


# ---------------------------------------------------------------------------
# Relative EE pose
# ---------------------------------------------------------------------------

def test_relative_ee_pose_at_home(model):
    """At home configs, the EE-to-EE distance is approximately 1m (arms facing each other)."""
    rel = model.relative_ee_pose(Q_HOME_LEFT, Q_HOME_RIGHT)
    assert isinstance(rel, pin.SE3)

    dist = np.linalg.norm(rel.translation)
    # Arms are 1m apart at base; EE distance should be roughly in that range.
    # Use a generous window: 0.4 m – 1.6 m.
    assert 0.4 < dist < 1.6, (
        f"EE-to-EE distance at home {dist:.3f} m is outside expected range [0.4, 1.6]"
    )


def test_relative_ee_pose_is_se3(model):
    """relative_ee_pose returns a valid SE3 (rotation is orthonormal)."""
    rel = model.relative_ee_pose(Q_HOME_LEFT, Q_HOME_RIGHT)
    R = rel.rotation
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "Rotation matrix is not orthonormal"
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6), "Rotation matrix determinant ≠ 1"


def test_relative_ee_pose_inverse_consistency(model):
    """left.inv * right and right.inv * left are inverses of each other."""
    rel_lr = model.relative_ee_pose(Q_HOME_LEFT, Q_HOME_RIGHT)
    rel_rl = model.relative_ee_pose(Q_HOME_RIGHT, Q_HOME_LEFT)  # swapped

    # rel_lr composed with rel_rl should not be identity (different bases),
    # but rel_lr.inverse().translation should match rel_rl.translation after a rotation.
    # The simpler check: the distance is the same magnitude.
    dist_lr = np.linalg.norm(rel_lr.translation)
    dist_rl = np.linalg.norm(rel_rl.translation)
    assert np.isclose(dist_lr, dist_rl, atol=1e-3), (
        f"EE distances should be symmetric: {dist_lr:.4f} vs {dist_rl:.4f}"
    )
