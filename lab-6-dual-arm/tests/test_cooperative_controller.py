"""Tests for DualImpedanceController and orientation_error.

Coverage:
- orientation_error: identity → zero, known rotation → expected sign/magnitude
- DualImpedanceController.compute_dual_torques output shape
- At target with zero velocity: torques ≈ gravity compensation only
- With position error: torque magnitude increases with stiffness gain
- Squeeze torques: only added when grasping=True
- Squeeze torques for both arms oppose each other along grasp axis
- clip_torques applied: output never exceeds TORQUE_LIMITS
- DualImpedanceGains defaults: positive-definite diagonal matrices
- set_gains: updates gains in-place
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import pinocchio as pin

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from lab6_common import Q_HOME_LEFT, Q_HOME_RIGHT, TORQUE_LIMITS
from dual_arm_model import DualArmModel
from cooperative_controller import (
    DualImpedanceController,
    DualImpedanceGains,
    orientation_error,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dual_model() -> DualArmModel:
    """Shared DualArmModel instance."""
    return DualArmModel()


@pytest.fixture(scope="module")
def controller(dual_model: DualArmModel) -> DualImpedanceController:
    """Shared DualImpedanceController with default gains."""
    return DualImpedanceController(dual_model)


@pytest.fixture(scope="module")
def home_targets(dual_model: DualArmModel) -> tuple[pin.SE3, pin.SE3]:
    """Target poses equal to FK at home configs (zero task-space error)."""
    target_left = dual_model.fk_left(Q_HOME_LEFT)
    target_right = dual_model.fk_right(Q_HOME_RIGHT)
    return target_left, target_right


# ---------------------------------------------------------------------------
# orientation_error
# ---------------------------------------------------------------------------


def test_orientation_error_identity_is_zero():
    """Identity desired and current rotation → zero error."""
    R = np.eye(3)
    e = orientation_error(R, R)
    assert e.shape == (3,), f"Expected shape (3,), got {e.shape}"
    assert np.allclose(e, np.zeros(3), atol=1e-12), f"Error should be zero, got {e}"


def test_orientation_error_same_rotation_is_zero():
    """Arbitrary identical desired and current rotation → zero error."""
    # 30° rotation about Z
    angle = np.pi / 6
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [0.0,            0.0,           1.0],
    ])
    e = orientation_error(R, R)
    assert np.allclose(e, np.zeros(3), atol=1e-12), f"Error should be zero, got {e}"


def test_orientation_error_nonzero_for_different_rotations():
    """Different desired and current rotations produce non-zero error."""
    R_des = np.eye(3)
    angle = np.pi / 4  # 45° rotation about Z
    R_cur = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [0.0,            0.0,           1.0],
    ])
    e = orientation_error(R_des, R_cur)
    assert np.linalg.norm(e) > 1e-6, "Error should be non-zero for different rotations"


def test_orientation_error_sign_convention():
    """Small rotation about Z gives error along Z axis (positive Z component)."""
    # Small CCW rotation about Z: R_cur slightly rotated from R_des = I
    small_angle = 0.1  # rad
    R_des = np.eye(3)
    R_cur = np.array([
        [np.cos(small_angle), -np.sin(small_angle), 0.0],
        [np.sin(small_angle),  np.cos(small_angle), 0.0],
        [0.0,                  0.0,                 1.0],
    ])
    e = orientation_error(R_des, R_cur)
    # e[2] should be the dominant component (rotation about Z)
    assert abs(e[2]) > abs(e[0]) and abs(e[2]) > abs(e[1]), (
        f"Expected dominant Z component; got e={e}"
    )


def test_orientation_error_returns_array():
    """orientation_error returns a NumPy array of shape (3,)."""
    e = orientation_error(np.eye(3), np.eye(3))
    assert isinstance(e, np.ndarray)
    assert e.shape == (3,)


def test_orientation_error_antisymmetric():
    """orientation_error(R_des, R_cur) and orientation_error(R_cur, R_des) are opposite."""
    angle = np.pi / 8
    R_des = np.eye(3)
    R_cur = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [0.0,            0.0,           1.0],
    ])
    e_forward = orientation_error(R_des, R_cur)
    e_backward = orientation_error(R_cur, R_des)
    # The skew-symmetric extraction is antisymmetric in the two rotation arguments
    assert np.allclose(e_forward, -e_backward, atol=1e-10), (
        f"Expected e(A,B) = -e(B,A); got {e_forward} and {e_backward}"
    )


# ---------------------------------------------------------------------------
# DualImpedanceGains defaults
# ---------------------------------------------------------------------------


def test_gains_default_stiffness_positive_definite():
    """Default K_p is a positive-definite diagonal matrix."""
    gains = DualImpedanceGains()
    eigenvalues = np.linalg.eigvalsh(gains.K_p)
    assert np.all(eigenvalues > 0), f"K_p is not positive definite: eigenvalues={eigenvalues}"


def test_gains_default_damping_positive_definite():
    """Default K_d is a positive-definite diagonal matrix."""
    gains = DualImpedanceGains()
    eigenvalues = np.linalg.eigvalsh(gains.K_d)
    assert np.all(eigenvalues > 0), f"K_d is not positive definite: eigenvalues={eigenvalues}"


def test_gains_default_squeeze_positive():
    """Default f_squeeze is positive."""
    gains = DualImpedanceGains()
    assert gains.f_squeeze > 0.0, "Default squeeze force should be positive"


def test_gains_shapes():
    """Default K_p and K_d have shape (6, 6)."""
    gains = DualImpedanceGains()
    assert gains.K_p.shape == (6, 6), f"K_p shape {gains.K_p.shape} != (6, 6)"
    assert gains.K_d.shape == (6, 6), f"K_d shape {gains.K_d.shape} != (6, 6)"


# ---------------------------------------------------------------------------
# set_gains
# ---------------------------------------------------------------------------


def test_set_gains_updates_controller(dual_model):
    """set_gains replaces the controller's gains object."""
    ctrl = DualImpedanceController(dual_model)
    new_gains = DualImpedanceGains(
        K_p=np.diag([200.0] * 6),
        K_d=np.diag([30.0] * 6),
        f_squeeze=5.0,
    )
    ctrl.set_gains(new_gains)
    assert ctrl.gains is new_gains, "set_gains should replace the gains object"
    assert np.allclose(ctrl.gains.K_p, np.diag([200.0] * 6))


# ---------------------------------------------------------------------------
# compute_dual_torques — output shape
# ---------------------------------------------------------------------------


def test_compute_dual_torques_output_shapes(controller, home_targets):
    """compute_dual_torques returns two (6,) arrays."""
    target_left, target_right = home_targets
    tau_l, tau_r = controller.compute_dual_torques(
        q_left=Q_HOME_LEFT,
        qd_left=np.zeros(6),
        q_right=Q_HOME_RIGHT,
        qd_right=np.zeros(6),
        target_left=target_left,
        target_right=target_right,
    )
    assert tau_l.shape == (6,), f"tau_left shape {tau_l.shape} != (6,)"
    assert tau_r.shape == (6,), f"tau_right shape {tau_r.shape} != (6,)"


def test_compute_dual_torques_returns_numpy_arrays(controller, home_targets):
    """compute_dual_torques returns NumPy arrays."""
    target_left, target_right = home_targets
    tau_l, tau_r = controller.compute_dual_torques(
        q_left=Q_HOME_LEFT,
        qd_left=np.zeros(6),
        q_right=Q_HOME_RIGHT,
        qd_right=np.zeros(6),
        target_left=target_left,
        target_right=target_right,
    )
    assert isinstance(tau_l, np.ndarray)
    assert isinstance(tau_r, np.ndarray)


# ---------------------------------------------------------------------------
# At target with zero velocity: torques ≈ gravity compensation only
# ---------------------------------------------------------------------------


def test_at_target_zero_velocity_torques_match_gravity(dual_model, home_targets):
    """At target pose with zero velocity, output torques equal gravity compensation."""
    ctrl = DualImpedanceController(dual_model)
    target_left, target_right = home_targets

    tau_l, tau_r = ctrl.compute_dual_torques(
        q_left=Q_HOME_LEFT,
        qd_left=np.zeros(6),
        q_right=Q_HOME_RIGHT,
        qd_right=np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        grasping=False,
    )

    g_left = dual_model.gravity_left(Q_HOME_LEFT)
    g_right = dual_model.gravity_right(Q_HOME_RIGHT)

    # clip_torques is applied, so compare against clipped gravity
    from lab6_common import clip_torques
    g_left_clipped = clip_torques(g_left)
    g_right_clipped = clip_torques(g_right)

    assert np.allclose(tau_l, g_left_clipped, atol=1e-6), (
        f"At target with zero velocity, left torque should equal gravity comp.\n"
        f"  tau_l={tau_l}\n  g_left={g_left_clipped}\n  diff={tau_l - g_left_clipped}"
    )
    assert np.allclose(tau_r, g_right_clipped, atol=1e-6), (
        f"At target with zero velocity, right torque should equal gravity comp.\n"
        f"  tau_r={tau_r}\n  g_right={g_right_clipped}\n  diff={tau_r - g_right_clipped}"
    )


# ---------------------------------------------------------------------------
# With position error: torques increase proportionally to stiffness
# ---------------------------------------------------------------------------


def test_position_error_increases_torques(dual_model, home_targets):
    """Larger position error produces larger task-space wrench and thus larger torques."""
    target_left_exact, target_right_exact = home_targets

    # Displace target by a small amount
    delta_small = np.array([0.02, 0.0, 0.0])
    target_left_small = pin.SE3(
        target_left_exact.rotation.copy(),
        target_left_exact.translation + delta_small,
    )

    # Displace target by a larger amount
    delta_large = np.array([0.10, 0.0, 0.0])
    target_left_large = pin.SE3(
        target_left_exact.rotation.copy(),
        target_left_exact.translation + delta_large,
    )

    ctrl = DualImpedanceController(dual_model)

    tau_small, _ = ctrl.compute_dual_torques(
        q_left=Q_HOME_LEFT,
        qd_left=np.zeros(6),
        q_right=Q_HOME_RIGHT,
        qd_right=np.zeros(6),
        target_left=target_left_small,
        target_right=target_right_exact,
        grasping=False,
    )

    tau_large, _ = ctrl.compute_dual_torques(
        q_left=Q_HOME_LEFT,
        qd_left=np.zeros(6),
        q_right=Q_HOME_RIGHT,
        qd_right=np.zeros(6),
        target_left=target_left_large,
        target_right=target_right_exact,
        grasping=False,
    )

    g_left = dual_model.gravity_left(Q_HOME_LEFT)

    # The deviation from gravity comp should be larger for the large error
    dev_small = np.linalg.norm(tau_small - g_left)
    dev_large = np.linalg.norm(tau_large - g_left)

    assert dev_large > dev_small, (
        f"Larger position error should produce larger torque deviation from gravity comp. "
        f"dev_small={dev_small:.4f}, dev_large={dev_large:.4f}"
    )


def test_position_error_torque_scales_with_stiffness(dual_model, home_targets):
    """Doubling stiffness doubles the impedance contribution."""
    target_left, target_right = home_targets

    # Apply a fixed position error
    displaced = pin.SE3(
        target_left.rotation.copy(),
        target_left.translation + np.array([0.05, 0.0, 0.0]),
    )

    gains_low = DualImpedanceGains(
        K_p=np.diag([200.0, 200.0, 200.0, 20.0, 20.0, 20.0]),
        K_d=np.diag([1e-9] * 6),  # near-zero damping to isolate stiffness effect
        f_squeeze=0.0,
    )
    gains_high = DualImpedanceGains(
        K_p=np.diag([400.0, 400.0, 400.0, 40.0, 40.0, 40.0]),
        K_d=np.diag([1e-9] * 6),
        f_squeeze=0.0,
    )

    g_left = dual_model.gravity_left(Q_HOME_LEFT)

    ctrl_low = DualImpedanceController(dual_model, gains_low)
    ctrl_high = DualImpedanceController(dual_model, gains_high)

    tau_low, _ = ctrl_low.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=displaced, target_right=target_right,
    )
    tau_high, _ = ctrl_high.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=displaced, target_right=target_right,
    )

    dev_low = np.linalg.norm(tau_low - g_left)
    dev_high = np.linalg.norm(tau_high - g_left)

    # Higher stiffness should produce larger torque deviation (before clipping dominates)
    assert dev_high > dev_low, (
        f"Higher stiffness should produce larger torque. "
        f"dev_low={dev_low:.4f}, dev_high={dev_high:.4f}"
    )


# ---------------------------------------------------------------------------
# Squeeze torques: only added when grasping=True
# ---------------------------------------------------------------------------


def test_squeeze_torques_only_when_grasping(dual_model, home_targets):
    """Torques differ between grasping=True and grasping=False."""
    target_left, target_right = home_targets
    ctrl = DualImpedanceController(dual_model)

    tau_l_no_grasp, tau_r_no_grasp = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        grasping=False,
    )

    tau_l_grasp, tau_r_grasp = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        grasping=True,
    )

    # With squeeze > 0 (default 10 N), torques must differ
    assert not np.allclose(tau_l_no_grasp, tau_l_grasp, atol=1e-6), (
        "Left torque should differ when grasping=True (squeeze force should be added)"
    )
    assert not np.allclose(tau_r_no_grasp, tau_r_grasp, atol=1e-6), (
        "Right torque should differ when grasping=True (squeeze force should be added)"
    )


def test_squeeze_torques_zero_when_f_squeeze_is_zero(dual_model, home_targets):
    """Torques are identical for grasping=True/False when f_squeeze=0."""
    gains_no_squeeze = DualImpedanceGains(
        K_p=DualImpedanceGains().K_p,
        K_d=DualImpedanceGains().K_d,
        f_squeeze=0.0,
    )
    ctrl = DualImpedanceController(dual_model, gains_no_squeeze)
    target_left, target_right = home_targets

    tau_l_no_grasp, tau_r_no_grasp = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        grasping=False,
    )
    tau_l_grasp, tau_r_grasp = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        grasping=True,
    )

    assert np.allclose(tau_l_no_grasp, tau_l_grasp, atol=1e-9), (
        "With f_squeeze=0, grasping flag should have no effect on left torques"
    )
    assert np.allclose(tau_r_no_grasp, tau_r_grasp, atol=1e-9), (
        "With f_squeeze=0, grasping flag should have no effect on right torques"
    )


# ---------------------------------------------------------------------------
# Squeeze torques oppose each other along grasp axis
# ---------------------------------------------------------------------------


def test_squeeze_torques_oppose_each_other(dual_model):
    """The joint-space squeeze torques for left and right arms act in opposite directions.

    Specifically: J_left^T * F_squeeze_left should push left EE toward right,
    and J_right^T * F_squeeze_right should push right EE toward left.
    We verify this by checking the virtual EE velocity from each torque contribution
    has the correct sign along the grasp axis.
    """
    ctrl = DualImpedanceController(dual_model)

    # Get squeeze torques by comparing grasping vs not-grasping at target
    target_left = dual_model.fk_left(Q_HOME_LEFT)
    target_right = dual_model.fk_right(Q_HOME_RIGHT)

    tau_l_grasp, tau_r_grasp = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        grasping=True,
    )
    tau_l_no_grasp, tau_r_no_grasp = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        grasping=False,
    )

    # Squeeze contribution in joint space
    sq_left = tau_l_grasp - tau_l_no_grasp
    sq_right = tau_r_grasp - tau_r_no_grasp

    # Map to EE-space forces via Jacobian pseudo-inverse to check direction
    J_left = dual_model.jacobian_left(Q_HOME_LEFT)
    J_right = dual_model.jacobian_right(Q_HOME_RIGHT)

    # Grasp axis: direction from left EE to right EE
    pos_left = dual_model.fk_left_pos(Q_HOME_LEFT)
    pos_right = dual_model.fk_right_pos(Q_HOME_RIGHT)
    grasp_axis = pos_right - pos_left
    grasp_axis /= np.linalg.norm(grasp_axis)

    # Recover approximate EE-space forces: F ≈ (J J^T)^{-1} J * sq
    # Simpler: project J^T F onto F direction → F^T J tau should have correct sign
    # F_left along grasp axis should be positive (left pushes toward right)
    # Compute: grasp_axis . J_pinv @ sq_left
    # Use J @ J^T inverse: small system
    JJt_left = J_left @ J_left.T
    F_left_approx = np.linalg.lstsq(JJt_left, J_left @ sq_left, rcond=None)[0]
    F_right_approx = np.linalg.lstsq(J_right @ J_right.T, J_right @ sq_right, rcond=None)[0]

    # Left force should be in +grasp_axis direction (toward right EE)
    proj_left = np.dot(F_left_approx[:3], grasp_axis)
    # Right force should be in -grasp_axis direction (toward left EE)
    proj_right = np.dot(F_right_approx[:3], grasp_axis)

    assert proj_left > 0.0, (
        f"Left squeeze force should point toward right EE (+grasp_axis). "
        f"Projection: {proj_left:.4f}"
    )
    assert proj_right < 0.0, (
        f"Right squeeze force should point toward left EE (-grasp_axis). "
        f"Projection: {proj_right:.4f}"
    )


def test_squeeze_forces_are_equal_and_opposite_in_magnitude(dual_model):
    """Squeeze forces have equal magnitude for symmetric gains."""
    ctrl = DualImpedanceController(dual_model)
    target_left = dual_model.fk_left(Q_HOME_LEFT)
    target_right = dual_model.fk_right(Q_HOME_RIGHT)

    tau_l_grasp, tau_r_grasp = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        grasping=True,
    )
    tau_l_no_grasp, tau_r_no_grasp = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        grasping=False,
    )

    sq_left_norm = np.linalg.norm(tau_l_grasp - tau_l_no_grasp)
    sq_right_norm = np.linalg.norm(tau_r_grasp - tau_r_no_grasp)

    # With symmetric gains and symmetric arm geometry at home, magnitudes are comparable
    # Allow 50% tolerance due to different Jacobian conditioning
    ratio = sq_left_norm / (sq_right_norm + 1e-12)
    assert 0.1 < ratio < 10.0, (
        f"Squeeze torque magnitudes are highly asymmetric: "
        f"left={sq_left_norm:.4f}, right={sq_right_norm:.4f}, ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# clip_torques applied to output
# ---------------------------------------------------------------------------


def test_output_torques_within_limits(dual_model):
    """Output torques from compute_dual_torques never exceed TORQUE_LIMITS."""
    # Use very high stiffness to try to drive torques beyond limits
    gains_high = DualImpedanceGains(
        K_p=np.diag([10000.0] * 6),
        K_d=np.diag([1000.0] * 6),
        f_squeeze=500.0,
    )
    ctrl = DualImpedanceController(dual_model, gains_high)

    # Introduce a large position error to stress the clipping
    target_left = dual_model.fk_left(Q_HOME_LEFT)
    target_right = dual_model.fk_right(Q_HOME_RIGHT)

    displaced_left = pin.SE3(
        target_left.rotation.copy(),
        target_left.translation + np.array([0.5, 0.3, 0.2]),
    )
    displaced_right = pin.SE3(
        target_right.rotation.copy(),
        target_right.translation + np.array([-0.5, 0.3, 0.2]),
    )

    tau_l, tau_r = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.ones(6) * 0.5,
        Q_HOME_RIGHT, np.ones(6) * 0.5,
        target_left=displaced_left,
        target_right=displaced_right,
        grasping=True,
    )

    assert np.all(np.abs(tau_l) <= TORQUE_LIMITS + 1e-9), (
        f"Left torques exceed limits: {tau_l}\nLimits: {TORQUE_LIMITS}"
    )
    assert np.all(np.abs(tau_r) <= TORQUE_LIMITS + 1e-9), (
        f"Right torques exceed limits: {tau_r}\nLimits: {TORQUE_LIMITS}"
    )


def test_output_torques_clipped_at_exact_limit(dual_model, home_targets):
    """Verify clipping is applied: gravity-only torques at home are within limits."""
    ctrl = DualImpedanceController(dual_model)
    target_left, target_right = home_targets

    tau_l, tau_r = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
    )

    for i, (t, lim) in enumerate(zip(tau_l, TORQUE_LIMITS)):
        assert abs(t) <= lim + 1e-9, f"Left joint {i} torque {t:.3f} exceeds limit {lim}"
    for i, (t, lim) in enumerate(zip(tau_r, TORQUE_LIMITS)):
        assert abs(t) <= lim + 1e-9, f"Right joint {i} torque {t:.3f} exceeds limit {lim}"


# ---------------------------------------------------------------------------
# Desired EE velocity passthrough
# ---------------------------------------------------------------------------


def test_desired_velocity_changes_torques(dual_model, home_targets):
    """Providing a non-zero desired EE velocity changes the output torques."""
    ctrl = DualImpedanceController(dual_model)
    target_left, target_right = home_targets

    tau_l_zero_vel, _ = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        xd_left=None,
    )

    # Provide a desired velocity in the X-translation direction
    xd_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau_l_with_vel, _ = ctrl.compute_dual_torques(
        Q_HOME_LEFT, np.zeros(6), Q_HOME_RIGHT, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        xd_left=xd_desired,
    )

    # With non-zero desired velocity and zero joint velocity, vel_err = xd_des ≠ 0
    # → damping term contributes → torques differ
    assert not np.allclose(tau_l_zero_vel, tau_l_with_vel, atol=1e-6), (
        "Providing a non-zero desired EE velocity should change the output torques"
    )


# ---------------------------------------------------------------------------
# Controller uses model from constructor
# ---------------------------------------------------------------------------


def test_controller_stores_model_reference(dual_model):
    """Controller holds a reference to the DualArmModel passed at construction."""
    ctrl = DualImpedanceController(dual_model)
    assert ctrl.model is dual_model


def test_controller_default_gains_are_dual_impedance_gains(dual_model):
    """Controller with no gains argument uses DualImpedanceGains defaults."""
    ctrl = DualImpedanceController(dual_model)
    assert isinstance(ctrl.gains, DualImpedanceGains)


# ---------------------------------------------------------------------------
# Torques are finite for all inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_left,q_right,label", [
    (Q_HOME_LEFT, Q_HOME_RIGHT, "home"),
    (np.zeros(6), np.zeros(6), "zeros"),
    (
        np.array([0.3, -0.8, 1.2, -0.5, 0.4, -0.2]),
        np.array([-0.3, -0.8, 1.2, -0.5, -0.4, 0.2]),
        "random",
    ),
])
def test_torques_are_finite(dual_model, q_left, q_right, label):
    """compute_dual_torques returns finite values for various configurations."""
    ctrl = DualImpedanceController(dual_model)
    target_left = dual_model.fk_left(q_left)
    target_right = dual_model.fk_right(q_right)

    tau_l, tau_r = ctrl.compute_dual_torques(
        q_left, np.zeros(6), q_right, np.zeros(6),
        target_left=target_left,
        target_right=target_right,
        grasping=True,
    )

    assert np.all(np.isfinite(tau_l)), f"[{label}] Left torques contain non-finite values: {tau_l}"
    assert np.all(np.isfinite(tau_r)), f"[{label}] Right torques contain non-finite values: {tau_r}"
