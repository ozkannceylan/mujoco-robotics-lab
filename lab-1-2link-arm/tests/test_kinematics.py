"""Unit tests for Lab 1 kinematics modules (A2 FK, A3 Jacobian, A4 IK).

Pure-analytical tests only: no MuJoCo model loading, no rendering, no viewer,
so the whole suite runs in well under a second.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from a2_forward_kinematics import (  # noqa: E402
    L1,
    L2,
    fk_all_joints,
    fk_endeffector,
    fk_homogeneous,
)
from a3_jacobian import (  # noqa: E402
    EE_OFFSET,
    analytic_jacobian,
    determinant,
    endeffector_velocity,
    max_abs_diff,
    numeric_jacobian,
)
from a4_inverse_kinematics import (  # noqa: E402
    analytic_ik,
    fk_endeffector as fk_with_offset,
    is_reachable,
    normalize_angle,
    numeric_ik,
    target_radius_limits,
)

# Configurations sampled across the workspace (radians).
SAMPLE_CONFIGS = [
    (0.0, 0.0),
    (math.radians(30.0), math.radians(45.0)),
    (math.radians(45.0), math.radians(-45.0)),
    (math.radians(60.0), math.radians(60.0)),
    (math.radians(90.0), math.radians(-90.0)),
    (math.radians(-30.0), math.radians(10.0)),
]


# ── A2: Forward kinematics ──────────────────────────────────────────


def test_fk_fully_extended_configuration() -> None:
    """Both joints at zero puts the tip at (L1 + L2, 0)."""
    assert np.allclose(fk_endeffector(0.0, 0.0), [L1 + L2, 0.0], atol=1e-12)


def test_fk_quarter_turn_configuration() -> None:
    """theta1 = 90 deg with a straight elbow points the arm along +y."""
    assert np.allclose(fk_endeffector(math.pi / 2, 0.0), [0.0, L1 + L2], atol=1e-12)


def test_fk_folded_configuration_returns_to_base() -> None:
    """Equal link lengths folded back on themselves reach the origin."""
    assert np.allclose(fk_endeffector(math.radians(37.0), math.pi), [0.0, 0.0], atol=1e-12)


def test_fk_site_offset_extends_reach() -> None:
    """The MuJoCo site offset lengthens the effective second link."""
    with_offset = fk_endeffector(0.0, 0.0, include_site_offset=True)
    assert np.allclose(with_offset, [L1 + L2 + EE_OFFSET, 0.0], atol=1e-12)


def test_fk_homogeneous_matches_geometric_form() -> None:
    """The 3x3 homogeneous chain agrees with the closed-form expression."""
    for theta1, theta2 in SAMPLE_CONFIGS:
        transform = fk_homogeneous(theta1, theta2)
        assert np.allclose(transform[:2, 2], fk_endeffector(theta1, theta2), atol=1e-12)
        # Rotation block must encode the accumulated joint angle.
        assert transform[0, 0] == pytest.approx(math.cos(theta1 + theta2), abs=1e-12)
        assert transform[1, 0] == pytest.approx(math.sin(theta1 + theta2), abs=1e-12)


def test_fk_all_joints_elbow_lies_on_first_link() -> None:
    """The elbow point is always exactly L1 away from the base."""
    for theta1, theta2 in SAMPLE_CONFIGS:
        points = fk_all_joints(theta1, theta2)
        assert np.allclose(points["base"], [0.0, 0.0], atol=1e-12)
        assert float(np.linalg.norm(points["joint2"])) == pytest.approx(L1, abs=1e-12)
        assert np.allclose(points["ee"], fk_endeffector(theta1, theta2), atol=1e-12)


# ── A3: Jacobian ───────────────────────────────────────────────────


def test_analytic_jacobian_matches_finite_differences() -> None:
    """Analytic Jacobian equals the central-difference Jacobian everywhere sampled."""
    for theta1, theta2 in SAMPLE_CONFIGS:
        error = max_abs_diff(analytic_jacobian(theta1, theta2), numeric_jacobian(theta1, theta2))
        assert error < 1e-6, f"Jacobian mismatch {error} at ({theta1}, {theta2})"


def test_jacobian_determinant_is_singular_at_straight_elbow() -> None:
    """det(J) = L1 * L2_eff * sin(theta2) vanishes when the elbow is straight."""
    assert determinant(analytic_jacobian(math.radians(20.0), 0.0)) == pytest.approx(0.0, abs=1e-12)
    assert determinant(analytic_jacobian(math.radians(20.0), math.pi)) == pytest.approx(0.0, abs=1e-12)
    # Away from those two branches the arm is well conditioned.
    assert abs(determinant(analytic_jacobian(math.radians(20.0), math.radians(90.0)))) > 1e-3


def test_endeffector_velocity_equals_jacobian_product() -> None:
    """endeffector_velocity() is exactly J * theta_dot."""
    theta1, theta2 = math.radians(30.0), math.radians(45.0)
    theta_dot = np.array([0.5, -0.2])
    jac = np.array(analytic_jacobian(theta1, theta2))
    assert np.allclose(endeffector_velocity(theta1, theta2, *theta_dot), jac @ theta_dot, atol=1e-12)


# ── A4: Inverse kinematics ─────────────────────────────────────────


def test_normalize_angle_wraps_into_pi_range() -> None:
    """Angles are folded into [-pi, pi] without changing their direction."""
    for raw in (3.0 * math.pi, -3.0 * math.pi, 0.7, -5.0):
        wrapped = normalize_angle(raw)
        assert -math.pi - 1e-12 <= wrapped <= math.pi + 1e-12
        assert math.cos(wrapped) == pytest.approx(math.cos(raw), abs=1e-12)
        assert math.sin(wrapped) == pytest.approx(math.sin(raw), abs=1e-12)


def test_analytic_ik_roundtrip_for_both_branches() -> None:
    """Both elbow branches feed back through FK onto the requested target."""
    for target in [(0.34, 0.28), (0.20, 0.30), (0.40, 0.10), (-0.25, 0.35)]:
        solutions = analytic_ik(target)
        assert {solution.branch for solution in solutions} == {"elbow_up", "elbow_down"}
        for solution in solutions:
            assert solution.error_norm < 1e-9
            recovered = fk_with_offset(solution.theta1, solution.theta2)
            assert np.allclose(recovered, target, atol=1e-9)


def test_analytic_ik_rejects_unreachable_targets() -> None:
    """Targets outside the annulus raise instead of returning a bogus pose."""
    inner, outer = target_radius_limits()
    assert not is_reachable((outer + 0.1, 0.0))
    assert is_reachable((0.34, 0.28))
    with pytest.raises(ValueError):
        analytic_ik((outer + 0.1, 0.0))
    with pytest.raises(ValueError):
        analytic_ik((inner * 0.5, 0.0))


@pytest.mark.parametrize("method", ["pinv", "dls"])
def test_numeric_ik_converges_to_target(method: str) -> None:
    """Pseudo-inverse and DLS solvers both reach the target within tolerance."""
    target = (0.34, 0.28)
    result = numeric_ik(target, method=method)
    assert result.success, f"{method} failed: {result.reason}"
    assert result.error_norm < 1e-5
    assert np.allclose(fk_with_offset(result.theta1, result.theta2), target, atol=1e-4)


def test_numeric_ik_rejects_unknown_method() -> None:
    """An unsupported solver name is a programming error, not a silent fallback."""
    with pytest.raises(ValueError):
        numeric_ik((0.34, 0.28), method="newton")
