"""Unit tests for Lab 1 trajectory generation (B1) and PD control (B2).

All simulations here use the lightweight analytical plant from b2_pd_controller,
so the suite stays fast and deterministic (no MuJoCo, no rendering).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from b1_trajectory_generation import (  # noqa: E402
    cartesian_trajectory,
    cubic_profile,
    joint_trajectory,
    max_line_deviation,
    quintic_profile,
)
from b2_pd_controller import (  # noqa: E402
    fixed_target_simulation,
    pd_control,
    trajectory_tracking_simulation,
)

DURATION = 2.0
Q0 = math.radians(-20.0)
QF = math.radians(75.0)


# ── B1: Trajectory generation ──────────────────────────────────────


def test_cubic_profile_boundary_conditions() -> None:
    """Cubic profile hits both endpoints with zero velocity."""
    q_start, qd_start, _ = cubic_profile(Q0, QF, DURATION, 0.0)
    q_end, qd_end, _ = cubic_profile(Q0, QF, DURATION, DURATION)

    assert q_start == pytest.approx(Q0, abs=1e-12)
    assert q_end == pytest.approx(QF, abs=1e-12)
    assert qd_start == pytest.approx(0.0, abs=1e-12)
    assert qd_end == pytest.approx(0.0, abs=1e-12)


def test_quintic_profile_boundary_conditions() -> None:
    """Quintic profile additionally zeroes acceleration at both endpoints."""
    q_start, qd_start, qdd_start = quintic_profile(Q0, QF, DURATION, 0.0)
    q_end, qd_end, qdd_end = quintic_profile(Q0, QF, DURATION, DURATION)

    assert q_start == pytest.approx(Q0, abs=1e-12)
    assert q_end == pytest.approx(QF, abs=1e-12)
    assert qd_start == pytest.approx(0.0, abs=1e-12)
    assert qd_end == pytest.approx(0.0, abs=1e-12)
    assert qdd_start == pytest.approx(0.0, abs=1e-12)
    assert qdd_end == pytest.approx(0.0, abs=1e-12)


def test_quintic_midpoint_is_halfway_and_at_peak_speed() -> None:
    """Symmetric profile: half the travel and maximum velocity at t = T/2."""
    q_mid, qd_mid, qdd_mid = quintic_profile(Q0, QF, DURATION, DURATION / 2.0)
    assert q_mid == pytest.approx(0.5 * (Q0 + QF), abs=1e-12)
    assert qdd_mid == pytest.approx(0.0, abs=1e-12)
    assert qd_mid == pytest.approx(1.875 * (QF - Q0) / DURATION, abs=1e-12)


def test_quintic_profile_rejects_nonpositive_duration() -> None:
    """A zero-length segment is a caller error, not a divide-by-zero."""
    with pytest.raises(ValueError):
        quintic_profile(Q0, QF, 0.0, 0.0)


def test_joint_trajectory_endpoints_and_time_grid() -> None:
    """Sampled joint trajectory starts/ends on the requested configurations."""
    start = (math.radians(10.0), math.radians(40.0))
    end = (math.radians(-35.0), math.radians(95.0))
    samples = 51

    for mode in ("cubic", "quintic"):
        traj = joint_trajectory(start, end, DURATION, samples, mode=mode)
        assert len(traj) == samples
        assert traj[0].time == pytest.approx(0.0, abs=1e-12)
        assert traj[-1].time == pytest.approx(DURATION, abs=1e-12)
        assert traj[0].theta1 == pytest.approx(start[0], abs=1e-12)
        assert traj[0].theta2 == pytest.approx(start[1], abs=1e-12)
        assert traj[-1].theta1 == pytest.approx(end[0], abs=1e-12)
        assert traj[-1].theta2 == pytest.approx(end[1], abs=1e-12)
        # Monotonically increasing sample times.
        assert all(b.time > a.time for a, b in zip(traj, traj[1:]))


def test_joint_trajectory_rejects_unknown_mode() -> None:
    """Only cubic and quintic interpolation are supported."""
    with pytest.raises(ValueError):
        joint_trajectory((0.0, 0.0), (0.1, 0.1), DURATION, 5, mode="trapezoidal")


def test_cartesian_trajectory_stays_on_the_straight_line() -> None:
    """Cartesian interpolation + IK tracks the line far better than joint-space."""
    start_xy = (0.20, 0.30)
    end_xy = (0.40, 0.10)
    samples = 51

    cartesian = cartesian_trajectory(start_xy, end_xy, DURATION, samples)
    cartesian_points = [(s.x, s.y) for s in cartesian]
    cartesian_deviation = max_line_deviation(cartesian_points, start_xy, end_xy)

    joint_space = joint_trajectory(
        (cartesian[0].theta1, cartesian[0].theta2),
        (cartesian[-1].theta1, cartesian[-1].theta2),
        DURATION,
        samples,
        mode="quintic",
    )
    joint_deviation = max_line_deviation([(s.x, s.y) for s in joint_space], start_xy, end_xy)

    assert cartesian_deviation < 1e-9
    assert joint_deviation > cartesian_deviation


# ── B2: PD control ─────────────────────────────────────────────────


def test_pd_control_zero_error_returns_only_gravity_term() -> None:
    """With no error the command reduces to the feedforward gravity torque."""
    q = (0.3, -0.4)
    tau = pd_control(q, (0.0, 0.0), q, (0.0, 0.0), kp=(20.0, 15.0), kd=(5.0, 4.0), gravity_term=(1.1, -0.6))
    assert tau == pytest.approx((1.1, -0.6), abs=1e-12)


def test_pd_control_sign_and_saturation() -> None:
    """The command opposes position error and respects the torque limit."""
    tau = pd_control((1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), kp=(20.0, 15.0), kd=(5.0, 4.0))
    assert tau[0] > 0.0  # drive joint 1 towards the larger desired angle

    saturated = pd_control(
        (10.0, -10.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
        kp=(200.0, 200.0), kd=(0.0, 0.0), torque_limit=5.0,
    )
    assert saturated == pytest.approx((5.0, -5.0), abs=1e-12)


def test_gravity_compensation_reduces_steady_state_error() -> None:
    """PD + g(q) removes the droop that plain PD leaves at the step target."""
    _, no_gc = fixed_target_simulation(use_gravity_comp=False)
    _, with_gc = fixed_target_simulation(use_gravity_comp=True)

    assert with_gc["final_error_norm"] < no_gc["final_error_norm"]
    assert with_gc["final_error_norm"] < 1e-3


def test_gravity_compensation_reduces_tracking_error() -> None:
    """The same holds while tracking a cubic joint trajectory."""
    _, no_gc = trajectory_tracking_simulation(use_gravity_comp=False)
    _, with_gc = trajectory_tracking_simulation(use_gravity_comp=True)

    assert with_gc["rms_error"] < no_gc["rms_error"]
    assert with_gc["final_error_norm"] < no_gc["final_error_norm"]
