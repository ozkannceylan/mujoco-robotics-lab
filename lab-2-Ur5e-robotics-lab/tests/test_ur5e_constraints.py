"""Tests for UR5e constraint utilities (b3 module)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

LAB2_SRC = Path(__file__).resolve().parents[1] / "src"
if str(LAB2_SRC) not in sys.path:
    sys.path.insert(0, str(LAB2_SRC))

from ur5e_common import Q_HOME, VELOCITY_LIMITS, TORQUE_LIMITS
from b3_constraints import (
    check_joint_limits,
    clamp_joint_positions,
    scale_velocity,
    scale_delta_q,
    saturate_torques,
    joint_limit_repulsion,
)


class TestJointLimits(unittest.TestCase):

    def test_home_within_limits(self) -> None:
        valid, margin = check_joint_limits(Q_HOME)
        self.assertTrue(valid)
        self.assertTrue(np.all(margin > 0))

    def test_beyond_limits_detected(self) -> None:
        q_bad = np.array([7.0, -7.0, 4.0, -7.0, 7.0, -7.0])
        valid, margin = check_joint_limits(q_bad)
        self.assertFalse(valid)

    def test_clamping_brings_within_limits(self) -> None:
        q_bad = np.array([7.0, -7.0, 4.0, -7.0, 7.0, -7.0])
        q_clamped = clamp_joint_positions(q_bad)
        valid, _ = check_joint_limits(q_clamped)
        self.assertTrue(valid)


class TestVelocityScaling(unittest.TestCase):

    def test_within_limits_unchanged(self) -> None:
        qd = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        qd_scaled = scale_velocity(qd)
        np.testing.assert_allclose(qd, qd_scaled)

    def test_exceeding_limits_scaled_down(self) -> None:
        qd = np.array([5.0, 5.0, 5.0, 10.0, 10.0, 10.0])
        qd_scaled = scale_velocity(qd)
        ratios = np.abs(qd_scaled) / VELOCITY_LIMITS
        self.assertLessEqual(np.max(ratios), 1.0 + 1e-10)

    def test_delta_q_respects_limits(self) -> None:
        dq = np.array([0.5, -0.5, 0.5, -1.0, 1.0, 1.0])
        dt = 0.002
        dq_scaled = scale_delta_q(dq, dt)
        implied_vel = np.abs(dq_scaled) / dt
        self.assertTrue(np.all(implied_vel <= VELOCITY_LIMITS + 1e-9))


class TestTorqueSaturation(unittest.TestCase):

    def test_within_limits_unchanged(self) -> None:
        tau = np.array([50.0, -50.0, 50.0, 10.0, -10.0, 10.0])
        tau_sat = saturate_torques(tau)
        np.testing.assert_allclose(tau, tau_sat)

    def test_exceeding_limits_clipped(self) -> None:
        tau = np.array([200.0, -200.0, 200.0, 50.0, -50.0, 50.0])
        tau_sat = saturate_torques(tau)
        self.assertTrue(np.all(np.abs(tau_sat) <= TORQUE_LIMITS + 1e-10))


class TestJointLimitRepulsion(unittest.TestCase):

    def test_home_no_repulsion(self) -> None:
        tau = joint_limit_repulsion(Q_HOME)
        self.assertLess(np.max(np.abs(tau)), 1e-6)

    def test_near_limit_produces_torque(self) -> None:
        q_near = np.array([6.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau = joint_limit_repulsion(q_near)
        self.assertLess(tau[0], 0)


if __name__ == "__main__":
    unittest.main()
