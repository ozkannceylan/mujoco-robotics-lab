"""Tests for UR5e trajectory generation (b1 module)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

LAB2_SRC = Path(__file__).resolve().parents[1] / "src"
if str(LAB2_SRC) not in sys.path:
    sys.path.insert(0, str(LAB2_SRC))

from ur5e_common import Q_HOME, NUM_JOINTS, VELOCITY_LIMITS
from b1_trajectory_generation import (
    cubic_trajectory,
    quintic_trajectory,
    trapezoidal_trajectory,
    multi_segment_trajectory,
)


Q_END = np.array([0.0, -np.pi / 2, np.pi / 3, -np.pi / 3, -np.pi / 2, 0.0])


class TestCubicTrajectory(unittest.TestCase):

    def test_endpoint_positions(self) -> None:
        traj = cubic_trajectory(Q_HOME, Q_END, 2.0, 0.01)
        np.testing.assert_allclose(traj[0].q, Q_HOME, atol=1e-10)
        np.testing.assert_allclose(traj[-1].q, Q_END, atol=1e-6)

    def test_zero_endpoint_velocity(self) -> None:
        traj = cubic_trajectory(Q_HOME, Q_END, 2.0, 0.01)
        self.assertLess(np.linalg.norm(traj[0].qd), 1e-10)
        self.assertLess(np.linalg.norm(traj[-1].qd), 1e-6)


class TestQuinticTrajectory(unittest.TestCase):

    def test_zero_endpoint_velocity(self) -> None:
        traj = quintic_trajectory(Q_HOME, Q_END, 2.0, 0.01)
        self.assertLess(np.linalg.norm(traj[0].qd), 1e-10)
        self.assertLess(np.linalg.norm(traj[-1].qd), 1e-6)

    def test_zero_endpoint_acceleration(self) -> None:
        traj = quintic_trajectory(Q_HOME, Q_END, 2.0, 0.01)
        self.assertLess(np.linalg.norm(traj[0].qdd), 1e-10)
        self.assertLess(np.linalg.norm(traj[-1].qdd), 1e-6)


class TestTrapezoidalTrajectory(unittest.TestCase):

    def test_respects_velocity_limits(self) -> None:
        v_max = VELOCITY_LIMITS * 0.5
        a_max = np.full(NUM_JOINTS, 4.0)
        traj = trapezoidal_trajectory(Q_HOME, Q_END, v_max, a_max, 0.01)
        for pt in traj:
            max_vel = np.max(np.abs(pt.qd))
            self.assertLess(max_vel, np.max(v_max) + 0.01)

    def test_endpoints(self) -> None:
        v_max = VELOCITY_LIMITS * 0.5
        a_max = np.full(NUM_JOINTS, 4.0)
        traj = trapezoidal_trajectory(Q_HOME, Q_END, v_max, a_max, 0.01)
        np.testing.assert_allclose(traj[0].q, Q_HOME, atol=1e-6)


class TestMultiSegmentTrajectory(unittest.TestCase):

    def test_lands_on_last_waypoint(self) -> None:
        q_mid = np.array([-np.pi / 4, -np.pi / 2, np.pi / 4,
                           -np.pi / 4, -np.pi / 2, 0.0])
        waypoints = [Q_HOME, q_mid, Q_END]
        traj = multi_segment_trajectory(waypoints, [1.5, 1.5], 0.01)
        np.testing.assert_allclose(traj[-1].q, Q_END, atol=1e-6)

    def test_point_count(self) -> None:
        traj = quintic_trajectory(Q_HOME, Q_END, 2.0, 0.01)
        self.assertGreater(len(traj), 100)


if __name__ == "__main__":
    unittest.main()
