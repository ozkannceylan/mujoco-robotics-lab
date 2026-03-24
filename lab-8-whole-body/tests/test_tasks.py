"""Tests for Lab 8 task definitions: CoMTask, FootPoseTask, PostureTask."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab8_common import NQ, NV, NUM_ACTUATED, NQ_FREEFLYER, Q_STAND, TORSO_HEIGHT
from lab8_common import LEFT_FOOT_FRAME
from g1_model import G1WholeBodyModel
from tasks import CoMTask, FootPoseTask, PostureTask


def _make_q_stand() -> np.ndarray:
    """Build standing configuration."""
    q = np.zeros(NQ)
    q[0:3] = [0.0, 0.0, TORSO_HEIGHT]
    q[3:7] = [0.0, 0.0, 0.0, 1.0]
    q[NQ_FREEFLYER:] = Q_STAND
    return q


class TestCoMTask(unittest.TestCase):
    """CoMTask error and Jacobian dimensions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1WholeBodyModel()
        cls.q = _make_q_stand()
        cls.v = np.zeros(NV)
        com = cls.g1.compute_com(cls.q)
        cls.task = CoMTask(target_com=com + np.array([0.01, 0, 0]))

    def test_error_shape(self) -> None:
        """CoM task error is 3D."""
        e = self.task.compute_error(self.g1, self.q, self.v)
        self.assertEqual(e.shape, (3,))

    def test_jacobian_shape(self) -> None:
        """CoM task Jacobian is (3, nv)."""
        J = self.task.compute_jacobian(self.g1, self.q)
        self.assertEqual(J.shape, (3, NV))

    def test_error_nonzero(self) -> None:
        """Error is nonzero when target differs from current."""
        e = self.task.compute_error(self.g1, self.q, self.v)
        self.assertGreater(np.linalg.norm(e), 0.005)


class TestFootPoseTask(unittest.TestCase):
    """FootPoseTask error and Jacobian dimensions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1WholeBodyModel()
        cls.q = _make_q_stand()
        cls.v = np.zeros(NV)
        target = cls.g1.get_frame_pose(cls.q, LEFT_FOOT_FRAME)
        cls.task = FootPoseTask(LEFT_FOOT_FRAME, target_pose=target)

    def test_error_shape(self) -> None:
        """Foot task error is 6D (pos + rot)."""
        e = self.task.compute_error(self.g1, self.q, self.v)
        self.assertEqual(e.shape, (6,))

    def test_jacobian_shape(self) -> None:
        """Foot task Jacobian is (6, nv)."""
        J = self.task.compute_jacobian(self.g1, self.q)
        self.assertEqual(J.shape, (6, NV))

    def test_error_near_zero_at_target(self) -> None:
        """Error is near zero when at target pose."""
        e = self.task.compute_error(self.g1, self.q, self.v)
        self.assertLess(np.linalg.norm(e), 0.01)


class TestPostureTask(unittest.TestCase):
    """PostureTask error and Jacobian dimensions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1WholeBodyModel()
        cls.q = _make_q_stand()
        cls.v = np.zeros(NV)
        cls.task = PostureTask(target_posture=Q_STAND)

    def test_error_shape(self) -> None:
        """Posture task error has na dimensions."""
        e = self.task.compute_error(self.g1, self.q, self.v)
        self.assertEqual(e.shape, (NUM_ACTUATED,))

    def test_jacobian_shape(self) -> None:
        """Posture task Jacobian is (na, nv)."""
        J = self.task.compute_jacobian(self.g1, self.q)
        self.assertEqual(J.shape, (NUM_ACTUATED, NV))

    def test_error_near_zero_at_target(self) -> None:
        """Error near zero when at target posture."""
        e = self.task.compute_error(self.g1, self.q, self.v)
        self.assertLess(np.linalg.norm(e), 0.01)


if __name__ == "__main__":
    unittest.main()
