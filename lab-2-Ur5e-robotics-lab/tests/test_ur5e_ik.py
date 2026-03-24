"""Tests for UR5e inverse kinematics (a4 module)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pinocchio as pin

LAB2_SRC = Path(__file__).resolve().parents[1] / "src"
if str(LAB2_SRC) not in sys.path:
    sys.path.insert(0, str(LAB2_SRC))

from ur5e_common import Q_HOME, NUM_JOINTS, load_pinocchio_model
from a4_inverse_kinematics import ik_pseudoinverse, ik_damped_least_squares, validate_ik_solution


class TestIKPseudoinverse(unittest.TestCase):
    """Pseudo-inverse IK solver tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.data, cls.ee_fid = load_pinocchio_model()

    def _make_target(self, q: np.ndarray):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return (
            self.data.oMf[self.ee_fid].translation.copy(),
            self.data.oMf[self.ee_fid].rotation.copy(),
        )

    def test_recovers_known_config(self) -> None:
        q_target = np.array([0.3, -1.2, 1.5, -1.6, 1.57, 0.0])
        p, R = self._make_target(q_target)
        result = ik_pseudoinverse(self.model, self.data, self.ee_fid,
                                   p, R, Q_HOME.copy(), max_iter=300)
        self.assertTrue(result.success, f"IK failed: err={result.final_error:.6f}")
        self.assertLess(result.position_error, 1e-3)


class TestIKDampedLS(unittest.TestCase):
    """Damped Least Squares IK solver tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.data, cls.ee_fid = load_pinocchio_model()

    def _make_target(self, q: np.ndarray):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return (
            self.data.oMf[self.ee_fid].translation.copy(),
            self.data.oMf[self.ee_fid].rotation.copy(),
        )

    def test_recovers_known_config(self) -> None:
        q_target = np.array([-0.5, -1.0, 1.2, -1.8, -1.57, 0.3])
        p, R = self._make_target(q_target)
        result = ik_damped_least_squares(self.model, self.data, self.ee_fid,
                                          p, R, Q_HOME.copy())
        self.assertTrue(result.success, f"IK failed: err={result.final_error:.6f}")
        self.assertLess(result.position_error, 1e-3)

    def test_fk_roundtrip(self) -> None:
        q_target = np.array([0.3, -1.2, 1.5, -1.6, 1.57, 0.0])
        p, R = self._make_target(q_target)
        result = ik_damped_least_squares(self.model, self.data, self.ee_fid,
                                          p, R, Q_HOME.copy())
        valid, pos_err, rot_err = validate_ik_solution(
            self.model, self.data, self.ee_fid, result.q, p, R
        )
        self.assertTrue(valid, f"FK roundtrip failed: pos={pos_err:.6f}, rot={rot_err:.6f}")


if __name__ == "__main__":
    unittest.main()
