"""Tests for UR5e forward kinematics and Jacobian (a2, a3 modules)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

LAB2_SRC = Path(__file__).resolve().parents[1] / "src"
if str(LAB2_SRC) not in sys.path:
    sys.path.insert(0, str(LAB2_SRC))

from ur5e_common import Q_HOME, Q_ZEROS, NUM_JOINTS, load_pinocchio_model, load_mujoco_model, get_mj_ee_site_id
import pinocchio as pin
import mujoco


class TestFKCrossValidation(unittest.TestCase):
    """Pinocchio vs MuJoCo FK agreement."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pin_model, cls.pin_data, cls.ee_fid = load_pinocchio_model()
        cls.mj_model, cls.mj_data = load_mujoco_model()
        cls.site_id = get_mj_ee_site_id(cls.mj_model)

    def _compare_fk(self, q: np.ndarray, tol_mm: float = 1.0) -> None:
        """Assert Pin and MuJoCo FK agree within tol_mm."""
        # Pinocchio
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        p_pin = self.pin_data.oMf[self.ee_fid].translation.copy()

        # MuJoCo
        self.mj_data.qpos[:NUM_JOINTS] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        p_mj = self.mj_data.site_xpos[self.site_id].copy()

        err_mm = np.linalg.norm(p_pin - p_mj) * 1000
        self.assertLess(err_mm, tol_mm, f"FK mismatch: {err_mm:.3f} mm")

    def test_fk_home(self) -> None:
        self._compare_fk(Q_HOME)

    def test_fk_zeros(self) -> None:
        self._compare_fk(Q_ZEROS)

    def test_fk_random(self) -> None:
        self._compare_fk(np.array([0.3, -1.2, 0.8, -0.6, 1.1, -0.4]))


class TestJacobian(unittest.TestCase):
    """Jacobian cross-validation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pin_model, cls.pin_data, cls.ee_fid = load_pinocchio_model()

    def test_geometric_vs_pinocchio_jacobian(self) -> None:
        from a3_jacobian import geometric_jacobian, pinocchio_jacobian
        q = np.array([0.3, -1.2, 0.8, -0.6, 1.1, -0.4])
        J_geo = geometric_jacobian(q)
        J_pin = pinocchio_jacobian(q)
        err = np.linalg.norm(J_geo - J_pin, "fro")
        self.assertLess(err, 1e-4)

    def test_numerical_vs_pinocchio_jacobian(self) -> None:
        from a3_jacobian import numerical_jacobian, pinocchio_jacobian
        q = Q_HOME.copy()
        J_num = numerical_jacobian(q)
        J_pin = pinocchio_jacobian(q)
        err = np.linalg.norm(J_num - J_pin, "fro")
        self.assertLess(err, 1e-4)

    def test_jacobian_shape(self) -> None:
        from a3_jacobian import pinocchio_jacobian
        J = pinocchio_jacobian(Q_HOME)
        self.assertEqual(J.shape, (6, 6))


if __name__ == "__main__":
    unittest.main()
