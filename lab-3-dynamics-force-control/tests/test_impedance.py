"""Tests for Lab 3 Cartesian impedance controller."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab3_common import Q_HOME, get_ee_pose, load_pinocchio_model
from b1_impedance_controller import (
    ImpedanceGains,
    orientation_error,
    run_impedance_sim,
)


class TestOrientationError(unittest.TestCase):
    """Tests for the orientation error function."""

    def test_zero_error(self) -> None:
        """Same rotation should give zero error."""
        R = np.eye(3)
        e = orientation_error(R, R)
        np.testing.assert_allclose(e, 0.0, atol=1e-12)

    def test_small_rotation(self) -> None:
        """Small rotation about Z should give error along Z."""
        theta = 0.05  # ~3 degrees
        R_des = np.eye(3)
        R_cur = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        e = orientation_error(R_des, R_cur)
        # Error should be approximately [0, 0, -theta] for small angles
        self.assertAlmostEqual(abs(e[2]), theta, places=3)
        np.testing.assert_allclose(e[:2], 0.0, atol=1e-4)

    def test_identity_rotation(self) -> None:
        """Random rotation compared to itself gives zero."""
        from scipy.spatial.transform import Rotation
        R = Rotation.random(random_state=42).as_matrix()
        e = orientation_error(R, R)
        np.testing.assert_allclose(e, 0.0, atol=1e-10)


class TestImpedance3D(unittest.TestCase):
    """Tests for 3D translational impedance controller."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pin_model, cls.pin_data, cls.ee_fid = load_pinocchio_model()
        cls.x_home, cls.R_home = get_ee_pose(
            cls.pin_model, cls.pin_data, cls.ee_fid, Q_HOME
        )

    def test_reaches_target(self) -> None:
        """3D impedance with K=500 should reach target within 5mm."""
        x_target = self.x_home + np.array([0.05, 0.0, 0.05])
        gains = ImpedanceGains(
            K_p=np.diag([500.0, 500.0, 500.0]),
            K_d=np.diag([50.0, 50.0, 50.0]),
        )
        results = run_impedance_sim(x_des=x_target, gains=gains, duration=3.0)
        final_err_mm = np.linalg.norm(results["ee_pos_err"][-1]) * 1000
        self.assertLess(final_err_mm, 5.0,
                        f"Final position error {final_err_mm:.2f} mm > 5 mm")

    def test_stiffness_scaling_monotonic(self) -> None:
        """Higher stiffness should give lower steady-state error."""
        x_target = self.x_home + np.array([0.0, 0.0, 0.08])
        errors = []
        for K in [100.0, 500.0, 2000.0]:
            gains = ImpedanceGains(
                K_p=np.diag([K, K, K]),
                K_d=np.diag([2 * np.sqrt(K)] * 3),
            )
            results = run_impedance_sim(x_des=x_target, gains=gains, duration=2.0)
            # Measure error at t=1.5s (well after transient)
            idx = int(1.5 / 0.001)
            err = np.linalg.norm(results["ee_pos_err"][idx]) * 1000
            errors.append(err)

        # Errors should be monotonically decreasing
        for i in range(len(errors) - 1):
            self.assertGreaterEqual(
                errors[i], errors[i + 1],
                f"Stiffness scaling not monotonic: {errors}"
            )


class TestImpedance6D(unittest.TestCase):
    """Tests for 6D impedance controller (position + orientation)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pin_model, cls.pin_data, cls.ee_fid = load_pinocchio_model()
        cls.x_home, cls.R_home = get_ee_pose(
            cls.pin_model, cls.pin_data, cls.ee_fid, Q_HOME
        )

    def test_reaches_target_6d(self) -> None:
        """6D impedance should reach target within 5mm position, 2 deg orientation."""
        x_target = self.x_home + np.array([0.05, 0.0, 0.05])
        gains = ImpedanceGains.make_6d(K_pos=500.0, K_ori=50.0, D_pos=50.0, D_ori=5.0)
        results = run_impedance_sim(
            x_des=x_target, R_des=self.R_home, gains=gains, duration=3.0,
        )
        final_pos_err_mm = np.linalg.norm(results["ee_pos_err"][-1]) * 1000
        final_ori_err_deg = np.degrees(np.linalg.norm(results["ee_ori_err"][-1]))

        self.assertLess(final_pos_err_mm, 5.0,
                        f"Position error {final_pos_err_mm:.2f} mm > 5 mm")
        self.assertLess(final_ori_err_deg, 2.0,
                        f"Orientation error {final_ori_err_deg:.2f} deg > 2 deg")

    def test_perturbation_recovery(self) -> None:
        """6D impedance should recover after perturbation."""
        x_target = self.x_home + np.array([0.0, 0.0, 0.08])
        gains = ImpedanceGains.make_6d(K_pos=500.0, K_ori=50.0, D_pos=50.0, D_ori=5.0)
        results = run_impedance_sim(
            x_des=x_target, R_des=self.R_home, gains=gains, duration=4.0,
            perturb_at=1.5, perturb_force=np.array([30.0, 0.0, 0.0]),
        )
        # Check final error (should recover)
        final_pos_err_mm = np.linalg.norm(results["ee_pos_err"][-1]) * 1000
        self.assertLess(final_pos_err_mm, 5.0,
                        f"Post-perturbation error {final_pos_err_mm:.2f} mm > 5 mm")


if __name__ == "__main__":
    unittest.main()
