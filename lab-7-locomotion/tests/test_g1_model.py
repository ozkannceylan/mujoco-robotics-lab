"""Tests for Lab 7 G1Model class: FK, CoM, Jacobians, mass matrix, IK."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab7_common import (
    NQ_PIN,
    NV_PIN,
    NUM_ACTUATED,
    NV_FREEFLYER,
    build_home_q_pin,
)
from g1_model import G1Model


class TestModelLoads(unittest.TestCase):
    """Verify URDF loads with freeflyer and has correct dimensions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1Model()

    def test_model_loads(self) -> None:
        """URDF loads with freeflyer; nq, nv, na are correct."""
        model = self.g1.model
        # Freeflyer: nq = 7 + actuated, nv = 6 + actuated
        self.assertEqual(model.nq, NQ_PIN,
                         f"Expected nq={NQ_PIN}, got {model.nq}")
        self.assertEqual(model.nv, NV_PIN,
                         f"Expected nv={NV_PIN}, got {model.nv}")
        # na = nv - 6 (freeflyer dofs)
        expected_na = NV_PIN - NV_FREEFLYER
        self.assertEqual(expected_na, NUM_ACTUATED,
                         f"Expected na={NUM_ACTUATED}, got {expected_na}")

    def test_frame_ids_valid(self) -> None:
        """Left and right foot frame IDs are valid."""
        self.assertLess(self.g1.left_foot_id, self.g1.model.nframes)
        self.assertLess(self.g1.right_foot_id, self.g1.model.nframes)

    def test_total_mass_positive(self) -> None:
        """Total mass is positive and reasonable for a humanoid."""
        self.assertGreater(self.g1.total_mass, 10.0,
                           "Total mass too low for humanoid")
        self.assertLess(self.g1.total_mass, 200.0,
                        "Total mass too high for humanoid")


class TestFKAtHome(unittest.TestCase):
    """FK at Q_HOME returns reasonable foot positions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1Model()
        cls.q_home = build_home_q_pin()

    def test_feet_below_torso(self) -> None:
        """Both feet should be below the torso/pelvis z-coordinate."""
        q = self.q_home
        lf = self.g1.get_left_foot_pos(q)
        rf = self.g1.get_right_foot_pos(q)
        torso_z = q[2]  # base z position

        self.assertLess(lf[2], torso_z,
                        f"Left foot z={lf[2]:.3f} not below torso z={torso_z:.3f}")
        self.assertLess(rf[2], torso_z,
                        f"Right foot z={rf[2]:.3f} not below torso z={torso_z:.3f}")

    def test_feet_near_ground(self) -> None:
        """Feet should be near ground level (z close to 0)."""
        q = self.q_home
        lf = self.g1.get_left_foot_pos(q)
        rf = self.g1.get_right_foot_pos(q)

        np.testing.assert_allclose(lf[2], 0.0, atol=0.15,
                                   err_msg="Left foot too far from ground")
        np.testing.assert_allclose(rf[2], 0.0, atol=0.15,
                                   err_msg="Right foot too far from ground")

    def test_feet_symmetric_y(self) -> None:
        """Left and right foot y-positions should be approximately symmetric."""
        q = self.q_home
        lf = self.g1.get_left_foot_pos(q)
        rf = self.g1.get_right_foot_pos(q)

        # Left foot should be positive y, right foot negative y
        self.assertGreater(lf[1], 0.0,
                           f"Left foot y={lf[1]:.3f} should be positive")
        self.assertLess(rf[1], 0.0,
                        f"Right foot y={rf[1]:.3f} should be negative")

        # Symmetric about y=0
        np.testing.assert_allclose(lf[1], -rf[1], atol=0.05,
                                   err_msg="Feet not symmetric in Y")

    def test_feet_symmetric_x(self) -> None:
        """Left and right foot x-positions should be close at home."""
        q = self.q_home
        lf = self.g1.get_left_foot_pos(q)
        rf = self.g1.get_right_foot_pos(q)

        np.testing.assert_allclose(lf[0], rf[0], atol=0.05,
                                   err_msg="Feet not aligned in X at home")


class TestCoMAtHome(unittest.TestCase):
    """CoM at home configuration is between feet and at expected height."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1Model()
        cls.q_home = build_home_q_pin()

    def test_com_between_feet_x(self) -> None:
        """CoM x should be near the midpoint of the feet."""
        q = self.q_home
        com = self.g1.compute_com(q)
        lf = self.g1.get_left_foot_pos(q)
        rf = self.g1.get_right_foot_pos(q)
        mid_x = (lf[0] + rf[0]) / 2.0

        np.testing.assert_allclose(com[0], mid_x, atol=0.15,
                                   err_msg="CoM X not between feet")

    def test_com_between_feet_y(self) -> None:
        """CoM y should be near zero (between feet)."""
        q = self.q_home
        com = self.g1.compute_com(q)

        np.testing.assert_allclose(com[1], 0.0, atol=0.10,
                                   err_msg="CoM Y not centered between feet")

    def test_com_height(self) -> None:
        """CoM z should be at a reasonable height for a standing humanoid."""
        q = self.q_home
        com = self.g1.compute_com(q)

        # CoM should be above ground and below torso
        self.assertGreater(com[2], 0.2,
                           f"CoM height {com[2]:.3f}m too low")
        self.assertLess(com[2], 1.2,
                        f"CoM height {com[2]:.3f}m too high")


class TestJacobianShape(unittest.TestCase):
    """Jacobians have correct shapes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1Model()
        cls.q_home = build_home_q_pin()

    def test_com_jacobian_shape(self) -> None:
        """CoM Jacobian is (3, nv)."""
        J_com = self.g1.compute_jacobian_com(self.q_home)
        self.assertEqual(J_com.shape, (3, NV_PIN),
                         f"Expected (3, {NV_PIN}), got {J_com.shape}")

    def test_foot_jacobian_shape_left(self) -> None:
        """Left foot Jacobian is (6, nv)."""
        J_lf = self.g1.compute_frame_jacobian(self.q_home, self.g1.left_foot_id)
        self.assertEqual(J_lf.shape, (6, NV_PIN),
                         f"Expected (6, {NV_PIN}), got {J_lf.shape}")

    def test_foot_jacobian_shape_right(self) -> None:
        """Right foot Jacobian is (6, nv)."""
        J_rf = self.g1.compute_frame_jacobian(self.q_home, self.g1.right_foot_id)
        self.assertEqual(J_rf.shape, (6, NV_PIN),
                         f"Expected (6, {NV_PIN}), got {J_rf.shape}")

    def test_com_jacobian_not_all_zero(self) -> None:
        """CoM Jacobian should have nonzero entries."""
        J_com = self.g1.compute_jacobian_com(self.q_home)
        self.assertGreater(np.linalg.norm(J_com), 0.0,
                           "CoM Jacobian is all zeros")


class TestMassMatrixSymmetricPD(unittest.TestCase):
    """Mass matrix M(q) is symmetric and positive definite."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1Model()
        cls.q_home = build_home_q_pin()

    def test_symmetric(self) -> None:
        """M(q) must be symmetric."""
        M = self.g1.compute_mass_matrix(self.q_home)
        np.testing.assert_allclose(M, M.T, atol=1e-10,
                                   err_msg="M(q) is not symmetric")

    def test_positive_definite(self) -> None:
        """M(q) must be positive definite (all eigenvalues > 0)."""
        M = self.g1.compute_mass_matrix(self.q_home)
        eigvals = np.linalg.eigvalsh(M)
        self.assertTrue(np.all(eigvals > 0),
                        f"M(q) not positive definite: min eigval={eigvals.min():.2e}")

    def test_shape(self) -> None:
        """M(q) must be (nv, nv)."""
        M = self.g1.compute_mass_matrix(self.q_home)
        self.assertEqual(M.shape, (NV_PIN, NV_PIN),
                         f"Expected ({NV_PIN}, {NV_PIN}), got {M.shape}")

    def test_symmetric_at_random_config(self) -> None:
        """M(q) is symmetric at a random configuration."""
        np.random.seed(42)
        q = self.q_home.copy()
        q[7:] += np.random.uniform(-0.3, 0.3, NUM_ACTUATED)
        M = self.g1.compute_mass_matrix(q)
        np.testing.assert_allclose(M, M.T, atol=1e-10,
                                   err_msg="M(q) not symmetric at random config")


class TestIKConvergence(unittest.TestCase):
    """whole_body_ik converges for a small CoM offset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1Model()
        cls.q_home = build_home_q_pin()

    def test_ik_small_com_offset(self) -> None:
        """IK converges when CoM target is offset by a small amount."""
        com_home = self.g1.compute_com(self.q_home)
        lf_home = self.g1.get_left_foot_pos(self.q_home)
        rf_home = self.g1.get_right_foot_pos(self.q_home)

        # Small CoM offset (2 cm in x)
        com_target = com_home.copy()
        com_target[0] += 0.02

        q_sol = self.g1.whole_body_ik(
            q_current=self.q_home.copy(),
            com_des=com_target,
            left_foot_des=lf_home,
            right_foot_des=rf_home,
            max_iter=50,
            tol=1e-3,
        )

        # Check IK returned a valid solution
        self.assertIsNotNone(q_sol, "IK returned None")
        self.assertEqual(q_sol.shape, (NQ_PIN,),
                         f"IK solution has wrong shape: {q_sol.shape}")

        # Verify CoM moved toward target
        com_sol = self.g1.compute_com(q_sol)
        error = np.linalg.norm(com_target - com_sol)
        self.assertLess(error, 0.02,
                        f"IK CoM error too large: {error:.4f}m")

    def test_ik_foot_position_maintained(self) -> None:
        """IK should maintain foot positions when they are given as targets."""
        lf_home = self.g1.get_left_foot_pos(self.q_home)
        rf_home = self.g1.get_right_foot_pos(self.q_home)
        com_home = self.g1.compute_com(self.q_home)

        # Small offset
        com_target = com_home.copy()
        com_target[0] += 0.01

        q_sol = self.g1.whole_body_ik(
            q_current=self.q_home.copy(),
            com_des=com_target,
            left_foot_des=lf_home,
            right_foot_des=rf_home,
            max_iter=50,
            tol=1e-3,
        )

        lf_sol = self.g1.get_left_foot_pos(q_sol)
        rf_sol = self.g1.get_right_foot_pos(q_sol)

        np.testing.assert_allclose(lf_sol, lf_home, atol=0.01,
                                   err_msg="Left foot moved too much during IK")
        np.testing.assert_allclose(rf_sol, rf_home, atol=0.01,
                                   err_msg="Right foot moved too much during IK")


if __name__ == "__main__":
    unittest.main()
