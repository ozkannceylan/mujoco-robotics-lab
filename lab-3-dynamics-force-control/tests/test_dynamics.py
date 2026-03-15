"""Tests for Lab 3 dynamics fundamentals: M(q), C(q,q̇), g(q) cross-validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab3_common import Q_HOME, Q_ZEROS, NUM_JOINTS, load_pinocchio_model, load_mujoco_model
from a1_dynamics_fundamentals import (
    compute_mass_matrix,
    compute_coriolis_matrix,
    compute_gravity_vector,
    cross_validate_gravity,
    cross_validate_mass_matrix,
)


class TestMassMatrix(unittest.TestCase):
    """Tests for the mass matrix M(q)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pin_model, cls.pin_data, _ = load_pinocchio_model()

    def test_symmetric(self) -> None:
        """M(q) must be symmetric."""
        for q in [Q_HOME, Q_ZEROS, np.random.uniform(-2, 2, NUM_JOINTS)]:
            M = compute_mass_matrix(self.pin_model, self.pin_data, q)
            np.testing.assert_allclose(M, M.T, atol=1e-10,
                                       err_msg="M(q) is not symmetric")

    def test_positive_definite(self) -> None:
        """M(q) must be positive definite (all eigenvalues > 0)."""
        for q in [Q_HOME, Q_ZEROS, np.random.uniform(-2, 2, NUM_JOINTS)]:
            M = compute_mass_matrix(self.pin_model, self.pin_data, q)
            eigvals = np.linalg.eigvalsh(M)
            self.assertTrue(np.all(eigvals > 0),
                            f"M(q) not positive definite: eigvals={eigvals}")

    def test_shape(self) -> None:
        """M(q) must be 6x6."""
        M = compute_mass_matrix(self.pin_model, self.pin_data, Q_HOME)
        self.assertEqual(M.shape, (NUM_JOINTS, NUM_JOINTS))


class TestGravityVector(unittest.TestCase):
    """Tests for the gravity vector g(q)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pin_model, cls.pin_data, _ = load_pinocchio_model()

    def test_shape(self) -> None:
        """g(q) must be (6,)."""
        g = compute_gravity_vector(self.pin_model, self.pin_data, Q_HOME)
        self.assertEqual(g.shape, (NUM_JOINTS,))

    def test_nonzero_at_home(self) -> None:
        """g(q) should be nonzero at Q_HOME (gravity acts on the arm)."""
        g = compute_gravity_vector(self.pin_model, self.pin_data, Q_HOME)
        self.assertGreater(np.linalg.norm(g), 1.0)


class TestCoriolisMatrix(unittest.TestCase):
    """Tests for the Coriolis matrix C(q, q̇)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pin_model, cls.pin_data, _ = load_pinocchio_model()

    def test_zero_velocity(self) -> None:
        """C(q, 0) should be zero when velocity is zero."""
        C = compute_coriolis_matrix(self.pin_model, self.pin_data,
                                    Q_HOME, np.zeros(NUM_JOINTS))
        np.testing.assert_allclose(C, 0.0, atol=1e-12)

    def test_shape(self) -> None:
        """C(q, q̇) must be 6x6."""
        qd = np.random.uniform(-1, 1, NUM_JOINTS)
        C = compute_coriolis_matrix(self.pin_model, self.pin_data, Q_HOME, qd)
        self.assertEqual(C.shape, (NUM_JOINTS, NUM_JOINTS))

    def test_skew_symmetric_property(self) -> None:
        """Ṁ - 2C should be skew-symmetric (Christoffel property)."""
        q = np.random.uniform(-2, 2, NUM_JOINTS)
        qd = np.random.uniform(-1, 1, NUM_JOINTS)
        C = compute_coriolis_matrix(self.pin_model, self.pin_data, q, qd)

        # Numerical Ṁ via finite difference
        eps = 1e-6
        q_plus = q + eps * qd
        M_plus = compute_mass_matrix(self.pin_model, self.pin_data, q_plus)
        M = compute_mass_matrix(self.pin_model, self.pin_data, q)
        M_dot = (M_plus - M) / eps

        S = M_dot - 2 * C
        # S should be skew-symmetric: S + S^T ≈ 0
        np.testing.assert_allclose(S + S.T, 0.0, atol=1e-3,
                                   err_msg="Ṁ - 2C is not skew-symmetric")


class TestEOMConsistency(unittest.TestCase):
    """Test that τ = M*qdd + C*qd + g matches RNEA (inverse dynamics)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pin_model, cls.pin_data, _ = load_pinocchio_model()

    def test_rnea_matches_mcg(self) -> None:
        """RNEA(q, qd, qdd) should equal M*qdd + C*qd + g."""
        import pinocchio as pin

        q = np.random.uniform(-2, 2, NUM_JOINTS)
        qd = np.random.uniform(-1, 1, NUM_JOINTS)
        qdd = np.random.uniform(-1, 1, NUM_JOINTS)

        # RNEA
        tau_rnea = pin.rnea(self.pin_model, self.pin_data, q, qd, qdd)

        # M*qdd + C*qd + g
        M = compute_mass_matrix(self.pin_model, self.pin_data, q)
        C = compute_coriolis_matrix(self.pin_model, self.pin_data, q, qd)
        g = compute_gravity_vector(self.pin_model, self.pin_data, q)
        tau_mcg = M @ qdd + C @ qd + g

        np.testing.assert_allclose(tau_rnea, tau_mcg, atol=1e-8,
                                   err_msg="RNEA != M*qdd + C*qd + g")


class TestCrossValidation(unittest.TestCase):
    """Cross-validate Pinocchio vs MuJoCo dynamics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pin_model, cls.pin_data, _ = load_pinocchio_model()
        cls.mj_model, cls.mj_data = load_mujoco_model()

    def _test_config(self, q: np.ndarray, name: str) -> None:
        """Helper: cross-validate at a given config."""
        g_err = cross_validate_gravity(
            self.pin_model, self.pin_data, self.mj_model, self.mj_data, q
        )
        self.assertLess(g_err, 0.01,
                        f"g(q) mismatch at {name}: {g_err:.6f} Nm")

        M_err = cross_validate_mass_matrix(
            self.pin_model, self.pin_data, self.mj_model, self.mj_data, q
        )
        self.assertLess(M_err, 0.001,
                        f"M(q) mismatch at {name}: {M_err:.6f}")

    def test_cross_val_home(self) -> None:
        self._test_config(Q_HOME, "Q_HOME")

    def test_cross_val_zeros(self) -> None:
        self._test_config(Q_ZEROS, "Q_ZEROS")

    def test_cross_val_random(self) -> None:
        np.random.seed(42)
        for i in range(3):
            q = np.random.uniform(-2, 2, NUM_JOINTS)
            self._test_config(q, f"random_{i}")


if __name__ == "__main__":
    unittest.main()
