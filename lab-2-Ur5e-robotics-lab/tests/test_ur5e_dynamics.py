"""Tests for UR5e dynamics (a5 module)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

LAB2_SRC = Path(__file__).resolve().parents[1] / "src"
if str(LAB2_SRC) not in sys.path:
    sys.path.insert(0, str(LAB2_SRC))

from ur5e_common import Q_HOME, NUM_JOINTS, load_pinocchio_model
from a5_dynamics import (
    compute_mass_matrix,
    compute_gravity_vector,
    compute_rnea,
    compute_aba,
)


class TestMassMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.data, _ = load_pinocchio_model()

    def test_symmetric(self) -> None:
        M = compute_mass_matrix(self.model, self.data, Q_HOME)
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_positive_definite(self) -> None:
        M = compute_mass_matrix(self.model, self.data, Q_HOME)
        eigvals = np.linalg.eigvalsh(M)
        self.assertTrue(np.all(eigvals > 0))

    def test_shape(self) -> None:
        M = compute_mass_matrix(self.model, self.data, Q_HOME)
        self.assertEqual(M.shape, (NUM_JOINTS, NUM_JOINTS))


class TestGravityVector(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.data, _ = load_pinocchio_model()

    def test_shape(self) -> None:
        g = compute_gravity_vector(self.model, self.data, Q_HOME)
        self.assertEqual(g.shape, (NUM_JOINTS,))

    def test_nonzero(self) -> None:
        g = compute_gravity_vector(self.model, self.data, Q_HOME)
        self.assertGreater(np.linalg.norm(g), 0.1)


class TestRNEA(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.data, _ = load_pinocchio_model()

    def test_static_equals_gravity(self) -> None:
        """RNEA with v=0, a=0 should equal gravity vector."""
        v = np.zeros(NUM_JOINTS)
        a = np.zeros(NUM_JOINTS)
        tau = compute_rnea(self.model, self.data, Q_HOME, v, a)
        g = compute_gravity_vector(self.model, self.data, Q_HOME)
        np.testing.assert_allclose(tau, g, atol=1e-10)


class TestABA(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.data, _ = load_pinocchio_model()

    def test_aba_matches_minv_method(self) -> None:
        """ABA(q, 0, 0) should equal M^-1 * (-g)."""
        v = np.zeros(NUM_JOINTS)
        tau = np.zeros(NUM_JOINTS)
        qdd_aba = compute_aba(self.model, self.data, Q_HOME, v, tau)
        M = compute_mass_matrix(self.model, self.data, Q_HOME)
        g = compute_gravity_vector(self.model, self.data, Q_HOME)
        qdd_manual = np.linalg.solve(M, tau - g)
        np.testing.assert_allclose(qdd_aba, qdd_manual, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
