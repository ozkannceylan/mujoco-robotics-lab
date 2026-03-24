"""Tests for Lab 8 G1WholeBodyModel: model loading, FK, Jacobians, CoM, dynamics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab8_common import NQ, NV, NUM_ACTUATED, NQ_FREEFLYER, Q_STAND, TORSO_HEIGHT
from lab8_common import LEFT_FOOT_FRAME, RIGHT_FOOT_FRAME
from g1_model import G1WholeBodyModel


def _make_q_stand() -> np.ndarray:
    """Build standing configuration."""
    q = np.zeros(NQ)
    q[0:3] = [0.0, 0.0, TORSO_HEIGHT]
    q[3:7] = [0.0, 0.0, 0.0, 1.0]
    q[NQ_FREEFLYER:] = Q_STAND
    return q


class TestModelLoads(unittest.TestCase):
    """G1 whole-body model loads correctly from URDF."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1WholeBodyModel()

    def test_dimensions(self) -> None:
        """Model has expected nq, nv, na."""
        self.assertEqual(self.g1.nq, NQ)
        self.assertEqual(self.g1.nv, NV)
        self.assertEqual(self.g1.na, NUM_ACTUATED)

    def test_frame_ids_valid(self) -> None:
        """Key frames (feet) are found in the model."""
        for name in [LEFT_FOOT_FRAME, RIGHT_FOOT_FRAME]:
            fid = self.g1.model.getFrameId(name)
            self.assertLess(fid, self.g1.model.nframes,
                            f"Frame '{name}' not found")

    def test_total_mass_positive(self) -> None:
        """Total robot mass is reasonable."""
        mass = sum(self.g1.model.inertias[i].mass
                   for i in range(self.g1.model.njoints))
        self.assertGreater(mass, 10.0)
        self.assertLess(mass, 200.0)

    def test_selection_matrix_shape(self) -> None:
        """Selection matrix S has shape (na, nv)."""
        S = self.g1.S
        self.assertEqual(S.shape, (NUM_ACTUATED, NV))


class TestFKAtStand(unittest.TestCase):
    """FK at standing configuration gives reasonable positions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1WholeBodyModel()
        cls.q = _make_q_stand()

    def test_feet_below_torso(self) -> None:
        """Both feet should be below the torso."""
        lf = self.g1.get_frame_position(self.q, LEFT_FOOT_FRAME)
        rf = self.g1.get_frame_position(self.q, RIGHT_FOOT_FRAME)
        self.assertLess(lf[2], TORSO_HEIGHT)
        self.assertLess(rf[2], TORSO_HEIGHT)

    def test_feet_symmetric_y(self) -> None:
        """Feet should be roughly symmetric about y=0."""
        lf = self.g1.get_frame_position(self.q, LEFT_FOOT_FRAME)
        rf = self.g1.get_frame_position(self.q, RIGHT_FOOT_FRAME)
        np.testing.assert_allclose(lf[1], -rf[1], atol=0.10,
                                   err_msg="Feet not symmetric in Y")


class TestCoM(unittest.TestCase):
    """CoM computation gives reasonable results."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1WholeBodyModel()
        cls.q = _make_q_stand()

    def test_com_height_reasonable(self) -> None:
        """CoM should be at a reasonable height."""
        com = self.g1.compute_com(self.q)
        self.assertGreater(com[2], 0.2)
        self.assertLess(com[2], TORSO_HEIGHT + 0.2)

    def test_com_jacobian_shape(self) -> None:
        """CoM Jacobian should have shape (3, nv)."""
        J = self.g1.compute_com_jacobian(self.q)
        self.assertEqual(J.shape, (3, NV))


class TestDynamics(unittest.TestCase):
    """Mass matrix and dynamics computations."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g1 = G1WholeBodyModel()
        cls.q = _make_q_stand()
        cls.v = np.zeros(NV)

    def test_mass_matrix_shape(self) -> None:
        """M(q) has shape (nv, nv)."""
        M = self.g1.mass_matrix(self.q)
        self.assertEqual(M.shape, (NV, NV))

    def test_mass_matrix_symmetric(self) -> None:
        """M(q) is symmetric."""
        M = self.g1.mass_matrix(self.q)
        np.testing.assert_allclose(M, M.T, atol=1e-10,
                                   err_msg="M(q) not symmetric")

    def test_mass_matrix_positive_definite(self) -> None:
        """M(q) is positive definite."""
        M = self.g1.mass_matrix(self.q)
        eigenvalues = np.linalg.eigvalsh(M)
        self.assertTrue(np.all(eigenvalues > 0),
                        f"M(q) not PD. Min eigenvalue: {eigenvalues.min()}")

    def test_nonlinear_effects_shape(self) -> None:
        """h(q, v) has shape (nv,)."""
        h = self.g1.nonlinear_effects(self.q, self.v)
        self.assertEqual(h.shape, (NV,))

    def test_gravity_vector_shape(self) -> None:
        """g(q) has shape (nv,)."""
        g = self.g1.gravity_vector(self.q)
        self.assertEqual(g.shape, (NV,))


if __name__ == "__main__":
    unittest.main()
