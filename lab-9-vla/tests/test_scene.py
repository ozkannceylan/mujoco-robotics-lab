"""Tests for Lab 9 VLA scene and data pipeline utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab9_common import load_mujoco_model, get_proprioception, get_ee_position, get_object_position


class TestSceneLoading(unittest.TestCase):
    """VLA scene loads and contains expected elements."""

    @classmethod
    def setUpClass(cls) -> None:
        import mujoco
        cls.mj_model, cls.mj_data = load_mujoco_model()
        mujoco.mj_forward(cls.mj_model, cls.mj_data)

    def test_model_loads(self) -> None:
        """Scene XML loads without error."""
        self.assertGreater(self.mj_model.nq, 0)
        self.assertGreater(self.mj_model.nv, 0)

    def test_has_actuators(self) -> None:
        """Scene has actuators for control."""
        self.assertGreater(self.mj_model.nu, 0)


class TestProprioception(unittest.TestCase):
    """Proprioception extraction."""

    @classmethod
    def setUpClass(cls) -> None:
        import mujoco
        cls.mj_model, cls.mj_data = load_mujoco_model()
        mujoco.mj_forward(cls.mj_model, cls.mj_data)

    def test_proprioception_shape(self) -> None:
        """Proprioception vector has expected shape."""
        prop = get_proprioception(self.mj_data)
        self.assertEqual(prop.ndim, 1)
        self.assertGreater(len(prop), 0)

    def test_proprioception_finite(self) -> None:
        """All proprioception values are finite."""
        prop = get_proprioception(self.mj_data)
        self.assertTrue(np.all(np.isfinite(prop)))


class TestEndEffector(unittest.TestCase):
    """End-effector position query."""

    @classmethod
    def setUpClass(cls) -> None:
        import mujoco
        cls.mj_model, cls.mj_data = load_mujoco_model()
        mujoco.mj_forward(cls.mj_model, cls.mj_data)

    def test_ee_position_shape(self) -> None:
        """EE position is 3D."""
        ee = get_ee_position(self.mj_model, self.mj_data)
        self.assertEqual(ee.shape, (3,))

    def test_ee_position_finite(self) -> None:
        """EE position values are finite."""
        ee = get_ee_position(self.mj_model, self.mj_data)
        self.assertTrue(np.all(np.isfinite(ee)))


class TestObjectPosition(unittest.TestCase):
    """Object position query."""

    @classmethod
    def setUpClass(cls) -> None:
        import mujoco
        cls.mj_model, cls.mj_data = load_mujoco_model()
        mujoco.mj_forward(cls.mj_model, cls.mj_data)

    def test_object_position_shape(self) -> None:
        """Object position is 3D."""
        pos = get_object_position(self.mj_model, self.mj_data, "red_cup")
        self.assertEqual(pos.shape, (3,))

    def test_object_on_table(self) -> None:
        """Red cup should be above the ground (on the table)."""
        pos = get_object_position(self.mj_model, self.mj_data, "red_cup")
        self.assertGreater(pos[2], 0.5, "Object should be on the table")


class TestDomainRandomizer(unittest.TestCase):
    """Domain randomizer generates valid randomizations."""

    @classmethod
    def setUpClass(cls) -> None:
        from domain_randomizer import DomainRandomizer
        import mujoco
        cls.mj_model, cls.mj_data = load_mujoco_model()
        mujoco.mj_forward(cls.mj_model, cls.mj_data)
        cls.randomizer = DomainRandomizer(seed=42)

    def test_randomize_runs(self) -> None:
        """Randomize does not raise."""
        import mujoco
        self.randomizer.randomize(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def test_randomize_changes_state(self) -> None:
        """Randomize changes object positions from default."""
        import mujoco

        # Get default position
        mj_model2, mj_data2 = load_mujoco_model()
        mujoco.mj_forward(mj_model2, mj_data2)
        default_pos = get_object_position(mj_model2, mj_data2, "red_cup").copy()

        # Randomize
        from domain_randomizer import DomainRandomizer
        r = DomainRandomizer(seed=42)
        r.randomize(mj_model2, mj_data2)
        mujoco.mj_forward(mj_model2, mj_data2)
        rand_pos = get_object_position(mj_model2, mj_data2, "red_cup").copy()

        # Object should still be on the table but in a different position
        self.assertGreater(rand_pos[2], 0.5, "Object fell off table")
        # XY should differ from default (with high probability)
        self.assertGreater(np.linalg.norm(rand_pos[:2] - default_pos[:2]), 0.001,
                           "Randomization did not change object position")


if __name__ == "__main__":
    unittest.main()
