"""Tests for Lab 3 gravity compensation controller."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab3_common import Q_HOME, NUM_JOINTS
from a2_gravity_compensation import run_gravity_comp_sim


class TestGravityCompHold(unittest.TestCase):
    """Test that gravity compensation holds position."""

    def test_hold_position_home(self) -> None:
        """Robot should hold Q_HOME with < 0.01 rad drift over 3 seconds."""
        results = run_gravity_comp_sim(duration=3.0, q_init=Q_HOME)
        max_err_rad = np.max(np.abs(results["q_error"]))
        self.assertLess(max_err_rad, 0.01,
                        f"Position drift too large: {max_err_rad:.6f} rad")

    def test_hold_position_zeros(self) -> None:
        """Robot should hold Q_ZEROS with < 0.01 rad drift over 3 seconds."""
        results = run_gravity_comp_sim(duration=3.0, q_init=np.zeros(NUM_JOINTS))
        max_err_rad = np.max(np.abs(results["q_error"]))
        self.assertLess(max_err_rad, 0.01,
                        f"Position drift too large: {max_err_rad:.6f} rad")

    def test_hold_position_arbitrary(self) -> None:
        """Robot should hold an arbitrary config with < 0.01 rad drift."""
        q = np.array([0.5, -1.0, 1.2, -0.8, 0.3, -0.5])
        results = run_gravity_comp_sim(duration=3.0, q_init=q)
        max_err_rad = np.max(np.abs(results["q_error"]))
        self.assertLess(max_err_rad, 0.01,
                        f"Position drift too large: {max_err_rad:.6f} rad")


class TestGravityCompPerturbation(unittest.TestCase):
    """Test that arm settles after perturbation."""

    def test_perturbation_settles(self) -> None:
        """After a perturbation pulse, velocity should decay due to joint damping."""
        results = run_gravity_comp_sim(
            duration=5.0,
            q_init=Q_HOME,
            perturb_at=0.5,
            perturb_torque=np.array([0.0, 15.0, 10.0, 5.0, 0.0, 0.0]),
            perturb_duration=0.2,
        )
        # Check velocity in last 200 steps (0.2s) is low
        final_max_vel = np.max(np.abs(results["qd"][-200:]))
        self.assertLess(final_max_vel, 0.5,
                        f"Final velocity too high: {final_max_vel:.4f} rad/s")

    def test_gravity_comp_torques_nonzero(self) -> None:
        """Controller should apply nonzero torques (gravity exists)."""
        results = run_gravity_comp_sim(duration=1.0)
        max_tau = np.max(np.abs(results["tau"]))
        self.assertGreater(max_tau, 1.0,
                           f"Torques too small: {max_tau:.4f} Nm")


if __name__ == "__main__":
    unittest.main()
