"""Tests for Lab 7 StandingBalanceController: standing stability and CoM support polygon."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab7_common import (
    DT,
    NUM_ACTUATED,
    Q_HOME,
    load_mujoco_model,
    get_state_from_mujoco,
    build_home_q_pin,
)
from g1_model import G1Model
from balance_controller import StandingBalanceController


def _run_standing_sim(
    duration: float = 5.0,
    use_com_control: bool = True,
) -> dict:
    """Run a standing balance simulation and collect metrics.

    Args:
        duration: Simulation duration [s].
        use_com_control: If True, use full CoM control. Otherwise, use simple hold.

    Returns:
        Dictionary with time series data.
    """
    import mujoco

    mj_model, mj_data = load_mujoco_model()
    g1 = G1Model()
    controller = StandingBalanceController(g1)

    # Set initial pose to home
    mj_data.qpos[7:7 + NUM_ACTUATED] = Q_HOME.copy()
    mujoco.mj_forward(mj_model, mj_data)

    n_steps = int(duration / DT)
    torso_z_history = np.zeros(n_steps)
    com_xy_history = np.zeros((n_steps, 2))

    q_home_pin = build_home_q_pin()
    com_home = g1.compute_com(q_home_pin)

    for i in range(n_steps):
        if use_com_control:
            ctrl = controller.compute_control(mj_model, mj_data)
        else:
            ctrl = controller.compute_control_simple(mj_model, mj_data)

        mj_data.ctrl[:NUM_ACTUATED] = ctrl
        mujoco.mj_step(mj_model, mj_data)

        # Record torso height (base body z)
        torso_z_history[i] = mj_data.qpos[2]

        # Record CoM from Pinocchio
        q_pin, v_pin = get_state_from_mujoco(mj_model, mj_data)
        com = g1.compute_com(q_pin)
        com_xy_history[i] = com[:2]

    # Get final foot positions for support polygon
    q_pin_final, _ = get_state_from_mujoco(mj_model, mj_data)
    lf = g1.get_left_foot_pos(q_pin_final)
    rf = g1.get_right_foot_pos(q_pin_final)

    return {
        "torso_z": torso_z_history,
        "com_xy": com_xy_history,
        "left_foot": lf,
        "right_foot": rf,
        "com_home_xy": com_home[:2],
    }


class TestStandingStable(unittest.TestCase):
    """Robot stands for 5 seconds without falling."""

    _results = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._results = _run_standing_sim(duration=5.0, use_com_control=True)

    @pytest.mark.slow
    def test_torso_stays_above_threshold(self) -> None:
        """Torso height stays above 0.6m for the entire duration."""
        min_z = np.min(self._results["torso_z"])
        self.assertGreater(min_z, 0.6,
                           f"Torso dropped to {min_z:.3f}m (threshold: 0.6m)")

    @pytest.mark.slow
    def test_torso_height_stable(self) -> None:
        """Torso height does not deviate much from initial."""
        z = self._results["torso_z"]
        # After settling (first 0.5s), check stability
        settle_idx = int(0.5 / DT)
        z_settled = z[settle_idx:]
        z_range = np.max(z_settled) - np.min(z_settled)
        self.assertLess(z_range, 0.15,
                        f"Torso height range too large: {z_range:.3f}m")

    @pytest.mark.slow
    def test_no_falling(self) -> None:
        """Torso height at end should be reasonable (robot did not fall)."""
        final_z = self._results["torso_z"][-1]
        self.assertGreater(final_z, 0.5,
                           f"Robot may have fallen: final torso z={final_z:.3f}m")


class TestCoMInsideSupport(unittest.TestCase):
    """CoM projection stays within the support polygon during standing."""

    _results = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._results = _run_standing_sim(duration=5.0, use_com_control=True)

    @pytest.mark.slow
    def test_com_inside_support_polygon(self) -> None:
        """CoM XY projection stays inside the support polygon.

        The support polygon for double support is the convex hull
        of both foot contact patches. We use a simplified rectangular
        bounding box with margin.
        """
        lf = self._results["left_foot"]
        rf = self._results["right_foot"]
        com_xy = self._results["com_xy"]

        # Build a bounding box from foot positions with margin
        margin = 0.10  # generous margin for foot contact patch size
        x_min = min(lf[0], rf[0]) - margin
        x_max = max(lf[0], rf[0]) + margin
        y_min = min(lf[1], rf[1]) - margin
        y_max = max(lf[1], rf[1]) + margin

        # Check all CoM positions (after settling)
        settle_idx = int(0.5 / DT)
        com_settled = com_xy[settle_idx:]

        x_inside = np.all((com_settled[:, 0] >= x_min) & (com_settled[:, 0] <= x_max))
        y_inside = np.all((com_settled[:, 1] >= y_min) & (com_settled[:, 1] <= y_max))

        self.assertTrue(x_inside,
                        f"CoM X left support polygon. "
                        f"Range: [{com_settled[:, 0].min():.3f}, {com_settled[:, 0].max():.3f}], "
                        f"Bounds: [{x_min:.3f}, {x_max:.3f}]")
        self.assertTrue(y_inside,
                        f"CoM Y left support polygon. "
                        f"Range: [{com_settled[:, 1].min():.3f}, {com_settled[:, 1].max():.3f}], "
                        f"Bounds: [{y_min:.3f}, {y_max:.3f}]")

    @pytest.mark.slow
    def test_com_near_center(self) -> None:
        """CoM XY should stay near the center of support (mean over time)."""
        com_xy = self._results["com_xy"]
        settle_idx = int(1.0 / DT)
        com_mean = np.mean(com_xy[settle_idx:], axis=0)

        lf = self._results["left_foot"]
        rf = self._results["right_foot"]
        center = np.array([(lf[0] + rf[0]) / 2, (lf[1] + rf[1]) / 2])

        dist = np.linalg.norm(com_mean - center)
        self.assertLess(dist, 0.15,
                        f"Mean CoM {com_mean} too far from center {center}: {dist:.3f}m")


if __name__ == "__main__":
    unittest.main()
