"""Tests for Lab 3 Phase 3: Force Control & Contact.

Tests hybrid position-force controller and contact force measurement.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from c1_force_control import (
    HybridGains,
    Q_ABOVE_TABLE,
    compute_hybrid_torque,
    get_ee_and_table_ids,
    read_contact_force_z,
    run_hybrid_force_sim,
)
from c2_line_trace import generate_line_trajectory, run_line_trace
from lab3_common import NUM_JOINTS, SCENE_TABLE_PATH, load_mujoco_model


# ---------------------------------------------------------------------------
# Contact detection tests
# ---------------------------------------------------------------------------

class TestContactDetection:
    """Tests for contact force reading and geom ID lookup."""

    def test_get_ee_and_table_ids(self) -> None:
        """EE body IDs and table geom ID are valid."""
        mj_model, _ = load_mujoco_model(SCENE_TABLE_PATH)
        ee_ids, table_gid = get_ee_and_table_ids(mj_model)
        assert len(ee_ids) > 0
        assert table_gid >= 0
        assert all(bid >= 0 for bid in ee_ids)

    def test_no_contact_at_home(self) -> None:
        """No contact force when arm is above table."""
        import mujoco

        mj_model, mj_data = load_mujoco_model(SCENE_TABLE_PATH)
        ee_ids, table_gid = get_ee_and_table_ids(mj_model)

        mj_data.qpos[:NUM_JOINTS] = Q_ABOVE_TABLE
        mj_data.qvel[:NUM_JOINTS] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        fz = read_contact_force_z(mj_model, mj_data, ee_ids, table_gid)
        assert fz == 0.0, f"Expected no contact force, got {fz}"


# ---------------------------------------------------------------------------
# Trajectory generation tests
# ---------------------------------------------------------------------------

class TestTrajectory:
    """Tests for line trajectory generation."""

    def test_start_and_end(self) -> None:
        """Trajectory starts at start and ends at end."""
        xy_start = np.array([0.3, 0.1])
        xy_end = np.array([0.5, -0.1])
        pos, vel = generate_line_trajectory(xy_start, xy_end, 2.0)

        assert np.allclose(pos[0], xy_start, atol=1e-10)
        assert np.allclose(pos[-1], xy_end, atol=1e-10)

    def test_zero_endpoint_velocity(self) -> None:
        """Velocity is zero at start and end (5th-order polynomial)."""
        pos, vel = generate_line_trajectory(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), 2.0
        )
        assert np.allclose(vel[0], 0.0, atol=1e-10)
        assert np.allclose(vel[-1], 0.0, atol=1e-10)

    def test_monotonic_x(self) -> None:
        """X position is monotonically increasing for X-aligned line."""
        pos, _ = generate_line_trajectory(
            np.array([0.3, 0.0]), np.array([0.5, 0.0]), 2.0
        )
        dx = np.diff(pos[:, 0])
        assert np.all(dx >= -1e-15), "X should be monotonically increasing"


# ---------------------------------------------------------------------------
# Hybrid controller tests
# ---------------------------------------------------------------------------

class TestHybridForceControl:
    """Tests for hybrid position-force controller."""

    def test_static_force_convergence(self) -> None:
        """Force converges to target within ±1N at a static XY position."""
        results = run_hybrid_force_sim(
            xy_des=np.array([0.4, 0.0]),
            gains=HybridGains(F_desired=5.0),
            approach_height=0.30,
            duration=8.0,
        )

        phase = results["phase"]
        hybrid_mask = phase == 1
        assert np.any(hybrid_mask), "Hybrid phase never reached"

        # After settling (0.5s into hybrid)
        hybrid_start = np.argmax(hybrid_mask)
        settle_idx = min(hybrid_start + 500, len(results["force_z"]) - 1)
        fz_settled = results["force_z"][settle_idx:]

        mean_force = np.mean(fz_settled)
        assert abs(mean_force - 5.0) < 1.0, f"Mean force {mean_force:.2f}N not near 5N"

        pct_in_band = np.mean(np.abs(fz_settled - 5.0) < 1.0) * 100
        assert pct_in_band > 90, f"Only {pct_in_band:.1f}% within ±1N (need >90%)"

    def test_xy_position_hold(self) -> None:
        """XY position stays within 5mm of target during force control."""
        xy_des = np.array([0.4, 0.0])
        results = run_hybrid_force_sim(
            xy_des=xy_des,
            gains=HybridGains(F_desired=5.0),
            approach_height=0.30,
            duration=8.0,
        )

        phase = results["phase"]
        hybrid_mask = phase == 1
        if not np.any(hybrid_mask):
            pytest.skip("Hybrid phase not reached")

        hybrid_start = np.argmax(hybrid_mask)
        settle_idx = min(hybrid_start + 500, len(results["force_z"]) - 1)
        xy_err = np.sqrt(
            (results["ee_pos"][settle_idx:, 0] - xy_des[0]) ** 2
            + (results["ee_pos"][settle_idx:, 1] - xy_des[1]) ** 2
        ) * 1000
        assert np.max(xy_err) < 5.0, f"Max XY error {np.max(xy_err):.1f}mm > 5mm"

    def test_contact_established(self) -> None:
        """Contact is established during approach phase."""
        results = run_hybrid_force_sim(
            xy_des=np.array([0.4, 0.0]),
            approach_height=0.30,
            duration=6.0,
        )
        assert np.any(results["phase"] == 1), "Contact never established"


# ---------------------------------------------------------------------------
# Line trace tests
# ---------------------------------------------------------------------------

class TestLineTrace:
    """Tests for capstone constant-force line tracing."""

    def test_trace_force_in_band(self) -> None:
        """Force stays within ±1N for >90% of trace phase."""
        results = run_line_trace(
            xy_start=np.array([0.40, 0.0]),
            xy_end=np.array([0.45, 0.0]),
            gains=HybridGains(K_p=2000.0, K_d=100.0, F_desired=5.0),
            approach_height=0.30,
            settle_time=1.5,
            trace_duration=6.0,
            post_time=0.5,
        )

        trace_mask = results["phase"] == 2
        assert np.any(trace_mask), "Trace phase never reached"

        fz_trace = results["force_z"][trace_mask]
        pct_in_band = np.mean(np.abs(fz_trace - 5.0) < 1.0) * 100
        assert pct_in_band > 90, f"Only {pct_in_band:.1f}% within ±1N (need >90%)"

    def test_trace_xy_tracking(self) -> None:
        """XY tracking error stays below 5mm during trace."""
        results = run_line_trace(
            xy_start=np.array([0.40, 0.0]),
            xy_end=np.array([0.45, 0.0]),
            gains=HybridGains(K_p=2000.0, K_d=100.0, F_desired=5.0),
            approach_height=0.30,
            settle_time=1.5,
            trace_duration=6.0,
            post_time=0.5,
        )

        trace_mask = results["phase"] == 2
        if not np.any(trace_mask):
            pytest.skip("Trace phase not reached")

        xy_actual = results["ee_pos"][trace_mask, :2]
        xy_desired = results["xy_des"][trace_mask]
        xy_err = np.linalg.norm(xy_actual - xy_desired, axis=1) * 1000
        assert np.max(xy_err) < 5.0, f"Max XY error {np.max(xy_err):.1f}mm > 5mm"
