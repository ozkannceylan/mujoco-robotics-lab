"""Tests for Lab 4 Phase 3 — Trajectory smoothing and execution."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collision_checker import CollisionChecker
from lab4_common import ACC_LIMITS, Q_HOME, VEL_LIMITS
from rrt_planner import RRTStarPlanner
from trajectory_executor import execute_trajectory
from trajectory_smoother import parameterize_topp_ra, shortcut_path


@pytest.fixture(scope="module")
def cc() -> CollisionChecker:
    return CollisionChecker()


@pytest.fixture(scope="module")
def raw_path(cc: CollisionChecker) -> list[np.ndarray]:
    """Plan a raw RRT path for testing."""
    q_start = Q_HOME.copy()
    q_goal = Q_HOME.copy()
    q_goal[0] += 1.5
    planner = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.15)
    path = planner.plan(q_start, q_goal, max_iter=3000, rrt_star=False, seed=42)
    assert path is not None
    return path


# ── Shortcutting ─────────────────────────────────────────────────────────


class TestShortcutting:
    """Test path shortcutting."""

    def test_shortens_path(
        self, raw_path: list[np.ndarray], cc: CollisionChecker
    ) -> None:
        short = shortcut_path(raw_path, cc, max_iter=200, seed=42)
        assert len(short) <= len(raw_path)

    def test_preserves_endpoints(
        self, raw_path: list[np.ndarray], cc: CollisionChecker
    ) -> None:
        short = shortcut_path(raw_path, cc, max_iter=200, seed=42)
        assert np.allclose(short[0], raw_path[0])
        assert np.allclose(short[-1], raw_path[-1])

    def test_shortened_path_collision_free(
        self, raw_path: list[np.ndarray], cc: CollisionChecker
    ) -> None:
        short = shortcut_path(raw_path, cc, max_iter=200, seed=42)
        for i in range(len(short) - 1):
            assert cc.is_path_free(short[i], short[i + 1])

    def test_two_point_path_unchanged(self, cc: CollisionChecker) -> None:
        path = [Q_HOME.copy(), Q_HOME.copy()]
        path[1][0] += 0.1
        short = shortcut_path(path, cc, max_iter=100)
        assert len(short) == 2

    def test_seed_reproducibility(
        self, raw_path: list[np.ndarray], cc: CollisionChecker
    ) -> None:
        s1 = shortcut_path(raw_path, cc, max_iter=100, seed=77)
        s2 = shortcut_path(raw_path, cc, max_iter=100, seed=77)
        assert len(s1) == len(s2)
        for q1, q2 in zip(s1, s2):
            assert np.allclose(q1, q2)


# ── TOPP-RA ──────────────────────────────────────────────────────────────


class TestTOPPRA:
    """Test time-optimal path parameterization."""

    def test_returns_valid_trajectory(
        self, raw_path: list[np.ndarray]
    ) -> None:
        short = shortcut_path(raw_path, CollisionChecker(), max_iter=200, seed=42)
        times, pos, vel, acc = parameterize_topp_ra(
            short, VEL_LIMITS, ACC_LIMITS
        )
        assert len(times) > 1
        assert pos.shape == (len(times), 6)
        assert vel.shape == (len(times), 6)
        assert acc.shape == (len(times), 6)

    def test_starts_and_ends_correctly(
        self, raw_path: list[np.ndarray]
    ) -> None:
        short = shortcut_path(raw_path, CollisionChecker(), max_iter=200, seed=42)
        times, pos, vel, acc = parameterize_topp_ra(
            short, VEL_LIMITS, ACC_LIMITS
        )
        assert np.allclose(pos[0], short[0], atol=1e-4)
        assert np.allclose(pos[-1], short[-1], atol=1e-4)

    def test_velocity_limits_respected(
        self, raw_path: list[np.ndarray]
    ) -> None:
        short = shortcut_path(raw_path, CollisionChecker(), max_iter=200, seed=42)
        _, _, vel, _ = parameterize_topp_ra(short, VEL_LIMITS, ACC_LIMITS)
        assert np.all(np.abs(vel) <= VEL_LIMITS + 0.05), (
            f"Velocity violation: max={np.abs(vel).max(axis=0)}"
        )

    def test_acceleration_limits_respected(
        self, raw_path: list[np.ndarray]
    ) -> None:
        short = shortcut_path(raw_path, CollisionChecker(), max_iter=200, seed=42)
        _, _, _, acc = parameterize_topp_ra(short, VEL_LIMITS, ACC_LIMITS)
        assert np.all(np.abs(acc) <= ACC_LIMITS + 0.5), (
            f"Acceleration violation: max={np.abs(acc).max(axis=0)}"
        )

    def test_time_strictly_increasing(
        self, raw_path: list[np.ndarray]
    ) -> None:
        short = shortcut_path(raw_path, CollisionChecker(), max_iter=200, seed=42)
        times, _, _, _ = parameterize_topp_ra(short, VEL_LIMITS, ACC_LIMITS)
        assert np.all(np.diff(times) >= 0)


# ── Execution ────────────────────────────────────────────────────────────


class TestExecution:
    """Test trajectory execution on MuJoCo."""

    def test_tracking_error_below_threshold(
        self, raw_path: list[np.ndarray]
    ) -> None:
        cc = CollisionChecker()
        short = shortcut_path(raw_path, cc, max_iter=200, seed=42)
        times, pos, vel, _ = parameterize_topp_ra(
            short, VEL_LIMITS, ACC_LIMITS
        )
        result = execute_trajectory(times, pos, vel)
        rms = np.sqrt(np.mean((result["q_actual"] - result["q_desired"]) ** 2))
        assert rms < 0.05, f"RMS tracking error {rms:.4f} exceeds 0.05 rad"

    def test_final_position_close_to_goal(
        self, raw_path: list[np.ndarray]
    ) -> None:
        cc = CollisionChecker()
        short = shortcut_path(raw_path, cc, max_iter=200, seed=42)
        times, pos, vel, _ = parameterize_topp_ra(
            short, VEL_LIMITS, ACC_LIMITS
        )
        result = execute_trajectory(times, pos, vel)
        final_err = np.linalg.norm(result["q_actual"][-1] - raw_path[-1])
        assert final_err < 0.1, f"Final error {final_err:.4f} exceeds 0.1 rad"

    def test_result_dict_keys(self, raw_path: list[np.ndarray]) -> None:
        cc = CollisionChecker()
        short = shortcut_path(raw_path, cc, max_iter=200, seed=42)
        times, pos, vel, _ = parameterize_topp_ra(
            short, VEL_LIMITS, ACC_LIMITS
        )
        result = execute_trajectory(times, pos, vel)
        for key in ["time", "q_actual", "q_desired", "tau", "ee_pos"]:
            assert key in result
