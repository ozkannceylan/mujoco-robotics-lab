"""Tests for Lab 4 Phase 2 — RRT / RRT* planner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collision_checker import CollisionChecker
from lab4_common import JOINT_LOWER, JOINT_UPPER, MEDIA_DIR, Q_HOME
from rrt_planner import RRTNode, RRTStarPlanner, visualize_plan


@pytest.fixture(scope="module")
def cc() -> CollisionChecker:
    return CollisionChecker()


@pytest.fixture(scope="module")
def goals() -> tuple[np.ndarray, np.ndarray]:
    """Standard start/goal pair for testing."""
    q_start = Q_HOME.copy()
    q_goal = Q_HOME.copy()
    q_goal[0] += 1.5
    return q_start, q_goal


# ── RRT basic ────────────────────────────────────────────────────────────


class TestRRTBasic:
    """Test basic RRT planner."""

    def test_finds_path(
        self, cc: CollisionChecker, goals: tuple[np.ndarray, np.ndarray]
    ) -> None:
        q_start, q_goal = goals
        planner = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.15)
        path = planner.plan(q_start, q_goal, max_iter=3000, rrt_star=False, seed=42)
        assert path is not None, "RRT failed to find a path"
        assert len(path) >= 2

    def test_path_starts_and_ends_correctly(
        self, cc: CollisionChecker, goals: tuple[np.ndarray, np.ndarray]
    ) -> None:
        q_start, q_goal = goals
        planner = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.15)
        path = planner.plan(q_start, q_goal, max_iter=3000, rrt_star=False, seed=42)
        assert path is not None
        assert np.allclose(path[0], q_start, atol=1e-8)
        assert np.allclose(path[-1], q_goal, atol=1e-8)

    def test_all_waypoints_collision_free(
        self, cc: CollisionChecker, goals: tuple[np.ndarray, np.ndarray]
    ) -> None:
        q_start, q_goal = goals
        planner = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.15)
        path = planner.plan(q_start, q_goal, max_iter=3000, rrt_star=False, seed=42)
        assert path is not None
        for i, q in enumerate(path):
            assert cc.is_collision_free(q), f"Waypoint {i} in collision"

    def test_all_edges_collision_free(
        self, cc: CollisionChecker, goals: tuple[np.ndarray, np.ndarray]
    ) -> None:
        q_start, q_goal = goals
        planner = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.15)
        path = planner.plan(q_start, q_goal, max_iter=3000, rrt_star=False, seed=42)
        assert path is not None
        for i in range(len(path) - 1):
            assert cc.is_path_free(path[i], path[i + 1]), f"Edge {i}-{i+1} in collision"

    def test_returns_none_for_colliding_start(self, cc: CollisionChecker) -> None:
        q_coll = np.array([0.84, -2.0, 2.0, 2.13, -0.79, 4.18])
        planner = RRTStarPlanner(cc)
        path = planner.plan(q_coll, Q_HOME, max_iter=100, rrt_star=False)
        assert path is None

    def test_returns_none_for_colliding_goal(self, cc: CollisionChecker) -> None:
        q_coll = np.array([0.84, -2.0, 2.0, 2.13, -0.79, 4.18])
        planner = RRTStarPlanner(cc)
        path = planner.plan(Q_HOME, q_coll, max_iter=100, rrt_star=False)
        assert path is None


# ── RRT* ─────────────────────────────────────────────────────────────────


class TestRRTStar:
    """Test RRT* with rewiring."""

    def test_finds_path(
        self, cc: CollisionChecker, goals: tuple[np.ndarray, np.ndarray]
    ) -> None:
        q_start, q_goal = goals
        planner = RRTStarPlanner(
            cc, step_size=0.3, goal_bias=0.1, rewire_radius=1.0
        )
        path = planner.plan(q_start, q_goal, max_iter=3000, rrt_star=True, seed=42)
        assert path is not None

    def test_rrt_star_shorter_than_rrt(
        self, cc: CollisionChecker, goals: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """RRT* should produce a shorter or equal path compared to RRT."""
        q_start, q_goal = goals

        planner_rrt = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.15)
        path_rrt = planner_rrt.plan(
            q_start, q_goal, max_iter=3000, rrt_star=False, seed=42
        )

        planner_star = RRTStarPlanner(
            cc, step_size=0.3, goal_bias=0.1, rewire_radius=1.0
        )
        path_star = planner_star.plan(
            q_start, q_goal, max_iter=3000, rrt_star=True, seed=42
        )

        assert path_rrt is not None and path_star is not None

        def path_cost(p: list[np.ndarray]) -> float:
            return sum(np.linalg.norm(p[i + 1] - p[i]) for i in range(len(p) - 1))

        cost_rrt = path_cost(path_rrt)
        cost_star = path_cost(path_star)
        # RRT* should generally be shorter, allow small margin
        assert cost_star <= cost_rrt * 1.1, (
            f"RRT* cost {cost_star:.3f} > RRT cost {cost_rrt:.3f} * 1.1"
        )

    def test_tree_has_nodes(
        self, cc: CollisionChecker, goals: tuple[np.ndarray, np.ndarray]
    ) -> None:
        q_start, q_goal = goals
        planner = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.1)
        planner.plan(q_start, q_goal, max_iter=1000, rrt_star=True, seed=42)
        assert len(planner.tree) > 1

    def test_seed_reproducibility(
        self, cc: CollisionChecker, goals: tuple[np.ndarray, np.ndarray]
    ) -> None:
        q_start, q_goal = goals
        p1 = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.1)
        path1 = p1.plan(q_start, q_goal, max_iter=2000, rrt_star=True, seed=99)

        p2 = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.1)
        path2 = p2.plan(q_start, q_goal, max_iter=2000, rrt_star=True, seed=99)

        assert path1 is not None and path2 is not None
        assert len(path1) == len(path2)
        for q1, q2 in zip(path1, path2):
            assert np.allclose(q1, q2)


# ── Visualization ────────────────────────────────────────────────────────


class TestVisualization:
    """Test visualization runs without error."""

    def test_visualize_saves_file(
        self, cc: CollisionChecker, goals: tuple[np.ndarray, np.ndarray]
    ) -> None:
        q_start, q_goal = goals
        planner = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.15)
        path = planner.plan(q_start, q_goal, max_iter=2000, rrt_star=False, seed=42)
        assert path is not None

        ee_fid = cc.model.getFrameId("ee_link")
        out = MEDIA_DIR / "test_viz.png"
        visualize_plan(planner, path, ee_fid, save_path=out)
        assert out.exists()
        out.unlink()  # cleanup
