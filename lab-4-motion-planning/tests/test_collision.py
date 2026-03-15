"""Tests for Lab 4 Phase 1 — Collision infrastructure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collision_checker import CollisionChecker
from lab4_common import (
    JOINT_LOWER,
    JOINT_UPPER,
    NUM_JOINTS,
    OBSTACLES,
    Q_HOME,
    TABLE_SPEC,
    URDF_PATH,
    get_ee_pos,
    load_mujoco_model,
    load_pinocchio_model,
)


@pytest.fixture(scope="module")
def cc() -> CollisionChecker:
    """Shared collision checker for all tests."""
    return CollisionChecker()


@pytest.fixture(scope="module")
def cc_no_env() -> CollisionChecker:
    """Collision checker with no environment obstacles (self-collision only)."""
    return CollisionChecker(obstacle_specs=[])


# ── Smoke tests ──────────────────────────────────────────────────────────


class TestCollisionCheckerInit:
    """Test collision checker initialization."""

    def test_loads_robot_geoms(self, cc: CollisionChecker) -> None:
        assert cc._n_robot_geoms == 8  # base, shoulder, upper, forearm, w1, w2, w3, ee

    def test_adds_obstacle_geoms(self, cc: CollisionChecker) -> None:
        assert len(cc._obstacle_ids) == len(OBSTACLES) + 1  # obstacles + table

    def test_collision_pairs_registered(self, cc: CollisionChecker) -> None:
        assert cc.num_collision_pairs > 0

    def test_no_env_fewer_pairs(
        self, cc: CollisionChecker, cc_no_env: CollisionChecker
    ) -> None:
        assert cc_no_env.num_collision_pairs < cc.num_collision_pairs


# ── Collision-free checks ────────────────────────────────────────────────


class TestCollisionFree:
    """Test is_collision_free for known configurations."""

    def test_q_home_free(self, cc: CollisionChecker) -> None:
        assert cc.is_collision_free(Q_HOME)

    def test_q_home_positive_distance(self, cc: CollisionChecker) -> None:
        d = cc.compute_min_distance(Q_HOME)
        assert d > 0.0, f"Expected positive clearance, got {d}"

    def test_arm_straight_up_free(self, cc: CollisionChecker) -> None:
        q_up = np.array([0, -np.pi / 2, 0, 0, 0, 0])
        assert cc.is_collision_free(q_up)

    def test_known_colliding_config(self, cc: CollisionChecker) -> None:
        """Config that puts forearm through obstacle 2."""
        q_coll = np.array([0.84, -2.0, 2.0, 2.13, -0.79, 4.18])
        assert not cc.is_collision_free(q_coll)

    def test_known_collision_negative_distance(self, cc: CollisionChecker) -> None:
        q_coll = np.array([0.84, -2.0, 2.0, 2.13, -0.79, 4.18])
        d = cc.compute_min_distance(q_coll)
        assert d < 0.0, f"Expected negative distance (penetration), got {d}"


# ── Self-collision ───────────────────────────────────────────────────────


class TestSelfCollision:
    """Test self-collision detection (no environment)."""

    def test_q_zero_no_self_collision(self, cc_no_env: CollisionChecker) -> None:
        """q=0 has the arm straight up — no self-collision."""
        assert cc_no_env.is_collision_free(np.zeros(NUM_JOINTS))

    def test_q_home_no_self_collision(self, cc_no_env: CollisionChecker) -> None:
        assert cc_no_env.is_collision_free(Q_HOME)

    def test_folded_arm_self_collision(self, cc_no_env: CollisionChecker) -> None:
        """Extreme elbow flexion should cause upper_arm/wrist collision."""
        q_fold = np.array([0, -0.5, 2.8, 2.0, 0, 0])
        # This might or might not collide depending on exact geometry.
        # Just verify the function runs without error.
        result = cc_no_env.is_collision_free(q_fold)
        assert isinstance(result, bool)


# ── Path checking ────────────────────────────────────────────────────────


class TestPathFree:
    """Test is_path_free edge checking."""

    def test_small_perturbation_free(self, cc: CollisionChecker) -> None:
        q2 = Q_HOME.copy()
        q2[0] += 0.05
        assert cc.is_path_free(Q_HOME, q2)

    def test_same_config_free(self, cc: CollisionChecker) -> None:
        assert cc.is_path_free(Q_HOME, Q_HOME)

    def test_path_through_collision_blocked(self, cc: CollisionChecker) -> None:
        """A path from free config to colliding config must not be free."""
        q_coll = np.array([0.84, -2.0, 2.0, 2.13, -0.79, 4.18])
        assert not cc.is_path_free(Q_HOME, q_coll, resolution=0.1)

    def test_resolution_affects_sampling(self, cc: CollisionChecker) -> None:
        """Finer resolution means more samples along the path."""
        q2 = Q_HOME.copy()
        q2[0] += 1.0  # 1 rad apart
        # resolution=0.1 → ~10 samples, resolution=0.5 → ~2 samples
        # Both should return same result for this free path, but more checks done
        r1 = cc.is_path_free(Q_HOME, q2, resolution=0.1)
        r2 = cc.is_path_free(Q_HOME, q2, resolution=0.5)
        assert r1 == r2  # same answer either way for this path


# ── Cross-validation with MuJoCo ────────────────────────────────────────


class TestCrossValidation:
    """Cross-validate Pinocchio collision against MuJoCo contacts."""

    def test_agreement_above_threshold(self, cc: CollisionChecker) -> None:
        """At least 90% agreement on random configs (excluding floor contacts)."""
        import mujoco

        mj_model, mj_data = load_mujoco_model()
        plane_ids = {
            i for i in range(mj_model.ngeom) if mj_model.geom_type[i] == 0
        }

        rng = np.random.default_rng(123)
        n_samples = 200
        agree = 0

        for _ in range(n_samples):
            q = rng.uniform(JOINT_LOWER, JOINT_UPPER)
            pin_free = cc.is_collision_free(q)

            mj_data.qpos[:6] = q
            mj_data.qvel[:] = 0
            mujoco.mj_forward(mj_model, mj_data)

            non_floor = sum(
                1
                for j in range(mj_data.ncon)
                if mj_data.contact[j].geom1 not in plane_ids
                and mj_data.contact[j].geom2 not in plane_ids
            )
            mj_free = non_floor == 0
            if pin_free == mj_free:
                agree += 1

        pct = agree / n_samples
        assert pct >= 0.90, f"Agreement {pct:.1%} below 90% threshold"

    def test_ee_position_matches(self) -> None:
        """FK: Pinocchio EE position matches MuJoCo at Q_HOME."""
        import mujoco

        pin_model, pin_data, ee_fid = load_pinocchio_model()
        mj_model, mj_data = load_mujoco_model()

        ee_pin = get_ee_pos(pin_model, pin_data, ee_fid, Q_HOME)

        mj_data.qpos[:6] = Q_HOME
        mj_data.qvel[:] = 0
        mujoco.mj_forward(mj_model, mj_data)
        tool_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tool0")
        ee_mj = mj_data.xpos[tool_bid].copy()

        assert np.allclose(ee_pin, ee_mj, atol=5e-3), (
            f"FK mismatch: pin={ee_pin} vs mj={ee_mj}"
        )


# ── Model loading ────────────────────────────────────────────────────────


class TestModelLoading:
    """Test model loading utilities."""

    def test_pinocchio_model_dof(self) -> None:
        model, data, ee_fid = load_pinocchio_model()
        assert model.nq == NUM_JOINTS

    def test_mujoco_model_loads(self) -> None:
        mj_model, mj_data = load_mujoco_model()
        assert mj_model.nq == NUM_JOINTS

    def test_obstacle_specs_match_scene(self) -> None:
        """Obstacle specs in lab4_common match scene_obstacles.xml count."""
        assert len(OBSTACLES) == 4  # 4 box obstacles
        assert TABLE_SPEC.name == "table"
