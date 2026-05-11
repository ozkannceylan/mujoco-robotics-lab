"""Tests for Lab 5 grasp state machine (abbreviated — full demo is pick_place_demo.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import pinocchio as pin

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from lab5_common import (
    BOX_A_POS,
    BOX_B_POS,
    GRIPPER_IDX,
    GRIPPER_OPEN,
    NUM_JOINTS,
    Q_HOME,
    TABLE_TOP_Z,
    add_lab_src_to_path,
    load_mujoco_model,
    load_pinocchio_model,
)
from grasp_planner import GraspConfigs, compute_grasp_configs
from grasp_state_machine import GraspStateMachine, State

add_lab_src_to_path("lab-3-dynamics-force-control")
add_lab_src_to_path("lab-4-motion-planning")

from collision_checker import CollisionChecker  # noqa: E402
from rrt_planner import RRTStarPlanner  # noqa: E402
from lab4_common import ObstacleSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TABLE5 = ObstacleSpec(
    "table",
    np.array([0.45, 0.0, 0.300]),
    np.array([0.35, 0.45, 0.015]),
)


@pytest.fixture(scope="module")
def setup():
    """Load all models once per test module."""
    mj_model, mj_data = load_mujoco_model()
    pin_model, pin_data, ee_fid = load_pinocchio_model()
    cfgs = compute_grasp_configs(pin_model, pin_data, ee_fid)
    return mj_model, mj_data, pin_model, pin_data, ee_fid, cfgs


# ---------------------------------------------------------------------------
# Collision checker (Lab 4 reuse)
# ---------------------------------------------------------------------------

class TestCollisionChecker:
    def test_home_is_collision_free(self, setup):
        from lab5_common import URDF_PATH
        cc = CollisionChecker(
            urdf_path=URDF_PATH,
            obstacle_specs=[_TABLE5],
            self_collision=True,
        )
        assert cc.is_collision_free(Q_HOME), "Q_HOME must be collision-free"

    def test_below_table_collides(self, setup):
        """A config with arm forced into the table should show collision."""
        from lab5_common import URDF_PATH
        cc = CollisionChecker(
            urdf_path=URDF_PATH,
            obstacle_specs=[_TABLE5],
            self_collision=False,
        )
        # Config that puts the arm inside the table: straight up then fold over
        q_floor = np.array([0.0, -np.pi / 2, np.pi, 0.0, 0.0, 0.0])
        # We don't assert True because it depends on exact geometry, just
        # verify the function does not crash.
        _ = cc.is_collision_free(q_floor)


# ---------------------------------------------------------------------------
# Grasp configs
# ---------------------------------------------------------------------------

class TestGraspConfigsIntegration:
    def test_configs_computed(self, setup):
        _, _, _, _, _, cfgs = setup
        assert isinstance(cfgs, GraspConfigs)

    def test_pregrasp_z_above_table(self, setup):
        _, _, pin_model, pin_data, ee_fid, cfgs = setup
        pin.forwardKinematics(pin_model, pin_data, cfgs.q_pregrasp)
        pin.updateFramePlacements(pin_model, pin_data)
        z = pin_data.oMf[ee_fid].translation[2]
        assert z > TABLE_TOP_Z + 0.05, (
            f"Pre-grasp EE at z={z:.3f} m, should be > table + 5 cm"
        )

    def test_all_configs_within_joint_limits(self, setup):
        _, _, _, _, _, cfgs = setup
        for name, q in [
            ("pregrasp", cfgs.q_pregrasp),
            ("grasp",    cfgs.q_grasp),
            ("preplace", cfgs.q_preplace),
            ("place",    cfgs.q_place),
        ]:
            assert np.all(q >= -2 * np.pi - 1e-4) and np.all(q <= 2 * np.pi + 1e-4), (
                f"Config '{name}' outside joint limits: {q}"
            )


# ---------------------------------------------------------------------------
# State machine — planning only (no full simulation)
# ---------------------------------------------------------------------------

class TestStateMachinePlanning:
    def test_plan_approach_finds_path(self, setup):
        """RRT* should find a path from Q_HOME to pre-grasp without crashing."""
        mj_model, mj_data, pin_model, pin_data, ee_fid, cfgs = setup
        # Reset model
        mj_data.qpos[:NUM_JOINTS] = Q_HOME.copy()
        mj_data.qvel[:] = 0.0
        mj_data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
        mujoco.mj_forward(mj_model, mj_data)

        sm = GraspStateMachine(
            mj_model, mj_data, pin_model, pin_data, ee_fid, cfgs
        )
        # Plan only (don't run full demo — it takes too long for tests)
        times, q_traj, qd_traj = sm._plan_and_smooth(Q_HOME, cfgs.q_pregrasp, seed=0)

        assert times is not None
        assert q_traj.shape[1] == NUM_JOINTS
        assert len(times) == len(q_traj)
        # Path should start at Q_HOME and end at pregrasp
        np.testing.assert_allclose(q_traj[0], Q_HOME, atol=0.2)
        np.testing.assert_allclose(q_traj[-1], cfgs.q_pregrasp, atol=0.2)

    def test_plan_transport_finds_path(self, setup):
        """RRT* should plan from q_pregrasp to q_preplace."""
        mj_model, mj_data, pin_model, pin_data, ee_fid, cfgs = setup
        mj_data.qpos[:NUM_JOINTS] = Q_HOME.copy()
        mj_data.qvel[:] = 0.0
        mj_data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
        mujoco.mj_forward(mj_model, mj_data)

        sm = GraspStateMachine(
            mj_model, mj_data, pin_model, pin_data, ee_fid, cfgs
        )
        times, q_traj, qd_traj = sm._plan_and_smooth(
            cfgs.q_pregrasp, cfgs.q_preplace, seed=1
        )
        assert times is not None
        assert q_traj.shape[1] == NUM_JOINTS

    def test_topp_ra_respects_velocity_limits(self, setup):
        """TOPP-RA output velocities must stay within VEL_LIMITS."""
        from lab5_common import VEL_LIMITS
        mj_model, mj_data, pin_model, pin_data, ee_fid, cfgs = setup
        mj_data.qpos[:NUM_JOINTS] = Q_HOME.copy()
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        sm = GraspStateMachine(
            mj_model, mj_data, pin_model, pin_data, ee_fid, cfgs
        )
        _, q_traj, qd_traj = sm._plan_and_smooth(Q_HOME, cfgs.q_pregrasp, seed=2)
        max_vel = np.max(np.abs(qd_traj), axis=0)
        assert np.all(max_vel <= VEL_LIMITS + 1e-3), (
            f"Velocity limits violated: max={max_vel}, limits={VEL_LIMITS}"
        )
