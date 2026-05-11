"""Tests for Lab 5 grasp planner (IK + grasp configs)."""

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
    GRIPPER_TIP_OFFSET,
    NUM_JOINTS,
    PREGRASP_CLEARANCE,
    Q_HOME,
    get_ee_pose,
    load_pinocchio_model,
    get_topdown_rotation,
)
from grasp_planner import GraspConfigs, compute_grasp_configs, compute_ik


@pytest.fixture
def pin_setup():
    model, data, fid = load_pinocchio_model()
    return model, data, fid


# ---------------------------------------------------------------------------
# IK solver
# ---------------------------------------------------------------------------

class TestComputeIK:
    def test_ik_converges_to_home(self, pin_setup):
        """IK should recover Q_HOME when targeting the EE pose at Q_HOME."""
        model, data, fid = pin_setup
        x_home, R_home = get_ee_pose(model, data, fid, Q_HOME)
        q_sol = compute_ik(model, data, fid, x_home, R_home, Q_HOME)
        assert q_sol is not None, "IK should converge for Q_HOME target"
        x_achieved, _ = get_ee_pose(model, data, fid, q_sol)
        err = np.linalg.norm(x_home - x_achieved) * 1000
        assert err < 1.0, f"IK position error {err:.2f} mm > 1 mm"

    def test_ik_position_accuracy(self, pin_setup):
        """IK should reach a known reachable position within 2 mm."""
        model, data, fid = pin_setup
        R_topdown = get_topdown_rotation(model, data, fid)
        # Directly above box A
        x_tgt = BOX_A_POS + np.array([0.0, 0.0, GRIPPER_TIP_OFFSET + 0.10])
        q_sol = compute_ik(model, data, fid, x_tgt, R_topdown, Q_HOME)
        assert q_sol is not None, f"IK should converge for target {x_tgt}"
        x_achieved, _ = get_ee_pose(model, data, fid, q_sol)
        err = np.linalg.norm(x_tgt - x_achieved) * 1000
        assert err < 2.0, f"IK error {err:.2f} mm exceeds 2 mm"

    def test_ik_unreachable_returns_none(self, pin_setup):
        """IK for a target far outside workspace should return None."""
        model, data, fid = pin_setup
        R_topdown = get_topdown_rotation(model, data, fid)
        x_far = np.array([5.0, 5.0, 5.0])   # definitely unreachable
        q_sol = compute_ik(model, data, fid, x_far, R_topdown, Q_HOME, max_iter=50)
        # Either None (if detected) or large error — we just check no crash
        if q_sol is not None:
            x_achieved, _ = get_ee_pose(model, data, fid, q_sol)
            err = np.linalg.norm(x_far - x_achieved)
            assert err > 0.5, "Should not converge to unreachable target"

    def test_ik_joint_limits_respected(self, pin_setup):
        """IK solution must stay within joint limits ±2π."""
        model, data, fid = pin_setup
        R_topdown = get_topdown_rotation(model, data, fid)
        x_tgt = BOX_A_POS + np.array([0.0, 0.0, GRIPPER_TIP_OFFSET])
        q_sol = compute_ik(model, data, fid, x_tgt, R_topdown, Q_HOME)
        if q_sol is not None:
            assert np.all(q_sol >= -2 * np.pi - 1e-6)
            assert np.all(q_sol <= 2 * np.pi + 1e-6)


# ---------------------------------------------------------------------------
# Grasp configurations
# ---------------------------------------------------------------------------

class TestComputeGraspConfigs:
    @pytest.fixture
    def cfgs(self, pin_setup):
        model, data, fid = pin_setup
        return compute_grasp_configs(model, data, fid), model, data, fid

    def test_returns_grasp_configs(self, cfgs):
        gc, *_ = cfgs
        assert isinstance(gc, GraspConfigs)

    def test_q_home_correct(self, cfgs):
        gc, *_ = cfgs
        np.testing.assert_allclose(gc.q_home, Q_HOME, atol=1e-6)

    def test_configs_correct_shape(self, cfgs):
        gc, *_ = cfgs
        for name in ["q_pregrasp", "q_grasp", "q_preplace", "q_place"]:
            q = getattr(gc, name)
            assert q.shape == (NUM_JOINTS,), f"{name} has wrong shape {q.shape}"

    def test_pregrasp_above_grasp(self, cfgs):
        """Pre-grasp EE should be above grasp EE in Z."""
        gc, model, data, fid = cfgs
        _, x_pre = get_ee_pose(model, data, fid, gc.q_pregrasp)[0], \
                   get_ee_pose(model, data, fid, gc.q_pregrasp)[0]
        x_grasp = get_ee_pose(model, data, fid, gc.q_grasp)[0]
        x_preg = get_ee_pose(model, data, fid, gc.q_pregrasp)[0]
        assert x_preg[2] > x_grasp[2] + 0.10, (
            f"Pre-grasp Z ({x_preg[2]:.3f}) should be > grasp Z ({x_grasp[2]:.3f}) + 10cm"
        )

    def test_grasp_fingertip_at_box_a(self, cfgs):
        """Grasp config: fingertip centre should be within 5 mm of box A centre."""
        gc, model, data, fid = cfgs
        x_ee = get_ee_pose(model, data, fid, gc.q_grasp)[0]
        # Fingertip is GRIPPER_TIP_OFFSET below tool0 when arm points down
        fingertip_z = x_ee[2] - GRIPPER_TIP_OFFSET
        err_xy = np.linalg.norm(x_ee[:2] - BOX_A_POS[:2]) * 1000
        err_z = abs(fingertip_z - BOX_A_POS[2]) * 1000
        assert err_xy < 5.0, f"XY error to box A: {err_xy:.1f} mm"
        assert err_z < 10.0, f"Z error to box A: {err_z:.1f} mm"

    def test_place_fingertip_at_box_b(self, cfgs):
        """Place config: fingertip centre should be within 5 mm of box B."""
        gc, model, data, fid = cfgs
        x_ee = get_ee_pose(model, data, fid, gc.q_place)[0]
        fingertip_z = x_ee[2] - GRIPPER_TIP_OFFSET
        err_xy = np.linalg.norm(x_ee[:2] - BOX_B_POS[:2]) * 1000
        assert err_xy < 5.0, f"XY error to box B: {err_xy:.1f} mm"

    def test_r_topdown_z_axis_points_down(self, cfgs):
        """R_topdown Z-column should point in world -Z (or very close)."""
        gc, *_ = cfgs
        z_col = gc.R_topdown[:, 2]
        # For top-down: EE Z in world frame should point downward
        assert z_col[2] < -0.9, (
            f"R_topdown Z-column {z_col} does not point downward (expected z < -0.9)"
        )
