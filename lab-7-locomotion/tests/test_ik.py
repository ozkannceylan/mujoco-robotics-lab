"""Lab 7 — Phase 3 tests: WholeBodyIK convergence and correctness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab7_common import (
    PELVIS_MJCF_Z,
    PELVIS_Z_STAND,
    build_pin_q_standing,
    load_g1_pinocchio,
    pelvis_world_to_pin_base,
)
from whole_body_ik import WholeBodyIK


def pelvis_world_from_q(q_pin: np.ndarray) -> np.ndarray:
    """Get pelvis world position from Pinocchio q."""
    p = q_pin[0:3].copy()
    p[2] += PELVIS_MJCF_Z
    return p


@pytest.fixture(scope="module")
def ik_solver():
    """Create a WholeBodyIK solver once for the test module."""
    model, data = load_g1_pinocchio()
    return WholeBodyIK(model, data, lam=1e-2, step_size=0.5)


@pytest.fixture(scope="module")
def q_stand():
    return build_pin_q_standing()


# ---------------------------------------------------------------------------
# Basic convergence
# ---------------------------------------------------------------------------


def test_ik_converges_from_standing(ik_solver, q_stand):
    """IK should converge when targets match the standing foot positions."""
    lf0, rf0 = ik_solver.get_foot_positions(q_stand)
    pelvis_world = pelvis_world_from_q(q_stand)

    q_sol, converged = ik_solver.solve(q_stand, pelvis_world, lf0, rf0, max_iter=50, tol=1e-4)
    assert converged, "IK should converge for trivial (standing) target"

    lf_sol, rf_sol = ik_solver.get_foot_positions(q_sol)
    np.testing.assert_allclose(lf_sol, lf0, atol=1e-3, err_msg="Left foot position error")
    np.testing.assert_allclose(rf_sol, rf0, atol=1e-3, err_msg="Right foot position error")


def test_ik_small_forward_step(ik_solver, q_stand):
    """IK should converge for a small forward step (5 cm)."""
    lf0, rf0 = ik_solver.get_foot_positions(q_stand)
    pelvis_world = pelvis_world_from_q(q_stand)

    lf_des = lf0 + np.array([0.05, 0.0, 0.0])
    rf_des = rf0 + np.array([0.05, 0.0, 0.0])

    q_sol, converged = ik_solver.solve(q_stand, pelvis_world, lf_des, rf_des, max_iter=200, tol=3e-4)

    lf_sol, rf_sol = ik_solver.get_foot_positions(q_sol)
    lf_err = np.linalg.norm(lf_des - lf_sol) * 1000
    rf_err = np.linalg.norm(rf_des - rf_sol) * 1000
    assert lf_err < 5.0, f"Left foot IK error too large: {lf_err:.1f} mm"
    assert rf_err < 5.0, f"Right foot IK error too large: {rf_err:.1f} mm"


def test_ik_lateral_shift(ik_solver, q_stand):
    """IK should converge when one foot is shifted laterally (swing step)."""
    lf0, rf0 = ik_solver.get_foot_positions(q_stand)
    pelvis_world = pelvis_world_from_q(q_stand)

    # Left foot swings forward and laterally
    lf_des = np.array([0.10, 0.10, 0.0])
    rf_des = rf0.copy()  # right foot stays

    q_sol, converged = ik_solver.solve(q_stand, pelvis_world, lf_des, rf_des, max_iter=300, tol=5e-4)

    lf_sol, _ = ik_solver.get_foot_positions(q_sol)
    lf_err = np.linalg.norm(lf_des - lf_sol) * 1000
    assert lf_err < 10.0, f"Left foot IK error: {lf_err:.1f} mm (limit 10 mm)"


def test_ik_pelvis_stays_fixed(ik_solver, q_stand):
    """The pelvis world position should stay at the desired position after IK."""
    from lab7_common import PELVIS_MJCF_Z
    lf0, rf0 = ik_solver.get_foot_positions(q_stand)
    # pelvis world position = q[0:3] + [0, 0, PELVIS_MJCF_Z]
    pelvis_world = q_stand[0:3].copy()
    pelvis_world[2] += PELVIS_MJCF_Z

    q_sol, _ = ik_solver.solve(q_stand, pelvis_world, lf0, rf0)

    # Check that the Pinocchio base translation places pelvis at desired world z
    from lab7_common import pelvis_world_to_pin_base
    expected_base = pelvis_world_to_pin_base(pelvis_world)
    np.testing.assert_allclose(q_sol[0:3], expected_base, atol=1e-9,
                                err_msg="Pelvis FreeFlyer translation must be fixed during IK")


def test_ik_upright_orientation(ik_solver, q_stand):
    """The pelvis orientation should remain upright after IK (quat = [0,0,0,1])."""
    lf0, rf0 = ik_solver.get_foot_positions(q_stand)
    pelvis_world = pelvis_world_from_q(q_stand)

    q_sol, _ = ik_solver.solve(q_stand, pelvis_world, lf0, rf0)

    # Pinocchio upright: q[3:7] = [qx, qy, qz, qw] = [0, 0, 0, 1]
    np.testing.assert_allclose(q_sol[3:7], [0.0, 0.0, 0.0, 1.0], atol=1e-6,
                                err_msg="Pelvis quaternion should remain upright")


# ---------------------------------------------------------------------------
# COM computation
# ---------------------------------------------------------------------------


def test_com_at_standing(ik_solver, q_stand):
    """CoM at standing should be near (0, 0, Z_C) (within 15 cm)."""
    from lab7_common import Z_C
    com = ik_solver.get_com_position(q_stand)
    pelvis_world = pelvis_world_from_q(q_stand)
    # CoM should be below the pelvis and above the ground
    assert 0.3 < com[2] < pelvis_world[2] + 0.1, (
        f"Standing CoM z = {com[2]:.3f}, pelvis at {pelvis_world[2]:.3f}"
    )
    # CoM should be within 15 cm of expected Z_C
    assert abs(com[2] - Z_C) < 0.15, f"Standing CoM z = {com[2]:.3f}, expected ~{Z_C:.3f}"


# ---------------------------------------------------------------------------
# Multiple solve calls (stateful correctness)
# ---------------------------------------------------------------------------


def test_sequential_ik_steps(ik_solver, q_stand):
    """Sequentially shifting the foot target should converge at each step."""
    lf0, rf0 = ik_solver.get_foot_positions(q_stand)
    pelvis_world = pelvis_world_from_q(q_stand)
    q_cur = q_stand.copy()

    for dx in [0.02, 0.04, 0.06, 0.08, 0.10]:
        lf_des = lf0 + np.array([dx, 0.0, 0.0])
        rf_des = rf0 + np.array([dx, 0.0, 0.0])
        q_cur, converged = ik_solver.solve(q_cur, pelvis_world, lf_des, rf_des, max_iter=200, tol=3e-4)
        lf_sol, rf_sol = ik_solver.get_foot_positions(q_cur)
        lf_err = np.linalg.norm(lf_des - lf_sol) * 1000
        assert lf_err < 10.0, f"IK diverged at dx={dx:.2f}: error={lf_err:.1f} mm"
