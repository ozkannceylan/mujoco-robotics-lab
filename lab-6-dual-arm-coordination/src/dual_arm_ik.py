"""Lab 6 — Dual-arm IK and bimanual grasp configuration planner.

Each arm gets an independent DLS IK call (same algorithm as Lab 5).
The bar is the "object-centric frame": left arm grips its -X end,
right arm grips its +X end.  Both use the same top-down EE orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

from lab6_common import (
    BAR_PICK_POS,
    BAR_PLACE_POS,
    GRIPPER_TIP_OFFSET,
    JOINT_LOWER,
    JOINT_UPPER,
    LEFT_ARM_BASE_POS,
    LEFT_GRIP_Y_OFFSET,
    NUM_JOINTS,
    PREGRASP_CLEARANCE,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    RIGHT_ARM_BASE_POS,
    RIGHT_GRIP_Y_OFFSET,
    get_topdown_rotation,
    load_both_pinocchio_models,
)


# ---------------------------------------------------------------------------
# DLS IK solver (identical to Lab 5 — parameterised, not duplicated)
# ---------------------------------------------------------------------------

def compute_ik(
    pin_model,
    pin_data,
    ee_fid: int,
    x_target: np.ndarray,
    R_target: np.ndarray,
    q_init: np.ndarray,
    max_iter: int = 400,
    tol: float = 1e-4,
    alpha: float = 0.5,
    lambda_sq: float = 1e-4,
) -> np.ndarray | None:
    """Damped Least Squares iterative IK for 6-DOF position + orientation.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
        x_target: Target EE position in world frame (3,).
        R_target: Target EE rotation matrix (3×3).
        q_init: Initial joint configuration (6,).
        max_iter: Maximum iterations.
        tol: Convergence threshold on task-space error norm.
        alpha: Step size (0, 1].
        lambda_sq: Damping factor squared for DLS regularisation.

    Returns:
        Converged joint configuration (6,) or None if IK fails.
    """
    q = q_init.copy()

    for _ in range(max_iter):
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        oMf = pin_data.oMf[ee_fid]

        pos_err = x_target - oMf.translation
        ori_err = pin.log3(R_target @ oMf.rotation.T)

        err = np.concatenate([pos_err, ori_err])
        if np.linalg.norm(err) < tol:
            return q

        pin.computeJointJacobians(pin_model, pin_data, q)
        J = pin.getFrameJacobian(
            pin_model, pin_data, ee_fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        JJt = J @ J.T + lambda_sq * np.eye(6)
        dq = alpha * J.T @ np.linalg.solve(JJt, err)

        q = pin.integrate(pin_model, q, dq)
        q = np.clip(q, JOINT_LOWER, JOINT_UPPER)

    return None


# ---------------------------------------------------------------------------
# Per-arm GraspConfigs
# ---------------------------------------------------------------------------

@dataclass
class GraspConfigs:
    """Joint configurations for pick-place phases of one arm.

    Attributes:
        q_home: Rest configuration for this arm.
        q_pregrasp: Above the grasp target.
        q_grasp: Fingertips at the grasp target.
        q_preplace: Above the place target.
        q_place: Fingertips at the place target.
        R_topdown: Top-down EE orientation (fixed throughout).
    """
    q_home:     np.ndarray
    q_pregrasp: np.ndarray
    q_grasp:    np.ndarray
    q_preplace: np.ndarray
    q_place:    np.ndarray
    R_topdown:  np.ndarray


@dataclass
class BimanualGraspConfigs:
    """Grasp configurations for both arms.

    Attributes:
        left: GraspConfigs for the left arm.
        right: GraspConfigs for the right arm.
    """
    left:  GraspConfigs
    right: GraspConfigs


# ---------------------------------------------------------------------------
# Configuration computation
# ---------------------------------------------------------------------------

def _compute_arm_configs(
    pin_model,
    pin_data,
    ee_fid: int,
    q_home: np.ndarray,
    grip_y_offset: float,
    arm_base_pos: np.ndarray,
    bar_pick_pos: np.ndarray,
    bar_place_pos: np.ndarray,
) -> GraspConfigs:
    """Compute IK joint configs for one arm given its grip Y-offset from bar centre.

    The bar is Y-oriented (long axis = Y).  Each arm grips one Y-end:
      grip_world = bar_pos + [0, grip_y_offset, 0]
      tool0_world = grip_world + [0, 0, GRIPPER_TIP_OFFSET]

    Pinocchio computes FK with the arm base at its local origin.  All world
    targets are converted to arm-local frame before passing to IK:
        local_target = world_target - arm_base_pos

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
        q_home: Home configuration (6,).
        grip_y_offset: Y offset from bar centre to grip point in world frame (m).
        arm_base_pos: World position of this arm's base (3,).
        bar_pick_pos: Bar centre world position at pick (3,).
        bar_place_pos: Bar centre world position at place (3,).

    Returns:
        GraspConfigs with all five joint configurations.

    Raises:
        RuntimeError: If IK fails or position error > 5 mm.
    """
    R_td = get_topdown_rotation(pin_model, pin_data, ee_fid, q_home)

    def _tool0_tgt(bar_pos: np.ndarray) -> np.ndarray:
        """World-frame tool0 target, then converted to arm-local frame."""
        grip_world = bar_pos + np.array([0.0, grip_y_offset, 0.0])
        tool0_world = grip_world + np.array([0.0, 0.0, GRIPPER_TIP_OFFSET])
        return tool0_world - arm_base_pos  # arm-local frame for Pinocchio

    def _pregrasp_tgt(bar_pos: np.ndarray) -> np.ndarray:
        return _tool0_tgt(bar_pos) + np.array([0.0, 0.0, PREGRASP_CLEARANCE])

    def _solve(name: str, tgt: np.ndarray, q_hint: np.ndarray) -> np.ndarray:
        q_sol = compute_ik(pin_model, pin_data, ee_fid, tgt, R_td, q_hint)
        if q_sol is None:
            raise RuntimeError(
                f"IK failed for '{name}' at {tgt}. "
                "Check target reachability and joint limits."
            )
        pin.forwardKinematics(pin_model, pin_data, q_sol)
        pin.updateFramePlacements(pin_model, pin_data)
        achieved = pin_data.oMf[ee_fid].translation
        err_m = np.linalg.norm(tgt - achieved)
        if err_m > 5e-3:
            raise RuntimeError(
                f"IK '{name}' converged but position error {err_m*1000:.1f} mm > 5 mm. "
                f"Target: {tgt}, Achieved: {achieved}"
            )
        return q_sol

    q_pg = _solve("pregrasp", _pregrasp_tgt(bar_pick_pos),  q_home)
    q_g  = _solve("grasp",    _tool0_tgt(bar_pick_pos),      q_home)

    # Seed place configs from pregrasp (same approximate region)
    q_pp = _solve("preplace", _pregrasp_tgt(bar_place_pos), q_pg)
    q_p  = _solve("place",    _tool0_tgt(bar_place_pos),    q_pg)

    return GraspConfigs(
        q_home=q_home.copy(),
        q_pregrasp=q_pg,
        q_grasp=q_g,
        q_preplace=q_pp,
        q_place=q_p,
        R_topdown=R_td,
    )


def compute_bimanual_configs(
    pin_L,
    data_L,
    ee_fid_L: int,
    pin_R,
    data_R,
    ee_fid_R: int,
    bar_pick_pos:  np.ndarray | None = None,
    bar_place_pos: np.ndarray | None = None,
) -> BimanualGraspConfigs:
    """Compute IK configurations for both arms from the bar's object-centric frame.

    Left arm grips bar -X end (grip_x_offset = LEFT_GRIP_X_OFFSET).
    Right arm grips bar +X end (grip_x_offset = RIGHT_GRIP_X_OFFSET).

    Args:
        pin_L: Pinocchio model for left arm.
        data_L: Pinocchio data for left arm.
        ee_fid_L: EE frame ID for left arm.
        pin_R: Pinocchio model for right arm.
        data_R: Pinocchio data for right arm.
        ee_fid_R: EE frame ID for right arm.
        bar_pick_pos: Bar centre at pick. Defaults to BAR_PICK_POS.
        bar_place_pos: Bar centre at place. Defaults to BAR_PLACE_POS.

    Returns:
        BimanualGraspConfigs with left and right arm configs.
    """
    if bar_pick_pos is None:
        bar_pick_pos = BAR_PICK_POS
    if bar_place_pos is None:
        bar_place_pos = BAR_PLACE_POS

    print("  Computing IK for left arm...")
    left_cfgs = _compute_arm_configs(
        pin_L, data_L, ee_fid_L,
        Q_HOME_LEFT, LEFT_GRIP_Y_OFFSET,
        LEFT_ARM_BASE_POS,
        bar_pick_pos, bar_place_pos,
    )
    print("  Computing IK for right arm...")
    right_cfgs = _compute_arm_configs(
        pin_R, data_R, ee_fid_R,
        Q_HOME_RIGHT, RIGHT_GRIP_Y_OFFSET,
        RIGHT_ARM_BASE_POS,
        bar_pick_pos, bar_place_pos,
    )
    return BimanualGraspConfigs(left=left_cfgs, right=right_cfgs)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def print_config_summary(
    cfgs: BimanualGraspConfigs,
    pin_L, data_L, ee_fid_L: int,
    pin_R, data_R, ee_fid_R: int,
) -> None:
    """Print EE positions for all configs of both arms.

    Args:
        cfgs: Bimanual grasp configurations.
        pin_L, data_L, ee_fid_L: Left arm Pinocchio triple.
        pin_R, data_R, ee_fid_R: Right arm Pinocchio triple.
    """
    def _ee_local(pm, pd, fid, q):
        pin.forwardKinematics(pm, pd, q)
        pin.updateFramePlacements(pm, pd)
        return pd.oMf[fid].translation.copy()

    print("\n=== Bimanual Grasp Configurations — EE Positions (world frame) ===")
    for arm_name, arm_cfgs, pm, pd, fid, base in [
        ("LEFT",  cfgs.left,  pin_L, data_L, ee_fid_L, LEFT_ARM_BASE_POS),
        ("RIGHT", cfgs.right, pin_R, data_R, ee_fid_R, RIGHT_ARM_BASE_POS),
    ]:
        print(f"\n  {arm_name} ARM (base at {base}):")
        for phase, q in [
            ("home",     arm_cfgs.q_home),
            ("pregrasp", arm_cfgs.q_pregrasp),
            ("grasp",    arm_cfgs.q_grasp),
            ("preplace", arm_cfgs.q_preplace),
            ("place",    arm_cfgs.q_place),
        ]:
            local = _ee_local(pm, pd, fid, q)
            world = local + base   # arm-local → world frame
            print(f"    {phase:10s}: local={local.round(3)}, world={world.round(3)}")
    print()


if __name__ == "__main__":
    (pin_L, data_L, ee_L), (pin_R, data_R, ee_R) = load_both_pinocchio_models()
    cfgs = compute_bimanual_configs(pin_L, data_L, ee_L, pin_R, data_R, ee_R)
    print_config_summary(cfgs, pin_L, data_L, ee_L, pin_R, data_R, ee_R)
