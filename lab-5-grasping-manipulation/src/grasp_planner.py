"""Lab 5 — Grasp configuration planner.

Computes IK-based joint configurations for each phase of the pick-and-place
cycle using Damped Least Squares (DLS) iterative IK via Pinocchio.

All grasps use a fixed top-down EE orientation (tool Z points world -Z),
computed from FK at Q_HOME. Only the EE position varies between phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

from lab5_common import (
    BOX_A_POS,
    BOX_B_POS,
    GRIPPER_TIP_OFFSET,
    JOINT_LOWER,
    JOINT_UPPER,
    NUM_JOINTS,
    PREGRASP_CLEARANCE,
    Q_HOME,
    get_topdown_rotation,
    load_pinocchio_model,
)


# ---------------------------------------------------------------------------
# IK solver
# ---------------------------------------------------------------------------

def compute_ik(
    pin_model,
    pin_data,
    ee_fid: int,
    x_target: np.ndarray,
    R_target: np.ndarray,
    q_init: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-4,
    alpha: float = 0.5,
    lambda_sq: float = 1e-4,
) -> np.ndarray | None:
    """Damped Least Squares (DLS) iterative IK for 6-DOF position + orientation.

    Minimises task-space error:
        err = [Δposition(3), Δorientation(3)]
    using the update rule:
        Δq = J^T (J J^T + λ² I)^{-1} err

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
        x_target: Target EE position in world frame (3,).
        R_target: Target EE rotation matrix (3×3).
        q_init: Initial joint configuration (6,).
        max_iter: Maximum iterations.
        tol: Convergence threshold on task-space error norm.
        alpha: Step size (0–1].
        lambda_sq: Damping factor squared for DLS regularisation.

    Returns:
        Converged joint configuration (6,) or None if IK fails.
    """
    q = q_init.copy()

    for _ in range(max_iter):
        # FK
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        oMf = pin_data.oMf[ee_fid]

        # Position error
        pos_err = x_target - oMf.translation

        # Orientation error using Lie algebra log3 (world frame, LOCAL_WORLD_ALIGNED Jacobian).
        # The skew-symmetric formula  -0.5*(R_target.T@R_cur - R_cur.T@R_target)  has a
        # 180° singularity: any symmetric 180°-rotation gives zero anti-symmetric part,
        # so the IK silently converges to a completely wrong orientation.
        # log3(R_target @ R_cur.T) correctly reports ~π·axis for 180° errors.
        R_cur = oMf.rotation
        ori_err = pin.log3(R_target @ R_cur.T)

        err = np.concatenate([pos_err, ori_err])
        if np.linalg.norm(err) < tol:
            return q

        # Jacobian (6 × 6) in world-aligned local frame
        pin.computeJointJacobians(pin_model, pin_data, q)
        J = pin.getFrameJacobian(
            pin_model, pin_data, ee_fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        # DLS step
        JJt = J @ J.T + lambda_sq * np.eye(6)
        dq = alpha * J.T @ np.linalg.solve(JJt, err)

        # Integrate with joint limits
        q = pin.integrate(pin_model, q, dq)
        q = np.clip(q, JOINT_LOWER, JOINT_UPPER)

    return None  # IK did not converge


# ---------------------------------------------------------------------------
# Grasp configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class GraspConfigs:
    """Joint configurations for each pick-and-place phase.

    All configs assume top-down EE orientation (R_topdown).

    Attributes:
        q_home: Rest/home configuration.
        q_pregrasp: Above box A — arm moves here via RRT* first.
        q_grasp: Fingertips at box A centre — arm descends here with impedance.
        q_preplace: Above target B — arm moves here via RRT* after lift.
        q_place: Fingertips at target B centre — arm descends to release.
        R_topdown: EE orientation for all above configs.
    """
    q_home: np.ndarray
    q_pregrasp: np.ndarray
    q_grasp: np.ndarray
    q_preplace: np.ndarray
    q_place: np.ndarray
    R_topdown: np.ndarray


def compute_grasp_configs(
    pin_model,
    pin_data,
    ee_fid: int,
    box_a_pos: np.ndarray | None = None,
    box_b_pos: np.ndarray | None = None,
) -> GraspConfigs:
    """Compute IK joint configurations for the full pick-and-place sequence.

    For each target position, the tool0 EE frame must be offset above the
    box by GRIPPER_TIP_OFFSET (the distance from tool0 origin to fingertip
    centre), so that fingertips land exactly at the box centre when the arm
    reaches that config.

    tool0_target = box_pos + [0, 0, GRIPPER_TIP_OFFSET]
        (world +Z offset because gripper tip is below tool0 when arm points down)

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
        box_a_pos: Box pick position in world frame. Defaults to BOX_A_POS.
        box_b_pos: Box place position in world frame. Defaults to BOX_B_POS.

    Returns:
        GraspConfigs with all five joint configurations.

    Raises:
        RuntimeError: If IK fails for any configuration.
    """
    if box_a_pos is None:
        box_a_pos = BOX_A_POS
    if box_b_pos is None:
        box_b_pos = BOX_B_POS

    R_topdown = get_topdown_rotation(pin_model, pin_data, ee_fid)

    def _tool0_target(box_pos: np.ndarray) -> np.ndarray:
        """Offset box position to get tool0 target (fingertip centred on box)."""
        return box_pos + np.array([0.0, 0.0, GRIPPER_TIP_OFFSET])

    def _pregrasp_target(box_pos: np.ndarray) -> np.ndarray:
        """Tool0 target for pre-grasp (15 cm above grasp level)."""
        return _tool0_target(box_pos) + np.array([0.0, 0.0, PREGRASP_CLEARANCE])

    # --- Compute IK for each phase ---
    # Solve in order so we can use earlier solutions as seeds for later ones.
    # For box_b (Y-mirrored side), seed from the box_a pregrasp solution with
    # shoulder_pan negated — this keeps the IK in a continuous branch and avoids
    # the solver getting stuck in a local minimum far from the target.

    configs: dict[str, np.ndarray] = {}

    def _solve(name: str, x_tgt: np.ndarray, q_hint: np.ndarray) -> np.ndarray:
        q_sol = compute_ik(pin_model, pin_data, ee_fid, x_tgt, R_topdown, q_hint)
        if q_sol is None:
            raise RuntimeError(
                f"IK failed for '{name}' target at {x_tgt}. "
                "Check target reachability and joint limits."
            )
        pin.forwardKinematics(pin_model, pin_data, q_sol)
        pin.updateFramePlacements(pin_model, pin_data)
        achieved = pin_data.oMf[ee_fid].translation
        err_m = np.linalg.norm(x_tgt - achieved)
        if err_m > 5e-3:
            raise RuntimeError(
                f"IK for '{name}' converged but position error {err_m*1000:.1f} mm "
                f"exceeds 5 mm. Target: {x_tgt}, Achieved: {achieved}"
            )
        return q_sol

    configs["pregrasp"] = _solve("pregrasp", _pregrasp_target(box_a_pos), Q_HOME)
    configs["grasp"]    = _solve("grasp",    _tool0_target(box_a_pos),    Q_HOME)

    # Seed preplace/place from pregrasp with shoulder_pan mirrored (joint 0)
    q_hint_b = configs["pregrasp"].copy()
    q_hint_b[0] = -q_hint_b[0]
    configs["preplace"] = _solve("preplace", _pregrasp_target(box_b_pos), q_hint_b)
    configs["place"]    = _solve("place",    _tool0_target(box_b_pos),    q_hint_b)

    return GraspConfigs(
        q_home=Q_HOME.copy(),
        q_pregrasp=configs["pregrasp"],
        q_grasp=configs["grasp"],
        q_preplace=configs["preplace"],
        q_place=configs["place"],
        R_topdown=R_topdown,
    )


# ---------------------------------------------------------------------------
# Convenience: print config summary
# ---------------------------------------------------------------------------

def print_config_summary(cfgs: GraspConfigs, pin_model, pin_data, ee_fid: int) -> None:
    """Print EE positions for each configuration in GraspConfigs.

    Args:
        cfgs: Grasp configurations.
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
    """
    def _ee_pos(q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        return pin_data.oMf[ee_fid].translation.copy()

    print("\n=== Grasp Configurations — EE Positions (tool0 origin) ===")
    for name, q in [
        ("home",      cfgs.q_home),
        ("pregrasp",  cfgs.q_pregrasp),
        ("grasp",     cfgs.q_grasp),
        ("preplace",  cfgs.q_preplace),
        ("place",     cfgs.q_place),
    ]:
        pos = _ee_pos(q)
        print(f"  {name:10s}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
    print()


if __name__ == "__main__":
    model, data, fid = load_pinocchio_model()
    cfgs = compute_grasp_configs(model, data, fid)
    print_config_summary(cfgs, model, data, fid)
