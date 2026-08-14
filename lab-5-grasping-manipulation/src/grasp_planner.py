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
# Joint-branch normalisation
# ---------------------------------------------------------------------------

def nearest_joint_branch(
    q: np.ndarray,
    q_ref: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    """Shift each revolute joint by multiples of 2π to sit closest to `q_ref`.

    All six UR5e joints are revolute with a ±2π range, so ``q_j`` and
    ``q_j ± 2π`` are kinematically identical yet several radians apart in joint
    space. DLS IK returns whichever branch its seed happened to fall into. When
    two configurations end up on *different* branches, a joint-space planner has
    to sweep the long way round — for the pick-and-place cycle that is a 5.2 rad
    shoulder_pan sweep instead of 1.0 rad, which RRT* cannot bridge inside its
    sampling bounds and iteration budget.

    For every joint independently this picks the kinematically-equivalent value
    that lies inside the joint limits and is closest to the reference.

    Args:
        q: Configuration to normalise (6,).
        q_ref: Reference configuration to stay close to (6,).
        lower: Lower joint limits. Defaults to `JOINT_LOWER`.
        upper: Upper joint limits. Defaults to `JOINT_UPPER`.

    Returns:
        Kinematically-identical configuration (6,) nearest to `q_ref`.
    """
    lo = JOINT_LOWER if lower is None else lower
    hi = JOINT_UPPER if upper is None else upper
    out = np.asarray(q, dtype=float).copy()
    ref = np.asarray(q_ref, dtype=float)
    for j in range(out.size):
        best = out[j]
        for k in (-1, 1):
            cand = out[j] + 2.0 * np.pi * k
            if cand < lo[j] - 1e-9 or cand > hi[j] + 1e-9:
                continue
            if abs(cand - ref[j]) < abs(best - ref[j]):
                best = cand
        out[j] = best
    return out


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


# Posture prior for the A-side pregrasp solve. This is the solution family the
# collision-validated pro demo (Step 5.4) uses: elbow high over the table, and
# — critically — its base-swing image over box B is ALSO collision-free against
# the Lab 4 real-geometry checker. Seeding from Q_HOME can converge to a
# different family whose B-side swing dips the upper arm into the table.
Q_PREGRASP_SEED = np.array([-2.96, -1.80, 1.50, -1.28, -1.57, -1.39])


def _is_scene_collision_free(mj_model, q: np.ndarray) -> bool:
    """Check an arm configuration against the MuJoCo scene (arm vs table/floor).

    IK solvers know nothing about obstacles (CLAUDE.md known issue) and DLS
    can converge onto a solution *branch* whose elbow or upper arm dips into
    the table even though the EE pose is perfect — exactly what broke the
    Step 6.1 transport plan. Contacts involving the grasp box are ignored:
    the box legitimately rests on the table and may sit between the fingers.

    Args:
        mj_model: MuJoCo model of the lab scene.
        q: Arm configuration (6,).

    Returns:
        True if no arm/gripper geom touches the table or floor at q.
    """
    import mujoco

    data = mujoco.MjData(mj_model)
    data.qpos[:NUM_JOINTS] = q
    mujoco.mj_forward(mj_model, data)

    box_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "grasp_box")
    for cid in range(data.ncon):
        c = data.contact[cid]
        b1 = mj_model.geom_bodyid[c.geom1]
        b2 = mj_model.geom_bodyid[c.geom2]
        if box_bid in (b1, b2):
            continue
        if c.dist < -1e-6:
            return False
    return True


def compute_grasp_configs(
    pin_model,
    pin_data,
    ee_fid: int,
    box_a_pos: np.ndarray | None = None,
    box_b_pos: np.ndarray | None = None,
    mj_model=None,
    validate_fn=None,
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
        mj_model: Optional MuJoCo scene model. When given, every IK solution
            is additionally validated collision-free against the scene, and
            alternative seeds are tried if a solution branch collides.
        validate_fn: Optional callable q -> bool giving an external collision
            verdict (e.g. the Lab 4 CollisionChecker the RRT* planner uses).
            Pass the *planner's own* checker so that no accepted config can
            later become an unreachable planning goal. Note the two checks
            are not redundant: the lab MJCF may omit arm-link collision geoms
            that the Lab 4 real-geometry model does have.

    Returns:
        GraspConfigs with all five joint configurations.

    Raises:
        RuntimeError: If IK fails (or only colliding branches are found) for
            any configuration.
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

    def _solve(
        name: str,
        x_tgt: np.ndarray,
        q_hints: list[np.ndarray],
        branch_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve IK trying seeds in order; validate accuracy and (optionally)
        scene collisions. Different seeds converge to different solution
        *families* (elbow-up vs elbow-down), so when a family collides with
        the table the next seed is tried rather than giving up.
        """
        last_reason = "no seed converged"
        rng = np.random.default_rng(0)
        random_restarts = [rng.uniform(-np.pi, np.pi, NUM_JOINTS) for _ in range(40)]
        for q_hint in [*q_hints, *random_restarts]:
            q_sol = compute_ik(pin_model, pin_data, ee_fid, x_tgt, R_topdown, q_hint)
            if q_sol is None:
                last_reason = "IK did not converge"
                continue
            pin.forwardKinematics(pin_model, pin_data, q_sol)
            pin.updateFramePlacements(pin_model, pin_data)
            achieved = pin_data.oMf[ee_fid].translation
            err_m = np.linalg.norm(x_tgt - achieved)
            if err_m > 5e-3:
                last_reason = f"position error {err_m*1000:.1f} mm > 5 mm"
                continue
            if branch_ref is not None:
                q_sol = nearest_joint_branch(q_sol, branch_ref)
            if mj_model is not None and not _is_scene_collision_free(mj_model, q_sol):
                last_reason = "solution branch collides with scene"
                continue
            if validate_fn is not None and not validate_fn(q_sol):
                last_reason = "solution branch rejected by validate_fn (planner collision truth)"
                continue
            return q_sol
        raise RuntimeError(
            f"IK failed for '{name}' target at {x_tgt}: {last_reason}. "
            "Check target reachability, joint limits, and seed list."
        )

    configs["pregrasp"] = _solve(
        "pregrasp", _pregrasp_target(box_a_pos), [Q_PREGRASP_SEED, Q_HOME]
    )
    configs["grasp"] = _solve(
        "grasp", _tool0_target(box_a_pos), [configs["pregrasp"], Q_HOME]
    )

    # Seeds for the B-side (preplace/place). The primary seed is a *base
    # swing*: rotate shoulder_pan by the world-z angle from box A to box B and
    # counter-rotate wrist_3 to keep the top-down orientation world-fixed.
    # This preserves the pregrasp solution's arm shape (elbow/shoulder
    # family), so the seed is already near the correct, collision-free branch.
    # The old mirrored-pan seed (q[0] → −q[0]) is kept as a fallback, but with
    # the MJCF-built model it converges to a family whose upper arm dips into
    # the table — which is why it is no longer first choice.
    delta_pan = float(
        np.arctan2(box_b_pos[1], box_b_pos[0]) - np.arctan2(box_a_pos[1], box_a_pos[0])
    )
    swing_plus = configs["pregrasp"].copy()
    swing_plus[0] += delta_pan
    swing_plus[5] += delta_pan
    swing_minus = configs["pregrasp"].copy()
    swing_minus[0] += delta_pan
    swing_minus[5] -= delta_pan

    q_mirror = configs["pregrasp"].copy()
    q_mirror[0] = -q_mirror[0]

    configs["preplace"] = _solve(
        "preplace",
        _pregrasp_target(box_b_pos),
        [swing_plus, swing_minus, q_mirror, configs["pregrasp"], Q_HOME],
        branch_ref=configs["pregrasp"],
    )
    configs["place"] = _solve(
        "place",
        _tool0_target(box_b_pos),
        [configs["preplace"], swing_plus, swing_minus, q_mirror],
        branch_ref=configs["preplace"],
    )

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
