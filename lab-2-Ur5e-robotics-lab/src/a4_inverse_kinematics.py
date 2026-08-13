"""Phase 2.2 — Inverse Kinematics for the UR5e.

This script implements:
  1. Numerical IK using Jacobian pseudo-inverse (full 6-DOF: position + orientation)
  2. Damped Least Squares IK with adaptive lambda
  3. IK solution validation via FK roundtrip
  4. Benchmark on 50 random reachable targets
  5. MuJoCo verification: solve IK then set joint angles in the simulator

Run:
    python3 lab-2-Ur5e-robotics-lab/src/a4_inverse_kinematics.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pinocchio as pin

from ur5e_common import (
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    NUM_JOINTS,
    Q_HOME,
    Q_ZEROS,
    get_mj_ee_site_id,
    load_mujoco_model,
    load_pinocchio_model,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IKResult:
    """Result of an IK solve attempt."""

    success: bool
    q: np.ndarray
    iterations: int
    final_error: float
    position_error: float
    orientation_error: float
    elapsed_s: float


# ---------------------------------------------------------------------------
# Pseudo-inverse IK (position + orientation)
# ---------------------------------------------------------------------------


def ik_pseudoinverse(
    model: pin.Model,
    data: pin.Data,
    ee_fid: int,
    p_target: np.ndarray,
    R_target: np.ndarray,
    q_init: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-4,
    alpha: float = 0.5,
) -> IKResult:
    """Solve 6-DOF IK using the Jacobian pseudo-inverse.

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        ee_fid: End-effector frame ID.
        p_target: Desired EE position (3,).
        R_target: Desired EE rotation matrix (3, 3).
        q_init: Initial joint configuration (NUM_JOINTS,).
        max_iter: Maximum iterations.
        tol: Convergence tolerance on the 6D error norm.
        alpha: Step size.

    Returns:
        IKResult with solution details.
    """
    q = q_init.copy().astype(float)
    t0 = time.perf_counter()

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        p_current = data.oMf[ee_fid].translation
        R_current = data.oMf[ee_fid].rotation

        e_pos = p_target - p_current
        e_rot = pin.log3(R_target @ R_current.T)
        e = np.concatenate([e_pos, e_rot])

        err_norm = np.linalg.norm(e)
        if err_norm < tol:
            elapsed = time.perf_counter() - t0
            return IKResult(
                success=True,
                q=q.copy(),
                iterations=i,
                final_error=err_norm,
                position_error=np.linalg.norm(e_pos),
                orientation_error=np.linalg.norm(e_rot),
                elapsed_s=elapsed,
            )

        J = pin.computeFrameJacobian(
            model, data, q, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # Moore-Penrose pseudo-inverse
        dq = np.linalg.pinv(J) @ e
        q = q + alpha * dq
        q = np.clip(q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    elapsed = time.perf_counter() - t0
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    e_pos = p_target - data.oMf[ee_fid].translation
    e_rot = pin.log3(R_target @ data.oMf[ee_fid].rotation.T)
    return IKResult(
        success=False,
        q=q.copy(),
        iterations=max_iter,
        final_error=np.linalg.norm(np.concatenate([e_pos, e_rot])),
        position_error=np.linalg.norm(e_pos),
        orientation_error=np.linalg.norm(e_rot),
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Damped Least Squares IK with adaptive lambda
# ---------------------------------------------------------------------------


def ik_damped_least_squares(
    model: pin.Model,
    data: pin.Data,
    ee_fid: int,
    p_target: np.ndarray,
    R_target: np.ndarray,
    q_init: np.ndarray,
    *,
    max_iter: int = 300,
    tol: float = 1e-4,
    alpha: float = 0.5,
    lam_init: float = 0.1,
    lam_min: float = 1e-4,
    lam_max: float = 1.0,
) -> IKResult:
    """Solve 6-DOF IK using Damped Least Squares with adaptive damping.

    The damping factor lambda is adapted based on error reduction:
      - If error decreases: reduce lambda (closer to pure pseudo-inverse)
      - If error increases: increase lambda (more conservative steps)

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        ee_fid: End-effector frame ID.
        p_target: Desired EE position (3,).
        R_target: Desired EE rotation matrix (3, 3).
        q_init: Initial joint configuration (NUM_JOINTS,).
        max_iter: Maximum iterations.
        tol: Convergence tolerance on the 6D error norm.
        alpha: Step size.
        lam_init: Initial damping factor.
        lam_min: Minimum damping factor.
        lam_max: Maximum damping factor.

    Returns:
        IKResult with solution details.
    """
    q = q_init.copy().astype(float)
    lam = lam_init
    prev_err = float("inf")
    t0 = time.perf_counter()

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        p_current = data.oMf[ee_fid].translation
        R_current = data.oMf[ee_fid].rotation

        e_pos = p_target - p_current
        e_rot = pin.log3(R_target @ R_current.T)
        e = np.concatenate([e_pos, e_rot])

        err_norm = np.linalg.norm(e)

        if err_norm < tol:
            elapsed = time.perf_counter() - t0
            return IKResult(
                success=True,
                q=q.copy(),
                iterations=i,
                final_error=err_norm,
                position_error=np.linalg.norm(e_pos),
                orientation_error=np.linalg.norm(e_rot),
                elapsed_s=elapsed,
            )

        # Adaptive lambda
        if err_norm < prev_err:
            lam = max(lam / 2.0, lam_min)
        else:
            lam = min(lam * 2.0, lam_max)
        prev_err = err_norm

        J = pin.computeFrameJacobian(
            model, data, q, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # Damped Least Squares: dq = J^T (J J^T + lam^2 I)^{-1} e
        JJT = J @ J.T
        dq = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(6), e)

        q = q + alpha * dq
        q = np.clip(q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    elapsed = time.perf_counter() - t0
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    e_pos = p_target - data.oMf[ee_fid].translation
    e_rot = pin.log3(R_target @ data.oMf[ee_fid].rotation.T)
    return IKResult(
        success=False,
        q=q.copy(),
        iterations=max_iter,
        final_error=np.linalg.norm(np.concatenate([e_pos, e_rot])),
        position_error=np.linalg.norm(e_pos),
        orientation_error=np.linalg.norm(e_rot),
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# FK roundtrip validation
# ---------------------------------------------------------------------------


def validate_ik_solution(
    model: pin.Model,
    data: pin.Data,
    ee_fid: int,
    q_solution: np.ndarray,
    p_target: np.ndarray,
    R_target: np.ndarray,
    pos_tol: float = 1e-3,
    rot_tol: float = 1e-2,
) -> tuple[bool, float, float]:
    """Validate an IK solution by running FK and comparing to the target.

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        ee_fid: End-effector frame ID.
        q_solution: Joint angles from IK solver.
        p_target: Target position (3,).
        R_target: Target rotation (3, 3).
        pos_tol: Position tolerance (m).
        rot_tol: Orientation tolerance (rad).

    Returns:
        Tuple of (valid, position_error, orientation_error).
    """
    pin.forwardKinematics(model, data, q_solution)
    pin.updateFramePlacements(model, data)

    p_fk = data.oMf[ee_fid].translation
    R_fk = data.oMf[ee_fid].rotation

    pos_err = np.linalg.norm(p_target - p_fk)
    rot_err = np.linalg.norm(pin.log3(R_target @ R_fk.T))

    valid = (pos_err < pos_tol) and (rot_err < rot_tol)
    return valid, pos_err, rot_err


# ---------------------------------------------------------------------------
# MuJoCo verification
# ---------------------------------------------------------------------------


def apply_ik_in_mujoco(
    p_target: np.ndarray,
    R_target: np.ndarray,
    q_init: np.ndarray | None = None,
) -> tuple[IKResult, np.ndarray]:
    """Solve IK with Pinocchio, then verify by setting joints in MuJoCo.

    Args:
        p_target: Desired EE position (3,).
        R_target: Desired EE rotation (3, 3).
        q_init: Initial guess. Defaults to Q_HOME.

    Returns:
        Tuple of (IKResult, mj_ee_pos) where mj_ee_pos is the EE position
        read back from MuJoCo after applying the IK solution.
    """
    import mujoco

    if q_init is None:
        q_init = Q_HOME.copy()

    # Solve IK with Pinocchio
    pin_model, pin_data, ee_fid = load_pinocchio_model()
    result = ik_damped_least_squares(
        pin_model, pin_data, ee_fid, p_target, R_target, q_init
    )

    # Apply in MuJoCo
    mj_model, mj_data = load_mujoco_model()
    site_id = get_mj_ee_site_id(mj_model)
    mj_data.qpos[:NUM_JOINTS] = result.q
    mujoco.mj_forward(mj_model, mj_data)
    mj_ee = mj_data.site_xpos[site_id].copy()

    return result, mj_ee


# ---------------------------------------------------------------------------
# Generate random reachable targets via FK on random configs
# ---------------------------------------------------------------------------


def generate_random_targets(
    model: pin.Model,
    data: pin.Data,
    ee_fid: int,
    count: int = 50,
    rng_seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate random reachable targets by sampling joint configs and running FK.

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        ee_fid: End-effector frame ID.
        count: Number of targets.
        rng_seed: Random seed for reproducibility.

    Returns:
        List of (p_target, R_target, q_true) tuples.
    """
    rng = np.random.default_rng(rng_seed)
    targets = []

    for _ in range(count):
        q_rand = rng.uniform(JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
        pin.forwardKinematics(model, data, q_rand)
        pin.updateFramePlacements(model, data)
        p = data.oMf[ee_fid].translation.copy()
        R = data.oMf[ee_fid].rotation.copy()
        targets.append((p, R, q_rand))

    return targets


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    model: pin.Model,
    data: pin.Data,
    ee_fid: int,
    n_targets: int = 50,
) -> list[dict]:
    """Benchmark IK on random reachable targets.

    Tests both pseudo-inverse and damped least squares methods. Reports
    success rate, mean iterations, and mean computation time.

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        ee_fid: End-effector frame ID.
        n_targets: Number of random targets.

    Returns:
        List of per-target result dicts.
    """
    targets = generate_random_targets(model, data, ee_fid, count=n_targets)
    results = []

    for idx, (p_target, R_target, q_true) in enumerate(targets):
        # Use a perturbed initial guess (not the true solution)
        rng = np.random.default_rng(idx + 1000)
        q_init = Q_HOME + rng.normal(0, 0.3, NUM_JOINTS)
        q_init = np.clip(q_init, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

        # Pseudo-inverse IK
        res_pinv = ik_pseudoinverse(
            model, data, ee_fid, p_target, R_target, q_init
        )

        # Damped Least Squares IK
        res_dls = ik_damped_least_squares(
            model, data, ee_fid, p_target, R_target, q_init
        )

        # FK roundtrip validation for DLS solution
        valid, pos_err_fk, rot_err_fk = validate_ik_solution(
            model, data, ee_fid, res_dls.q, p_target, R_target
        )

        results.append({
            "idx": idx,
            "p_target": p_target,
            "pinv_success": res_pinv.success,
            "pinv_iters": res_pinv.iterations,
            "pinv_pos_err": res_pinv.position_error,
            "pinv_rot_err": res_pinv.orientation_error,
            "pinv_time": res_pinv.elapsed_s,
            "dls_success": res_dls.success,
            "dls_iters": res_dls.iterations,
            "dls_pos_err": res_dls.position_error,
            "dls_rot_err": res_dls.orientation_error,
            "dls_time": res_dls.elapsed_s,
            "fk_valid": valid,
            "fk_pos_err": pos_err_fk,
            "fk_rot_err": rot_err_fk,
        })

    return results


def print_benchmark_summary(results: list[dict]) -> None:
    """Print a summary table from benchmark results.

    Args:
        results: Output of run_benchmark().
    """
    n = len(results)

    for method, prefix in [("Pseudo-Inverse", "pinv"), ("Damped Least Squares", "dls")]:
        successes = sum(1 for r in results if r[f"{prefix}_success"])
        iters = [r[f"{prefix}_iters"] for r in results if r[f"{prefix}_success"]]
        times = [r[f"{prefix}_time"] for r in results if r[f"{prefix}_success"]]
        pos_errs = [r[f"{prefix}_pos_err"] for r in results if r[f"{prefix}_success"]]
        rot_errs = [r[f"{prefix}_rot_err"] for r in results if r[f"{prefix}_success"]]

        print(f"\n  {method}:")
        print(f"    Success rate:      {successes}/{n} ({100 * successes / n:.1f}%)")
        if iters:
            print(f"    Mean iterations:   {np.mean(iters):.1f}")
            print(f"    Mean time:         {np.mean(times) * 1000:.2f} ms")
            print(f"    Mean pos error:    {np.mean(pos_errs):.6f} m")
            print(f"    Mean rot error:    {np.mean(rot_errs):.6f} rad")
        else:
            print("    (no successes)")

    # FK roundtrip validation stats
    fk_valid = sum(1 for r in results if r["fk_valid"])
    print(f"\n  FK roundtrip validation (DLS solutions):")
    print(f"    Valid:  {fk_valid}/{n} ({100 * fk_valid / n:.1f}%)")


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------


def demo_single_target() -> None:
    """Demonstrate IK on a single target with both methods."""
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    print("=" * 72)
    print("Part 1: Single-Target IK Demo")
    print("=" * 72)

    # Generate a target from a known configuration
    q_target = np.array([0.3, -1.2, 1.5, -1.6, 1.57, 0.0])
    pin.forwardKinematics(pin_model, pin_data, q_target)
    pin.updateFramePlacements(pin_model, pin_data)
    p_target = pin_data.oMf[ee_fid].translation.copy()
    R_target = pin_data.oMf[ee_fid].rotation.copy()

    print(f"\n  Target position:  [{p_target[0]:.4f}, {p_target[1]:.4f}, {p_target[2]:.4f}]")
    print(f"  Initial guess:    Q_HOME")
    print()

    # Pseudo-inverse
    res_pinv = ik_pseudoinverse(
        pin_model, pin_data, ee_fid, p_target, R_target, Q_HOME.copy()
    )
    print(f"  Pseudo-Inverse IK:")
    print(f"    Success:     {res_pinv.success}")
    print(f"    Iterations:  {res_pinv.iterations}")
    print(f"    Pos error:   {res_pinv.position_error:.6f} m")
    print(f"    Rot error:   {res_pinv.orientation_error:.6f} rad")
    print(f"    Time:        {res_pinv.elapsed_s * 1000:.2f} ms")

    # Damped Least Squares
    res_dls = ik_damped_least_squares(
        pin_model, pin_data, ee_fid, p_target, R_target, Q_HOME.copy()
    )
    print(f"\n  Damped Least Squares IK:")
    print(f"    Success:     {res_dls.success}")
    print(f"    Iterations:  {res_dls.iterations}")
    print(f"    Pos error:   {res_dls.position_error:.6f} m")
    print(f"    Rot error:   {res_dls.orientation_error:.6f} rad")
    print(f"    Time:        {res_dls.elapsed_s * 1000:.2f} ms")

    # FK roundtrip
    valid, pe, re = validate_ik_solution(
        pin_model, pin_data, ee_fid, res_dls.q, p_target, R_target
    )
    print(f"\n  FK Roundtrip Validation (DLS):")
    print(f"    Valid:     {valid}")
    print(f"    Pos err:   {pe:.6f} m")
    print(f"    Rot err:   {re:.6f} rad")


def demo_mujoco_verification() -> None:
    """Demonstrate IK solution applied in MuJoCo."""
    print("\n" + "=" * 72)
    print("Part 2: MuJoCo Verification")
    print("=" * 72)

    pin_model, pin_data, ee_fid = load_pinocchio_model()

    # Create a target
    q_target = np.array([-0.5, -1.0, 1.2, -1.8, -1.57, 0.3])
    pin.forwardKinematics(pin_model, pin_data, q_target)
    pin.updateFramePlacements(pin_model, pin_data)
    p_target = pin_data.oMf[ee_fid].translation.copy()
    R_target = pin_data.oMf[ee_fid].rotation.copy()

    print(f"\n  Target position: [{p_target[0]:.4f}, {p_target[1]:.4f}, {p_target[2]:.4f}]")

    result, mj_ee = apply_ik_in_mujoco(p_target, R_target)

    print(f"\n  IK Solution:")
    print(f"    Success:     {result.success}")
    print(f"    Iterations:  {result.iterations}")
    print(f"    Pos error:   {result.position_error:.6f} m")

    pin_ee = pin_data.oMf[ee_fid].translation
    pin.forwardKinematics(pin_model, pin_data, result.q)
    pin.updateFramePlacements(pin_model, pin_data)
    pin_ee = pin_data.oMf[ee_fid].translation.copy()

    print(f"\n  Cross-validation:")
    print(f"    Pinocchio EE: [{pin_ee[0]:.4f}, {pin_ee[1]:.4f}, {pin_ee[2]:.4f}]")
    print(f"    MuJoCo EE:    [{mj_ee[0]:.4f}, {mj_ee[1]:.4f}, {mj_ee[2]:.4f}]")

    mismatch = np.linalg.norm(pin_ee - mj_ee) * 1000
    status = "PASS" if mismatch < 1.0 else "FAIL"
    print(f"    Mismatch:     {mismatch:.3f} mm [{status}]")


def demo_benchmark() -> None:
    """Run the IK benchmark and print results."""
    print("\n" + "=" * 72)
    print("Part 3: IK Benchmark (50 random reachable targets)")
    print("=" * 72)

    pin_model, pin_data, ee_fid = load_pinocchio_model()
    results = run_benchmark(pin_model, pin_data, ee_fid, n_targets=50)
    print_benchmark_summary(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for A4: Inverse Kinematics."""
    demo_single_target()
    demo_mujoco_verification()
    demo_benchmark()

    print("\n" + "=" * 72)
    print("Phase 2.2 — Inverse Kinematics: COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
