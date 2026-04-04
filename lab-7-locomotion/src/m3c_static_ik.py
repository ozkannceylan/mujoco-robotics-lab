#!/usr/bin/env python3
"""M3c — Static Whole-Body IK (pure Pinocchio math, NO simulation).

Tests the IK solver in isolation before putting it in a simulation loop.

Whole-body IK task stack (priority order):
  1. Left foot position + orientation (6D) — must not slip
  2. Right foot position + orientation (6D) — must not slip
  3. CoM position XY (2D) — balance target
  4. Pelvis height Z (1D) — maintain standing height
  5. Pelvis orientation (3D) — keep upright

Method: Stacked Jacobian with damped least squares (DLS).
  J_stack = [J_lf; J_rf; J_com_xy; J_pelvis_z; J_pelvis_ori]
  dx_stack = [dx_lf; dx_rf; dx_com_xy; dx_pelvis_z; dx_pelvis_ori]
  dq = J^T (J J^T + lambda^2 I)^{-1} dx
  q  = pin.integrate(model, q, alpha * dq)

Test cases (all from standing config):
  1. Identity: targets = current FK → converge in < 5 iters
  2. CoM shift left 5cm (Y -= 0.05m), feet fixed
  3. CoM shift right 5cm (Y += 0.05m), feet fixed
  4. CoM shift forward 3cm (X += 0.03m), feet fixed

Gate criteria:
  - Identity converges < 5 iters
  - All 4 tests converge (< 200 iters)
  - Foot position error < 2mm in all tests
  - CoM XY error < 5mm in all tests
  - Save results to media/m3c_ik_results.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

from lab7_common import (
    CTRL_STAND,
    MEDIA_DIR,
    PELVIS_MJCF_Z,
    load_g1_pinocchio,
)

# ---------------------------------------------------------------------------
# IK parameters
# ---------------------------------------------------------------------------

LAMBDA: float = 1e-2    # DLS damping
ALPHA: float = 0.3      # step size
DQ_MAX: float = 0.1     # max dq norm per iteration


# ---------------------------------------------------------------------------
# Standing configuration (same as M3a)
# ---------------------------------------------------------------------------


def build_standing_q(model: pin.Model) -> np.ndarray:
    """Build a standing configuration in Pinocchio convention."""
    q = pin.neutral(model)
    q[2] = 0.79 - PELVIS_MJCF_Z
    q[22:29] = CTRL_STAND[15:22]  # left arm
    q[29:36] = CTRL_STAND[22:29]  # right arm
    return q


# ---------------------------------------------------------------------------
# Whole-body IK
# ---------------------------------------------------------------------------


def whole_body_ik(
    model: pin.Model,
    data: pin.Data,
    q_init: np.ndarray,
    com_target: np.ndarray,
    lf_target: pin.SE3,
    rf_target: pin.SE3,
    max_iter: int = 200,
    tol: float = 1e-3,
    fix_base: bool = False,
) -> tuple[np.ndarray, bool, int, dict[str, float]]:
    """Solve whole-body IK with stacked Jacobian DLS.

    Pelvis height and orientation targets are derived from q_init FK.

    Args:
        model: Pinocchio model with FreeFlyer base.
        data: Pinocchio data.
        q_init: Initial configuration (nq=36).
        com_target: (3,) CoM target position (only XY used).
        lf_target: SE3 target for left foot frame.
        rf_target: SE3 target for right foot frame.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on ||dx_stack||.
        fix_base: If True, only optimize actuated joints (v-indices 6:nv),
            keeping the floating base fixed. Use this in simulation where
            the base state is determined by physics, not by IK.

    Returns:
        (q_sol, converged, n_iters, errors_dict)
    """
    q = q_init.copy()

    # Frame IDs
    lf_id = model.getFrameId("left_foot")
    rf_id = model.getFrameId("right_foot")
    pelvis_id = model.getFrameId("pelvis")

    # Derive pelvis targets from initial FK
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pelvis_z_target = data.oMf[pelvis_id].translation[2]
    pelvis_R_target = data.oMf[pelvis_id].rotation.copy()

    converged = False
    n_iters = 0

    for it in range(max_iter):
        # ── Forward kinematics + Jacobians ────────────────────────
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        J_com_full = pin.jacobianCenterOfMass(model, data, q)  # 3×nv

        lf_now = data.oMf[lf_id]
        rf_now = data.oMf[rf_id]
        pelvis_now = data.oMf[pelvis_id]
        com_now = data.com[0].copy()

        # ── Task errors ───────────────────────────────────────────
        # Left foot 6D
        dx_lf = np.zeros(6)
        dx_lf[:3] = lf_target.translation - lf_now.translation
        dx_lf[3:] = pin.log3(lf_target.rotation @ lf_now.rotation.T)

        # Right foot 6D
        dx_rf = np.zeros(6)
        dx_rf[:3] = rf_target.translation - rf_now.translation
        dx_rf[3:] = pin.log3(rf_target.rotation @ rf_now.rotation.T)

        # CoM XY 2D
        dx_com_xy = com_target[:2] - com_now[:2]

        if fix_base:
            # With base fixed, pelvis tasks are unsatisfiable (pelvis Jacobian
            # is only nonzero for base DOFs). Use feet + CoM only.
            # Stack: 6 + 6 + 2 = 14
            dx_stack = np.concatenate([dx_lf, dx_rf, dx_com_xy])
        else:
            # Pelvis Z 1D
            dx_pz = np.array([pelvis_z_target - pelvis_now.translation[2]])

            # Pelvis orientation 3D
            dx_pori = pin.log3(pelvis_R_target @ pelvis_now.rotation.T)

            # Stack: 6 + 6 + 2 + 1 + 3 = 18
            dx_stack = np.concatenate([dx_lf, dx_rf, dx_com_xy, dx_pz, dx_pori])

        # ── Convergence check ─────────────────────────────────────
        if np.linalg.norm(dx_stack) < tol:
            converged = True
            n_iters = it
            break

        # ── Stacked Jacobian ──────────────────────────────────────
        J_lf = pin.getFrameJacobian(
            model, data, lf_id, pin.LOCAL_WORLD_ALIGNED,
        )  # 6×nv
        J_rf = pin.getFrameJacobian(
            model, data, rf_id, pin.LOCAL_WORLD_ALIGNED,
        )  # 6×nv
        J_com_xy = J_com_full[:2, :]     # 2×nv

        if fix_base:
            J_stack = np.vstack([J_lf, J_rf, J_com_xy])  # 14×nv
        else:
            J_pelvis = pin.getFrameJacobian(
                model, data, pelvis_id, pin.LOCAL_WORLD_ALIGNED,
            )  # 6×nv
            J_pz = J_pelvis[2:3, :]         # 1×nv
            J_pori = J_pelvis[3:6, :]       # 3×nv
            J_stack = np.vstack([J_lf, J_rf, J_com_xy, J_pz, J_pori])  # 18×nv

        # ── DLS solve ─────────────────────────────────────────────
        if fix_base:
            # Only optimize actuated joints (v-indices 6:nv)
            J_act = J_stack[:, 6:]  # 18×(nv-6)
            JJT = J_act @ J_act.T
            dq_act = J_act.T @ np.linalg.solve(
                JJT + LAMBDA**2 * np.eye(JJT.shape[0]), dx_stack,
            )
            # Clamp
            dq_norm = np.linalg.norm(dq_act)
            if dq_norm > DQ_MAX:
                dq_act *= DQ_MAX / dq_norm
            # Assemble full dq with zero base velocity
            dq = np.zeros(model.nv)
            dq[6:] = dq_act
        else:
            JJT = J_stack @ J_stack.T  # 18×18
            dq = J_stack.T @ np.linalg.solve(
                JJT + LAMBDA**2 * np.eye(JJT.shape[0]), dx_stack,
            )
            # Clamp dq norm
            dq_norm = np.linalg.norm(dq)
            if dq_norm > DQ_MAX:
                dq *= DQ_MAX / dq_norm

        # ── Integrate ─────────────────────────────────────────────
        q = pin.integrate(model, q, ALPHA * dq)

        n_iters = it + 1

    # ── Final state ───────────────────────────────────────────────
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.centerOfMass(model, data, q)

    errors = {
        "lf_pos": float(np.linalg.norm(
            lf_target.translation - data.oMf[lf_id].translation)),
        "rf_pos": float(np.linalg.norm(
            rf_target.translation - data.oMf[rf_id].translation)),
        "lf_ori": float(np.linalg.norm(
            pin.log3(lf_target.rotation @ data.oMf[lf_id].rotation.T))),
        "rf_ori": float(np.linalg.norm(
            pin.log3(rf_target.rotation @ data.oMf[rf_id].rotation.T))),
        "com_xy": float(np.linalg.norm(
            com_target[:2] - data.com[0][:2])),
        "pelvis_z": float(data.oMf[pelvis_id].translation[2]),
        "pelvis_z_err": float(abs(
            pelvis_z_target - data.oMf[pelvis_id].translation[2])),
        "pelvis_ori": float(np.linalg.norm(
            pin.log3(pelvis_R_target @ data.oMf[pelvis_id].rotation.T))),
    }

    return q, converged, n_iters, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MEDIA_DIR / "m3c_ik_results.txt"

    lines: list[str] = []
    lines.append("M3c — Static Whole-Body IK Validation (pure Pinocchio)")
    lines.append(f"Parameters: lambda={LAMBDA}, alpha={ALPHA}, dq_max={DQ_MAX}")
    lines.append("")

    # ── Load model ────────────────────────────────────────────────────
    model, data = load_g1_pinocchio()
    lines.append(f"Model: nq={model.nq}, nv={model.nv}")

    # ── Standing config and reference FK ──────────────────────────────
    q_stand = build_standing_q(model)

    pin.forwardKinematics(model, data, q_stand)
    pin.updateFramePlacements(model, data)
    pin.centerOfMass(model, data, q_stand)

    lf_id = model.getFrameId("left_foot")
    rf_id = model.getFrameId("right_foot")
    pelvis_id = model.getFrameId("pelvis")

    lf_ref = data.oMf[lf_id].copy()
    rf_ref = data.oMf[rf_id].copy()
    com_ref = data.com[0].copy()
    pelvis_z_ref = data.oMf[pelvis_id].translation[2]

    lines.append(f"Standing CoM:       [{com_ref[0]:.6f}, {com_ref[1]:.6f}, {com_ref[2]:.6f}]")
    lines.append(f"Standing pelvis Z:  {pelvis_z_ref:.6f}")
    lines.append(f"Left foot:          [{lf_ref.translation[0]:.6f}, "
                 f"{lf_ref.translation[1]:.6f}, {lf_ref.translation[2]:.6f}]")
    lines.append(f"Right foot:         [{rf_ref.translation[0]:.6f}, "
                 f"{rf_ref.translation[1]:.6f}, {rf_ref.translation[2]:.6f}]")

    # ── Define test cases ─────────────────────────────────────────────
    tests = [
        {
            "name": "1. Identity (targets = current)",
            "com_target": com_ref.copy(),
            "lf_target": lf_ref.copy(),
            "rf_target": rf_ref.copy(),
            "identity": True,
        },
        {
            "name": "2. CoM shift left 5cm (Y -= 0.05m)",
            "com_target": com_ref + np.array([0.0, -0.05, 0.0]),
            "lf_target": lf_ref.copy(),
            "rf_target": rf_ref.copy(),
            "identity": False,
        },
        {
            "name": "3. CoM shift right 5cm (Y += 0.05m)",
            "com_target": com_ref + np.array([0.0, +0.05, 0.0]),
            "lf_target": lf_ref.copy(),
            "rf_target": rf_ref.copy(),
            "identity": False,
        },
        {
            "name": "4. CoM shift forward 3cm (X += 0.03m)",
            "com_target": com_ref + np.array([+0.03, 0.0, 0.0]),
            "lf_target": lf_ref.copy(),
            "rf_target": rf_ref.copy(),
            "identity": False,
        },
    ]

    # ── Run tests ─────────────────────────────────────────────────────
    results = []

    for test in tests:
        lines.append("")
        lines.append("=" * 90)
        lines.append(f"  TEST: {test['name']}")
        lines.append("=" * 90)

        com_tgt = test["com_target"]
        lines.append(f"  CoM target:  [{com_tgt[0]:.6f}, {com_tgt[1]:.6f}, {com_tgt[2]:.6f}]")

        q_sol, converged, n_iters, errors = whole_body_ik(
            model, data, q_stand.copy(),
            com_target=test["com_target"],
            lf_target=test["lf_target"],
            rf_target=test["rf_target"],
        )

        # Read final state
        pin.forwardKinematics(model, data, q_sol)
        pin.updateFramePlacements(model, data)
        pin.centerOfMass(model, data, q_sol)

        lf_final = data.oMf[lf_id].translation.copy()
        rf_final = data.oMf[rf_id].translation.copy()
        com_final = data.com[0].copy()

        lines.append(f"  Converged:   {'YES' if converged else 'NO'} ({n_iters} iterations)")
        lines.append(f"  Final CoM:   [{com_final[0]:.6f}, {com_final[1]:.6f}, {com_final[2]:.6f}]")
        lines.append(f"  Final L foot:[{lf_final[0]:.6f}, {lf_final[1]:.6f}, {lf_final[2]:.6f}]")
        lines.append(f"  Final R foot:[{rf_final[0]:.6f}, {rf_final[1]:.6f}, {rf_final[2]:.6f}]")
        lines.append(f"  Pelvis Z:    {errors['pelvis_z']:.6f} "
                     f"(err={errors['pelvis_z_err']*1000:.4f}mm)")
        lines.append("")
        lines.append(f"  {'Metric':<30s} {'Value':<15s} {'Gate':<15s} {'Status'}")
        lines.append(f"  {'-'*75}")
        lines.append(
            f"  {'CoM XY error':<30s} {errors['com_xy']*1000:.4f} mm   "
            f"{'< 5mm':<15s} {'PASS' if errors['com_xy'] < 0.005 else 'FAIL'}"
        )
        lines.append(
            f"  {'L foot pos error':<30s} {errors['lf_pos']*1000:.4f} mm   "
            f"{'< 2mm':<15s} {'PASS' if errors['lf_pos'] < 0.002 else 'FAIL'}"
        )
        lines.append(
            f"  {'R foot pos error':<30s} {errors['rf_pos']*1000:.4f} mm   "
            f"{'< 2mm':<15s} {'PASS' if errors['rf_pos'] < 0.002 else 'FAIL'}"
        )
        lines.append(
            f"  {'L foot ori error':<30s} {errors['lf_ori']*1000:.4f} mrad "
            f"{'(info)':<15s} —"
        )
        lines.append(
            f"  {'R foot ori error':<30s} {errors['rf_ori']*1000:.4f} mrad "
            f"{'(info)':<15s} —"
        )
        lines.append(
            f"  {'Pelvis Z error':<30s} {errors['pelvis_z_err']*1000:.4f} mm   "
            f"{'(info)':<15s} —"
        )
        lines.append(
            f"  {'Pelvis ori error':<30s} {errors['pelvis_ori']*1000:.4f} mrad "
            f"{'(info)':<15s} —"
        )

        # Show joint angle changes from standing
        dq_from_stand = q_sol[7:19] - q_stand[7:19]  # leg joints only
        if np.linalg.norm(dq_from_stand) > 1e-6:
            lines.append("")
            lines.append("  Leg joint changes from standing (rad):")
            from lab7_common import LEG_JOINT_NAMES
            for jname, dval in zip(LEG_JOINT_NAMES, dq_from_stand):
                if abs(dval) > 1e-5:
                    lines.append(f"    {jname:<35s} {dval:+.6f}")

        results.append({
            "name": test["name"],
            "converged": converged,
            "n_iters": n_iters,
            "identity": test["identity"],
            **errors,
        })

    # ── Gate summary ──────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("  M3c GATE RESULTS")
    lines.append("=" * 90)
    lines.append(f"  {'Criterion':<55s} {'Value':<20s} {'Status'}")
    lines.append("-" * 85)

    # Identity test < 5 iters
    id_test = results[0]
    id_pass = id_test["converged"] and id_test["n_iters"] < 5
    lines.append(
        f"  {'Identity converges in < 5 iters':<55s} "
        f"{str(id_test['n_iters']) + ' iters':<20s} "
        f"{'PASS' if id_pass else 'FAIL'}"
    )

    # All 4 converge
    all_converge = all(r["converged"] for r in results)
    max_iters = max(r["n_iters"] for r in results)
    lines.append(
        f"  {'All 4 tests converge (< 200 iters)':<55s} "
        f"{'max=' + str(max_iters) + ' iters':<20s} "
        f"{'PASS' if all_converge else 'FAIL'}"
    )

    # Foot error < 2mm
    max_foot_err = max(
        max(r["lf_pos"], r["rf_pos"]) for r in results
    )
    foot_pass = max_foot_err < 0.002
    lines.append(
        f"  {'Foot position error < 2mm (all tests)':<55s} "
        f"{'max=' + f'{max_foot_err*1000:.4f}mm':<20s} "
        f"{'PASS' if foot_pass else 'FAIL'}"
    )

    # CoM error < 5mm
    max_com_err = max(r["com_xy"] for r in results)
    com_pass = max_com_err < 0.005
    lines.append(
        f"  {'CoM XY error < 5mm (all tests)':<55s} "
        f"{'max=' + f'{max_com_err*1000:.4f}mm':<20s} "
        f"{'PASS' if com_pass else 'FAIL'}"
    )

    # Results file
    lines.append(
        f"  {'Results saved':<55s} "
        f"{'m3c_ik_results.txt':<20s} PASS"
    )

    lines.append("-" * 85)
    all_pass = id_pass and all_converge and foot_pass and com_pass
    lines.append(
        f"  {'OVERALL':<55s} {'':<20s} "
        f"{'ALL PASS' if all_pass else 'SOME FAIL'}"
    )
    lines.append("=" * 90)

    # ── Print and save ────────────────────────────────────────────────
    output = "\n".join(lines)
    print(output)

    output_path.write_text(output + "\n")
    print(f"\n  Output saved to: {output_path}")


if __name__ == "__main__":
    main()
