#!/usr/bin/env python3
"""M3a — Pinocchio Frame Convention Validation (pure math, NO simulation).

Validates that analytical Jacobians from Pinocchio match numerical finite
differences for:
  1. Center of Mass (CoM) — 3×nv Jacobian
  2. Left foot frame position — linear part of 6×nv frame Jacobian
  3. Right foot frame position — same

For each of the 12 leg joints, we:
  - Perturb by eps using pin.integrate(model, q, dq)  [correct for quaternion]
  - Recompute FK / CoM
  - Compare (new - old) / eps with analytical Jacobian column

Gate criteria:
  - All 12 CoM Jacobian columns: error < 1e-4
  - All foot Jacobian columns: error < 1e-4
  - Print frame IDs and names used for each foot
  - Save full output to media/m3a_jacobian_validation.txt
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
    LEG_JOINT_NAMES,
    MEDIA_DIR,
    PELVIS_MJCF_Z,
    load_g1_pinocchio,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

EPS: float = 1e-3  # perturbation magnitude [rad]
TOL: float = 1e-4  # max acceptable error per Jacobian column


# ---------------------------------------------------------------------------
# Build standing configuration in Pinocchio convention
# ---------------------------------------------------------------------------


def build_standing_q(model: pin.Model) -> np.ndarray:
    """Build a standing configuration in Pinocchio convention.

    Base at the Menagerie keyframe height, identity quaternion, leg joints
    at zero, arm joints at their neutral (stand keyframe) values.
    """
    q = pin.neutral(model)
    # Base position: x=0, y=0, z = keyframe_height - MJCF_offset
    q[0] = 0.0
    q[1] = 0.0
    q[2] = 0.79 - PELVIS_MJCF_Z  # ≈ -0.003
    # Quaternion (x,y,z,w) = identity → already (0,0,0,1) from neutral

    # Leg joints (indices 7-18): all zero → already zero from neutral

    # Arm joints from CTRL_STAND (indices 22-35 in q)
    q[22:29] = CTRL_STAND[15:22]  # left arm
    q[29:36] = CTRL_STAND[22:29]  # right arm

    return q


# ---------------------------------------------------------------------------
# Jacobian validation helpers
# ---------------------------------------------------------------------------


def validate_com_jacobian(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    joint_indices_v: list[int],
    joint_names: list[str],
    lines: list[str],
) -> int:
    """Validate CoM Jacobian columns against finite differences.

    Args:
        joint_indices_v: velocity indices (nv) for the 12 leg joints.
        joint_names: human-readable names for each joint.
        lines: output lines buffer.

    Returns:
        Number of failures.
    """
    # Analytical CoM Jacobian
    pin.centerOfMass(model, data, q)
    J_com = pin.jacobianCenterOfMass(model, data, q)  # (3, nv)
    com_ref = data.com[0].copy()

    lines.append("")
    lines.append("=" * 90)
    lines.append("  CoM JACOBIAN VALIDATION")
    lines.append("=" * 90)
    lines.append(f"  CoM position: [{com_ref[0]:.6f}, {com_ref[1]:.6f}, {com_ref[2]:.6f}]")
    lines.append(f"  J_com shape: {J_com.shape}")
    lines.append("")
    lines.append(f"  {'Joint':<30s} {'Analytical (x,y,z)':<36s} {'Numerical (x,y,z)':<36s} {'Error':<12s} {'Match'}")
    lines.append("-" * 130)

    n_fail = 0

    for name, iv in zip(joint_names, joint_indices_v):
        # Central differences: (f(q+eps) - f(q-eps)) / (2*eps) → O(eps^2) error
        dq = np.zeros(model.nv)

        # Forward perturbation
        dq[iv] = +EPS
        q_fwd = pin.integrate(model, q, dq)
        pin.centerOfMass(model, data, q_fwd)
        com_fwd = data.com[0].copy()

        # Backward perturbation
        dq[iv] = -EPS
        q_bwd = pin.integrate(model, q, dq)
        pin.centerOfMass(model, data, q_bwd)
        com_bwd = data.com[0].copy()

        # Numerical Jacobian column (central difference)
        num_col = (com_fwd - com_bwd) / (2.0 * EPS)

        # Analytical Jacobian column
        ana_col = J_com[:, iv]

        # Error
        err = np.linalg.norm(ana_col - num_col)
        match = err < TOL

        if not match:
            n_fail += 1
            # Detailed sign analysis
            sign_ana = np.sign(ana_col)
            sign_num = np.sign(num_col)
            sign_mismatch = not np.allclose(sign_ana, sign_num, atol=0.5)
            extra = "  SIGN MISMATCH" if sign_mismatch else "  MAGNITUDE"
        else:
            extra = ""

        ana_str = f"[{ana_col[0]:+.6f}, {ana_col[1]:+.6f}, {ana_col[2]:+.6f}]"
        num_str = f"[{num_col[0]:+.6f}, {num_col[1]:+.6f}, {num_col[2]:+.6f}]"
        status = "YES" if match else f"NO{extra}"

        lines.append(f"  {name:<30s} {ana_str:<36s} {num_str:<36s} {err:<12.2e} {status}")

    # Restore data to reference config
    pin.centerOfMass(model, data, q)

    return n_fail


def validate_foot_jacobian(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    frame_name: str,
    joint_indices_v: list[int],
    joint_names: list[str],
    lines: list[str],
) -> int:
    """Validate foot frame Jacobian (linear part) against finite differences.

    Returns:
        Number of failures.
    """
    frame_id = model.getFrameId(frame_name)
    frame_obj = model.frames[frame_id]

    lines.append("")
    lines.append("=" * 90)
    lines.append(f"  FOOT JACOBIAN VALIDATION: {frame_name}")
    lines.append("=" * 90)
    lines.append(f"  Frame ID: {frame_id}")
    lines.append(f"  Frame name: {frame_obj.name}")
    lines.append(f"  Frame type: {frame_obj.type}")
    lines.append(f"  Parent joint: {frame_obj.parentJoint} ({model.names[frame_obj.parentJoint]})")

    # Compute reference foot position
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    foot_ref = data.oMf[frame_id].translation.copy()

    lines.append(f"  Foot position: [{foot_ref[0]:.6f}, {foot_ref[1]:.6f}, {foot_ref[2]:.6f}]")

    # Analytical frame Jacobian (LOCAL_WORLD_ALIGNED)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J_foot_full = pin.getFrameJacobian(
        model, data, frame_id, pin.LOCAL_WORLD_ALIGNED,
    )  # (6, nv): linear (0:3) + angular (3:6)
    J_foot_lin = J_foot_full[:3, :]  # linear part only

    lines.append(f"  J_foot shape: {J_foot_full.shape} (using LINEAR part [:3, :])")
    lines.append(f"  Reference frame: LOCAL_WORLD_ALIGNED")
    lines.append("")
    lines.append(f"  {'Joint':<30s} {'Analytical (x,y,z)':<36s} {'Numerical (x,y,z)':<36s} {'Error':<12s} {'Match'}")
    lines.append("-" * 130)

    n_fail = 0

    for name, iv in zip(joint_names, joint_indices_v):
        # Central differences: (f(q+eps) - f(q-eps)) / (2*eps) → O(eps^2) error
        dq = np.zeros(model.nv)

        # Forward perturbation
        dq[iv] = +EPS
        q_fwd = pin.integrate(model, q, dq)
        pin.forwardKinematics(model, data, q_fwd)
        pin.updateFramePlacements(model, data)
        foot_fwd = data.oMf[frame_id].translation.copy()

        # Backward perturbation
        dq[iv] = -EPS
        q_bwd = pin.integrate(model, q, dq)
        pin.forwardKinematics(model, data, q_bwd)
        pin.updateFramePlacements(model, data)
        foot_bwd = data.oMf[frame_id].translation.copy()

        # Numerical column (central difference)
        num_col = (foot_fwd - foot_bwd) / (2.0 * EPS)

        # Analytical column
        ana_col = J_foot_lin[:, iv]

        # Error
        err = np.linalg.norm(ana_col - num_col)
        match = err < TOL

        if not match:
            n_fail += 1
            sign_ana = np.sign(ana_col)
            sign_num = np.sign(num_col)
            sign_mismatch = not np.allclose(sign_ana, sign_num, atol=0.5)
            extra = "  SIGN MISMATCH" if sign_mismatch else "  MAGNITUDE"
        else:
            extra = ""

        ana_str = f"[{ana_col[0]:+.6f}, {ana_col[1]:+.6f}, {ana_col[2]:+.6f}]"
        num_str = f"[{num_col[0]:+.6f}, {num_col[1]:+.6f}, {num_col[2]:+.6f}]"
        status = "YES" if match else f"NO{extra}"

        lines.append(f"  {name:<30s} {ana_str:<36s} {num_str:<36s} {err:<12.2e} {status}")

    # Restore reference state
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    return n_fail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MEDIA_DIR / "m3a_jacobian_validation.txt"

    lines: list[str] = []
    lines.append("M3a — Pinocchio Jacobian Validation (pure math)")
    lines.append(f"eps = {EPS}, tolerance = {TOL}")

    # ── Load Pinocchio model ──────────────────────────────────────────
    model, data = load_g1_pinocchio()
    lines.append(f"Model: nq={model.nq}, nv={model.nv}, njoints={model.njoints}")

    # ── Standing configuration ────────────────────────────────────────
    q = build_standing_q(model)
    lines.append(f"Standing q[0:7] (base): {q[0:7]}")
    lines.append(f"Standing q[7:19] (legs): {q[7:19]}")

    # ── Map leg joint names to velocity indices ───────────────────────
    joint_iv_list: list[int] = []
    for jname in LEG_JOINT_NAMES:
        jid = model.getJointId(jname)
        iv = model.joints[jid].idx_v
        joint_iv_list.append(iv)

    lines.append("")
    lines.append("Leg joint → velocity index mapping:")
    for name, iv in zip(LEG_JOINT_NAMES, joint_iv_list):
        lines.append(f"  {name:<35s} idx_v = {iv}")

    # ── 1. CoM Jacobian validation ────────────────────────────────────
    com_fails = validate_com_jacobian(
        model, data, q, joint_iv_list, LEG_JOINT_NAMES, lines,
    )

    # ── 2. Left foot Jacobian validation ──────────────────────────────
    lf_fails = validate_foot_jacobian(
        model, data, q, "left_ankle_roll_link",
        joint_iv_list, LEG_JOINT_NAMES, lines,
    )

    # ── 3. Right foot Jacobian validation ─────────────────────────────
    rf_fails = validate_foot_jacobian(
        model, data, q, "right_ankle_roll_link",
        joint_iv_list, LEG_JOINT_NAMES, lines,
    )

    # ── Also validate with OP_FRAME foot frames ──────────────────────
    lf_op_fails = validate_foot_jacobian(
        model, data, q, "left_foot",
        joint_iv_list, LEG_JOINT_NAMES, lines,
    )
    rf_op_fails = validate_foot_jacobian(
        model, data, q, "right_foot",
        joint_iv_list, LEG_JOINT_NAMES, lines,
    )

    # ── Gate summary ──────────────────────────────────────────────────
    total_fails = com_fails + lf_fails + rf_fails
    total_tests = 3 * len(LEG_JOINT_NAMES)  # 3 Jacobians × 12 joints

    lines.append("")
    lines.append("=" * 90)
    lines.append("  M3a GATE RESULTS")
    lines.append("=" * 90)
    lines.append(f"  {'Test':<50s} {'Failures':<15s} {'Status'}")
    lines.append("-" * 80)
    lines.append(
        f"  {'CoM Jacobian (12 leg joints)':<50s} "
        f"{com_fails:<15d} {'PASS' if com_fails == 0 else 'FAIL'}"
    )
    lines.append(
        f"  {'Left foot Jacobian (left_ankle_roll_link)':<50s} "
        f"{lf_fails:<15d} {'PASS' if lf_fails == 0 else 'FAIL'}"
    )
    lines.append(
        f"  {'Right foot Jacobian (right_ankle_roll_link)':<50s} "
        f"{rf_fails:<15d} {'PASS' if rf_fails == 0 else 'FAIL'}"
    )
    lines.append(
        f"  {'Left foot OP_FRAME (left_foot)':<50s} "
        f"{lf_op_fails:<15d} {'PASS' if lf_op_fails == 0 else 'FAIL'}"
    )
    lines.append(
        f"  {'Right foot OP_FRAME (right_foot)':<50s} "
        f"{rf_op_fails:<15d} {'PASS' if rf_op_fails == 0 else 'FAIL'}"
    )
    lines.append("-" * 80)
    lines.append(
        f"  {'TOTAL (CoM + ankle_roll_link feet)':<50s} "
        f"{str(total_fails) + '/' + str(total_tests):<15s} "
        f"{'ALL PASS' if total_fails == 0 else 'SOME FAIL'}"
    )
    lines.append("=" * 90)

    # ── Frame summary ─────────────────────────────────────────────────
    lines.append("")
    lines.append("Frame summary for future IK use:")
    for fname in [
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_foot", "right_foot",
    ]:
        fid = model.getFrameId(fname)
        fobj = model.frames[fid]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pos = data.oMf[fid].translation
        lines.append(
            f"  {fname:<30s} ID={fid:<4d} type={str(fobj.type):<20s} "
            f"pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
        )

    # ── Print and save ────────────────────────────────────────────────
    output = "\n".join(lines)
    print(output)

    output_path.write_text(output + "\n")
    print(f"\n  Output saved to: {output_path}")


if __name__ == "__main__":
    main()
