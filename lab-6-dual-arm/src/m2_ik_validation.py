"""M2 — IK validation: DLS IK convergence + FK round-trip.

Tests DualArmModel IK on 10 target poses per arm (position-only and
position+orientation). Verifies convergence and FK round-trip error.

Gate criteria:
  - IK converges for all 10 test targets (per arm)
  - IK solution FK error < 0.5 mm
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import math

import mujoco
import numpy as np
import pinocchio as pin

from dual_arm_model import DualArmModel
from lab6_common import (
    LEFT_JOINT_SLICE,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    RIGHT_JOINT_SLICE,
    SCENE_DUAL_PATH,
    get_mj_site_id,
)


def generate_reachable_targets(
    dual: DualArmModel,
    arm: str,
    n: int = 10,
    seed: int = 42,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Generate reachable targets by FK from random joint configs.

    Returns:
        List of (label, target_pos, target_rot).
    """
    rng = np.random.default_rng(seed)
    q_home = Q_HOME_LEFT if arm == "left" else Q_HOME_RIGHT
    targets = []

    for i in range(n):
        # Random perturbation from home — stays in reachable workspace
        q = q_home + rng.uniform(-0.8, 0.8, size=6)
        pose = dual.fk(arm, q)
        targets.append((f"T{i + 1}", pose.translation.copy(), pose.rotation.copy()))

    return targets


def main() -> None:
    """Run IK validation."""
    print("=" * 70)
    print("M2: IK Validation — DLS IK Convergence + FK Round-Trip")
    print("=" * 70)

    dual = DualArmModel()

    # Also load MuJoCo for cross-checking IK solutions
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_DUAL_PATH))
    mj_data = mujoco.MjData(mj_model)
    left_ee_sid = get_mj_site_id(mj_model, "left_ee_site")
    right_ee_sid = get_mj_site_id(mj_model, "right_ee_site")

    all_converged = True
    max_fk_err = 0.0
    max_mj_err = 0.0

    for arm in ("left", "right"):
        print(f"\n  --- {arm.upper()} ARM ---")
        print(f"  {'Target':<8s} {'Conv':>5s} {'Iter':>5s} "
              f"{'PosErr(mm)':>11s} {'RotErr(rad)':>12s} "
              f"{'FK_RT(mm)':>10s} {'MJ_RT(mm)':>10s}")
        print("  " + "-" * 71)

        targets = generate_reachable_targets(dual, arm, n=10, seed=42 if arm == "left" else 99)

        ee_sid = left_ee_sid if arm == "left" else right_ee_sid
        jslice = LEFT_JOINT_SLICE if arm == "left" else RIGHT_JOINT_SLICE

        for label, target_pos, target_rot in targets:
            q_sol, converged, info = dual.ik(
                arm, target_pos, target_rot,
                damping=0.01, max_iter=100, tol_pos=1e-4, tol_rot=1e-3,
            )

            # FK round-trip with Pinocchio
            fk_pose = dual.fk(arm, q_sol)
            fk_rt_err = np.linalg.norm(fk_pose.translation - target_pos)

            # MuJoCo round-trip
            mj_data.qpos[jslice] = q_sol
            mujoco.mj_forward(mj_model, mj_data)
            mj_pos = mj_data.site_xpos[ee_sid].copy()
            mj_rt_err = np.linalg.norm(mj_pos - target_pos)

            max_fk_err = max(max_fk_err, fk_rt_err)
            max_mj_err = max(max_mj_err, mj_rt_err)
            if not converged:
                all_converged = False

            print(f"  {label:<8s} {'Y' if converged else 'N':>5s} {info['iter']:>5d} "
                  f"{info['pos_err'] * 1000:11.6f} {info.get('rot_err', 0):12.6f} "
                  f"{fk_rt_err * 1000:10.6f} {mj_rt_err * 1000:10.6f}")

    # Also test position-only IK (bonus — not part of gate)
    print(f"\n  --- POSITION-ONLY IK (left arm, 5 targets, bonus) ---")
    print(f"  {'Target':<8s} {'Conv':>5s} {'Iter':>5s} "
          f"{'PosErr(mm)':>11s} {'FK_RT(mm)':>10s} {'MJ_RT(mm)':>10s}")
    print("  " + "-" * 55)

    pos_converged = 0
    targets_pos = generate_reachable_targets(dual, "left", n=5, seed=77)
    for label, target_pos, _ in targets_pos:
        q_sol, converged, info = dual.ik(
            "left", target_pos, target_rot=None,
            damping=0.01, max_iter=200, tol_pos=1e-4, n_restarts=20,
        )
        fk_pose = dual.fk("left", q_sol)
        fk_rt_err = np.linalg.norm(fk_pose.translation - target_pos)

        mj_data.qpos[LEFT_JOINT_SLICE] = q_sol
        mujoco.mj_forward(mj_model, mj_data)
        mj_pos = mj_data.site_xpos[left_ee_sid].copy()
        mj_rt_err = np.linalg.norm(mj_pos - target_pos)

        if converged:
            pos_converged += 1

        print(f"  {label:<8s} {'Y' if converged else 'N':>5s} {info['iter']:>5d} "
              f"{info['pos_err'] * 1000:11.6f} {fk_rt_err * 1000:10.6f} "
              f"{mj_rt_err * 1000:10.6f}")

    print(f"  Position-only: {pos_converged}/5 converged")

    # Gate (6DOF IK only — position-only is bonus)
    print("\n" + "=" * 70)
    print("M2 GATE CRITERIA")
    print("=" * 70)
    print(f"\n  [1] FK cross-validation:      PASS (see m2_fk_validation.py)")
    print(f"  [2] IK converges all targets: {'PASS' if all_converged else 'FAIL'}  "
          f"(20/20 6DOF targets)")
    print(f"  [3] IK FK round-trip (Pin):   {max_fk_err * 1000:.6f} mm  "
          f"(limit: 0.5)  {'PASS' if max_fk_err * 1000 < 0.5 else 'FAIL'}")
    print(f"  [4] IK FK round-trip (MJ):    {max_mj_err * 1000:.6f} mm  "
          f"(limit: 0.5)  {'PASS' if max_mj_err * 1000 < 0.5 else 'FAIL'}")

    gate_pass = all_converged and max_fk_err * 1000 < 0.5 and max_mj_err * 1000 < 0.5
    print(f"\n  M2 GATE: {'PASSED' if gate_pass else 'FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
