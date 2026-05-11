"""M2 — FK cross-validation: Pinocchio vs MuJoCo.

Tests DualArmModel FK against MuJoCo site positions for both arms
across multiple joint configurations. Prints gate results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import math

import mujoco
import numpy as np

from dual_arm_model import DualArmModel
from lab6_common import (
    LEFT_JOINT_SLICE,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    RIGHT_JOINT_SLICE,
    SCENE_DUAL_PATH,
    get_mj_site_id,
)

# Test configs: (label, q_left, q_right)
TEST_CONFIGS = [
    (
        "Home",
        Q_HOME_LEFT.copy(),
        Q_HOME_RIGHT.copy(),
    ),
    (
        "Shoulder lift varied",
        np.array([-1.7, -1.2, 1.0, -0.8, -1.0, 0.3]),
        np.array([-1.4, -1.0, 0.8, -1.2, -1.8, -0.3]),
    ),
    (
        "Wrist dominant",
        np.array([-1.5, -1.0, 0.8, -1.5, -1.8, 0.5]),
        np.array([-1.7, -1.8, 1.8, -0.5, -1.2, 0.8]),
    ),
    (
        "Near home",
        np.array([-1.6, -1.6, 1.5, -1.4, -1.4, 0.1]),
        np.array([-1.5, -1.5, 1.6, -1.6, -1.6, -0.1]),
    ),
    (
        "Elbow up",
        np.array([-math.pi / 2, -0.5, 0.5, -1.0, -1.0, 0.0]),
        np.array([-math.pi / 2, -2.0, 2.0, -1.5, -2.0, 0.0]),
    ),
]


def main() -> None:
    """Run FK cross-validation."""
    print("=" * 70)
    print("M2: FK Cross-Validation — Pinocchio vs MuJoCo")
    print("=" * 70)

    # Load models
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_DUAL_PATH))
    mj_data = mujoco.MjData(mj_model)
    left_ee_sid = get_mj_site_id(mj_model, "left_ee_site")
    right_ee_sid = get_mj_site_id(mj_model, "right_ee_site")

    dual = DualArmModel()

    print(f"\n  {'Config':<30s} {'ErrL (mm)':>10s} {'ErrR (mm)':>10s}")
    print("  " + "-" * 50)

    max_err = 0.0
    results = []

    for label, q_left, q_right in TEST_CONFIGS:
        # MuJoCo FK
        mj_data.qpos[LEFT_JOINT_SLICE] = q_left
        mj_data.qpos[RIGHT_JOINT_SLICE] = q_right
        mujoco.mj_forward(mj_model, mj_data)
        mj_left_pos = mj_data.site_xpos[left_ee_sid].copy()
        mj_right_pos = mj_data.site_xpos[right_ee_sid].copy()

        # Pinocchio FK
        pin_left = dual.fk_left(q_left)
        pin_right = dual.fk_right(q_right)

        err_l = np.linalg.norm(pin_left.translation - mj_left_pos)
        err_r = np.linalg.norm(pin_right.translation - mj_right_pos)
        max_err = max(max_err, err_l, err_r)

        print(f"  {label:<30s} {err_l * 1000:10.6f} {err_r * 1000:10.6f}")
        results.append((label, err_l, err_r))

    # Add 15 random configs
    rng = np.random.default_rng(42)
    for i in range(15):
        q_l = rng.uniform(-2.0, 2.0, size=6)
        q_r = rng.uniform(-2.0, 2.0, size=6)

        mj_data.qpos[LEFT_JOINT_SLICE] = q_l
        mj_data.qpos[RIGHT_JOINT_SLICE] = q_r
        mujoco.mj_forward(mj_model, mj_data)
        mj_left_pos = mj_data.site_xpos[left_ee_sid].copy()
        mj_right_pos = mj_data.site_xpos[right_ee_sid].copy()

        pin_left = dual.fk_left(q_l)
        pin_right = dual.fk_right(q_r)

        err_l = np.linalg.norm(pin_left.translation - mj_left_pos)
        err_r = np.linalg.norm(pin_right.translation - mj_right_pos)
        max_err = max(max_err, err_l, err_r)

        label = f"Random #{i + 1}"
        print(f"  {label:<30s} {err_l * 1000:10.6f} {err_r * 1000:10.6f}")

    print()
    print(f"  Max FK error: {max_err * 1000:.6f} mm")
    print(f"  Gate (<1 mm): {'PASS' if max_err * 1000 < 1.0 else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
