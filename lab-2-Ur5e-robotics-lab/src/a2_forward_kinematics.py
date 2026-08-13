"""Phase 1.2 — Forward Kinematics + DH Parameters.

This script implements:
  1. DH-based FK from scratch (standard DH convention, UR5e parameters)
  2. Pinocchio FK (URDF-based)
  3. MuJoCo FK (MJCF-based)
  4. Cross-validation between Pinocchio and MuJoCo (< 0.001 m)
  5. DH FK as an educational exercise (different frame convention than URDF)

Key insight: DH and URDF are two different parameterizations of the same
kinematic chain. They produce the same physical motion but use different
reference frames. The lab uses Pinocchio/MuJoCo as the ground truth.

Run:
    python3 lab-2-Ur5e-robotics-lab/src/a2_forward_kinematics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ur5e_common import (
    DH_PARAMS,
    NUM_JOINTS,
    Q_HOME,
    Q_ZEROS,
    load_mujoco_model,
    load_pinocchio_model,
    get_mj_ee_site_id,
)


# ---------------------------------------------------------------------------
# DH-based Forward Kinematics (Standard DH Convention)
# ---------------------------------------------------------------------------


def dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Compute the 4x4 Standard DH homogeneous transform.

    T_i = Rot_z(theta) @ Trans_z(d) @ Trans_x(a) @ Rot_x(alpha)

    This is the standard (not modified) DH convention used by Universal Robots.

    Args:
        alpha: Twist angle (rad).
        a: Link length (m).
        d: Link offset (m).
        theta: Joint angle (rad).

    Returns:
        4x4 homogeneous transformation matrix.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st * ca,  st * sa,  a * ct],
        [st,  ct * ca, -ct * sa,  a * st],
        [0,   sa,       ca,       d     ],
        [0,   0,        0,        1     ],
    ])


def fk_dh(q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute FK using the UR5e Standard DH parameters.

    The result is in the UR5e DH base frame convention.

    Args:
        q: Joint angles (6,).

    Returns:
        Tuple of:
          - T_0_to_6: 4x4 transform from base to frame 6.
          - frames: List of 4x4 intermediate transforms (7 total).
    """
    T = np.eye(4)
    frames = [T.copy()]

    for i, dh in enumerate(DH_PARAMS):
        T_i = dh_transform(dh.alpha, dh.a, dh.d, q[i])
        T = T @ T_i
        frames.append(T.copy())

    return T, frames


def rotation_to_rpy(R: np.ndarray) -> np.ndarray:
    """Extract Roll-Pitch-Yaw (XYZ) from a rotation matrix.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Array of [roll, pitch, yaw] in radians.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def demo_dh_fk() -> None:
    """Demonstrate DH FK at various configurations."""
    print("=" * 72)
    print("Part 1: DH-based FK (Standard DH Convention)")
    print("=" * 72)
    print()
    print("UR5e DH Parameter Table:")
    print(f"  {'Joint':<8} {'alpha (rad)':<14} {'a (m)':<12} {'d (m)':<12}")
    print("  " + "-" * 44)
    for i, dh in enumerate(DH_PARAMS):
        print(f"  {i + 1:<8} {dh.alpha:<14.4f} {dh.a:<12.4f} {dh.d:<12.4f}")

    configs = {
        "zeros":  Q_ZEROS.copy(),
        "home":   Q_HOME.copy(),
        "arm_up": np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0]),
    }

    print(f"\n  {'Config':<10} {'DH Frame 6 position (x, y, z)':>36}")
    print("  " + "-" * 48)
    for name, q in configs.items():
        T, frames = fk_dh(q)
        p = T[:3, 3]
        print(f"  {name:<10} [{p[0]:8.4f}, {p[1]:8.4f}, {p[2]:8.4f}]")

    # Show intermediate frames at zeros
    print("\n  Intermediate frames at q=zeros:")
    T, frames = fk_dh(Q_ZEROS)
    for i, F in enumerate(frames):
        p = F[:3, 3]
        label = "base" if i == 0 else f"frame_{i}"
        print(f"    {label:<10} [{p[0]:8.4f}, {p[1]:8.4f}, {p[2]:8.4f}]")


def validate_pin_vs_mujoco() -> None:
    """Cross-validate Pinocchio FK vs MuJoCo FK (the ground truth pair)."""
    import mujoco
    import pinocchio as pin

    print("\n" + "=" * 72)
    print("Part 2: Pinocchio vs MuJoCo Cross-Validation")
    print("=" * 72)

    mj_model, mj_data = load_mujoco_model()
    site_id = get_mj_ee_site_id(mj_model)
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    configs = {
        "zeros":       Q_ZEROS,
        "home":        Q_HOME,
        "reach_right": np.array([np.pi / 4, -np.pi / 3, np.pi / 4,
                                 -np.pi / 4, np.pi / 2, 0]),
        "reach_left":  np.array([-np.pi / 4, -np.pi / 3, np.pi / 4,
                                 -np.pi / 4, np.pi / 2, 0]),
        "arm_up":      np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0]),
        "random":      np.array([0.3, -1.2, 0.8, -0.6, 1.1, -0.4]),
    }

    print(f"\n  {'Config':<14} {'MuJoCo EE pos':>28} {'Pinocchio EE pos':>28} {'Error':>10}")
    print("  " + "-" * 84)

    all_ok = True
    for name, q in configs.items():
        # MuJoCo
        mj_data.qpos[:NUM_JOINTS] = q
        mujoco.mj_forward(mj_model, mj_data)
        mj_ee = mj_data.site_xpos[site_id].copy()

        # Pinocchio
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        pin_ee = pin_data.oMf[ee_fid].translation.copy()

        error_mm = np.linalg.norm(mj_ee - pin_ee) * 1000
        status = "OK" if error_mm < 1.0 else "FAIL"

        def fmt(p):
            return f"[{p[0]:7.4f}, {p[1]:7.4f}, {p[2]:7.4f}]"

        print(f"  {name:<14} {fmt(mj_ee):>28} {fmt(pin_ee):>28} "
              f"{error_mm:>6.3f} mm [{status}]")
        if error_mm >= 1.0:
            all_ok = False

    print()
    if all_ok:
        print("  PASS: All configurations match (< 1.0 mm error)")
    else:
        print("  FAIL: Some configurations exceed tolerance!")

    # Orientation comparison at home
    print("\n  Orientation at HOME config:")
    mj_data.qpos[:NUM_JOINTS] = Q_HOME
    mujoco.mj_forward(mj_model, mj_data)
    R_mj = mj_data.site_xmat[site_id].reshape(3, 3)

    pin.forwardKinematics(pin_model, pin_data, Q_HOME)
    pin.updateFramePlacements(pin_model, pin_data)
    R_pin = pin_data.oMf[ee_fid].rotation

    rpy_mj = np.degrees(rotation_to_rpy(R_mj))
    rpy_pin = np.degrees(rotation_to_rpy(R_pin))
    print(f"    MuJoCo RPY:    [{rpy_mj[0]:7.2f}, {rpy_mj[1]:7.2f}, {rpy_mj[2]:7.2f}] deg")
    print(f"    Pinocchio RPY: [{rpy_pin[0]:7.2f}, {rpy_pin[1]:7.2f}, {rpy_pin[2]:7.2f}] deg")


def note_on_dh_vs_urdf() -> None:
    """Explain why DH FK numbers differ from Pinocchio/MuJoCo."""
    print("\n" + "=" * 72)
    print("Part 3: DH vs URDF Frame Convention (Important!)")
    print("=" * 72)
    print("""
  The DH and URDF/MJCF represent the SAME physical robot but use DIFFERENT
  reference frame conventions:

  DH Convention (Standard):
    - All joints rotate around local z-axis
    - Frames defined by (alpha, a, d, theta) parameters
    - Base frame follows UR DH convention

  URDF/MJCF Convention:
    - Joints can rotate around any axis (x, y, or z)
    - Frames defined by body tree with explicit transforms
    - Base frame may have additional rotations (e.g., 180° around z)

  The Pinocchio (URDF) and MuJoCo (MJCF) models match PERFECTLY because
  they encode the same body tree with the same frame conventions.

  The DH FK uses the published UR5e parameters which differ slightly from
  the CAD-based dimensions used in the Menagerie model:
    DH d1 = 0.1625 m  vs  Menagerie = 0.163 m
    DH d4 = 0.1333 m  vs  Menagerie ≈ 0.127 m (shoulder offset)

  For ALL practical work in this lab, we use Pinocchio and MuJoCo as the
  ground truth. The DH FK is presented as an educational exercise to
  understand the mathematical foundation.
""")


def main() -> None:
    """Run FK validation."""
    demo_dh_fk()
    validate_pin_vs_mujoco()
    note_on_dh_vs_urdf()

    print("=" * 72)
    print("Phase 1.2 — Forward Kinematics: COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
