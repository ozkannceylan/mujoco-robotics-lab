"""Phase 1.1 — Environment Setup.

Verifies that MuJoCo and Pinocchio load the UR5e model correctly,
cross-validates FK between both engines, and renders the scene.

Run:
    python3 lab-2-Ur5e-robotics-lab/src/a1_model_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure this lab's src/ is on sys.path so imports work from any CWD
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ur5e_common import (
    MJCF_SCENE_PATH,
    URDF_PATH,
    Q_HOME,
    Q_ZEROS,
    NUM_JOINTS,
    JOINT_NAMES,
    load_mujoco_model,
    load_pinocchio_model,
    get_mj_ee_site_id,
)


def check_dependencies() -> None:
    """Print versions of all required packages."""
    print("=" * 60)
    print("Dependency Check")
    print("=" * 60)

    deps = {
        "numpy": "numpy",
        "mujoco": "mujoco",
        "pinocchio": "pinocchio",
        "matplotlib": "matplotlib",
    }
    for name, module in deps.items():
        try:
            mod = __import__(module)
            print(f"  {name:15s} {mod.__version__}")
        except ImportError:
            print(f"  {name:15s} NOT INSTALLED")


def check_model_files() -> None:
    """Verify that all model files exist."""
    print("\n" + "=" * 60)
    print("Model Files")
    print("=" * 60)

    files = {
        "MJCF scene": MJCF_SCENE_PATH,
        "URDF":       URDF_PATH,
    }
    for name, path in files.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {name:15s} [{status}] {path}")


def verify_mujoco():
    """Load the UR5e in MuJoCo and print model info."""
    import mujoco

    print("\n" + "=" * 60)
    print("MuJoCo Model Verification")
    print("=" * 60)

    mj_model, mj_data = load_mujoco_model()
    site_id = get_mj_ee_site_id(mj_model)

    print(f"  nq (joints):   {mj_model.nq}")
    print(f"  nv (DOF):      {mj_model.nv}")
    print(f"  nu (actuators):{mj_model.nu}")
    print(f"  nbody:         {mj_model.nbody}")
    print(f"  timestep:      {mj_model.opt.timestep} s")

    # Print joint names
    print("\n  Joints:")
    for i in range(min(mj_model.nq, NUM_JOINTS)):
        jnt = mj_model.joint(i)
        print(f"    {i}: {jnt.name}")

    # Set home configuration
    mj_data.qpos[:NUM_JOINTS] = Q_HOME
    mujoco.mj_forward(mj_model, mj_data)
    ee_pos = mj_data.site_xpos[site_id]
    print(f"\n  Home config qpos: {Q_HOME}")
    print(f"  Home EE position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")

    # Step 100 times from home
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    for _ in range(100):
        mujoco.mj_step(mj_model, mj_data)
    qpos_after = mj_data.qpos[:NUM_JOINTS].copy()
    print(f"  After 100 steps: qpos = [{', '.join(f'{q:.4f}' for q in qpos_after)}]")

    return mj_model, mj_data, site_id


def verify_pinocchio():
    """Load the UR5e in Pinocchio and print model info."""
    import pinocchio as pin

    print("\n" + "=" * 60)
    print("Pinocchio Model Verification")
    print("=" * 60)

    pin_model, pin_data, ee_fid = load_pinocchio_model()

    print(f"  Model name:    {pin_model.name}")
    print(f"  nq (config):   {pin_model.nq}")
    print(f"  nv (velocity): {pin_model.nv}")
    print(f"  njoints:       {pin_model.njoints}")
    print(f"  nframes:       {pin_model.nframes}")

    print("\n  Joints:")
    for i in range(1, pin_model.njoints):  # skip universe
        print(f"    {i}: {pin_model.names[i]}")

    print(f"\n  Frames:")
    for i, f in enumerate(pin_model.frames):
        print(f"    {i}: {f.name:30s} (parent joint: {f.parentJoint})")

    # FK at home
    pin.forwardKinematics(pin_model, pin_data, Q_HOME)
    pin.updateFramePlacements(pin_model, pin_data)
    ee_pos = pin_data.oMf[ee_fid].translation
    print(f"\n  Home EE position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")

    return pin_model, pin_data, ee_fid


def cross_validate_fk() -> None:
    """Compare FK between MuJoCo and Pinocchio at multiple configurations."""
    import mujoco
    import pinocchio as pin

    print("\n" + "=" * 60)
    print("FK Cross-Validation: MuJoCo vs Pinocchio")
    print("=" * 60)

    mj_model, mj_data = load_mujoco_model()
    site_id = get_mj_ee_site_id(mj_model)
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    configs = {
        "zeros":    Q_ZEROS,
        "home":     Q_HOME,
        "random_1": np.array([0.5, -1.0, 1.2, -0.5, 0.8, -0.3]),
        "random_2": np.array([-1.0, -2.0, 0.5, 1.0, -0.5, 2.0]),
        "extended": np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0]),
        "folded":   np.array([0, -2.5, 2.5, -1.5, -1.5, 0]),
    }

    print(f"\n  {'Config':<12} {'Error (mm)':>10}")
    print("  " + "-" * 24)

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
        print(f"  {name:<12} {error_mm:>8.3f} mm  [{status}]")
        if error_mm >= 1.0:
            all_ok = False

    if all_ok:
        print("\n  All configurations match (< 1.0 mm).")
    else:
        print("\n  WARNING: Some configurations have > 1.0 mm mismatch!")


def main() -> None:
    """Run all verification steps."""
    check_dependencies()
    check_model_files()
    verify_mujoco()
    verify_pinocchio()
    cross_validate_fk()

    print("\n" + "=" * 60)
    print("Phase 1.1 — Environment Setup: COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
