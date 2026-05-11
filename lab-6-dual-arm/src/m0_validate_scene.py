"""M0 — Build and validate the dual-arm scene.

Loads scene_dual.xml, prints all joint and actuator names,
confirms 12 joints + 12 actuators, sets home configuration,
renders one frame to media/m0_scene.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing lab6_common from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

from lab6_common import (
    MEDIA_DIR,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    SCENE_DUAL_PATH,
)


def main() -> None:
    """Load scene, validate, render."""
    print("=" * 60)
    print("M0: Dual-Arm Scene Validation")
    print("=" * 60)

    # Load model
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_DUAL_PATH))
    mj_data = mujoco.MjData(mj_model)
    print(f"\nModel loaded: {SCENE_DUAL_PATH.name}")
    print(f"  nq = {mj_model.nq}  (generalized coords)")
    print(f"  nv = {mj_model.nv}  (generalized velocities)")
    print(f"  nu = {mj_model.nu}  (actuators)")

    # Print joint names
    print(f"\n--- Joints ({mj_model.njnt}) ---")
    joint_names = []
    for i in range(mj_model.njnt):
        name = mj_model.joint(i).name
        joint_names.append(name)
        jnt_type = mj_model.joint(i).type[0]
        type_str = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[jnt_type]
        print(f"  [{i:2d}] {name:40s} type={type_str}")

    # Print actuator names
    print(f"\n--- Actuators ({mj_model.nu}) ---")
    actuator_names = []
    for i in range(mj_model.nu):
        name = mj_model.actuator(i).name
        actuator_names.append(name)
        print(f"  [{i:2d}] {name}")

    # Count arm joints (hinge type only, excluding freejoint)
    arm_joints = [n for n in joint_names if "shoulder" in n or "elbow" in n or "wrist" in n]
    n_arm_joints = len(arm_joints)
    print(f"\n--- Validation ---")
    print(f"  Arm joints:  {n_arm_joints} (expected 12)")
    print(f"  Actuators:   {mj_model.nu} (expected 12)")

    assert n_arm_joints == 12, f"Expected 12 arm joints, got {n_arm_joints}"
    assert mj_model.nu == 12, f"Expected 12 actuators, got {mj_model.nu}"
    print("  PASS: 12 joints, 12 actuators confirmed.")

    # Set home configuration
    # Arm joints are qpos[0:6] (left) and qpos[6:12] (right).
    # Freejoint adds 7 more at the end: qpos[12:19].
    mj_data.qpos[0:6] = Q_HOME_LEFT
    mj_data.qpos[6:12] = Q_HOME_RIGHT
    mujoco.mj_forward(mj_model, mj_data)
    print(f"\n  Home config set. Left EE site, Right EE site positions:")

    # Print EE site positions and orientations
    left_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_ee_site")
    right_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_ee_site")

    minus_z = np.array([0.0, 0.0, -1.0])
    for label, sid in [("left_ee_site", left_site_id), ("right_ee_site", right_site_id)]:
        pos = mj_data.site_xpos[sid]
        z_axis = mj_data.site_xmat[sid].reshape(3, 3)[:, 2]
        dot_down = float(np.dot(z_axis, minus_z))
        status = "PASS" if dot_down > 0.9 else "FAIL"
        print(f"    {label:18s} pos={np.round(pos, 3)}  z_axis={np.round(z_axis, 3)}  dot(-z)={dot_down:.4f}  {status}")
        assert dot_down > 0.9, f"{label} EE z-axis not pointing down: dot={dot_down:.4f}"
    print("  PASS: Both EE z-axes point downward (dot > 0.9).")

    # Render
    renderer = mujoco.Renderer(mj_model, height=1080, width=1920)
    renderer.update_scene(mj_data)
    pixels = renderer.render()
    renderer.close()

    # Save screenshot
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MEDIA_DIR / "m0_scene.png"

    from PIL import Image
    img = Image.fromarray(pixels)
    img.save(str(out_path))
    print(f"\n  Screenshot saved: {out_path}")

    print("\n" + "=" * 60)
    print("M0 GATE: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
