"""M2 — Visual IK verification.

For each arm, pick 2 IK test targets. Set IK solution in MuJoCo, render
with a sphere marker at the target position. Saves 4 screenshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

from dual_arm_model import DualArmModel
from lab6_common import (
    LEFT_JOINT_SLICE,
    MEDIA_DIR,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    RIGHT_JOINT_SLICE,
    SCENE_DUAL_PATH,
)

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# Use same seeds as m2_ik_validation.py
LEFT_SEED = 42
RIGHT_SEED = 99


def generate_reachable_targets(
    dual: DualArmModel, arm: str, n: int = 10, seed: int = 42,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Same generator as m2_ik_validation.py."""
    rng = np.random.default_rng(seed)
    q_home = Q_HOME_LEFT if arm == "left" else Q_HOME_RIGHT
    targets = []
    for i in range(n):
        q = q_home + rng.uniform(-0.8, 0.8, size=6)
        pose = dual.fk(arm, q)
        targets.append((f"T{i + 1}", pose.translation.copy(), pose.rotation.copy()))
    return targets


def add_marker(scene: mujoco.MjvScene, pos: np.ndarray, rgba: np.ndarray, size: float = 0.02) -> None:
    """Add a sphere marker to the MjvScene via ngeom slot."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:] = [size, size, size]
    g.pos[:] = pos
    g.mat[:] = np.eye(3)
    g.rgba[:] = rgba
    g.dataid = -1
    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid = -1
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    g.emission = 0.5
    scene.ngeom += 1


def render_ik_shot(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    renderer: mujoco.Renderer,
    q_left: np.ndarray,
    q_right: np.ndarray,
    marker_pos: np.ndarray,
    marker_rgba: np.ndarray,
) -> np.ndarray:
    """Set joint config, render with marker, return image."""
    mj_data.qpos[LEFT_JOINT_SLICE] = q_left
    mj_data.qpos[RIGHT_JOINT_SLICE] = q_right
    mujoco.mj_forward(mj_model, mj_data)

    # Update scene then inject marker
    renderer.update_scene(mj_data)
    scene = renderer._scene
    add_marker(scene, marker_pos, marker_rgba, size=0.025)

    return renderer.render().copy()


def main() -> None:
    """Render 4 IK verification screenshots."""
    print("M2 Visual IK Verification")
    print("=" * 50)

    dual = DualArmModel()
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_DUAL_PATH))
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    # Pick targets: T1 and T5 for each arm (varied configs)
    left_targets = generate_reachable_targets(dual, "left", n=10, seed=LEFT_SEED)
    right_targets = generate_reachable_targets(dual, "right", n=10, seed=RIGHT_SEED)

    picks = [
        ("left",  0, left_targets,  np.array([0.0, 1.0, 0.2, 1.0])),  # T1, green marker
        ("left",  4, left_targets,  np.array([0.0, 1.0, 0.2, 1.0])),  # T5, green marker
        ("right", 0, right_targets, np.array([1.0, 0.2, 0.2, 1.0])),  # T1, red marker
        ("right", 4, right_targets, np.array([1.0, 0.2, 0.2, 1.0])),  # T5, red marker
    ]

    for idx, (arm, tidx, targets, rgba) in enumerate(picks):
        label, target_pos, target_rot = targets[tidx]

        # Solve IK
        q_sol, converged, info = dual.ik(arm, target_pos, target_rot)
        assert converged, f"{arm} {label} IK did not converge"

        # Set IK solution on the active arm, home on the other
        if arm == "left":
            q_l, q_r = q_sol, Q_HOME_RIGHT
        else:
            q_l, q_r = Q_HOME_LEFT, q_sol

        img = render_ik_shot(mj_model, mj_data, renderer, q_l, q_r, target_pos, rgba)

        shot_num = (idx % 2) + 1
        fname = f"m2_ik_{arm}_{shot_num}.png"
        out_path = MEDIA_DIR / fname

        import imageio
        imageio.imwrite(str(out_path), img)
        print(f"  {fname}: {arm} {label}, pos_err={info['pos_err']*1000:.3f} mm, "
              f"target={target_pos.round(3)}")

    renderer.close()
    print("\nDone — 4 screenshots saved to media/")


if __name__ == "__main__":
    main()
