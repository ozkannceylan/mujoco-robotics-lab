"""Lab 5 — Record pick-and-place demo video.

Runs the full GraspStateMachine with MuJoCo offscreen rendering and saves
an MP4 video to media/pick_place_demo.mp4.

Usage:
    cd lab-5-grasping-manipulation/src
    python record_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import imageio
import mujoco
import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab5_common import (
    MEDIA_DIR,
    load_mujoco_model,
    load_pinocchio_model,
)
from grasp_planner import compute_grasp_configs
from grasp_state_machine import GraspStateMachine

# ---------------------------------------------------------------------------
# Video parameters
# ---------------------------------------------------------------------------

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 60
RENDER_EVERY = 4         # render 1 frame per N simulation steps (1 kHz / 4 → 250 samples/s → 60 fps output)
OUTPUT_PATH = MEDIA_DIR / "pick_place_demo.mp4"


def main() -> None:
    """Record full pick-and-place cycle to MP4."""
    print("=" * 65)
    print(" Lab 5: Recording Pick-and-Place Demo Video")
    print("=" * 65)

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    print("\n[1/3] Loading models...")
    mj_model, mj_data = load_mujoco_model()
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    # Compute grasp configurations
    print("[2/3] Computing IK grasp configurations...")
    cfgs = compute_grasp_configs(pin_model, pin_data, ee_fid)

    # Set up offscreen renderer
    renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

    # Camera: isometric view of the scene
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.35, 0.0, 0.35]
    camera.distance = 1.8
    camera.azimuth = 160.0
    camera.elevation = -22.0

    # Intercept mujoco.mj_step to capture frames while the state machine runs
    frames: list[np.ndarray] = []
    step_count_box = [0]  # use list so closure can mutate
    original_mj_step = mujoco.mj_step

    def recording_mj_step(m, d):
        """Wrapper: step + optionally render a frame."""
        original_mj_step(m, d)
        step_count_box[0] += 1
        if step_count_box[0] % RENDER_EVERY == 0:
            renderer.update_scene(d, camera=camera)
            frames.append(renderer.render().copy())

    # Run state machine with recording hook
    print("[3/3] Running simulation + rendering...")
    mujoco.mj_step = recording_mj_step
    t0 = time.time()
    try:
        sm = GraspStateMachine(
            mj_model, mj_data, pin_model, pin_data, ee_fid, cfgs,
            Kp_joint=400.0, Kd_joint=40.0,
            Kp_cart=600.0, Kd_cart=60.0,
        )
        sm.run()
    finally:
        mujoco.mj_step = original_mj_step  # always restore

    elapsed = time.time() - t0
    n_steps = step_count_box[0]
    print(f"\n  Simulation: {n_steps} steps, {n_steps * 0.001:.1f}s sim time, "
          f"{len(frames)} frames, {elapsed:.0f}s wall time")

    # Write video
    print(f"\n  Writing video to {OUTPUT_PATH} ...")
    writer = imageio.get_writer(
        str(OUTPUT_PATH),
        fps=VIDEO_FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"  Video saved: {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print("\nDone.")


if __name__ == "__main__":
    main()
