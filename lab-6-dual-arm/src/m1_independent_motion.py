"""M1 — Independent joint PD control for dual UR5e arms.

Moves each arm through 3 target configs, holds 2s at each target,
records steady-state error. Saves video to media/m1_independent.mp4.

Gate criteria:
  - Steady-state joint error < 0.001 rad
  - Steady-state joint velocity < 0.01 rad/s
  - No torque saturation warnings
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio
import mujoco
import numpy as np

from lab6_common import (
    DT,
    LEFT_JOINT_SLICE,
    MEDIA_DIR,
    NUM_JOINTS_PER_ARM,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    RIGHT_JOINT_SLICE,
    SCENE_DUAL_PATH,
)
from joint_pd_controller import DualArmJointPD

# Video settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
FRAME_SKIP = int(1.0 / (DT * VIDEO_FPS))  # render every Nth step

# Timing
MOVE_TIME = 3.0       # seconds to reach target
HOLD_TIME = 2.0       # seconds to hold at target
SETTLE_WINDOW = 0.5   # last N seconds of hold used for steady-state measurement

# Target configs — moderate angles, avoid 0 and ±pi.
# Keep shoulder_pan near home (-pi/2 ≈ -1.571) to avoid sweeping arm through table.
# Each is (label, q_left(6), q_right(6))
TARGETS = [
    (
        "T1: shoulder lift + elbow",
        np.array([-1.7, -1.2, 1.0, -0.8, -1.0, 0.3]),
        np.array([-1.4, -1.0, 0.8, -1.2, -1.8, -0.3]),
    ),
    (
        "T2: wrist-dominant motion",
        np.array([-1.5, -1.0, 0.8, -1.5, -1.8, 0.5]),
        np.array([-1.7, -1.8, 1.8, -0.5, -1.2, 0.8]),
    ),
    (
        "T3: return near home",
        np.array([-1.6, -1.6, 1.5, -1.4, -1.4, 0.1]),
        np.array([-1.5, -1.5, 1.6, -1.6, -1.6, -0.1]),
    ),
]


def main() -> None:
    """Run M1 independent motion test."""
    print("=" * 70)
    print("M1: Independent Joint PD Control")
    print("=" * 70)

    # Load model
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_DUAL_PATH))
    mj_data = mujoco.MjData(mj_model)

    # Initialize at home
    mj_data.qpos[LEFT_JOINT_SLICE] = Q_HOME_LEFT
    mj_data.qpos[RIGHT_JOINT_SLICE] = Q_HOME_RIGHT
    mujoco.mj_forward(mj_model, mj_data)

    # Controller
    controller = DualArmJointPD(kp=100.0, kd=10.0)

    # Renderer
    renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

    frames: list[np.ndarray] = []
    results: list[dict] = []
    saturation_warnings: list[str] = []

    wall_start = time.time()

    # Start from home — settle for 1s
    print("\n  Settling at home for 1.0s...")
    n_settle = int(1.0 / DT)
    for step in range(n_settle):
        controller.compute(mj_data, Q_HOME_LEFT, Q_HOME_RIGHT)
        mujoco.mj_step(mj_model, mj_data)
        if step % FRAME_SKIP == 0:
            renderer.update_scene(mj_data)
            frames.append(renderer.render().copy())

    # Run each target
    current_target_left = Q_HOME_LEFT.copy()
    current_target_right = Q_HOME_RIGHT.copy()

    for t_idx, (label, q_left_target, q_right_target) in enumerate(TARGETS):
        print(f"\n  [{t_idx + 1}/{len(TARGETS)}] {label}")

        # Phase 1: Move to target
        n_move = int(MOVE_TIME / DT)
        for step in range(n_move):
            controller.compute(mj_data, q_left_target, q_right_target)
            mujoco.mj_step(mj_model, mj_data)
            if controller.saturated:
                saturation_warnings.append(f"{label} move step {step}")
            if step % FRAME_SKIP == 0:
                renderer.update_scene(mj_data)
                frames.append(renderer.render().copy())

        # Phase 2: Hold at target, collect steady-state data
        n_hold = int(HOLD_TIME / DT)
        n_measure_start = int((HOLD_TIME - SETTLE_WINDOW) / DT)

        errors_left: list[np.ndarray] = []
        errors_right: list[np.ndarray] = []
        vels_left: list[np.ndarray] = []
        vels_right: list[np.ndarray] = []

        for step in range(n_hold):
            controller.compute(mj_data, q_left_target, q_right_target)
            mujoco.mj_step(mj_model, mj_data)
            if controller.saturated:
                saturation_warnings.append(f"{label} hold step {step}")
            if step % FRAME_SKIP == 0:
                renderer.update_scene(mj_data)
                frames.append(renderer.render().copy())

            # Collect data in settle window
            if step >= n_measure_start:
                errors_left.append(np.abs(q_left_target - mj_data.qpos[LEFT_JOINT_SLICE]))
                errors_right.append(np.abs(q_right_target - mj_data.qpos[RIGHT_JOINT_SLICE]))
                vels_left.append(np.abs(mj_data.qvel[LEFT_JOINT_SLICE].copy()))
                vels_right.append(np.abs(mj_data.qvel[RIGHT_JOINT_SLICE].copy()))

        # Compute steady-state metrics
        max_err_left = np.max(errors_left)
        max_err_right = np.max(errors_right)
        max_vel_left = np.max(vels_left)
        max_vel_right = np.max(vels_right)

        results.append({
            "label": label,
            "max_err_left": max_err_left,
            "max_err_right": max_err_right,
            "max_vel_left": max_vel_left,
            "max_vel_right": max_vel_right,
        })

        print(f"    Left  — max err: {max_err_left:.6f} rad, max vel: {max_vel_left:.6f} rad/s")
        print(f"    Right — max err: {max_err_right:.6f} rad, max vel: {max_vel_right:.6f} rad/s")

    renderer.close()

    wall_elapsed = time.time() - wall_start
    print(f"\n  Simulation: {len(frames)} frames, {wall_elapsed:.1f}s wall time")

    # Write video
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    video_path = MEDIA_DIR / "m1_independent.mp4"
    print(f"  Writing video to {video_path} ...")
    writer = imageio.get_writer(
        str(video_path),
        fps=VIDEO_FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Video saved: {video_path}")

    # ---- Gate criteria ----
    print("\n" + "=" * 70)
    print("M1 GATE CRITERIA")
    print("=" * 70)

    # Error table
    print(f"\n  {'Target':<35s} {'MaxErr_L':>10s} {'MaxErr_R':>10s} {'MaxVel_L':>10s} {'MaxVel_R':>10s}")
    print("  " + "-" * 75)
    for r in results:
        print(f"  {r['label']:<35s} {r['max_err_left']:10.6f} {r['max_err_right']:10.6f} "
              f"{r['max_vel_left']:10.6f} {r['max_vel_right']:10.6f}")

    all_max_err = max(max(r["max_err_left"], r["max_err_right"]) for r in results)
    all_max_vel = max(max(r["max_vel_left"], r["max_vel_right"]) for r in results)
    n_sat = len(saturation_warnings)

    print(f"\n  [1] Max joint error:    {all_max_err:.6f} rad  (limit: 0.001)  "
          f"{'PASS' if all_max_err < 0.001 else 'FAIL'}")
    print(f"  [2] Max joint velocity: {all_max_vel:.6f} rad/s (limit: 0.01)   "
          f"{'PASS' if all_max_vel < 0.01 else 'FAIL'}")
    print(f"  [3] Torque saturation:  {n_sat} events  "
          f"{'PASS' if n_sat == 0 else f'WARN ({n_sat} events)'}")
    print(f"  [4] Video saved:        {video_path.exists()}  "
          f"{'PASS' if video_path.exists() else 'FAIL'}")

    all_pass = all_max_err < 0.001 and all_max_vel < 0.01 and video_path.exists()
    print(f"\n  M1 GATE: {'PASSED' if all_pass else 'FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
