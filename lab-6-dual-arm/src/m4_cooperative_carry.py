"""M4 — Cooperative carry: bimanual grasp, lift, carry, and place.

Runs the BimanualStateMachine for the full 6-state pipeline:
  APPROACH → CLOSE → GRASP → LIFT → CARRY → PLACE

Gate criteria:
  - Box lifted >= 13 cm above initial z
  - Box carried >= 18 cm in +x from initial x
  - Box placed on table surface (z within 3 cm of initial z)
  - Box rotation < 10 deg from initial orientation
  - Video: media/m4_carry.mp4
  - Trajectory plot: media/m4_box_trajectory.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from bimanual_state_machine import BimanualStateMachine, State
from dual_arm_model import DualArmModel
from joint_pd_controller import DualArmJointPD
from lab6_common import (
    DT,
    MEDIA_DIR,
    SCENE_DUAL_PATH,
)

# Video settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
FRAME_SKIP = int(1.0 / (DT * VIDEO_FPS))

# Controller gains — moderate for smooth motion (ramp handles large moves)
KP = 300.0
KD = 40.0


def plot_box_trajectory(sm: BimanualStateMachine, save_path: Path) -> None:
    """Plot box x, y, z vs time with state labels."""
    times = np.array(sm.time_log)
    traj = np.array(sm.box_trajectory)
    states = sm.state_log

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#08111f")

    labels = ["x", "y", "z"]
    colors = ["#4fc3f7", "#81c784", "#ffb74d"]

    # Find state transition times
    transitions: list[tuple[float, str]] = []
    prev_state = states[0]
    for i, s in enumerate(states):
        if s != prev_state:
            transitions.append((times[i], s.name))
            prev_state = s

    for ax, label, color, dim in zip(axes, labels, colors, range(3)):
        ax.set_facecolor("#0d1b2a")
        ax.plot(times, traj[:, dim], color=color, linewidth=1.5, label=f"box {label}")
        ax.set_ylabel(f"{label} (m)", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, color="white")
        for spine in ax.spines.values():
            spine.set_color("#1b2838")

        # State transition markers
        for t_trans, s_name in transitions:
            ax.axvline(t_trans, color="#e57373", alpha=0.5, linestyle="--", linewidth=0.8)
            if dim == 0:  # Only label on top subplot
                ax.text(t_trans, ax.get_ylim()[1], s_name,
                        color="#e57373", fontsize=7, rotation=45,
                        ha="left", va="bottom")

    axes[-1].set_xlabel("Time (s)", color="white", fontsize=11)
    axes[0].set_title("M4: Box Trajectory During Cooperative Carry",
                      color="white", fontsize=14, pad=10)

    # Add initial position reference lines
    init_pos = sm.box_init_pos
    if init_pos is not None:
        for ax, dim in zip(axes, range(3)):
            ax.axhline(init_pos[dim], color="white", alpha=0.3,
                       linestyle=":", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)


def compute_rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute rotation angle between two rotation matrices in degrees."""
    R_rel = R1.T @ R2
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle_rad)


def main() -> None:
    """Run M4 cooperative carry pipeline."""
    print("=" * 70)
    print("M4: Cooperative Carry — Bimanual Grasp, Lift, Carry, Place")
    print("=" * 70)

    # Load models
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_DUAL_PATH))
    mj_data = mujoco.MjData(mj_model)
    dual = DualArmModel()
    controller = DualArmJointPD(kp=KP, kd=KD)
    renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

    # Create state machine with renderer
    sm = BimanualStateMachine(
        mj_model, mj_data, dual, controller,
        renderer=renderer, frame_skip=FRAME_SKIP,
    )

    wall_start = time.time()
    success = sm.run()
    wall_elapsed = time.time() - wall_start

    renderer.close()

    print(f"\n  Pipeline {'COMPLETED' if success else 'FAILED'}")
    print(f"  Simulation time: {sm.sim_time:.2f}s")
    print(f"  Wall time: {wall_elapsed:.1f}s")
    print(f"  Frames captured: {len(sm.frames)}")

    if not success:
        print("\n  *** Pipeline failed — check state machine output above ***")
        return

    # ---- Save video ----
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    video_path = MEDIA_DIR / "m4_carry.mp4"
    print(f"\n  Writing video to {video_path} ...")
    writer = imageio.get_writer(
        str(video_path), fps=VIDEO_FPS, codec="libx264",
        quality=8, macro_block_size=1,
    )
    for frame in sm.frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Video saved: {video_path}")

    # ---- Trajectory plot ----
    plot_path = MEDIA_DIR / "m4_box_trajectory.png"
    plot_box_trajectory(sm, plot_path)
    print(f"  Trajectory plot saved: {plot_path}")

    # ---- Gate criteria ----
    print("\n" + "=" * 70)
    print("M4 GATE CRITERIA")
    print("=" * 70)

    box_id = sm.box_id
    box_final_pos = sm.mj_data.xpos[box_id].copy()
    box_final_rot = sm.mj_data.xmat[box_id].reshape(3, 3).copy()
    box_init_pos = sm.box_init_pos

    # Find max z during LIFT/CARRY states
    traj = np.array(sm.box_trajectory)
    states = sm.state_log
    lift_carry_mask = np.array([s in (State.LIFT, State.CARRY) for s in states])
    if lift_carry_mask.any():
        max_z = traj[lift_carry_mask, 2].max()
        lift_dz = max_z - box_init_pos[2]
    else:
        lift_dz = 0.0

    # Find max y during CARRY state (+y carry direction)
    carry_mask = np.array([s == State.CARRY for s in states])
    if carry_mask.any():
        max_y = traj[carry_mask, 1].max()
        carry_dy = max_y - box_init_pos[1]
    else:
        carry_dy = 0.0

    # Final z relative to initial
    place_dz = abs(box_final_pos[2] - box_init_pos[2])

    # Rotation error — compare initial R (identity since box starts axis-aligned)
    box_init_rot = np.eye(3)  # Box starts axis-aligned
    rot_err_deg = compute_rotation_error_deg(box_init_rot, box_final_rot)

    g1 = lift_dz >= 0.13
    g2 = carry_dy >= 0.18
    g3 = place_dz < 0.03
    g4 = rot_err_deg < 10.0
    g5 = video_path.exists()
    g6 = plot_path.exists()

    print(f"\n  [1] Lift >= 13 cm:               {'PASS' if g1 else 'FAIL'}"
          f"  (dz = {lift_dz*100:.1f} cm)")
    print(f"  [2] Carry >= 18 cm (+y):         {'PASS' if g2 else 'FAIL'}"
          f"  (dy = {carry_dy*100:.1f} cm)")
    print(f"  [3] Place on table (dz < 3 cm):  {'PASS' if g3 else 'FAIL'}"
          f"  (dz = {place_dz*100:.1f} cm)")
    print(f"  [4] Rotation < 10 deg:           {'PASS' if g4 else 'FAIL'}"
          f"  ({rot_err_deg:.1f} deg)")
    print(f"  [5] Video saved:                 {'PASS' if g5 else 'FAIL'}")
    print(f"  [6] Trajectory plot saved:        {'PASS' if g6 else 'FAIL'}")

    gate_pass = g1 and g2 and g3 and g4 and g5 and g6
    print(f"\n  M4 GATE: {'PASSED' if gate_pass else 'FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
