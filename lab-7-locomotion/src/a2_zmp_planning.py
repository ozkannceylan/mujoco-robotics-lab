"""Demo A2: ZMP trajectory planning visualization.

Plans a 10-step walk and visualizes the ZMP reference, planned CoM,
and footstep positions. No MuJoCo simulation -- pure planning.

Saves plots to media/ directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab7_common import DT, HIP_WIDTH, ensure_media_dir
from footstep_planner import FootstepPlanner
from lipm_planner import LIPMPlanner


def run_zmp_planning_demo() -> None:
    """Run the ZMP planning visualization demo."""
    print("=" * 60)
    print("Lab 7 - Demo A2: ZMP Trajectory Planning")
    print("=" * 60)

    # Parameters
    num_steps = 10
    step_length = 0.15
    step_width = 2 * HIP_WIDTH
    com_height = 0.55

    # Plan footsteps
    print("\n[1/4] Planning footsteps...")
    fp = FootstepPlanner(
        step_length=step_length,
        step_width=step_width,
        step_height=0.04,
        step_duration=0.5,
        double_support_duration=0.2,
    )
    steps = fp.plan_footsteps(num_steps=num_steps, first_foot="right")

    for i, s in enumerate(steps):
        side = "L" if s.is_left else "R"
        print(f"  Step {i+1} ({side}): x={s.x:.3f}, y={s.y:.3f}, "
              f"t=[{s.t_start:.2f}, {s.t_end:.2f}]")

    # Generate ZMP reference
    print("\n[2/4] Generating ZMP reference...")
    total_time = steps[-1].t_end + fp.double_support_duration + 1.0
    t_arr, zmp_ref_x, zmp_ref_y = fp.generate_zmp_reference(
        steps, total_time=total_time
    )
    print(f"  Total time: {total_time:.2f}s")
    print(f"  ZMP trajectory length: {len(zmp_ref_x)} samples")

    # Plan CoM trajectory via LIPM
    print("\n[3/4] Computing CoM trajectory via LIPM preview control...")
    lipm = LIPMPlanner(z_c=com_height, dt=DT, preview_steps=150)
    com_x, com_y, zmp_actual_x, zmp_actual_y = lipm.plan_com_trajectory(
        zmp_ref_x, zmp_ref_y
    )

    # Compute tracking error
    zmp_err_x = zmp_ref_x - zmp_actual_x
    zmp_err_y = zmp_ref_y - zmp_actual_y
    rms_x = np.sqrt(np.mean(zmp_err_x**2))
    rms_y = np.sqrt(np.mean(zmp_err_y**2))
    print(f"  ZMP tracking RMS: X={rms_x*1000:.2f}mm, Y={rms_y*1000:.2f}mm")

    # Save plots
    print("\n[4/4] Generating plots...")
    media_dir = ensure_media_dir()

    # Plot 1: Top-down view (CoM, ZMP, footsteps)
    fig, ax = plt.subplots(figsize=(14, 6))

    # Footstep rectangles
    foot_len = 0.12
    foot_wid = 0.06
    for s in steps:
        color = "cornflowerblue" if s.is_left else "salmon"
        rect = plt.Rectangle(
            (s.x - foot_len / 2, s.y - foot_wid / 2),
            foot_len, foot_wid,
            fill=True, facecolor=color, edgecolor="black",
            linewidth=1.0, alpha=0.6,
        )
        ax.add_patch(rect)
        label = "L" if s.is_left else "R"
        ax.text(s.x, s.y, label, ha="center", va="center", fontsize=7)

    # Initial foot positions
    for y, color, label in [
        (step_width / 2, "cornflowerblue", "L"),
        (-step_width / 2, "salmon", "R"),
    ]:
        rect = plt.Rectangle(
            (0 - foot_len / 2, y - foot_wid / 2),
            foot_len, foot_wid,
            fill=True, facecolor=color, edgecolor="black",
            linewidth=1.0, alpha=0.6,
        )
        ax.add_patch(rect)
        ax.text(0, y, label, ha="center", va="center", fontsize=7)

    # ZMP reference
    ax.plot(zmp_ref_x, zmp_ref_y, "g--", linewidth=1.0, alpha=0.7, label="ZMP ref")
    # ZMP actual
    ax.plot(zmp_actual_x, zmp_actual_y, "g-", linewidth=1.5, label="ZMP actual")
    # CoM trajectory
    ax.plot(com_x, com_y, "b-", linewidth=2.0, label="CoM")
    ax.plot(com_x[0], com_y[0], "bo", markersize=8)
    ax.plot(com_x[-1], com_y[-1], "bs", markersize=8)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"LIPM Preview Control: {num_steps}-Step Walk Plan (Top-Down)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(media_dir / "a2_topdown_plan.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'a2_topdown_plan.png'}")

    # Plot 2: X trajectories vs time
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(t_arr, zmp_ref_x, "g--", linewidth=1.0, alpha=0.7, label="ZMP ref")
    ax.plot(t_arr, zmp_actual_x, "g-", linewidth=1.5, label="ZMP actual")
    ax.plot(t_arr, com_x, "b-", linewidth=2.0, label="CoM")
    # Mark step phases
    for s in steps:
        color = "cornflowerblue" if s.is_left else "salmon"
        ax.axvspan(s.t_start, s.t_end, alpha=0.1, color=color)
    ax.set_ylabel("X [m]")
    ax.set_title("LIPM Trajectories: X Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_arr, zmp_ref_y, "g--", linewidth=1.0, alpha=0.7, label="ZMP ref")
    ax.plot(t_arr, zmp_actual_y, "g-", linewidth=1.5, label="ZMP actual")
    ax.plot(t_arr, com_y, "b-", linewidth=2.0, label="CoM")
    for s in steps:
        color = "cornflowerblue" if s.is_left else "salmon"
        ax.axvspan(s.t_start, s.t_end, alpha=0.1, color=color)
    ax.set_ylabel("Y [m]")
    ax.set_xlabel("Time [s]")
    ax.set_title("LIPM Trajectories: Y Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(media_dir / "a2_trajectories.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'a2_trajectories.png'}")

    # Plot 3: ZMP tracking error
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t_arr, zmp_err_x * 1000, "r-", linewidth=1.0)
    axes[0].set_ylabel("ZMP Error X [mm]")
    axes[0].set_title(f"ZMP Tracking Error (RMS: X={rms_x*1000:.2f}mm, Y={rms_y*1000:.2f}mm)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_arr, zmp_err_y * 1000, "r-", linewidth=1.0)
    axes[1].set_ylabel("ZMP Error Y [mm]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(media_dir / "a2_zmp_error.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'a2_zmp_error.png'}")

    # Plot 4: Foot swing trajectory example
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    start = np.array([0.0, -step_width / 2, 0.0])
    end = np.array([step_length * 0.5, -step_width / 2, 0.0])
    swing_times, swing_pos = fp.generate_foot_trajectory(start, end, 0.5)

    axes[0].plot(swing_times, swing_pos[:, 0], "b-", linewidth=2.0)
    axes[0].set_ylabel("X [m]")
    axes[0].set_title("Quintic Foot Swing Trajectory")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(swing_times, swing_pos[:, 1], "b-", linewidth=2.0)
    axes[1].set_ylabel("Y [m]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(swing_times, swing_pos[:, 2] * 1000, "b-", linewidth=2.0)
    axes[2].set_ylabel("Z [mm]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(media_dir / "a2_foot_swing.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'a2_foot_swing.png'}")

    print("\nDemo A2 complete!")
    print(f"  Steps planned: {num_steps}")
    print(f"  Total distance: {steps[-1].x:.3f}m")
    print(f"  ZMP tracking: {rms_x*1000:.2f}mm (X), {rms_y*1000:.2f}mm (Y)")


if __name__ == "__main__":
    run_zmp_planning_demo()
