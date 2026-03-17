"""Lab 5 — Capstone Pick-and-Place Demo.

Runs the full pick-and-place cycle:
  1. Load scene (UR5e + gripper + table + graspable box)
  2. Compute IK grasp configurations
  3. Execute GraspStateMachine (RRT* planning + impedance control + gripper)
  4. Log and plot results

Usage:
    cd lab-5-grasping-manipulation/src
    python pick_place_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab5_common import (
    BOX_A_POS,
    BOX_B_POS,
    MEDIA_DIR,
    Q_HOME,
    load_mujoco_model,
    load_pinocchio_model,
)
from grasp_planner import compute_grasp_configs, print_config_summary
from grasp_state_machine import GraspStateMachine


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(log: dict, save_dir: Path) -> None:
    """Generate and save pick-and-place result plots.

    Args:
        log: Log dict from GraspStateMachine.run().
        save_dir: Directory to save PNG files.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    t = log["time"]
    ee = log["ee_pos"]
    grip = log["gripper_pos"]
    states = log["state"]

    # --- 1. EE trajectory (XY top-down view) ---
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(ee[:, 0], ee[:, 1], c=t, cmap="viridis", s=1, alpha=0.7)
    ax.scatter(BOX_A_POS[0], BOX_A_POS[1], c="blue", s=150, marker="^",
               zorder=5, label="Box A (pick)")
    ax.scatter(BOX_B_POS[0], BOX_B_POS[1], c="green", s=150, marker="*",
               zorder=5, label="Box B (place)")
    plt.colorbar(sc, ax=ax, label="Time (s)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("End-Effector Trajectory — Top View (X-Y)")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "ee_trajectory_3d.png", dpi=150)
    plt.close(fig)

    # --- 2. EE position vs time ---
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    labels = ["X", "Y", "Z"]
    targets_a = [BOX_A_POS, BOX_B_POS]
    for j, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, ee[:, j] * 1000, "C0", linewidth=0.8, label="Actual")
        ax.axhline(BOX_A_POS[j] * 1000, color="C1", linestyle="--", alpha=0.7, label="Box A")
        ax.axhline(BOX_B_POS[j] * 1000, color="C2", linestyle="--", alpha=0.7, label="Box B")
        ax.set_ylabel(f"{label} (mm)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("EE Position vs Time")
    _add_state_spans(axes, t, states)
    fig.tight_layout()
    fig.savefig(save_dir / "ee_position_vs_time.png", dpi=150)
    plt.close(fig)

    # --- 3. Gripper position vs time ---
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(t, grip * 1000, "C3", linewidth=1.0)
    ax.axhline(30.0, color="C0", linestyle="--", alpha=0.5, label="Open (30 mm)")
    ax.axhline(0.0, color="C2", linestyle="--", alpha=0.5, label="Closed (0 mm)")
    ax.set_ylabel("Finger position (mm)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Gripper Opening vs Time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _add_state_spans([ax], t, states)
    fig.tight_layout()
    fig.savefig(save_dir / "gripper_vs_time.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {save_dir}/")


def _add_state_spans(axes, t: np.ndarray, states: list[str]) -> None:
    """Add vertical shaded regions for each state transition.

    Args:
        axes: List of matplotlib Axes to annotate.
        t: Time array.
        states: State name per timestep.
    """
    unique_states: list[tuple[str, float, float]] = []
    cur_state = states[0]
    t_start = t[0]
    for i in range(1, len(states)):
        if states[i] != cur_state:
            unique_states.append((cur_state, t_start, t[i - 1]))
            cur_state = states[i]
            t_start = t[i]
    unique_states.append((cur_state, t_start, t[-1]))

    colours = plt.cm.Set3(np.linspace(0, 1, len(unique_states)))
    for ax in axes:
        for (name, t0, t1), col in zip(unique_states, colours):
            ax.axvspan(t0, t1, alpha=0.15, color=col, label=name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full pick-and-place demo."""
    print("=" * 65)
    print(" Lab 5: Grasping & Manipulation — Pick-and-Place Demo")
    print("=" * 65)

    # Load models
    print("\n[1/3] Loading models...")
    mj_model, mj_data = load_mujoco_model()
    pin_model, pin_data, ee_fid = load_pinocchio_model()
    print(f"  MuJoCo: nq={mj_model.nq}, nu={mj_model.nu}, ngeom={mj_model.ngeom}")

    # Compute grasp configurations
    print("\n[2/3] Computing IK grasp configurations...")
    cfgs = compute_grasp_configs(pin_model, pin_data, ee_fid)
    print_config_summary(cfgs, pin_model, pin_data, ee_fid)

    # Run state machine
    print("\n[3/3] Running pick-and-place state machine...")
    sm = GraspStateMachine(
        mj_model, mj_data, pin_model, pin_data, ee_fid, cfgs,
        Kp_joint=400.0, Kd_joint=40.0,
        Kp_cart=600.0, Kd_cart=60.0,
    )
    log = sm.run()

    # Results summary
    t_total = log["time"][-1]
    ee_final = log["ee_pos"][-1]
    box_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "grasp_box")
    box_final = mj_data.xpos[box_bid].copy()

    print(f"\n{'='*65}")
    print(f"  Total time:       {t_total:.1f} s")
    print(f"  Box final pos:    [{box_final[0]:.3f}, {box_final[1]:.3f}, {box_final[2]:.3f}] m")
    print(f"  Target (Box B):   [{BOX_B_POS[0]:.3f}, {BOX_B_POS[1]:.3f}, {BOX_B_POS[2]:.3f}] m")
    dist = np.linalg.norm(box_final[:2] - BOX_B_POS[:2]) * 1000
    print(f"  Lateral error:    {dist:.1f} mm")
    print(f"{'='*65}\n")

    # Plot
    plot_results(log, MEDIA_DIR)


if __name__ == "__main__":
    main()
