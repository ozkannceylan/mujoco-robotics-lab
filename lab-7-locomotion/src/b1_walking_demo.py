"""Demo B1: Walking execution in MuJoCo.

Plans a 10+ step walk, executes it in MuJoCo simulation, and
records CoM, ZMP, foot positions, and joint data. Saves plots.
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

from lab7_common import (
    DT,
    NUM_ACTUATED,
    Q_HOME,
    JOINT_NAMES,
    NQ_FREEFLYER,
    NV_FREEFLYER,
    HIP_WIDTH,
    load_mujoco_model,
    get_state_from_mujoco,
    get_mj_joint_dofadr,
    ensure_media_dir,
)
from g1_model import G1Model
from balance_controller import StandingBalanceController
from walking_controller import WalkingController


def run_walking_demo() -> None:
    """Run the walking execution demo."""
    import mujoco

    print("=" * 60)
    print("Lab 7 - Demo B1: Walking Execution (10+ Steps)")
    print("=" * 60)

    # Load models
    print("\n[1/6] Loading models...")
    mj_model, mj_data = load_mujoco_model()
    g1 = G1Model()
    dof_addrs = get_mj_joint_dofadr(mj_model)

    # Set initial configuration
    print("[2/6] Setting initial configuration...")
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = mj_model.jnt_qposadr[jid]
        mj_data.qpos[qpos_adr] = Q_HOME[i]
    mujoco.mj_forward(mj_model, mj_data)

    # Create controllers
    balance = StandingBalanceController(g1)
    walker = WalkingController(
        g1,
        step_length=0.12,
        step_height=0.03,
        step_duration=0.5,
        double_support_duration=0.25,
        com_height=0.55,
    )

    # Plan walk
    print("[3/6] Planning 12-step walk...")
    num_walk_steps = 12
    total_walk_time = walker.plan(num_steps=num_walk_steps)

    for i, s in enumerate(walker.steps):
        side = "L" if s.is_left else "R"
        print(f"  Step {i+1} ({side}): x={s.x:.3f}, y={s.y:.3f}")

    # Settle time before walking
    settle_time = 1.0
    total_sim_time = settle_time + total_walk_time
    num_sim_steps = int(total_sim_time / DT)

    # Data collection
    times = np.zeros(num_sim_steps)
    com_actual = np.zeros((num_sim_steps, 3))
    base_pos = np.zeros((num_sim_steps, 3))
    base_quat = np.zeros((num_sim_steps, 4))
    joint_pos = np.zeros((num_sim_steps, NUM_ACTUATED))
    joint_vel = np.zeros((num_sim_steps, NUM_ACTUATED))
    ctrl_log = np.zeros((num_sim_steps, NUM_ACTUATED))
    left_foot_log = np.zeros((num_sim_steps, 3))
    right_foot_log = np.zeros((num_sim_steps, 3))

    print(f"[4/6] Running simulation ({total_sim_time:.1f}s)...")
    settle_steps = int(settle_time / DT)

    for step in range(num_sim_steps):
        t_sim = step * DT
        times[step] = t_sim

        # Get state
        q_pin, v_pin = get_state_from_mujoco(mj_model, mj_data)
        com = g1.compute_com(q_pin)
        com_actual[step] = com
        base_pos[step] = mj_data.qpos[0:3]
        base_quat[step] = mj_data.qpos[3:7]
        joint_pos[step] = q_pin[NQ_FREEFLYER:]
        joint_vel[step] = v_pin[NV_FREEFLYER:]

        # Get foot positions
        lf_pos = g1.get_left_foot_pos(q_pin)
        rf_pos = g1.get_right_foot_pos(q_pin)
        left_foot_log[step] = lf_pos
        right_foot_log[step] = rf_pos

        # Control
        if step < settle_steps:
            # Settle phase: balance controller
            ctrl = balance.compute_control_simple(mj_model, mj_data)
        else:
            # Walking phase
            t_walk = t_sim - settle_time
            ctrl = walker.compute_control(mj_model, mj_data, t_walk)

        ctrl_log[step] = ctrl
        mj_data.ctrl[:] = ctrl

        # Step simulation
        mujoco.mj_step(mj_model, mj_data)

        # Progress
        if step % (num_sim_steps // 10) == 0:
            print(f"  t={t_sim:.1f}s  CoM=[{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]  "
                  f"base_z={base_pos[step, 2]:.3f}")

    # Check results
    final_z = base_pos[-1, 2]
    distance = base_pos[-1, 0] - base_pos[0, 0]
    print(f"\n  Final torso height: {final_z:.3f}m")
    print(f"  Distance walked (X): {distance:.3f}m")
    print(f"  Final base position: [{base_pos[-1, 0]:.3f}, {base_pos[-1, 1]:.3f}, {base_pos[-1, 2]:.3f}]")

    if final_z > 0.4:
        print("  Status: WALKING/STANDING (success)")
    else:
        print("  Status: FALLEN (needs tuning)")

    # Save plots
    print("\n[5/6] Generating plots...")
    media_dir = ensure_media_dir()

    # Get planned data
    planned = walker.get_planned_data()
    t_planned = planned["time"] + settle_time  # shift to match sim time

    # Plot 1: CoM trajectory comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    labels_xyz = ["X", "Y", "Z"]

    for i, (ax, label) in enumerate(zip(axes, labels_xyz)):
        ax.plot(times, com_actual[:, i], "b-", linewidth=1.5, label="Actual CoM")
        if i == 0 and planned["com_x"] is not None:
            ax.plot(t_planned[:len(planned["com_x"])], planned["com_x"],
                    "r--", linewidth=1.0, label="Planned CoM")
        elif i == 1 and planned["com_y"] is not None:
            ax.plot(t_planned[:len(planned["com_y"])], planned["com_y"],
                    "r--", linewidth=1.0, label="Planned CoM")
        ax.set_ylabel(f"CoM {label} [m]")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Walking: CoM Trajectory (Actual vs Planned)")
    fig.tight_layout()
    fig.savefig(str(media_dir / "b1_com_trajectory.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'b1_com_trajectory.png'}")

    # Plot 2: Top-down view
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(com_actual[:, 0], com_actual[:, 1], "b-", linewidth=1.5, label="Actual CoM")
    if planned["com_x"] is not None:
        ax.plot(planned["com_x"], planned["com_y"], "r--", linewidth=1.0, label="Planned CoM")
    ax.plot(left_foot_log[:, 0], left_foot_log[:, 1], "c.", markersize=1, alpha=0.3, label="Left foot")
    ax.plot(right_foot_log[:, 0], right_foot_log[:, 1], "m.", markersize=1, alpha=0.3, label="Right foot")

    # Draw footstep rectangles
    foot_len = 0.10
    foot_wid = 0.05
    for s in walker.steps:
        color = "cornflowerblue" if s.is_left else "salmon"
        rect = plt.Rectangle(
            (s.x - foot_len / 2, s.y - foot_wid / 2),
            foot_len, foot_wid,
            fill=True, facecolor=color, edgecolor="black",
            linewidth=0.8, alpha=0.4,
        )
        ax.add_patch(rect)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Walking: Top-Down View")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(media_dir / "b1_topdown.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'b1_topdown.png'}")

    # Plot 3: Foot heights
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, left_foot_log[:, 2] * 1000, "c-", linewidth=1.0, label="Left foot Z")
    ax.plot(times, right_foot_log[:, 2] * 1000, "m-", linewidth=1.0, label="Right foot Z")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Foot Height [mm]")
    ax.set_title("Walking: Foot Heights")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(media_dir / "b1_foot_heights.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'b1_foot_heights.png'}")

    # Plot 4: Joint positions
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    joint_groups = [
        ("Waist", [0]),
        ("Left Hip/Knee", [3, 4]),
        ("Left Ankle", [5, 6]),
        ("Right Hip/Knee", [9, 10]),
    ]
    for ax, (group_name, indices) in zip(axes, joint_groups):
        for idx in indices:
            ax.plot(times, np.degrees(joint_pos[:, idx]),
                    linewidth=1.0, label=JOINT_NAMES[idx])
        ax.set_ylabel("Angle [deg]")
        ax.set_title(group_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(str(media_dir / "b1_joint_positions.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'b1_joint_positions.png'}")

    print("\n[6/6] Demo B1 complete!")
    print(f"  Steps planned: {num_walk_steps}")
    print(f"  Distance walked: {distance:.3f}m")
    print(f"  Final height: {final_z:.3f}m")


if __name__ == "__main__":
    run_walking_demo()
