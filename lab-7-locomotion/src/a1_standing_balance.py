"""Demo A1: Standing balance with perturbation recovery.

Simulates the G1 humanoid standing still and recovering from
external force perturbations applied at t=3s and t=6s.

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

from lab7_common import (
    DT,
    NUM_ACTUATED,
    Q_HOME,
    JOINT_NAMES,
    load_mujoco_model,
    get_state_from_mujoco,
    get_mj_joint_dofadr,
    ensure_media_dir,
)
from g1_model import G1Model
from balance_controller import StandingBalanceController


def run_standing_balance_demo() -> None:
    """Run the standing balance demo with perturbation recovery."""
    import mujoco

    print("=" * 60)
    print("Lab 7 - Demo A1: Standing Balance with Perturbation Recovery")
    print("=" * 60)

    # Load models
    print("\n[1/5] Loading models...")
    mj_model, mj_data = load_mujoco_model()
    g1 = G1Model()
    controller = StandingBalanceController(g1)

    # Get joint addresses for setting initial state
    dof_addrs = get_mj_joint_dofadr(mj_model)

    # Set initial joint positions to home configuration
    print("[2/5] Setting initial configuration...")
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = mj_model.jnt_qposadr[jid]
        mj_data.qpos[qpos_adr] = Q_HOME[i]

    # Forward to compute initial state
    mujoco.mj_forward(mj_model, mj_data)

    # Simulation parameters
    sim_time = 10.0
    num_steps = int(sim_time / DT)
    perturb_times = [3.0, 6.0]  # Apply perturbation at these times
    perturb_force = [80.0, -60.0]  # Force magnitude in x-direction [N]
    perturb_duration = 0.1  # Duration of each perturbation [s]

    # Data collection
    times = np.zeros(num_steps)
    com_pos_log = np.zeros((num_steps, 3))
    com_vel_log = np.zeros((num_steps, 3))
    torque_log = np.zeros((num_steps, NUM_ACTUATED))
    ctrl_log = np.zeros((num_steps, NUM_ACTUATED))
    base_pos_log = np.zeros((num_steps, 3))

    # First settle with simple control for 0.5s
    settle_steps = int(0.5 / DT)

    print("[3/5] Running simulation (10s)...")
    print(f"  Perturbation 1: t=3.0s, Fx={perturb_force[0]}N")
    print(f"  Perturbation 2: t=6.0s, Fx={perturb_force[1]}N")

    for step in range(num_steps):
        t = step * DT
        times[step] = t

        # Get state for logging
        q_pin, v_pin = get_state_from_mujoco(mj_model, mj_data)
        com = g1.compute_com(q_pin)
        com_vel = g1.compute_com_velocity(q_pin, v_pin)
        com_pos_log[step] = com
        com_vel_log[step] = com_vel
        base_pos_log[step] = mj_data.qpos[0:3]

        # Compute control
        if step < settle_steps:
            ctrl = controller.compute_control_simple(mj_model, mj_data)
        else:
            ctrl = controller.compute_control(mj_model, mj_data)

        ctrl_log[step] = ctrl
        mj_data.ctrl[:] = ctrl

        # Apply perturbation forces
        for p_time, p_force in zip(perturb_times, perturb_force):
            if p_time <= t < p_time + perturb_duration:
                # Apply force to torso body
                torso_id = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
                )
                mj_data.xfrc_applied[torso_id, 0] = p_force
            elif abs(t - (p_time + perturb_duration)) < DT:
                torso_id = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
                )
                mj_data.xfrc_applied[torso_id, :] = 0.0

        # Log actuator torques
        torque_log[step] = mj_data.actuator_force[:NUM_ACTUATED]

        # Step simulation
        mujoco.mj_step(mj_model, mj_data)

        # Print progress
        if step % (num_steps // 10) == 0:
            print(f"  t={t:.1f}s  CoM=[{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}]")

    # Check robot didn't fall
    final_z = base_pos_log[-1, 2]
    print(f"\n  Final torso height: {final_z:.3f}m")
    if final_z > 0.5:
        print("  Status: STANDING (success)")
    else:
        print("  Status: FALLEN (needs tuning)")

    # Save plots
    print("[4/5] Generating plots...")
    media_dir = ensure_media_dir()

    # Plot 1: CoM position
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["X", "Y", "Z"]
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(times, com_pos_log[:, i], "b-", linewidth=1.0)
        ax.set_ylabel(f"CoM {label} [m]")
        ax.grid(True, alpha=0.3)
        # Mark perturbation times
        for pt in perturb_times:
            ax.axvline(pt, color="r", linestyle="--", alpha=0.5, label="perturbation")
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Center of Mass Position During Standing Balance")
    fig.tight_layout()
    fig.savefig(str(media_dir / "a1_com_position.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'a1_com_position.png'}")

    # Plot 2: CoM velocity
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(times, com_vel_log[:, i], "b-", linewidth=1.0)
        ax.set_ylabel(f"CoM V{label} [m/s]")
        ax.grid(True, alpha=0.3)
        for pt in perturb_times:
            ax.axvline(pt, color="r", linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Center of Mass Velocity During Standing Balance")
    fig.tight_layout()
    fig.savefig(str(media_dir / "a1_com_velocity.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'a1_com_velocity.png'}")

    # Plot 3: Selected joint torques
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    torque_indices = [3, 4, 5]  # left hip_pitch, knee, ankle
    torque_names = ["L Hip Pitch", "L Knee Pitch", "L Ankle Pitch"]
    for ax, idx, name in zip(axes, torque_indices, torque_names):
        ax.plot(times, torque_log[:, idx], "b-", linewidth=0.8)
        ax.set_ylabel(f"{name} [Nm]")
        ax.grid(True, alpha=0.3)
        for pt in perturb_times:
            ax.axvline(pt, color="r", linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Joint Torques During Standing Balance")
    fig.tight_layout()
    fig.savefig(str(media_dir / "a1_joint_torques.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'a1_joint_torques.png'}")

    # Plot 4: Base position (x,y top-down)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(base_pos_log[:, 0], base_pos_log[:, 1], "b-", linewidth=1.0)
    ax.plot(base_pos_log[0, 0], base_pos_log[0, 1], "go", markersize=8, label="Start")
    ax.plot(base_pos_log[-1, 0], base_pos_log[-1, 1], "rs", markersize=8, label="End")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Base Position (Top-Down View)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(media_dir / "a1_base_topdown.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {media_dir / 'a1_base_topdown.png'}")

    print("\n[5/5] Demo A1 complete!")
    print(f"  CoM X range: [{com_pos_log[:, 0].min():.4f}, {com_pos_log[:, 0].max():.4f}] m")
    print(f"  CoM Y range: [{com_pos_log[:, 1].min():.4f}, {com_pos_log[:, 1].max():.4f}] m")
    print(f"  CoM Z range: [{com_pos_log[:, 2].min():.4f}, {com_pos_log[:, 2].max():.4f}] m")


if __name__ == "__main__":
    run_standing_balance_demo()
