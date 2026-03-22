"""Lab 6 — Phase 3 Demo: Cooperative carry pipeline.

Full bimanual pick-carry-place: approach → grasp → lift → carry → lower → release → retreat.
Uses BimanualGraspStateMachine with DualImpedanceController.
Generates plots and metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure src is on path
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab6_common import (
    MEDIA_DIR,
    SCENE_DUAL_PATH,
    DT,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    BOX_INIT_POS,
)
from dual_arm_model import DualArmModel
from cooperative_controller import DualImpedanceGains
from bimanual_grasp import BimanualGraspStateMachine, BimanualTaskConfig

# ---------------------------------------------------------------------------
# Joint / actuator name definitions
# ---------------------------------------------------------------------------

LEFT_JOINT_NAMES = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
]

RIGHT_JOINT_NAMES = [
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_elbow_joint",
    "right_wrist_1_joint",
    "right_wrist_2_joint",
    "right_wrist_3_joint",
]

LEFT_ACTUATOR_NAMES = [
    "left_motor1",
    "left_motor2",
    "left_motor3",
    "left_motor4",
    "left_motor5",
    "left_motor6",
]

RIGHT_ACTUATOR_NAMES = [
    "right_motor1",
    "right_motor2",
    "right_motor3",
    "right_motor4",
    "right_motor5",
    "right_motor6",
]

# ---------------------------------------------------------------------------
# MuJoCo index helpers
# ---------------------------------------------------------------------------


def get_joint_qpos_indices(mj_model, joint_names: list[str]) -> list[int]:
    """Return qpos address for each named joint.

    Args:
        mj_model: MuJoCo model.
        joint_names: List of joint name strings.

    Returns:
        List of integer qpos indices.
    """
    return [
        mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in joint_names
    ]


def get_joint_dof_indices(mj_model, joint_names: list[str]) -> list[int]:
    """Return DOF (velocity) address for each named joint.

    Args:
        mj_model: MuJoCo model.
        joint_names: List of joint name strings.

    Returns:
        List of integer dof indices.
    """
    return [
        mj_model.jnt_dofadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in joint_names
    ]


def get_actuator_indices(mj_model, act_names: list[str]) -> list[int]:
    """Return actuator ID for each named actuator.

    Args:
        mj_model: MuJoCo model.
        act_names: List of actuator name strings.

    Returns:
        List of integer actuator IDs.
    """
    return [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in act_names
    ]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Assign a colour to each state for timeline plot
_STATE_COLORS = {
    "IDLE":      "#aaaaaa",
    "APPROACH":  "#4e9af1",
    "PRE_GRASP": "#6ec6f5",
    "GRASP":     "#e67e22",
    "LIFT":      "#e74c3c",
    "CARRY":     "#9b59b6",
    "LOWER":     "#27ae60",
    "RELEASE":   "#f1c40f",
    "RETREAT":   "#1abc9c",
    "DONE":      "#2ecc71",
}


def _plot_state_timeline(
    times: np.ndarray,
    states: list[str],
    total_time: float,
    state_log: list[tuple[float, str]],
) -> None:
    """Horizontal bar chart showing which state was active over time.

    Args:
        times: Logged time array.
        states: State name at each timestep.
        total_time: Total simulation duration (s).
        state_log: State machine transition log [(t, name), ...].
    """
    fig, ax = plt.subplots(figsize=(12, 3))

    # Build intervals from state_log
    intervals: list[tuple[float, float, str]] = []
    for i, (t_start, sname) in enumerate(state_log):
        t_end = state_log[i + 1][0] if i + 1 < len(state_log) else total_time
        intervals.append((t_start, t_end, sname))

    # Add any time before first logged state (IDLE period before transition)
    if state_log and state_log[0][0] > 0.0:
        intervals.insert(0, (0.0, state_log[0][0], "IDLE"))

    for t_start, t_end, sname in intervals:
        color = _STATE_COLORS.get(sname, "#888888")
        ax.barh(
            0, t_end - t_start, left=t_start, height=0.5,
            color=color, edgecolor="white", linewidth=0.5,
        )
        mid = (t_start + t_end) / 2.0
        if t_end - t_start > 0.3:
            ax.text(mid, 0, sname, ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")

    # Legend
    patches = [
        mpatches.Patch(color=_STATE_COLORS.get(s, "#888888"), label=s)
        for s in list(dict.fromkeys(sn for _, _, sn in intervals))
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=5)

    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Cooperative Carry — State Timeline")
    fig.tight_layout()
    out = MEDIA_DIR / "cooperative_carry_state_timeline.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_ee_positions(
    times: np.ndarray,
    ee_left: np.ndarray,
    ee_right: np.ndarray,
) -> None:
    """Plot left and right EE XYZ positions over time.

    Args:
        times: Time array (N,).
        ee_left: Left EE positions (N, 3).
        ee_right: Right EE positions (N, 3).
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["X (m)", "Y (m)", "Z (m)"]
    colors_left = ["#e74c3c", "#e67e22", "#f1c40f"]
    colors_right = ["#3498db", "#2ecc71", "#9b59b6"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(times, ee_left[:, i], color=colors_left[i], label=f"Left EE {label}", lw=1.2)
        ax.plot(times, ee_right[:, i], color=colors_right[i], label=f"Right EE {label}",
                lw=1.2, linestyle="--")
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Cooperative Carry — End-Effector Positions")
    fig.tight_layout()
    out = MEDIA_DIR / "cooperative_carry_ee_positions.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_box_trajectory(
    times: np.ndarray,
    box_pos: np.ndarray,
    state_log: list[tuple[float, str]],
) -> None:
    """Plot box X/Y/Z position over time with state transitions marked.

    Args:
        times: Time array (N,).
        box_pos: Box positions (N, 3).
        state_log: State transition log [(t, name), ...].
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["X (m)", "Y (m)", "Z (m)"]
    colors = ["#e74c3c", "#27ae60", "#3498db"]

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(times, box_pos[:, i], color=color, lw=1.5, label=f"Box {label}")
        # Mark state transitions
        for t_trans, sname in state_log:
            ax.axvline(t_trans, color="#aaaaaa", linestyle=":", lw=0.8, alpha=0.7)
            if i == 0:
                ax.text(t_trans, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else box_pos[:, i].max(),
                        sname, rotation=90, fontsize=6, va="top", color="#555555")
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Cooperative Carry — Box Trajectory")
    fig.tight_layout()
    out = MEDIA_DIR / "cooperative_carry_box_trajectory.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_torques(
    times: np.ndarray,
    tau_left: np.ndarray,
    tau_right: np.ndarray,
) -> None:
    """Plot joint torques for both arms.

    Args:
        times: Time array (N,).
        tau_left: Left arm torques (N, 6).
        tau_right: Right arm torques (N, 6).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    joint_labels = ["J1 (pan)", "J2 (lift)", "J3 (elbow)", "J4 (w1)", "J5 (w2)", "J6 (w3)"]
    cmap = plt.get_cmap("tab10")

    for j in range(6):
        axes[0].plot(times, tau_left[:, j], color=cmap(j), label=joint_labels[j], lw=1.0)
        axes[1].plot(times, tau_right[:, j], color=cmap(j), label=joint_labels[j], lw=1.0)

    axes[0].set_ylabel("Torque (Nm)")
    axes[0].set_title("Left Arm Joint Torques")
    axes[0].legend(loc="upper right", fontsize=7, ncol=3)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="k", lw=0.5)

    axes[1].set_ylabel("Torque (Nm)")
    axes[1].set_title("Right Arm Joint Torques")
    axes[1].legend(loc="upper right", fontsize=7, ncol=3)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="k", lw=0.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Cooperative Carry — Joint Torques")
    fig.tight_layout()
    out = MEDIA_DIR / "cooperative_carry_torques.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


def _print_summary(
    state_log: list[tuple[float, str]],
    total_time: float,
    box_pos_history: np.ndarray,
    target_pos: np.ndarray,
) -> None:
    """Print simulation summary with key metrics.

    Args:
        state_log: State transition log from the state machine.
        total_time: Total simulation wall time (s).
        box_pos_history: Box positions over time (N, 3).
        target_pos: Desired final box center (3,).
    """
    print("\n" + "=" * 60)
    print("COOPERATIVE CARRY PIPELINE — SUMMARY")
    print("=" * 60)
    print(f"  Total simulation time : {total_time:.2f} s")
    print()

    # State durations
    print("  State durations:")
    for i, (t_start, sname) in enumerate(state_log):
        t_end = state_log[i + 1][0] if i + 1 < len(state_log) else total_time
        duration = t_end - t_start
        print(f"    {sname:<12s} : {t_start:5.2f}s → {t_end:5.2f}s  ({duration:.2f}s)")
    print()

    # Box displacement
    box_init = box_pos_history[0]
    box_final = box_pos_history[-1]
    displacement = np.linalg.norm(box_final[:2] - box_init[:2])  # XY only
    print(f"  Box initial position  : [{box_init[0]:.3f}, {box_init[1]:.3f}, {box_init[2]:.3f}]")
    print(f"  Box final position    : [{box_final[0]:.3f}, {box_final[1]:.3f}, {box_final[2]:.3f}]")
    print(f"  Box XY displacement   : {displacement:.4f} m")

    # Placement error vs target
    placement_err = np.linalg.norm(box_final[:2] - target_pos[:2])
    print(f"  Target XY position    : [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
    print(f"  Placement XY error    : {placement_err:.4f} m")

    # Max lift height
    max_z = box_pos_history[:, 2].max()
    print(f"  Peak box height (Z)   : {max_z:.4f} m")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full bimanual pick-carry-place pipeline and generate plots."""

    print("Lab 6 — Cooperative Carry Pipeline Demo")
    print("-" * 40)

    # ------------------------------------------------------------------
    # 1. Load MuJoCo model
    # ------------------------------------------------------------------
    print("Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_DUAL_PATH))
    mj_data = mujoco.MjData(mj_model)

    # ------------------------------------------------------------------
    # 2. Build index arrays for joint reads and actuator writes
    # ------------------------------------------------------------------
    left_qpos_idx = get_joint_qpos_indices(mj_model, LEFT_JOINT_NAMES)
    left_dof_idx = get_joint_dof_indices(mj_model, LEFT_JOINT_NAMES)
    right_qpos_idx = get_joint_qpos_indices(mj_model, RIGHT_JOINT_NAMES)
    right_dof_idx = get_joint_dof_indices(mj_model, RIGHT_JOINT_NAMES)
    left_act_idx = get_actuator_indices(mj_model, LEFT_ACTUATOR_NAMES)
    right_act_idx = get_actuator_indices(mj_model, RIGHT_ACTUATOR_NAMES)
    box_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")

    print(f"  Left qpos indices  : {left_qpos_idx}")
    print(f"  Right qpos indices : {right_qpos_idx}")
    print(f"  Box body id        : {box_bid}")

    # ------------------------------------------------------------------
    # 3. Set initial joint positions and forward kinematics
    # ------------------------------------------------------------------
    for i, idx in enumerate(left_qpos_idx):
        mj_data.qpos[idx] = Q_HOME_LEFT[i]
    for i, idx in enumerate(right_qpos_idx):
        mj_data.qpos[idx] = Q_HOME_RIGHT[i]
    mujoco.mj_forward(mj_model, mj_data)

    # ------------------------------------------------------------------
    # 4. Create Pinocchio dual-arm model and controller components
    # ------------------------------------------------------------------
    print("Creating DualArmModel and BimanualGraspStateMachine...")
    dual_model = DualArmModel()

    # Task: pick box from (0.5, 0, 0.245), carry +0.3m in Y to (0.5, 0.3, 0.245)
    object_pos = np.array([0.5, 0.0, 0.245])
    target_pos = np.array([0.5, 0.3, 0.245])

    task_config = BimanualTaskConfig(
        object_pos=object_pos,
        target_pos=target_pos,
        lift_height=0.15,
        approach_clearance=0.08,
        grasp_offset_x=0.18,
        position_tolerance=0.015,
        settle_time=0.5,
    )

    gains = DualImpedanceGains(
        K_p=np.diag([400.0, 400.0, 400.0, 40.0, 40.0, 40.0]),
        K_d=np.diag([60.0, 60.0, 60.0, 6.0, 6.0, 6.0]),
        f_squeeze=10.0,
    )

    state_machine = BimanualGraspStateMachine(dual_model, task_config, gains)

    # ------------------------------------------------------------------
    # 5. Simulation loop
    # ------------------------------------------------------------------
    sim_duration = 15.0  # seconds
    n_steps = int(sim_duration / DT)
    log_every = 10        # log every 10 steps → 100 Hz log rate

    # Pre-allocate log arrays
    n_log = n_steps // log_every + 1
    log_times = np.zeros(n_log)
    log_states: list[str] = []
    log_ee_left = np.zeros((n_log, 3))
    log_ee_right = np.zeros((n_log, 3))
    log_box_pos = np.zeros((n_log, 3))
    log_tau_left = np.zeros((n_log, 6))
    log_tau_right = np.zeros((n_log, 6))

    log_idx = 0
    last_state_printed = ""

    print(f"Running simulation for {sim_duration:.1f}s ({n_steps} steps)...")

    for step in range(n_steps):
        t = mj_data.time

        # --- Read joint states from MuJoCo ----------------------------------
        q_left = np.array([mj_data.qpos[idx] for idx in left_qpos_idx])
        qd_left = np.array([mj_data.qvel[idx] for idx in left_dof_idx])
        q_right = np.array([mj_data.qpos[idx] for idx in right_qpos_idx])
        qd_right = np.array([mj_data.qvel[idx] for idx in right_dof_idx])

        # --- State machine step → torques -----------------------------------
        tau_left, tau_right = state_machine.step(
            q_left=q_left,
            qd_left=qd_left,
            q_right=q_right,
            qd_right=qd_right,
            t=t,
            mj_model=mj_model,
            mj_data=mj_data,
        )

        # --- Apply torques to MuJoCo actuators ------------------------------
        for i, act_id in enumerate(left_act_idx):
            mj_data.ctrl[act_id] = tau_left[i]
        for i, act_id in enumerate(right_act_idx):
            mj_data.ctrl[act_id] = tau_right[i]

        # --- Log data -------------------------------------------------------
        if step % log_every == 0 and log_idx < n_log:
            log_times[log_idx] = t
            log_states.append(state_machine.state.name)

            ee_l = dual_model.fk_left_pos(q_left)
            ee_r = dual_model.fk_right_pos(q_right)
            log_ee_left[log_idx] = ee_l
            log_ee_right[log_idx] = ee_r
            log_box_pos[log_idx] = mj_data.xpos[box_bid].copy()
            log_tau_left[log_idx] = tau_left
            log_tau_right[log_idx] = tau_right
            log_idx += 1

        # Print state transitions to console
        current_state = state_machine.state.name
        if current_state != last_state_printed:
            print(f"  t={t:6.3f}s  →  {current_state}")
            last_state_printed = current_state

        # --- Step MuJoCo ----------------------------------------------------
        mujoco.mj_step(mj_model, mj_data)

        # Stop early if done
        if state_machine.is_done:
            # Run a few more steps for the arms to fully settle
            settle_steps = int(0.5 / DT)
            for _ in range(settle_steps):
                q_left = np.array([mj_data.qpos[idx] for idx in left_qpos_idx])
                qd_left = np.array([mj_data.qvel[idx] for idx in left_dof_idx])
                q_right = np.array([mj_data.qpos[idx] for idx in right_qpos_idx])
                qd_right = np.array([mj_data.qvel[idx] for idx in right_dof_idx])

                tau_left, tau_right = state_machine.step(
                    q_left, qd_left, q_right, qd_right, mj_data.time,
                    mj_model=mj_model, mj_data=mj_data,
                )
                for i, act_id in enumerate(left_act_idx):
                    mj_data.ctrl[act_id] = tau_left[i]
                for i, act_id in enumerate(right_act_idx):
                    mj_data.ctrl[act_id] = tau_right[i]
                mujoco.mj_step(mj_model, mj_data)

                if log_idx < n_log:
                    log_times[log_idx] = mj_data.time
                    log_states.append(state_machine.state.name)
                    log_ee_left[log_idx] = dual_model.fk_left_pos(q_left)
                    log_ee_right[log_idx] = dual_model.fk_right_pos(q_right)
                    log_box_pos[log_idx] = mj_data.xpos[box_bid].copy()
                    log_tau_left[log_idx] = tau_left
                    log_tau_right[log_idx] = tau_right
                    log_idx += 1

            break

    total_time = mj_data.time
    print(f"\nSimulation complete. Simulated {total_time:.3f} s.")

    # Trim logs to actual data
    times = log_times[:log_idx]
    states_arr = log_states[:log_idx]
    ee_left_arr = log_ee_left[:log_idx]
    ee_right_arr = log_ee_right[:log_idx]
    box_pos_arr = log_box_pos[:log_idx]
    tau_left_arr = log_tau_left[:log_idx]
    tau_right_arr = log_tau_right[:log_idx]

    # ------------------------------------------------------------------
    # 6. Generate plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    state_log = state_machine.get_state_log()
    _plot_state_timeline(times, states_arr, total_time, state_log)
    _plot_ee_positions(times, ee_left_arr, ee_right_arr)
    _plot_box_trajectory(times, box_pos_arr, state_log)
    _plot_torques(times, tau_left_arr, tau_right_arr)

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    _print_summary(state_log, total_time, box_pos_arr, target_pos)


if __name__ == "__main__":
    main()
