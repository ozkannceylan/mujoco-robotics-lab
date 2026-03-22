"""Lab 6 Capstone — Full bimanual pick-carry-place demonstration.

Demonstrates all Lab 6 capabilities:
1. Coordinated approach (both arms simultaneously)
2. Bimanual grasp with weld constraints
3. Cooperative lift and carry
4. Coordinated place and release
5. Collision-free retreat

Collects comprehensive metrics and generates publication-quality plots.
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
)
from dual_arm_model import DualArmModel
from cooperative_controller import DualImpedanceGains
from bimanual_grasp import BimanualGraspStateMachine, BimanualTaskConfig
from dual_collision_checker import DualCollisionChecker

# ---------------------------------------------------------------------------
# Joint / actuator name definitions (same as b1_cooperative_carry.py)
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
# MuJoCo index helpers (reused from b1_cooperative_carry.py)
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
# State color palette (shared with b1_cooperative_carry.py)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Quaternion rotation utility
# ---------------------------------------------------------------------------


def _quat_angle_from_identity(q_wxyz: np.ndarray) -> float:
    """Return the rotation angle (degrees) from the identity quaternion.

    Clamps the w component to [-1, 1] to avoid NaN from floating-point error.

    Args:
        q_wxyz: Quaternion in (w, x, y, z) MuJoCo convention.

    Returns:
        Angle in degrees (always >= 0).
    """
    w = float(np.clip(q_wxyz[0], -1.0, 1.0))
    angle_rad = 2.0 * np.arccos(abs(w))
    return float(np.degrees(angle_rad))


# ---------------------------------------------------------------------------
# Plot 1 — State timeline
# ---------------------------------------------------------------------------


def _plot_state_timeline(
    total_time: float,
    state_log: list[tuple[float, str]],
) -> None:
    """Horizontal bar chart showing which state was active over time.

    Args:
        total_time: Total simulation duration (s).
        state_log: State machine transition log [(t, name), ...].
    """
    fig, ax = plt.subplots(figsize=(14, 3))

    # Build intervals from state_log
    intervals: list[tuple[float, float, str]] = []
    for i, (t_start, sname) in enumerate(state_log):
        t_end = state_log[i + 1][0] if i + 1 < len(state_log) else total_time
        intervals.append((t_start, t_end, sname))

    # Prepend an IDLE bar if the simulation started before the first logged state
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
            ax.text(
                mid, 0, sname,
                ha="center", va="center",
                fontsize=7, color="white", fontweight="bold",
            )

    # Legend (ordered by first appearance)
    seen: dict[str, bool] = {}
    patches = []
    for _, _, sname in intervals:
        if sname not in seen:
            seen[sname] = True
            patches.append(
                mpatches.Patch(color=_STATE_COLORS.get(sname, "#888888"), label=sname)
            )
    ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=5)

    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Lab 6 Capstone — State Timeline")
    fig.tight_layout()

    out = MEDIA_DIR / "capstone_state_timeline.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — EE tracking (2x3 subplot: left XYZ | right XYZ)
# ---------------------------------------------------------------------------


def _plot_ee_tracking(
    times: np.ndarray,
    ee_left: np.ndarray,
    ee_right: np.ndarray,
) -> None:
    """2x3 subplot showing XYZ position of left and right end-effectors.

    Args:
        times: Time array (N,).
        ee_left: Left EE world positions (N, 3).
        ee_right: Right EE world positions (N, 3).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    axis_labels = ["X (m)", "Y (m)", "Z (m)"]
    colors = ["#e74c3c", "#27ae60", "#3498db"]

    for col, (label, color) in enumerate(zip(axis_labels, colors)):
        # Left arm row
        axes[0, col].plot(times, ee_left[:, col], color=color, lw=1.2)
        axes[0, col].set_ylabel(label)
        axes[0, col].set_title(f"Left EE — {label}")
        axes[0, col].grid(True, alpha=0.3)

        # Right arm row
        axes[1, col].plot(times, ee_right[:, col], color=color, lw=1.2, linestyle="--")
        axes[1, col].set_ylabel(label)
        axes[1, col].set_title(f"Right EE — {label}")
        axes[1, col].set_xlabel("Time (s)")
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle("Lab 6 Capstone — End-Effector Tracking (Left / Right)", fontsize=12)
    fig.tight_layout()

    out = MEDIA_DIR / "capstone_ee_tracking.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Box trajectory with state annotations
# ---------------------------------------------------------------------------


def _plot_box_trajectory(
    times: np.ndarray,
    box_pos: np.ndarray,
    state_log: list[tuple[float, str]],
) -> None:
    """Box XYZ over time with vertical lines at state transitions.

    Args:
        times: Time array (N,).
        box_pos: Box center world positions (N, 3).
        state_log: State transition log [(t, name), ...].
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    labels = ["X (m)", "Y (m)", "Z (m)"]
    colors = ["#e74c3c", "#27ae60", "#3498db"]

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(times, box_pos[:, i], color=color, lw=1.5, label=f"Box {label}")

        # Vertical lines at state transitions
        for t_trans, sname in state_log:
            ax.axvline(t_trans, color="#aaaaaa", linestyle=":", lw=0.8, alpha=0.7)
            if i == 0:  # annotate only on top subplot to avoid clutter
                y_top = box_pos[:, i].max()
                ax.text(
                    t_trans, y_top, sname,
                    rotation=90, fontsize=6, va="top", color="#555555",
                )

        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Lab 6 Capstone — Box Trajectory with State Annotations")
    fig.tight_layout()

    out = MEDIA_DIR / "capstone_box_trajectory.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Arm-arm minimum distance over time
# ---------------------------------------------------------------------------


def _plot_min_distance(
    times_dist: np.ndarray,
    min_distances: np.ndarray,
) -> None:
    """Arm-arm minimum distance vs. time.

    Args:
        times_dist: Sparse time array for distance samples (M,).
        min_distances: Minimum arm-arm distance at each sample (M,).
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(times_dist, min_distances * 1000.0, color="#e74c3c", lw=1.5,
            label="Min arm-arm distance")
    ax.axhline(0.0, color="k", lw=1.0, linestyle="--", label="Collision threshold")
    ax.fill_between(times_dist, 0, min_distances * 1000.0,
                    where=(min_distances >= 0), alpha=0.15, color="#27ae60",
                    label="Clearance zone")
    ax.fill_between(times_dist, min_distances * 1000.0, 0,
                    where=(min_distances < 0), alpha=0.3, color="#e74c3c",
                    label="Penetration zone")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (mm)")
    ax.set_title("Lab 6 Capstone — Arm-Arm Minimum Distance Over Time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = MEDIA_DIR / "capstone_min_distance.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 5 — Joint torques (2 subplots: left, right)
# ---------------------------------------------------------------------------


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
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    joint_labels = ["J1 (pan)", "J2 (lift)", "J3 (elbow)", "J4 (w1)", "J5 (w2)", "J6 (w3)"]
    cmap = plt.get_cmap("tab10")

    for j in range(6):
        axes[0].plot(times, tau_left[:, j], color=cmap(j), label=joint_labels[j], lw=1.0)
        axes[1].plot(times, tau_right[:, j], color=cmap(j), label=joint_labels[j], lw=1.0)

    for ax, title in zip(axes, ["Left Arm Joint Torques", "Right Arm Joint Torques"]):
        ax.set_ylabel("Torque (Nm)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Lab 6 Capstone — Joint Torques", fontsize=12)
    fig.tight_layout()

    out = MEDIA_DIR / "capstone_torques.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(
    total_time: float,
    state_log: list[tuple[float, str]],
    box_pos_init: np.ndarray,
    box_pos_final: np.ndarray,
    target_pos: np.ndarray,
    box_quat_history: np.ndarray,
    min_distances: np.ndarray,
    tau_left_all: np.ndarray,
    tau_right_all: np.ndarray,
) -> dict:
    """Print the comprehensive capstone summary table and validate success criteria.

    Args:
        total_time: Total simulated time (s).
        state_log: State transition log [(t, name), ...].
        box_pos_init: Box position at t=0 (3,).
        box_pos_final: Box position at end (3,).
        target_pos: Desired final box center (3,).
        box_quat_history: Box quaternion (w,x,y,z) at each logged step (N, 4).
        min_distances: Arm-arm minimum distances at sample points (M,).
        tau_left_all: Left torques (N, 6).
        tau_right_all: Right torques (N, 6).

    Returns:
        Dict with computed metric values for validation.
    """
    # --- Compute metrics ---
    n_transitions = len(state_log)

    placement_err_m = np.linalg.norm(box_pos_final[:2] - target_pos[:2])
    placement_err_mm = placement_err_m * 1000.0

    # Box rotation: max angle deviation from initial orientation during entire run
    # Reference: first logged quaternion (should be close to identity if box starts upright)
    ref_quat = box_quat_history[0]
    ref_w = float(np.clip(ref_quat[0], -1.0, 1.0))
    angles_deg = np.array([_quat_angle_from_identity(q) for q in box_quat_history])
    # Angle relative to initial orientation: use relative rotation
    # Compute deviation as angle between q[i] and q[0]:
    #   dot product of unit quaternions = cos(angle/2)
    dots = np.clip(np.abs(box_quat_history @ ref_quat), 0.0, 1.0)
    relative_angles_deg = np.degrees(2.0 * np.arccos(dots))
    max_rotation_deg = float(relative_angles_deg.max())

    min_arm_dist_mm = float(min_distances.min()) * 1000.0 if len(min_distances) > 0 else float("nan")

    max_tau_left = float(np.abs(tau_left_all).max())
    max_tau_right = float(np.abs(tau_right_all).max())

    # --- Print table ---
    W = 63
    print()
    print("=" * W)
    print("Lab 6 Capstone: Bimanual Pick-Carry-Place")
    print("=" * W)
    print(f"{'Metric':<35} {'Value':>25}")
    print("-" * W)
    print(f"{'Total time (s)':<35} {total_time:>25.1f}")
    print(f"{'State transitions':<35} {n_transitions:>25d}")
    print(f"{'Box initial position':<35} {str([round(v, 3) for v in box_pos_init.tolist()]):>25}")
    print(f"{'Box final position':<35} {str([round(v, 3) for v in box_pos_final.tolist()]):>25}")
    print(f"{'Placement error (mm)':<35} {placement_err_mm:>25.1f}")
    print(f"{'Box rotation during carry (deg)':<35} {max_rotation_deg:>25.1f}")
    print(f"{'Min arm-arm distance (mm)':<35} {min_arm_dist_mm:>25.1f}")
    print(f"{'Max left torque (Nm)':<35} {max_tau_left:>25.1f}")
    print(f"{'Max right torque (Nm)':<35} {max_tau_right:>25.1f}")
    print("=" * W)

    # --- Validate success criteria ---
    print()
    print("Success Criteria Validation")
    print("-" * W)

    ok_placement = placement_err_mm < 10.0
    ok_rotation = max_rotation_deg < 5.0
    ok_collision = min_arm_dist_mm > 0.0 if not np.isnan(min_arm_dist_mm) else True

    status = lambda ok: "PASS" if ok else "FAIL"
    print(f"  Object relocated (placement error < 10 mm)  : "
          f"{placement_err_mm:.1f} mm  [{status(ok_placement)}]")
    print(f"  No excessive rotation (< 5 deg)             : "
          f"{max_rotation_deg:.1f} deg  [{status(ok_rotation)}]")
    print(f"  No arm-arm collision (min dist > 0)          : "
          f"{min_arm_dist_mm:.1f} mm  [{status(ok_collision)}]")

    all_pass = ok_placement and ok_rotation and ok_collision
    print()
    if all_pass:
        print("  OVERALL: ALL CRITERIA PASSED")
    else:
        print("  OVERALL: SOME CRITERIA FAILED — review logs")
    print("=" * W)

    return {
        "placement_err_mm": placement_err_mm,
        "max_rotation_deg": max_rotation_deg,
        "min_arm_dist_mm": min_arm_dist_mm,
        "max_tau_left": max_tau_left,
        "max_tau_right": max_tau_right,
        "all_pass": all_pass,
    }


# ---------------------------------------------------------------------------
# Main capstone routine
# ---------------------------------------------------------------------------


def run_capstone() -> None:
    """Run the full Lab 6 capstone demo: bimanual pick, carry, and place."""

    print("=" * 63)
    print("Lab 6 Capstone — Bimanual Pick-Carry-Place")
    print("=" * 63)

    # ------------------------------------------------------------------
    # 1. Load MuJoCo model
    # ------------------------------------------------------------------
    print("\n[1/5] Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_DUAL_PATH))
    mj_data = mujoco.MjData(mj_model)

    # ------------------------------------------------------------------
    # 2. Build named index arrays for joints / actuators / box body
    # ------------------------------------------------------------------
    print("[2/5] Building joint / actuator index maps...")
    left_qpos_idx = get_joint_qpos_indices(mj_model, LEFT_JOINT_NAMES)
    left_dof_idx = get_joint_dof_indices(mj_model, LEFT_JOINT_NAMES)
    right_qpos_idx = get_joint_qpos_indices(mj_model, RIGHT_JOINT_NAMES)
    right_dof_idx = get_joint_dof_indices(mj_model, RIGHT_JOINT_NAMES)
    left_act_idx = get_actuator_indices(mj_model, LEFT_ACTUATOR_NAMES)
    right_act_idx = get_actuator_indices(mj_model, RIGHT_ACTUATOR_NAMES)
    box_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")

    # Box quaternion index in qpos (freejoint: 3 pos + 4 quat)
    box_jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
    if box_jnt_id >= 0:
        box_qpos_adr = mj_model.jnt_qposadr[box_jnt_id]
        box_quat_adr = box_qpos_adr + 3  # quaternion starts after 3 translation DOFs
    else:
        box_quat_adr = -1  # fallback: will use xquat from mj_data

    print(f"  Left qpos indices  : {left_qpos_idx}")
    print(f"  Right qpos indices : {right_qpos_idx}")
    print(f"  Box body id        : {box_bid}")

    # ------------------------------------------------------------------
    # 3. Set initial joint positions and run forward kinematics
    # ------------------------------------------------------------------
    print("[3/5] Setting home configuration...")
    for i, idx in enumerate(left_qpos_idx):
        mj_data.qpos[idx] = Q_HOME_LEFT[i]
    for i, idx in enumerate(right_qpos_idx):
        mj_data.qpos[idx] = Q_HOME_RIGHT[i]
    mujoco.mj_forward(mj_model, mj_data)

    # ------------------------------------------------------------------
    # 4. Create Pinocchio models and state machine
    # ------------------------------------------------------------------
    print("[4/5] Creating DualArmModel, BimanualGraspStateMachine, "
          "DualCollisionChecker...")

    dual_model = DualArmModel()

    # Task: pick box from (0.5, 0, 0.245), place at (0.5, -0.25, 0.245)
    object_pos = np.array([0.5, 0.0, 0.245])
    target_pos = np.array([0.5, -0.25, 0.245])

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

    # Collision checker for arm-arm distance logging
    collision_checker = DualCollisionChecker()

    # ------------------------------------------------------------------
    # 5. Simulation loop
    # ------------------------------------------------------------------
    print("[5/5] Running simulation (15 s)...")

    sim_duration = 15.0  # seconds
    n_steps = int(sim_duration / DT)
    log_every = 10            # 100 Hz log rate (1 kHz sim / 10 = 100 Hz)
    dist_check_every = 100    # check collision distance every 100 steps to save time

    n_log = n_steps // log_every + 1
    log_times = np.zeros(n_log)
    log_states: list[str] = []
    log_ee_left = np.zeros((n_log, 3))
    log_ee_right = np.zeros((n_log, 3))
    log_box_pos = np.zeros((n_log, 3))
    log_box_quat = np.zeros((n_log, 4))     # (w, x, y, z)
    log_tau_left = np.zeros((n_log, 6))
    log_tau_right = np.zeros((n_log, 6))

    # Sparse distance log (checked every dist_check_every steps)
    n_dist = n_steps // dist_check_every + 1
    log_dist_times = np.zeros(n_dist)
    log_min_dist = np.zeros(n_dist)

    log_idx = 0
    dist_idx = 0
    last_state_printed = ""

    def _read_box_quat(mj_data) -> np.ndarray:
        """Read box quaternion (w,x,y,z) from MuJoCo data."""
        if box_quat_adr >= 0:
            return mj_data.qpos[box_quat_adr: box_quat_adr + 4].copy()
        # Fallback to xquat (body rotation quaternion in world frame)
        return mj_data.xquat[box_bid].copy()

    for step in range(n_steps):
        t = mj_data.time

        # --- Read joint states --------------------------------------------
        q_left = np.array([mj_data.qpos[idx] for idx in left_qpos_idx])
        qd_left = np.array([mj_data.qvel[idx] for idx in left_dof_idx])
        q_right = np.array([mj_data.qpos[idx] for idx in right_qpos_idx])
        qd_right = np.array([mj_data.qvel[idx] for idx in right_dof_idx])

        # --- State machine step → torques ----------------------------------
        tau_left, tau_right = state_machine.step(
            q_left=q_left,
            qd_left=qd_left,
            q_right=q_right,
            qd_right=qd_right,
            t=t,
            mj_model=mj_model,
            mj_data=mj_data,
        )

        # --- Apply torques --------------------------------------------------
        for i, act_id in enumerate(left_act_idx):
            mj_data.ctrl[act_id] = tau_left[i]
        for i, act_id in enumerate(right_act_idx):
            mj_data.ctrl[act_id] = tau_right[i]

        # --- Log main data (every log_every steps) -------------------------
        if step % log_every == 0 and log_idx < n_log:
            log_times[log_idx] = t
            log_states.append(state_machine.state.name)
            log_ee_left[log_idx] = dual_model.fk_left_pos(q_left)
            log_ee_right[log_idx] = dual_model.fk_right_pos(q_right)
            log_box_pos[log_idx] = mj_data.xpos[box_bid].copy()
            log_box_quat[log_idx] = _read_box_quat(mj_data)
            log_tau_left[log_idx] = tau_left
            log_tau_right[log_idx] = tau_right
            log_idx += 1

        # --- Log arm-arm distance (every dist_check_every steps) -----------
        if step % dist_check_every == 0 and dist_idx < n_dist:
            d = collision_checker.get_min_distance(q_left, q_right)
            log_dist_times[dist_idx] = t
            log_min_dist[dist_idx] = d
            dist_idx += 1

        # --- Console state transitions -------------------------------------
        current_state = state_machine.state.name
        if current_state != last_state_printed:
            print(f"  t={t:6.3f}s  ->  {current_state}")
            last_state_printed = current_state

        # --- Step MuJoCo ---------------------------------------------------
        mujoco.mj_step(mj_model, mj_data)

        # Stop early once DONE; run short settle window
        if state_machine.is_done:
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
                    log_box_quat[log_idx] = _read_box_quat(mj_data)
                    log_tau_left[log_idx] = tau_left
                    log_tau_right[log_idx] = tau_right
                    log_idx += 1

            break

    total_time = mj_data.time
    print(f"\nSimulation complete. Simulated {total_time:.3f} s.")

    # Trim to actual logged data
    times = log_times[:log_idx]
    ee_left_arr = log_ee_left[:log_idx]
    ee_right_arr = log_ee_right[:log_idx]
    box_pos_arr = log_box_pos[:log_idx]
    box_quat_arr = log_box_quat[:log_idx]
    tau_left_arr = log_tau_left[:log_idx]
    tau_right_arr = log_tau_right[:log_idx]
    dist_times = log_dist_times[:dist_idx]
    min_dist_arr = log_min_dist[:dist_idx]

    state_log = state_machine.get_state_log()

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    _plot_state_timeline(total_time, state_log)
    _plot_ee_tracking(times, ee_left_arr, ee_right_arr)
    _plot_box_trajectory(times, box_pos_arr, state_log)
    _plot_min_distance(dist_times, min_dist_arr)
    _plot_torques(times, tau_left_arr, tau_right_arr)

    # ------------------------------------------------------------------
    # Print summary and validate success criteria
    # ------------------------------------------------------------------
    _print_summary(
        total_time=total_time,
        state_log=state_log,
        box_pos_init=box_pos_arr[0],
        box_pos_final=box_pos_arr[-1],
        target_pos=target_pos,
        box_quat_history=box_quat_arr,
        min_distances=min_dist_arr,
        tau_left_all=tau_left_arr,
        tau_right_all=tau_right_arr,
    )


if __name__ == "__main__":
    run_capstone()
