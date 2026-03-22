"""Lab 6 — Phase 1 Demo: Independent arm motion.

Moves each arm through waypoints independently using impedance control.
Verifies no arm-arm or arm-environment collisions.
Generates plots of EE trajectories and joint tracking.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

# Ensure src is on the path
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mujoco

from lab6_common import (
    DT,
    MEDIA_DIR,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    TABLE_SURFACE_Z,
    load_mujoco_model,
)
from dual_arm_model import DualArmModel
from cooperative_controller import DualImpedanceController, DualImpedanceGains
from dual_collision_checker import DualCollisionChecker

# ---------------------------------------------------------------------------
# MuJoCo joint / actuator address helpers
# ---------------------------------------------------------------------------

_LEFT_JOINT_NAMES = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
]

_RIGHT_JOINT_NAMES = [
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_elbow_joint",
    "right_wrist_1_joint",
    "right_wrist_2_joint",
    "right_wrist_3_joint",
]

_LEFT_ACT_NAMES = [f"left_motor{i}" for i in range(1, 7)]
_RIGHT_ACT_NAMES = [f"right_motor{i}" for i in range(1, 7)]


def _get_joint_addresses(mj_model) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return (left_qadr, right_qadr, left_vadr, right_vadr).

    Each element is a list of 6 integer addresses into mj_data.qpos / qvel.
    """
    left_qadr = [
        mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in _LEFT_JOINT_NAMES
    ]
    right_qadr = [
        mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in _RIGHT_JOINT_NAMES
    ]
    left_vadr = [
        mj_model.jnt_dofadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in _LEFT_JOINT_NAMES
    ]
    right_vadr = [
        mj_model.jnt_dofadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in _RIGHT_JOINT_NAMES
    ]
    return left_qadr, right_qadr, left_vadr, right_vadr


def _get_actuator_ids(mj_model) -> tuple[list[int], list[int]]:
    """Return (left_act_ids, right_act_ids) — indices into mj_data.ctrl."""
    left_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in _LEFT_ACT_NAMES
    ]
    right_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in _RIGHT_ACT_NAMES
    ]
    return left_ids, right_ids


# ---------------------------------------------------------------------------
# Waypoint definitions
# ---------------------------------------------------------------------------

def _build_waypoints(dual_model: DualArmModel) -> tuple[list[pin.SE3], list[pin.SE3]]:
    """Define Cartesian waypoints for each arm.

    Left arm moves during [0, 4) seconds; right arm during [4, 8) seconds.
    Waypoints are expressed as SE3 targets in the world frame.

    Returns:
        (left_waypoints, right_waypoints) — each a list of pin.SE3 objects.
    """
    # Orientation: end-effector pointing downward (−Z in world → tool Z-axis)
    # UR5e home FK gives a reasonable end-effector orientation; we reuse it
    # for all waypoints and only vary position.
    home_left_pose = dual_model.fk_left(Q_HOME_LEFT)
    home_right_pose = dual_model.fk_right(Q_HOME_RIGHT)

    R_left = home_left_pose.rotation.copy()
    R_right = home_right_pose.rotation.copy()

    # ---------- left arm waypoints ----------
    # WP0: home position (used as start/end anchor)
    p_left_home = home_left_pose.translation.copy()

    # WP1: reach forward-left of the table at safe height with y-offset to clear
    #      the table surface. y=0.20 keeps arm links well clear of table top.
    p_left_reach = np.array([0.35, 0.20, 0.40])

    # WP2: lift up while holding forward position
    p_left_lift = np.array([0.38, 0.20, 0.50])

    # WP3: return to home (same as WP0; controller handles it)
    # Waypoint list drives sequencing: home → reach → lift → home
    left_waypoints = [
        pin.SE3(R_left, p_left_home),
        pin.SE3(R_left, p_left_reach),
        pin.SE3(R_left, p_left_lift),
        pin.SE3(R_left, p_left_home),
    ]

    # ---------- right arm waypoints ----------
    p_right_home = home_right_pose.translation.copy()

    # Right arm reaches forward-right of the table (mirrored, keeping y=0.20 offset)
    p_right_reach = np.array([0.65, 0.20, 0.40])
    p_right_lift = np.array([0.62, 0.20, 0.50])

    right_waypoints = [
        pin.SE3(R_right, p_right_home),
        pin.SE3(R_right, p_right_reach),
        pin.SE3(R_right, p_right_lift),
        pin.SE3(R_right, p_right_home),
    ]

    return left_waypoints, right_waypoints


# ---------------------------------------------------------------------------
# Initialise MuJoCo simulation state
# ---------------------------------------------------------------------------

def _set_home(mj_data, left_qadr: list[int], right_qadr: list[int]) -> None:
    """Write home joint angles into mj_data.qpos."""
    for k, adr in enumerate(left_qadr):
        mj_data.qpos[adr] = Q_HOME_LEFT[k]
    for k, adr in enumerate(right_qadr):
        mj_data.qpos[adr] = Q_HOME_RIGHT[k]


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    total_time: float = 8.0,
    collision_check_every: int = 50,
) -> dict:
    """Run the independent motion demo.

    Left arm moves for the first 4 s; right arm moves for the last 4 s.
    Both arms use impedance control at all times (hold position when idle).

    Args:
        total_time: Simulation duration in seconds.
        collision_check_every: Run collision check every N steps (expensive).

    Returns:
        Dictionary with logged time-series arrays:
            times, ee_left, ee_right, q_left_log, q_right_log,
            err_left, err_right, min_dist_log, collision_events.
    """
    # --- Load models ---
    mj_model, mj_data = load_mujoco_model()
    dual_model = DualArmModel()
    controller = DualImpedanceController(
        dual_model,
        DualImpedanceGains(
            K_p=np.diag([800.0, 800.0, 800.0, 80.0, 80.0, 80.0]),
            K_d=np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0]),
        ),
    )
    collision_checker = DualCollisionChecker()

    # --- Address maps ---
    left_qadr, right_qadr, left_vadr, right_vadr = _get_joint_addresses(mj_model)
    left_act_ids, right_act_ids = _get_actuator_ids(mj_model)

    # --- Set home configuration ---
    _set_home(mj_data, left_qadr, right_qadr)
    mujoco.mj_forward(mj_model, mj_data)

    # --- Waypoints ---
    left_waypoints, right_waypoints = _build_waypoints(dual_model)

    # Timing: left arm uses [0, 4 s], right arm uses [4 s, 8 s]
    half_time = total_time / 2.0
    # Each arm visits 4 waypoints (including home start/end).
    # Segment duration = half_time / (num_waypoints - 1)
    n_waypoints = len(left_waypoints)  # 4
    segment_duration = half_time / (n_waypoints - 1)  # ~1.33 s per segment

    # --- Storage ---
    n_steps = int(total_time / DT)
    times = np.zeros(n_steps)
    ee_left_log = np.zeros((n_steps, 3))
    ee_right_log = np.zeros((n_steps, 3))
    q_left_log = np.zeros((n_steps, 6))
    q_right_log = np.zeros((n_steps, 6))
    err_left_log = np.zeros(n_steps)
    err_right_log = np.zeros(n_steps)
    min_dist_log = np.full(n_steps, np.nan)
    collision_events: list[float] = []

    print(f"[a1] Starting simulation: {total_time:.1f} s, {n_steps} steps")
    print(f"[a1] Left arm: t=[0, {half_time:.1f}] s  |  "
          f"Right arm: t=[{half_time:.1f}, {total_time:.1f}] s")

    # --- Simulation loop ---
    for step in range(n_steps):
        t = step * DT
        times[step] = t

        # Read joint states
        q_left = np.array([mj_data.qpos[a] for a in left_qadr])
        qd_left = np.array([mj_data.qvel[a] for a in left_vadr])
        q_right = np.array([mj_data.qpos[a] for a in right_qadr])
        qd_right = np.array([mj_data.qvel[a] for a in right_vadr])

        # Determine active target for each arm
        if t < half_time:
            # Left arm is moving; determine which segment we are in
            seg_idx = min(int(t / segment_duration), n_waypoints - 2)
            alpha = (t - seg_idx * segment_duration) / segment_duration
            alpha = np.clip(alpha, 0.0, 1.0)
            # Interpolate position between consecutive waypoints
            p0 = left_waypoints[seg_idx].translation
            p1 = left_waypoints[seg_idx + 1].translation
            R_interp = left_waypoints[seg_idx].rotation  # keep orientation fixed
            target_left = pin.SE3(R_interp, p0 + alpha * (p1 - p0))
        else:
            # Left arm holds home
            target_left = left_waypoints[0]

        if t >= half_time:
            # Right arm is moving
            t_right = t - half_time
            seg_idx = min(int(t_right / segment_duration), n_waypoints - 2)
            alpha = (t_right - seg_idx * segment_duration) / segment_duration
            alpha = np.clip(alpha, 0.0, 1.0)
            p0 = right_waypoints[seg_idx].translation
            p1 = right_waypoints[seg_idx + 1].translation
            R_interp = right_waypoints[seg_idx].rotation
            target_right = pin.SE3(R_interp, p0 + alpha * (p1 - p0))
        else:
            # Right arm holds home
            target_right = right_waypoints[0]

        # Compute impedance torques
        tau_left, tau_right = controller.compute_dual_torques(
            q_left, qd_left,
            q_right, qd_right,
            target_left, target_right,
            grasping=False,
        )

        # Write torques to MuJoCo ctrl
        for k, aid in enumerate(left_act_ids):
            mj_data.ctrl[aid] = tau_left[k]
        for k, aid in enumerate(right_act_ids):
            mj_data.ctrl[aid] = tau_right[k]

        # Step simulation
        mujoco.mj_step(mj_model, mj_data)

        # Log EE positions (from Pinocchio FK for consistency)
        ee_left_log[step] = dual_model.fk_left_pos(q_left)
        ee_right_log[step] = dual_model.fk_right_pos(q_right)
        q_left_log[step] = q_left
        q_right_log[step] = q_right

        # Position tracking error
        err_left_log[step] = np.linalg.norm(
            target_left.translation - ee_left_log[step]
        )
        err_right_log[step] = np.linalg.norm(
            target_right.translation - ee_right_log[step]
        )

        # Collision check (sparse, expensive)
        if step % collision_check_every == 0:
            min_d = collision_checker.get_min_distance(q_left, q_right)
            min_dist_log[step] = min_d
            if not collision_checker.is_collision_free(q_left, q_right):
                collision_events.append(t)
                print(f"[a1] COLLISION detected at t={t:.3f} s  (min_dist={min_d:.4f} m)")

        # Progress print
        if step % 1000 == 0:
            print(
                f"[a1] t={t:.2f}s  "
                f"err_L={err_left_log[step]*1e3:.1f}mm  "
                f"err_R={err_right_log[step]*1e3:.1f}mm"
            )

    print(f"\n[a1] Simulation complete. "
          f"Collision events: {len(collision_events)}")
    if not collision_events:
        print("[a1] All configurations COLLISION-FREE.")

    return {
        "times": times,
        "ee_left": ee_left_log,
        "ee_right": ee_right_log,
        "q_left": q_left_log,
        "q_right": q_right_log,
        "err_left": err_left_log,
        "err_right": err_right_log,
        "min_dist": min_dist_log,
        "collision_events": collision_events,
        "left_waypoints": left_waypoints,
        "right_waypoints": right_waypoints,
        "half_time": half_time,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ee_trajectories(data: dict, out_path: Path) -> None:
    """3D plot of both EE trajectories.

    Args:
        data: Dictionary returned by run_simulation().
        out_path: Output PNG path.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    times = data["times"]
    half_time = data["half_time"]
    ee_left = data["ee_left"]
    ee_right = data["ee_right"]

    # Colour-map: blue gradient for left, red gradient for right
    n = len(times)
    left_mask = times < half_time
    right_mask = times >= half_time

    # Left arm trajectory (blue)
    ax.plot(
        ee_left[left_mask, 0],
        ee_left[left_mask, 1],
        ee_left[left_mask, 2],
        color="royalblue",
        linewidth=1.5,
        label="Left arm (0–4 s)",
        alpha=0.85,
    )
    # Static hold during right arm phase (dashed)
    ax.plot(
        ee_left[right_mask, 0],
        ee_left[right_mask, 1],
        ee_left[right_mask, 2],
        color="royalblue",
        linewidth=0.8,
        linestyle="--",
        alpha=0.35,
        label="Left arm hold (4–8 s)",
    )

    # Right arm trajectory (red)
    ax.plot(
        ee_right[left_mask, 0],
        ee_right[left_mask, 1],
        ee_right[left_mask, 2],
        color="tomato",
        linewidth=0.8,
        linestyle="--",
        alpha=0.35,
        label="Right arm hold (0–4 s)",
    )
    ax.plot(
        ee_right[right_mask, 0],
        ee_right[right_mask, 1],
        ee_right[right_mask, 2],
        color="tomato",
        linewidth=1.5,
        label="Right arm (4–8 s)",
        alpha=0.85,
    )

    # Mark waypoints
    for wp in data["left_waypoints"]:
        ax.scatter(*wp.translation, color="navy", s=40, zorder=5)
    for wp in data["right_waypoints"]:
        ax.scatter(*wp.translation, color="darkred", s=40, zorder=5)

    # Mark collision events
    for t_col in data["collision_events"]:
        idx = int(t_col / DT)
        ax.scatter(
            ee_left[idx, 0], ee_left[idx, 1], ee_left[idx, 2],
            color="magenta", s=80, marker="X", zorder=10,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Lab 6 — Independent Motion: EE Trajectories")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[a1] Saved: {out_path}")


def plot_tracking(data: dict, out_path: Path) -> None:
    """Position tracking error over time for both arms.

    Args:
        data: Dictionary returned by run_simulation().
        out_path: Output PNG path.
    """
    times = data["times"]
    err_left = data["err_left"] * 1e3   # m → mm
    err_right = data["err_right"] * 1e3
    half_time = data["half_time"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- subplot 1: left arm error ---
    ax = axes[0]
    ax.plot(times, err_left, color="royalblue", linewidth=1.0, label="Left EE error")
    ax.axvspan(0, half_time, alpha=0.08, color="royalblue", label="Left active phase")
    ax.set_ylabel("EE position error (mm)")
    ax.set_title("Left Arm — EE Position Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- subplot 2: right arm error ---
    ax = axes[1]
    ax.plot(times, err_right, color="tomato", linewidth=1.0, label="Right EE error")
    ax.axvspan(half_time, times[-1], alpha=0.08, color="tomato", label="Right active phase")
    ax.set_ylabel("EE position error (mm)")
    ax.set_title("Right Arm — EE Position Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- subplot 3: minimum collision distance ---
    ax = axes[2]
    min_dist = data["min_dist"]
    valid = ~np.isnan(min_dist)
    t_valid = times[valid]
    d_valid = min_dist[valid] * 1e2  # m → cm
    ax.plot(t_valid, d_valid, color="darkorange", linewidth=1.2,
            marker="o", markersize=3, label="Min collision distance")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0, label="Collision threshold (0 m)")
    ax.set_ylabel("Min distance (cm)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Minimum Collision Distance (arm-arm & arm-env)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mark collision events
    for t_col in data["collision_events"]:
        for a in axes:
            a.axvline(t_col, color="magenta", linewidth=1.0, linestyle=":")

    # Vertical line separating phases
    for a in axes:
        a.axvline(half_time, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.suptitle("Lab 6 — Phase 1: Independent Motion Tracking", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[a1] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the independent motion demo and generate plots."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    # Run simulation
    data = run_simulation(total_time=8.0, collision_check_every=50)

    # Summary statistics
    half_time = data["half_time"]
    times = data["times"]
    err_left = data["err_left"]
    err_right = data["err_right"]
    left_active = times < half_time
    right_active = times >= half_time

    print("\n--- Tracking Summary ---")
    print(
        f"Left arm  (active phase): mean={err_left[left_active].mean()*1e3:.1f} mm  "
        f"max={err_left[left_active].max()*1e3:.1f} mm"
    )
    print(
        f"Right arm (active phase): mean={err_right[right_active].mean()*1e3:.1f} mm  "
        f"max={err_right[right_active].max()*1e3:.1f} mm"
    )

    # Minimum distance summary
    min_dist = data["min_dist"]
    valid = ~np.isnan(min_dist)
    if valid.any():
        print(
            f"Min collision distance over run: "
            f"{min_dist[valid].min()*1e2:.2f} cm  "
            f"(mean: {min_dist[valid].mean()*1e2:.2f} cm)"
        )

    if not data["collision_events"]:
        print("Collision check: PASS — no collisions detected.")
    else:
        print(f"Collision check: FAIL — {len(data['collision_events'])} event(s).")

    # Generate plots
    plot_ee_trajectories(
        data,
        MEDIA_DIR / "independent_motion_ee_trajectories.png",
    )
    plot_tracking(
        data,
        MEDIA_DIR / "independent_motion_tracking.png",
    )

    print("\n[a1] Done.")


if __name__ == "__main__":
    main()
