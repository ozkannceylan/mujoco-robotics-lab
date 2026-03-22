"""Lab 6 — Phase 2 Demo: Coordinated simultaneous approach.

Both arms start at home, simultaneously move to pre-grasp poses
on opposite sides of the box. Demonstrates synchronized timing
and arm-arm collision avoidance.
"""

from __future__ import annotations

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
    BOX_INIT_POS,
    DT,
    MEDIA_DIR,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    load_mujoco_model,
)
from dual_arm_model import DualArmModel
from cooperative_controller import DualImpedanceController, DualImpedanceGains
from dual_collision_checker import DualCollisionChecker

# ---------------------------------------------------------------------------
# MuJoCo joint / actuator address helpers (same pattern as a1)
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

    Args:
        mj_model: MuJoCo model.

    Returns:
        Tuple of four lists of address integers.
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
    """Return (left_act_ids, right_act_ids) — indices into mj_data.ctrl.

    Args:
        mj_model: MuJoCo model.

    Returns:
        Tuple of two lists of actuator indices.
    """
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
# Pre-grasp target computation
# ---------------------------------------------------------------------------

# Box geometry: half-extent along X = 0.15 m.
# Pre-grasp clearance: 5 cm beyond the box face.
# Box centre: BOX_INIT_POS = (0.5, 0, 0.245)
# Left EE approaches from the -X side:  x = 0.5 - 0.15 - 0.05 = 0.30
# Right EE approaches from the +X side: x = 0.5 + 0.15 + 0.05 = 0.70
# Both stay at box centre height z = 0.245, y = 0.0.

_BOX_HALF_X = 0.15   # m, box half-extent along X
_CLEARANCE = 0.05    # m, 5 cm standoff from box face

PREGRASP_LEFT_POS = np.array([
    BOX_INIT_POS[0] - _BOX_HALF_X - _CLEARANCE,   # 0.30
    BOX_INIT_POS[1],                                 # 0.0
    BOX_INIT_POS[2],                                 # 0.245
])

PREGRASP_RIGHT_POS = np.array([
    BOX_INIT_POS[0] + _BOX_HALF_X + _CLEARANCE,   # 0.70
    BOX_INIT_POS[1],                                 # 0.0
    BOX_INIT_POS[2],                                 # 0.245
])


def _build_pregrasp_targets(dual_model: DualArmModel) -> tuple[pin.SE3, pin.SE3]:
    """Compute pre-grasp SE3 targets for both arms.

    The end-effectors approach the box along the X-axis from opposite sides.
    Orientation is taken from home FK so it remains physically achievable.

    Args:
        dual_model: DualArmModel for FK.

    Returns:
        Tuple of (target_left, target_right) as pin.SE3 in world frame.
    """
    home_left = dual_model.fk_left(Q_HOME_LEFT)
    home_right = dual_model.fk_right(Q_HOME_RIGHT)

    # Keep home orientation — only move position to pre-grasp
    target_left = pin.SE3(home_left.rotation.copy(), PREGRASP_LEFT_POS.copy())
    target_right = pin.SE3(home_right.rotation.copy(), PREGRASP_RIGHT_POS.copy())
    return target_left, target_right


def _solve_pregrasp_ik(
    dual_model: DualArmModel,
    target_left: pin.SE3,
    target_right: pin.SE3,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Solve IK for pre-grasp targets using home as the initial guess.

    Args:
        dual_model: DualArmModel with IK solvers.
        target_left: Desired left EE pose in world frame.
        target_right: Desired right EE pose in world frame.

    Returns:
        Tuple of (q_pregrasp_left, q_pregrasp_right), either may be None on failure.
    """
    q_left = dual_model.ik_left(target_left, Q_HOME_LEFT.copy(), max_iter=200, tol=1e-4)
    q_right = dual_model.ik_right(target_right, Q_HOME_RIGHT.copy(), max_iter=200, tol=1e-4)
    return q_left, q_right


# ---------------------------------------------------------------------------
# Initialise MuJoCo simulation state
# ---------------------------------------------------------------------------

def _set_home(mj_data, left_qadr: list[int], right_qadr: list[int]) -> None:
    """Write home joint angles into mj_data.qpos.

    Args:
        mj_data: MuJoCo data.
        left_qadr: Left joint qpos addresses.
        right_qadr: Right joint qpos addresses.
    """
    for k, adr in enumerate(left_qadr):
        mj_data.qpos[adr] = Q_HOME_LEFT[k]
    for k, adr in enumerate(right_qadr):
        mj_data.qpos[adr] = Q_HOME_RIGHT[k]


# ---------------------------------------------------------------------------
# Arrival detection
# ---------------------------------------------------------------------------

_ARRIVAL_THRESHOLD = 0.010  # 10 mm — declare "arrived" within this radius


def _arrival_time(times: np.ndarray, err: np.ndarray) -> float | None:
    """Return the first time err drops below the arrival threshold.

    Args:
        times: Time array (N,).
        err: Position error array (N,).

    Returns:
        Time of first arrival, or None if never reached.
    """
    idx = np.where(err < _ARRIVAL_THRESHOLD)[0]
    if len(idx) == 0:
        return None
    return float(times[idx[0]])


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    total_time: float = 5.0,
    collision_check_every: int = 50,
) -> dict:
    """Run the coordinated approach demo.

    Both arms start at home and simultaneously approach pre-grasp poses
    on opposite sides of the box (5 cm standoff from each face along X).
    Impedance control drives both arms at the same time.

    Args:
        total_time: Simulation duration in seconds (default 5 s).
        collision_check_every: Run Pinocchio collision check every N steps.

    Returns:
        Dictionary with logged arrays:
            times, ee_left, ee_right, q_left_log, q_right_log,
            err_left, err_right, min_dist_log, collision_events,
            target_left, target_right,
            q_pregrasp_left, q_pregrasp_right,
            arrival_time_left, arrival_time_right.
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

    # --- Pre-grasp targets ---
    target_left, target_right = _build_pregrasp_targets(dual_model)

    print("[a2] Pre-grasp target positions:")
    print(f"[a2]   Left  EE target: {target_left.translation}")
    print(f"[a2]   Right EE target: {target_right.translation}")
    print(f"[a2]   Box centre:      {BOX_INIT_POS}")

    # --- IK solutions ---
    q_pregrasp_left, q_pregrasp_right = _solve_pregrasp_ik(
        dual_model, target_left, target_right
    )

    if q_pregrasp_left is None:
        print("[a2] WARNING: Left arm IK failed — using home as fallback.")
        q_pregrasp_left = Q_HOME_LEFT.copy()
    else:
        ee_check_l = dual_model.fk_left_pos(q_pregrasp_left)
        print(
            f"[a2] Left  IK solved. FK check: {ee_check_l}  "
            f"(err={np.linalg.norm(ee_check_l - target_left.translation)*1e3:.2f} mm)"
        )

    if q_pregrasp_right is None:
        print("[a2] WARNING: Right arm IK failed — using home as fallback.")
        q_pregrasp_right = Q_HOME_RIGHT.copy()
    else:
        ee_check_r = dual_model.fk_right_pos(q_pregrasp_right)
        print(
            f"[a2] Right IK solved. FK check: {ee_check_r}  "
            f"(err={np.linalg.norm(ee_check_r - target_right.translation)*1e3:.2f} mm)"
        )

    # --- Check IK collision-free ---
    if not collision_checker.is_collision_free(q_pregrasp_left, q_pregrasp_right):
        print("[a2] WARNING: IK pre-grasp configuration has collision — check clearance.")
    else:
        print("[a2] IK pre-grasp configuration: COLLISION-FREE.")

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

    print(f"\n[a2] Starting simulation: {total_time:.1f} s, {n_steps} steps")
    print("[a2] Both arms move simultaneously from home to pre-grasp.")

    # --- Simulation loop ---
    for step in range(n_steps):
        t = step * DT
        times[step] = t

        # Read joint states from MuJoCo
        q_left = np.array([mj_data.qpos[a] for a in left_qadr])
        qd_left = np.array([mj_data.qvel[a] for a in left_vadr])
        q_right = np.array([mj_data.qpos[a] for a in right_qadr])
        qd_right = np.array([mj_data.qvel[a] for a in right_vadr])

        # Both arms target pre-grasp simultaneously for the full duration
        # (impedance control will drive them there and hold)
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

        # Log EE positions (Pinocchio FK for consistency)
        ee_left_log[step] = dual_model.fk_left_pos(q_left)
        ee_right_log[step] = dual_model.fk_right_pos(q_right)
        q_left_log[step] = q_left
        q_right_log[step] = q_right

        # Position tracking errors
        err_left_log[step] = np.linalg.norm(
            target_left.translation - ee_left_log[step]
        )
        err_right_log[step] = np.linalg.norm(
            target_right.translation - ee_right_log[step]
        )

        # Collision check (sparse)
        if step % collision_check_every == 0:
            min_d = collision_checker.get_min_distance(q_left, q_right)
            min_dist_log[step] = min_d
            if not collision_checker.is_collision_free(q_left, q_right):
                collision_events.append(t)
                print(
                    f"[a2] COLLISION detected at t={t:.3f} s  "
                    f"(min_dist={min_d:.4f} m)"
                )

        # Progress print
        if step % 1000 == 0:
            print(
                f"[a2] t={t:.2f}s  "
                f"err_L={err_left_log[step]*1e3:.1f}mm  "
                f"err_R={err_right_log[step]*1e3:.1f}mm"
            )

    print(f"\n[a2] Simulation complete. Collision events: {len(collision_events)}")
    if not collision_events:
        print("[a2] All configurations COLLISION-FREE.")

    # --- Arrival times ---
    arrival_time_left = _arrival_time(times, err_left_log)
    arrival_time_right = _arrival_time(times, err_right_log)

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
        "target_left": target_left,
        "target_right": target_right,
        "q_pregrasp_left": q_pregrasp_left,
        "q_pregrasp_right": q_pregrasp_right,
        "arrival_time_left": arrival_time_left,
        "arrival_time_right": arrival_time_right,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_coordinated_approach(data: dict, out_path: Path) -> None:
    """Generate a 4-panel figure showing both EE positions converging to targets.

    Panels:
      (top-left)     EE X positions vs time — convergence to target X
      (top-right)    EE Z positions vs time — convergence to target Z
      (bottom-left)  Position tracking errors (mm)
      (bottom-right) Minimum collision distance (cm)

    Args:
        data: Dictionary returned by run_simulation().
        out_path: Output PNG file path.
    """
    times = data["times"]
    ee_left = data["ee_left"]
    ee_right = data["ee_right"]
    err_left = data["err_left"] * 1e3     # m → mm
    err_right = data["err_right"] * 1e3
    target_left = data["target_left"]
    target_right = data["target_right"]
    arrival_l = data["arrival_time_left"]
    arrival_r = data["arrival_time_right"]
    collision_events = data["collision_events"]
    min_dist = data["min_dist"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Lab 6 — Phase 2: Coordinated Simultaneous Approach", fontsize=13)

    # ---- Panel 1: EE X positions ----
    ax = axes[0, 0]
    ax.plot(times, ee_left[:, 0], color="royalblue", linewidth=1.2, label="Left EE X")
    ax.plot(times, ee_right[:, 0], color="tomato", linewidth=1.2, label="Right EE X")
    ax.axhline(
        target_left.translation[0], color="royalblue", linestyle="--",
        linewidth=0.9, alpha=0.7, label=f"Left target X = {target_left.translation[0]:.3f} m"
    )
    ax.axhline(
        target_right.translation[0], color="tomato", linestyle="--",
        linewidth=0.9, alpha=0.7, label=f"Right target X = {target_right.translation[0]:.3f} m"
    )
    # Box boundaries
    box_x = 0.5
    box_half_x = 0.15
    ax.axvspan(
        times[0], times[-1],
        ymin=0, ymax=1,
        alpha=0.0,  # transparent — just for legend entry
    )
    ax.axhline(box_x - box_half_x, color="gray", linestyle=":", linewidth=0.8, alpha=0.7,
               label="Box left face (X=0.35)")
    ax.axhline(box_x + box_half_x, color="gray", linestyle=":", linewidth=0.8, alpha=0.7,
               label="Box right face (X=0.65)")
    if arrival_l is not None:
        ax.axvline(arrival_l, color="royalblue", linestyle=":", linewidth=1.0,
                   alpha=0.8, label=f"Left arrived t={arrival_l:.2f}s")
    if arrival_r is not None:
        ax.axvline(arrival_r, color="tomato", linestyle=":", linewidth=1.0,
                   alpha=0.8, label=f"Right arrived t={arrival_r:.2f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("X position (m)")
    ax.set_title("EE X Position — Convergence from Opposite Sides")
    ax.legend(fontsize=7, loc="center right")
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: EE Z positions ----
    ax = axes[0, 1]
    ax.plot(times, ee_left[:, 2], color="royalblue", linewidth=1.2, label="Left EE Z")
    ax.plot(times, ee_right[:, 2], color="tomato", linewidth=1.2, label="Right EE Z")
    ax.axhline(
        target_left.translation[2], color="royalblue", linestyle="--",
        linewidth=0.9, alpha=0.7, label=f"Target Z = {target_left.translation[2]:.3f} m"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z position (m)")
    ax.set_title("EE Z Position — Height During Approach")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Tracking errors ----
    ax = axes[1, 0]
    ax.plot(times, err_left, color="royalblue", linewidth=1.0, label="Left EE error")
    ax.plot(times, err_right, color="tomato", linewidth=1.0, label="Right EE error")
    ax.axhline(
        _ARRIVAL_THRESHOLD * 1e3, color="green", linestyle="--",
        linewidth=0.9, label=f"Arrival threshold ({_ARRIVAL_THRESHOLD*1e3:.0f} mm)"
    )
    if arrival_l is not None:
        ax.axvline(arrival_l, color="royalblue", linestyle=":", linewidth=1.0, alpha=0.8,
                   label=f"Left arrived t={arrival_l:.2f}s")
    if arrival_r is not None:
        ax.axvline(arrival_r, color="tomato", linestyle=":", linewidth=1.0, alpha=0.8,
                   label=f"Right arrived t={arrival_r:.2f}s")
    for t_col in collision_events:
        ax.axvline(t_col, color="magenta", linewidth=1.2, linestyle=":", alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EE position error (mm)")
    ax.set_title("Tracking Error — Simultaneous Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Minimum collision distance ----
    ax = axes[1, 1]
    valid = ~np.isnan(min_dist)
    t_valid = times[valid]
    d_valid = min_dist[valid] * 1e2   # m → cm
    ax.plot(
        t_valid, d_valid,
        color="darkorange", linewidth=1.2,
        marker="o", markersize=2, label="Min collision distance"
    )
    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.9, label="Collision threshold")
    for t_col in collision_events:
        ax.axvline(t_col, color="magenta", linewidth=1.2, linestyle=":",
                   alpha=0.9, label=f"Collision t={t_col:.2f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min distance (cm)")
    ax.set_title("Arm-Arm Collision Clearance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[a2] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the coordinated approach demo and generate plots."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    # Run simulation
    data = run_simulation(total_time=5.0, collision_check_every=50)

    # --- Summary ---
    times = data["times"]
    err_left = data["err_left"]
    err_right = data["err_right"]
    arrival_l = data["arrival_time_left"]
    arrival_r = data["arrival_time_right"]

    print("\n--- Coordinated Approach Summary ---")
    print(
        f"Left  arm: final error = {err_left[-1]*1e3:.1f} mm  |  "
        f"mean = {err_left.mean()*1e3:.1f} mm  |  "
        f"arrival = {arrival_l:.3f} s" if arrival_l is not None
        else f"Left  arm: final error = {err_left[-1]*1e3:.1f} mm  |  "
             f"mean = {err_left.mean()*1e3:.1f} mm  |  arrival = NOT REACHED"
    )
    print(
        f"Right arm: final error = {err_right[-1]*1e3:.1f} mm  |  "
        f"mean = {err_right.mean()*1e3:.1f} mm  |  "
        f"arrival = {arrival_r:.3f} s" if arrival_r is not None
        else f"Right arm: final error = {err_right[-1]*1e3:.1f} mm  |  "
             f"mean = {err_right.mean()*1e3:.1f} mm  |  arrival = NOT REACHED"
    )

    # Timing symmetry
    if arrival_l is not None and arrival_r is not None:
        delta_t = abs(arrival_l - arrival_r)
        print(f"Arrival time difference: {delta_t*1e3:.1f} ms  "
              f"({'synchronized' if delta_t < 0.1 else 'asymmetric'})")

    # Final EE positions
    q_left_final = data["q_left"][-1]
    q_right_final = data["q_right"][-1]
    dual_model = DualArmModel()
    ee_l_final = dual_model.fk_left_pos(q_left_final)
    ee_r_final = dual_model.fk_right_pos(q_right_final)
    target_left = data["target_left"]
    target_right = data["target_right"]

    print(f"\nFinal EE positions:")
    print(f"  Left  EE: {ee_l_final}  (target: {target_left.translation})")
    print(f"  Right EE: {ee_r_final}  (target: {target_right.translation})")
    print(f"  Left  final error: {np.linalg.norm(ee_l_final - target_left.translation)*1e3:.2f} mm")
    print(f"  Right final error: {np.linalg.norm(ee_r_final - target_right.translation)*1e3:.2f} mm")

    # Clearance from box faces
    box_x_left_face = 0.5 - 0.15   # 0.35
    box_x_right_face = 0.5 + 0.15  # 0.65
    clearance_l = abs(ee_l_final[0] - box_x_left_face)
    clearance_r = abs(ee_r_final[0] - box_x_right_face)
    print(f"\nBox face clearance:")
    print(f"  Left  EE to box left face (+X side): {clearance_l*1e3:.1f} mm  "
          f"(target: {_CLEARANCE*1e3:.0f} mm)")
    print(f"  Right EE to box right face (-X side): {clearance_r*1e3:.1f} mm  "
          f"(target: {_CLEARANCE*1e3:.0f} mm)")

    # Minimum collision distance
    min_dist = data["min_dist"]
    valid = ~np.isnan(min_dist)
    if valid.any():
        print(
            f"\nCollision distance: min={min_dist[valid].min()*1e2:.2f} cm  "
            f"mean={min_dist[valid].mean()*1e2:.2f} cm"
        )

    if not data["collision_events"]:
        print("Collision check: PASS — no collisions detected.")
    else:
        print(f"Collision check: FAIL — {len(data['collision_events'])} event(s).")

    # Generate plot
    plot_coordinated_approach(
        data,
        MEDIA_DIR / "coordinated_approach.png",
    )

    print("\n[a2] Done.")


if __name__ == "__main__":
    main()
