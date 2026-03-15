"""Lab 3 — Phase 3 Capstone: Constant-Force Line Tracing.

EE descends to table, establishes contact, then traces a straight line
in XY while maintaining a constant 5N downward force via hybrid control.

Demonstrates:
  - Approach → contact detection → hybrid mode switching
  - Simultaneous position tracking (XY) + force regulation (Z)
  - Smooth trajectory generation on contact surface
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mujoco
import pinocchio as pin

from lab3_common import (
    DT,
    NUM_JOINTS,
    MEDIA_DIR,
    SCENE_TABLE_PATH,
    clip_torques,
    load_mujoco_model,
    load_pinocchio_model,
)
from c1_force_control import (
    HybridGains,
    Q_ABOVE_TABLE,
    compute_hybrid_torque,
    get_ee_and_table_ids,
    read_contact_force_z,
)


def generate_line_trajectory(
    xy_start: np.ndarray,
    xy_end: np.ndarray,
    duration: float,
    dt: float = DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a straight-line XY trajectory with 5th-order polynomial profile.

    Uses s(t) = 10t^3 - 15t^4 + 6t^5 for zero velocity/acceleration at endpoints.

    Args:
        xy_start: Start XY position (2,).
        xy_end: End XY position (2,).
        duration: Time to complete the line (s).
        dt: Timestep.

    Returns:
        Tuple of (positions (N, 2), velocities (N, 2)).
    """
    n = int(duration / dt)
    t_arr = np.linspace(0, 1, n)

    s = 10 * t_arr**3 - 15 * t_arr**4 + 6 * t_arr**5
    ds = (30 * t_arr**2 - 60 * t_arr**3 + 30 * t_arr**4) / duration

    direction = xy_end - xy_start
    positions = xy_start[np.newaxis, :] + s[:, np.newaxis] * direction[np.newaxis, :]
    velocities = ds[:, np.newaxis] * direction[np.newaxis, :]

    return positions, velocities


def run_line_trace(
    xy_start: np.ndarray,
    xy_end: np.ndarray,
    gains: HybridGains | None = None,
    approach_height: float = 0.30,
    settle_time: float = 1.0,
    trace_duration: float = 4.0,
    post_time: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Run constant-force line tracing capstone demo.

    Phases:
      0 = approach (impedance descent to table)
      1 = settle (hold position, let force stabilize)
      2 = trace (follow line trajectory with hybrid control)
      3 = hold (maintain final position with hybrid control)

    Args:
        xy_start: Start XY on table (2,).
        xy_end: End XY on table (2,).
        gains: HybridGains.
        approach_height: Initial Z height (m).
        settle_time: Time to settle after contact (s).
        trace_duration: Time to trace the line (s).
        post_time: Time to hold at end (s).

    Returns:
        Dict with: time, q, qd, tau, ee_pos, force_z, phase, xy_des.
    """
    if gains is None:
        gains = HybridGains()

    pin_model, pin_data, ee_fid = load_pinocchio_model()
    mj_model, mj_data = load_mujoco_model(SCENE_TABLE_PATH)
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    ee_body_ids, table_geom_id = get_ee_and_table_ids(mj_model)
    table_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_z = mj_model.body_pos[table_body][2] + 0.02

    # Generate line trajectory
    xy_traj, xyd_traj = generate_line_trajectory(
        xy_start, xy_end, trace_duration, DT
    )

    # Initialize at Q_ABOVE_TABLE (EE at ~(0.4, 0, 0.35))
    mj_data.qpos[:NUM_JOINTS] = Q_ABOVE_TABLE.copy()
    mj_data.qvel[:NUM_JOINTS] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    total_duration = 15.0
    n_steps = int(total_duration / DT)

    time_list = []
    q_list = []
    qd_list = []
    tau_list = []
    ee_pos_list = []
    force_z_list = []
    phase_list = []
    xy_des_list = []

    force_integral = 0.0
    fz_filtered = 0.0
    alpha_filter = 0.2
    contact_established = False
    contact_time = None
    settle_done = False
    trace_start_time = None
    trace_done = False

    approach_Kp = np.diag([800.0, 800.0, 800.0])
    approach_Kd = np.diag([80.0, 80.0, 80.0])
    descent_speed = 0.05

    for i in range(n_steps):
        t = i * DT
        q = mj_data.qpos[:NUM_JOINTS].copy()
        qd = mj_data.qvel[:NUM_JOINTS].copy()

        # Read and filter force
        fz_raw = read_contact_force_z(mj_model, mj_data, ee_body_ids, table_geom_id)
        fz_filtered = alpha_filter * fz_raw + (1 - alpha_filter) * fz_filtered
        fz = fz_filtered

        ee_pos = mj_data.site_xpos[site_id].copy()

        # Contact detection
        if not contact_established and fz > 0.5:
            contact_established = True
            contact_time = t

        if not contact_established:
            # PHASE 0: Approach — impedance descent toward table
            phase = 0
            z_target = max(approach_height - descent_speed * t, table_z - 0.01)
            x_target = np.array([xy_start[0], xy_start[1], z_target])
            xy_des_cur = xy_start.copy()

            pin.forwardKinematics(pin_model, pin_data, q)
            pin.updateFramePlacements(pin_model, pin_data)
            pin.computeJointJacobians(pin_model, pin_data, q)
            J = pin.getFrameJacobian(
                pin_model, pin_data, ee_fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]
            x_cur = pin_data.oMf[ee_fid].translation.copy()
            v_cur = J @ qd
            F = approach_Kp @ (x_target - x_cur) + approach_Kd @ (-v_cur)
            pin.computeGeneralizedGravity(pin_model, pin_data, q)
            tau = J.T @ F + pin_data.g.copy()

        elif not settle_done:
            # PHASE 1: Settle — hold start position, let force stabilize
            phase = 1
            xy_des_cur = xy_start.copy()

            tau, force_integral = compute_hybrid_torque(
                pin_model, pin_data, ee_fid,
                q, qd, xy_start, None,
                fz, force_integral, gains,
            )

            if t - contact_time >= settle_time:
                settle_done = True
                trace_start_time = t

        elif not trace_done:
            # PHASE 2: Trace — follow line with hybrid XY+force control
            phase = 2
            t_since_trace = t - trace_start_time
            traj_idx = min(int(t_since_trace / DT), len(xy_traj) - 1)
            xy_des_cur = xy_traj[traj_idx]
            xyd_cur = xyd_traj[traj_idx]

            tau, force_integral = compute_hybrid_torque(
                pin_model, pin_data, ee_fid,
                q, qd, xy_des_cur, xyd_cur,
                fz, force_integral, gains,
            )

            if t_since_trace >= trace_duration:
                trace_done = True

        else:
            # PHASE 3: Hold at end position
            phase = 3
            xy_des_cur = xy_end.copy()

            tau, force_integral = compute_hybrid_torque(
                pin_model, pin_data, ee_fid,
                q, qd, xy_end, None,
                fz, force_integral, gains,
            )

            if t - (trace_start_time + trace_duration) >= post_time:
                tau = clip_torques(tau)
                time_list.append(t)
                q_list.append(q)
                qd_list.append(qd)
                tau_list.append(tau)
                ee_pos_list.append(ee_pos)
                force_z_list.append(fz)
                phase_list.append(phase)
                xy_des_list.append(xy_des_cur)
                break

        tau = clip_torques(tau)
        time_list.append(t)
        q_list.append(q)
        qd_list.append(qd)
        tau_list.append(tau)
        ee_pos_list.append(ee_pos)
        force_z_list.append(fz)
        phase_list.append(phase)
        xy_des_list.append(xy_des_cur)

        mj_data.ctrl[:NUM_JOINTS] = tau
        mujoco.mj_step(mj_model, mj_data)

    if contact_time is not None:
        print(f"  Contact at t={contact_time:.3f}s")
    else:
        print("  WARNING: No contact established!")
    if trace_start_time is not None:
        print(f"  Trace started at t={trace_start_time:.3f}s")

    return {
        "time": np.array(time_list),
        "q": np.array(q_list),
        "qd": np.array(qd_list),
        "tau": np.array(tau_list),
        "ee_pos": np.array(ee_pos_list),
        "force_z": np.array(force_z_list),
        "phase": np.array(phase_list),
        "xy_des": np.array(xy_des_list),
    }


def plot_line_trace(
    results: Dict[str, np.ndarray],
    xy_start: np.ndarray,
    xy_end: np.ndarray,
    F_desired: float = 5.0,
    save: bool = True,
) -> None:
    """Plot capstone line trace results.

    Args:
        results: Output from run_line_trace().
        xy_start: Line start XY.
        xy_end: Line end XY.
        F_desired: Desired force.
        save: Save to media/.
    """
    import matplotlib.pyplot as plt

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    t = results["time"]
    phase = results["phase"]
    trace_mask = phase == 2

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # --- Plot 1: Force vs time ---
    ax = axes[0]
    colors = {0: "#9E9E9E", 1: "#2196F3", 2: "#4CAF50", 3: "#FF9800"}
    phase_labels = {0: "Approach", 1: "Settle", 2: "Trace", 3: "Hold"}

    for p_val in sorted(set(phase)):
        mask = phase == p_val
        ax.plot(t[mask], results["force_z"][mask], ".", markersize=1,
                color=colors[p_val], label=phase_labels[p_val])

    ax.axhline(F_desired, color="r", linestyle="--", linewidth=1.5,
               label=f"Target ({F_desired}N)")
    ax.axhline(F_desired - 1, color="r", linestyle=":", alpha=0.3)
    ax.axhline(F_desired + 1, color="r", linestyle=":", alpha=0.3)
    ax.set_ylabel("Force Z (N)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Constant-Force Line Tracing — Force Profile")
    ax.legend(fontsize=9, ncol=5)
    ax.grid(alpha=0.3)

    # --- Plot 2: XY path ---
    ax = axes[1]
    ax.plot([xy_start[0] * 1000, xy_end[0] * 1000],
            [xy_start[1] * 1000, xy_end[1] * 1000],
            "r--", linewidth=2, label="Desired")
    if np.any(trace_mask):
        ax.plot(results["ee_pos"][trace_mask, 0] * 1000,
                results["ee_pos"][trace_mask, 1] * 1000,
                "g-", linewidth=1.5, label="Actual (trace)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("XY Path on Table Surface")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # --- Plot 3: XY tracking error during trace ---
    ax = axes[2]
    if np.any(trace_mask):
        trace_t = t[trace_mask]
        xy_actual = results["ee_pos"][trace_mask, :2]
        xy_desired = results["xy_des"][trace_mask]
        xy_err = np.linalg.norm(xy_actual - xy_desired, axis=1) * 1000
        ax.plot(trace_t, xy_err, "b-", linewidth=1.5)
        ax.axhline(5.0, color="r", linestyle="--", alpha=0.5, label="5mm threshold")
    ax.set_ylabel("XY Error (mm)")
    ax.set_xlabel("Time (s)")
    ax.set_title("XY Tracking Error During Trace")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(MEDIA_DIR / "capstone_line_trace.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run capstone constant-force line tracing demo."""
    print("=" * 60)
    print("Lab 3 — Capstone: Constant-Force Line Tracing")
    print("=" * 60)

    # 50mm line along X, centered on table
    xy_start = np.array([0.40, 0.0])
    xy_end = np.array([0.45, 0.0])
    gains = HybridGains(K_p=2000.0, K_d=100.0, K_fp=5.0, K_fi=10.0, F_desired=5.0)

    line_length = np.linalg.norm(xy_end - xy_start) * 1000
    print(f"\nLine: {xy_start} -> {xy_end} ({line_length:.0f}mm)")
    print(f"F_desired: {gains.F_desired} N")
    print("Phases: approach -> settle (1.5s) -> trace (6s) -> hold (1s)\n")

    results = run_line_trace(
        xy_start=xy_start,
        xy_end=xy_end,
        gains=gains,
        approach_height=0.30,
        settle_time=1.5,
        trace_duration=6.0,
        post_time=1.0,
    )

    # Analyze trace phase
    phase = results["phase"]
    trace_mask = phase == 2
    hybrid_mask = phase >= 1

    if np.any(trace_mask):
        fz_trace = results["force_z"][trace_mask]
        xy_actual = results["ee_pos"][trace_mask, :2]
        xy_desired = results["xy_des"][trace_mask]
        xy_err = np.linalg.norm(xy_actual - xy_desired, axis=1) * 1000

        print("\n--- Trace Phase Analysis ---")
        print(f"  Mean force:     {np.mean(fz_trace):.2f} N")
        print(f"  Std force:      {np.std(fz_trace):.2f} N")
        print(f"  Force in 5+/-1N: {np.mean(np.abs(fz_trace - gains.F_desired) < 1.0)*100:.1f}%")
        print(f"  Mean XY error:  {np.mean(xy_err):.2f} mm")
        print(f"  Max XY error:   {np.max(xy_err):.2f} mm")
        print(f"  XY < 5mm:       {np.mean(xy_err < 5.0)*100:.1f}%")

        fz_contact = results["force_z"][hybrid_mask]
        print(f"\n--- All Contact Phases ---")
        print(f"  Mean force:     {np.mean(fz_contact):.2f} N")
        print(f"  Force in 5+/-1N: {np.mean(np.abs(fz_contact - gains.F_desired) < 1.0)*100:.1f}%")
    else:
        print("\nWARNING: Trace phase never reached!")

    print("\nGenerating plots...")
    plot_line_trace(results, xy_start, xy_end, gains.F_desired)
    print(f"Plots saved to {MEDIA_DIR}/")


if __name__ == "__main__":
    main()
