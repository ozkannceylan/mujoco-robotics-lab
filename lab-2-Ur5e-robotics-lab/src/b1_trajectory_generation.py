"""Phase 3.1 — Trajectory Generation.

Implements joint-space and Cartesian-space trajectory generators:
  1. Cubic polynomial (joint space)
  2. Quintic polynomial (joint space)
  3. Trapezoidal velocity profile (joint space)
  4. Minimum jerk (Cartesian space)
  5. Multi-segment via-point trajectory

Run:
    python3 lab-2-Ur5e-robotics-lab/src/b1_trajectory_generation.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ur5e_common import (
    NUM_JOINTS,
    Q_HOME,
    VELOCITY_LIMITS,
    load_pinocchio_model,
)


@dataclass
class JointTrajectoryPoint:
    """A single point in a joint-space trajectory."""
    t: float
    q: np.ndarray       # positions (6,)
    qd: np.ndarray      # velocities (6,)
    qdd: np.ndarray     # accelerations (6,)


# ---------------------------------------------------------------------------
# Joint-space trajectory generators
# ---------------------------------------------------------------------------


def cubic_trajectory(q_start: np.ndarray, q_end: np.ndarray,
                     T: float, dt: float) -> list[JointTrajectoryPoint]:
    """Generate a cubic polynomial trajectory.

    Boundary conditions: zero velocity at start and end.

    Args:
        q_start: Start joint angles (6,).
        q_end: End joint angles (6,).
        T: Duration (seconds).
        dt: Timestep (seconds).

    Returns:
        List of trajectory points.
    """
    a0 = q_start.copy()
    a1 = np.zeros(NUM_JOINTS)
    a2 = 3.0 * (q_end - q_start) / T**2
    a3 = -2.0 * (q_end - q_start) / T**3

    points = []
    t = 0.0
    while t <= T + dt / 2:
        t_clamped = min(t, T)
        q = a0 + a1 * t_clamped + a2 * t_clamped**2 + a3 * t_clamped**3
        qd = a1 + 2 * a2 * t_clamped + 3 * a3 * t_clamped**2
        qdd = 2 * a2 + 6 * a3 * t_clamped
        points.append(JointTrajectoryPoint(t=t_clamped, q=q, qd=qd, qdd=qdd))
        t += dt
    return points


def quintic_trajectory(q_start: np.ndarray, q_end: np.ndarray,
                       T: float, dt: float) -> list[JointTrajectoryPoint]:
    """Generate a quintic polynomial trajectory.

    Boundary conditions: zero velocity AND acceleration at start and end.

    Args:
        q_start: Start joint angles (6,).
        q_end: End joint angles (6,).
        T: Duration (seconds).
        dt: Timestep (seconds).

    Returns:
        List of trajectory points.
    """
    dq = q_end - q_start
    a0 = q_start.copy()
    a3 = 10.0 * dq / T**3
    a4 = -15.0 * dq / T**4
    a5 = 6.0 * dq / T**5

    points = []
    t = 0.0
    while t <= T + dt / 2:
        t_c = min(t, T)
        q = a0 + a3 * t_c**3 + a4 * t_c**4 + a5 * t_c**5
        qd = 3 * a3 * t_c**2 + 4 * a4 * t_c**3 + 5 * a5 * t_c**4
        qdd = 6 * a3 * t_c + 12 * a4 * t_c**2 + 20 * a5 * t_c**3
        points.append(JointTrajectoryPoint(t=t_c, q=q, qd=qd, qdd=qdd))
        t += dt
    return points


def trapezoidal_trajectory(q_start: np.ndarray, q_end: np.ndarray,
                           v_max: np.ndarray, a_max: np.ndarray,
                           dt: float) -> list[JointTrajectoryPoint]:
    """Generate a trapezoidal velocity profile trajectory.

    Three phases: acceleration, cruise, deceleration.
    All joints are synchronized to finish at the same time.

    Args:
        q_start: Start joint angles (6,).
        q_end: End joint angles (6,).
        v_max: Maximum velocity per joint (6,).
        a_max: Maximum acceleration per joint (6,).
        dt: Timestep (seconds).

    Returns:
        List of trajectory points.
    """
    dq = q_end - q_start
    signs = np.sign(dq)
    dq_abs = np.abs(dq)

    # Compute per-joint durations
    T_per_joint = np.zeros(NUM_JOINTS)
    for i in range(NUM_JOINTS):
        if dq_abs[i] < 1e-10:
            continue
        # Time to accelerate to v_max
        t_acc = v_max[i] / a_max[i]
        # Distance during acceleration
        d_acc = 0.5 * a_max[i] * t_acc**2
        if 2 * d_acc >= dq_abs[i]:
            # Triangle profile (no cruise phase)
            t_acc = np.sqrt(dq_abs[i] / a_max[i])
            T_per_joint[i] = 2 * t_acc
        else:
            # Trapezoidal profile
            d_cruise = dq_abs[i] - 2 * d_acc
            t_cruise = d_cruise / v_max[i]
            T_per_joint[i] = 2 * t_acc + t_cruise

    # Synchronize: all joints finish at the same time
    T_total = np.max(T_per_joint)
    if T_total < dt:
        return [JointTrajectoryPoint(t=0, q=q_start.copy(),
                                      qd=np.zeros(NUM_JOINTS),
                                      qdd=np.zeros(NUM_JOINTS))]

    points = []
    t = 0.0
    while t <= T_total + dt / 2:
        t_c = min(t, T_total)
        q = np.zeros(NUM_JOINTS)
        qd = np.zeros(NUM_JOINTS)
        qdd = np.zeros(NUM_JOINTS)

        for i in range(NUM_JOINTS):
            if dq_abs[i] < 1e-10:
                q[i] = q_start[i]
                continue

            # Recompute per-joint profile with synchronized T
            T_j = T_total
            # Effective acceleration for this joint
            v_eff = min(v_max[i], dq_abs[i] / (T_j / 2))
            a_eff = v_eff / (T_j / 2) if dq_abs[i] <= v_eff * T_j / 2 else a_max[i]
            t_acc_j = v_eff / a_eff
            t_dec_start = T_j - t_acc_j

            if t_c <= t_acc_j:
                # Acceleration phase
                q[i] = q_start[i] + signs[i] * 0.5 * a_eff * t_c**2
                qd[i] = signs[i] * a_eff * t_c
                qdd[i] = signs[i] * a_eff
            elif t_c <= t_dec_start:
                # Cruise phase
                d_acc = 0.5 * a_eff * t_acc_j**2
                q[i] = q_start[i] + signs[i] * (d_acc + v_eff * (t_c - t_acc_j))
                qd[i] = signs[i] * v_eff
                qdd[i] = 0.0
            else:
                # Deceleration phase
                t_dec = t_c - t_dec_start
                d_acc = 0.5 * a_eff * t_acc_j**2
                d_cruise = v_eff * (t_dec_start - t_acc_j)
                q[i] = q_start[i] + signs[i] * (d_acc + d_cruise
                        + v_eff * t_dec - 0.5 * a_eff * t_dec**2)
                qd[i] = signs[i] * (v_eff - a_eff * t_dec)
                qdd[i] = -signs[i] * a_eff

        points.append(JointTrajectoryPoint(t=t_c, q=q, qd=qd, qdd=qdd))
        t += dt
    return points


# ---------------------------------------------------------------------------
# Cartesian-space trajectory
# ---------------------------------------------------------------------------


def minimum_jerk_trajectory(p_start: np.ndarray, p_end: np.ndarray,
                            T: float, dt: float) -> list[tuple[float, np.ndarray]]:
    """Generate a minimum-jerk Cartesian trajectory (position only).

    s(t) = 10(t/T)^3 - 15(t/T)^4 + 6(t/T)^5

    Args:
        p_start: Start position (3,).
        p_end: End position (3,).
        T: Duration (seconds).
        dt: Timestep (seconds).

    Returns:
        List of (time, position) tuples.
    """
    points = []
    t = 0.0
    while t <= T + dt / 2:
        t_c = min(t, T)
        s = t_c / T
        sigma = 10 * s**3 - 15 * s**4 + 6 * s**5
        p = p_start + sigma * (p_end - p_start)
        points.append((t_c, p.copy()))
        t += dt
    return points


def multi_segment_trajectory(waypoints: list[np.ndarray],
                             durations: list[float],
                             dt: float) -> list[JointTrajectoryPoint]:
    """Generate a multi-segment quintic trajectory through waypoints.

    Args:
        waypoints: List of joint angle arrays [(6,), ...].
        durations: Duration for each segment [T1, T2, ...].
        dt: Timestep (seconds).

    Returns:
        List of trajectory points.
    """
    all_points = []
    t_offset = 0.0

    for seg in range(len(waypoints) - 1):
        seg_points = quintic_trajectory(
            waypoints[seg], waypoints[seg + 1], durations[seg], dt
        )
        for pt in seg_points:
            all_points.append(JointTrajectoryPoint(
                t=pt.t + t_offset, q=pt.q, qd=pt.qd, qdd=pt.qdd
            ))
        t_offset += durations[seg]

    return all_points


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate all trajectory types."""
    print("=" * 72)
    print("Phase 3.1 — Trajectory Generation")
    print("=" * 72)

    pin_model, pin_data, ee_fid = load_pinocchio_model()

    q_start = Q_HOME.copy()
    q_end = np.array([0.0, -np.pi / 2, np.pi / 3, -np.pi / 3, -np.pi / 2, 0.0])
    T = 2.0
    dt = 0.01

    # 1. Cubic
    print("\n1. Cubic polynomial trajectory")
    traj_cubic = cubic_trajectory(q_start, q_end, T, dt)
    print(f"   Points: {len(traj_cubic)}, Duration: {traj_cubic[-1].t:.2f}s")
    print(f"   Start vel: {np.linalg.norm(traj_cubic[0].qd):.6f} (should be ~0)")
    print(f"   End vel:   {np.linalg.norm(traj_cubic[-1].qd):.6f} (should be ~0)")

    # 2. Quintic
    print("\n2. Quintic polynomial trajectory")
    traj_quintic = quintic_trajectory(q_start, q_end, T, dt)
    print(f"   Points: {len(traj_quintic)}, Duration: {traj_quintic[-1].t:.2f}s")
    print(f"   Start vel: {np.linalg.norm(traj_quintic[0].qd):.6f}")
    print(f"   End vel:   {np.linalg.norm(traj_quintic[-1].qd):.6f}")
    print(f"   Start acc: {np.linalg.norm(traj_quintic[0].qdd):.6f} (should be ~0)")
    print(f"   End acc:   {np.linalg.norm(traj_quintic[-1].qdd):.6f} (should be ~0)")

    # 3. Trapezoidal
    print("\n3. Trapezoidal velocity profile")
    v_max = VELOCITY_LIMITS * 0.5  # 50% of max velocity
    a_max = np.full(NUM_JOINTS, 4.0)  # 4 rad/s^2
    traj_trap = trapezoidal_trajectory(q_start, q_end, v_max, a_max, dt)
    print(f"   Points: {len(traj_trap)}, Duration: {traj_trap[-1].t:.2f}s")
    max_vel = max(np.max(np.abs(pt.qd)) for pt in traj_trap)
    print(f"   Max velocity: {max_vel:.3f} rad/s")

    # 4. Minimum jerk (Cartesian)
    print("\n4. Minimum jerk (Cartesian space)")
    pin.forwardKinematics(pin_model, pin_data, q_start)
    pin.updateFramePlacements(pin_model, pin_data)
    p_start = pin_data.oMf[ee_fid].translation.copy()

    pin.forwardKinematics(pin_model, pin_data, q_end)
    pin.updateFramePlacements(pin_model, pin_data)
    p_end = pin_data.oMf[ee_fid].translation.copy()

    traj_mj = minimum_jerk_trajectory(p_start, p_end, T, dt)
    print(f"   Points: {len(traj_mj)}, Duration: {traj_mj[-1][0]:.2f}s")
    print(f"   Start: [{p_start[0]:.4f}, {p_start[1]:.4f}, {p_start[2]:.4f}]")
    print(f"   End:   [{p_end[0]:.4f}, {p_end[1]:.4f}, {p_end[2]:.4f}]")

    # 5. Multi-segment
    print("\n5. Multi-segment via-point trajectory")
    q_mid = np.array([-np.pi / 4, -np.pi / 2, np.pi / 4,
                       -np.pi / 4, -np.pi / 2, 0.0])
    waypoints = [q_start, q_mid, q_end]
    durations = [1.5, 1.5]
    traj_multi = multi_segment_trajectory(waypoints, durations, dt)
    print(f"   Waypoints: {len(waypoints)}, Segments: {len(durations)}")
    print(f"   Total points: {len(traj_multi)}, Duration: {traj_multi[-1].t:.2f}s")

    # Feasibility check
    print("\n--- Feasibility Check (UR5e velocity limits) ---")
    for name, traj in [("cubic", traj_cubic), ("quintic", traj_quintic),
                        ("trapezoidal", traj_trap)]:
        max_vels = np.max(np.abs(np.array([pt.qd for pt in traj])), axis=0)
        violations = max_vels > VELOCITY_LIMITS
        if np.any(violations):
            print(f"   {name}: VIOLATION on joints {np.where(violations)[0]}")
        else:
            print(f"   {name}: OK (all within limits)")

    print("\n" + "=" * 72)
    print("Phase 3.1 — Trajectory Generation: COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
