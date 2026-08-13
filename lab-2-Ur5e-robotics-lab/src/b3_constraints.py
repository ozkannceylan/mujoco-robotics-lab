"""Phase 3.3 — Joint Constraints and Safety.

Implements safety constraints for UR5e control:
  1. Joint limit checking and clamping
  2. Velocity limit scaling
  3. Torque saturation
  4. Self-collision detection (geometric heuristic)
  5. Constraint-aware IK wrapper

Run:
    python3 lab-2-Ur5e-robotics-lab/src/b3_constraints.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ur5e_common import (
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    JOINT_NAMES,
    NUM_JOINTS,
    Q_HOME,
    TORQUE_LIMITS,
    VELOCITY_LIMITS,
    load_pinocchio_model,
)


# ---------------------------------------------------------------------------
# 1. Joint limit checking and clamping
# ---------------------------------------------------------------------------


def check_joint_limits(q: np.ndarray) -> tuple[bool, np.ndarray]:
    """Check if joint angles are within limits.

    Args:
        q: Joint angles (6,).

    Returns:
        Tuple of (all_within_limits, margin_to_limits).
        Margin is positive when within limits, negative when violated.
    """
    margin_lower = q - JOINT_LIMITS_LOWER
    margin_upper = JOINT_LIMITS_UPPER - q
    margin = np.minimum(margin_lower, margin_upper)
    return bool(np.all(margin >= 0)), margin


def clamp_joint_positions(q: np.ndarray) -> np.ndarray:
    """Clamp joint angles to within limits.

    Args:
        q: Joint angles (6,).

    Returns:
        Clamped joint angles (6,).
    """
    return np.clip(q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)


def joint_limit_repulsion(q: np.ndarray, buffer: float = 0.1,
                          gain: float = 5.0) -> np.ndarray:
    """Compute repulsive torques that push joints away from limits.

    Uses a spring-like potential in the buffer zone near limits.

    Args:
        q: Joint angles (6,).
        buffer: Buffer zone width (rad) near limits.
        gain: Repulsion gain.

    Returns:
        Repulsive torques (6,).
    """
    tau = np.zeros(NUM_JOINTS)
    for i in range(NUM_JOINTS):
        dist_lower = q[i] - JOINT_LIMITS_LOWER[i]
        dist_upper = JOINT_LIMITS_UPPER[i] - q[i]

        if dist_lower < buffer:
            tau[i] += gain * (buffer - dist_lower)
        if dist_upper < buffer:
            tau[i] -= gain * (buffer - dist_upper)
    return tau


# ---------------------------------------------------------------------------
# 2. Velocity limit scaling
# ---------------------------------------------------------------------------


def scale_velocity(qd: np.ndarray, limits: np.ndarray | None = None) -> np.ndarray:
    """Scale velocity vector to respect per-joint limits.

    If any joint exceeds its limit, the entire vector is scaled down
    proportionally to maintain the direction of motion.

    Args:
        qd: Joint velocities (6,).
        limits: Per-joint velocity limits. Defaults to VELOCITY_LIMITS.

    Returns:
        Scaled velocity (6,).
    """
    if limits is None:
        limits = VELOCITY_LIMITS
    ratios = np.abs(qd) / limits
    max_ratio = np.max(ratios)
    if max_ratio > 1.0:
        return qd / max_ratio
    return qd.copy()


def scale_delta_q(dq: np.ndarray, dt: float) -> np.ndarray:
    """Scale a position increment so the implied velocity stays within limits.

    Args:
        dq: Joint position increment (6,).
        dt: Timestep (seconds).

    Returns:
        Scaled position increment (6,).
    """
    implied_vel = dq / dt
    scaled_vel = scale_velocity(implied_vel)
    return scaled_vel * dt


# ---------------------------------------------------------------------------
# 3. Torque saturation
# ---------------------------------------------------------------------------


def saturate_torques(tau: np.ndarray,
                     limits: np.ndarray | None = None) -> np.ndarray:
    """Clip torques to within actuator limits.

    Args:
        tau: Commanded torques (6,).
        limits: Per-joint torque limits. Defaults to TORQUE_LIMITS.

    Returns:
        Saturated torques (6,).
    """
    if limits is None:
        limits = TORQUE_LIMITS
    return np.clip(tau, -limits, limits)


# ---------------------------------------------------------------------------
# 4. Self-collision detection (geometric heuristic)
# ---------------------------------------------------------------------------


def check_self_collision(q: np.ndarray, min_dist: float = 0.05) -> tuple[bool, list[str]]:
    """Check for self-collision using Pinocchio FK and geometric heuristics.

    Computes distances between non-adjacent link frames and flags if any
    are closer than min_dist. This is a simplified heuristic — for production
    use, HPP-FCL collision checking through Pinocchio would be preferred.

    Args:
        q: Joint angles (6,).
        min_dist: Minimum allowed distance (m) between non-adjacent links.

    Returns:
        Tuple of (collision_detected, list of collision descriptions).
    """
    model, data, _ = load_pinocchio_model()
    pin.forwardKinematics(model, data, q)

    collisions = []
    n_joints = model.njoints

    for i in range(1, n_joints):
        for j in range(i + 2, n_joints):  # skip adjacent links
            p_i = data.oMi[i].translation
            p_j = data.oMi[j].translation
            dist = np.linalg.norm(p_j - p_i)
            if dist < min_dist:
                name_i = model.names[i] if i < len(model.names) else f"joint_{i}"
                name_j = model.names[j] if j < len(model.names) else f"joint_{j}"
                collisions.append(
                    f"{name_i} <-> {name_j}: {dist:.4f}m (< {min_dist}m)"
                )

    return len(collisions) > 0, collisions


def self_collision_score(q: np.ndarray) -> float:
    """Compute a scalar collision proximity score.

    Returns the minimum distance between non-adjacent link frames.
    Lower values mean closer to self-collision.

    Args:
        q: Joint angles (6,).

    Returns:
        Minimum inter-link distance (m).
    """
    model, data, _ = load_pinocchio_model()
    pin.forwardKinematics(model, data, q)

    min_dist = float("inf")
    n_joints = model.njoints

    for i in range(1, n_joints):
        for j in range(i + 2, n_joints):
            p_i = data.oMi[i].translation
            p_j = data.oMi[j].translation
            dist = np.linalg.norm(p_j - p_i)
            min_dist = min(min_dist, dist)

    return min_dist


# ---------------------------------------------------------------------------
# 5. Constraint-aware command filter
# ---------------------------------------------------------------------------


def safe_command(q: np.ndarray, qd: np.ndarray, tau: np.ndarray,
                 dt: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply all safety constraints to a control command.

    Args:
        q: Current joint positions (6,).
        qd: Commanded joint velocities (6,).
        tau: Commanded torques (6,).
        dt: Timestep (seconds).

    Returns:
        Tuple of (safe_qd, safe_tau, info_dict).
    """
    info = {}

    # Velocity scaling
    qd_safe = scale_velocity(qd)
    info["vel_scaled"] = not np.allclose(qd, qd_safe)

    # Torque saturation
    tau_safe = saturate_torques(tau)
    info["tau_saturated"] = not np.allclose(tau, tau_safe)

    # Joint limit repulsion
    tau_repulse = joint_limit_repulsion(q)
    tau_safe = tau_safe + tau_repulse
    tau_safe = saturate_torques(tau_safe)
    info["repulsion_active"] = bool(np.any(np.abs(tau_repulse) > 1e-6))

    # Joint limit check
    q_next = q + qd_safe * dt
    valid, margin = check_joint_limits(q_next)
    if not valid:
        q_next = clamp_joint_positions(q_next)
        qd_safe = (q_next - q) / dt
        info["position_clamped"] = True
    else:
        info["position_clamped"] = False

    info["min_joint_margin"] = float(np.min(margin))

    return qd_safe, tau_safe, info


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate constraint checking on various configurations."""
    print("=" * 72)
    print("Phase 3.3 — Joint Constraints and Safety")
    print("=" * 72)

    # 1. Joint limit checking
    print("\n--- 1. Joint Limit Checking ---")
    configs = {
        "home": Q_HOME.copy(),
        "near_limit": np.array([6.0, -6.0, 3.0, -6.0, 6.0, -6.0]),
        "beyond_limit": np.array([7.0, -7.0, 4.0, -7.0, 7.0, -7.0]),
    }
    for name, q in configs.items():
        valid, margin = check_joint_limits(q)
        clamped = clamp_joint_positions(q)
        print(f"\n  {name}:")
        print(f"    Valid: {valid}")
        print(f"    Min margin: {np.min(margin):.4f} rad")
        if not valid:
            diff = np.linalg.norm(clamped - q)
            print(f"    Clamping correction: {diff:.4f} rad")

    # 2. Velocity scaling
    print("\n--- 2. Velocity Scaling ---")
    test_velocities = [
        ("within_limits", np.array([1.0, 2.0, 1.5, 3.0, 4.0, 5.0])),
        ("exceeding_limits", np.array([5.0, 5.0, 5.0, 10.0, 10.0, 10.0])),
    ]
    for name, qd in test_velocities:
        qd_scaled = scale_velocity(qd)
        print(f"\n  {name}:")
        print(f"    Original: [{', '.join(f'{v:.2f}' for v in qd)}]")
        print(f"    Scaled:   [{', '.join(f'{v:.2f}' for v in qd_scaled)}]")
        print(f"    Max ratio: {np.max(np.abs(qd_scaled) / VELOCITY_LIMITS):.4f}")

    # 3. Torque saturation
    print("\n--- 3. Torque Saturation ---")
    tau_test = np.array([200.0, -200.0, 100.0, 50.0, -40.0, 30.0])
    tau_sat = saturate_torques(tau_test)
    print(f"  Original: [{', '.join(f'{v:.1f}' for v in tau_test)}]")
    print(f"  Saturated: [{', '.join(f'{v:.1f}' for v in tau_sat)}]")

    # 4. Self-collision detection
    print("\n--- 4. Self-Collision Detection ---")
    collision_configs = {
        "home": Q_HOME.copy(),
        "folded": np.array([0.0, -np.pi / 2, np.pi, -np.pi / 2, 0.0, 0.0]),
        "extended": np.array([0.0, -np.pi / 4, np.pi / 2, -np.pi / 4, -np.pi / 2, 0.0]),
    }
    for name, q in collision_configs.items():
        collision, details = check_self_collision(q)
        score = self_collision_score(q)
        print(f"\n  {name}:")
        print(f"    Collision: {collision}")
        print(f"    Min distance: {score:.4f} m")
        if details:
            for d in details[:3]:  # show max 3
                print(f"      {d}")

    # 5. Joint limit repulsion
    print("\n--- 5. Joint Limit Repulsion ---")
    q_near = np.array([6.1, -6.1, 2.9, 0.0, 0.0, 0.0])
    tau_repulse = joint_limit_repulsion(q_near)
    print(f"  q near limits: [{', '.join(f'{v:.2f}' for v in q_near)}]")
    print(f"  Repulsion tau: [{', '.join(f'{v:.4f}' for v in tau_repulse)}]")

    # 6. Safe command filter
    print("\n--- 6. Safe Command Filter ---")
    q_test = Q_HOME.copy()
    qd_test = np.array([5.0, 5.0, 5.0, 10.0, 10.0, 10.0])
    tau_cmd = np.array([200.0, -200.0, 100.0, 50.0, -40.0, 30.0])
    qd_safe, tau_safe, info = safe_command(q_test, qd_test, tau_cmd, dt=0.002)
    print(f"  Velocity scaled: {info['vel_scaled']}")
    print(f"  Torque saturated: {info['tau_saturated']}")
    print(f"  Repulsion active: {info['repulsion_active']}")
    print(f"  Position clamped: {info['position_clamped']}")
    print(f"  Min joint margin: {info['min_joint_margin']:.4f} rad")

    print("\n" + "=" * 72)
    print("Phase 3.3 — Constraints: COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
