"""Lab 4 — Trajectory post-processing: shortcutting + TOPP-RA.

Takes a raw RRT/RRT* path and produces a time-optimal, smooth trajectory
respecting joint velocity and acceleration limits.
"""

from __future__ import annotations

import numpy as np
import toppra as ta
import toppra.constraint as constraint

from collision_checker import CollisionChecker


def shortcut_path(
    path: list[np.ndarray],
    collision_checker: CollisionChecker,
    max_iter: int = 200,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Shorten a C-space path by removing unnecessary waypoints.

    Randomly selects two non-adjacent waypoints and checks if the
    straight-line segment between them is collision-free. If so,
    removes all intermediate waypoints.

    Args:
        path: Ordered list of configurations.
        collision_checker: For edge collision checking.
        max_iter: Number of shortcut attempts.
        seed: Random seed for reproducibility.

    Returns:
        Shortened path (may be same length if no shortcuts found).
    """
    if len(path) <= 2:
        return list(path)

    rng = np.random.default_rng(seed)
    result = [q.copy() for q in path]

    for _ in range(max_iter):
        if len(result) <= 2:
            break
        # Pick two random indices with gap >= 2
        i = rng.integers(0, len(result) - 2)
        j = rng.integers(i + 2, len(result))

        if collision_checker.is_path_free(result[i], result[j]):
            # Remove intermediate waypoints
            result = result[: i + 1] + result[j:]

    return result


def parameterize_topp_ra(
    path: list[np.ndarray],
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-optimal path parameterization via TOPP-RA.

    Fits a cubic spline through the waypoints and computes the fastest
    feasible timing subject to velocity and acceleration constraints.

    Args:
        path: Ordered list of configurations (at least 2).
        vel_limits: Joint velocity limits (6,).
        acc_limits: Joint acceleration limits (6,).
        dt: Output sampling interval (seconds).

    Returns:
        Tuple of (times, positions, velocities, accelerations), each as
        numpy arrays. positions/velocities/accelerations have shape (N, 6).
    """
    n_waypoints = len(path)
    dof = path[0].shape[0]
    waypoints = np.array(path)  # (n_waypoints, dof)

    # Remove duplicate (or near-duplicate) consecutive waypoints
    filtered = [waypoints[0]]
    for i in range(1, n_waypoints):
        if np.linalg.norm(waypoints[i] - filtered[-1]) > 1e-8:
            filtered.append(waypoints[i])
    waypoints = np.array(filtered)
    n_waypoints = len(waypoints)

    # Handle degenerate case
    if n_waypoints < 2:
        times = np.array([0.0])
        return times, waypoints[:1], np.zeros((1, dof)), np.zeros((1, dof))

    # Assign path parameter values based on cumulative arc length
    ss = np.zeros(n_waypoints)
    for i in range(1, n_waypoints):
        ss[i] = ss[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])

    # Normalize to [0, 1]
    ss /= ss[-1]

    # Create spline interpolation
    path_spline = ta.SplineInterpolator(ss, waypoints)

    # Velocity and acceleration constraints
    vlim = constraint.JointVelocityConstraint(vel_limits)
    alim = constraint.JointAccelerationConstraint(acc_limits)

    # Solve TOPP-RA
    instance = ta.algorithm.TOPPRA(
        [vlim, alim], path_spline, parametrizer="ParametrizeConstAccel"
    )
    traj = instance.compute_trajectory()

    if traj is None:
        raise RuntimeError("TOPP-RA failed to find a feasible parameterization")

    # Sample the trajectory at dt intervals
    duration = traj.duration
    times = np.arange(0, duration + dt / 2, dt)
    # Clip to ensure we don't exceed duration due to floating point
    times = np.minimum(times, duration)

    positions = np.array([traj(t) for t in times])
    velocities = np.array([traj(t, 1) for t in times])
    accelerations = np.array([traj(t, 2) for t in times])

    return times, positions, velocities, accelerations
