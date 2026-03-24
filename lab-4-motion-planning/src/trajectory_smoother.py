"""Lab 4 — Trajectory post-processing: shortcutting + time parameterization."""

from __future__ import annotations

import math

import numpy as np

try:
    import toppra as ta
    import toppra.constraint as constraint

    _TOPPRA_AVAILABLE = True
except Exception:
    ta = None
    constraint = None
    _TOPPRA_AVAILABLE = False

from collision_checker import CollisionChecker


def shortcut_path(
    path: list[np.ndarray],
    collision_checker: CollisionChecker,
    max_iter: int = 200,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Shorten a C-space path by removing unnecessary waypoints."""
    if len(path) <= 2:
        return list(path)

    rng = np.random.default_rng(seed)
    result = [q.copy() for q in path]

    for _ in range(max_iter):
        if len(result) <= 2:
            break
        i = rng.integers(0, len(result) - 2)
        j = rng.integers(i + 2, len(result))
        if collision_checker.is_path_free(result[i], result[j]):
            result = result[: i + 1] + result[j:]

    return result


def _filter_duplicate_waypoints(path: list[np.ndarray]) -> np.ndarray:
    """Remove duplicate consecutive waypoints."""
    waypoints = np.array(path, dtype=float)
    filtered = [waypoints[0]]
    for i in range(1, len(waypoints)):
        if np.linalg.norm(waypoints[i] - filtered[-1]) > 1e-8:
            filtered.append(waypoints[i])
    return np.array(filtered, dtype=float)


def _parameterize_quintic(
    path: list[np.ndarray],
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conservative quintic timing fallback when TOPP-RA is unavailable."""
    waypoints = _filter_duplicate_waypoints(path)
    dof = waypoints.shape[1]
    if len(waypoints) < 2:
        times = np.array([0.0])
        zeros = np.zeros((1, dof))
        return times, waypoints[:1], zeros, zeros

    all_t = []
    all_q = []
    all_qd = []
    all_qdd = []
    t_offset = 0.0

    vel_scale = 1.90
    acc_scale = 6.0

    for seg_idx in range(len(waypoints) - 1):
        q0 = waypoints[seg_idx]
        q1 = waypoints[seg_idx + 1]
        delta = q1 - q0

        T_vel = vel_scale * float(np.max(np.abs(delta) / np.maximum(vel_limits, 1e-8)))
        T_acc = math.sqrt(acc_scale * float(np.max(np.abs(delta) / np.maximum(acc_limits, 1e-8))))
        T = max(T_vel, T_acc, 5 * dt)

        n_steps = max(2, int(np.ceil(T / dt)) + 1)
        u = np.linspace(0.0, 1.0, n_steps)
        if seg_idx < len(waypoints) - 2:
            u = u[:-1]

        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / T
        dds = (60 * u - 180 * u**2 + 120 * u**3) / (T**2)

        q_seg = q0[np.newaxis, :] + s[:, np.newaxis] * delta[np.newaxis, :]
        qd_seg = ds[:, np.newaxis] * delta[np.newaxis, :]
        qdd_seg = dds[:, np.newaxis] * delta[np.newaxis, :]
        t_seg = t_offset + u * T

        all_t.append(t_seg)
        all_q.append(q_seg)
        all_qd.append(qd_seg)
        all_qdd.append(qdd_seg)

        t_offset += T

    times = np.concatenate(all_t)
    positions = np.vstack(all_q)
    velocities = np.vstack(all_qd)
    accelerations = np.vstack(all_qdd)
    return times, positions, velocities, accelerations


def parameterize_topp_ra(
    path: list[np.ndarray],
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-parameterize a path using TOPP-RA when available, else quintic fallback."""
    waypoints = _filter_duplicate_waypoints(path)
    if len(waypoints) < 2:
        times = np.array([0.0])
        zeros = np.zeros((1, waypoints.shape[1]))
        return times, waypoints[:1], zeros, zeros

    if not _TOPPRA_AVAILABLE:
        return _parameterize_quintic(list(waypoints), vel_limits, acc_limits, dt)

    n_waypoints = len(waypoints)
    dof = waypoints.shape[1]
    ss = np.zeros(n_waypoints)
    for i in range(1, n_waypoints):
        ss[i] = ss[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])

    ss /= ss[-1]
    path_spline = ta.SplineInterpolator(ss, waypoints)
    vlim = constraint.JointVelocityConstraint(vel_limits)
    alim = constraint.JointAccelerationConstraint(acc_limits)
    instance = ta.algorithm.TOPPRA(
        [vlim, alim], path_spline, parametrizer="ParametrizeConstAccel"
    )
    traj = instance.compute_trajectory()
    if traj is None:
        return _parameterize_quintic(list(waypoints), vel_limits, acc_limits, dt)

    duration = traj.duration
    times = np.arange(0.0, duration + dt / 2.0, dt)
    times = np.minimum(times, duration)
    positions = np.array([traj(t) for t in times])
    velocities = np.array([traj(t, 1) for t in times])
    accelerations = np.array([traj(t, 2) for t in times])

    return times, positions, velocities, accelerations
