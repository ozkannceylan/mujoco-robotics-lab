"""Lab 4 — Execute timed trajectories on the torque-controlled UR5e.

Uses joint-space impedance control: tau = Kp*(q_d - q) + Kd*(qd_d - qd) + g(q)
where g(q) is the gravity compensation term from Pinocchio RNEA.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

from lab4_common import (
    DT,
    NUM_JOINTS,
    SCENE_PATH,
    clip_torques,
    load_mujoco_model,
    load_pinocchio_model,
)


def execute_trajectory(
    times: np.ndarray,
    q_traj: np.ndarray,
    qd_traj: np.ndarray,
    scene_path: Path | None = None,
    Kp: float = 400.0,
    Kd: float = 40.0,
) -> dict[str, np.ndarray]:
    """Execute a timed trajectory on the torque-controlled UR5e.

    Uses joint-space impedance control with gravity compensation.
    Interpolates the trajectory at the simulation timestep (1 kHz).

    Args:
        times: Time array (N,) in seconds.
        q_traj: Joint position trajectory (N, 6).
        qd_traj: Joint velocity trajectory (N, 6).
        scene_path: Path to MJCF scene. Defaults to SCENE_PATH.
        Kp: Proportional gain.
        Kd: Derivative gain.

    Returns:
        Dict with keys: time, q_actual, q_desired, tau, ee_pos.
        Each value is a numpy array.
    """
    # Load models
    mj_model, mj_data = load_mujoco_model(scene_path)
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    # Initialize MuJoCo at trajectory start
    mj_data.qpos[:NUM_JOINTS] = q_traj[0]
    mj_data.qvel[:NUM_JOINTS] = qd_traj[0]
    mujoco.mj_forward(mj_model, mj_data)

    # Get tool0 body ID for EE tracking
    tool_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tool0")

    # Simulation duration: trajectory time + 0.5s settling
    traj_duration = times[-1]
    total_duration = traj_duration + 0.5
    n_steps = int(total_duration / DT) + 1

    # Storage
    log_time = np.zeros(n_steps)
    log_q_actual = np.zeros((n_steps, NUM_JOINTS))
    log_q_desired = np.zeros((n_steps, NUM_JOINTS))
    log_tau = np.zeros((n_steps, NUM_JOINTS))
    log_ee_pos = np.zeros((n_steps, 3))

    for step in range(n_steps):
        t = step * DT

        # Interpolate desired trajectory
        if t <= traj_duration:
            q_d = np.interp(t, times, q_traj.T).T if q_traj.ndim == 1 else np.array(
                [np.interp(t, times, q_traj[:, j]) for j in range(NUM_JOINTS)]
            )
            qd_d = np.array(
                [np.interp(t, times, qd_traj[:, j]) for j in range(NUM_JOINTS)]
            )
        else:
            # Hold final position during settling
            q_d = q_traj[-1]
            qd_d = np.zeros(NUM_JOINTS)

        # Current state from MuJoCo
        q = mj_data.qpos[:NUM_JOINTS].copy()
        qd = mj_data.qvel[:NUM_JOINTS].copy()

        # Gravity compensation via Pinocchio RNEA
        pin.computeGeneralizedGravity(pin_model, pin_data, q)
        g = pin_data.g.copy()

        # Joint-space impedance control
        tau = Kp * (q_d - q) + Kd * (qd_d - qd) + g
        tau = clip_torques(tau)

        # Apply torques
        mj_data.ctrl[:NUM_JOINTS] = tau

        # Log
        log_time[step] = t
        log_q_actual[step] = q
        log_q_desired[step] = q_d
        log_tau[step] = tau
        log_ee_pos[step] = mj_data.xpos[tool_bid].copy()

        # Step simulation
        mujoco.mj_step(mj_model, mj_data)

    return {
        "time": log_time,
        "q_actual": log_q_actual,
        "q_desired": log_q_desired,
        "tau": log_tau,
        "ee_pos": log_ee_pos,
    }
