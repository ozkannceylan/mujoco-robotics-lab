"""Phase 3.2 — Control Hierarchy.

Implements and compares four controller levels:
  1. Joint PD + gravity compensation
  2. Computed torque (inverse dynamics) control
  3. Task-space impedance control
  4. Operational Space Control (OSC)

All controllers are tested on trajectory tracking in MuJoCo.

Run:
    python3 lab-2-Ur5e-robotics-lab/src/b2_control_hierarchy.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
import pinocchio as pin

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ur5e_common import (
    NUM_JOINTS,
    Q_HOME,
    load_pinocchio_model,
)
from mujoco_sim import UR5eSimulator
from b1_trajectory_generation import quintic_trajectory, JointTrajectoryPoint


@dataclass
class ControlLog:
    """Log of controller performance over a trajectory."""
    name: str
    times: list[float] = field(default_factory=list)
    pos_errors: list[float] = field(default_factory=list)    # joint error norm
    torques: list[float] = field(default_factory=list)       # torque norm
    ee_errors: list[float] = field(default_factory=list)     # EE position error


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


def pd_gravity_control(
    q: np.ndarray, qd: np.ndarray,
    q_des: np.ndarray, qd_des: np.ndarray,
    pin_model, pin_data,
    Kp: np.ndarray, Kd: np.ndarray,
) -> np.ndarray:
    """Joint PD control with gravity compensation.

    tau = Kp * (q_des - q) + Kd * (qd_des - qd) + g(q)

    Args:
        q: Current joint positions (6,).
        qd: Current joint velocities (6,).
        q_des: Desired joint positions (6,).
        qd_des: Desired joint velocities (6,).
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        Kp: Proportional gains (6,).
        Kd: Derivative gains (6,).

    Returns:
        Joint torques (6,).
    """
    pin.computeGeneralizedGravity(pin_model, pin_data, q)
    g = pin_data.g.copy()

    tau = Kp * (q_des - q) + Kd * (qd_des - qd) + g
    return tau


def computed_torque_control(
    q: np.ndarray, qd: np.ndarray,
    q_des: np.ndarray, qd_des: np.ndarray, qdd_des: np.ndarray,
    pin_model, pin_data,
    Kp: np.ndarray, Kd: np.ndarray,
) -> np.ndarray:
    """Computed torque (inverse dynamics) control.

    tau = M(q) * (qdd_des + Kp*(q_des-q) + Kd*(qd_des-qd)) + C*qd + g

    Args:
        q, qd: Current state.
        q_des, qd_des, qdd_des: Desired trajectory point.
        pin_model, pin_data: Pinocchio model and data.
        Kp, Kd: Gains (6,).

    Returns:
        Joint torques (6,).
    """
    e = q_des - q
    ed = qd_des - qd
    a_cmd = qdd_des + Kp * e + Kd * ed

    tau = pin.rnea(pin_model, pin_data, q, qd, a_cmd)
    return tau


def task_space_impedance_control(
    q: np.ndarray, qd: np.ndarray,
    x_des: np.ndarray, R_des: np.ndarray,
    pin_model, pin_data, ee_fid: int,
    Kp_pos: float = 200.0, Kd_pos: float = 40.0,
    Kp_rot: float = 50.0, Kd_rot: float = 10.0,
) -> np.ndarray:
    """Task-space impedance control.

    tau = J^T * F + g(q)
    F = [Kp_pos * e_pos + Kd_pos * (-J_v * qd);
         Kp_rot * e_rot + Kd_rot * (-J_w * qd)]

    Args:
        q, qd: Current state.
        x_des: Desired EE position (3,).
        R_des: Desired EE rotation (3,3).
        pin_model, pin_data: Pinocchio model and data.
        ee_fid: End-effector frame ID.
        Kp_pos, Kd_pos: Position stiffness/damping.
        Kp_rot, Kd_rot: Orientation stiffness/damping.

    Returns:
        Joint torques (6,).
    """
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    pin.computeJointJacobians(pin_model, pin_data, q)

    oMf = pin_data.oMf[ee_fid]
    x_cur = oMf.translation
    R_cur = oMf.rotation

    # Position error
    e_pos = x_des - x_cur

    # Orientation error (log map)
    R_err = R_des @ R_cur.T
    e_rot = pin.log3(R_err)

    # Jacobian in world-aligned frame
    J = pin.getFrameJacobian(pin_model, pin_data, ee_fid,
                              pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    # Cartesian velocity (approximate)
    xd = J @ qd

    # Wrench
    F = np.zeros(6)
    F[:3] = Kp_pos * e_pos - Kd_pos * xd[:3]
    F[3:] = Kp_rot * e_rot - Kd_rot * xd[3:]

    # Gravity compensation
    pin.computeGeneralizedGravity(pin_model, pin_data, q)
    g = pin_data.g.copy()

    tau = J.T @ F + g
    return tau


def osc_control(
    q: np.ndarray, qd: np.ndarray,
    x_des: np.ndarray, R_des: np.ndarray,
    xdd_des: np.ndarray,
    pin_model, pin_data, ee_fid: int,
    Kp: float = 200.0, Kd: float = 40.0,
) -> np.ndarray:
    """Operational Space Control (Khatib 1987).

    tau = J^T * Lambda * (xdd_des + Kp*e_x + Kd*(-xd)) + J^T*mu + g(q)
    Lambda = (J * M^{-1} * J^T)^{-1}

    Args:
        q, qd: Current state.
        x_des: Desired EE position (3,).
        R_des: Desired EE rotation (3,3).
        xdd_des: Desired Cartesian acceleration (6,).
        pin_model, pin_data: Pinocchio model and data.
        ee_fid: End-effector frame ID.
        Kp, Kd: Gains.

    Returns:
        Joint torques (6,).
    """
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    pin.computeJointJacobians(pin_model, pin_data, q)

    oMf = pin_data.oMf[ee_fid]
    x_cur = oMf.translation
    R_cur = oMf.rotation

    e_pos = x_des - x_cur
    e_rot = pin.log3(R_des @ R_cur.T)
    e = np.concatenate([e_pos, e_rot])

    J = pin.getFrameJacobian(pin_model, pin_data, ee_fid,
                              pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    xd = J @ qd

    # Mass matrix
    pin.crba(pin_model, pin_data, q)
    M = pin_data.M.copy()
    M_inv = np.linalg.inv(M)

    # Task-space inertia
    JMinvJt = J @ M_inv @ J.T
    Lambda = np.linalg.inv(JMinvJt + 1e-6 * np.eye(6))

    # Task-space command
    F = Lambda @ (xdd_des + Kp * e - Kd * xd)

    # Gravity + Coriolis
    v_zero = np.zeros(NUM_JOINTS)
    h = pin.rnea(pin_model, pin_data, q, qd, v_zero)

    tau = J.T @ F + h
    return tau


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------


def run_controller(controller_fn: Callable, name: str,
                   traj: list[JointTrajectoryPoint],
                   sim: UR5eSimulator,
                   pin_model, pin_data, ee_fid: int) -> ControlLog:
    """Run a controller on a trajectory and log performance.

    Args:
        controller_fn: Function that takes (t, q, qd, traj_point) and returns torques.
        name: Controller name for logging.
        traj: Joint trajectory.
        sim: MuJoCo simulator.
        pin_model, pin_data, ee_fid: Pinocchio model.

    Returns:
        ControlLog with performance metrics.
    """
    log = ControlLog(name=name)

    # Reset to start of trajectory
    sim.set_qpos(traj[0].q)
    sim.data.qvel[:NUM_JOINTS] = traj[0].qd
    mujoco.mj_forward(sim.model, sim.data)

    # Desired EE trajectory (for logging)
    pin.forwardKinematics(pin_model, pin_data, traj[0].q)
    pin.updateFramePlacements(pin_model, pin_data)

    traj_idx = 0
    while sim.time < traj[-1].t + 0.01:
        # Find current trajectory point
        while traj_idx < len(traj) - 1 and traj[traj_idx + 1].t <= sim.time:
            traj_idx += 1
        pt = traj[traj_idx]

        q = sim.data.qpos[:NUM_JOINTS].copy()
        qd = sim.data.qvel[:NUM_JOINTS].copy()

        # Compute control
        tau = controller_fn(sim.time, q, qd, pt)
        tau = np.clip(tau, -150, 150)  # torque limits

        sim.set_ctrl(tau)
        sim.step()

        # Log
        state = sim.get_state()
        pos_err = np.linalg.norm(pt.q - q)

        pin.forwardKinematics(pin_model, pin_data, pt.q)
        pin.updateFramePlacements(pin_model, pin_data)
        ee_des = pin_data.oMf[ee_fid].translation.copy()
        ee_err = np.linalg.norm(ee_des - state.ee_pos)

        log.times.append(sim.time)
        log.pos_errors.append(pos_err)
        log.torques.append(np.linalg.norm(tau))
        log.ee_errors.append(ee_err)

    return log


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Compare all four controllers on the same trajectory."""
    print("=" * 72)
    print("Phase 3.2 — Control Hierarchy")
    print("=" * 72)

    pin_model, pin_data, ee_fid = load_pinocchio_model()

    # Generate a test trajectory
    q_start = Q_HOME.copy()
    q_end = np.array([0.0, -np.pi / 2, np.pi / 3, -np.pi / 3, -np.pi / 2, 0.0])
    T = 3.0
    dt = 0.002  # match MuJoCo timestep
    traj = quintic_trajectory(q_start, q_end, T, dt)

    # Gains
    Kp = np.array([2000, 2000, 2000, 500, 500, 500], dtype=float)
    Kd = np.array([400, 400, 400, 100, 100, 100], dtype=float)

    # Pre-compute desired EE poses for task-space controllers
    pin.forwardKinematics(pin_model, pin_data, q_end)
    pin.updateFramePlacements(pin_model, pin_data)
    x_des_end = pin_data.oMf[ee_fid].translation.copy()
    R_des_end = pin_data.oMf[ee_fid].rotation.copy()

    results = []

    # 1. PD + gravity compensation
    print("\n  Running PD + gravity compensation...")
    sim1 = UR5eSimulator()
    log1 = run_controller(
        lambda t, q, qd, pt: pd_gravity_control(q, qd, pt.q, pt.qd,
                                                   pin_model, pin_data, Kp, Kd),
        "PD+g", traj, sim1, pin_model, pin_data, ee_fid
    )
    results.append(log1)

    # 2. Computed torque
    print("  Running computed torque control...")
    sim2 = UR5eSimulator()
    log2 = run_controller(
        lambda t, q, qd, pt: computed_torque_control(q, qd, pt.q, pt.qd, pt.qdd,
                                                       pin_model, pin_data, Kp, Kd),
        "CT", traj, sim2, pin_model, pin_data, ee_fid
    )
    results.append(log2)

    # 3. Task-space impedance
    print("  Running task-space impedance control...")
    sim3 = UR5eSimulator()

    def impedance_ctrl(t, q, qd, pt):
        pin.forwardKinematics(pin_model, pin_data, pt.q)
        pin.updateFramePlacements(pin_model, pin_data)
        x_d = pin_data.oMf[ee_fid].translation.copy()
        R_d = pin_data.oMf[ee_fid].rotation.copy()
        return task_space_impedance_control(q, qd, x_d, R_d,
                                             pin_model, pin_data, ee_fid)

    log3 = run_controller(impedance_ctrl, "Impedance", traj, sim3,
                           pin_model, pin_data, ee_fid)
    results.append(log3)

    # Print results
    print(f"\n  {'Controller':<15} {'RMS joint err':>14} {'RMS EE err':>12} "
          f"{'Mean |τ|':>10} {'Max |τ|':>10}")
    print("  " + "-" * 64)
    for log in results:
        rms_pos = np.sqrt(np.mean(np.array(log.pos_errors)**2))
        rms_ee = np.sqrt(np.mean(np.array(log.ee_errors)**2))
        mean_tau = np.mean(log.torques)
        max_tau = np.max(log.torques)
        print(f"  {log.name:<15} {rms_pos:>12.6f} rad {rms_ee:>10.6f} m "
              f"{mean_tau:>8.2f} Nm {max_tau:>8.2f} Nm")

    print("\n" + "=" * 72)
    print("Phase 3.2 — Control Hierarchy: COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
