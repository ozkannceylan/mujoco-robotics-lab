"""Standing balance controller for the G1 humanoid.

Implements CoM-based PD control with gravity compensation feedforward.
Uses position-servo actuators: ctrl = q_des + qfrc_bias/Kp.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from lab7_common import (
    ACTUATOR_KP,
    ACTUATOR_KD,
    NUM_ACTUATED,
    NQ_FREEFLYER,
    NV_FREEFLYER,
    NV_PIN,
    Q_HOME,
    GRAVITY_MAGNITUDE,
    build_home_q_pin,
    get_state_from_mujoco,
    get_mj_joint_qpos_adr,
    get_mj_joint_dofadr,
)
from g1_model import G1Model


class StandingBalanceController:
    """PD balance controller that maintains upright standing.

    Strategy:
      1. Compute desired CoM acceleration via PD on CoM position error.
      2. Map CoM acceleration to joint accelerations via J_com pseudoinverse.
      3. Compute desired joint positions as q_des = q_home + delta_q.
      4. Send to position servos with gravity feedforward:
         ctrl[i] = q_des[i] + qfrc_bias[joint_adr[i]] / Kp[i]
    """

    def __init__(
        self,
        g1: G1Model,
        kp_com: float = 100.0,
        kd_com: float = 20.0,
        kp_posture: float = 30.0,
        kd_posture: float = 5.0,
    ) -> None:
        """Initialize the standing balance controller.

        Args:
            g1: G1 Pinocchio model wrapper.
            kp_com: Proportional gain for CoM PD.
            kd_com: Derivative gain for CoM PD.
            kp_posture: Proportional gain for posture regulation.
            kd_posture: Derivative gain for posture regulation.
        """
        self.g1 = g1
        self.kp_com = kp_com
        self.kd_com = kd_com
        self.kp_posture = kp_posture
        self.kd_posture = kd_posture

        # Compute CoM target from home configuration
        q_home = build_home_q_pin()
        self.com_target = g1.compute_com(q_home).copy()
        self.q_home_joints = Q_HOME.copy()

        # Cache joint addresses (set on first call)
        self._qpos_addrs = None
        self._dof_addrs = None

    def _ensure_addrs(self, mj_model) -> None:
        """Cache MuJoCo joint addresses on first call.

        Args:
            mj_model: MuJoCo model.
        """
        if self._qpos_addrs is None:
            self._qpos_addrs = get_mj_joint_qpos_adr(mj_model)
            self._dof_addrs = get_mj_joint_dofadr(mj_model)

    def set_com_target(self, target: np.ndarray) -> None:
        """Update the desired CoM position.

        Args:
            target: Desired CoM position [3].
        """
        self.com_target = target.copy()

    def compute_control(
        self,
        mj_model,
        mj_data,
        com_target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute actuator control signals for standing balance.

        Args:
            mj_model: MuJoCo model.
            mj_data: MuJoCo data.
            com_target: Optional override for CoM target [3].

        Returns:
            Control signal array [NUM_ACTUATED].
        """
        self._ensure_addrs(mj_model)

        # Get current state in Pinocchio convention
        q_pin, v_pin = get_state_from_mujoco(mj_model, mj_data)

        # Current CoM and velocity
        com = self.g1.compute_com(q_pin)
        com_vel = self.g1.compute_com_velocity(q_pin, v_pin)

        # Target
        target = com_target if com_target is not None else self.com_target

        # CoM PD control (in x, y only -- z stays constant)
        com_error = target - com
        com_error[2] = 0.0  # don't control height via CoM PD
        com_vel_error = -com_vel.copy()
        com_vel_error[2] = 0.0

        com_acc_des = self.kp_com * com_error + self.kd_com * com_vel_error

        # CoM Jacobian -> joint velocity increment
        J_com = self.g1.compute_jacobian_com(q_pin)
        # Only use actuated part of Jacobian
        J_com_act = J_com[:, NV_FREEFLYER:]

        # Damped pseudoinverse
        lam = 1e-3
        JtJ = J_com_act.T @ J_com_act + lam * np.eye(NUM_ACTUATED)
        dq_com = np.linalg.solve(JtJ, J_com_act.T @ com_acc_des) * 0.01

        # Posture regulation
        q_joints = q_pin[NQ_FREEFLYER:]
        v_joints = v_pin[NV_FREEFLYER:]
        dq_posture = (
            self.kp_posture * (self.q_home_joints - q_joints)
            - self.kd_posture * v_joints
        ) * 0.001

        # Desired joint positions
        q_des = q_joints + dq_com + dq_posture

        # Build control with gravity feedforward
        ctrl = np.zeros(NUM_ACTUATED)
        for i in range(NUM_ACTUATED):
            bias = mj_data.qfrc_bias[self._dof_addrs[i]]
            ctrl[i] = q_des[i] + bias / ACTUATOR_KP[i]

        return ctrl

    def compute_control_simple(
        self,
        mj_model,
        mj_data,
    ) -> np.ndarray:
        """Simplified control: hold home pose with gravity feedforward.

        This is a simpler version that just holds the home pose without
        CoM feedback. Useful as a fallback or for initial standing.

        Args:
            mj_model: MuJoCo model.
            mj_data: MuJoCo data.

        Returns:
            Control signal array [NUM_ACTUATED].
        """
        self._ensure_addrs(mj_model)

        ctrl = np.zeros(NUM_ACTUATED)
        for i in range(NUM_ACTUATED):
            bias = mj_data.qfrc_bias[self._dof_addrs[i]]
            ctrl[i] = self.q_home_joints[i] + bias / ACTUATOR_KP[i]

        return ctrl
