"""Baseline standing balance controller using pseudoinverse (no QP).

Provides a simple whole-body balance controller that maintains standing
posture using weighted pseudoinverse task-priority control. This serves
as a baseline before the QP-based controller.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin

from g1_model import G1WholeBodyModel
from lab8_common import (
    Q_STAND,
    NUM_ACTUATED,
    NV,
    NV_FREEFLYER,
    LEFT_FOOT_FRAME,
    RIGHT_FOOT_FRAME,
    TORSO_HEIGHT,
    ACTUATOR_KP,
    ACTUATOR_KV,
    compute_feedforward_ctrl,
    get_actuated_qfrc_bias,
)


class BalanceController:
    """Standing balance controller using task-space pseudoinverse.

    Maintains the robot in a standing posture by tracking:
    1. CoM position (keep above support polygon center)
    2. Foot positions (keep both feet planted)
    3. Posture regularization (default standing pose)

    Uses position actuators with gravity feedforward.

    Attributes:
        wb_model: G1 whole-body model.
        q_ref: Reference posture for actuated joints [na].
        com_target: Desired CoM position [3].
        kp_com: CoM tracking proportional gain.
        kd_com: CoM tracking derivative gain.
        kp_posture: Posture proportional gain.
        kd_posture: Posture derivative gain.
    """

    def __init__(
        self,
        wb_model: G1WholeBodyModel,
        q_ref: Optional[np.ndarray] = None,
        kp_com: float = 50.0,
        kd_com: float = 10.0,
        kp_posture: float = 20.0,
        kd_posture: float = 5.0,
    ) -> None:
        """Initialize balance controller.

        Args:
            wb_model: G1 whole-body model.
            q_ref: Reference joint posture [na]. Defaults to Q_STAND.
            kp_com: CoM proportional gain.
            kd_com: CoM derivative gain.
            kp_posture: Posture proportional gain.
            kd_posture: Posture derivative gain.
        """
        self.wb_model = wb_model
        self.q_ref = q_ref.copy() if q_ref is not None else Q_STAND.copy()
        self.kp_com = kp_com
        self.kd_com = kd_com
        self.kp_posture = kp_posture
        self.kd_posture = kd_posture
        self.com_target: Optional[np.ndarray] = None

        # Foot target poses (set from initial configuration)
        self._left_foot_target: Optional[pin.SE3] = None
        self._right_foot_target: Optional[pin.SE3] = None

    def initialize(self, q: np.ndarray) -> None:
        """Initialize controller targets from current configuration.

        Sets CoM target and foot targets from the initial robot state.

        Args:
            q: Current generalized positions [nq].
        """
        self.com_target = self.wb_model.compute_com(q)
        self._left_foot_target = self.wb_model.get_frame_pose(q, LEFT_FOOT_FRAME)
        self._right_foot_target = self.wb_model.get_frame_pose(q, RIGHT_FOOT_FRAME)

    def compute_desired_qpos(
        self,
        q: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute desired actuated joint positions for position actuators.

        Uses a simple pseudoinverse approach:
        1. Compute CoM error and project to joint space
        2. Add posture regularization
        3. Return q_des for the position actuators

        Args:
            q: Current generalized positions [nq].
            v: Current generalized velocities [nv].

        Returns:
            Desired actuated joint positions [na].
        """
        na = NUM_ACTUATED
        q_act = q[7:7 + na]
        v_act = v[NV_FREEFLYER:NV_FREEFLYER + na]

        # Start with posture reference
        dq = np.zeros(na)

        # CoM tracking
        if self.com_target is not None:
            com = self.wb_model.compute_com(q)
            com_vel = self.wb_model.compute_com_velocity(q, v)
            Jcom = self.wb_model.compute_com_jacobian(q)
            # Only use actuated part of Jacobian
            Jcom_act = Jcom[:, NV_FREEFLYER:]

            com_error = com - self.com_target
            com_acc_des = -self.kp_com * com_error - self.kd_com * com_vel

            # Pseudoinverse for CoM
            Jcom_pinv = np.linalg.pinv(Jcom_act, rcond=1e-4)
            dq_com = Jcom_pinv @ com_acc_des * 0.01  # scale for position control
            dq += dq_com

        # Posture regularization
        posture_error = q_act - self.q_ref
        dq += -self.kp_posture * posture_error * 0.002  # scale for DT
        dq += -self.kd_posture * v_act * 0.002

        # Desired position = current + correction
        q_des = q_act + dq

        return q_des

    def compute_ctrl(
        self,
        mj_model,
        mj_data,
        q: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute control signal for MuJoCo position actuators.

        Applies gravity feedforward to the desired joint positions.

        Args:
            mj_model: MuJoCo model.
            mj_data: MuJoCo data.
            q: Current generalized positions (Pinocchio convention) [nq].
            v: Current generalized velocities [nv].

        Returns:
            Control signal [na] for MuJoCo actuators.
        """
        q_des = self.compute_desired_qpos(q, v)
        qfrc_bias = get_actuated_qfrc_bias(mj_data)
        qd_des = np.zeros(NUM_ACTUATED)
        ctrl = compute_feedforward_ctrl(q_des, qd_des, qfrc_bias)
        return ctrl


class SimpleStandingController:
    """Even simpler standing controller: just hold Q_STAND with feedforward.

    Useful as a baseline to verify the model stands without falling.

    Attributes:
        q_des: Desired joint positions [na].
    """

    def __init__(self, q_des: Optional[np.ndarray] = None) -> None:
        """Initialize simple standing controller.

        Args:
            q_des: Desired standing posture [na]. Defaults to Q_STAND.
        """
        self.q_des = q_des.copy() if q_des is not None else Q_STAND.copy()

    def compute_ctrl(self, mj_data) -> np.ndarray:
        """Compute control signal with gravity feedforward.

        Args:
            mj_data: MuJoCo data (after mj_forward).

        Returns:
            Control signal [na].
        """
        qfrc_bias = get_actuated_qfrc_bias(mj_data)
        qd_des = np.zeros(NUM_ACTUATED)
        return compute_feedforward_ctrl(self.q_des, qd_des, qfrc_bias)
