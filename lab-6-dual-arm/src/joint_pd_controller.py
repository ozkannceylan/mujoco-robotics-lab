"""Joint-space PD controller with MuJoCo gravity compensation for dual UR5e arms."""

from __future__ import annotations

import numpy as np

from lab6_common import (
    LEFT_CTRL_SLICE,
    LEFT_JOINT_SLICE,
    NUM_JOINTS_PER_ARM,
    RIGHT_CTRL_SLICE,
    RIGHT_JOINT_SLICE,
    TORQUE_LIMITS,
)


class DualArmJointPD:
    """PD controller for two independent UR5e arms.

    Control law per arm:
        tau = Kp * (q_target - q) + Kd * (0 - qvel) + qfrc_bias

    qfrc_bias from MuJoCo includes gravity + Coriolis compensation.

    Args:
        kp: Proportional gain (scalar or array of 6).
        kd: Derivative gain (scalar or array of 6).
    """

    def __init__(self, kp: float | np.ndarray = 100.0, kd: float | np.ndarray = 10.0) -> None:
        self.kp = np.broadcast_to(np.asarray(kp, dtype=np.float64), NUM_JOINTS_PER_ARM).copy()
        self.kd = np.broadcast_to(np.asarray(kd, dtype=np.float64), NUM_JOINTS_PER_ARM).copy()
        self._saturated = False

    @property
    def saturated(self) -> bool:
        """True if any torque was clipped on the last call."""
        return self._saturated

    def compute(
        self,
        mj_data,
        q_target_left: np.ndarray,
        q_target_right: np.ndarray,
    ) -> None:
        """Compute and apply PD torques to both arms.

        Args:
            mj_data: MuJoCo data object (modified in-place via ctrl).
            q_target_left: Target joint angles for left arm (6,).
            q_target_right: Target joint angles for right arm (6,).
        """
        self._saturated = False

        for q_target, jslice, cslice in [
            (q_target_left, LEFT_JOINT_SLICE, LEFT_CTRL_SLICE),
            (q_target_right, RIGHT_JOINT_SLICE, RIGHT_CTRL_SLICE),
        ]:
            q = mj_data.qpos[jslice]
            qd = mj_data.qvel[jslice]
            bias = mj_data.qfrc_bias[jslice]

            tau = self.kp * (q_target - q) + self.kd * (0.0 - qd) + bias
            tau_clipped = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

            if not np.allclose(tau, tau_clipped):
                self._saturated = True

            mj_data.ctrl[cslice] = tau_clipped
