"""Lab 6 — Dual impedance controller with internal force coupling.

Provides:
- Independent impedance control for each arm
- Symmetric gains for balanced internal forces
- Internal squeeze force along grasp axis when grasping
- Gravity compensation for both arms
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pinocchio as pin

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab6_common import TORQUE_LIMITS, clip_torques
from dual_arm_model import DualArmModel


def orientation_error(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Compute orientation error between desired and current rotation matrices.

    Uses the skew-symmetric extraction method:
      e_R = 0.5 * vee(R_d^T R - R^T R_d)

    Args:
        R_des: Desired rotation matrix [3x3].
        R_cur: Current rotation matrix [3x3].

    Returns:
        Orientation error vector (3,).
    """
    R_err = R_des.T @ R_cur - R_cur.T @ R_des
    e = 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])
    return e


@dataclass
class DualImpedanceGains:
    """Impedance gains for dual-arm control (symmetric for both arms).

    Attributes:
        K_p: Position stiffness (6x6) - translation (N/m) + rotation (Nm/rad).
        K_d: Damping (6x6).
        f_squeeze: Internal squeeze force along grasp axis (N).
    """

    K_p: np.ndarray = field(
        default_factory=lambda: np.diag([400.0, 400.0, 400.0, 40.0, 40.0, 40.0])
    )
    K_d: np.ndarray = field(
        default_factory=lambda: np.diag([60.0, 60.0, 60.0, 6.0, 6.0, 6.0])
    )
    f_squeeze: float = 10.0  # N, internal squeeze force


class DualImpedanceController:
    """Dual-arm impedance controller with internal force coupling.

    Each arm runs impedance control: tau = J^T * F + g(q)
    where F = K_p * error + K_d * vel_error

    When grasping, adds an internal force along the grasp axis
    (line connecting the two EEs) to maintain grip.

    Uses symmetric gains so internal forces naturally balance.

    Args:
        dual_model: DualArmModel for FK, Jacobians, gravity.
        gains: Impedance gains (shared by both arms).
    """

    def __init__(
        self,
        dual_model: DualArmModel,
        gains: DualImpedanceGains | None = None,
    ) -> None:
        self.model = dual_model
        self.gains = gains or DualImpedanceGains()

    def compute_dual_torques(
        self,
        q_left: np.ndarray,
        qd_left: np.ndarray,
        q_right: np.ndarray,
        qd_right: np.ndarray,
        target_left: pin.SE3,
        target_right: pin.SE3,
        xd_left: np.ndarray | None = None,
        xd_right: np.ndarray | None = None,
        grasping: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute joint torques for both arms.

        Args:
            q_left: Left arm joint angles (6,).
            qd_left: Left arm joint velocities (6,).
            q_right: Right arm joint angles (6,).
            qd_right: Right arm joint velocities (6,).
            target_left: Left EE target pose in world frame.
            target_right: Right EE target pose in world frame.
            xd_left: Desired left EE velocity (6,) or None.
            xd_right: Desired right EE velocity (6,) or None.
            grasping: If True, add internal squeeze force.

        Returns:
            Tuple of (tau_left, tau_right), each (6,).
        """
        # Compute torques for each arm
        tau_left = self._compute_arm_torque(
            q_left, qd_left, target_left, xd_left, "left"
        )
        tau_right = self._compute_arm_torque(
            q_right, qd_right, target_right, xd_right, "right"
        )

        # Add internal squeeze force when grasping
        if grasping and self.gains.f_squeeze > 0:
            tau_sq_left, tau_sq_right = self._compute_squeeze_torques(
                q_left, q_right
            )
            tau_left += tau_sq_left
            tau_right += tau_sq_right

        return clip_torques(tau_left), clip_torques(tau_right)

    def _compute_arm_torque(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        target: pin.SE3,
        xd_des: np.ndarray | None,
        side: str,
    ) -> np.ndarray:
        """Compute impedance torque for a single arm.

        Args:
            q: Joint angles (6,).
            qd: Joint velocities (6,).
            target: Desired EE pose in world frame.
            xd_des: Desired EE velocity (6,) or None.
            side: "left" or "right".

        Returns:
            Joint torques (6,).
        """
        if side == "left":
            ee_pose = self.model.fk_left(q)
            J = self.model.jacobian_left(q)
            g = self.model.gravity_left(q)
        else:
            ee_pose = self.model.fk_right(q)
            J = self.model.jacobian_right(q)
            g = self.model.gravity_right(q)

        # 6D error: position + orientation
        pos_err = target.translation - ee_pose.translation
        ori_err = orientation_error(target.rotation, ee_pose.rotation)
        error = np.concatenate([pos_err, ori_err])

        # Velocity error
        v_cur = J @ qd
        if xd_des is None:
            xd_des = np.zeros(6)
        vel_err = xd_des - v_cur

        # Impedance wrench
        F = self.gains.K_p @ error + self.gains.K_d @ vel_err

        # Joint torque
        tau = J.T @ F + g
        return tau

    def _compute_squeeze_torques(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute internal squeeze force torques.

        Applies force along the line connecting the two EEs,
        pushing each EE toward the other (squeeze).

        Args:
            q_left: Left arm joint angles (6,).
            q_right: Right arm joint angles (6,).

        Returns:
            Tuple of (tau_squeeze_left, tau_squeeze_right).
        """
        # Get EE positions
        pos_left = self.model.fk_left_pos(q_left)
        pos_right = self.model.fk_right_pos(q_right)

        # Grasp axis: direction from left EE to right EE
        diff = pos_right - pos_left
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.zeros(6), np.zeros(6)
        grasp_axis = diff / dist

        # Squeeze force: push each EE toward the other
        F_squeeze_left = np.zeros(6)
        F_squeeze_left[:3] = self.gains.f_squeeze * grasp_axis  # left pushes toward right

        F_squeeze_right = np.zeros(6)
        F_squeeze_right[:3] = -self.gains.f_squeeze * grasp_axis  # right pushes toward left

        # Map to joint torques via Jacobian transpose
        J_left = self.model.jacobian_left(q_left)
        J_right = self.model.jacobian_right(q_right)

        tau_left = J_left.T @ F_squeeze_left
        tau_right = J_right.T @ F_squeeze_right

        return tau_left, tau_right

    def set_gains(self, gains: DualImpedanceGains) -> None:
        """Update impedance gains.

        Args:
            gains: New gains.
        """
        self.gains = gains
