"""Lab 6 — Dual-arm kinematics model.

Manages two separate Pinocchio UR5e models with base transforms,
providing FK, Jacobians, gravity computation, IK, and object-centric frame abstraction.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab6_common import (
    LEFT_BASE_SE3,
    RIGHT_BASE_SE3,
    URDF_PATH,
    NUM_JOINTS_PER_ARM,
)


@dataclass
class ObjectFrame:
    """Object-centric frame for bimanual coordination.

    The object pose in world frame, plus the grasp offsets for each arm.
    Each arm's target = object_pose * grasp_offset.

    Attributes:
        pose: Object pose in world frame (SE3).
        grasp_offset_left: Left EE pose relative to object frame (SE3).
        grasp_offset_right: Right EE pose relative to object frame (SE3).
    """

    pose: pin.SE3
    grasp_offset_left: pin.SE3
    grasp_offset_right: pin.SE3

    def get_left_target(self) -> pin.SE3:
        """Get left arm target in world frame."""
        return self.pose * self.grasp_offset_left

    def get_right_target(self) -> pin.SE3:
        """Get right arm target in world frame."""
        return self.pose * self.grasp_offset_right

    @classmethod
    def from_ee_poses(
        cls,
        object_pose: pin.SE3,
        left_ee: pin.SE3,
        right_ee: pin.SE3,
    ) -> "ObjectFrame":
        """Create ObjectFrame from current EE poses and known object pose.

        Computes grasp offsets as: offset = object_pose.inverse() * ee_pose

        Args:
            object_pose: Object pose in world frame.
            left_ee: Left EE pose in world frame.
            right_ee: Right EE pose in world frame.

        Returns:
            ObjectFrame with computed grasp offsets.
        """
        offset_left = object_pose.inverse() * left_ee
        offset_right = object_pose.inverse() * right_ee
        return cls(
            pose=object_pose,
            grasp_offset_left=offset_left,
            grasp_offset_right=offset_right,
        )

    def moved_to(self, new_pose: pin.SE3) -> "ObjectFrame":
        """Return a new ObjectFrame with the object at a new pose, keeping grasp offsets.

        Args:
            new_pose: New object pose in world frame.

        Returns:
            New ObjectFrame with same grasp offsets but updated pose.
        """
        return ObjectFrame(
            pose=new_pose,
            grasp_offset_left=self.grasp_offset_left,
            grasp_offset_right=self.grasp_offset_right,
        )


class DualArmModel:
    """Dual UR5e kinematics using two separate Pinocchio models with base offsets.

    Each arm is a standard 6-DOF UR5e loaded from the same URDF.
    Base SE3 transforms position each arm in the world frame.

    All FK/Jacobian/gravity outputs are in the WORLD frame.

    Args:
        urdf_path: Path to UR5e URDF (shared by both arms).
        base_left: SE3 transform for left arm base in world frame.
        base_right: SE3 transform for right arm base in world frame.
    """

    def __init__(
        self,
        urdf_path: Path | None = None,
        base_left: pin.SE3 | None = None,
        base_right: pin.SE3 | None = None,
    ) -> None:
        path = urdf_path or URDF_PATH
        self.base_left = base_left if base_left is not None else LEFT_BASE_SE3
        self.base_right = base_right if base_right is not None else RIGHT_BASE_SE3

        # Load left arm model
        self.model_left = pin.buildModelFromUrdf(str(path))
        self.model_left.armature[:] = 0.01
        self.data_left = self.model_left.createData()
        self.ee_fid_left = self.model_left.getFrameId("ee_link")

        # Load right arm model
        self.model_right = pin.buildModelFromUrdf(str(path))
        self.model_right.armature[:] = 0.01
        self.data_right = self.model_right.createData()
        self.ee_fid_right = self.model_right.getFrameId("ee_link")

    def fk_left(self, q: np.ndarray) -> pin.SE3:
        """Forward kinematics for left arm in world frame.

        Args:
            q: Left arm joint angles (6,).

        Returns:
            EE pose in world frame (SE3).
        """
        pin.forwardKinematics(self.model_left, self.data_left, q)
        pin.updateFramePlacements(self.model_left, self.data_left)
        local_pose = self.data_left.oMf[self.ee_fid_left]
        return self.base_left * local_pose

    def fk_right(self, q: np.ndarray) -> pin.SE3:
        """Forward kinematics for right arm in world frame.

        Args:
            q: Right arm joint angles (6,).

        Returns:
            EE pose in world frame (SE3).
        """
        pin.forwardKinematics(self.model_right, self.data_right, q)
        pin.updateFramePlacements(self.model_right, self.data_right)
        local_pose = self.data_right.oMf[self.ee_fid_right]
        return self.base_right * local_pose

    def jacobian_left(self, q: np.ndarray) -> np.ndarray:
        """Compute 6x6 Jacobian for left arm in world frame.

        The Jacobian maps joint velocities to EE spatial velocity (linear, angular)
        in the world frame. The base rotation is applied to transform from
        local frame to world frame.

        Args:
            q: Left arm joint angles (6,).

        Returns:
            Jacobian matrix (6, 6) in world frame.
        """
        pin.forwardKinematics(self.model_left, self.data_left, q)
        pin.updateFramePlacements(self.model_left, self.data_left)
        pin.computeJointJacobians(self.model_left, self.data_left, q)
        J_local = pin.getFrameJacobian(
            self.model_left,
            self.data_left,
            self.ee_fid_left,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        # Rotate Jacobian to world frame
        R = self.base_left.rotation
        J_world = np.zeros_like(J_local)
        J_world[:3, :] = R @ J_local[:3, :]  # linear part
        J_world[3:, :] = R @ J_local[3:, :]  # angular part
        return J_world

    def jacobian_right(self, q: np.ndarray) -> np.ndarray:
        """Compute 6x6 Jacobian for right arm in world frame.

        Args:
            q: Right arm joint angles (6,).

        Returns:
            Jacobian matrix (6, 6) in world frame.
        """
        pin.forwardKinematics(self.model_right, self.data_right, q)
        pin.updateFramePlacements(self.model_right, self.data_right)
        pin.computeJointJacobians(self.model_right, self.data_right, q)
        J_local = pin.getFrameJacobian(
            self.model_right,
            self.data_right,
            self.ee_fid_right,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        R = self.base_right.rotation
        J_world = np.zeros_like(J_local)
        J_world[:3, :] = R @ J_local[:3, :]
        J_world[3:, :] = R @ J_local[3:, :]
        return J_world

    def gravity_left(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity torques for left arm.

        Note: gravity direction in the arm's local frame must account for base rotation.
        For the left arm at identity, gravity is [0, 0, -9.81] (standard).
        For the right arm rotated 180° about Z, gravity in local frame is still [0, 0, -9.81]
        since rotation about Z doesn't change gravity direction.

        Args:
            q: Joint angles (6,).

        Returns:
            Gravity torques (6,).
        """
        # Gravity in world frame
        g_world = np.array([0.0, 0.0, -9.81])
        # Transform to arm's local frame
        g_local = self.base_left.rotation.T @ g_world

        # Temporarily set model gravity
        old_gravity = self.model_left.gravity.linear.copy()
        self.model_left.gravity.linear[:] = g_local

        pin.computeGeneralizedGravity(self.model_left, self.data_left, q)
        result = self.data_left.g.copy()

        # Restore
        self.model_left.gravity.linear[:] = old_gravity
        return result

    def gravity_right(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity torques for right arm.

        Args:
            q: Joint angles (6,).

        Returns:
            Gravity torques (6,).
        """
        g_world = np.array([0.0, 0.0, -9.81])
        g_local = self.base_right.rotation.T @ g_world

        old_gravity = self.model_right.gravity.linear.copy()
        self.model_right.gravity.linear[:] = g_local

        pin.computeGeneralizedGravity(self.model_right, self.data_right, q)
        result = self.data_right.g.copy()

        self.model_right.gravity.linear[:] = old_gravity
        return result

    def ik_left(
        self,
        target: pin.SE3,
        q_init: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> np.ndarray | None:
        """Numerical IK for left arm.

        Solves for joint angles that place the left EE at the target pose in world frame.
        Uses damped least squares (Levenberg-Marquardt).

        Args:
            target: Desired EE pose in world frame (SE3).
            q_init: Initial joint guess (6,).
            max_iter: Maximum iterations.
            tol: Convergence tolerance (norm of 6D error).

        Returns:
            Joint angles (6,) or None if IK failed.
        """
        # Convert world target to arm-local target
        local_target = self.base_left.inverse() * target
        return self._solve_ik(
            self.model_left,
            self.data_left,
            self.ee_fid_left,
            local_target,
            q_init,
            max_iter,
            tol,
        )

    def ik_right(
        self,
        target: pin.SE3,
        q_init: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> np.ndarray | None:
        """Numerical IK for right arm.

        Args:
            target: Desired EE pose in world frame (SE3).
            q_init: Initial joint guess (6,).
            max_iter: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Joint angles (6,) or None if IK failed.
        """
        local_target = self.base_right.inverse() * target
        return self._solve_ik(
            self.model_right,
            self.data_right,
            self.ee_fid_right,
            local_target,
            q_init,
            max_iter,
            tol,
        )

    @staticmethod
    def _solve_ik(
        model,
        data,
        ee_fid: int,
        target: pin.SE3,
        q_init: np.ndarray,
        max_iter: int,
        tol: float,
    ) -> np.ndarray | None:
        """Solve IK using damped least squares.

        Args:
            model: Pinocchio model.
            data: Pinocchio data.
            ee_fid: EE frame ID.
            target: Target pose in arm-local frame.
            q_init: Initial joint angles.
            max_iter: Max iterations.
            tol: Convergence tolerance.

        Returns:
            Joint angles or None.
        """
        q = q_init.copy()
        damp = 1e-6
        alpha = 0.5  # step size

        for _ in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            current = data.oMf[ee_fid]
            error_se3 = pin.log6(current.inverse() * target)
            err = error_se3.vector

            if np.linalg.norm(err) < tol:
                return q

            pin.computeJointJacobians(model, data, q)
            J = pin.getFrameJacobian(
                model,
                data,
                ee_fid,
                pin.ReferenceFrame.LOCAL,
            )

            # Damped least squares
            JtJ = J.T @ J + damp * np.eye(NUM_JOINTS_PER_ARM)
            dq = np.linalg.solve(JtJ, J.T @ err)
            q = q + alpha * dq

            # Clamp to joint limits
            q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

        return None

    def relative_ee_pose(self, q_left: np.ndarray, q_right: np.ndarray) -> pin.SE3:
        """Compute relative pose between left and right EE.

        Returns: left_ee.inverse() * right_ee (right EE expressed in left EE frame).

        Args:
            q_left: Left arm joint angles (6,).
            q_right: Right arm joint angles (6,).

        Returns:
            Relative SE3 pose.
        """
        left_ee = self.fk_left(q_left)
        right_ee = self.fk_right(q_right)
        return left_ee.inverse() * right_ee

    def fk_left_pos(self, q: np.ndarray) -> np.ndarray:
        """Get left EE position in world frame (convenience).

        Args:
            q: Joint angles (6,).

        Returns:
            Position (3,).
        """
        return self.fk_left(q).translation.copy()

    def fk_right_pos(self, q: np.ndarray) -> np.ndarray:
        """Get right EE position in world frame (convenience).

        Args:
            q: Joint angles (6,).

        Returns:
            Position (3,).
        """
        return self.fk_right(q).translation.copy()
