"""Task hierarchy for whole-body control.

Defines task-space objectives (CoM, foot pose, hand pose, posture) that
produce error vectors and Jacobians for the QP/WLS controller.

Each task computes:
  - error e (dimension depends on task)
  - Jacobian J (dim x nv)
  - desired acceleration a_d = -kp * e - kd * de/dt
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pinocchio as pin

from g1_model import G1WholeBodyModel
from lab8_common import NUM_ACTUATED, NV_FREEFLYER


class Task(ABC):
    """Abstract base class for whole-body control tasks.

    Attributes:
        name: Human-readable task name.
        dim: Dimension of the task error vector.
        weight: Scalar weight for this task in the cost function.
        kp: Proportional gain.
        kd: Derivative gain.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        weight: float,
        kp: float,
        kd: float,
    ) -> None:
        """Initialize task.

        Args:
            name: Task name.
            dim: Error vector dimension.
            weight: Task weight in cost.
            kp: Proportional gain.
            kd: Derivative gain.
        """
        self.name = name
        self.dim = dim
        self.weight = weight
        self.kp = kp
        self.kd = kd

    @abstractmethod
    def compute_error(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute task error vector.

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].
            v: Generalized velocities [nv].

        Returns:
            Error vector [dim].
        """
        ...

    @abstractmethod
    def compute_jacobian(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
    ) -> np.ndarray:
        """Compute task Jacobian.

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].

        Returns:
            Jacobian [dim, nv].
        """
        ...

    def compute_error_derivative(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute task error derivative (de/dt = J * v).

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].
            v: Generalized velocities [nv].

        Returns:
            Error derivative [dim].
        """
        J = self.compute_jacobian(model, q)
        return J @ v

    def compute_desired_acceleration(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute desired task-space acceleration via PD law.

        a_d = -kp * e - kd * de/dt

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].
            v: Generalized velocities [nv].

        Returns:
            Desired acceleration [dim].
        """
        e = self.compute_error(model, q, v)
        de = self.compute_error_derivative(model, q, v)
        return -self.kp * e - self.kd * de


class CoMTask(Task):
    """Track a desired center of mass position (3D).

    Attributes:
        target_com: Desired CoM position [3].
    """

    def __init__(
        self,
        target_com: np.ndarray,
        weight: float = 1000.0,
        kp: float = 100.0,
        kd: float = 20.0,
    ) -> None:
        """Initialize CoM task.

        Args:
            target_com: Desired CoM position [3].
            weight: Task weight.
            kp: Proportional gain.
            kd: Derivative gain.
        """
        super().__init__("com", 3, weight, kp, kd)
        self.target_com = target_com.copy()

    def set_target(self, target_com: np.ndarray) -> None:
        """Update the target CoM position.

        Args:
            target_com: New desired CoM [3].
        """
        self.target_com = target_com.copy()

    def compute_error(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute CoM position error.

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].
            v: Generalized velocities [nv].

        Returns:
            Position error [3].
        """
        com = model.compute_com(q)
        return com - self.target_com

    def compute_jacobian(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
    ) -> np.ndarray:
        """Compute CoM Jacobian (3 x nv).

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].

        Returns:
            CoM Jacobian [3, nv].
        """
        return model.compute_com_jacobian(q)


class FootPoseTask(Task):
    """Track a desired foot SE3 pose (6D: position + orientation).

    Uses the Pinocchio SE3 logarithmic map for orientation error.

    Attributes:
        frame_name: Name of the foot frame.
        target_pose: Desired SE3 pose.
    """

    def __init__(
        self,
        frame_name: str,
        target_pose: pin.SE3,
        weight: float = 100.0,
        kp: float = 200.0,
        kd: float = 30.0,
    ) -> None:
        """Initialize foot pose task.

        Args:
            frame_name: Foot frame name.
            target_pose: Desired SE3 pose.
            weight: Task weight.
            kp: Proportional gain.
            kd: Derivative gain.
        """
        super().__init__(f"foot_{frame_name}", 6, weight, kp, kd)
        self.frame_name = frame_name
        self.target_pose = target_pose.copy()

    def set_target(self, target_pose: pin.SE3) -> None:
        """Update target pose.

        Args:
            target_pose: New desired SE3 pose.
        """
        self.target_pose = target_pose.copy()

    def compute_error(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute 6D pose error using SE3 log.

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].
            v: Generalized velocities [nv].

        Returns:
            6D error [linear(3), angular(3)].
        """
        current_pose = model.get_frame_pose(q, self.frame_name)
        # SE3 error in world-aligned frame
        error_se3 = pin.log6(self.target_pose.actInv(current_pose))
        return error_se3.vector.copy()

    def compute_jacobian(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
    ) -> np.ndarray:
        """Compute 6xnv frame Jacobian.

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].

        Returns:
            Jacobian [6, nv].
        """
        return model.get_frame_jacobian(q, self.frame_name)


class HandPoseTask(Task):
    """Track a desired hand SE3 pose (6D).

    Same structure as FootPoseTask but with different default gains.

    Attributes:
        frame_name: Name of the hand frame.
        target_pose: Desired SE3 pose.
    """

    def __init__(
        self,
        frame_name: str,
        target_pose: pin.SE3,
        weight: float = 10.0,
        kp: float = 100.0,
        kd: float = 20.0,
    ) -> None:
        """Initialize hand pose task.

        Args:
            frame_name: Hand frame name.
            target_pose: Desired SE3 pose.
            weight: Task weight.
            kp: Proportional gain.
            kd: Derivative gain.
        """
        super().__init__(f"hand_{frame_name}", 6, weight, kp, kd)
        self.frame_name = frame_name
        self.target_pose = target_pose.copy()

    def set_target(self, target_pose: pin.SE3) -> None:
        """Update target pose.

        Args:
            target_pose: New desired SE3 pose.
        """
        self.target_pose = target_pose.copy()

    def compute_error(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute 6D pose error using SE3 log.

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].
            v: Generalized velocities [nv].

        Returns:
            6D error [linear(3), angular(3)].
        """
        current_pose = model.get_frame_pose(q, self.frame_name)
        error_se3 = pin.log6(self.target_pose.actInv(current_pose))
        return error_se3.vector.copy()

    def compute_jacobian(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
    ) -> np.ndarray:
        """Compute 6xnv frame Jacobian.

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].

        Returns:
            Jacobian [6, nv].
        """
        return model.get_frame_jacobian(q, self.frame_name)


class PostureTask(Task):
    """Regularize actuated joint positions toward a reference posture.

    Only penalizes actuated joints (na-dimensional), not freeflyer.

    Attributes:
        target_posture: Reference joint angles for actuated joints [na].
    """

    def __init__(
        self,
        target_posture: np.ndarray,
        weight: float = 1.0,
        kp: float = 10.0,
        kd: float = 3.0,
    ) -> None:
        """Initialize posture task.

        Args:
            target_posture: Desired actuated joint positions [na].
            weight: Task weight.
            kp: Proportional gain.
            kd: Derivative gain.
        """
        super().__init__("posture", NUM_ACTUATED, weight, kp, kd)
        self.target_posture = target_posture.copy()

    def set_target(self, target_posture: np.ndarray) -> None:
        """Update reference posture.

        Args:
            target_posture: New desired joint positions [na].
        """
        self.target_posture = target_posture.copy()

    def compute_error(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute actuated joint position error.

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].
            v: Generalized velocities [nv].

        Returns:
            Joint position error [na].
        """
        q_act = q[7:]  # skip freeflyer pos(3) + quat(4)
        return q_act - self.target_posture

    def compute_jacobian(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
    ) -> np.ndarray:
        """Compute posture Jacobian: selects actuated joints only.

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].

        Returns:
            Selection Jacobian [na, nv].
        """
        # J = [0_{na x 6} | I_{na x na}]
        return model.S.copy()

    def compute_error_derivative(
        self,
        model: G1WholeBodyModel,
        q: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute actuated joint velocity (error derivative).

        Args:
            model: G1 whole-body model.
            q: Generalized positions [nq].
            v: Generalized velocities [nv].

        Returns:
            Joint velocity error [na].
        """
        return v[NV_FREEFLYER:]
