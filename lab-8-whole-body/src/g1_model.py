"""G1 Whole-Body Model wrapping Pinocchio for analytical computations.

Provides FK, Jacobians, CoM, dynamics (CRBA, RNEA) for the G1 humanoid
with a freeflyer base. Uses Pinocchio as the analytical engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin

from lab8_common import (
    URDF_PATH,
    NUM_ACTUATED,
    NV,
    NV_FREEFLYER,
    LEFT_FOOT_FRAME,
    RIGHT_FOOT_FRAME,
    LEFT_HAND_FRAME,
    RIGHT_HAND_FRAME,
)


class G1WholeBodyModel:
    """Whole-body analytical model for the G1 humanoid using Pinocchio.

    Attributes:
        model: Pinocchio model with freeflyer root.
        data: Pinocchio data.
        nq: Number of generalized coordinates (32).
        nv: Number of generalized velocities (31).
        na: Number of actuated joints (25).
    """

    def __init__(self, urdf_path: Path | None = None) -> None:
        """Initialize the G1 model from URDF.

        Args:
            urdf_path: Path to URDF. Defaults to URDF_PATH.
        """
        path = urdf_path or URDF_PATH
        self.model = pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())
        self.model.armature[:] = 0.01  # match MuJoCo armature

        self.data = self.model.createData()

        self.nq: int = self.model.nq
        self.nv: int = self.model.nv
        self.na: int = NUM_ACTUATED

        # Cache frame IDs
        self._frame_ids: dict[str, int] = {}
        self._init_frame_ids()

        # Selection matrix: maps actuated torques to full generalized forces
        # S is (na x nv): tau_full = S^T @ tau_actuated
        self._S = np.zeros((self.na, self.nv))
        self._S[:, NV_FREEFLYER:] = np.eye(self.na)

    def _init_frame_ids(self) -> None:
        """Cache commonly used frame IDs."""
        for name in [LEFT_FOOT_FRAME, RIGHT_FOOT_FRAME,
                     LEFT_HAND_FRAME, RIGHT_HAND_FRAME]:
            fid = self.model.getFrameId(name)
            self._frame_ids[name] = fid

    @property
    def S(self) -> np.ndarray:
        """Selection matrix S: (na x nv). tau_gen = S^T @ tau_act."""
        return self._S

    def get_frame_id(self, frame_name: str) -> int:
        """Get Pinocchio frame ID by name (cached).

        Args:
            frame_name: Name of the frame.

        Returns:
            Frame ID.
        """
        if frame_name not in self._frame_ids:
            self._frame_ids[frame_name] = self.model.getFrameId(frame_name)
        return self._frame_ids[frame_name]

    # -----------------------------------------------------------------
    # Forward Kinematics
    # -----------------------------------------------------------------

    def forward_kinematics(self, q: np.ndarray, v: Optional[np.ndarray] = None) -> None:
        """Run forward kinematics (and optionally velocity kinematics).

        Args:
            q: Generalized positions [nq].
            v: Generalized velocities [nv]. If provided, also computes velocities.
        """
        if v is not None:
            pin.forwardKinematics(self.model, self.data, q, v)
        else:
            pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def get_frame_pose(self, q: np.ndarray, frame_name: str) -> pin.SE3:
        """Compute the SE3 pose of a frame.

        Args:
            q: Generalized positions [nq].
            frame_name: Name of the frame.

        Returns:
            SE3 pose of the frame in world coordinates.
        """
        self.forward_kinematics(q)
        fid = self.get_frame_id(frame_name)
        return self.data.oMf[fid].copy()

    def get_frame_position(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        """Compute the 3D position of a frame.

        Args:
            q: Generalized positions [nq].
            frame_name: Name of the frame.

        Returns:
            Position [3].
        """
        pose = self.get_frame_pose(q, frame_name)
        return pose.translation.copy()

    # -----------------------------------------------------------------
    # Jacobians
    # -----------------------------------------------------------------

    def get_frame_jacobian(
        self,
        q: np.ndarray,
        frame_name: str,
        ref_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    ) -> np.ndarray:
        """Compute the 6xnv Jacobian of a frame.

        Args:
            q: Generalized positions [nq].
            frame_name: Name of the frame.
            ref_frame: Reference frame for the Jacobian. Default LOCAL_WORLD_ALIGNED.

        Returns:
            Jacobian [6, nv].
        """
        self.forward_kinematics(q)
        pin.computeJointJacobians(self.model, self.data, q)
        fid = self.get_frame_id(frame_name)
        return pin.getFrameJacobian(self.model, self.data, fid, ref_frame).copy()

    # -----------------------------------------------------------------
    # Center of Mass
    # -----------------------------------------------------------------

    def compute_com(self, q: np.ndarray, v: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute center of mass position.

        Args:
            q: Generalized positions [nq].
            v: Generalized velocities [nv]. If provided, also computes CoM velocity.

        Returns:
            CoM position [3].
        """
        if v is not None:
            pin.centerOfMass(self.model, self.data, q, v)
        else:
            pin.centerOfMass(self.model, self.data, q)
        return self.data.com[0].copy()

    def compute_com_velocity(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute center of mass velocity.

        Args:
            q: Generalized positions [nq].
            v: Generalized velocities [nv].

        Returns:
            CoM velocity [3].
        """
        pin.centerOfMass(self.model, self.data, q, v)
        return self.data.vcom[0].copy()

    def compute_com_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute the CoM Jacobian (3 x nv).

        Args:
            q: Generalized positions [nq].

        Returns:
            CoM Jacobian [3, nv].
        """
        pin.jacobianCenterOfMass(self.model, self.data, q)
        return self.data.Jcom.copy()

    def compute_com_with_load(
        self,
        q: np.ndarray,
        load_mass: float,
        load_pos_world: np.ndarray,
    ) -> np.ndarray:
        """Compute effective CoM including a carried load.

        Args:
            q: Generalized positions [nq].
            load_mass: Mass of the carried object [kg].
            load_pos_world: Position of the load in world frame [3].

        Returns:
            Effective CoM position [3].
        """
        robot_com = self.compute_com(q)
        robot_mass = self.total_mass()
        effective_com = (robot_mass * robot_com + load_mass * load_pos_world) / (
            robot_mass + load_mass
        )
        return effective_com

    def total_mass(self) -> float:
        """Return total robot mass.

        Returns:
            Total mass [kg].
        """
        return pin.computeTotalMass(self.model)

    # -----------------------------------------------------------------
    # Dynamics
    # -----------------------------------------------------------------

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute the mass matrix M(q) via CRBA.

        Args:
            q: Generalized positions [nq].

        Returns:
            Mass matrix [nv, nv] (symmetric).
        """
        M = pin.crba(self.model, self.data, q)
        # CRBA only fills upper triangle; symmetrize
        M_full = np.triu(M) + np.triu(M, 1).T
        return M_full

    def nonlinear_effects(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute nonlinear effects h(q, v) = C(q,v)*v + g(q) via RNEA.

        Args:
            q: Generalized positions [nq].
            v: Generalized velocities [nv].

        Returns:
            Nonlinear effects vector [nv].
        """
        return pin.nonLinearEffects(self.model, self.data, q, v).copy()

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity vector g(q).

        Args:
            q: Generalized positions [nq].

        Returns:
            Gravity vector [nv].
        """
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def neutral_configuration(self) -> np.ndarray:
        """Return neutral (zero) configuration with proper freeflyer quaternion.

        Returns:
            Neutral configuration [nq].
        """
        return pin.neutral(self.model).copy()

    def random_configuration(self) -> np.ndarray:
        """Return a random valid configuration.

        Returns:
            Random configuration [nq].
        """
        return pin.randomConfiguration(self.model).copy()
