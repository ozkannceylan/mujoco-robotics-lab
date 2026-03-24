"""G1 humanoid model wrapper using Pinocchio.

Provides analytical FK, CoM, Jacobians, mass matrix, and whole-body IK
for the G1 humanoid with freeflyer base.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from lab7_common import (
    URDF_PATH,
    NQ_PIN,
    NV_PIN,
    NQ_FREEFLYER,
    NV_FREEFLYER,
    NUM_ACTUATED,
    Q_HOME,
    build_home_q_pin,
    load_pinocchio_model,
)


class G1Model:
    """Pinocchio-based analytical model for the G1 humanoid.

    Wraps Pinocchio model with convenience methods for FK, CoM,
    Jacobians, mass matrix, and whole-body IK.
    """

    def __init__(self) -> None:
        """Initialize G1 model by loading URDF with freeflyer root."""
        import pinocchio as pin

        self.pin = pin
        self.model, self.data = load_pinocchio_model()

        # Cache frame IDs
        self.left_foot_id = self.model.getFrameId("left_foot")
        self.right_foot_id = self.model.getFrameId("right_foot")
        self.torso_id = self.model.getFrameId("torso")

        # Verify frame IDs
        assert self.left_foot_id < self.model.nframes, "left_foot frame not found"
        assert self.right_foot_id < self.model.nframes, "right_foot frame not found"

        # Compute total mass
        self.total_mass = sum(
            self.model.inertias[i].mass for i in range(self.model.njoints)
        )

        # Home configuration
        self.q_home = build_home_q_pin()

    def forward_kinematics(
        self, q: np.ndarray, v: Optional[np.ndarray] = None
    ) -> None:
        """Run forward kinematics and update frame placements.

        Args:
            q: Configuration vector [NQ_PIN].
            v: Velocity vector [NV_PIN], optional.
        """
        if v is not None:
            self.pin.forwardKinematics(self.model, self.data, q, v)
        else:
            self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)

    def get_frame_position(self, q: np.ndarray, frame_id: int) -> np.ndarray:
        """Get position of a frame in world coordinates.

        Args:
            q: Configuration vector [NQ_PIN].
            frame_id: Pinocchio frame ID.

        Returns:
            Position vector [3].
        """
        self.forward_kinematics(q)
        return self.data.oMf[frame_id].translation.copy()

    def get_frame_placement(self, q: np.ndarray, frame_id: int):
        """Get SE3 placement of a frame.

        Args:
            q: Configuration vector [NQ_PIN].
            frame_id: Pinocchio frame ID.

        Returns:
            Pinocchio SE3 placement.
        """
        self.forward_kinematics(q)
        return self.data.oMf[frame_id].copy()

    def get_left_foot_pos(self, q: np.ndarray) -> np.ndarray:
        """Get left foot position in world frame.

        Args:
            q: Configuration vector [NQ_PIN].

        Returns:
            Position [3].
        """
        return self.get_frame_position(q, self.left_foot_id)

    def get_right_foot_pos(self, q: np.ndarray) -> np.ndarray:
        """Get right foot position in world frame.

        Args:
            q: Configuration vector [NQ_PIN].

        Returns:
            Position [3].
        """
        return self.get_frame_position(q, self.right_foot_id)

    def compute_com(self, q: np.ndarray) -> np.ndarray:
        """Compute center of mass position.

        Args:
            q: Configuration vector [NQ_PIN].

        Returns:
            CoM position [3].
        """
        self.pin.centerOfMass(self.model, self.data, q)
        return self.data.com[0].copy()

    def compute_com_velocity(
        self, q: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Compute center of mass velocity.

        Args:
            q: Configuration vector [NQ_PIN].
            v: Velocity vector [NV_PIN].

        Returns:
            CoM velocity [3].
        """
        self.pin.centerOfMass(self.model, self.data, q, v)
        return self.data.vcom[0].copy()

    def compute_jacobian_com(self, q: np.ndarray) -> np.ndarray:
        """Compute CoM Jacobian (3 x nv).

        Args:
            q: Configuration vector [NQ_PIN].

        Returns:
            CoM Jacobian [3, NV_PIN].
        """
        self.pin.jacobianCenterOfMass(self.model, self.data, q)
        return self.data.Jcom.copy()

    def compute_frame_jacobian(
        self, q: np.ndarray, frame_id: int
    ) -> np.ndarray:
        """Compute 6D frame Jacobian in LOCAL_WORLD_ALIGNED frame.

        Args:
            q: Configuration vector [NQ_PIN].
            frame_id: Pinocchio frame ID.

        Returns:
            Jacobian [6, NV_PIN].
        """
        self.pin.computeJointJacobians(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        J = self.pin.getFrameJacobian(
            self.model, self.data, frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return J.copy()

    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute joint-space mass matrix M(q).

        Args:
            q: Configuration vector [NQ_PIN].

        Returns:
            Mass matrix [NV_PIN, NV_PIN].
        """
        self.pin.crba(self.model, self.data, q)
        M = self.data.M.copy()
        # Ensure symmetry (CRBA may only fill upper triangle in some versions)
        M = (M + M.T) / 2.0
        return M

    def compute_gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity torque vector g(q).

        Args:
            q: Configuration vector [NQ_PIN].

        Returns:
            Gravity vector [NV_PIN].
        """
        self.pin.computeGeneralizedGravity(self.model, self.data, q)
        return self.data.g.copy()

    def compute_nonlinear_effects(
        self, q: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Compute nonlinear effects h(q,v) = C(q,v)*v + g(q).

        Args:
            q: Configuration vector [NQ_PIN].
            v: Velocity vector [NV_PIN].

        Returns:
            Nonlinear effects vector [NV_PIN].
        """
        self.pin.nonLinearEffects(self.model, self.data, q, v)
        return self.data.nle.copy()

    def whole_body_ik(
        self,
        q_current: np.ndarray,
        com_des: Optional[np.ndarray] = None,
        left_foot_des: Optional[np.ndarray] = None,
        right_foot_des: Optional[np.ndarray] = None,
        left_foot_rot: Optional[np.ndarray] = None,
        right_foot_rot: Optional[np.ndarray] = None,
        posture_ref: Optional[np.ndarray] = None,
        max_iter: int = 30,
        dt_ik: float = 0.1,
        tol: float = 1e-3,
        w_com: float = 10.0,
        w_foot: float = 50.0,
        w_foot_rot: float = 20.0,
        w_posture: float = 0.1,
        damping: float = 1e-4,
    ) -> np.ndarray:
        """Whole-body IK with task priority: feet > CoM > posture.

        Uses damped least-squares to solve for joint configuration that
        achieves desired CoM, foot positions/orientations, and posture.

        Args:
            q_current: Current configuration [NQ_PIN].
            com_des: Desired CoM position [3], optional.
            left_foot_des: Desired left foot position [3], optional.
            right_foot_des: Desired right foot position [3], optional.
            left_foot_rot: Desired left foot rotation [3x3], optional.
            right_foot_rot: Desired right foot rotation [3x3], optional.
            posture_ref: Reference posture for actuated joints [NUM_ACTUATED], optional.
            max_iter: Maximum iterations.
            dt_ik: Step size for integration.
            tol: Convergence tolerance on task error norm.
            w_com: Weight for CoM task.
            w_foot: Weight for foot position task.
            w_foot_rot: Weight for foot rotation task.
            w_posture: Weight for posture regularization.
            damping: Damping factor for pseudoinverse.

        Returns:
            Solution configuration [NQ_PIN].
        """
        q = q_current.copy()
        if posture_ref is None:
            posture_ref = Q_HOME.copy()

        for _ in range(max_iter):
            self.forward_kinematics(q)
            self.pin.computeJointJacobians(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)

            # Build stacked task
            tasks_J = []
            tasks_e = []

            # Foot position tasks (highest priority)
            if left_foot_des is not None:
                J_lf = self.pin.getFrameJacobian(
                    self.model, self.data, self.left_foot_id,
                    self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                pos_lf = self.data.oMf[self.left_foot_id].translation
                e_lf = left_foot_des - pos_lf
                tasks_J.append(w_foot * J_lf[:3])
                tasks_e.append(w_foot * e_lf)

            if right_foot_des is not None:
                J_rf = self.pin.getFrameJacobian(
                    self.model, self.data, self.right_foot_id,
                    self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                pos_rf = self.data.oMf[self.right_foot_id].translation
                e_rf = right_foot_des - pos_rf
                tasks_J.append(w_foot * J_rf[:3])
                tasks_e.append(w_foot * e_rf)

            # Foot rotation tasks
            if left_foot_rot is not None:
                J_lf_full = self.pin.getFrameJacobian(
                    self.model, self.data, self.left_foot_id,
                    self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                R_lf = self.data.oMf[self.left_foot_id].rotation
                R_err = left_foot_rot @ R_lf.T
                e_rot_lf = self.pin.log3(R_err)
                tasks_J.append(w_foot_rot * J_lf_full[3:])
                tasks_e.append(w_foot_rot * e_rot_lf)

            if right_foot_rot is not None:
                J_rf_full = self.pin.getFrameJacobian(
                    self.model, self.data, self.right_foot_id,
                    self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                R_rf = self.data.oMf[self.right_foot_id].rotation
                R_err = right_foot_rot @ R_rf.T
                e_rot_rf = self.pin.log3(R_err)
                tasks_J.append(w_foot_rot * J_rf_full[3:])
                tasks_e.append(w_foot_rot * e_rot_rf)

            # CoM task
            if com_des is not None:
                self.pin.jacobianCenterOfMass(self.model, self.data, q)
                J_com = self.data.Jcom
                self.pin.centerOfMass(self.model, self.data, q)
                com_cur = self.data.com[0]
                e_com = com_des - com_cur
                tasks_J.append(w_com * J_com)
                tasks_e.append(w_com * e_com)

            # Posture regularization (only actuated joints)
            J_posture = np.zeros((NUM_ACTUATED, NV_PIN))
            for i in range(NUM_ACTUATED):
                J_posture[i, NV_FREEFLYER + i] = 1.0
            e_posture = posture_ref - q[NQ_FREEFLYER:]
            tasks_J.append(w_posture * J_posture)
            tasks_e.append(w_posture * e_posture)

            if len(tasks_J) == 0:
                break

            J_stack = np.vstack(tasks_J)
            e_stack = np.concatenate(tasks_e)

            # Check convergence
            if np.linalg.norm(e_stack) < tol:
                break

            # Damped least-squares
            JtJ = J_stack.T @ J_stack + damping * np.eye(NV_PIN)
            dv = np.linalg.solve(JtJ, J_stack.T @ e_stack)

            # Integrate
            q = self.pin.integrate(self.model, q, dv * dt_ik)

        return q
