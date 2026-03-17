"""Lab 7 — Phase 3: Whole-body inverse kinematics for the G1.

Solves for leg joint angles given desired pelvis position and foot positions
using Pinocchio task-space Jacobian control with damped least squares.

Architecture:
    Pinocchio model with FreeFlyer root.
    Pelvis is treated as a "virtual" fixed frame (held at desired position).
    Left and right legs are solved independently (decoupled IK).
    Waist joints are kept at neutral (zero) as a posture regularisation task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab7_common import (
    V_LEFT_LEG,
    V_RIGHT_LEG,
    V_WAIST,
    Q_STAND_JOINTS,
    NQ,
    NV,
    LEFT_FOOT_FRAME,
    RIGHT_FOOT_FRAME,
    load_g1_pinocchio,
    build_pin_q_standing,
    pelvis_world_to_pin_base,
)


class WholeBodyIK:
    """Jacobian-based whole-body IK for the G1 during walking.

    Fixes the pelvis at a desired world position (upright) and iteratively
    solves for leg joint angles to reach desired foot positions.

    Each leg is solved independently:
        Task: 3-D foot position (3 DOF)
        DOFs: 6 leg joints
        Null space: posture regularisation toward q_rest

    Args:
        pin_model: Pinocchio model (nq=36 with FreeFlyer).
        pin_data:  Associated Pinocchio data object.
        left_foot_frame: Frame name for the left foot.
        right_foot_frame: Frame name for the right foot.
        lam: Damped-LS regularisation factor λ (λ² enters the diagonal).
        step_size: IK integration step size (0 < step_size ≤ 1).
    """

    def __init__(
        self,
        pin_model: pin.Model,
        pin_data: pin.Data,
        left_foot_frame: str = LEFT_FOOT_FRAME,
        right_foot_frame: str = RIGHT_FOOT_FRAME,
        lam: float = 1e-2,
        step_size: float = 0.5,
    ) -> None:
        self.model = pin_model
        self.data = pin_data
        self.lam = lam
        self.step_size = step_size

        self.lf_id = pin_model.getFrameId(left_foot_frame)
        self.rf_id = pin_model.getFrameId(right_foot_frame)

        if self.lf_id >= pin_model.nframes:
            raise ValueError(f"Frame '{left_foot_frame}' not found in Pinocchio model.")
        if self.rf_id >= pin_model.nframes:
            raise ValueError(f"Frame '{right_foot_frame}' not found in Pinocchio model.")

        # Neutral posture for null-space regularisation (leg joints only)
        self._q_rest_left = np.zeros(6)    # leg joints at zero
        self._q_rest_right = np.zeros(6)

    # ------------------------------------------------------------------
    # Main IK solver
    # ------------------------------------------------------------------

    def solve(
        self,
        q_current: np.ndarray,
        pelvis_pos_des: np.ndarray,
        lfoot_pos_des: np.ndarray,
        rfoot_pos_des: np.ndarray,
        max_iter: int = 200,
        tol: float = 3e-4,
    ) -> tuple[np.ndarray, bool]:
        """Solve whole-body IK.

        The pelvis is fixed at ``pelvis_pos_des`` with upright (identity)
        orientation.  Left and right leg joints are updated to minimise
        foot position errors.

        Args:
            q_current: Current full Pinocchio q (nq=36).
            pelvis_pos_des: Desired pelvis position [x, y, z] in world frame.
            lfoot_pos_des: Desired left foot position [x, y, z] in world frame.
            rfoot_pos_des: Desired right foot position [x, y, z] in world frame.
            max_iter: Maximum Gauss–Newton iterations.
            tol: Convergence tolerance (sum of foot position error norms) [m].

        Returns:
            (q_des, converged):
                q_des     — updated full q (nq=36), Pinocchio convention
                converged — True if both feet converged within tol
        """
        q = q_current.copy()

        # Fix pelvis at desired WORLD position.
        # Pinocchio q[0:3] is the FreeFlyer TRANSLATION, NOT the world position.
        # pelvis_world = q[0:3] + [0, 0, 0.793]  →  q[0:3] = pelvis_world - [0, 0, 0.793]
        q[0:3] = pelvis_world_to_pin_base(pelvis_pos_des)
        q[3:7] = [0.0, 0.0, 0.0, 1.0]   # upright (Pinocchio quat: x,y,z,w)

        for iteration in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            lf_cur = self.data.oMf[self.lf_id].translation.copy()
            rf_cur = self.data.oMf[self.rf_id].translation.copy()

            lf_err = lfoot_pos_des - lf_cur
            rf_err = rfoot_pos_des - rf_cur

            err_norm = np.linalg.norm(lf_err) + np.linalg.norm(rf_err)
            if err_norm < tol:
                return q, True

            # Jacobians (6 × nv, LOCAL_WORLD_ALIGNED → world-frame position rows)
            pin.computeJointJacobians(self.model, self.data, q)
            J_lf_full = pin.getFrameJacobian(
                self.model, self.data, self.lf_id, pin.LOCAL_WORLD_ALIGNED
            )[:3, :]   # 3 × nv
            J_rf_full = pin.getFrameJacobian(
                self.model, self.data, self.rf_id, pin.LOCAL_WORLD_ALIGNED
            )[:3, :]   # 3 × nv

            # Extract leg-only columns
            J_lf_leg = J_lf_full[:, V_LEFT_LEG]    # 3 × 6
            J_rf_leg = J_rf_full[:, V_RIGHT_LEG]   # 3 × 6

            # Damped LS + null-space posture regularisation
            dv_left = self._dls_with_posture(
                J_lf_leg, lf_err, q[7:13], self._q_rest_left
            )
            dv_right = self._dls_with_posture(
                J_rf_leg, rf_err, q[13:19], self._q_rest_right
            )

            # Integrate using pin.integrate for correct quaternion handling
            v = np.zeros(NV)
            v[V_LEFT_LEG] = dv_left * self.step_size
            v[V_RIGHT_LEG] = dv_right * self.step_size
            q = pin.integrate(self.model, q, v)

            # Re-fix pelvis (pin.integrate may drift base if v[:6] != 0)
            q[0:3] = pelvis_world_to_pin_base(pelvis_pos_des)
            q[3:7] = [0.0, 0.0, 0.0, 1.0]

            # Clamp joint limits
            q = pin.normalize(self.model, q)

        return q, False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dls_with_posture(
        self,
        J: np.ndarray,
        err: np.ndarray,
        q_cur: np.ndarray,
        q_rest: np.ndarray,
        k_null: float = 0.1,
    ) -> np.ndarray:
        """Damped LS with null-space posture regularisation.

        Primary task: J @ dq ≈ err
        Null-space:   minimise ||q_cur + dq - q_rest||

        Args:
            J: Task Jacobian (m × n).
            err: Task error (m,).
            q_cur: Current joint angles (n,).
            q_rest: Desired resting joint angles (n,).
            k_null: Null-space posture gain.

        Returns:
            Joint velocity (n,).
        """
        m, n = J.shape
        lam2 = self.lam * self.lam

        # Jacobian pseudo-inverse: J† = Jᵀ (J Jᵀ + λ²I)⁻¹
        JJT = J @ J.T + lam2 * np.eye(m)
        J_pinv = J.T @ np.linalg.solve(JJT, np.eye(m))   # n × m

        # Primary task velocity
        v_primary = J_pinv @ err

        # Null-space projector: N = I - J† J
        N = np.eye(n) - J_pinv @ J

        # Null-space posture task
        v_null = k_null * (q_rest - q_cur)

        return v_primary + N @ v_null

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_foot_positions(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return current left and right foot positions for a given q.

        Args:
            q: Full Pinocchio q (nq=36).

        Returns:
            (lfoot_pos, rfoot_pos): Each (3,) world position.
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return (
            self.data.oMf[self.lf_id].translation.copy(),
            self.data.oMf[self.rf_id].translation.copy(),
        )

    def get_com_position(self, q: np.ndarray) -> np.ndarray:
        """Return the CoM world position for a given q.

        Args:
            q: Full Pinocchio q (nq=36).

        Returns:
            (3,) CoM world position.
        """
        return pin.centerOfMass(self.model, self.data, q, False).copy()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Check that IK converges for a known feasible configuration."""
    pin_model, pin_data = load_g1_pinocchio()
    ik = WholeBodyIK(pin_model, pin_data)

    # Start from standing configuration
    q0 = build_pin_q_standing()

    # Get initial foot positions
    lf0, rf0 = ik.get_foot_positions(q0)
    print(f"  Standing left foot:  {lf0}")
    print(f"  Standing right foot: {rf0}")

    # Shift desired feet slightly forward (5 cm)
    lf_des = lf0 + np.array([0.05, 0.0, 0.0])
    rf_des = rf0 + np.array([0.05, 0.0, 0.0])
    pelvis_des = q0[0:3].copy()

    q_sol, conv = ik.solve(q0, pelvis_des, lf_des, rf_des)
    lf_sol, rf_sol = ik.get_foot_positions(q_sol)

    print(f"  IK converged: {conv}")
    print(f"  Left foot error:  {np.linalg.norm(lf_des - lf_sol)*1000:.2f} mm")
    print(f"  Right foot error: {np.linalg.norm(rf_des - rf_sol)*1000:.2f} mm")


if __name__ == "__main__":
    print("Running WholeBodyIK smoke test …")
    _smoke_test()
