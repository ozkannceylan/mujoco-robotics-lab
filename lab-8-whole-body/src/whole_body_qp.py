"""Whole-body QP (weighted least-squares) controller.

Solves a simplified whole-body control problem:
  min  sum_i  w_i * ||J_i * qdd - a_d_i||^2
  s.t. M * qdd + h = S^T * tau + Jc^T * fc  (dynamics)
       friction cone on fc
       joint torque limits

Implementation uses a weighted least-squares approach with
regularized pseudoinverse, suitable for real-time control without
requiring OSQP or other QP solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from g1_model import G1WholeBodyModel
from tasks import Task
from contact_model import ContactInfo, build_friction_cone_constraints
from lab8_common import NUM_ACTUATED, NV, NV_FREEFLYER, ACTUATOR_KP


@dataclass
class QPResult:
    """Result of the whole-body QP solve.

    Attributes:
        qdd: Joint accelerations [nv].
        tau: Actuated joint torques [na].
        contact_forces: Contact forces {frame_name: force [3]}.
        cost: Total cost value.
        success: Whether the solve succeeded.
    """
    qdd: np.ndarray
    tau: np.ndarray
    contact_forces: Dict[str, np.ndarray] = field(default_factory=dict)
    cost: float = 0.0
    success: bool = True


class WholeBodyQP:
    """Weighted least-squares whole-body controller.

    Solves for joint accelerations that best track multiple task objectives
    while respecting dynamics and contact constraints.

    The approach:
    1. Stack weighted task objectives: min ||W^(1/2) (J_stack * qdd - a_stack)||^2
    2. Add regularization for numerical stability
    3. Solve for qdd via regularized normal equations
    4. Recover torques from dynamics: tau = S * (M * qdd + h - Jc^T * fc)
    5. Contact forces are estimated via projection

    Attributes:
        wb_model: G1 whole-body model.
        reg: Regularization parameter.
    """

    def __init__(
        self,
        wb_model: G1WholeBodyModel,
        regularization: float = 1e-6,
    ) -> None:
        """Initialize the WB-QP controller.

        Args:
            wb_model: G1 whole-body model.
            regularization: Tikhonov regularization for the least-squares solve.
        """
        self.wb_model = wb_model
        self.reg = regularization

    def solve(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tasks: List[Task],
        active_contacts: List[ContactInfo],
        tau_max: Optional[np.ndarray] = None,
    ) -> QPResult:
        """Solve the whole-body weighted least-squares problem.

        Args:
            q: Generalized positions [nq].
            v: Generalized velocities [nv].
            tasks: List of active tasks.
            active_contacts: List of active contact infos.
            tau_max: Optional torque limits [na].

        Returns:
            QPResult with accelerations, torques, and contact forces.
        """
        nv = self.wb_model.nv
        na = self.wb_model.na
        nc = len(active_contacts)  # number of contact points
        nf = nc * 3  # total contact force dimensions

        # Compute dynamics quantities
        M = self.wb_model.mass_matrix(q)
        h = self.wb_model.nonlinear_effects(q, v)
        S = self.wb_model.S  # (na x nv)

        # Contact Jacobians (stacked: only translation part, 3 rows per contact)
        Jc_list = []
        for ci in active_contacts:
            J_full = self.wb_model.get_frame_jacobian(q, ci.frame_name)
            Jc_list.append(J_full[:3, :])  # translational part only

        Jc = np.vstack(Jc_list) if Jc_list else np.zeros((0, nv))

        # Stack weighted task objectives
        J_rows = []
        a_rows = []
        w_rows = []

        for task in tasks:
            J_t = task.compute_jacobian(self.wb_model, q)
            a_t = task.compute_desired_acceleration(self.wb_model, q, v)
            w_t = task.weight

            J_rows.append(J_t)
            a_rows.append(a_t)
            w_rows.append(np.full(task.dim, w_t))

        if not J_rows:
            # No tasks, return zero
            return QPResult(
                qdd=np.zeros(nv),
                tau=np.zeros(na),
                success=True,
            )

        J_stack = np.vstack(J_rows)
        a_stack = np.concatenate(a_rows)
        w_stack = np.concatenate(w_rows)

        # Weighted Jacobian and desired acceleration
        W_sqrt = np.diag(np.sqrt(w_stack))
        Jw = W_sqrt @ J_stack
        aw = W_sqrt @ a_stack

        # Solve weighted least squares: min ||Jw * qdd - aw||^2 + reg * ||qdd||^2
        # Normal equations: (Jw^T Jw + reg*I) qdd = Jw^T aw
        JtJ = Jw.T @ Jw + self.reg * np.eye(nv)
        Jta = Jw.T @ aw

        # If contacts exist, we need to enforce dynamics consistency
        if nc > 0:
            # Solve as an augmented system:
            # We want qdd such that unactuated base dynamics are consistent:
            #   M[:6,:] @ qdd + h[:6] = Jc[:, :6]^T @ fc (no actuation on base)
            # And actuated dynamics give torques:
            #   tau = M[6:,:] @ qdd + h[6:] - Jc[:, 6:]^T @ fc

            # Estimate contact forces to support the robot against gravity
            # Use least-norm solution for contact forces from base dynamics
            fc = self._estimate_contact_forces(M, h, Jc, JtJ, Jta, nv, nc)

            # Solve for qdd accounting for contact forces
            # Add dynamics constraint as a high-weight task:
            # Base dynamics: M_base @ qdd + h_base - Jc_base^T @ fc = 0
            # This is a 6D constraint (unactuated base DOFs)
            dynamics_weight = 10000.0
            M_base = M[:NV_FREEFLYER, :]
            h_base = h[:NV_FREEFLYER]
            Jc_base_T = Jc[:, :NV_FREEFLYER].T  # (6 x nf)

            rhs_base = Jc_base_T @ fc - h_base
            JtJ += dynamics_weight * M_base.T @ M_base
            Jta += dynamics_weight * M_base.T @ rhs_base

            try:
                qdd = np.linalg.solve(JtJ, Jta)
            except np.linalg.LinAlgError:
                qdd = np.linalg.lstsq(JtJ, Jta, rcond=None)[0]

            # Recover torques from full dynamics
            tau = S @ (M @ qdd + h - Jc.T @ fc)

            contact_forces = {}
            for i, ci in enumerate(active_contacts):
                contact_forces[ci.frame_name] = fc[3 * i: 3 * (i + 1)]
        else:
            # No contacts (flight phase) - just solve WLS
            try:
                qdd = np.linalg.solve(JtJ, Jta)
            except np.linalg.LinAlgError:
                qdd = np.linalg.lstsq(JtJ, Jta, rcond=None)[0]

            tau = S @ (M @ qdd + h)
            fc = np.zeros(0)
            contact_forces = {}

        # Clip torques if limits provided
        if tau_max is not None:
            tau = np.clip(tau, -tau_max, tau_max)

        # Compute cost for diagnostics
        cost = float(np.sum((Jw @ qdd - aw) ** 2))

        return QPResult(
            qdd=qdd,
            tau=tau,
            contact_forces=contact_forces,
            cost=cost,
            success=True,
        )

    def _estimate_contact_forces(
        self,
        M: np.ndarray,
        h: np.ndarray,
        Jc: np.ndarray,
        JtJ: np.ndarray,
        Jta: np.ndarray,
        nv: int,
        nc: int,
    ) -> np.ndarray:
        """Estimate contact forces from base dynamics.

        For static/quasi-static cases, we estimate fc such that the
        base (unactuated) dynamics are close to zero acceleration:
          M_base @ qdd + h_base = Jc_base^T @ fc

        Uses least-norm solution.

        Args:
            M: Mass matrix [nv, nv].
            h: Nonlinear effects [nv].
            Jc: Stacked contact Jacobians [3*nc, nv].
            JtJ: Pre-computed J^T W J.
            Jta: Pre-computed J^T W a.
            nv: Number of velocity DOFs.
            nc: Number of contacts.

        Returns:
            Contact forces [3*nc].
        """
        nf = 3 * nc
        # For the base DOFs (unactuated), in quasi-static:
        # Jc_base^T @ fc ≈ h_base (gravity compensation)
        Jc_base_T = Jc[:, :NV_FREEFLYER].T  # (6 x nf)
        h_base = h[:NV_FREEFLYER]

        # Solve: Jc_base_T @ fc = h_base
        # This is underdetermined if nc > 2, least-norm otherwise
        try:
            fc, _, _, _ = np.linalg.lstsq(Jc_base_T, h_base, rcond=None)
        except np.linalg.LinAlgError:
            fc = np.zeros(nf)

        # Enforce positive normal forces
        for i in range(nc):
            if fc[3 * i + 2] < 0:
                fc[3 * i + 2] = 1.0  # minimum normal force

        return fc


def solve_whole_body_qp(
    wb_model: G1WholeBodyModel,
    q: np.ndarray,
    v: np.ndarray,
    tasks: List[Task],
    active_contacts: List[ContactInfo],
    regularization: float = 1e-6,
    tau_max: Optional[np.ndarray] = None,
) -> QPResult:
    """Convenience function to create and solve a whole-body QP.

    Args:
        wb_model: G1 whole-body model.
        q: Generalized positions [nq].
        v: Generalized velocities [nv].
        tasks: List of active tasks.
        active_contacts: List of active contact infos.
        regularization: Tikhonov regularization.
        tau_max: Optional torque limits [na].

    Returns:
        QPResult.
    """
    qp = WholeBodyQP(wb_model, regularization)
    return qp.solve(q, v, tasks, active_contacts, tau_max)
