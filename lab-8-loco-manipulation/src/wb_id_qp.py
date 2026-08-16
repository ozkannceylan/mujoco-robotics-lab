"""Lab 8 — M1 Step 1.2: whole-body inverse-dynamics QP (acceleration level).

Solves, once per control tick, for joint accelerations **and** contact wrenches
together::

    min_{q̈, f}   Σ_i w_i ‖ J_i q̈ + J̇_i q̇ − ẍ_i ‖²  +  λ_a‖q̈‖² + λ_f‖f‖²

    s.t.   M q̈ + h = Sᵀτ + J_cᵀ f        (rigid-body dynamics)
           J_c q̈ + J̇_c q̇ = 0             (stance feet do not accelerate)
           friction cone, unilateral f_z, centre-of-pressure inside the foot
           |τ| ≤ τ_max

The first six rows of the dynamics involve no actuator (the floating base is
unactuated), so they enter as an equality constraint on ``(q̈, f)``; the
remaining 29 rows *define* the torque, which is read out after the solve.

Why not the velocity-level QP in `wb_qp.py`
-------------------------------------------
That formulation was written first and measured to fail (LESSONS L-M1-a): a
kinematic QP can hold `J_com q̇ = 0` perfectly while the robot topples,
because CoM motion is not something joint velocity commands — it is produced
by contact forces. Making the hand task stronger reliably made the G1 fall
*faster*, which is the signature of a controller optimising the wrong
variable. Here the contact wrenches are decision variables constrained by
friction and the support geometry, so "keep the CoM over the feet" becomes a
statement the solver can actually enforce.

`wb_qp.WholeBodyQP` is kept for pure kinematic problems (e.g. swing-foot
retargeting in M2), not for balance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import osqp
import pinocchio as pin
from scipy import sparse

from lab8_common import NU, NV

__all__ = ["ContactSpec", "IDQPResult", "WholeBodyIDQP"]

_CONTACT_DIM = 6  # 6D wrench per foot


@dataclass
class ContactSpec:
    """A flat, non-slipping foot contact.

    Defaults describe the **actual** Menagerie G1 foot: four contact spheres of
    radius 5 mm at x ∈ {−0.05, 0.12}, y ∈ {±0.025, ±0.03}, z = −0.03 in the
    ankle-roll frame. Three consequences the original symmetric ±0.08 box got
    wrong, all of which matter for walking (LESSONS L-M3-d):

    * The patch is **not centred on the ankle**. It reaches 0.12 m forward but
      only 0.05 m back, so the honest description is a half-length of 0.085
      about a centre 0.035 m ahead of the frame. Assuming ±0.08 simultaneously
      *over*-claims 30 mm of rearward CoP the foot does not have — the QP plans
      contact wrenches MuJoCo then refuses to produce — and throws away 40 mm
      of the forward authority that decelerating the CoM before touchdown
      needs.
    * The sole sits 0.035 m **below** the frame origin the wrench is expressed
      about, so `CoP = −m_y/f_z` is only right when f_x is zero. Walking is
      exactly when it is not: at 0.3·mg of shear that is a 10 mm CoP error, on
      an axis with 85 mm of total travel.
    * The rear pair is 0.025 m off centre, not 0.03.

    Args:
        frame_name: Pinocchio frame of the foot (ankle roll link).
        friction: Coulomb coefficient (Menagerie G1 foot geoms use 0.6).
        half_length: Contact patch half-extent along local +x [m].
        half_width: Contact patch half-extent along local +y [m].
        center_x / center_y: Patch centre relative to the frame origin [m].
        origin_height: Height of the frame origin above the contact plane [m].
        min_normal_force: Lower bound on f_z [N] — keeps the solver from
            "letting go" of a stance foot to make a task easier.
    """

    frame_name: str
    friction: float = 0.6
    half_length: float = 0.085
    half_width: float = 0.025
    center_x: float = 0.035
    center_y: float = 0.0
    origin_height: float = 0.035
    min_normal_force: float = 1.0


@dataclass
class IDQPResult:
    """Solution of one inverse-dynamics QP tick."""

    tau: np.ndarray
    qddot: np.ndarray
    forces: np.ndarray
    status: str
    iterations: int
    solve_time_ms: float = 0.0
    task_errors: dict[str, float] = field(default_factory=dict)


class QPInfeasible(RuntimeError):
    """Raised when the tick's QP has no usable solution."""


class WholeBodyIDQP:
    """Task-space inverse dynamics with contact wrenches as decision variables.

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        contacts: Stance contacts held during this phase.
        torque_limits: (nu, 2) actuator limits [N·m].
        tolerance: OSQP primal/dual tolerance. The stack spans task weights
            from 1e4 down to 1e1 against a 1e-4 regularisation, so demanding
            1e-6 on a problem conditioned like that costs iterations without
            buying control accuracy: at 1e-6 the walking gait hit the 4000
            iteration cap on 38 % of ticks and averaged 12.6 ms a solve, versus
            a 1 ms budget. 1e-4 is well below the acceleration resolution the
            robot can act on.
        max_iterations: Iteration cap. A tick that hits it still returns a
            usable, dynamically-consistent solution (the base-dynamics residual
            stays ≲0.02 N·m) — it is a warning about conditioning, not a
            failure.
        acc_regularisation: λ_a on ‖q̈‖².
        force_regularisation: λ_f on ‖f‖² — prefers the smallest wrench that
            does the job, which also keeps the two feet from fighting.
    """

    def __init__(
        self,
        model: pin.Model,
        data: pin.Data,
        contacts: list[ContactSpec],
        torque_limits: np.ndarray,
        acc_regularisation: float = 1e-4,
        force_regularisation: float = 1e-5,
        tolerance: float = 1e-4,
        max_iterations: int = 2000,
    ) -> None:
        self.model = model
        self.data = data
        self.contacts = list(contacts)
        self.torque_limits = np.asarray(torque_limits, dtype=float)
        self.acc_reg = float(acc_regularisation)
        self.force_reg = float(force_regularisation)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)

        self._solver: osqp.OSQP | None = None
        self.set_contacts(self.contacts)

    def set_contacts(self, contacts: list[ContactSpec]) -> None:
        """Replace the stance set — how a gait switches support phases.

        The problem *size* changes with the number of contacts (6 decision
        variables each), so the OSQP instance is discarded rather than hot
        updated: its factorisation is built for a fixed dimension. At ~0.1 ms
        per solve a rebuild on the two phase transitions per step is
        negligible, and correctness beats reusing a stale factorisation.
        """
        self.contacts = list(contacts)
        self.contact_ids = [self.model.getFrameId(c.frame_name) for c in self.contacts]
        self.n_forces = _CONTACT_DIM * len(self.contacts)
        self.n_vars = NV + self.n_forces
        self._triu_rows, self._triu_cols = np.triu_indices(self.n_vars)
        self._solver = None

    @property
    def contact_frames(self) -> list[str]:
        """Names of the frames currently in contact."""
        return [c.frame_name for c in self.contacts]

    # -- model terms -------------------------------------------------------

    def _contact_jacobian(self, q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stacked contact Jacobian (6n × nv) and its drift `J̇_c q̇` (6n,)."""
        jacobians, drifts = [], []
        for frame_id in self.contact_ids:
            jacobians.append(
                pin.getFrameJacobian(
                    self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
            )
            acc = pin.getFrameClassicalAcceleration(
                self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            drifts.append(np.concatenate([acc.linear, acc.angular]))
        del q, v
        return np.vstack(jacobians), np.concatenate(drifts)

    def _friction_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearised friction pyramid + CoP bounds for all contacts.

        Rows act on the force block only; the caller pads the q̈ columns.

        The wrench `[fx, fy, fz, mx, my, mz]` is world-aligned about the foot
        frame's origin, which sits `h` above the ground, so the CoP on the
        contact plane is::

            CoP_x = (−m_y − h·f_x) / f_z
            CoP_y = ( m_x − h·f_y) / f_z

        and the bound `CoP ∈ centre ± half` becomes the four linear rows below
        after multiplying through by `f_z > 0`. World-aligned axes make the
        patch offsets world-axis offsets too — exact for this straight-line
        gait, and the place a turning gait would have to rotate the patch
        corners before taking these bounds.
        """
        blocks, lower, upper = [], [], []
        for contact in self.contacts:
            mu, hl, hw = contact.friction, contact.half_length, contact.half_width
            cx, cy, h = contact.center_x, contact.center_y, contact.origin_height
            # Ordering of a 6D wrench row: [fx, fy, fz, mx, my, mz]
            rows = [
                ([0, 0, mu, 0, 0, 0], 0.0, np.inf),          # unilateral (with fz>=fmin below)
                ([1, 0, mu, 0, 0, 0], 0.0, np.inf),          # fx ≥ −mu fz
                ([-1, 0, mu, 0, 0, 0], 0.0, np.inf),         # fx ≤  mu fz
                ([0, 1, mu, 0, 0, 0], 0.0, np.inf),          # fy ≥ −mu fz
                ([0, -1, mu, 0, 0, 0], 0.0, np.inf),         # fy ≤  mu fz
                ([0, h, cy + hw, -1, 0, 0], 0.0, np.inf),    # CoP_y ≤ cy + hw
                ([0, -h, hw - cy, 1, 0, 0], 0.0, np.inf),    # CoP_y ≥ cy − hw
                ([h, 0, cx + hl, 0, 1, 0], 0.0, np.inf),     # CoP_x ≤ cx + hl
                ([-h, 0, hl - cx, 0, -1, 0], 0.0, np.inf),   # CoP_x ≥ cx − hl
                ([0, 0, 1, 0, 0, 0], contact.min_normal_force, np.inf),
            ]
            blocks.append(np.array([r[0] for r in rows], dtype=float))
            lower.extend(r[1] for r in rows)
            upper.extend(r[2] for r in rows)

        n_rows = sum(b.shape[0] for b in blocks)
        matrix = np.zeros((n_rows, self.n_forces))
        row = 0
        for i, block in enumerate(blocks):
            matrix[row:row + block.shape[0], i * _CONTACT_DIM:(i + 1) * _CONTACT_DIM] = block
            row += block.shape[0]
        return matrix, np.array(lower), np.array(upper)

    # -- solve -------------------------------------------------------------

    def solve(self, stack, q: np.ndarray, v: np.ndarray) -> IDQPResult:
        """Solve one tick. `stack.update_dynamics(q, v)` must precede this."""
        mass_matrix = pin.crba(self.model, self.data, q)
        mass_matrix = np.triu(mass_matrix) + np.triu(mass_matrix, 1).T
        nonlinear = pin.nonLinearEffects(self.model, self.data, q, v)

        contact_jac, contact_drift = self._contact_jacobian(q, v)

        # ---- cost -------------------------------------------------------
        hessian = np.zeros((self.n_vars, self.n_vars))
        gradient = np.zeros(self.n_vars)
        for task in stack.active:
            jac = task.jacobian(self.model, self.data, q)
            target = task.desired_acceleration(self.model, self.data, q, v) - task.drift(
                self.model, self.data
            )
            hessian[:NV, :NV] += task.weight * (jac.T @ jac)
            gradient[:NV] += -task.weight * (jac.T @ target)
        hessian[:NV, :NV] += self.acc_reg * np.eye(NV)
        hessian[NV:, NV:] += self.force_reg * np.eye(self.n_forces)
        hessian = 0.5 * (hessian + hessian.T)

        # ---- constraints -------------------------------------------------
        rows_a, rows_l, rows_u = [], [], []

        # (1) unactuated base dynamics: M[:6] q̈ − J_cᵀ[:6] f = −h[:6]
        base = np.zeros((6, self.n_vars))
        base[:, :NV] = mass_matrix[:6, :]
        base[:, NV:] = -contact_jac.T[:6, :]
        rows_a.append(base)
        rows_l.append(-nonlinear[:6])
        rows_u.append(-nonlinear[:6])

        # (2) stance feet hold: J_c q̈ = −J̇_c q̇
        contact_rows = np.zeros((contact_jac.shape[0], self.n_vars))
        contact_rows[:, :NV] = contact_jac
        rows_a.append(contact_rows)
        rows_l.append(-contact_drift)
        rows_u.append(-contact_drift)

        # (3) friction cone / CoP / unilateral
        friction, f_lower, f_upper = self._friction_constraints()
        friction_rows = np.zeros((friction.shape[0], self.n_vars))
        friction_rows[:, NV:] = friction
        rows_a.append(friction_rows)
        rows_l.append(f_lower)
        rows_u.append(f_upper)

        # (4) torque limits: τ = M[6:] q̈ + h[6:] − J_cᵀ[6:] f
        torque_rows = np.zeros((NU, self.n_vars))
        torque_rows[:, :NV] = mass_matrix[6:, :]
        torque_rows[:, NV:] = -contact_jac.T[6:, :]
        rows_a.append(torque_rows)
        rows_l.append(self.torque_limits[:, 0] - nonlinear[6:])
        rows_u.append(self.torque_limits[:, 1] - nonlinear[6:])

        constraint = np.vstack(rows_a)
        lower = np.concatenate(rows_l)
        upper = np.concatenate(rows_u)

        # ---- solve -------------------------------------------------------
        p_values = hessian[self._triu_rows, self._triu_cols]
        p_sparse = sparse.csc_matrix(
            (p_values, (self._triu_rows, self._triu_cols)), shape=(self.n_vars, self.n_vars)
        )
        a_sparse = sparse.csc_matrix(constraint)

        if self._solver is None:
            self._solver = osqp.OSQP()
            self._solver.setup(
                P=p_sparse, q=gradient, A=a_sparse, l=lower, u=upper,
                verbose=False, polishing=False,
                eps_abs=self.tolerance, eps_rel=self.tolerance,
                max_iter=self.max_iterations,
            )
            self._a_nnz = a_sparse.nnz
        elif a_sparse.nnz != self._a_nnz:
            # Contact set changed shape — rebuild rather than corrupt the solver.
            self._solver = osqp.OSQP()
            self._solver.setup(
                P=p_sparse, q=gradient, A=a_sparse, l=lower, u=upper,
                verbose=False, polishing=False,
                eps_abs=self.tolerance, eps_rel=self.tolerance,
                max_iter=self.max_iterations,
            )
            self._a_nnz = a_sparse.nnz
        else:
            self._solver.update(Px=p_sparse.data, q=gradient, Ax=a_sparse.data, l=lower, u=upper)

        solution = self._solver.solve()
        status = str(solution.info.status)
        if solution.x is None or not np.all(np.isfinite(solution.x)):
            raise QPInfeasible(f"OSQP returned no usable solution (status='{status}')")

        qddot = np.asarray(solution.x[:NV], dtype=float)
        forces = np.asarray(solution.x[NV:], dtype=float)
        tau = mass_matrix[6:, :] @ qddot + nonlinear[6:] - contact_jac.T[6:, :] @ forces

        return IDQPResult(
            tau=np.clip(tau, self.torque_limits[:, 0], self.torque_limits[:, 1]),
            qddot=qddot,
            forces=forces,
            status=status,
            iterations=int(solution.info.iter),
            solve_time_ms=float(solution.info.solve_time) * 1e3,
            task_errors=stack.errors(q),
        )
