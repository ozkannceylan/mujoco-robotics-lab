"""Lab 8 — M1 Step 1.2: whole-body QP (velocity level, OSQP).

Solves, once per control tick, for the joint velocity that best satisfies the
whole task stack::

    min_q̇   Σ_i w_i ‖ J_i q̇ − ẋ_i ‖²  +  λ ‖q̇‖²
    s.t.     q̇_min ≤ q̇ ≤ q̇_max

which OSQP takes in the standard form ``min ½ q̇ᵀP q̇ + pᵀq̇  s.t.  l ≤ A q̇ ≤ u``
with ``P = Σ w_i J_iᵀJ_i + λI`` and ``p = −Σ w_i J_iᵀ ẋ_i``.

**Weighted, not strictly hierarchical.** `plan/LAB_08.md` argues for strict
priorities, and it is right that balance must never be traded for
manipulation. This implementation approximates that with a weight ladder
(feet 1e6 → CoM 1e4 → hand 1e2 → posture 1) rather than a cascade of
null-space projections, because:

* it is one QP instead of N, which matters at 1 kHz on a 35-DOF model;
* the residual of a lower level cannot visibly perturb a higher one at a 1e2
  weight gap — and, unlike a strict hierarchy, a slightly infeasible high
  task degrades gracefully instead of leaving the lower levels unsolvable;
* the per-task errors are logged, so if a lower task ever *does* start
  bending a higher one the gate tables will show it rather than hide it.

If M2/M3 evidence shows leakage, the escalation path is real hierarchical QP
(Escande et al.) — that decision is recorded in `tasks/ARCHITECTURE.md`.

Joint position limits enter as velocity bounds (``q̇ ≤ (q_max − q)/dt``): a
purely velocity-level QP has no other way to see them, and clamping the
solution afterwards would silently violate the tasks the QP just balanced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import osqp
from scipy import sparse

from lab8_common import NU, NV

__all__ = ["QPInfeasible", "QPResult", "WholeBodyQP"]


class QPInfeasible(RuntimeError):
    """Raised when OSQP cannot solve the tick's problem.

    Deliberately an exception rather than a fallback: a silently clamped or
    zeroed solution looks like a working controller while the robot falls
    over, and that failure mode is much harder to debug than a stack trace.
    """


@dataclass
class QPResult:
    """Solution of one QP tick."""

    qdot: np.ndarray
    status: str
    iterations: int
    task_errors: dict[str, float] = field(default_factory=dict)
    solve_time_ms: float = 0.0


class WholeBodyQP:
    """Velocity-level whole-body QP over a `TaskStack`.

    Args:
        model: Pinocchio model (used for joint limits).
        dt: Control timestep [s] — converts position limits into velocity bounds.
        damping: Tikhonov weight λ on ‖q̇‖². Regularises the redundant
            null space and keeps P positive definite when tasks under-determine
            the solution.
        velocity_limit: Max |q̇| per actuated joint [rad/s].
        position_limit_margin: Stop-band [rad] before a joint limit at which
            the velocity bound starts closing.
    """

    def __init__(
        self,
        model,
        dt: float,
        damping: float = 1e-4,
        velocity_limit: float = 8.0,
        position_limit_margin: float = 0.05,
    ) -> None:
        self.model = model
        self.dt = float(dt)
        self.damping = float(damping)
        self.velocity_limit = float(velocity_limit)
        self.position_limit_margin = float(position_limit_margin)

        # Pinocchio stores limits over nq; the actuated joints are the tail.
        self.q_lower = np.asarray(model.lowerPositionLimit, dtype=float)[7:]
        self.q_upper = np.asarray(model.upperPositionLimit, dtype=float)[7:]

        self._solver: osqp.OSQP | None = None
        # OSQP reads only the upper triangle of P and requires the sparsity
        # pattern to stay fixed across hot updates. Keeping every upper-triangle
        # entry — including numerically zero ones — guarantees that; letting
        # scipy drop zeros makes nnz drift between ticks and update() fails with
        # "new number of elements out of bounds".
        self._triu_rows, self._triu_cols = np.triu_indices(NV)

    # -- bounds ------------------------------------------------------------

    def velocity_bounds(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-DOF velocity bounds (nv,), floating base left unconstrained."""
        lower = np.full(NV, -np.inf)
        upper = np.full(NV, np.inf)

        lower[6:] = -self.velocity_limit
        upper[6:] = self.velocity_limit

        q_joints = np.asarray(q, dtype=float)[7:]
        finite = np.isfinite(self.q_lower) & np.isfinite(self.q_upper)
        if np.any(finite):
            margin = self.position_limit_margin
            room_up = (self.q_upper - q_joints - margin) / self.dt
            room_down = (self.q_lower - q_joints + margin) / self.dt
            upper[6:][finite] = np.minimum(upper[6:][finite], np.maximum(room_up[finite], 0.0))
            lower[6:][finite] = np.maximum(lower[6:][finite], np.minimum(room_down[finite], 0.0))
        return lower, upper

    # -- solve -------------------------------------------------------------

    def solve(self, stack, q: np.ndarray) -> QPResult:
        """Solve one tick. `stack.update(q)` must have been called already."""
        jacobian, xdot_des, weights = stack.assemble(q)
        if jacobian.shape[0] == 0:
            return QPResult(np.zeros(NV), "no_tasks", 0)

        weighted = jacobian * weights[:, None]
        hessian = jacobian.T @ weighted + self.damping * np.eye(NV)
        hessian = 0.5 * (hessian + hessian.T)  # enforce exact symmetry for OSQP
        gradient = -weighted.T @ xdot_des

        lower, upper = self.velocity_bounds(q)
        constraint = sparse.eye(NV, format="csc")

        p_values = hessian[self._triu_rows, self._triu_cols]
        p_sparse = sparse.csc_matrix(
            (p_values, (self._triu_rows, self._triu_cols)), shape=(NV, NV)
        )
        if self._solver is None:
            self._solver = osqp.OSQP()
            self._solver.setup(
                P=p_sparse, q=gradient, A=constraint, l=lower, u=upper,
                verbose=False, polishing=False, eps_abs=1e-6, eps_rel=1e-6,
            )
        else:
            # Hot path: same pattern, new values.
            self._solver.update(Px=p_sparse.data, q=gradient, l=lower, u=upper)

        solution = self._solver.solve()
        status = str(solution.info.status)
        if solution.x is None or not np.all(np.isfinite(solution.x)):
            raise QPInfeasible(
                f"OSQP returned no usable solution (status='{status}'). "
                f"tasks={[t.name for t in stack.active]}"
            )
        if "solved" not in status:
            raise QPInfeasible(
                f"OSQP status '{status}' with tasks {[t.name for t in stack.active]}. "
                "Check for conflicting high-weight tasks or over-tight limits."
            )

        return QPResult(
            qdot=np.asarray(solution.x, dtype=float),
            status=status,
            iterations=int(solution.info.iter),
            task_errors=stack.errors(q),
            solve_time_ms=float(solution.info.solve_time) * 1e3,
        )
