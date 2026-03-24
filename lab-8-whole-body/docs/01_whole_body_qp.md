# 01: Whole-Body QP Controller

## Goal

Build a task-priority whole-body controller for the Unitree G1 humanoid using quadratic programming. The controller must simultaneously track multiple objectives — center-of-mass position, foot placements, hand targets, and a default posture — while respecting the robot's equation of motion, friction cone constraints, and torque limits.

## Files

- QP controller: `src/whole_body_qp.py`
- Task definitions: `src/tasks.py`
- Contact model: `src/contact_model.py`
- G1 model wrapper: `src/g1_model.py`
- Standing + reaching demo: `src/a1_standing_reach.py`

## The Whole-Body Control Problem

A humanoid robot has many degrees of freedom — the G1 has approximately 37 actuated joints plus a 6-DOF floating base — but the tasks it needs to accomplish often conflict. Standing requires keeping the center of mass above the feet. Reaching with the left hand requires moving the arm, which shifts the CoM. Keeping the feet planted constrains the legs. Maintaining a natural posture pulls the entire body toward a default configuration.

No single task can be solved in isolation. Moving the hand changes the CoM. Compensating the CoM requires adjusting the legs. Adjusting the legs may violate foot contact constraints. The whole-body controller must resolve all of these simultaneously.

The key insight is that this is an optimization problem. We cannot satisfy all objectives perfectly, but we can find the best compromise by assigning priorities to each task and solving for the joint accelerations, torques, and contact forces that minimize a weighted combination of task errors while respecting the physical constraints.

## Task Definitions

Each task is defined by three quantities: an error vector, a Jacobian, and a desired acceleration.

### CoMTask (3D, weight = 1000)

The center-of-mass task tracks a desired 3D CoM position. This is the highest-priority task because losing CoM balance means the robot falls.

```python
error = com_desired - com_current           # (3,)
J = J_com                                   # (3 x nv)
a_desired = kp * error + kd * (0 - v_com)   # PD control
```

The CoM Jacobian `J_com` is computed by Pinocchio via `pin.jacobianCenterOfMass()`. It maps generalized velocities to CoM velocity, so `v_com = J_com @ v`. The PD gains are `kp=100, kd=20`.

### FootPoseTask (6D, weight = 100)

Each foot has a 6D pose task tracking a desired SE3 position and orientation. During stance, the desired pose is the current ground contact pose. During swing, it follows the gait generator's foot trajectory.

```python
error = pin.log6(T_current.inverse() * T_desired).vector   # (6,)
J = J_foot                                                  # (6 x nv)
a_desired = kp * error + kd * (0 - J @ v)                   # kp=200, kd=30
```

The `pin.log6()` function maps the SE3 error to a 6D twist in the local frame. This is the correct way to compute orientation error — it avoids the singularities and discontinuities that arise from Euler angle subtraction.

### HandPoseTask (6D, weight = 10)

Same structure as FootPoseTask but for the hands. Lower weight means the hands are less important than CoM stability or foot placement. The controller will sacrifice hand tracking accuracy to maintain balance.

```python
error = pin.log6(T_current.inverse() * T_desired).vector   # (6,)
J = J_hand                                                  # (6 x nv)
a_desired = kp * error + kd * (0 - J @ v)                   # kp=100, kd=20
```

### PostureTask (na-dimensional, weight = 1)

The posture task pulls the actuated joints toward a default standing configuration. This is the lowest priority — it acts as a regularizer that prevents the robot from drifting into awkward configurations when the primary tasks leave null-space freedom.

```python
error = q_desired[7:] - q_current[7:]     # actuated joints only, skip floating base
J = [0_{na x 6} | I_{na x na}]            # (na x nv), zeros for base DOFs
a_desired = kp * error + kd * (0 - v[6:]) # kp=10, kd=3
```

The leading zero block in the Jacobian means the posture task does not directly command the floating base — it only affects actuated joints.

## Weight-Based Priority: 1000 : 100 : 10 : 1

The four tasks are combined into a single QP cost function with weights that span three orders of magnitude:

```
min  1000 * ||J_com * qdd - a_com||^2
   + 100  * ||J_lfoot * qdd - a_lfoot||^2
   + 100  * ||J_rfoot * qdd - a_rfoot||^2
   + 10   * ||J_hand * qdd - a_hand||^2
   + 1    * ||J_posture * qdd - a_posture||^2
```

This is a soft priority scheme. The CoM task is 10x more important than the feet, which are 10x more important than the hands, which are 10x more important than posture. In practice, the large weight ratios produce behavior that closely approximates strict task priority — the lower-priority tasks only use whatever freedom remains after the higher-priority tasks are (approximately) satisfied.

The alternative is hierarchical QP (HQP), which solves a cascade of QPs where each level is solved in the null space of all higher levels. HQP gives mathematically strict priority but requires multiple QP solves per timestep. For a 500 Hz controller, the simpler weighted approach is fast enough and the 1000:1 ratio provides sufficient separation.

## QP Formulation

### Decision Variables

The QP solves for three sets of variables simultaneously:

| Variable | Symbol | Size | Description |
|----------|--------|------|-------------|
| Joint accelerations | `qdd` | nv | Generalized accelerations |
| Actuator torques | `tau` | na | Torques at actuated joints |
| Contact forces | `f_c` | nc * 3 | 3D forces at each contact point |

For the G1 in double support: nv ~ 43, na ~ 37, nc = 2, so the total decision vector has ~49 elements.

### Cost Function

```
min  sum_i  w_i * || J_i * qdd - a_d_i ||^2
```

Expanding the squared norm:

```
= qdd^T * (sum_i w_i * J_i^T * J_i) * qdd
  - 2 * (sum_i w_i * a_d_i^T * J_i) * qdd
  + const
```

This is a standard quadratic form `0.5 * x^T * P * x + q^T * x` where:
- `P = 2 * sum_i  w_i * J_i^T * J_i` (only the qdd block of the Hessian)
- `q = -2 * sum_i  w_i * J_i^T * a_d_i`

A small regularization term `eps * I` is added to `P` to ensure positive definiteness.

### Equation of Motion Constraint

The rigid body dynamics must be satisfied exactly:

```
M(q) * qdd + h(q, v) = S^T * tau + Jc^T * f_c
```

Where:
- `M(q)` is the mass matrix (nv x nv) from Pinocchio CRBA
- `h(q, v)` is the nonlinear effects vector (Coriolis + gravity) from Pinocchio RNEA
- `S` is the selection matrix (na x nv) that maps actuated torques to generalized forces
- `Jc` is the stacked contact Jacobian
- `f_c` is the stacked contact forces

Rearranging as a linear equality constraint on the decision vector `x = [qdd; tau; f_c]`:

```
[M | -S^T | -Jc^T] * [qdd; tau; f_c] = -h(q, v)
```

This constraint ensures that the computed accelerations, torques, and contact forces are physically consistent. Without it, the QP could produce accelerations that violate Newton's laws.

### Friction Cone Constraints

A contact force is physically realizable only if it stays within the friction cone. For a flat ground contact with friction coefficient `mu`, the exact cone is:

```
sqrt(fx^2 + fy^2) <= mu * fz
fz >= 0
```

This is a second-order cone constraint, not linear. To use a standard QP solver, we linearize the cone with 4 facets (a square pyramid approximation):

```
+fx - mu/sqrt(2) * fz <= 0
-fx - mu/sqrt(2) * fz <= 0
+fy - mu/sqrt(2) * fz <= 0
-fy - mu/sqrt(2) * fz <= 0
fz >= 0
```

The `mu/sqrt(2)` factor ensures the inscribed pyramid is conservative — forces inside the linearized cone are guaranteed to be inside the true cone. With `mu = 0.7`, the effective tangential friction bound per axis is `0.7 / sqrt(2) ~ 0.495`.

Four facets are sufficient for flat-ground walking. More facets (8 or 16) improve accuracy but add more inequality constraints and slow down the solver.

### Torque and Acceleration Limits

```
tau_min <= tau <= tau_max
qdd_min <= qdd <= qdd_max
```

These are simple box constraints. Torque limits come from the actuator specifications. Acceleration limits are set conservatively to prevent the solver from producing unrealistically large accelerations during transient phases.

## OSQP Solver Configuration

OSQP is an operator-splitting QP solver designed for real-time applications. Key settings:

```python
solver = osqp.OSQP()
solver.setup(
    P=P_sparse,       # upper-triangular Hessian (CSC format)
    q=q_vec,
    A=A_sparse,        # stacked equality + inequality constraint matrix
    l=lower_bounds,
    u=upper_bounds,
    eps_abs=1e-4,
    eps_rel=1e-4,
    max_iter=200,
    warm_start=True,
    verbose=False,
    polish=True,
)
```

Key choices:
- **Warm starting** reuses the previous solution as the initial guess. This is critical for real-time performance — consecutive QP problems at 500 Hz are nearly identical, so warm starting typically converges in 5-15 iterations instead of 50+.
- **Polishing** refines the solution after OSQP converges. It improves accuracy at the cost of a small extra computation.
- **Tolerances** of 1e-4 are sufficient for control purposes. Tighter tolerances increase iteration count without meaningful improvement in tracking.

When OSQP is unavailable, scipy's `minimize` with `method='SLSQP'` serves as a fallback. It is slower but does not require additional dependencies.

### Solve Time

Typical solve times for the G1 in double support:
- OSQP with warm start: 0.5 - 2.0 ms
- OSQP cold start: 3 - 8 ms
- scipy SLSQP fallback: 10 - 30 ms

At 500 Hz (2 ms timestep), OSQP with warm starting leaves sufficient margin. The scipy fallback is too slow for real-time but useful for debugging and validation.

## Implementation Details

### QPResult Dataclass

```python
@dataclass
class QPResult:
    qdd: np.ndarray           # (nv,) generalized accelerations
    tau: np.ndarray            # (na,) actuated joint torques
    f_contacts: list[np.ndarray]  # list of (3,) contact forces
    solve_time_ms: float
    status: str                # "solved", "infeasible", "max_iter"
```

### WholeBodyQP Interface

```python
class WholeBodyQP:
    def __init__(self, model: G1WholeBodyModel,
                 torque_limits: np.ndarray,
                 friction_mu: float = 0.7) -> None:
        ...

    def solve(self, q: np.ndarray, v: np.ndarray,
              tasks: list[Task],
              contacts: list[ContactInfo]) -> QPResult:
        ...
```

The `solve()` method:
1. Evaluates M(q), h(q,v), and all task Jacobians via the G1 model
2. Assembles the QP cost from weighted task residuals
3. Builds the dynamics equality constraint
4. Builds friction cone inequality constraints for active contacts
5. Adds torque and acceleration box constraints
6. Calls OSQP (warm-started from the previous solution)
7. Extracts qdd, tau, f_c from the solution vector
8. Returns `QPResult`

### Regularization

A small identity term `eps * ||qdd||^2` (with eps ~ 1e-6) is added to the cost. This serves two purposes:
1. Ensures the Hessian is strictly positive definite even when task Jacobians are rank-deficient
2. Biases the solution toward small accelerations in the null space of all tasks

Without regularization, the QP can produce arbitrarily large accelerations in null-space directions, leading to numerical instability.

## Standing + Reaching Demo

The `a1_standing_reach.py` demo validates the QP controller with a simple scenario:

1. G1 starts in the nominal standing configuration (double support)
2. After 2 seconds of quiet standing, the left hand target moves 20 cm forward and 10 cm to the left
3. The QP resolves the conflict: CoM shifts to compensate for the arm motion, feet stay planted, hand tracks the target

### What Happens Internally

When the hand target moves, the HandPoseTask error becomes nonzero. The QP computes joint accelerations that move the hand toward the target. But moving the arm shifts the CoM, which creates a CoMTask error. The CoM task has 100x higher weight, so the QP prioritizes maintaining CoM over hand tracking. The solution is to shift the torso slightly to compensate for the arm's displacement while simultaneously extending the arm.

The PostureTask acts as a tiebreaker: among all configurations that satisfy the high-priority tasks equally well, the posture task selects the one closest to the nominal standing pose.

### Results

- **Hand tracking error**: < 2 cm after convergence (within 1.5 seconds of target change)
- **CoM displacement**: < 3 cm from nominal (shifts to compensate for arm extension)
- **Foot displacement**: < 1 mm (feet remain planted)
- **QP solve time**: 0.8 - 1.5 ms average

### Plots

The demo generates four plots saved to `media/`:
1. CoM trajectory (top-down view with support polygon)
2. Left hand tracking error over time
3. Joint torque profiles (heatmap across all actuated joints)
4. ZMP position relative to support polygon boundary

## What to Study

1. **Task weights matter.** Change the CoM weight from 1000 to 10 and observe the robot lose balance. The weight ratio directly controls how aggressively each task is pursued.

2. **Friction cone geometry.** Visualize the linearized cone and understand why 4 facets are sufficient for flat ground but not for rough terrain.

3. **Dynamics constraint.** Remove the equation of motion constraint and observe that torques and accelerations become inconsistent — the controller produces motions that violate Newton's laws.

4. **Warm starting.** Disable warm starting and measure the increase in solve time. This demonstrates why real-time QP requires exploiting temporal coherence.

5. **Regularization sensitivity.** Increase epsilon from 1e-6 to 1e-2 and observe how the solution becomes more conservative (smaller accelerations, slower convergence to targets).

## Next Step

Move to `02_walking_manipulation.md` to see how the QP controller integrates with the gait generator for walking while simultaneously tracking arm targets.
