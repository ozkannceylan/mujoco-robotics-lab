# Grasp Pipeline — Lab 5

## Overview

Lab 5 implements a complete pick-and-place pipeline: the UR5e arm picks a 40mm cube from position A (world [0.35, +0.20, 0.335]) and places it at position B (world [0.35, -0.20, 0.335]). The pipeline integrates:

- **Pinocchio**: FK and Damped Least Squares (DLS) IK for computing arm configurations
- **Lab 4 RRT\***: collision-free motion planning between configurations
- **Lab 4 TOPP-RA**: time-optimal trajectory parameterization
- **Lab 3 impedance control**: smooth trajectory execution with joint-space stiffness
- **Lab 3 gravity compensation**: accurate torque feedforward to eliminate steady-state error

---

## State Machine

The `GraspStateMachine` class progresses through 11 states:

```
IDLE
  → PLAN_APPROACH   (RRT* from Q_HOME to q_pregrasp)
  → EXEC_APPROACH   (joint-space impedance tracking)
  → DESCEND         (Cartesian impedance descent to q_grasp)
  → CLOSE           (gripper closes, wait for settlement)
  → LIFT            (Cartesian impedance ascent back to q_pregrasp)
  → PLAN_TRANSPORT  (RRT* from q_pregrasp to q_preplace)
  → EXEC_TRANSPORT  (joint-space impedance tracking)
  → DESCEND_PLACE   (Cartesian impedance descent to q_place)
  → RELEASE         (gripper opens, wait for settlement)
  → RETRACT         (RRT* + impedance back to Q_HOME)
  → DONE
```

---

## IK Solver

**File:** `src/grasp_planner.py :: compute_ik()`

The IK solver uses **Damped Least Squares (DLS)**:

```
Δq = α · Jᵀ (J Jᵀ + λ² I)⁻¹ · e
```

Where:
- **J** (6×6): frame Jacobian in world-aligned local frame (`pin.getFrameJacobian`)
- **e** (6,): task-space error = [Δposition, Δorientation] — orientation error extracted from skew-symmetric R_err
- **α = 0.5**: step size (stability)
- **λ² = 1e-4**: damping (avoids singularity blow-up)
- **tol = 1e-4**: convergence threshold on ‖e‖
- **max_iter = 300**: iteration budget

The solver clamps each step to joint limits via `pin.integrate()` + `np.clip()`.

---

## Grasp Configurations

**File:** `src/grasp_planner.py :: compute_grasp_configs()`

Five joint configurations are computed offline before the pick-and-place cycle:

| Config | Tool0 target (world frame) | Purpose |
|--------|---------------------------|---------|
| `q_home` | — (Q_HOME directly) | Rest pose |
| `q_pregrasp` | BOX_A + [0, 0, 0.090 + 0.15] | Approach above box A |
| `q_grasp` | BOX_A + [0, 0, 0.090] | Fingertips at box A center |
| `q_preplace` | BOX_B + [0, 0, 0.090 + 0.15] | Transport hover above B |
| `q_place` | BOX_B + [0, 0, 0.090] | Fingertips at box B center |

The `GRIPPER_TIP_OFFSET = 0.090 m` accounts for the distance from tool0 origin to fingertip center. The `PREGRASP_CLEARANCE = 0.150 m` gives 15 cm of clearance above the box for the approach.

All configurations use the same fixed **top-down orientation** `R_topdown`, computed from FK at Q_HOME. This keeps the gripper Z axis pointing world -Z throughout, which is the natural approach direction for tabletop grasping.

---

## Motion Planning (Lab 4 integration)

**File:** `src/grasp_state_machine.py :: _plan_and_smooth()`

Each `PLAN_*` state calls the Lab 4 pipeline:

```python
# 1. RRT* collision-free path (6000 iterations)
path = rrt_planner.plan(q_start, q_goal, max_iter=6000, rrt_star=True)

# 2. Shortcutting (200 iterations, removes redundant waypoints)
path = shortcut_path(path, collision_checker, max_iter=200)

# 3. TOPP-RA time parameterization (returns times, q, qd, qdd)
times, q_traj, qd_traj, _ = parameterize_topp_ra(path, VEL_LIMITS, ACC_LIMITS)
```

**Collision checker:** Reuses Lab 4's `CollisionChecker` with only the table obstacle. No cluttered scene objects — the open-field assumption is valid for a clean tabletop.

**Velocity / acceleration limits:**
```python
VEL_LIMITS = [3.14, 3.14, 3.14, 6.28, 6.28, 6.28]  # rad/s
ACC_LIMITS  = [8.0,  8.0,  8.0, 16.0, 16.0, 16.0]  # rad/s²
```

---

## Trajectory Execution (Lab 3 integration)

### Joint-Space Impedance (EXEC_* states)

```python
τ = Kp(q_d - q) + Kd(qd_d - qd) + g(q)
```

- **Kp = diag([200, 200, 200, 100, 100, 100])** N·m/rad — joints 1–3 stiffer
- **Kd = 2√Kp** (critically damped)
- **g(q)**: Pinocchio RNEA gravity vector

The reference (q_d, qd_d) is read from the TOPP-RA trajectory at each simulation step.

### Cartesian Impedance (DESCEND, LIFT, DESCEND_PLACE)

Vertical descents and lifts use Lab 3's `compute_impedance_torque()` with a Cartesian stiffness matrix. Only the Z-axis displacement is commanded; X and Y are regulated to maintain alignment.

---

## Data Flow

```
compute_grasp_configs()   ← Pinocchio FK + DLS IK
         ↓ GraspConfigs (5 × q vectors)
GraspStateMachine.__init__()
         ↓ state transitions
_plan_and_smooth()        ← Lab 4 RRT* + TOPP-RA
         ↓ (times, q_traj, qd_traj)
_run_joint_impedance()    ← Lab 3 gravity comp + PD
         ↓ ctrl torques → MuJoCo mj_step()
         ↓ qpos, qvel, contact forces logged
```

---

## Failure Modes

| Mode | Symptom | Cause | Mitigation |
|------|---------|-------|------------|
| IK divergence | RuntimeError "IK failed" | Target unreachable | Check GRIPPER_TIP_OFFSET sign |
| RRT* timeout | RuntimeError "RRT* failed" | Narrow passage or bad limits | Increase max_iter, check joint limits |
| Box slips | State machine freezes in LIFT | Low friction / gripper gap too large | Increase μ_slide or reduce minimum gap |
| Box collides during transport | `data.ncon > 0` mid-flight | IK config has arm-table collision | Add Y-offset to keep arm clear of table edge |
