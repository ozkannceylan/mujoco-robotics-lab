# Lab 7: Locomotion Fundamentals — Architecture

## 1. System Overview

Lab 7 transitions from fixed-base manipulation (Labs 1–6, UR5e) to floating-base locomotion on the **Unitree G1 humanoid** (29 actuated DOFs, 33.34 kg). The lab explores the fundamental building blocks of bipedal locomotion: standing balance, center-of-mass tracking, whole-body inverse kinematics, and ZMP-based walking.

### Platform

| Property | Value |
|----------|-------|
| Robot | Unitree G1 (MuJoCo Menagerie) |
| Total DOFs (nq) | 36 (7 freejoint + 29 hinge) |
| Velocity DOFs (nv) | 35 (6 base velocity + 29 joint velocity) |
| Actuated joints (nu) | 29 (12 leg + 3 waist + 14 arm) |
| Actuator type | Position servos (Kp=500, Kd=0) |
| Total mass | 33.34 kg |
| Standing pelvis height | 0.79 m (keyframe), 0.757 m (settled) |
| Standing CoM height | ~0.66 m |
| Simulation timestep | 0.002 s |

### Architecture Stack

```
┌──────────────────────────────────────────────────────────┐
│  Planner Layer                                           │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ ZMP Reference  │→│ LIPM Preview │→│ Swing Foot   │  │
│  │ Generator      │  │ Controller   │  │ Trajectory   │  │
│  └───────────────┘  └──────────────┘  └──────────────┘  │
├──────────────────────────────────────────────────────────┤
│  IK Layer                                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Whole-Body IK (Pinocchio)                        │    │
│  │ Tasks: feet 6D×2 + CoM XY + pelvis Z + pelvis R │    │
│  │ Method: Stacked Jacobian + DLS                   │    │
│  └──────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────┤
│  Control Layer                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Gravity-Compensated PD                           │    │
│  │ ctrl = q_target + qfrc_bias/Kp - K_vel*qvel/Kp  │    │
│  └──────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────┤
│  Simulation Layer                                        │
│  ┌────────────────────┐  ┌───────────────────────────┐   │
│  │ MuJoCo Engine      │  │ Pinocchio Analytics      │   │
│  │ (physics, contact, │  │ (FK, Jacobians, CoM,     │   │
│  │  rendering)        │  │  IK, frame placements)   │   │
│  └────────────────────┘  └───────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### What Works vs. What Doesn't

| Capability | Status | Milestone |
|-----------|--------|-----------|
| Gravity-compensated standing (10s) | PASS | M1 |
| 5N push recovery | PASS | M1 |
| CoM cross-validation (Pin vs MuJoCo) | 0.000mm error | M2 |
| Support polygon computation | PASS | M2 |
| Pinocchio Jacobian validation (finite diff) | 0/36 failures | M3a |
| FK cross-validation (10 random configs) | 0.000mm error | M3b |
| Whole-body IK (CoM shift + foot constraint) | <0.51mm foot slip | M3c |
| Weight shift in simulation (5cm lateral) | 53.5mm shift, <1.4mm drift | M3d |
| Dynamic walking (single step or multi-step) | FAILED | M3e |

---

## 2. Floating-Base Robotics Theory

### Why nq != nv

Fixed-base robots (Labs 1–6) have `nq = nv` because every joint uses a single angle with a single velocity. Floating-base robots break this symmetry because of the **quaternion representation**.

The G1's freejoint contributes:
- **7 elements to qpos**: 3 position (x, y, z) + 4 quaternion (w, x, y, z)
- **6 elements to qvel**: 3 linear velocity + 3 angular velocity

The quaternion has 4 components but only 3 degrees of freedom (it lies on the unit sphere S3). The angular velocity lives in the tangent space of SO(3), which is 3-dimensional. This creates the mismatch: `nq = 36` but `nv = 35`.

**Critical implication:** You cannot update configurations by simple addition (`q += dq`). Instead, you must use `pin.integrate(model, q, dq)`, which applies the velocity tangent vector to the configuration manifold — updating the quaternion on S3 via the exponential map.

### MuJoCo vs Pinocchio Conventions

| Convention | MuJoCo | Pinocchio |
|-----------|--------|-----------|
| Quaternion order | (w, x, y, z) | (x, y, z, w) |
| Base Z in qpos | World position | World position - 0.793 |
| Velocity space | Same | Same |
| Joint ordering | Same as MJCF | Same as MJCF |

The Z offset exists because the G1's MJCF defines `<body name="pelvis" pos="0 0 0.793">`. Pinocchio's `buildModelFromMJCF` with `FreeFlyer` bakes this offset into the joint placement, so `pelvis_world_z = pin_q[2] + 0.793`.

Conversion functions in `lab7_common.py`:
- `mj_qpos_to_pin()`: swaps quaternion and subtracts Z offset
- `pin_q_to_mj()`: inverse operation
- `mj_quat_to_pin()` / `pin_quat_to_mj()`: quaternion-only conversion

### Position Actuator Model

The Menagerie G1 uses position actuators:

```
tau = Kp * (ctrl - qpos) - Kd * qvel
```

With Menagerie defaults: Kp=500, Kd=0 (from `dampratio=1` with zero `dof_damping`).

**Zero Kd is a critical design constraint.** Without damping, the actuators are underdamped oscillators. Lab 7 adds explicit velocity damping:

```
ctrl = q_target + qfrc_bias[6+i] / Kp - K_VEL * qvel[6+i] / Kp
```

This yields effective torque:
```
tau = Kp*(q_target - qpos) + qfrc_bias - K_VEL*qvel
```

With K_VEL=40: damping ratio zeta = 40 / (2*sqrt(500)) = 0.89 (near-critical).

---

## 3. Pinocchio Analytical Pipeline

Pinocchio handles all analytical computation. MuJoCo handles simulation.

### FK and Jacobian Computation

```python
pin.computeJointJacobians(model, data, q)
pin.updateFramePlacements(model, data)
pin.centerOfMass(model, data, q)

# Foot Jacobians (6×nv, LOCAL_WORLD_ALIGNED)
J_lf = pin.getFrameJacobian(model, data, lf_id, pin.LOCAL_WORLD_ALIGNED)
J_rf = pin.getFrameJacobian(model, data, rf_id, pin.LOCAL_WORLD_ALIGNED)

# CoM Jacobian (3×nv)
J_com = pin.jacobianCenterOfMass(model, data, q)
```

**LOCAL_WORLD_ALIGNED** is mandatory. This gives velocities expressed in the world frame but at the frame origin — the correct choice for task-space IK with stacked Jacobians.

### Whole-Body IK

The IK solver (`m3c_static_ik.py`) uses a stacked Jacobian with damped least squares (DLS):

**Task stack (18 dimensions for free-base, 14 for fixed-base):**
1. Left foot position + orientation (6D)
2. Right foot position + orientation (6D)
3. CoM XY position (2D)
4. Pelvis height Z (1D) — free-base only
5. Pelvis orientation (3D) — free-base only

**DLS solution:**
```
J_stack = [J_lf; J_rf; J_com_xy; J_pz; J_pori]   (18×35)
dx_stack = [dx_lf; dx_rf; dx_com_xy; dx_pz; dx_pori]
dq = J^T (J J^T + lambda^2 I)^{-1} dx
q_new = pin.integrate(model, q, alpha * dq)
```

Parameters: lambda=0.01, alpha=0.3, dq_max=0.1 rad/iter.

**Key insight:** The pelvis Jacobian is only nonzero for base DOFs (indices 0–5). This means pelvis tasks constrain the floating base while leg joints remain free for foot/CoM tasks — the tasks cooperate rather than compete.

### Cross-Validation Results

All Pinocchio-MuJoCo comparisons show machine-precision agreement:

| Quantity | Method | Max Error |
|----------|--------|-----------|
| Foot FK (10 random configs) | Position comparison | 0.000 mm |
| CoM position | subtree_com vs pin.com | 0.000 mm |
| CoM Jacobian columns | Central finite differences | 1.03e-08 |
| Foot Jacobian columns | Central finite differences | 1.09e-07 |

---

## 4. LIPM and ZMP Preview Control Theory

### Linear Inverted Pendulum Model

The LIPM models the robot as a point mass at constant height z_c, connected to the ground by a massless leg. Under the constant-height constraint:

```
x_ddot = (g / z_c) * (x_com - x_zmp)
```

The **Zero-Moment Point (ZMP)** is where the net moment of gravity plus inertial forces is zero. During single support, the ZMP must lie within the stance foot. During double support, it can be anywhere in the convex hull of both feet.

The LIPM decouples into independent X and Y dynamics, making the problem tractable.

### Preview Control (Kajita 2003)

The cart-table model is discretized with jerk (third derivative) as input:

```
State: [x, x_dot, x_ddot]
A = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]
B = [[dt^3/6], [dt^2/2], [dt]]
C = [1, 0, -z_c/g]   (ZMP output: p = x - (z_c/g)*x_ddot)
```

The system is augmented with an integrator on ZMP tracking error, then a discrete Riccati equation is solved to get:

- **Ki**: integral gain on cumulative ZMP error
- **Gx**: state feedback gain [1×3]
- **Gd[j]**: preview gains for future ZMP references

The control law looks 1.6s ahead (80 preview steps × 0.02s):
```
u(k) = -Ki * sum(e) - Gx @ state(k) + sum(Gd[j] * zmp_ref(k+j))
```

**Critical property:** Preview control starts moving the CoM BEFORE the support switches. The CoM "leads" the ZMP transition, which is essential for dynamic walking — the CoM must be traveling toward the next support before the current support phase ends.

### Footstep Plan and ZMP Reference

The footstep planner generates alternating left/right steps with configurable stride and timing:

```
Timeline:
  [0, t_init)              — Initial DS: ZMP moves from midpoint to first stance
  For each step i:
    [t_start, t_start+t_ss) — SS: ZMP at stance foot center
    [t_ss, t_ss+t_ds)       — DS: ZMP interpolates to next stance
  Final SS + Final DS       — ZMP returns to midpoint
```

Timing parameters: t_ss=0.8s (single support), t_ds=0.2s (double support), t_init=t_final=1.0s.

### Swing Foot Trajectory

During single support, the swing foot follows:
- **Horizontal (X, Y):** Cubic smooth interpolation: `3s^2 - 2s^3`
- **Vertical (Z):** Parabolic bump: `4 * step_height * s * (1-s)`, peak at s=0.5

This gives zero velocity at start and end (smooth liftoff and landing).

---

## 5. Walking Failures — Honest Analysis

### What We Attempted

After successfully implementing the entire LIPM + ZMP + IK pipeline (all modules verified independently), we attempted to execute a single forward step in MuJoCo simulation. Six distinct approaches were tried:

| Attempt | Key Change | Result | Fall Time |
|---------|-----------|--------|-----------|
| 1 | Basic IK + PD (Kp=400, t_ss=1.5s) | FELL | 3.64s |
| 2 | +Feedforward + ankle feedback | FELL | 3.79s |
| 3 | Higher gains (Kp=800), shorter SS (0.5s) | First SS survived, fell on alignment | 3.78s |
| 4 | Skip alignment step, keep both feet planted | FELL | 4.04s |
| 5 | Custom manual ZMP reference, no alignment | FELL (worse) | 2.84s |
| 6 | Online IK with measured base state | FELL | 3.04s |

### Root Cause Analysis

The failure is not in any single component — each module (LIPM planner, ZMP reference, IK solver, swing trajectory) works correctly in isolation. The failure lies in the **execution gap** between kinematic planning and dynamic simulation.

**The fundamental problem: Position actuators cannot provide ZMP control.**

ZMP-based walking requires precise control of the ground reaction forces, specifically the point where the net moment is zero. This requires controlling **ankle torques** — the primary mechanism for shifting the ZMP within the stance foot.

Position actuators (`tau = Kp*(q_ref - q) - Kd*qvel`) track joint angles, not torques. When the robot enters single support:

1. The IK planner computes joint angles assuming perfect tracking
2. The position actuators track those angles with some lag and error
3. Tracking errors create unintended torques, especially at the ankles
4. The unintended ankle torques shift the actual ZMP away from the planned ZMP
5. The CoM accelerates in the wrong direction (LIPM: `x_ddot ∝ x_com - x_zmp`)
6. The error compounds — within 0.5–1.5s, the robot falls

This is a **structural limitation**, not a tuning problem. No amount of gain adjustment can fix it because the control architecture cannot directly command ankle torques.

### What Would Fix It

1. **Torque-controlled actuators**: Replace position servos with direct torque commands. This requires a different MuJoCo actuator model and a whole-body dynamics controller (inverse dynamics, QP-based).

2. **Reinforcement learning**: Train a policy that maps state to position actuator commands. RL can implicitly learn the mapping from desired ZMP to position commands through trial and error, compensating for the actuator dynamics.

3. **Model Predictive Control (MPC)**: Predict the actuator dynamics forward and optimize position commands that achieve the desired torque profile. More complex but principled.

4. **Sim-to-real frameworks (e.g., Isaac Lab)**: Use GPU-parallelized environments to train robust walking policies. This is the current industry standard for humanoid locomotion.

### The Positive Side

Everything up to and including weight shift (M3d) works reliably:
- Gravity-compensated standing is rock-solid (1.6mm deviation over 10s)
- Push recovery from 5N lateral force works
- Pinocchio IK produces correct joint configurations for arbitrary CoM targets
- Pre-computed IK trajectories execute smoothly in simulation (53.5mm CoM shift, <1.4mm foot drift)

The failure occurs specifically when the robot must balance on one foot — the transition from static to dynamic balance. This is the core challenge of bipedal locomotion and explains why walking is one of the hardest problems in robotics.

---

## 6. Lessons Learned

### Lesson 1: The Pinocchio-MuJoCo Split is Powerful

Using Pinocchio for all analytical computation (FK, Jacobians, IK, CoM) and MuJoCo for physics simulation is the right architecture. With correct quaternion and Z-offset conversion, the two engines agree to machine precision. This gave us complete confidence in the kinematic pipeline and let us isolate dynamic execution as the failure mode.

### Lesson 2: Pre-computed IK >> Online IK for Position Actuators

Five attempts at online IK feedback (updating IK targets every timestep based on measured state) all failed. The kinematic Jacobian predicts different CoM-to-joint sensitivity than what MuJoCo dynamics produces, because servo torque reaction on the pelvis dominates over kinematic chain prediction.

Pre-computing the entire IK trajectory offline at N waypoints, then replaying with PD tracking, is far more robust. The key insight: for quasi-static motions, you don't need feedback on the IK layer — the PD controller handles small tracking errors, and gravity compensation keeps the robot upright.

### Lesson 3: Velocity Damping is the Most Important Parameter

The Menagerie G1 actuators have Kd=0. Without explicit velocity damping (K_VEL=40, giving zeta=0.89), the robot oscillates and falls. This single parameter change — adding `ctrl -= K_VEL * qvel / Kp` — was the difference between success and failure for weight shift.

### Lesson 4: Validate Jacobians with Central Finite Differences

Forward differences (O(eps)) gave false alarms for large-lever-arm joints. Central differences (O(eps^2)) brought errors to 1e-7, confirming the Jacobians are correct. This validation should be the first step for any new floating-base model.

### Lesson 5: The Execution Gap is Real

The gap between "kinematically correct" and "dynamically stable" is enormous for bipedal locomotion. Every robotics textbook teaches ZMP + LIPM + IK as the walking pipeline, but executing it on position-actuated servos in simulation reveals that the pipeline assumes perfect torque control. This assumption is invisible in the theory but fatal in practice.

### Lesson 6: Honest Documentation Matters

Documenting failures is as valuable as documenting successes. The walking failure taught us more about humanoid control than all the successful milestones combined. It motivated a deep understanding of why torque control matters, why RL has become dominant for locomotion, and what Lab 8+ will need to address.
