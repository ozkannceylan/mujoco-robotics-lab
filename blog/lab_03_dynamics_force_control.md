# Lab 3: From Position to Force — Why Contact Changes Everything

## The Leap from Kinematics to Dynamics

Labs 1 and 2 were about geometry: where is the arm, how do we move it to a target? But real manipulation isn't just about reaching a pose — it's about *interacting* with the environment. The moment the robot touches something, position control alone breaks down.

This is the conceptual leap of Lab 3: we need to control *forces*, not just positions.

## Why Position Control Fails at Contact

Consider a simple task: press the end-effector against a table with exactly 5 Newtons of force. A position controller would try to move the EE to a target *through* the table surface. The result? Either:
- The target is above the surface → no contact
- The target is below → the force depends on the stiffness of the table and the robot, not on what you commanded

The robot has no idea how hard it's pushing. It only knows *where* it is.

## The Dynamics Foundation

Before we can control forces, we need to understand the dynamics:

```
M(q)q̈ + C(q,q̇)q̇ + g(q) = τ
```

The mass matrix M(q) tells us how inertia distributes across joints. The gravity vector g(q) tells us what torque is needed just to hold position. The Coriolis matrix C(q,q̇) captures velocity-dependent effects.

Lab 3 starts here: computing these quantities with Pinocchio and cross-validating against MuJoCo. The agreement is excellent — gravity vectors match within 0.0005 Nm, mass matrices within 0.00004.

## Building Up: Gravity → Impedance → Force

The controller hierarchy is designed as layers:

**Layer 1: Gravity Compensation** — τ = g(q). The simplest dynamics controller. The arm floats in place, holding position with 0.0006 rad drift. This is the foundation everything else builds on.

**Layer 2: Impedance Control** — τ = J^T · (K·Δx + D·Δẋ) + g(q). The EE behaves like a virtual spring-damper. Tunable compliance: K=100 gives soft 104mm deflections, K=2000 gives stiff 17mm deflections under the same force.

**Layer 3: Hybrid Force/Position** — Position in XY, force in Z. This is where it gets interesting.

## The Hybrid Controller

The hybrid controller partitions the task space:
- XY: "go to this position" (impedance-like)
- Z: "apply this force" (PI controller with damping)

```
τ = J^T · (F_position + F_force) + g(q)
```

Getting this to work required solving several non-obvious problems:

**Contact force measurement**: MuJoCo's `<force>` sensor measures *all* constraint forces on a body, not just contact. We switched to `mj_contactForce()` per-contact pair.

**Contact chattering**: A naive PI force controller oscillated wildly (std=11N vs target 5N). The fix was three-fold: EMA force filtering (α=0.2), Z-velocity damping (K_dz=30), and anti-windup clamping.

**Collision filtering**: The wrist capsule geoms hit the table before the tool tip. MuJoCo's contype/conaffinity bitmasks isolate tool-table contact from accidental wrist contacts.

## The Capstone: Line Tracing Under Constant Force

The final demo ties everything together: the arm descends to a table, detects contact (>0.5N threshold), settles for 1.5 seconds, then traces a 50mm straight line while maintaining 5N of downward force.

Results:
- **Force**: 4.99N mean, 0.33N std, 98.1% within ±1N
- **Position**: 1.96mm mean XY error, 100% under 5mm

This is a real manipulation primitive — constant-force wiping, polishing, or surface inspection.

## What I Learned

1. **The Jacobian determines everything at contact.** In the vertical tool configuration, the Jacobian X-row is ~0.01. A 100N Cartesian force produces only 6Nm of joint torque. This fundamentally limits how fast you can move in XY while in contact.

2. **Dynamics cross-validation is worth the investment.** Rebuilding the URDF from scratch to match MuJoCo took time but paid off: every subsequent controller worked on the first try because the physics matched.

3. **Contact is hard.** The gap between "works in free space" and "works in contact" is enormous. Every assumption about smooth dynamics breaks at the contact boundary.

4. **Force filtering is not optional.** Raw contact forces in simulation are noisy. Real-world force/torque sensors are even noisier. Low-pass filtering and velocity damping are mandatory.

## Looking Ahead

Lab 3 gives us the ability to control *interaction* forces. Lab 4 will add motion planning and collision avoidance — moving safely through cluttered environments. Lab 5 combines both: grasping objects requires planning a path to the object, then switching to force control for the grasp itself.

The progression from kinematics → dynamics → force control → planning → grasping mirrors the real roadmap toward general manipulation.
