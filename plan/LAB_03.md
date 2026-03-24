# Lab 3: Dynamics & Force Control

> **Status:** Not Started  
> **Prerequisites:** Lab 1 (FK/IK), Lab 2 (6-DOF, Pinocchio)  
> **Platform:** MuJoCo Menagerie UR5e + mounted Robotiq 2F-85, analyzed with Pinocchio  
> **Capstone Demo:** End-effector maintains constant force against a surface while tracing a path

---

## Objectives

1. Understand the manipulator equations of motion: `M(q)q̈ + C(q,q̇)q̇ + g(q) = τ`
2. Implement gravity compensation using Pinocchio RNEA
3. Build a Cartesian impedance controller (spring-damper behavior)
4. Demonstrate compliant contact — the robot touches a surface gently, not crashes into it

---

## Why This Lab Matters

Labs 1–2 were purely kinematic — commanding positions. Real robots deal with forces. Without dynamics and force control, you cannot grasp objects (Lab 5), coordinate arms under load (Lab 6), or balance a walking humanoid (Lab 7). This is the single most important conceptual leap in the series.

---

## Theory Scope

- Rigid-body dynamics: inertia matrix `M(q)`, Coriolis `C(q,q̇)`, gravity `g(q)`
- Recursive Newton-Euler Algorithm (RNEA) — how Pinocchio computes inverse dynamics
- Gravity compensation: `τ = g(q)` makes the robot "float"
- Impedance control: `F = K(x_d - x) + D(ẋ_d - ẋ)` — virtual spring-damper
- Cartesian-to-joint torque mapping: `τ = J^T · F + g(q)`

---

## Architecture

```
Task Spec (desired pose + desired force)
        │
        ▼
┌──────────────────────┐
│  Impedance Controller │
│  F = K·Δx + D·Δẋ     │
│  + F_desired          │
└──────────┬───────────┘
           │ Cartesian force
           ▼
┌──────────────────────┐
│  Torque Computation   │
│  τ = J^T·F + g(q)    │
│  (Pinocchio RNEA)    │
└──────────┬───────────┘
           │ joint torques
           ▼
┌──────────────────────┐
│  MuJoCo Simulation    │
│  Torque-mode actuation│
│  → q, q̇, contacts    │
└──────────────────────┘
```

---

## Implementation Phases

### Phase 1 — Dynamics Fundamentals
- Start from the MuJoCo Menagerie UR5e model and preserve its real robot geometry
- Mount the Robotiq 2F-85 so Labs 3–5 share the same hardware baseline
- Switch the arm actuation path to a torque-level control workflow without replacing the robot model
- Compute and visualize `M(q)`, `C(q,q̇)`, `g(q)` using Pinocchio
- Implement gravity compensation — robot holds any pose hands-free
- Validate: perturb the arm, it should stay in place

### Phase 2 — Cartesian Impedance Controller
- Implement task-space impedance with tunable K and D matrices
- Test: command a Cartesian target, arm moves compliantly (not rigidly)
- Vary stiffness to show soft vs. stiff behavior
- Plot: position tracking error vs. stiffness

### Phase 3 — Force Control & Contact
- Add a flat surface (table) to the MuJoCo scene
- Implement hybrid position-force control: position in XY, force in Z
- Capstone: end-effector descends to surface, applies constant 5N force, traces a line

### Phase 4 — Documentation & Blog
- Write LAB_03.md with theory derivations, architecture diagrams, results
- Write blog post focusing on "why force control matters for manipulation"
- Record capstone demo video/gif

---

## Key Design Decisions for Claude Code

- **Torque mode is critical.** The MJCF actuator tags must change from `position` to `torque`. Without this, MuJoCo's internal PD controller masks the dynamics.
- **Use the real robot model.** Lab 3 must run on the Menagerie UR5e geometry, not a simplified/custom UR5e surrogate. Any Pinocchio model must match the MuJoCo robot actually used in simulation.
- **Use Pinocchio for dynamics, MuJoCo for simulation.** Pinocchio computes `M`, `C`, `g`, `J`. MuJoCo applies torques and steps physics. Keep them separate.
- **Impedance gains matter.** Start with low stiffness (K ~ 100–500 N/m) and increase. Too high = instability. Too low = no tracking.
- **Contact parameters.** MuJoCo's `solref` and `solimp` on the table surface affect how "real" the contact feels. Document the tuning.

---

## Success Criteria

- [ ] Robot holds position under gravity with zero position command (gravity comp only)
- [ ] Impedance controller tracks Cartesian waypoints with tunable compliance
- [ ] End-effector makes stable contact with surface at commanded force (±1N)
- [ ] Capstone demo: constant-force line tracing on table surface
- [ ] LAB_03.md complete with theory + architecture + results
- [ ] Blog post published

---

## References

- Siciliano et al., *Robotics: Modelling, Planning and Control* — Ch. 7 (Dynamics), Ch. 9 (Force Control)
- Pinocchio docs: RNEA, computeJointJacobians, forwardKinematics
- MuJoCo docs: Actuator model, contact parameters (solref, solimp)
