# Lab 7: Locomotion Fundamentals

A Unitree G1 humanoid walks on flat ground using ZMP-based gait planning
and whole-body inverse kinematics. This lab builds the locomotion foundation
that Labs 8 and 9 depend on.

---

## Architecture

```
FootstepPlanner
  │  footstep positions, ZMP reference, foot trajectories
  ▼
LIPMPlanner (Preview Control)
  │  smooth CoM trajectory from ZMP reference
  ▼
WalkingController (Whole-Body IK)
  │  Pinocchio damped least-squares IK
  │  Inputs: CoM(t), left_foot(t), right_foot(t)
  │  Output: q_desired(t)
  ▼
MuJoCo Position Servo + Feedforward
  │  ctrl = q_des + qfrc_bias/Kp + Kd*qd_des/Kp
  ▼
MuJoCo Simulation (500 Hz)
  │  Step physics → read joint state → loop
  └─→ loop
```

**Key principle:** Pinocchio computes all kinematics and dynamics analytically.
MuJoCo runs the physics simulation. Cross-validation between the two
confirms correctness.

---

## Key Concepts

### Linear Inverted Pendulum Model (LIPM)
Models the humanoid as a point mass at constant height above the ground.
Reduces the complex dynamics to a simple 2D linear system, making CoM
trajectory planning tractable.

### ZMP Preview Control (Kajita 2003)
Given a ZMP reference trajectory (derived from footstep positions), preview
control computes a smooth CoM trajectory that tracks the ZMP with a
look-ahead window. This guarantees dynamic balance — the ZMP stays inside
the support polygon.

### Whole-Body Inverse Kinematics
Damped least-squares IK converts task-space targets (CoM position, foot
poses) into joint angles. Task priority: CoM tracking > foot placement >
posture regulation.

### Gravity Feedforward for Menagerie Servos
MuJoCo Menagerie position servos suffer gravity droop. The fix:
`ctrl = q_des + qfrc_bias/Kp + Kd*qd_des/Kp` compensates for gravity
and velocity lag.

---

## Repository Structure

```
lab-7-locomotion/
├── src/
│   ├── lab7_common.py              # Paths, constants, joint mappings
│   ├── g1_model.py                 # G1 Pinocchio wrapper (FK, CoM, IK)
│   ├── balance_controller.py       # Standing balance (CoM PD)
│   ├── lipm_planner.py             # LIPM + preview control
│   ├── footstep_planner.py         # Footstep sequence + foot trajectories
│   ├── walking_controller.py       # Whole-body IK for walking
│   ├── a1_standing_balance.py      # Demo: standing + perturbation
│   ├── a2_zmp_planning.py          # Demo: ZMP trajectory visualization
│   ├── b1_walking_demo.py          # Demo: 10+ step walking
│   └── capstone_demo.py            # Capstone with metrics
│
├── models/
│   ├── g1_humanoid.xml             # Simplified G1 MJCF (~23 DOF)
│   ├── g1_humanoid.urdf            # G1 URDF for Pinocchio
│   └── scene_flat.xml              # G1 on flat ground
│
├── tests/                          # 4 test files
├── docs/                           # English docs (3 articles)
├── docs-turkish/                   # Turkish docs (3 articles)
├── tasks/                          # PLAN, ARCHITECTURE, TODO, LESSONS
└── media/                          # Videos, plots
```

---

## Dependencies

```
Python     >= 3.10
MuJoCo     >= 3.0
pinocchio  >= 2.6
numpy      >= 1.24
matplotlib >= 3.7
```

---

## Running

```bash
# Standing balance with perturbation recovery
python src/a1_standing_balance.py

# ZMP trajectory visualization
python src/a2_zmp_planning.py

# 10+ step walking demo
python src/b1_walking_demo.py

# Capstone demo with metrics and video
python src/capstone_demo.py

# Tests
pytest tests/ -v
```

---

## Results Summary

| Metric | Target | Actual |
|--------|--------|--------|
| FK cross-validation error | < 1 mm | — |
| Standing balance duration | 10s | — |
| Perturbation recovery | < 2s | — |
| Walking steps (flat ground) | 10+ | — |
| Max CoM deviation from plan | < 5 cm | — |
| ZMP within support polygon | 100% | — |

---

## Connection to Prior Labs

| Lab | Pattern reused |
|-----|---------------|
| Lab 3 | PD control, gravity compensation, Menagerie feedforward |
| All | Cross-validation pattern (Pinocchio vs MuJoCo) |

---

## What's Next

Lab 8 unlocks the arms and adds a whole-body QP controller for simultaneous
walking and manipulation. Lab 7's gait generator and LIPM planner are
reused as the locomotion backbone.
