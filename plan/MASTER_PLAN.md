# Robotics Lab — Master Plan

> **Goal:** Progressive robotics curriculum from first principles to VLA-controlled humanoid manipulation  
> **Stack:** MuJoCo · Pinocchio · ROS2 · Python  
> **Author:** M. Ozkan Ceylan

---

## Vision

A structured, hands-on lab series that builds robotics competency from the ground up. Each lab produces working code, rigorous documentation, and a blog post. The series begins with planar kinematics and ends with a VLA-controlled humanoid performing manipulation tasks from natural language commands.

```
Lab 1          Lab 2          Lab 3          Lab 4          Lab 5
2-Link      →  6-DOF Arm   →  Dynamics &  →  Motion      →  Grasping &
Planar         DH/Pinocchio   Force Ctrl     Planning       Manipulation
                                                              │
Lab 9          Lab 8          Lab 7          Lab 6          ◄─┘
VLA          ← Whole-Body  ← Locomotion  ← Dual-Arm
Integration    Loco-Manip     Fundamentals   Coordination
```

---

## Lab Summary

| Lab | Title | Capstone Demo |
|-----|-------|---------------|
| 1 | 2-Link Planar Robot | Draw a square with end-effector |
| 2 | 6-DOF Robot Arm (DH & Pinocchio) | Draw a cube in 3D space |
| 3 | Dynamics & Force Control | Constant-force surface contact |
| 4 | Motion Planning & Collision Avoidance | Obstacle-free trajectory in cluttered scene |
| 5 | Grasping & Manipulation | Pick and place an object |
| 6 | Dual-Arm Coordination | Two arms carry an object together |
| 7 | Locomotion Fundamentals | Stable bipedal walking on flat ground |
| 8 | Whole-Body Loco-Manipulation | Walk while carrying an object |
| 9 | VLA Integration | "Pick up the red cup" — end-to-end language-to-action |

---

## Progression Logic

```
POSITION CONTROL (Labs 1-2)
  Command WHERE the robot goes.
      │
      ▼
FORCE CONTROL (Lab 3)
  Command HOW the robot interacts.
      │
      ▼
PLANNING (Lab 4)
  Find PATHS through obstacles.
      │
      ▼
MANIPULATION (Lab 5)
  Combine position + force + planning to do useful work.
      │
      ▼
COORDINATION (Lab 6)
  Scale from one arm to two — bridge to humanoid upper body.
      │
      ▼
LOCOMOTION (Lab 7)
  Leave the fixed-base world. Floating base changes everything.
      │
      ▼
INTEGRATION (Labs 8-9)
  Locomotion + manipulation + perception + language.
```

---

## Platform Transitions

| Labs | Robot | Rationale |
|------|-------|-----------|
| 1 | Custom 2-link planar | Minimal complexity, focus on math |
| 2–5 | UR5e + Robotiq 2F-85 (MuJoCo Menagerie) | Industry-standard fixed-base manipulation stack |
| 6 | Dual UR5e or G1 upper body | Transition point |
| 7–9 | Unitree G1 | Full humanoid with Dex3 hands |

### Platform Lock

For Labs 2–5, the robot baseline is fixed:

- Use the MuJoCo Menagerie `universal_robots_ur5e` model.
- Use the MuJoCo Menagerie `robotiq_2f85` gripper for manipulation labs.
- Do not replace Labs 3–5 with simplified/custom UR5e kinematics or a custom gripper as the primary implementation path.
- If a temporary simplified model is ever used for debugging, it must be clearly labeled as a temporary prototype and not counted as lab completion.

---

## Repo Structure

```
robotics-lab/
├── MASTER_PLAN.md
├── README.md
├── blog/
│   ├── lab_01_planar_robot.md
│   ├── lab_02_6dof_arm.md
│   └── ...
├── lab_01_planar_robot/
│   ├── README.md
│   ├── docs/
│   │   └── LAB_01.md
│   ├── src/
│   ├── models/
│   ├── config/
│   └── media/
├── lab_02_6dof_arm/
│   └── (same structure)
├── ...
├── shared/
│   ├── utils/
│   └── visualization/
└── requirements.txt
```

---

## Deliverables Per Lab

Each lab produces three artifacts:

1. **Code** — Working implementation under `lab_XX/src/`
2. **Documentation** — Technical writeup in `lab_XX/docs/LAB_XX.md`
3. **Blog Post** — Public-facing article in `blog/`

---

## Documentation Template

Every `LAB_XX.md`:

```
# Lab XX: [Title]
## Objectives
## Prerequisites
## Theory (math, diagrams)
## Implementation (architecture, design decisions)
## Results (demo, plots, metrics)
## Lessons Learned
## References
```

Every blog post:

```
# [Title]
## Context
## The Approach
## Key Insight
## Results
## What's Next
```

---

## Timeline Estimate

| Phase | Labs | Duration | Notes |
|-------|------|----------|-------|
| Foundations | 1–2 | ✅ Complete | |
| Control & Planning | 3–4 | ~2 weeks | |
| Manipulation | 5–6 | ~2 weeks | |
| Locomotion | 7–8 | ~3 weeks | Paradigm shift |
| VLA | 9 | ~2 weeks | Builds on humanoid_vla |

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| Contact physics instability (Lab 5) | condim/solref tuning; simpler grasp primitives as fallback |
| Bipedal walking divergence (Lab 7) | Start with standing balance; use reference trajectories |
| G1 model complexity (Labs 7–8) | Prototype on simplified biped first |
| GPU memory for VLA (Lab 9) | Cloud GPU for training; INT8 for local inference |
| Scope creep | Hard-scoped capstone demo per lab — ship when demo works |
