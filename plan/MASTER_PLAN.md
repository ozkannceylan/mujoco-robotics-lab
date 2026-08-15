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

_Status refreshed 2026-08-13 (project review). This table is the single source of truth for lab status; the root README roadmap table mirrors it._

| Lab | Title | Capstone Demo | Status |
|-----|-------|---------------|--------|
| 1 | 2-Link Planar Robot | Draw a square with end-effector | ✅ Complete (no unit tests — see review note) |
| 2 | 6-DOF Robot Arm (DH & Pinocchio) | Draw a cube in 3D space | ✅ Complete (34 tests) |
| 3 | Dynamics & Force Control | Constant-force surface contact | ✅ Complete (34 tests; blog post never written) |
| 4 | Motion Planning & Collision Avoidance | Slalom through 4 obstacles (RRT* + TOPP-RA) | ✅ Complete (45 tests; blog post never written) |
| 5 | Grasping & Manipulation | Pick and place an object | ✅ Complete — Phase 5 + Step 6.1 both closed 2026-08-13; capstone places box 5.7 mm from target with transport post-condition |
| 6 | Dual-Arm Coordination | Two arms cooperatively carry an object (weld-constraint) | ✅ Complete (milestone-gated M0–M5; unit tests intentionally removed) |
| 7 | Locomotion Fundamentals | Standing balance + quasi-static weight shift (M0–M3d); ZMP walking documented as structurally infeasible with position actuators | ✅ Complete at M3d scope (34 tests; M4 blocked by design) |
| 8 | Whole-Body Loco-Manipulation | Walk while carrying an object | 🚧 In Progress — M0 + M1 PASS (torque G1; whole-body ID QP, 7.08 mm hand tracking); M2 stepping next |
| 9 | VLA Integration | "Pick up the red cup" — end-to-end language-to-action | 📋 Planned |

### Lab 8 dependency note (from Lab 7 outcome)

`plan/LAB_08.md` Phase 2 says "combine Lab 7's gait generator with the whole-body QP" —
**Lab 7 has no working gait generator.** M4 ZMP walking is blocked: MuJoCo Menagerie G1
position actuators cannot track the dynamic reference (IK converges, PD replay fails —
6 attempts, see `lab-7-locomotion/README.md` Scope Deferral). Lab 8 must treat gait
generation as *its own deliverable*, built on the torque-level inverse-dynamics path
(Pinocchio RNEA → joint torques) that LAB_08's architecture already prescribes. Lab 9's
data pipeline depends on Lab 8's controllers, so this is the critical path for the series.

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
| 6 | Dual UR5e (shipped) | Transition point — G1 upper-body option was not used |
| 7–9 | Unitree G1 | Full humanoid with Dex3 hands |

### Platform Lock

For Labs 2–5, the robot baseline is fixed:

- Use the MuJoCo Menagerie `universal_robots_ur5e` model.
- Use the MuJoCo Menagerie `robotiq_2f85` gripper for manipulation labs.
- Do not replace Labs 3–5 with simplified/custom UR5e kinematics or a custom gripper as the primary implementation path.
- If a temporary simplified model is ever used for debugging, it must be clearly labeled as a temporary prototype and not counted as lab completion.

---

## Repo Structure

_Updated 2026-08-13 to match the actual layout (the original planned layout with
`shared/`, root `blog/`, and `lab_XX_*` naming was never adopted)._

```
mujoco-robotics-lab/
├── README.md                      # Portfolio front page with roadmap/status table
├── CLAUDE.md / AGENTS.md          # Agent workflow rules
├── plan/                          # Lab briefs (LAB_01..09) + this master plan
├── tasks/                         # Project-level status board + Labs 1–2 lessons
├── tools/
│   └── video_producer.py          # Reusable 3-phase demo video pipeline
├── lab-1-2link-arm/               # src/, models/, docs/, docs-turkish/, media/, blog/, ros2_bridge/
├── lab-2-Ur5e-robotics-lab/       #   + tests/ (labs 2–5, 7)
├── lab-3-dynamics-force-control/  # Labs 3–7 additionally have tasks/{PLAN,ARCHITECTURE,TODO,LESSONS}.md
├── lab-4-motion-planning/
├── lab-5-grasping-manipulation/
├── lab-6-dual-arm/                # No tests/ — milestone-gated verification instead
├── lab-7-locomotion/
└── lab-8-loco-manipulation/       # In progress (M0+M1 done); torque G1 + whole-body ID QP
```

---

## Deliverables Per Lab

Each lab produces three artifacts:

1. **Code** — Working implementation under `lab-N-<name>/src/`
2. **Documentation** — Technical writeup in `docs/` (EN) + `docs-turkish/` (TR).
   _Convention drift note: the original `docs/LAB_XX.md` single-file convention was
   abandoned; Labs 1–2 use per-module notes (`a1_*.md`…), Labs 3–7 use
   `ARCHITECTURE.md` + `CODE_WALKTHROUGH.md`. Both are acceptable._
3. **Blog Post** — Public-facing article in the lab's `blog/` folder.
   _Status: written for Labs 1, 2, 5, 6, 7. **Missing for Labs 3 and 4** despite being
   a success criterion in their briefs — tracked in `tasks/todo.md`._

---

## Documentation Template

Every lab writeup should cover:

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

## Timeline

| Phase | Labs | Status | Notes |
|-------|------|--------|-------|
| Foundations | 1–2 | ✅ Complete (2026-03) | |
| Control & Planning | 3–4 | ✅ Complete (2026-03, published 2026-05) | |
| Manipulation | 5–6 | ✅ Complete (2026-03/05; Lab 5 fully closed 2026-08-13) | |
| Locomotion | 7 | ✅ Complete at M3d scope (2026-05) | M4 ZMP walking blocked → moved to Lab 8 |
| Whole-Body | 8 | 🚧 In progress — M0 + M1 done (2026-08-15) | Owns gait generation via torque control (M2–M3) |
| VLA | 9 | 📋 Not started | Builds on humanoid_vla; needs Lab 8 controllers for demo data |

---

## Risk Register

| Risk | Status / Mitigation |
|------|---------------------|
| Contact physics instability (Lab 5) | ✅ Handled — condim/solref tuning worked; see Lab 5 LESSONS.md |
| Bipedal walking divergence (Lab 7) | ⚠️ **Materialized, different root cause**: not divergence but the Menagerie G1 *position-actuator model* — PD replay cannot track dynamic ZMP references (6 attempts). Finding feeds Lab 8's torque-control design. |
| G1 model complexity (Labs 7–8) | Partially retired — G1 stack works for quasi-static tasks |
| GPU memory for VLA (Lab 9) | Open — cloud GPU for training; INT8 for local inference |
| Scope creep | Working — hard-scoped capstones shipped for 7 labs |
| Doc drift (new, 2026-08) | Tracking docs went stale while code advanced (Lab 5 Phase 5) or regressed silently (Lab 6 test removal). Mitigation: project-level status board in `tasks/todo.md`, refreshed at each review. |
