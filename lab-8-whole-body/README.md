# Lab 8: Whole-Body Loco-Manipulation

A Unitree G1 humanoid walks to a table, picks up an object, and carries it
while walking — all controlled by a task-priority whole-body QP.

This lab merges everything from Labs 1–7: kinematics, dynamics, force control,
grasping, coordination, and locomotion into a single whole-body controller.
The robot must simultaneously maintain balance, execute a walking gait, and
perform arm manipulation — resolving conflicts through quadratic programming
with strict task priorities.

---

## Architecture

```
┌──────────────────────────────────────┐
│       LocoManipStateMachine           │
│  IDLE → WALK → STABILIZE → REACH →   │
│  GRASP → LIFT → WALK_WITH_OBJ →      │
│  STABILIZE → PLACE → RELEASE → DONE  │
└────────────────┬─────────────────────┘
                 │ active tasks + contacts
                 ▼
┌──────────────────┐   ┌──────────────────────┐
│  GaitGenerator   │   │  Task Definitions    │
│  LIPM + ZMP      │   │  CoM · Feet · Hands  │
│  foot swing      │   │  Posture             │
└────────┬─────────┘   └──────────┬───────────┘
         └──────────┬─────────────┘
                    ▼
┌──────────────────────────────────────┐
│         WholeBodyQP (OSQP)           │
│                                      │
│  min Σ w_i ‖J_i·q̈ - a_d_i‖²       │
│  s.t. dynamics, friction, limits     │
│                                      │
│  Priority: balance > feet > hands    │
│            > posture                 │
└────────────────┬─────────────────────┘
                 │ joint torques
                 ▼
┌──────────────────────────────────────┐
│  Pinocchio (analytical brain)        │
│  FK, Jacobians, M(q), h(q,v), CoM   │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│  MuJoCo Simulation (physics engine)  │
│  G1 + ground + table + object        │
│  Contact dynamics, rendering         │
└──────────────────────────────────────┘
```

---

## Key Concepts

### Task-Priority QP
All objectives (balance, feet, hands, posture) are least-squares terms in a
single QP. Weight gaps (1000:100:10:1) enforce near-strict priority: balance
is never sacrificed for manipulation.

### CoM Compensation
When the robot picks up an object, the combined CoM shifts. The balance
controller adjusts the CoM target to keep the combined system (robot + object)
balanced above the support polygon.

### Gait + Manipulation Integration
The gait generator provides CoM and foot trajectories. The QP resolves these
with simultaneous hand tracking. Arms comply when leg motion demands it.

### Weld Constraint Grasp
Grasping uses a MuJoCo equality constraint (following Labs 5/6). Focus is
whole-body coordination, not grasp mechanics.

---

## Repository Structure

```
lab-8-whole-body/
├── src/
│   ├── lab8_common.py, g1_model.py, tasks.py, contact_model.py
│   ├── whole_body_qp.py, balance_controller.py, gait_generator.py
│   ├── loco_manip_fsm.py
│   ├── a1_standing_reach.py, a2_walk_fixed_arms.py, a3_walk_and_reach.py
│   ├── b1_loco_manip_pipeline.py, capstone_demo.py
├── models/                         # G1 MJCF/URDF + scene
├── tests/                          # 10 test files
├── docs/ + docs-turkish/           # 3 articles each
├── tasks/                          # PLAN, ARCHITECTURE, TODO, LESSONS
└── media/
```

---

## Dependencies

```
Python     >= 3.10
MuJoCo     >= 3.0
pinocchio  >= 2.6
numpy      >= 1.24
matplotlib >= 3.7
osqp       >= 0.6
scipy      >= 1.10
```

---

## Running

```bash
python src/a1_standing_reach.py       # Stand + reach with QP
python src/a2_walk_fixed_arms.py      # Walk with fixed arms
python src/a3_walk_and_reach.py       # Walk while reaching
python src/b1_loco_manip_pipeline.py  # Full loco-manipulation
python src/capstone_demo.py           # Capstone with metrics
pytest tests/ -v
```

---

## Results Summary

| Metric | Target | Actual |
|--------|--------|--------|
| FK cross-validation | < 1 mm | — |
| Standing balance (5N perturbation) | Recovers < 2s | — |
| Hand reach error (standing) | < 2 cm | — |
| Arm drift during walk | < 5 cm | — |
| Walking steps without falling | 10+ | — |
| QP solve time | < 2 ms | — |
| Capstone: full sequence | No falls, no drops | — |

---

## Connection to Prior Labs

| Lab | Pattern reused |
|-----|---------------|
| Lab 3 | Dynamics, gravity compensation, impedance |
| Lab 5 | GraspStateMachine, weld constraint |
| Lab 6 | Coordinated state machine |
| Lab 7 | LIPM, ZMP, gait generation, standing balance |

---

## What's Next

Lab 9 (VLA Integration) uses this lab's whole-body controller as the expert
demonstrator. The VLA policy learns to replicate the hand-coded pipeline from
vision and language inputs alone.
