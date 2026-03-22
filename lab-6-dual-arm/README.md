# Lab 6: Dual-Arm Coordination

Two UR5e robots sharing a workspace, coordinating to pick up a large box,
carry it 30 cm across the table, and place it — without colliding with each
other and without dropping the object.

This lab builds on Labs 3–5. It introduces the object-centric frame
abstraction, three coordination modes (synchronized, master-slave, symmetric),
dual impedance control with internal force management, and a bimanual state
machine that orchestrates the full pick-carry-place pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                BimanualGraspStateMachine                 │
│   IDLE→APPROACH→PRE_GRASP→GRASP→LIFT→CARRY→LOWER        │
│   →RELEASE→RETREAT→DONE                                  │
└──────────────────────┬──────────────────────────────────┘
                       │ current state + targets
                       ▼
┌──────────────┐   ┌────────────────────┐   ┌─────────────────────┐
│ DualArmModel │──▶│ CoordinatedPlanner │──▶│ SynchronizedTraj    │
│              │   │                    │   │ q_left[T], q_right[T│
│ fk_left(q)   │   │ plan_synchronized  │   │ timestamps[T]       │
│ fk_right(q)  │   │ plan_master_slave  │   └──────────┬──────────┘
│ jacobian_*(q)│   │ plan_symmetric     │              │
│ ik_left(p)   │   └────────────────────┘              ▼
│ ik_right(p)  │                           ┌─────────────────────┐
└──────┬───────┘                           │ DualImpedanceCtrl   │
       │                                   │ tau = J^T*F + g(q)  │
       │   ┌──────────────────────┐        │ + squeeze force      │
       └──▶│ DualCollisionChecker │        └──────────┬──────────┘
           │ HPP-FCL: self, cross, │                  │ (tau_L, tau_R)
           │ env collision checks  │                  ▼
           └──────────────────────┘        ┌─────────────────────┐
                                           │   MuJoCo Simulation  │
                                           │ ctrl[:6]  = tau_L    │
                                           │ ctrl[6:12]= tau_R    │
                                           │ mj_step()            │
                                           └─────────────────────┘
```

Pinocchio is the analytical brain (FK, Jacobians, IK, gravity, collision
geometry). MuJoCo is the physics simulator (contact, dynamics, rendering).
Neither duplicates the other's computation.

---

## Key Concepts

### Object-centric frame
All cooperative motion is planned relative to the object, not relative to
individual arm bases. Grasp offsets (`left_ee = object_pose * offset_left`)
are computed once at grasp time and held constant during transport.

### Three coordination modes
- **Synchronized linear**: both arms move to independent SE3 targets,
  arriving simultaneously. Duration auto-computed from velocity limits.
- **Master-slave**: master follows prescribed waypoints; slave reconstructs the
  object pose from the master pose and tracks its grasp offset.
- **Symmetric**: both arms derive targets from an object trajectory directly.
  Used for lift/carry/lower phases.

### Dual impedance with internal force
```
tau = J^T * (K_p * e_x + K_d * e_xd) + g(q)
```
Symmetric gains prevent unintended internal forces. A 10 N squeeze force along
the inter-EE axis maintains grip during transport.

### Weld constraint
MuJoCo equality constraints (`left_grasp`, `right_grasp`) are activated at
contact and deactivated at release, simulating rigid grasps without needing
to model finger friction.

---

## Repository Structure

```
lab-6-dual-arm/
├── src/
│   ├── lab6_common.py              # Paths, constants, model loaders
│   ├── dual_arm_model.py           # DualArmModel + ObjectFrame
│   ├── dual_collision_checker.py   # HPP-FCL collision checking
│   ├── coordinated_planner.py      # 3 coordination modes
│   ├── cooperative_controller.py   # Dual impedance controller
│   ├── bimanual_grasp.py           # Bimanual state machine
│   ├── a1_independent_motion.py    # Demo: independent arm motion
│   ├── a2_coordinated_approach.py  # Demo: synchronized approach
│   ├── b1_cooperative_carry.py     # Demo: full carry pipeline
│   └── capstone_demo.py            # Full multi-scenario demo
│
├── models/
│   ├── ur5e_left.xml               # Left UR5e (prefixed names)
│   ├── ur5e_right.xml              # Right UR5e (180° yaw, prefixed names)
│   └── scene_dual.xml              # Full scene: arms + table + box + welds
│
├── tests/
│   ├── test_dual_model.py          # FK cross-validation, Jacobians
│   ├── test_dual_collision.py      # Collision/free config tests
│   ├── test_coordinated_planner.py # Timing sync, relative pose invariance
│   ├── test_bimanual_grasp.py      # State machine transitions
│   └── test_cooperative_controller.py
│
├── docs/
│   ├── 01_dual_arm_setup.md        # Kinematics, scene, collision checking
│   ├── 02_coordinated_motion.md    # Object-centric planning, 3 modes
│   └── 03_cooperative_manipulation.md  # Impedance control, state machine
│
├── docs-turkish/
│   └── ...                         # Turkish translations
│
├── tasks/
│   ├── PLAN.md
│   ├── ARCHITECTURE.md
│   ├── TODO.md
│   └── LESSONS.md
│
└── media/                          # Videos, plots
```

---

## Dependencies

```
Python   >= 3.10
MuJoCo   >= 3.0
pinocchio >= 2.6
hppfcl   >= 2.4      (ships with pinocchio on most installs)
numpy    >= 1.24
matplotlib >= 3.7
```

Install:
```bash
pip install mujoco pin numpy matplotlib
```

---

## Running the Demos

All demos are run from the `src/` directory or with the project root on
`PYTHONPATH`:

```bash
# Phase 1: independent arm motion test
python src/a1_independent_motion.py

# Phase 2: coordinated simultaneous approach to the box
python src/a2_coordinated_approach.py

# Phase 3: full cooperative carry pipeline
python src/b1_cooperative_carry.py

# Capstone: full multi-scenario demo with metrics
python src/capstone_demo.py
```

Run all tests:
```bash
pytest tests/ -v
```

---

## Results Summary

| Metric | Value |
|--------|-------|
| FK cross-validation error (Pinocchio vs MuJoCo) | < 1 mm |
| IK convergence (Levenberg-Marquardt) | < 100 iterations, tol = 1e-4 |
| Arm-arm collision check latency | < 1 ms (collision-free config) |
| Synchronized approach timing error | < 1 simulation timestep (1 ms) |
| EE position tracking error during carry | < 5 mm RMS |
| Object rotation during carry | < 5 degrees |
| Cooperative carry placement error | < 1 cm |
| Internal squeeze force | ~10 N along grasp axis |

---

## Docs

- `docs/01_dual_arm_setup.md` — Dual-arm kinematics, MuJoCo scene, FK
  cross-validation, arm-arm collision checking with HPP-FCL.
- `docs/02_coordinated_motion.md` — Object-centric frame, SE3 SLERP
  interpolation, three coordination modes, IK warm-starting.
- `docs/03_cooperative_manipulation.md` — Dual impedance control, internal
  force theory, squeeze force, bimanual state machine, full carry pipeline.

---

## Connection to Prior Labs

| Lab | Pattern reused in Lab 6 |
|-----|------------------------|
| Lab 3 | Impedance control loop: `tau = J^T * F + g(q)`, gravity compensation |
| Lab 4 | HPP-FCL collision checker structure, `is_collision_free`, `is_path_free` |
| Lab 5 | `GraspStateMachine` pattern: enum states, `step()`, weld constraint management |

All patterns are reimplemented within Lab 6 (no cross-lab imports) so each lab
remains independently runnable.
