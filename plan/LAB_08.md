# Lab 8: Whole-Body Loco-Manipulation

> **Status:** 🚧 In Progress — kickoff 2026-08-14; see `lab-8-loco-manipulation/tasks/TODO.md` (next: M0)
> **Prerequisites:** Lab 6 (dual-arm coordination), Lab 7 (locomotion)  
> **Platform:** Unitree G1 on MuJoCo + Pinocchio  
> **Capstone Demo:** G1 walks to a table, picks up an object, carries it while walking
> **Scope correction (2026-08-14):** Phase 2 below says "combine Lab 7's gait
> generator" — Lab 7 shipped **no gait generator** (M4 blocked by position actuators).
> Lab 8 owns gait generation on the torque/RNEA path as milestones M2–M3; Lab 7's
> LIPM planner and floating-base stack are reused. See the lab's `tasks/PLAN.md`.

---

## Objectives

1. Implement a whole-body controller that handles locomotion and manipulation simultaneously
2. Use task-priority QP (quadratic programming) to resolve conflicts between tasks
3. Combine the walking gait (Lab 7) with arm manipulation (Labs 3–5)
4. Execute a full loco-manipulation sequence on the G1

---

## Why This Lab Matters

This is where everything converges. Labs 1–6 built manipulation skills on a fixed base. Lab 7 made the robot walk. This lab removes the artificial separation: the humanoid must walk AND use its hands at the same time. This is the operational mode for Lab 9's VLA — the model will output whole-body joint commands, and you need to understand what "correct" whole-body behavior looks like before training a policy to produce it.

---

## Theory Scope

- Whole-body control: controlling all DOFs simultaneously with multiple objectives
- Task priority framework: primary task (balance) > secondary task (manipulation) > tertiary (posture)
- QP-based control: formulate as quadratic program with equality/inequality constraints
- Task hierarchy: balance constraints are never violated, manipulation is best-effort
- Momentum control: regulate centroidal momentum while performing arm tasks

---

## Architecture

```
Task Stack (priority ordered):
  1. Balance (CoM inside support polygon)
  2. Foot placement (gait trajectory)
  3. Hand task (reach / carry object)
  4. Posture regulation (stay close to nominal)
        │
        ▼
┌──────────────────────┐
│  Whole-Body QP        │
│                       │
│  min ‖J·q̇ - ẋ_d‖²   │
│  s.t. balance         │
│       joint limits    │
│       torque limits   │
│       contact forces  │
└──────────┬───────────┘
           │ q̇_desired (all DOFs)
           ▼
┌──────────────────────┐
│  Inverse Dynamics     │
│  (Pinocchio RNEA)     │
│  q̇_desired → τ       │
└──────────┬───────────┘
           │ joint torques
           ▼
┌──────────────────────┐
│  MuJoCo Simulation    │
│  G1 full body         │
└──────────────────────┘
```

---

## Implementation Phases

### Phase 1 — Whole-Body Controller Framework
- Implement task-priority QP using a solver (OSQP or qpOASES via Python bindings)
- Define tasks: CoM tracking, foot tracking, hand tracking, posture
- Test on standing G1: command hand position while maintaining balance
- Validate: moving the arm doesn't make the robot fall over

### Phase 2 — Walking + Arm Motion
- Combine Lab 7's gait generator with the whole-body QP
- Walk while keeping arms at a fixed Cartesian pose
- Then: walk while moving one arm to a target (reach for an object)
- Test: arm tracks target within 5cm while walking remains stable

### Phase 3 — Loco-Manipulation Sequence
- Design the full task: G1 starts 1m from table, walks to it, grasps object, walks back
- Implement task sequencing: walk → stop → reach → grasp → walk with object
- The QP must handle the load change (object mass affects CoM)
- Capstone: complete the full sequence without falling

### Phase 4 — Documentation & Blog
- Write LAB_08.md: whole-body QP formulation, task hierarchy, results
- Write blog post: "walking and working — whole-body humanoid control"
- Record capstone demo

---

## Key Design Decisions for Claude Code

- **QP solver choice matters.** OSQP (open source, Python bindings) is the pragmatic choice. Don't implement QP from scratch.
- **Task priorities, not task weights.** Strict priority (balance can never be violated for manipulation) is safer than weighted objectives for a walking robot.
- **Start with standing + reaching.** Get the whole-body QP working while the robot stands before adding locomotion. Walking + manipulation simultaneously is the hardest case.
- **Simplify the grasp.** Use Lab 5's gripper logic but keep it simple — the focus of this lab is whole-body coordination, not grasp sophistication.
- **CoM shift matters.** When the robot picks up an object, its CoM shifts. The balance controller must handle this or the robot falls during transport.

---

## Success Criteria

- [ ] Whole-body QP resolves arm + balance tasks without instability
- [ ] G1 reaches for objects while standing without losing balance
- [ ] G1 walks while maintaining arm pose (carrying behavior)
- [ ] Capstone demo: walk → grasp → walk with object → place
- [ ] LAB_08.md complete
- [ ] Blog post published

---

## References

- Sentis & Khatib, "Synthesis of Whole-Body Behaviors through Hierarchical Control of Behavioral Primitives" (2005)
- Escande et al., "Hierarchical Quadratic Programming" (2014)
- OSQP solver documentation
- Pinocchio: whole-body kinematics and dynamics
