# Lab 5: Grasping & Manipulation

> **Status:** Not Started  
> **Prerequisites:** Lab 3 (force control), Lab 4 (motion planning)  
> **Platform:** MuJoCo Menagerie UR5e + Robotiq 2F-85 on MuJoCo  
> **Capstone Demo:** Pick up a box from the table, place it at a target location

---

## Objectives

1. Understand MuJoCo contact physics: friction, condim, solref/solimp
2. Add a gripper to the UR5e and control it (open/close)
3. Implement grasp planning: approach → grasp → lift → transport → place
4. Combine force control (Lab 3) and motion planning (Lab 4) into a full pick-and-place pipeline

---

## Why This Lab Matters

This is the first lab where the robot does useful work. Everything before this was building blocks — kinematics, dynamics, planning. Here they all converge: the robot must plan a path (Lab 4), make gentle contact (Lab 3), apply appropriate grip force, and move an object. This is also the foundation for Lab 6 (two arms) and Lab 9 (VLA-commanded manipulation).

---

## Theory Scope

- Grasp analysis: force closure, friction cones (conceptual — not full grasp synthesis)
- Contact modeling in MuJoCo: condim, friction, solref, solimp tuning
- Grasp state machine: approach → descend → close gripper → lift → transport → open
- Gripper control: position-controlled parallel jaw

---

## Architecture

```
Task: "Pick object at A, place at B"
        │
        ▼
┌──────────────────────┐
│  Grasp State Machine  │
│                       │
│  APPROACH → DESCEND → │
│  GRASP → LIFT →       │
│  TRANSPORT → PLACE    │
└──────────┬───────────┘
           │ current state
           ▼
┌──────────────────────┐
│  Per-State Controller │
│                       │
│  APPROACH: RRT* plan  │
│  DESCEND: impedance Z │
│  GRASP: gripper close │
│  LIFT: impedance +Z   │
│  TRANSPORT: RRT* plan │
│  PLACE: impedance -Z  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Lab 3 Controller +   │
│  Lab 4 Planner +      │
│  Gripper actuator     │
└──────────────────────┘
```

---

## Implementation Phases

### Phase 1 — Gripper Integration
- Use the MuJoCo Menagerie Robotiq 2F-85 as the gripper baseline
- Keep the same UR5e + Robotiq hardware stack as Labs 3 and 4
- Implement gripper open/close control
- Test: open and close gripper at various arm configurations
- Tune contact parameters so objects don't slip or fly away

### Phase 2 — Contact Physics Tuning
- Add graspable objects to the scene (box, cylinder)
- Experiment with MuJoCo contact parameters: condim, friction, solref, solimp
- Document what each parameter does and what values work
- Test: can the gripper hold an object while the arm moves?

### Phase 3 — Pick and Place Pipeline
- Implement the grasp state machine
- Wire each state to the appropriate controller (impedance / planner / gripper)
- Implement grasp pose computation (top-down approach for box)
- Full pipeline: idle → approach → grasp → lift → transport → place → idle

### Phase 4 — Documentation & Blog
- Write LAB_05.md with contact physics deep dive, state machine diagram, results
- Write blog post: "building a pick-and-place pipeline from scratch"
- Record capstone demo

---

## Key Design Decisions for Claude Code

- **State machine, not end-to-end.** This lab uses explicit states, not learned policies. Lab 9 replaces this with VLA. Understanding the manual pipeline makes you appreciate what the VLA learns.
- **No custom hardware path.** Lab 5 should not continue with a custom gripper or simplified arm as the primary implementation. The canonical stack is Menagerie UR5e + Robotiq 2F-85, shared with Labs 3 and 4.
- **Top-down grasps only.** Don't implement full 6-DOF grasp synthesis — it's a rabbit hole. Approach from above, close gripper. Simple and effective for tabletop tasks.
- **Contact tuning is the hard part.** Expect to spend significant time on MuJoCo's solref/solimp. Document every tuning decision — this knowledge transfers directly to Labs 6–9.
- **Reuse Labs 3 + 4.** The impedance controller and RRT* planner should be imported, not reimplemented.

---

## Success Criteria

- [ ] Gripper reliably grasps and holds objects during arm motion
- [ ] Contact parameters documented with rationale
- [ ] State machine executes full pick-and-place cycle
- [ ] Capstone demo: pick box from position A, place at position B
- [ ] LAB_05.md complete
- [ ] Blog post published

---

## References

- MuJoCo docs: Contact model, condim, solref/solimp
- MuJoCo Menagerie: Robotiq gripper models
- Craig, *Introduction to Robotics* — Ch. on grasping
