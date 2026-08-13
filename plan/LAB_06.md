# Lab 6: Dual-Arm Coordination

> **Status:** ✅ Complete (2026-05, commit `242c4ac`) — milestone-gated M0–M5, see `lab-6-dual-arm/tasks/TODO.md`
> **Prerequisites:** Lab 5 (grasping & manipulation)  
> **Platform:** Dual UR5e on MuJoCo (G1 upper-body option not used)  
> **Capstone Demo:** Two arms cooperatively lift and transport a large object
> **Scope note:** Shipped with **weld-constraint** cooperative carry, not the internal
> force control described below. The brief text is preserved as originally planned;
> `lab-6-dual-arm/docs/ARCHITECTURE.md` describes what was actually built.

---

## Objectives

1. Set up a dual-arm system in MuJoCo (two UR5e or G1 torso + arms)
2. Implement coordinated motion: both arms move in sync to a shared target
3. Handle shared workspace collision avoidance (arm-to-arm)
4. Execute a bimanual task: cooperatively carry an object

---

## Why This Lab Matters

This is the bridge from industrial robotics to humanoid robotics. A humanoid has two arms that must work together. The challenges are fundamentally different from single-arm: shared workspace means the arms can collide with each other, cooperative tasks require force coordination (neither arm drops its side), and planning complexity doubles. Solving this prepares you for the G1's upper body in Labs 7–9.

---

## Theory Scope

- Dual-arm kinematics: relative pose between end-effectors, object-centric frames
- Cooperative control: master-slave vs. symmetric coordination
- Shared workspace collision: arm-arm collision checking
- Internal force control: maintain grip force on a shared object without crushing it

---

## Architecture

```
Task: "Carry object from A to B using both arms"
        │
        ▼
┌──────────────────────────┐
│  Task Allocator           │
│  Object pose → grasp      │
│  poses for left + right   │
└──────────┬───────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│  Left   │ │  Right  │
│  Arm    │ │  Arm    │
│  Planner│ │  Planner│
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌──────────────────────────┐
│  Coordination Layer       │
│  Sync timing, check       │
│  arm-arm collision,       │
│  maintain object force    │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Dual Impedance Controller│
│  (Lab 3 × 2 + coupling)  │
└──────────────────────────┘
```

---

## Implementation Phases

### Phase 1 — Dual-Arm Setup
- Create MuJoCo scene with two UR5e arms facing each other (or load G1 upper body)
- Extend Pinocchio model for dual-arm kinematics
- Implement arm-arm collision checking
- Test: move each arm independently without collision

### Phase 2 — Coordinated Motion
- Implement synchronized motion: both arms follow a coupled trajectory
- Master-slave mode: one arm leads, other mirrors relative to object
- Symmetric mode: both arms track object-centric waypoints
- Test: both arms approach an object from opposite sides simultaneously

### Phase 3 — Cooperative Manipulation
- Implement bimanual grasp: both arms grasp a large box from opposite sides
- Add internal force control: maintain grip without crushing (symmetric impedance)
- Execute cooperative carry: grasp → lift → transport → place
- Handle the "rigid grasp" constraint: object must not rotate in-hand

### Phase 4 — Documentation & Blog
- Write LAB_06.md with dual-arm coordination theory, architecture, results
- Write blog post: "from one arm to two — the coordination challenge"
- Record capstone demo

---

## Platform Decision Point

This lab has a choice:

**Option A — Dual UR5e:** Simpler, reuses all Lab 2–5 code, clean isolated test of coordination algorithms. Better for learning.

**Option B — G1 Upper Body:** More realistic, forces you onto the G1 platform early, but adds complexity (floating base even if locked, different joint structure). Better for portfolio.

Recommendation: Start with dual UR5e to validate the coordination algorithms, then port the capstone demo to G1 upper body (with locked base) if time allows.

---

## Key Design Decisions for Claude Code

- **Object-centric frame is the key abstraction.** Don't plan in individual arm joint spaces. Define the object frame and derive each arm's target from it.
- **Timing synchronization matters.** If one arm arrives at the grasp point 0.5s before the other, the task fails. Synchronized trajectory timing is essential.
- **Internal forces are tricky.** When two arms hold the same rigid object, there's a redundancy — infinite combinations of forces can maintain the grasp. Use symmetric impedance (equal stiffness on both arms) as the simplest stable solution.

---

## Success Criteria

- [ ] Dual-arm scene with arm-arm collision avoidance
- [ ] Synchronized approach: both arms arrive at grasp points simultaneously
- [ ] Stable cooperative carry: object doesn't slip, rotate, or drop
- [ ] Capstone demo: bimanual pick, carry, and place of a large object
- [ ] LAB_06.md complete
- [ ] Blog post published

---

## References

- Smith et al., "Dual arm manipulation — A survey" (2012)
- Pinocchio: multi-body model handling
- MuJoCo: equality constraints (weld) for rigid grasp modeling
