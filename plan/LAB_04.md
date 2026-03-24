# Lab 4: Motion Planning & Collision Avoidance

> **Status:** Not Started  
> **Prerequisites:** Lab 2 (6-DOF IK), Lab 3 (torque-level control)  
> **Platform:** MuJoCo Menagerie UR5e + mounted Robotiq 2F-85, planned with Pinocchio/HPP-FCL  
> **Capstone Demo:** Plan and execute collision-free trajectory through a cluttered tabletop

---

## Objectives

1. Implement sampling-based planners (RRT, RRT*) in configuration space
2. Integrate Pinocchio collision checking for self-collision and environment obstacles
3. Apply trajectory smoothing and time-optimal parameterization (TOPP-RA)
4. Execute planned trajectories on the torque-controlled UR5e from Lab 3

---

## Why This Lab Matters

Lab 2 moved the arm in free space. Lab 3 added force awareness. But neither can handle obstacles. In any real manipulation scenario — a cluttered desk, a shelf, a kitchen — the robot must find collision-free paths. Planning is the bridge between "I know where to go" and "I know how to get there safely."

---

## Theory Scope

- Configuration space (C-space) vs. task space
- Sampling-based planning: RRT, RRT*, PRM — when to use which
- Collision checking: Pinocchio `computeCollisions` with HPP-FCL
- Trajectory smoothing: shortcutting, B-spline smoothing
- Time-optimal path parameterization: TOPP-RA (given path, find fastest timing)

---

## Architecture

```
Goal (start config → target config)
        │
        ▼
┌──────────────────────┐
│  Collision Checker    │
│  Pinocchio + HPP-FCL  │
│  Self-collision +     │
│  environment obstacles│
└──────────┬───────────┘
           │ collision query
           ▼
┌──────────────────────┐
│  RRT* Planner         │
│  Sample → Extend →    │
│  Rewire → Path        │
└──────────┬───────────┘
           │ waypoints in C-space
           ▼
┌──────────────────────┐
│  Trajectory Smoother  │
│  Shortcutting +       │
│  TOPP-RA timing       │
└──────────┬───────────┘
           │ timed trajectory
           ▼
┌──────────────────────┐
│  Trajectory Executor  │
│  Feed to impedance    │
│  controller (Lab 3)   │
└──────────────────────┘
```

---

## Implementation Phases

### Phase 1 — Collision Infrastructure
- Set up Pinocchio collision model from the same Menagerie UR5e + Robotiq geometry used in simulation
- Add environment obstacles (boxes on table, shelves) to both MuJoCo scene and Pinocchio
- Implement `is_collision_free(q)` function
- Visualize: show collision vs. free configurations

### Phase 2 — RRT / RRT* Implementation
- Implement basic RRT in C-space (6D)
- Extend to RRT* with rewiring for shorter paths
- Tune: step size, goal bias, max iterations
- Visualize planned path in simulation

### Phase 3 — Trajectory Post-Processing
- Implement path shortcutting (iterative collision-free shortcuts)
- Integrate TOPP-RA for time-optimal velocity profile
- Feed smoothed trajectory to Lab 3's impedance controller for execution
- Compare raw RRT path vs. smoothed path (jerkiness, execution time)

### Phase 4 — Capstone & Documentation
- Design cluttered scene: table with 3–5 box obstacles
- Plan and execute pick-like motion (approach from above, navigate between obstacles)
- Write LAB_04.md with algorithm explanations, C-space diagrams, comparison plots
- Write blog post: "from free space to cluttered environments"

---

## Key Design Decisions for Claude Code

- **Implement RRT from scratch** — don't use MoveIt2 or OMPL. This lab is about understanding the algorithm, not using a library. Keep it pure Python + Pinocchio.
- **Plan on the real executed geometry.** Lab 4 must not use a simplified/custom UR5e collision surrogate as the primary planner model. The collision model must represent the Menagerie UR5e and mounted Robotiq gripper used in MuJoCo.
- **Collision checking is the bottleneck.** Cache where possible. Pinocchio's `computeCollisions` is fast but called thousands of times during planning.
- **TOPP-RA can be a library.** Unlike RRT, TOPP-RA implementation is not the learning goal. Use `toppra` Python package.
- **Reuse Lab 3's controller.** The trajectory executor should feed waypoints into the impedance controller. No new low-level control code.

---

## Success Criteria

- [ ] Collision checker correctly identifies self-collisions and environment contacts
- [ ] RRT* finds valid paths in cluttered 6-DOF C-space
- [ ] Smoothed trajectories execute on torque-controlled UR5e without jerks
- [ ] Capstone demo: navigate between obstacles to reach a target pose
- [ ] LAB_04.md complete
- [ ] Blog post published

---

## References

- LaValle, *Planning Algorithms* — Ch. 5 (RRT)
- Pham & Pham, "TOPP-RA: Time-Optimal Path Parameterization" (2018)
- Pinocchio collision tutorial: HPP-FCL integration
- `toppra` Python package docs
