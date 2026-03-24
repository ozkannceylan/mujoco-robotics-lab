# CLAUDE.md — Robotics Lab Series

You are working inside the `mujoco-robotics-lab` repository.

## Goal

Build a portfolio-ready robotics lab series using MuJoCo, progressing from simple planar arms to VLA-controlled humanoid manipulation. See `MASTER_PLAN.md` for the full roadmap.

## Context

- Engineer has a mechatronics background with a master's in RL for mobile robotics
- Lab 1 (2-link planar arm) is complete — FK, Jacobian, IK, PD control, trajectory generation
- Lab 2 (UR5e 6-DOF) is complete — scales Lab 1 foundations to an industrial arm with Pinocchio
- Lab 3 (Dynamics & Force Control) is complete — RNEA/CRBA, gravity compensation, Cartesian impedance, hybrid force control
- Lab 4 (Motion Planning) is complete — Pinocchio+HPP-FCL collision checking, RRT*, TOPP-RA trajectory parameterization
- Lab 5 (Grasping & Manipulation) is complete — custom parallel-jaw gripper, DLS IK, pick-and-place state machine, Lab 3+4 integration
- Labs 6–9 are planned — dual-arm, locomotion, whole-body, VLA
- End goals: strengthen fundamentals for humanoid VLA work, prepare for robotics interviews, build a portfolio demo

## Tech Stack

- **Python 3.10+**
- **MuJoCo** — physics simulation, rendering, contact dynamics
- **Pinocchio (pin)** — analytical FK, Jacobian, dynamics (RNEA, ABA, CRBA), collision checking (HPP-FCL)
- **NumPy** — all numerical computation
- **Matplotlib** — plotting, 3D visualization
- **meshcat-python** — optional interactive 3D viewer
- **ROS2 Humble** — bridge node integration (later labs)

## Architecture Principle

```
Pinocchio = analytical brain (FK, Jacobian, M, C, g, IK)
MuJoCo   = physics simulator (step, render, contact, sensor)
```

- Use Pinocchio for ALL analytical computations
- Use MuJoCo for simulation execution and rendering
- Never duplicate computation — if Pinocchio computes it, don't recompute in MuJoCo
- Cross-validate between the two as a correctness check

---

## Per-Lab Workflow

**This is the mandatory workflow for every lab. Follow it in order.**

### Step 1 — Read the lab brief

Each lab has a detailed plan file in the `plan/` directory: `plan/LAB_XX.md`. Read it fully before doing anything else. It contains objectives, theory scope, architecture, implementation phases, key design decisions, and success criteria.

### Step 2 — Create the lab folder and `tasks/` subfolder

```
lab-N-<name>/
├── tasks/
│   ├── PLAN.md           ← Step 3: write this first
│   ├── ARCHITECTURE.md   ← Step 4: write this before any code
│   ├── TODO.md           ← Step 5: create from PLAN, update after every step
│   └── LESSONS.md        ← Step 6: log bugs, debug strategies, insights
├── src/
├── models/
├── docs/
├── docs-turkish/
├── media/
├── tests/
├── ros2_bridge/
└── README.md
```

### Step 3 — Write `tasks/PLAN.md`

Break the lab brief into concrete implementation steps. This is your contract — you execute this plan, nothing more, nothing less.

Format:
```markdown
# Lab N: [Title] — Implementation Plan

## Phase 1: [Name]
### Step 1.1: [Specific task]
- What to build
- Expected output / how to verify
### Step 1.2: ...

## Phase 2: [Name]
### Step 2.1: ...
...
```

### Step 4 — Write `tasks/ARCHITECTURE.md`

Document the full technical architecture BEFORE writing any code. This file is the source of truth for how modules connect.

Must include:
- **Module map:** which Python files exist and what each one does
- **Data flow:** what data flows between modules (diagram or description)
- **Key interfaces:** function signatures for the main APIs
- **Model files:** which MJCF/URDF files are needed and where they come from
- **Dependencies on previous labs:** what is imported from `shared/` or earlier labs

### Step 5 — Create and maintain `tasks/TODO.md`

Generate from PLAN.md. Update after every completed step.

Format:
```markdown
# Lab N: TODO

## Phase 1: [Name]
- [x] Step 1.1: [description] — DONE (2026-03-16)
- [ ] Step 1.2: [description]
- [ ] Step 1.3: [description]

## Phase 2: [Name]
- [ ] Step 2.1: ...

## Current Focus
> Step 1.2: [what you're working on right now]

## Blockers
> None / [describe any blockers]
```

Rules:
- Check off items immediately after completion
- Update "Current Focus" before starting each step
- Log blockers as they appear
- Never skip ahead without updating TODO

### Step 6 — Maintain `tasks/LESSONS.md`

Log every bug, failed approach, and debug insight AS IT HAPPENS. This is not written after the fact — it's a live journal.

Format:
```markdown
# Lab N: Lessons Learned

## Bugs & Fixes

### [Date] — [Short description]
**Symptom:** What went wrong
**Root cause:** Why it happened
**Fix:** What resolved it
**Takeaway:** What to remember for future labs

## Debug Strategies

### [Technique name]
When to use it, how it helped

## Key Insights

### [Insight]
Brief explanation of something non-obvious learned during implementation
```

---

## Execution Rules

1. **Read LAB_XX.md → Write PLAN → Write ARCHITECTURE → Create TODO → Then code.** Never skip steps.
2. **Update TODO.md after every completed step.** If you forget, the next session starts with stale state.
3. **Log bugs in LESSONS.md immediately.** Don't wait until the end. Future labs will hit the same issues.
4. **One phase at a time.** Complete all steps in Phase N before starting Phase N+1.
5. **Tests before moving on.** Each phase should have passing tests before the next phase begins.
6. **Cross-validate Pinocchio vs MuJoCo** whenever both compute the same quantity.
7. **When resuming a lab**, read `tasks/TODO.md` first to find exactly where you left off.

---

## Project Structure

```
mujoco-robotics-lab/
├── CLAUDE.md                         # This file
├── plan/                             # Lab briefs (MASTER_PLAN.md, LAB_03.md … LAB_09.md)
├── README.md                         # Main project README
│
├── lab-1-2link-arm/                  # ✅ Complete
├── lab-2-Ur5e-robotics-lab/          # ✅ Complete
├── lab-3-dynamics-force-control/     # ✅ Complete
├── lab-4-motion-planning/            # ✅ Complete
│
│   # Each lab follows this layout:
│   ├── tasks/    # PLAN.md, ARCHITECTURE.md, TODO.md, LESSONS.md
│   ├── src/      # Python source (lab_N_common.py + modules)
│   ├── models/   # MJCF / URDF files
│   ├── docs/     # English documentation
│   ├── docs-turkish/
│   ├── media/    # Plots, GIFs, videos
│   ├── tests/    # test_*.py files
│   └── README.md
│
├── lab-5-grasping-manipulation/      # ✅ Complete
├── ... (labs 6–9 follow same structure)
│
└── blog/                             # Blog posts per lab
```

---

## Code Standards

- Every function: docstring + type hints
- Comments in English
- Test files in `<lab>/tests/` — naming: `test_{module}.py`
- Model files in `<lab>/models/`
- Use `pathlib.Path` for all file paths
- No hardcoded absolute paths — use relative paths from project root
- Numerical comparisons: use `np.allclose()` with explicit tolerances
- Documentation: always write both English (`docs/`) and Turkish (`docs-turkish/`)

## Common Patterns

### Loading models

```python
# Pinocchio
import pinocchio as pin
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, mesh_dir)
data = model.createData()

# MuJoCo
import mujoco
mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
mj_data = mujoco.MjData(mj_model)
```

### Cross-validation pattern

```python
# Always compare Pinocchio vs MuJoCo when both compute the same quantity
pin.forwardKinematics(model, data, q)
ee_pin = data.oMf[frame_id].translation

mujoco.mj_step(mj_model, mj_data)
ee_mj = mj_data.xpos[body_id]

assert np.allclose(ee_pin, ee_mj, atol=1e-3), f"FK mismatch: {ee_pin} vs {ee_mj}"
```

---

## Known Issues + Solutions

### Issue: Pinocchio and MuJoCo frame conventions may differ
Solution: Check the frame ordering. MuJoCo uses body indices, Pinocchio uses frame IDs. Map them explicitly once and store the mapping.

### Issue: UR5e URDF from different sources may have different joint naming
Solution: Standardize on the mujoco_menagerie naming convention. Print `model.names` on first load and verify.

### Issue: Pinocchio quaternion convention is (x, y, z, w), MuJoCo uses (w, x, y, z)
Solution: Always convert explicitly when passing quaternions between the two. Write a utility function `pin_quat_to_mj()` and `mj_quat_to_pin()`.

### Issue: MuJoCo Menagerie position servos have gravity droop and tracking lag
Solution: Menagerie models use `general` actuators with `tau = Kp*(ctrl-qpos) - Kd*qvel`. Naive `ctrl = q_des` causes steady-state offset from gravity and velocity lag. Fix with feedforward: `ctrl = q_des + qfrc_bias/Kp + Kd*qd_des/Kp`. This achieved 0.088 mm RMS (vs 133 mm without).

### Issue: IK solutions may collide with scene objects (table, etc.)
Solution: IK solvers don't know about obstacles. Always check `data.ncon` after setting `data.qpos` to each IK solution. If contacts exist, reposition the target or add a Y-offset to keep the arm clear.

### Issue: Pinocchio GeometryObject constructor order
Solution: Use `GeometryObject(name, parent_joint, parent_frame, placement, shape)`. The older order `(name, parent_joint, parent_frame, shape, placement)` is deprecated and silently wrong.

### Issue: Adjacent-link self-collision produces false positives
Solution: Skip collision pairs where parent joint indices differ by ≤1 (`adjacency_gap=1`). Adjacent links physically can't collide, and overlapping collision geometries at joints cause spurious collisions.

### Issue: TOPP-RA crashes on near-duplicate waypoints
Solution: Filter consecutive waypoints within `1e-8` of each other before constructing the arc-length spline. `scipy.interpolate.CubicSpline` requires strictly increasing arc-length values.

### Issue: Lab src/ files need sys.path when imported cross-lab
Solution: Each lab module that imports from another lab must add the foreign `src/` to `sys.path` at the top of the file using `Path(__file__).resolve()` and conditional `sys.path.insert(0, ...)`.

### Issue: MuJoCo freejoint body qpos layout
Solution: After arm joints (6) and gripper joints (2 with equality → 2 positions in qpos), the freejoint occupies qpos[8:15] (3 pos + 4 quat). Equality constraint does NOT reduce qpos size — both joint positions appear. Always verify with `mj_model.nq`.

### Issue: Gripper minimum gap must be less than object half-width
Solution: Compute `pad_inner_face = finger_body_y + pad_y_offset - pad_half_size` and verify it's less than `object_half_width` before writing any control code. A gap of even 1mm is enough for stiff contacts (`solimp="0.99"`). Test by checking `data.ncon` in a static scene.

### Issue: `is_gripper_in_contact` must check all finger geoms, not just pads
Solution: The structural finger body geom (`left_finger_geom`, `right_finger_geom`) contacts the object before the smaller pad geom. Check both geoms in the contact loop. Never limit to only pad geoms.

### Issue: Contact tests must check during closing, not after settling
Solution: A free-flying box with no arm gravity compensation will fall to the floor in ~1 second. Contact tests should break-and-check inside the step loop: `if is_gripper_in_contact(m, d): contact_detected = True; break`.

### Issue: `parameterize_topp_ra` returns 4-tuple (times, q, qd, qdd)
Solution: Unpack as `times, q_traj, qd_traj, _ = parameterize_topp_ra(...)`. The fourth element `qdd` (accelerations) is often not needed by the controller.

---

## Lab Progress

- [x] Lab 1: 2-Link Planar Arm (square drawing demo)
- [x] Lab 2: UR5e 6-DOF Arm (cube drawing demo)
- [x] Lab 3: Dynamics & Force Control (gravity comp, Cartesian impedance, force control)
- [x] Lab 4: Motion Planning & Collision Avoidance (RRT*, TOPP-RA, capstone demo)
- [x] Lab 5: Grasping & Manipulation (custom gripper, DLS IK, pick-and-place state machine)
- [ ] Lab 6: Dual-Arm Coordination
- [ ] Lab 7: Locomotion Fundamentals
- [ ] Lab 8: Whole-Body Loco-Manipulation
- [ ] Lab 9: VLA Integration

---

## Debugging Checklist

When something doesn't match between Pinocchio and MuJoCo:
1. Check joint angle ordering — are both using the same convention?
2. Check frame/body ID mapping — print names from both
3. Check quaternion convention — (w,x,y,z) vs (x,y,z,w)
4. Check if gravity direction matches in both models
5. Check units — Pinocchio uses SI (meters, radians, kg), verify MuJoCo model does too

---

## Session Start Protocol

When starting work on any lab:
1. Read this CLAUDE.md
2. Read the lab brief: `plan/LAB_XX.md`
3. Check `lab-N-<name>/tasks/TODO.md` for current state
4. Check `lab-N-<name>/tasks/LESSONS.md` for known issues
5. Resume from "Current Focus" in TODO.md
