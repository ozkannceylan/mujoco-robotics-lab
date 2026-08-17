# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read and follow: /home/ozkan/Documents/MyProjects/_meta/workflow-rules.md

## Goal

Build a portfolio-ready robotics lab series using MuJoCo, progressing from simple planar arms to VLA-controlled humanoid manipulation. See `plan/MASTER_PLAN.md` for the full roadmap.

## Context

- Engineer has a mechatronics background with a master's in RL for mobile robotics
- Lab 1 (2-link planar arm) is complete — FK, Jacobian, IK, PD control, trajectory generation
- Lab 2 (UR5e 6-DOF) is complete — scales Lab 1 foundations to an industrial arm with Pinocchio
- Lab 3 (Dynamics & Force Control) is complete — RNEA/CRBA, gravity compensation, Cartesian impedance, hybrid force control
- Lab 4 (Motion Planning) is complete — Pinocchio+HPP-FCL collision checking, RRT*, TOPP-RA trajectory parameterization
- Lab 5 (Grasping & Manipulation) is complete — custom parallel-jaw gripper, DLS IK, pick-and-place state machine, Lab 3+4 integration
- Lab 6 (Dual-Arm Coordination) is complete — dual UR5e, weld-constraint cooperative carry, milestone-gated M0–M5
- Lab 7 (Locomotion) is complete at M3d scope — G1 standing + weight shift; ZMP walking blocked by position actuators, deferred to Lab 8
- Lab 8 (Whole-Body Loco-Manipulation) is complete — milestone-gated M0–M6, closed 2026-08-17; owns gait generation via torque control (Lab 7's actuator finding). Torque-actuated G1, model parity 1e-16; whole-body inverse-dynamics QP with contact wrenches, 7.08 mm hand tracking; in-place stepping with ZMP 100 % inside support; **DCM forward walking — 12 steps, 1.18 m, 6.2 mm DCM RMS**; walk + two-handed carry pose via centroidal angular-momentum control, 14.5 mm hand RMS; **loco-manipulation capstone — walk→pick→carry→place, payload 11.8 mm from target**; docs EN/TR + code walkthrough + blog post; 97 tests
- Lab 9 (VLA) is planned — depends on Lab 8 controllers for demonstration data
- End goals: strengthen fundamentals for humanoid VLA work, prepare for robotics interviews, build a portfolio demo

---

## Common Commands

### Fresh environment setup

From a fresh clone, run the setup script first — a bare `git clone` is **not**
enough to run any lab:

```bash
./tools/setup_env.sh
export MUJOCO_GL=egl   # required for headless rendering (no display attached)
```

The script installs the Python deps, sparse-clones the MuJoCo Menagerie models
into both locations the labs expect, and best-effort installs `libegl1`.

Three things a fresh clone does **not** give you, and the script fixes:

1. **Menagerie models are not in the repo.** Labs 2–6 load them from
   `lab-2-Ur5e-robotics-lab/models/mujoco_menagerie/` and Lab 7 from
   `third_party/mujoco_menagerie/unitree_g1/`. Both paths are gitignored and
   must be populated by the setup script (or cloned manually) before any demo
   or test will run.
2. **Headless rendering needs EGL.** Without `MUJOCO_GL=egl` (and the `libegl1`
   system package) every render/record script fails on a machine with no
   display.
3. **The PyPI package `pinocchio` is the wrong package** — see below.

### Install dependencies

```bash
pip install mujoco numpy pin scipy "imageio[ffmpeg]" matplotlib pytest meshcat
```

> The real Pinocchio is `pip install pin`; it provides `import pinocchio`.
> The PyPI package literally named `pinocchio` is an unrelated project — do not
> install it.

### Run tests

```bash
# All tests for a specific lab
pytest lab-3-dynamics-force-control/tests/

# Single test file
pytest lab-4-motion-planning/tests/test_collision.py

# Single test method
pytest lab-5-grasping-manipulation/tests/test_gripper.py::TestGripperContact::test_contact_detection -v

# All tests across the project (303 as of 2026-08-16)
pytest lab-*/tests/
```

> Cross-lab runs work only because the repo-root `conftest.py` isolates
> same-named modules per lab (Labs 1 and 2 both define `a4_inverse_kinematics`,
> Labs 7 and 8 both define `standing_controller`). Without it the second lab
> collected imports the first lab's file. Don't delete it.

No pytest config files — uses defaults. Tests use both `unittest.TestCase` and pure pytest fixtures.

### Run demos

Each lab has numbered scripts (a1, a2, b1, c1, etc.) that run in order:

```bash
python3 lab-1-2link-arm/src/c1_draw_square.py        # Lab 1 capstone
python3 lab-2-Ur5e-robotics-lab/src/c3_draw_cube.py   # Lab 2 capstone
python3 lab-3-dynamics-force-control/src/c1_force_control.py
python3 lab-4-motion-planning/src/capstone_demo.py
python3 lab-5-grasping-manipulation/src/record_pro_demo.py
python3 lab-8-loco-manipulation/src/m3_walking.py   # G1 walks 1.18 m (needs MUJOCO_GL=egl)
```

---

## Architecture Principle

```
Pinocchio = analytical brain (FK, Jacobian, M, C, g, IK)
MuJoCo   = physics simulator (step, render, contact, sensor)
```

- Use Pinocchio for ALL analytical computations
- Use MuJoCo for simulation execution and rendering
- Never duplicate computation — if Pinocchio computes it, don't recompute in MuJoCo
- Cross-validate between the two as a correctness check

## Lab Common Module Pattern

Every lab has a `src/lab<N>_common.py` that is the central configuration hub:

- **Directory constants**: `LAB_DIR`, `PROJECT_ROOT`, `MODELS_DIR`, `MEDIA_DIR`
- **Physical constants**: `NUM_JOINTS`, `DT`, `GRAVITY`, joint/torque limits
- **Default configs**: `Q_HOME`, `Q_ZEROS`
- **Model paths**: URDF and MJCF file locations
- **Quaternion utilities**: `mj_quat_to_pin()`, `pin_quat_to_mj()`
- **Model loaders**: `load_mujoco_model()`, `load_pinocchio_model()`
- **Control helpers**: `apply_arm_torques()`, `get_mj_ee_site_id()`, etc.

## Cross-Lab Import Pattern

Later labs import from earlier labs via `sys.path` manipulation. The UR5e URDF from Lab 3 is reused by all subsequent labs.

```python
# In lab8_common.py — importing from Lab 7.
# APPEND the foreign lab and keep this lab's own src/ at position 0, or a
# shared module name (standing_controller, record_demo, …) resolves to the
# wrong lab. See Known Issues.
_LAB7_SRC_DIR = PROJECT_ROOT / "lab-7-locomotion" / "src"
if str(_LAB7_SRC_DIR) not in sys.path:
    sys.path.append(str(_LAB7_SRC_DIR))

from lab7_common import NQ, NV, PELVIS_MJCF_Z, mj_qpos_to_pin
```

Older labs (3–5) still use `sys.path.insert(0, ...)` for this; it works there
only because no module name collides yet.

Tests do the same to reach their lab's `src/`:
```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

## Video Production Pipeline

`tools/video_producer.py` provides a reusable 3-phase demo workflow:
1. Animated metrics presentation (Matplotlib)
2. Native MuJoCo simulation recording with overlays
3. ffmpeg composition into final H.264 artifact

Constants: 1920x1080, 30 FPS, dark theme (`#08111f` background).

---

## Per-Lab Workflow

**Mandatory for every new lab. Follow in order.**

1. **Read the lab brief**: `plan/LAB_XX.md` — read fully before anything else
2. **Create lab folder** with `tasks/`, `src/`, `models/`, `docs/`, `docs-turkish/`, `media/`, `tests/`
3. **Write `tasks/PLAN.md`**: Break lab brief into phased implementation steps
4. **Write `tasks/ARCHITECTURE.md`**: Module map, data flow, key interfaces, model files, cross-lab deps — before any code
5. **Create `tasks/TODO.md`**: Generated from PLAN.md, updated after every step. Must have "Current Focus" and "Blockers" sections
6. **Maintain `tasks/LESSONS.md`**: Live journal — log bugs/fixes/insights AS THEY HAPPEN with Symptom/Root cause/Fix/Takeaway format

## Execution Rules

1. **Read LAB_XX.md → Write PLAN → Write ARCHITECTURE → Create TODO → Then code.** Never skip steps.
2. **Update TODO.md after every completed step.** If you forget, the next session starts with stale state.
3. **Log bugs in LESSONS.md immediately.** Don't wait until the end. Future labs will hit the same issues.
4. **One phase at a time.** Complete all steps in Phase N before starting Phase N+1.
5. **Tests before moving on.** Each phase should have passing tests before the next phase begins.
6. **Cross-validate Pinocchio vs MuJoCo** whenever both compute the same quantity.
7. **When resuming a lab**, read `tasks/TODO.md` first to find exactly where you left off.

---

## Tech Stack

- **Python 3.10+**
- **MuJoCo** — physics simulation, rendering, contact dynamics
- **Pinocchio (pin)** — analytical FK, Jacobian, dynamics (RNEA, ABA, CRBA), collision checking (HPP-FCL)
- **NumPy** — all numerical computation
- **SciPy** — optimization (IK solvers, TOPP-RA splines)
- **Matplotlib** — plotting, 3D visualization
- **meshcat-python** — optional interactive 3D viewer
- **ROS2 Humble** — bridge node integration (later labs)

## Code Standards

- Every function: docstring + type hints
- Comments in English
- Test files in `<lab>/tests/` — naming: `test_{module}.py`
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
pin.forwardKinematics(model, data, q)
ee_pin = data.oMf[frame_id].translation

mujoco.mj_step(mj_model, mj_data)
ee_mj = mj_data.xpos[body_id]

assert np.allclose(ee_pin, ee_mj, atol=1e-3), f"FK mismatch: {ee_pin} vs {ee_mj}"
```

---

## Known Issues + Solutions

### Pinocchio vs MuJoCo frame conventions
MuJoCo uses body indices, Pinocchio uses frame IDs. Map them explicitly once and store the mapping.

### Pinocchio quaternion (x,y,z,w) vs MuJoCo (w,x,y,z)
Always convert explicitly. Use `pin_quat_to_mj()` and `mj_quat_to_pin()` from the lab common module.

### MuJoCo Menagerie position servos — gravity droop and tracking lag
Menagerie `general` actuators: `tau = Kp*(ctrl-qpos) - Kd*qvel`. Fix with feedforward: `ctrl = q_des + qfrc_bias/Kp + Kd*qd_des/Kp`. Achieved 0.088 mm RMS (vs 133 mm without).

### IK solutions may collide with scene objects
IK solvers don't know about obstacles. Check `data.ncon` after setting `data.qpos` to each IK solution.

### Pinocchio GeometryObject constructor order
Use `GeometryObject(name, parent_joint, parent_frame, placement, shape)`. The older order with shape before placement is deprecated and silently wrong.

### Adjacent-link self-collision false positives
Skip collision pairs where parent joint indices differ by ≤1 (`adjacency_gap=1`).

### TOPP-RA crashes on near-duplicate waypoints
Filter consecutive waypoints within `1e-8` before constructing arc-length spline. `scipy.interpolate.CubicSpline` requires strictly increasing values.

### Cross-lab imports need sys.path — but APPEND the foreign lab, never insert(0)
Each lab module importing from another lab must add the foreign `src/` to `sys.path` using `Path(__file__).resolve()`. Use `sys.path.append(...)` for the foreign lab and keep your own `src/` at position 0. Labs share module names (`standing_controller`, `grasp_planner`, `record_demo`), so `insert(0, foreign_src)` silently shadows the local module with another lab's implementation — Lab 8 hit this as `ImportError: cannot import name 'GravityMode' from 'standing_controller'` pointing at Lab 7's file.

### A contact model that a standing test cannot falsify is still worth checking
Lab 8 described the G1 foot as a symmetric ±0.08 m CoP box on the ankle frame.
The real Menagerie sole spans x ∈ [−0.05, 0.12], y ∈ ±0.025, 35 mm *below* that
frame. Standing never excites the difference; walking uses both ends of the foot
every step, and the error showed up as contact wrenches the QP planned and
MuJoCo refused to produce (realised CoM acceleration: slope 0.78, bias
−0.09 m/s²). Also include the shear term — the wrench is about a frame above the
ground, so `CoP_x = (−m_y − h·f_x)/f_z`, not `−m_y/f_z` (Lab 8 L-M3-d).

### An OSQP tolerance below your problem's conditioning costs accuracy, not buys it
`eps_abs = eps_rel = 1e-6` on a task stack spanning weights 1e4…1e1 against a
1e-4 regularisation made 38 % of Lab 8's control ticks return `maximum
iterations reached` at 12.6 ms/solve. At `1e-4` every tick converges in ~25
iterations and 0.073 ms — **and the constraint residual drops** (0.021 →
8.5e-5 N·m). A hit iteration cap is a correctness warning, not a perf note: the
returned point is wherever the solver happened to be (Lab 8 L-M3-e).

### Quasi-static balance references cannot walk; command the DCM instead
"Move the CoM over the foot that is about to take the load" works for stepping
in place and fails at every gain setting for forward walking — it needs a moment
of rest over each foot that walking never provides. Command the divergent
component `ξ = c + ċ/ω` (`ω = √(g/z_c)`) instead: plan a ZMP through the
footsteps, back-integrate `ξ̇ = ω(ξ − p)` from a terminal rest condition, and
issue `c̈ = ω²(c − p_cmd)` with `p_cmd = ξ − ξ̇_ref/ω + (k/ω)(ξ − ξ_ref)`. No CoM
*position* task on the control path at all (Lab 8 `dcm_planner.py`, L-M3-c).

### Lateral balance cost is set by stance width, not stride length
The ZMP crosses between the feet every step and the lateral DCM swings with the
same amplitude, so a wide stance demands a large excursion be arrested inside
one foot width. Lab 8's G1 went 7/12 steps at its 0.237 m rest stance and 12/12
at 0.18 m, with stride and everything else unchanged (L-M3-f).

### An arm task on a walking robot needs a centroidal momentum task, not re-weighting
Manipulation and balance are coupled through momentum: the CoM Jacobian includes
the arms, so a hand task disturbs the quantity keeping the robot upright. Lab 8
found no hand weight that both walked and tracked — and the failures were
*non-monotonic*, which is the tell that a term is missing rather than mis-tuned.
Adding `L = A_g(q) q̇` regulation took the same hand task from falling on step 7
to 12/12 steps with 3× better tracking (Lab 8 L-M4-c). Use `L → 0` for a held
pose; supply an `L_ref` when a task deliberately moves mass (Kajita's resolved
momentum control), or the term fights the trajectory it was added to enable.

### Perturb what should not matter before believing a pass
A Lab 8 walking+reach configuration passed 12/12 steps; shifting the commanded
circle's *starting phase* — which changes nothing about the task — dropped it to
9/12 and then 3/12. A result a no-op perturbation can destroy is a draw from a
distribution, not a controller property. The same check on the carry task came
back flat (12/12 across stride and double-support changes), which is what made
that gate trustworthy (Lab 8 L-M4-f).

### A MuJoCo weld holds its COMPILE-TIME relative pose
`eq_active` is a switch, not a "grasp here" instruction. `mjEQ_WELD` holds body2
at the relpose baked into `model.eq_data` at compile time, so activating it
where the hand happens to be commands a snap back to the rest configuration —
Lab 8 measured a 0.42 m lurch that threw the robot down. Write the live
relative pose into `eq_data[3:10]` (pos + wxyz quat, body2 in body1) before
setting `eq_active`, and refuse to close a weld the hand has not actually
reached (Lab 8 L-M5-b).

### When a known-good controller regresses in a new scene, read the contact list
Lab 8's identical M3 controller walks 12 steps on the bare model and fell on
step 4 once a pedestal was added. Logging every contact involving a scene prop
named it in one line — `pick_pedestal ↔ right_hip_roll_link` — where two hours
of balance-tuning hypotheses had not. Scene furniture at limb height is a
collision the balance controller cannot anticipate (Lab 8 L-M5-f).

### Servo the object, not the gripper
A hand target derived from a hand→object offset measured once before the motion
goes stale: a compliant grasp lets the load settle, and Lab 8 saw that become a
systematic 55 mm placement error. Recompute the hand target every tick from the
*live* offset toward the object's goal. Release accuracy went 65 mm → 18.9 mm
(Lab 8 L-M5-i). And check the goal is a pose the object can rest in — a target
0.09 m in from a 0.10 m half-extent shelf overhangs, tips and drops (L-M5-j).

### Floating base: do NOT inertia-shape joint PD gains with M(q)
The Lab 5 fix `τ = M(q)(Kp·e + Kd·ė) + g` is correct for a **fixed-base** arm. On a floating-base humanoid `M(q)[6:,6:]` is not the inertia felt through the closed leg chains, and shaping gains with it saturates actuators and makes the G1 fall at every gain setting. Use raw joint-space gains there (Lab 8 L-M0-b).

### A velocity-level (kinematic) QP cannot balance a floating-base robot
`min ‖J q̇ − ẋ_d‖²` can satisfy `J_com q̇ = 0` exactly while the robot topples: CoM motion is produced by contact forces, which a kinematic QP does not represent. Tell-tale symptom — strengthening a manipulation task makes the robot fall *sooner*. Solve at the acceleration level with contact wrenches as decision variables (Lab 8 `wb_id_qp.py`, L-M1-a).

### Gravity compensation alone cannot hold a standing humanoid
`τ = g(q)` cancels weight but adds no posture stiffness — a standing robot is an inverted pendulum and collapses in ~2 s. Position servos hide this behind their internal PD. When porting position→torque control, enumerate every stabilising term the servo provided and re-supply it. While in contact, prefer contact-consistent gravity (`g(q) − τ_constraint`) over free-space `g(q)`.

### MuJoCo freejoint body qpos layout
After arm joints (6) and gripper joints (2 with equality → 2 in qpos), freejoint occupies qpos[8:15] (3 pos + 4 quat). Equality constraint does NOT reduce qpos size. Verify with `mj_model.nq`.

### Gripper minimum gap must be less than object half-width
Compute `pad_inner_face = finger_body_y + pad_y_offset - pad_half_size` and verify < `object_half_width`. Test by checking `data.ncon` in a static scene.

### `is_gripper_in_contact` must check all finger geoms
The structural finger body geom contacts the object before the smaller pad geom. Check both in the contact loop.

### Contact tests must check during closing, not after settling
A free-flying box falls to floor in ~1s without gravity comp. Break-and-check inside the step loop.

### `parameterize_topp_ra` returns 4-tuple
Unpack as `times, q_traj, qd_traj, _ = parameterize_topp_ra(...)`. Fourth element (accelerations) is often unused.

### UR5e URDF joint naming
Standardize on mujoco_menagerie naming convention. Print `model.names` on first load to verify.

---

## Lab Progress

Published (portfolio-ready, documented in main README):
- [x] Lab 1: 2-Link Planar Arm (square drawing demo)
- [x] Lab 2: UR5e 6-DOF Arm (cube drawing demo)
- [x] Lab 3: Dynamics & Force Control (gravity comp, Cartesian impedance, hybrid force control on a real table-contact scene)
- [x] Lab 4: Motion Planning & Collision Avoidance (from-scratch RRT/RRT*, real-geometry collision truth, shortcutting + TOPP-RA, slalom capstone)
- [x] Lab 5: Grasping & Manipulation (custom MJCF jaw gripper, DLS IK, 11-state pick-and-place, Lab 3+4 integration). Phase 5 hardening closed 2026-08-13 (SO3-log IK, RRT* integration, pro demo re-recorded with 0 self-collisions). Step 6.1 capstone transport fixed same day: gripper friction pads were mounted on the OUTSIDE of the fingers (model bug) + 5 controller/planning fixes (inertia-scaled PD via crba, MJCF-built pin model, scene-derived collision truth, convergence-gated handoffs, touchdown-stop + ascend-before-retract). Capstone places box 5.7 mm from target with a transport post-condition assert.
- [x] Lab 6: Dual-Arm Coordination (two UR5e arms, Pinocchio dual-arm DLS IK, weld-constraint cooperative carry, milestone-gated verification M0-M5)
- [x] Lab 7: Locomotion Fundamentals (Unitree G1, floating-base Pinocchio, stacked-Jacobian whole-body IK, standing + weight shift on M3d scope; M4 ZMP walking deferred as structural limitation of position actuators)
- [x] Lab 8: Whole-Body Loco-Manipulation (torque-actuated G1, whole-body inverse-dynamics QP,
      DCM walking, centroidal angular-momentum control, walk→pick→carry→place capstone).
      Milestone-gated M0–M6, all passed; closed 2026-08-17; 97 tests. Two results
      reported honestly rather than claimed: M4's *moving*-hand sub-task is exploratory
      (it does not survive a no-op perturbation — see L-M4-f), and the QP is solved at
      the acceleration level rather than the velocity level `plan/LAB_08.md` specifies.

In progress (real work on disk, not yet portfolio-ready):
- *(none)*

Future (no folder yet — planned in main README roadmap only):
- [ ] Lab 9: VLA Integration — depends on Lab 8's controllers for demonstration data
      (`m3_walking.py`, `m4_walk_reach.py`, `m5_capstone.py`)

Platform transitions: Labs 1 uses custom 2-link. Labs 2–6 use UR5e + Robotiq 2F-85. Labs 7+ use Unitree G1 humanoid.

---

## Debugging Checklist

When Pinocchio and MuJoCo disagree:
1. Joint angle ordering — same convention?
2. Frame/body ID mapping — print names from both
3. Quaternion convention — (w,x,y,z) vs (x,y,z,w)
4. Gravity direction — matches in both models?
5. Units — Pinocchio uses SI, verify MuJoCo model does too

---

## Session Start Protocol

1. Read this CLAUDE.md
2. Read the lab brief: `plan/LAB_XX.md`
3. Check `lab-N-<name>/tasks/TODO.md` for current state
4. Check `lab-N-<name>/tasks/LESSONS.md` for known issues
5. Resume from "Current Focus" in TODO.md


## Lab 6 Rules
- ONE milestone per session. Do NOT proceed to next milestone.
- Every milestone ends with: gate criteria check + evidence (screenshot/plot/table).
- If gate fails, fix THIS milestone. Do not work around it.
- Controller mode changes per state (see state-controller map below).
- No impedance control for large motions. Joint PD only until within 10cm of target.

## Lab 7 Rules
- ONE milestone per session. Do NOT proceed to next milestone.
- Every milestone ends with: gate criteria check + evidence.
- Evidence MUST include video (media/mN_*.mp4) or screenshot (media/mN_*.png) for every milestone. No exceptions.
- If gate fails, fix THIS milestone. Do not work around it.
- Print numerical gate results in a table format.
- Do NOT implement anything beyond current milestone scope.

## Pinocchio Rules (MANDATORY)
- All foot/CoM control MUST go through Pinocchio IK. No open-loop joint offsets.
- If Pinocchio gives unexpected results, DEBUG THE FRAME CONVENTION. Do NOT replace with open-loop hacks.
- Always use pin.LOCAL_WORLD_ALIGNED for Jacobians (not LOCAL, not WORLD)
- Floating base: nq != nv (quaternion vs tangent). Use pin.integrate() for configuration updates, NEVER q += dq.
- Validate every Jacobian column with finite differences: perturb joint i by eps=1e-6, recompute FK, compare (FK_new - FK_old)/eps with Jacobian column i. If signs don't match, your frame is wrong.

## Domain knowledge retrieval

When stuck on a robotics, controls, or simulation problem (unexpected
behavior, cryptic error, physics that looks wrong), before web search
or guessing:

1. rg -i "<distinctive error token>" /opt/data/wiki-lessons/
2. No hit: read /opt/data/wiki-lessons/INDEX.md and match your symptom against
   the one-line rules.
3. Open the matching note. Apply the Rule. Respect Scope and limits.
4. Cite the lesson id in your output when you use one.