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
- Labs 6–9 are planned — dual-arm, locomotion, whole-body, VLA
- End goals: strengthen fundamentals for humanoid VLA work, prepare for robotics interviews, build a portfolio demo

---

## Common Commands

### Install dependencies

```bash
pip install mujoco numpy pinocchio scipy imageio[ffmpeg] matplotlib meshcat
```

### Run tests

```bash
# All tests for a specific lab
pytest lab-3-dynamics-force-control/tests/

# Single test file
pytest lab-4-motion-planning/tests/test_collision.py

# Single test method
pytest lab-5-grasping-manipulation/tests/test_gripper.py::TestGripperContact::test_contact_detection -v

# All tests across the project
pytest lab-*/tests/
```

No pytest config files — uses defaults. Tests use both `unittest.TestCase` and pure pytest fixtures.

### Run demos

Each lab has numbered scripts (a1, a2, b1, c1, etc.) that run in order:

```bash
python3 lab-1-2link-arm/src/c1_draw_square.py        # Lab 1 capstone
python3 lab-2-Ur5e-robotics-lab/src/c3_draw_cube.py   # Lab 2 capstone
python3 lab-3-dynamics-force-control/src/c1_force_control.py
python3 lab-4-motion-planning/src/capstone_demo.py
python3 lab-5-grasping-manipulation/src/record_pro_demo.py
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
# In lab4_common.py — importing from Lab 3
_LAB3_SRC_DIR = PROJECT_ROOT / "lab-3-dynamics-force-control" / "src"
if str(_LAB3_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB3_SRC_DIR))

from lab3_common import (
    DT, NUM_JOINTS, Q_HOME,
    load_pinocchio_model as load_lab3_pinocchio_model,
)
```

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

### Cross-lab imports need sys.path
Each lab module importing from another lab must add the foreign `src/` to `sys.path` using `Path(__file__).resolve()` and conditional `sys.path.insert(0, ...)`.

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

- [x] Lab 1: 2-Link Planar Arm (square drawing demo)
- [x] Lab 2: UR5e 6-DOF Arm (cube drawing demo)
- [x] Lab 3: Dynamics & Force Control (gravity comp, Cartesian impedance, force control)
- [x] Lab 4: Motion Planning & Collision Avoidance (RRT*, TOPP-RA, capstone demo)
- [x] Lab 5: Grasping & Manipulation (custom gripper, DLS IK, pick-and-place state machine)
- [ ] Lab 6: Dual-Arm Coordination
- [ ] Lab 7: Locomotion Fundamentals
- [ ] Lab 8: Whole-Body Loco-Manipulation
- [ ] Lab 9: VLA Integration

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