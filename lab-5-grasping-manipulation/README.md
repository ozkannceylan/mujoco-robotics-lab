# Lab 5: Grasping & Manipulation

A pick-and-place pipeline for the **UR5e + custom parallel-jaw gripper** in MuJoCo. The arm picks a 40 mm cube from table position A and places it at position B by chaining Lab 3 (gravity-compensated impedance control) and Lab 4 (RRT\* planning + smoothing) through an 11-state grasp state machine.

## Showcase

[`media/pick_place_demo.mp4`](media/pick_place_demo.mp4) — full pick-and-place cycle.

> The capstone runs the full pick-and-place cycle on a 150 g, 40 mm cube. DLS IK computes 4 grasp configurations (pregrasp, grasp, preplace, place), the state machine drives the gripper open/close + contact detection through 11 states, and Lab 3's `compute_impedance_torque` executes each segment under gravity compensation.

> **Status note**: Complete. The core pick-and-place pipeline (Phases 1-4) and the pro-demo hardening track (Phase 5) are both finished and tested — see [Pending Work](#pending-work) below for the Phase 5 close-out and the self-collision verification numbers.

## Key Results

| Metric | Value |
|---|---|
| Test suite | **33 passed** (3 files: gripper / planner / state machine) |
| IK position accuracy | < 0.1 mm |
| Joint tracking error | < 5 mrad |
| Gripper gap (open / closed) | 60 mm / 0 mm (pinch on a 40 mm cube) |
| Box mass | 150 g |
| Planning time per segment | 200–600 ms (RRT\*, 6000 iterations) |
| Pick success on the fixed scene | Reliable |

---

## Skills Demonstrated

- **Custom MJCF gripper from scratch**: parallel-jaw gripper with one position actuator, an equality constraint for finger symmetry, and contact-tuned pads (`condim`, friction, `solref`, `solimp`).
- **DLS IK with SO(3)-aware orientation**: damped least squares using `pin.log3` for the orientation error, no 180° singularity. Seeded across the 4 grasp poses so the wrist stays on one side.
- **Grasp state machine**: 11 states (HOME → APPROACH → DESCEND → GRIP → SETTLE → LIFT → CARRY → PREPLACE → PLACE → RELEASE → RETURN) with contact-tracked closing across the full settle window.
- **Cross-lab integration**: planning consumes Lab 4's `RRTStarPlanner` + `shortcut_path` + `parameterize_topp_ra`; execution consumes Lab 3's `compute_impedance_torque` + `ImpedanceGains`. Pinocchio is the analytical brain; MuJoCo is the simulator.
- **Real contact pipeline**: `is_gripper_in_contact` checks every finger geom (structural body + pad — the pad isn't always the first to touch), and the contact test runs during closing rather than after a 1 s settle so the box doesn't fall first.

---

## Architecture

```text
Pinocchio (FK, DLS IK)
        ↓ GraspConfigs (5 × q)
GraspStateMachine (11 states)
        ↓ for each PLAN_* state:
Lab 4 RRT* + shortcutting           → waypoints
Lab 4 parameterize_topp_ra          → (t, q, q̇, q̈)
        ↓ for each EXEC_* state:
Lab 3 compute_impedance_torque      → τ = Kp·Δq + Kd·Δq̇ + g(q)
        ↓
MuJoCo mj_step()
        ↓
qpos, qvel, contact forces, gripper state
```

Pinocchio handles all analytical computation (FK, IK, Jacobians, gravity). MuJoCo handles simulation and contact. Lab 3 and Lab 4 modules are imported via `add_lab_src_to_path()` in `lab5_common.py` — no logic is duplicated across labs.

---

## Modules

| File | Role |
|---|---|
| `src/lab5_common.py` | Paths, constants, MuJoCo + Pinocchio loaders, cross-lab path helpers |
| `src/gripper_controller.py` | `open_gripper` / `close_gripper` / `settle` / `is_gripper_in_contact` |
| `src/grasp_planner.py` | DLS IK, `GraspConfigs` dataclass, `compute_grasp_configs` |
| `src/grasp_state_machine.py` | 11-state pick-and-place orchestrator with Lab 3 + Lab 4 integration |
| `src/pick_place_demo.py` | Capstone demo — full cycle, plots to `media/` |
| `src/record_pro_demo.py` | Pro-demo recorder (Phase 5 work-in-progress, see below) |

---

## Quick Start

```bash
# From the repository root
pip install mujoco numpy pinocchio scipy "imageio[ffmpeg]" matplotlib
# toppra is optional — Lab 4 falls back to a quintic time-parameterization when not installed

# Run the full test suite (33 tests)
python3 -m pytest lab-5-grasping-manipulation/tests -q

# Run the pick-and-place capstone demo
python3 lab-5-grasping-manipulation/src/pick_place_demo.py
```

---

## Scene Constants

| Constant | Value | Description |
|---|---|---|
| `BOX_A_POS` | [0.35, +0.20, 0.335] m | Pick location |
| `BOX_B_POS` | [0.35, -0.20, 0.335] m | Place location |
| `GRIPPER_TIP_OFFSET` | 0.105 m | tool0 origin → fingertip pad center |
| `PREGRASP_CLEARANCE` | 0.150 m | Approach height above the box |
| `GRIPPER_OPEN` | 0.030 m on `ctrl[6]` | Finger slide open setpoint |
| `GRIPPER_CLOSED` | 0.000 m on `ctrl[6]` | Finger slide closed setpoint |
| `TABLE_TOP_Z` | 0.315 m | Table surface world Z |

---

## Cross-Lab Dependencies

| Component | Source |
|---|---|
| `ur5e.urdf` (Pinocchio model) | `lab-3-dynamics-force-control/models/` |
| `compute_impedance_torque`, `ImpedanceGains` | `lab-3-dynamics-force-control/src/` |
| `CollisionChecker` | `lab-4-motion-planning/src/` |
| `RRTStarPlanner`, `shortcut_path` | `lab-4-motion-planning/src/` |
| `parameterize_topp_ra` | `lab-4-motion-planning/src/` |

---

## Structure

```text
lab-5-grasping-manipulation/
├── src/              Source modules + capstone + pro-demo recorder
├── models/           ur5e_gripper.xml (UR5e + jaw gripper) + scene_grasp.xml (table + box)
├── docs/             English study notes (01–04)
├── docs-turkish/     Turkish study notes
├── blog/             Long-form blog post
├── media/            pick_place_demo.mp4, pick_place_pro.mp4
├── tests/            Pytest suite (33 tests across 3 files)
└── tasks/            PLAN / ARCHITECTURE / TODO / LESSONS
```

---

## Documentation

| Topic | English | Turkish |
|---|---|---|
| 01 — Contact physics | [Contact Physics](docs/01_contact_physics.md) | [Temas Fiziği](docs-turkish/01_temas_fizigi.md) |
| 02 — Gripper design | [Gripper Design](docs/02_gripper_design.md) | [Tutucu Tasarımı](docs-turkish/02_tutucu_tasarimi.md) |
| 03 — Grasp pipeline | [Grasp Pipeline](docs/03_grasp_pipeline.md) | [Kavrama Pipeline](docs-turkish/03_kavrama_pipeline.md) |
| 04 — Results | [Pick-and-Place Results](docs/04_pick_place_results.md) | [Al-Yerleştir Sonuçları](docs-turkish/04_al_yerlesir_sonuclari.md) |

Blog post: [`blog/lab5_blog_post.md`](blog/lab5_blog_post.md).

---

## Pending Work

**Phase 5 — Pro Demo Hardening is complete** (2026-08-13). All four items landed:

- **5.1** — SO(3) `log3` orientation error in `record_pro_demo.py` (`_so3_log`, no 180° singularity)
- **5.2** — Lab 4 RRT\* + shortcutting drives all four long-distance transitions
- **5.3** — matplotlib 3D import guarded in Lab 4's `rrt_planner.py`
- **5.4** — `pick_place_pro.mp4` re-recorded (1280×720, 60 fps, 23.1 s) and verified self-collision-free

### Self-collision verification

`record_pro_demo.py` carries a `SelfCollisionMonitor` that inspects every contact after **every**
simulation step (not every rendered frame — at 60 fps only 1 step in 8 is drawn, so a brief
interpenetration could otherwise slip between frames). Geoms are grouped by parent body into
`arm`, `grip` and `env`; an `arm↔arm` or `arm↔grip` contact is a self-collision, while the
2F-85's own linkage contacts are tallied separately. The recording run reports:

| Metric | Value |
|---|---|
| Simulation steps checked | 11050 |
| Steps with self-collision | **0** |
| Max penetration depth | 0.000 mm |
| Robot↔table contact pairs | 0 |

The script exits non-zero if this check ever fails, so the guarantee is re-checked on every
re-record rather than asserted once.

### Resolved (2026-08-13, same day) — capstone box transport (Step 6.1)

The capstone `pick_place_demo.py` used to reach `DONE` without moving the box
(400 mm lateral error, gripper closing on air). Root-causing it uncovered six
stacked defects — the deepest being a **model bug: the gripper's friction pads
were mounted on the outside of the fingers** and never touched the object, so
every grasp ran on the low-friction structural finger geoms and the box crept
out during transport. Full postmortem: `tasks/LESSONS.md` § "Step 6.1 Session".

The fixes (pads flipped inward; inertia-scaled joint PD via `pin.crba`;
scene-matched analytical model — `load_pinocchio_model(match_scene_inertias=True)`;
unified IK/planner collision truth via `SceneCollisionChecker`; convergence-gated
state handoffs with absolute 6D Cartesian targets; touchdown-stop on place and
vertical ascend before retract) close the loop end to end:

| Gate | Result |
|---|---|
| Box final position | `[0.350, −0.194, 0.335]` m |
| **Lateral error to Box B** | **5.7 mm** (tolerance 30 mm) |
| Joint settle residual (every state) | 10.0 mrad |
| Descend / lift / place settles | 2.4 / 3.0 / 4.1 mm |
| Full cycle | 34.7 s, `transport_ok = True`, exit 0 |
| Test suite | 33/33 passed |

`GraspStateMachine.run()` now returns `transport_ok` / `box_lateral_error_mm`
and both demo scripts exit non-zero on a failed transport — the state machine
can no longer claim success without the box actually arriving.

This does **not** affect the Phase 5 pro-demo result above. It does mean the "Pick success on the
fixed scene: Reliable" row in [Key Results](#key-results) currently overstates the capstone, and a
post-condition assertion on final box position should be added so a silent miss cannot pass again.

---

## Notes

- The gripper minimum gap must be verified `< object_half_width` before any test runs. Compute `pad_inner_face = finger_body_y + pad_y_offset - pad_half_size`; for a 40 mm cube this passes with margin to spare.
- `is_gripper_in_contact` checks every finger geom, not only the pads — the structural finger body geom contacts the object first. Restricting the check to pads underreports contact during closing.
- Contact tests must check during closing, not after settling. A free-flying 150 g box drops to the floor in ~1 s if the arm releases gravity compensation, so the test breaks out of the closing loop as soon as a contact is registered.

---

## License

The Lab 5 source code and original documentation are covered by the repository root [Apache-2.0 license](../LICENSE).

Bundled robot description packages and model assets in [`models/`](models/) and the Menagerie assets reused from Lab 2 keep their upstream licenses. See the repository root [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the exact carve-outs.
