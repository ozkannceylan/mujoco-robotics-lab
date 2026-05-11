# Lab 5: Grasping & Manipulation

A pick-and-place pipeline for the **UR5e + custom parallel-jaw gripper** in MuJoCo. The arm picks a 40 mm cube from table position A and places it at position B by chaining Lab 3 (gravity-compensated impedance control) and Lab 4 (RRT\* planning + smoothing) through an 11-state grasp state machine.

## Showcase

[`media/pick_place_demo.mp4`](media/pick_place_demo.mp4) — full pick-and-place cycle.

> The capstone runs the full pick-and-place cycle on a 150 g, 40 mm cube. DLS IK computes 4 grasp configurations (pregrasp, grasp, preplace, place), the state machine drives the gripper open/close + contact detection through 11 states, and Lab 3's `compute_impedance_torque` executes each segment under gravity compensation.

> **Status note**: The core pick-and-place pipeline is **complete and tested** (Phases 1-4). A *pro demo hardening* phase remains open — see [Pending Work](#pending-work) below.

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

The core pick-and-place pipeline is complete and shipped above. A separate **Phase 5 — Pro Demo Hardening** track remains open in [`tasks/TODO.md`](tasks/TODO.md):

- **5.1** — fix the IK orientation error formula in `record_pro_demo.py` (currently uses an older form, not the SO(3) `log3` formulation already adopted in `grasp_planner.py`)
- **5.2** — integrate Lab 4's RRT\* path through `record_pro_demo.py` for collision-free pro-demo planning
- **5.4** — re-record `pick_place_pro.mp4` once 5.1 + 5.2 land

These items do not affect the capstone `pick_place_demo.py`, the test suite, or any of the metrics in the table above. They harden the pro-demo recorder specifically.

---

## Notes

- The gripper minimum gap must be verified `< object_half_width` before any test runs. Compute `pad_inner_face = finger_body_y + pad_y_offset - pad_half_size`; for a 40 mm cube this passes with margin to spare.
- `is_gripper_in_contact` checks every finger geom, not only the pads — the structural finger body geom contacts the object first. Restricting the check to pads underreports contact during closing.
- Contact tests must check during closing, not after settling. A free-flying 150 g box drops to the floor in ~1 s if the arm releases gravity compensation, so the test breaks out of the closing loop as soon as a contact is registered.

---

## License

The Lab 5 source code and original documentation are covered by the repository root [Apache-2.0 license](../LICENSE).

Bundled robot description packages and model assets in [`models/`](models/) and the Menagerie assets reused from Lab 2 keep their upstream licenses. See the repository root [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the exact carve-outs.
