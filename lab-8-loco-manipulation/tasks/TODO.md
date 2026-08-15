# Lab 8: TODO

> Kickoff: 2026-08-14. Rules: ONE milestone per session; every milestone ends with a
> gate check + evidence in `media/mN_*`; if a gate fails, fix THIS milestone.

## Pre-M0 (kickoff session) — DONE 2026-08-14
- [x] Read `plan/LAB_08.md` fully
- [x] Create lab folder structure (tasks/, src/, models/, docs/, docs-turkish/, media/, tests/)
- [x] Write `tasks/PLAN.md` (milestone-gated M0–M6; gait ownership deviation documented)
- [x] Write `tasks/ARCHITECTURE.md` (module map, data flow, interfaces, cross-lab deps)
- [x] Verify OSQP availability — osqp 1.1.3 installs and imports cleanly
- [x] Update MASTER_PLAN / root status board / CLAUDE.md / plan/LAB_08.md status header

## M0 — Torque-Actuated G1 Bring-Up — ✅ DONE (2026-08-15), GATE PASSED 7/7
- [x] Step 0.1: torque-actuated G1 — `src/g1_torque_model.py` converts Menagerie's 29
      `<position>` servos to `<motor>` actuators via `MjSpec.set_to_motor()`, with
      ctrlrange taken from each joint's `actuatorfrcrange` (5–139 N·m) and the stand
      keyframe's position-target ctrl zeroed. **Deviation from PLAN**: built
      programmatically instead of as a committed `models/g1_torque.xml` — keeps
      Menagerie as the single source of truth, avoids the meshdir shim Lab 2 needed,
      and matches the `build_mujoco_scene_spec` convention of Labs 3–4. `export_xml()`
      can still emit a snapshot for inspection.
- [x] Step 0.2: `src/lab8_common.py` — torque-model + floating-base Pinocchio loaders,
      Lab 7 conventions re-exported, state conversion (incl. the world→body base-twist
      rotation), CoM / contact / support-polygon helpers, MuJoCo-3.11-safe `mj_fullM`.
- [x] Step 0.3: `src/standing_controller.py` — joint PD + selectable gravity mode;
      `src/m0_torque_standing.py` runs cross-validation, the gravity-mode ablation and
      the recorded gate run.
- [x] Tests: `tests/test_torque_model.py` — 18 tests (actuator semantics, torque limits,
      ctrl→force mapping, keyframe hygiene, g/M/CoM parity, base-velocity frame
      conversion, zero-torque-falls baseline). Also fixed the repo-wide
      `pytest lab-*/tests/` collision via a root `conftest.py` → **224 passed** across
      Labs 1–5, 7, 8 (see L-M0-e).
- [x] Gate — all criteria PASS:

      | criterion | result | measured |
      |---|---|---|
      | Stands 10 s without falling | PASS | no fall |
      | CoM horizontal drift < 30 mm | PASS | 0.71 mm |
      | Both feet in contact at end | PASS | yes |
      | CoM inside support polygon | PASS | 52.7 mm margin |
      | g(q) parity < 1e-6 relative | PASS | 1.74e-16 |
      | M(q) parity < 1e-6 relative | PASS | 9.32e-17 |
      | Torque command authority | PASS | motor actuators |

- [x] Gravity-mode ablation (the milestone's real finding, 10 s hold each):

      | mode | result | CoM drift | steady joint err | \|τ\|max |
      |---|---|---|---|---|
      | none (pure PD) | STAND | 0.18 mm | 2.77 mrad | 1.4 N·m |
      | free-space g(q) | STAND | 0.96 mm | 1.40 mrad | 1.6 N·m |
      | contact-consistent | STAND | 0.62 mm | **0.00 mrad** | 3.9 N·m |
      | *(g(q) alone, no PD)* | *FELL* | — | collapses to 0.097 m in ~2 s | — |

- [x] Evidence: `media/m0_torque_standing.mp4` + `media/m0_standing_metrics.png`
- [x] Lessons logged: L-M0-a … L-M0-d in LESSONS.md — notably **inherited lesson I-3
      (inertia-shaped PD) is invalid on a floating base** and was overturned by measurement.

## M1 — Whole-Body QP, Standing
- [ ] Step 1.1: `wb_tasks.py` (CoM / foot / hand / posture; FD-validated Jacobians + tests)
- [ ] Step 1.2: `wb_qp.py` (OSQP, weighted lexicographic stack, limit constraints)
- [ ] Step 1.3: `inverse_dynamics.py` (RNEA + **raw** joint-space PD — M0's L-M0-b
      showed M(q)-shaped gains destabilise a floating base)
- [ ] Step 1.4: standing reach demo
- [ ] Gate: hand RMS < 20 mm; CoM ≥ 20 mm inside support polygon; feet < 5 mm motion
- [ ] Evidence: `media/m1_standing_reach.mp4` + plots

## M2 — Torque-Level Stepping
- [ ] LIPM refs (adapt Lab 7 planner) + contact schedule + swing-foot task
- [ ] Gate: 4 in-place steps, ZMP inside support polygon > 95% of stance
- [ ] Evidence: `media/m2_stepping.mp4` + ZMP plot

## M3 — Forward Walking (retires Lab 7's deferred capstone)
- [ ] Gate: ≥ 10 steps, ≥ 1.0 m, no fall
- [ ] Evidence: `media/m3_walking.mp4` + stride/ZMP plots

## M4 — Walk + Arm Task
- [ ] Gate: M3 gate holds AND hand error < 50 mm while walking
- [ ] Evidence: `media/m4_walk_reach.mp4` + hand-error plot

## M5 — Loco-Manipulation Capstone
- [ ] WALK→REACH→GRASP(weld)→CARRY→PLACE sequence; payload mass in CoM model
- [ ] Gate: no fall; object within 50 mm of target; object-pose post-condition assert
- [ ] Evidence: `media/m5_capstone.mp4` + gate table

## M6 — Documentation & Blog
- [ ] docs/ + docs-turkish/ (ARCHITECTURE + CODE_WALKTHROUGH pattern)
- [ ] README with per-milestone evidence tables
- [ ] Blog post (do NOT defer — Labs 3–4 lesson)
- [ ] Root README / MASTER_PLAN / status board / plan/LAB_08.md updates

## Current Focus
> **M1 — Whole-Body QP, Standing.** M0 is closed: torque authority is established
> and the analytical model is verified against the simulated body. Next session
> starts at Step 1.1 (`wb_tasks.py` — task residuals + finite-difference-validated
> Jacobians), then `wb_qp.py` (OSQP) and `inverse_dynamics.py`.
>
> Two M0 results carry directly into M1:
> - Use **raw** joint-space gains, not M(q)-shaped ones (L-M0-b).
> - M1's inverse dynamics must be **contact-consistent**; M0's mode of the same
>   name reads constraint forces from the simulator, whereas M1 should obtain
>   them from the QP's own contact-wrench variables.

## Blockers
> None. OSQP verified (1.1.3). G1 menagerie assets present under
> `third_party/mujoco_menagerie/unitree_g1/` (fresh clones: run `tools/setup_env.sh`).
