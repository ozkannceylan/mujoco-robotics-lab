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

## M1 — Whole-Body QP, Standing — ✅ DONE (2026-08-15), GATE PASSED 5/5
- [x] Step 1.1: `wb_tasks.py` — CoM / frame-position / frame-pose / posture tasks with
      LOCAL_WORLD_ALIGNED Jacobians, `J̇q̇` drift terms, and feedforward (ẋ_ref/ẍ_ref).
      All four Jacobians finite-difference validated in tests.
- [x] Step 1.2: **`wb_id_qp.py`** — acceleration-level inverse-dynamics QP in OSQP with
      contact wrenches as decision variables (47 vars: 35 q̈ + 12 wrench), unactuated
      base dynamics + stance constraint as equalities, friction pyramid / CoP /
      unilateral / torque limits as inequalities. Mean solve **0.11 ms**.
      **Major deviation**: the planned velocity-level `wb_qp.py` was written first and
      measured to be structurally unable to balance (L-M1-a) — it is kept for kinematic
      sub-problems only. This is the single most important result of M1.
- [x] Step 1.3: torque comes straight out of the ID QP's actuated rows, so the separate
      `inverse_dynamics.py` servo is no longer on the control path (it remains as the
      M0-style tracker, useful if a future milestone needs velocity-command tracking).
- [x] Step 1.4: `m1_standing_reach.py` — 11 s demo: settle → reach → 2 laps of a 10 cm
      hand circle, both feet planted.
- [x] Tests: `tests/test_wb_tasks.py` — 24 tests (FD Jacobians, drift, feedforward,
      frame conventions, stack assembly, QP force balance / friction / CoP / torque
      limits / contact constraint). Lab 8 total **42 passed**.
- [x] Gate — all criteria PASS:

      | criterion | result | measured |
      |---|---|---|
      | No fall | PASS | stood 11 s |
      | Hand tracking RMS < 20 mm | PASS | **7.08 mm** |
      | CoM margin ≥ 20 mm | PASS | 51.7 mm min |
      | Stance feet move < 5 mm | PASS | 2.19 mm |
      | Torques within limits | PASS | 12.0 N·m peak (limit 139) |

- [x] Evidence: `media/m1_standing_reach.mp4` + `media/m1_reach_metrics.png`
- [x] Lessons: L-M1-a (kinematic QP cannot balance — the milestone's core finding),
      L-M1-b (feedforward cut hand RMS 18.63 → 7.08 mm)

## M2 — Torque-Level Stepping — ✅ DONE (2026-08-15), GATE PASSED 3/3
- [x] Step 2.1: `gait_planner.py` — `GaitSchedule` produces the phase timeline,
      contact sets, swing-foot references (position + velocity + acceleration
      feedforward) and the CoM weight-shift target. Reuses Lab 7's swing-arc idea
      but generates the CoM reference directly rather than through the LIPM preview
      controller — the QP already enforces balance via contact wrenches and CoP, so
      a second planner would only add a way for the two to disagree on phase timing.
- [x] Step 2.2: contact scheduling — `WholeBodyIDQP.set_contacts()` resizes the
      problem per phase; `SteppingController` intersects the scheduled stance with
      **measured** contact (L-M2-a) and ramps the swing task in at lift-off.
- [x] Step 2.3: `m2_stepping.py` — settle → 4 alternating in-place steps.
- [x] New task type: `FrameOrientationTask` (pelvis/torso), plus `measured_zmp()`
      and `point_in_support_polygon()` in `lab8_common`.
- [x] Tests: `tests/test_gait.py` — 20 tests (phase sequencing, contact sets never
      empty, swing continuity/clearance/zero-touchdown-velocity, feedforward vs
      finite differences, CoM shift direction and continuity, ZMP measurement,
      contact-set resizing). Lab 8 total **62 passed**.
- [x] Gate — all criteria PASS:

      | criterion | result | measured |
      |---|---|---|
      | 4 steps without falling | PASS | **4/4** |
      | ZMP inside support polygon > 95 % | PASS | **98.7 %** |
      | Torques within limits | PASS | 49.6 N·m peak (limit 139) |

- [x] Evidence: `media/m2_stepping.mp4` + `media/m2_stepping_metrics.png`
- [x] Lessons: L-M2-a (contact set must match measured contact), **L-M2-b (commanding
      CoM *height* saturated the waist actuators — the decisive fix, 3/4 → 4/4)**,
      L-M2-c (two plausible improvements that made it worse), L-M2-d (settle before
      measuring home poses).

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
> **M3 — Forward Walking** (≥ 10 steps, ≥ 1.0 m). `GaitSchedule` already takes a
> `step_length`, and a forward stride is covered by a test, so M3 is mostly about
> compressing the timeline: M2's 2.0 s double support / 0.5 s swing is quasi-static.
>
> What M2 leaves for M3 to solve:
> - The weight-shift strategy (move the CoM over the stance foot, *then* swing) has
>   a deadline problem as timing tightens — over-damping already failed for exactly
>   this reason (L-M2-c). Capture-point / DCM tracking is the expected replacement.
> - Torque headroom is now large (49.6 of 139 N·m), so the budget for faster motion
>   exists — provided no task re-introduces an over-specified demand (L-M2-b).
> - Contact switching is measured-contact driven and stable across 16 switches;
>   forward stepping adds touchdown at a *new* location, so landing detection will
>   matter more than it did stepping in place.

## Blockers
> None. OSQP verified (1.1.3). G1 menagerie assets present under
> `third_party/mujoco_menagerie/unitree_g1/` (fresh clones: run `tools/setup_env.sh`).
