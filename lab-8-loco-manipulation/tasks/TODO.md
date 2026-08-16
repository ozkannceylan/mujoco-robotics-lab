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

## M3 — Forward Walking (retires Lab 7's deferred capstone) — ✅ DONE (2026-08-16), GATE PASSED 4/4
- [x] Step 3.1: `src/dcm_planner.py` — piecewise-linear ZMP through the footsteps
      (hold → sweep onto stance → hold at the stance foot → sweep to the next),
      with the DCM back-integrated from a terminal rest condition. Backward is
      the only stable direction for an unstable system: it contracts the
      boundary error by `e^{−ωT}` where forward integration amplifies it.
      Segments are deliberately not one-per-gait-phase (L-M3-h).
- [x] Step 3.2: `wb_tasks.DCMTask` — commands `c̈ = ω²(c − p_cmd)` from
      `p_cmd = ξ − ξ̇_ref/ω + (k/ω)(ξ − ξ_ref)`, clamped into the stance feet.
      **The CoM position task is off the control path entirely**: nothing tells
      the robot where its CoM should be, only where its divergent component is
      going. Optional leaky integrator, defaulting off (L-M3-g).
- [x] Step 3.3: `gait_planner` gains `step_width` — the dominant gait parameter
      (L-M3-f). `lab8_common` gains `lipm_omega` / `divergent_component`.
- [x] Step 3.4: **Foot contact model corrected** (L-M3-d). `ContactSpec` now
      describes the real Menagerie G1 sole (x ∈ [−0.05, 0.12], y ∈ ±0.025,
      35 mm below the frame) instead of a symmetric ±0.08 guess, and the CoP
      rows carry the `h·f` shear term. This, not the control law, was the
      single biggest contributor.
- [x] Step 3.5: **QP solver tolerance fixed** (L-M3-e). 38 % of ticks were
      hitting the iteration cap at 12.6 ms; now 100 % converge in ~25
      iterations and 0.073 ms, with a *smaller* constraint residual.
- [x] Step 3.6: `m3_walking.py` — 12 steps × 10 cm, plus `--in-place` to re-run
      M2's gate through the DCM controller.
- [x] Tests: `tests/test_dcm.py` — 35 tests (LIPM primitives, segment algebra
      against the DCM ODE, plan continuity/terminal rest/foot-patch targeting,
      the control law reducing to the planned ZMP under perfect tracking, VRP
      clamping, and the CoP box against the real foot geometry). Lab 8 total
      **97 passed**.
- [x] Gate — all criteria PASS:

      | criterion | result | measured |
      |---|---|---|
      | ≥10 steps without falling | PASS | **12/12** |
      | Travelled ≥ 1.0 m | PASS | **1.18 m** |
      | ZMP inside support > 90 % | PASS | 99.3 % |
      | Torques within limits | PASS | 56.0 N·m (limit 139) |

- [x] Regressions: M2's in-place gate re-run under the DCM controller; M0 and
      M1 re-run after the QP changes. M2 *improved* (ZMP 98.7 % → 100 %).
- [x] Evidence: `media/m3_walking.mp4` + `media/m3_walking_metrics.png`
- [x] Lessons: L-M3-c (read saturation before error when a new controller
      underperforms), **L-M3-d (the contact model was the real bug)**,
      L-M3-e (a tighter QP tolerance made the answer worse), L-M3-f (stance
      width dominates stride length), L-M3-g (the integrator became harmful
      once the cause was fixed), L-M3-h (a well-motivated fix to the plan's
      initial condition halved the gait — a dynamic reference is allowed to
      want the robot already moving).

## M4 — Walk + Arm Task — 🚧 IN PROGRESS (carry PASSES 5/5, reach open)
- [x] `src/m4_walk_reach.py` — walks M3's gait with both hands on a Cartesian
      task. Two sub-tasks per PLAN: **carry** (both hands hold a pose fixed in
      the walking frame) and **reach** (right hand traces a circle).
- [x] `wb_tasks.CentroidalAngularMomentumTask` — regulates `L = A_g q̇` toward
      zero. **This is the milestone's real content** (L-M4-c): without it no
      hand weight both walks and tracks; with it the hand task that used to
      fall on step 7 walks all twelve and tracks three times better.
- [x] `m3_walking.make_plan` split out of `build` so the gait can be replanned
      after the robot changes posture (L-M4-a).
- [x] Carry pose reached under the whole-body QP with the DCM frozen, then the
      plan rebuilt on the achieved configuration.
- [x] Hand reference carries the plan's own CoM velocity **and acceleration**
      (`c̈ = ω²(c − p)`), not just position.
- [x] **Carry gate — 5/5 PASS**:

      | criterion | result | measured |
      |---|---|---|
      | ≥10 steps, no fall | PASS | **12/12** |
      | Travelled ≥ 1.0 m | PASS | 1.170 m |
      | ZMP inside support > 90 % | PASS | 99.1 % |
      | Hand error < 50 mm walking | PASS | **14.5 mm RMS, 25.7 mm max** |
      | Torques within limits | PASS | 55.2 N·m peak |

- [ ] **Reach gate NOT passed**: best 6 of 12 steps. Circle speed is not the
      cause (2/3/4 s periods, 0.08/0.10 m radii all fall at 3–4 steps); the
      lateral→sagittal plane change helped (4 → 6 steps) but did not close it.
- [ ] **Next implementation step**: give the momentum task a *reference* rather
      than zero. A deliberately moving hand generates angular momentum by
      design, so commanding `L → 0` fights the task itself; the reference
      should be the momentum the planned arm motion implies. Then re-check the
      carry gate, which must not regress.
- [x] Evidence (carry): `media/m4_walk_reach.mp4` + `media/m4_hand_error.png`
- [x] Lessons: L-M4-a (replan on the posture you will walk in; reach it with
      the QP, not joint PD), L-M4-b (a no-variance offset is two tasks fighting
      — but check what the losing side was buying), **L-M4-c (centroidal
      angular momentum is the missing term, not a gain)**.

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
> **M4 — Walk + Arm Task, continued.** The carry half is done and passes 5/5
> (12 steps, 1.17 m, hand 14.5 mm RMS). The reach half — right hand circling
> while walking — still falls at 6 of 12 steps.
>
> Start at the momentum **reference**, not at more tuning; the sweeps already
> run rule tuning out (hand weight, hand gain, carry offset, circle period and
> radius, circle plane — every one non-monotonic or downhill):
> 1. `CentroidalAngularMomentumTask` currently commands `L → 0`. A hand that is
>    deliberately moving generates angular momentum on purpose, so this term is
>    now fighting the very task it was added to enable.
> 2. Compute the reference momentum the commanded arm trajectory implies —
>    `A_g(q) q̇_ref` restricted to the arm columns is the cheap version — and
>    regulate `L → L_ref`.
> 3. Re-run the carry gate afterwards. It must still be 5/5; a regression there
>    means the reference is wrong rather than merely differently tuned.

## Blockers
> None. OSQP verified (1.1.3). G1 menagerie assets present under
> `third_party/mujoco_menagerie/unitree_g1/` (fresh clones: run `tools/setup_env.sh`).
