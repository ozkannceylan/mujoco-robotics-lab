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

## M0 — Torque-Actuated G1 Bring-Up
- [ ] Step 0.1: `models/g1_torque.xml` — replace 29 position actuators with `<motor>`,
      document per-joint torque limits; meshes from third_party menagerie clone
- [ ] Step 0.2: `src/lab8_common.py` — loaders (MuJoCo + floating-base pin from the SAME
      MJCF), joint map, constants; reuse lab7_common where sensible
- [ ] Step 0.3: gravity-compensated torque-only standing (τ = RNEA(q,0,0), both feet down)
- [ ] Gate: 10 s stand, CoM drift < 30 mm, no fall; g_pin vs qfrc_bias cross-validation table
- [ ] Evidence: `media/m0_torque_standing.mp4` + gate table printout

## M1 — Whole-Body QP, Standing
- [ ] Step 1.1: `wb_tasks.py` (CoM / foot / hand / posture; FD-validated Jacobians + tests)
- [ ] Step 1.2: `wb_qp.py` (OSQP, weighted lexicographic stack, limit constraints)
- [ ] Step 1.3: `inverse_dynamics.py` (RNEA + inertia-shaped PD)
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
> **M0 — Torque-Actuated G1 Bring-Up.** Next session starts at Step 0.1
> (`models/g1_torque.xml`). Everything downstream depends on torque authority —
> this is the direct answer to Lab 7's position-actuator blocker.

## Blockers
> None. OSQP verified (1.1.3). G1 menagerie assets present under
> `third_party/mujoco_menagerie/unitree_g1/` (fresh clones: run `tools/setup_env.sh`).
