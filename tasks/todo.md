# Project Status Board

> Refreshed: 2026-08-13 (evening) — full review + 8-agent cleanup sweep executed same day.
> Audit: `tasks/PROJECT_REVIEW_2026-08-13.md`. Per-lab detail: `lab-N-<name>/tasks/TODO.md`.

## Lab Status (summary)

- [x] **Lab 1: 2-Link Planar Arm** — Complete. Square-drawing capstone + video. Now has a real test suite (26 tests, all passing) and MuJoCo ≥3.11 compat.
- [x] **Lab 2: UR5e 6-DOF Arm** — Complete. Cube capstone; 34/34 tests green again (lost `lab_scene.xml` recreated as a tracked file); wrong-robot `ros2_bridge/` deleted; stale paths fixed; GIF 39→6.4 MB.
- [x] **Lab 3: Dynamics & Force Control** — Complete. 34/34 tests (MuJoCo 3.11 `qM` compat fixed). Blog post written.
- [x] **Lab 4: Motion Planning** — Complete. 45/45 tests. README metrics now the validated slalom numbers; unreproducible `slalom_metrics.json` deleted. Blog post written.
- [x] **Lab 5: Grasping & Manipulation** — Complete. **Phase 5 closed 2026-08-13**: 5.4 pro demo re-recorded (23.1 s, 720p60) with per-step self-collision monitor — 11,050 steps, **0 self-collisions**; a real IK branch-selection bug (`nearest_joint_branch`) was found and fixed on the way; 33/33 tests. ⚠️ New follow-up: **Step 6.1** below.
- [x] **Lab 6: Dual-Arm Coordination** — Complete. M2 FK cross-validation now has an archived artifact (`media/m2_fk_validation.txt`, max err 2.6e-12 mm).
- [x] **Lab 7: Locomotion (M3d scope)** — Complete. Test suite runs for real now (34/34 — imports were broken by the Menagerie rewrite); misleading M4 leftovers and scratch files deleted; media naming aligned. M4 ZMP walking remains BLOCKED by design → Lab 8.
- [ ] **Lab 8: Whole-Body Loco-Manipulation** — Not started. Must own gait generation on the torque/RNEA path (see MASTER_PLAN "Lab 8 dependency note").
- [ ] **Lab 9: VLA Integration** — Not started. Depends on Lab 8 controllers for demo data.

## Current Focus

> 1. **Lab 5 Step 6.1** — `pick_place_demo.py` (capstone, NOT the pro demo) reaches DONE without
>    moving the box: impedance tracking hands off to the gripper ~70–90 mm short of target,
>    fingers close on air, box never transported (400 mm lateral error). Needs
>    DESCEND/DESCEND_PLACE convergence gating + a final box-position assert. Details in
>    `lab-5-grasping-manipulation/tasks/TODO.md`.
> 2. **Lab 8 kickoff** per CLAUDE.md Per-Lab Workflow (read `plan/LAB_08.md` → PLAN →
>    ARCHITECTURE → TODO → code), honoring the gait-generator dependency note.

## Blockers

> None.

## Open Items (small)

- [ ] Lab 5 Step 6.2 — decide whether the two dropped capstone plots (joint tracking, state timeline) should be reinstated.
- [ ] Lab 2 — `c1_multi_waypoint_log.csv`, `c1_singularity_log.csv`, `c1_metrics_dashboard.csv` are archived outputs with no current producer script (noted in `media/README.md`); regenerate producers or accept as archived.
- [ ] `THIRD_PARTY_NOTICES.md` — verify its path list still matches reality now that model assets are gitignored clones (root README license note already updated).
- [ ] Lab 3/4 blog posts exist but are not yet published anywhere external (briefs' criterion is ticked for authorship).

## Completed 2026-08-13 cleanup sweep (was the Doc-Hygiene Backlog)

All Tier 1–3 items from the morning review were executed by the 8-agent sweep and verified:

- Tier 1: Lab 2 bridge deleted ✓ · Lab 1 tests claim made TRUE (26 new tests) ✓ · root README blanket claim fixed ✓ · Lab 2 media/README rewritten ✓ · Lab 2 stale sys.path (13 files) ✓ · Lab 7 orphaned media (5 files) deleted ✓ · Lab 7 scratch files (6) deleted ✓
- Tier 2: Lab 5 Step 5.4 done ✓ · Lab 4 metrics table + JSON ✓ · plan/LAB_06+07 status headers ✓ · Lab 6/7 tasks/PLAN superseded banners ✓ · Lab 3+4 blog posts written ✓ · Lab 7 media naming ✓ · Lab 3/4 unreferenced media documented ✓ · Lab 5 plots regenerated ✓ (exposed Step 6.1)
- Tier 3: duplicate plan files → pointers ✓ · Labs 1–2 tasks/TODO.md backfilled ✓ · Lab 6/7 ARCHITECTURE pointers ✓ · off-topic blog posts → attic/ ✓ · root README topics/structure/hero ✓ · Lab 1 module tables ✓ · 39 MB GIF re-encoded ✓ · Lab 6 m2_ik_visual + FK artifact ✓
- Reproducibility (new): `tools/setup_env.sh` ✓ · bogus gitlinks removed + gitignored ✓ · CLAUDE.md `pin` package fix ✓ · MuJoCo ≥3.11 `qM` compat in labs 1+3 ✓ · Lab 7 model path no longer points outside the repo ✓

## History

Pre-2026-08 version of this file tracked only Labs 1–2 ("Lab 3 TBD"); see git history.
