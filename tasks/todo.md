# Project Status Board

> Refreshed: 2026-08-13 (evening) — full review + 8-agent cleanup sweep executed same day.
> Audit: `tasks/PROJECT_REVIEW_2026-08-13.md`. Per-lab detail: `lab-N-<name>/tasks/TODO.md`.

## Lab Status (summary)

- [x] **Lab 1: 2-Link Planar Arm** — Complete. Square-drawing capstone + video. Now has a real test suite (26 tests, all passing) and MuJoCo ≥3.11 compat.
- [x] **Lab 2: UR5e 6-DOF Arm** — Complete. Cube capstone; 34/34 tests green again (lost `lab_scene.xml` recreated as a tracked file); wrong-robot `ros2_bridge/` deleted; stale paths fixed; GIF 39→6.4 MB.
- [x] **Lab 3: Dynamics & Force Control** — Complete. 34/34 tests (MuJoCo 3.11 `qM` compat fixed). Blog post written.
- [x] **Lab 4: Motion Planning** — Complete. 45/45 tests. README metrics now the validated slalom numbers; unreproducible `slalom_metrics.json` deleted. Blog post written.
- [x] **Lab 5: Grasping & Manipulation** — **Fully complete 2026-08-13**. Phase 5 (pro demo, 0 self-collisions) AND Step 6.1 (capstone transport) closed. 6.1 root cause: gripper friction pads mounted on the OUTSIDE of the fingers (model bug) + 5 controller/planning fixes. Capstone now places box **5.7 mm** from target with a transport post-condition assert; 33/33 tests; both evidence videos re-recorded.
- [x] **Lab 6: Dual-Arm Coordination** — Complete. M2 FK cross-validation now has an archived artifact (`media/m2_fk_validation.txt`, max err 2.6e-12 mm).
- [x] **Lab 7: Locomotion (M3d scope)** — Complete. Test suite runs for real now (34/34 — imports were broken by the Menagerie rewrite); misleading M4 leftovers and scratch files deleted; media naming aligned. M4 ZMP walking remains BLOCKED by design → Lab 8.
- [ ] **Lab 8: Whole-Body Loco-Manipulation** — 🚧 IN PROGRESS. **M0/M1/M2 PASSED 2026-08-15**: torque-actuated G1 (parity 1e-16), whole-body inverse-dynamics QP with contact wrenches (hand tracking 7.08 mm RMS, 0.11 ms solve), and torque-level stepping (4/4 in-place steps, ZMP 98.7 % inside support, peak torque 49.6 of 139 N·m). 62 tests. Next: **M3 — forward walking** (retires Lab 7's deferred 10-step capstone).
- [ ] **Lab 9: VLA Integration** — Not started. Depends on Lab 8 controllers for demo data.

## Current Focus

> **Lab 8 · M3 — Forward walking** (M0/M1/M2 passed 2026-08-15; one
> milestone per session). Labs 1–7 have no open code work items.

## Blockers

> None.

## Open Items (small)

- [x] Lab 5 Step 6.1 — capstone transport — DONE 2026-08-13 (see Lab 5 TODO/LESSONS; 5.7 mm placement, post-condition assert added).
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
