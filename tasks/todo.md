# Project Status Board

> Refreshed: 2026-08-18 — Lab 9 closed at measured scope (M0–M6). Lab 8 closed 2026-08-17.
> Audit: `tasks/PROJECT_REVIEW_2026-08-13.md`. Per-lab detail: `lab-N-<name>/tasks/TODO.md`.

## Lab Status (summary)

- [x] **Lab 1: 2-Link Planar Arm** — Complete. Square-drawing capstone + video. Now has a real test suite (26 tests, all passing) and MuJoCo ≥3.11 compat.
- [x] **Lab 2: UR5e 6-DOF Arm** — Complete. Cube capstone; 34/34 tests green again (lost `lab_scene.xml` recreated as a tracked file); wrong-robot `ros2_bridge/` deleted; stale paths fixed; GIF 39→6.4 MB.
- [x] **Lab 3: Dynamics & Force Control** — Complete. 34/34 tests (MuJoCo 3.11 `qM` compat fixed). Blog post written.
- [x] **Lab 4: Motion Planning** — Complete. 45/45 tests. README metrics now the validated slalom numbers; unreproducible `slalom_metrics.json` deleted. Blog post written.
- [x] **Lab 5: Grasping & Manipulation** — **Fully complete 2026-08-13**. Phase 5 (pro demo, 0 self-collisions) AND Step 6.1 (capstone transport) closed. 6.1 root cause: gripper friction pads mounted on the OUTSIDE of the fingers (model bug) + 5 controller/planning fixes. Capstone now places box **5.7 mm** from target with a transport post-condition assert; 33/33 tests; both evidence videos re-recorded.
- [x] **Lab 6: Dual-Arm Coordination** — Complete. M2 FK cross-validation now has an archived artifact (`media/m2_fk_validation.txt`, max err 2.6e-12 mm).
- [x] **Lab 7: Locomotion (M3d scope)** — Complete. Test suite runs for real now (34/34 — imports were broken by the Menagerie rewrite); misleading M4 leftovers and scratch files deleted; media naming aligned. M4 ZMP walking remains BLOCKED by design → Lab 8.
- [x] **Lab 8: Whole-Body Loco-Manipulation** — **Complete 2026-08-17 (M0–M6).** Torque-actuated G1 (model parity 1e-16), whole-body inverse-dynamics QP with contact wrenches (standing hand tracking 7.08 mm RMS), torque-level stepping (4/4 in-place, ZMP 100 % inside support), **DCM forward walking — 12/12 steps, 1.18 m, 6.2 mm DCM RMS**, **walk + two-handed carry — 12/12 steps, hand 14.5 mm RMS** via a centroidal angular-momentum task, and the **loco-manipulation capstone — walk→pick→carry→place, payload 11.8 mm from target, no fall**. M3 retires Lab 7's deferred 10-step capstone and confirms its actuator-model diagnosis. Along the way it corrected the foot contact model and the QP solver tolerance, which also improved M2 and cut solve time 12.6 ms → 0.073 ms. M6 shipped docs EN/TR, a code walkthrough and the blog post **in-milestone**. 97 tests. Two things deliberately not claimed: M4's *moving*-hand sub-task (exploratory — does not survive a no-op perturbation) and the velocity-level QP the brief specified (a kinematic QP cannot balance a floating base).
- [~] **Lab 9: VLA Integration** — **Closed at measured scope 2026-08-18 (M0–M6).** Language-conditioned ACT policy (15.8 M params) emitting the task-space references Lab 8's whole-body QP consumes, so balance is never learned. **M0–M3 pass**: two-object randomised scene with the pelvis's world x/y/yaw deliberately excluded from the state, expert 40/40, 240 demonstrations / 12,180 frames, training to **0.11x** the predict-the-mean baseline with **4.1 mm** hand-target error. **M4 and M5's task gate fail, cause measured**: the policy walks to the object and stops within **1 mm** of the right place and runs inference at **37 Hz** on 4 CPU cores, but **ignores its instruction** — 0.3 mm difference in the hand target between "the red cup" and "the blue box" — because the expert walks until the named object is the one in front of it, so "reach for the nearest object" is correct in every training frame. Walk sits at exactly 50 % because that is chance on a binary choice. The reach also plateaus 12 mm short of the grasp gate (compounding error). Task set cut from 3–5 to `walk` + `pick` because Lab 8's capstone scores 1/8 off its tuned configuration. 48 tests. Follow-up is a **re-collection**, not a retrain.

## Current Focus

> **None — the nine-lab series is closed.** Labs 1–8 are complete; Lab 9 is
> closed at its measured scope with a documented negative result.
>
> The two follow-ups Lab 9 identified, both recorded rather than done:
> 1. **Re-collect Lab 9's demonstrations so the instruction is load-bearing in
>    the data** — position the expert so both objects are equally reachable at
>    the reach and have it stop at a target-independent point, so "reach for the
>    nearest object" stops being a valid policy (L-M4-a).
> 2. **Lab 8 needs a hand-orientation task** before `place` can be restored as a
>    Lab 9 task — position-only hand tasks release an object at whatever tilt the
>    wrist has, and it rolls (L-M0-e).

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
