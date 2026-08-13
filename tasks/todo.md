# Project Status Board

> Refreshed: 2026-08-13 (full project review — see `tasks/PROJECT_REVIEW_2026-08-13.md`)
> Per-lab detail lives in `lab-N-<name>/tasks/TODO.md` (Labs 3–7). This file is the
> project-level index: what is done, what is open, and the doc-hygiene backlog.

## Lab Status (summary)

- [x] **Lab 1: 2-Link Planar Arm** — Complete. Square-drawing capstone + video. ⚠️ No unit tests exist (READMEs claiming `tests/` were wrong — see backlog).
- [x] **Lab 2: UR5e 6-DOF Arm** — Complete. Cube-drawing capstone, 34 tests / 5 files. ⚠️ `ros2_bridge/` folder contains Lab 1's 2-link bridge by mistake; `src/*.py` carry stale pre-reorg sys.path lines (see backlog).
- [x] **Lab 3: Dynamics & Force Control** — Complete. 34 tests. Blog post never written (was a success criterion).
- [x] **Lab 4: Motion Planning** — Complete (slalom capstone). 45 tests. Blog post never written; README metrics table predates the slalom redesign.
- [x] **Lab 5: Grasping & Manipulation** — Complete except **Step 5.4** (re-record pro demo + verify no self-collision). Steps 5.1 (SO3-log IK) and 5.2 (RRT* integration) are already in `record_pro_demo.py` — TODO was stale, now fixed.
- [x] **Lab 6: Dual-Arm Coordination** — Complete, milestone-gated M0–M5, evidence complete. Note: ~126 unit tests were deliberately deleted in the milestone restructure (commit `6c6dc86`) — only lab with zero automated tests.
- [x] **Lab 7: Locomotion Fundamentals** — Complete **at M3d scope** (standing + weight shift, 34 tests). M4 ZMP walking BLOCKED: position actuators can't track dynamic references (M3e failed 6×). M5 capstone documents the working phases + the finding.
- [ ] **Lab 8: Whole-Body Loco-Manipulation** — Not started. ⚠️ `plan/LAB_08.md` assumes a Lab 7 "gait generator" that does not exist; Lab 8 must build gait itself on the torque-level (RNEA) path. See MASTER_PLAN "Lab 8 dependency note".
- [ ] **Lab 9: VLA Integration** — Not started. Depends on Lab 8 controllers for demonstration data.

## Current Focus

> 1. **Lab 5 Step 5.4** — re-record `pick_place_pro.mp4` with the current script, verify no self-collision in any frame, log in Lab 5 LESSONS.md. This is the only open *code/demo* work item in Labs 1–7.
> 2. Doc-hygiene backlog below (Tier 1 first), then Lab 8 kickoff per the Per-Lab Workflow in CLAUDE.md.

## Blockers

> None for Lab 5 Step 5.4 or the backlog. Lab 8's M4-walking inheritance is a design
> constraint, not a blocker — the torque-control path is already prescribed in LAB_08.md.

---

## Doc-Hygiene Backlog (from 2026-08-13 review)

### Tier 1 — factually wrong, fix first
- [ ] **Lab 2 `ros2_bridge/`**: `mujoco_bridge.py` + `commander.py` are byte-identical copies of Lab 1's 2-link bridge (loads `two_link_torque.xml`, publishes 2 DOF in a 6-DOF lab). Delete in favor of `src/c2_ros2_bridge.py` or port to UR5e.
- [ ] **Lab 1 `tests/` claims**: `lab-1-2link-arm/README.md:103` and root `README.md` structure block claim a `tests/` dir that doesn't exist. Remove the claim or backfill tests.
- [ ] **Root `README.md` blanket claim** "each lab … tests" — false for Labs 1 and 6; soften.
- [ ] **Lab 2 `media/README.md`** says the folder is "reserved"/empty while `c3_draw_cube.{gif,mp4}` sit next to it. Rewrite.
- [ ] **Lab 2 `src/*.py` stale paths** (all 12 scripts): `sys.path.insert` targets deleted `src/lab-2-Ur5e-robotics-lab/` layout; docstrings + `c2_ros2_bridge.py:64` runtime print show wrong run commands. Works only by accident — clean up.
- [ ] **Lab 7 orphaned media**: `media/m4_walking.mp4`, `m4_zmp.png`, `m3_single_step.mp4`, `walking_results.png`, `lipm_trajectory.png` are outputs of the *deleted* open-loop pipeline that LESSONS.md disowns. They read as evidence for the blocked M4. Delete (or move to an `attic/` with a disclaimer).
- [ ] **Lab 7 scratch files in src/**: `test_claude.py`, `test2.py`, `test3.py`, `test4.py`, `test_write.py`, orphaned `walking_demo.py` — committed junk in a published lab; `test_*.py` names can hijack pytest collection. Delete.

### Tier 2 — stale status / misleading on resume
- [ ] **Lab 5 Step 5.4**: re-record pro demo (also listed under Current Focus). Then update lab README "Pending Work" + root README footnote `*`.
- [ ] **Lab 4 README metrics table**: still pre-slalom numbers (RMS 0.0125 rad vs validated 0.0027 rad); `media/slalom_metrics.json` is from a deleted script version and contradicts PLAN (round-trip vs forward, clearance 0.0 vs 0.034 m) — regenerate or delete.
- [ ] **`plan/LAB_06.md` / `plan/LAB_07.md`**: status headers said "Not Started" for complete labs — fixed 2026-08-13; keep briefs' scope text in mind (both promise features that were legitimately re-scoped: internal force control → welds; walking → M3d).
- [ ] **Lab 6/7 `tasks/PLAN.md`**: pre-implementation plans contradicting shipped reality (Lab 7's is 100% unchecked). Add a "superseded — see TODO.md" header or update.
- [ ] **Lab 3/4 blog posts**: never written though `plan/LAB_03.md` / `LAB_04.md` list them as success criteria. Decide: write them or formally drop the criterion.
- [ ] **Lab 7 media naming**: README cites `m3e_zmp_walking.mp4` but the script writes `m3e_single_step.mp4`; `m3e_single_step.py` writes `m3_step_analysis.png` (missing `e`). Align names.
- [ ] **Lab 3 unreferenced media**: `lab3_demo.mp4`, `lab3_metrics.mp4`, `lab3_simulation.mp4` + `record_lab3_demo.py` appear in no doc. Add to README module/media tables. Same for Lab 4's `lab4_{demo,metrics,simulation}.mp4`.
- [ ] **Lab 5 missing plots**: README/docs promise plots in `media/` from `pick_place_demo.py`; only mp4s exist. Regenerate or fix docs.

### Tier 3 — structural hygiene
- [ ] Duplicate plans: `plan/LAB_01.md` ≡ `lab-1…/PROJECT-PLAN-*.md`, `plan/LAB_02.md` ≡ `lab-2…/PROJECT-PLAN-*.md`. Keep the `plan/` copy, replace the lab copy with a pointer.
- [ ] Labs 1–2 lack `tasks/TODO.md` (CLAUDE.md mandates it); their PROJECT-PLAN files have 100% unchecked boxes despite completion. Backfill minimal TODO.md marking done state.
- [ ] Duplicate `tasks/ARCHITECTURE.md` stubs in Labs 6–7 shadowed by full `docs/ARCHITECTURE.md` — add pointer headers.
- [ ] Lab 1 `blog/` contains two off-topic posts (`2025-03-01-50-lessons-humanoid-vla.md`, `2026-03-12-hello-world-meet-rookie.md`) — move out of the lab.
- [ ] Root README: Topics Covered table stops at Lab 3; Lab 5 has no hero image; structure block omits `plan/`, `tasks/`, `tools/`.
- [ ] Lab 1 README module tables omit `a1_interactive_demo.py`, `a1_torque_demo.py`, and the B4/ROS2 module.
- [ ] `lab-2…/media/c3_draw_cube.gif` is 39 MB — re-encode or LFS.
- [ ] Lab 6: `m2_ik_visual.py` missing from README Quick Start; M2 FK cross-validation claim (0.000 mm) has no saved artifact.

## History

The pre-2026-08 version of this file only tracked Labs 1–2 ("Lab 3 TBD") and was two
months stale. Lab 1–2 step-level checklists were all complete and are preserved in git
history (`git log -- tasks/todo.md`).
