# Lab 5: TODO

## Phase 1: Gripper Integration
- [x] Step 1.1: Build MJCF scene with UR5e + parallel-jaw gripper — DONE (2026-03-16)
- [x] Step 1.2: Create common module (`lab5_common.py`) — DONE (2026-03-16)
- [x] Step 1.3: Implement gripper controller (`gripper_controller.py`) — DONE (2026-03-16)
- [x] Step 1.4: Write Phase 1 tests — DONE (2026-03-16)

## Phase 2: Contact Physics Tuning
- [x] Step 2.1: Tune contact parameters and document — DONE (2026-03-16)
- [x] Step 2.2: IK solver + grasp config computation (`grasp_planner.py`) — DONE (2026-03-16)
- [x] Step 2.3: Write Phase 2 tests — DONE (2026-03-16)

## Phase 3: Pick and Place Pipeline
- [x] Step 3.1: Implement grasp state machine (`grasp_state_machine.py`) — DONE (2026-03-16)
- [x] Step 3.2: Run pick and place demo (`pick_place_demo.py`) — DONE (2026-03-16)
- [x] Step 3.3: Write Phase 3 tests — DONE (2026-03-16)

## Phase 4: Documentation & Blog
- [x] Step 4.1: Write English documentation (`docs/`) — DONE (2026-03-16)
- [x] Step 4.2: Write Turkish documentation (`docs-turkish/`) — DONE (2026-03-16)
- [x] Step 4.3: Write blog post — DONE (2026-03-16)
- [x] Step 4.4: Write README.md — DONE (2026-03-16)

## Post-Completion Fixes (2026-03-17)
- [x] Fix `GRIPPER_TIP_OFFSET` 0.090 → 0.105 m (`lab5_common.py`) — DONE (2026-03-17)
- [x] Fix IK 180° orientation singularity: `pin.log3` (`grasp_planner.py`) — DONE (2026-03-17)
- [x] Fix IK seed for preplace/place: mirror shoulder_pan from pregrasp (`grasp_planner.py`) — DONE (2026-03-17)
- [x] Fix IK joint wrapping `% 2π` → `np.clip` (`record_pro_demo.py`) — DONE (2026-03-17)
- [x] Fix gripper kp 200 → 1000 (`ur5e_gripper.xml`) — DONE (2026-03-17)
- [x] Fix arm joint impedance during gripper close (`grasp_state_machine.py`) — DONE (2026-03-17)
- [x] Fix contact check: track across full settle window (`grasp_state_machine.py`) — DONE (2026-03-17)
- [x] Re-record `pick_place_demo.mp4` — contact: True ✓ — DONE (2026-03-17)
- [x] Re-record `pick_place_pro.mp4` — full cycle complete ✓ — DONE (2026-03-17)

## Phase 5: Pro Demo Hardening (2026-03-17)
- [x] Step 5.3: Fix matplotlib env issue in Lab 4 (rrt_planner.py + test skip) — DONE (2026-03-17)
- [x] Step 5.1: Fix IK orientation formula in record_pro_demo.py (SO3 log) — DONE (verified in code 2026-08-13: `_so3_log()` at record_pro_demo.py:294, used in `compute_ik` at :349)
- [x] Step 5.2: Integrate Lab 4 RRT* for collision-free planning in record_pro_demo.py — DONE (verified in code 2026-08-13: `plan_collision_free()` at :84 with `shortcut_path`; all 4 long-distance transitions use `run_phase_planned()` at :659/:682/:688/:711; short vertical moves intentionally stay on raw `run_phase`)
- [ ] Step 5.4: Re-record pro demo video, verify no self-collision in any frame
  - Note (2026-08-13 review): `media/pick_place_pro.mp4` exists but cannot be confirmed to post-date the 5.1/5.2 code. No "no self-collision" verification is recorded in LESSONS.md. Re-record with the current script and log the verification before checking this off.

## Current Focus
> Step 5.4: Re-record pro demo with current record_pro_demo.py (5.1+5.2 already in code), verify no self-collision, log result in LESSONS.md

## Blockers
> None

## Review Note (2026-08-13)
Project-wide review found TODO was stale: Steps 5.1 and 5.2 were already implemented
in `src/record_pro_demo.py` but never checked off here, and no LESSONS.md entry was
written for that work. Only Step 5.4 genuinely remains. See `tasks/PROJECT_REVIEW_2026-08-13.md`
at repo root for the full audit.
