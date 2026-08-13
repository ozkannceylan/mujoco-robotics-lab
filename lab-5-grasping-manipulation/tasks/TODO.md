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
- [x] Step 5.4: Re-record pro demo video, verify no self-collision in any frame — DONE (2026-08-13)
  - **Blocker found and fixed first**: the script could not complete a run. `q_preplace` IK
    (seeded from `Q_HOME`) stalled 127 mm short of target and returned a *colliding*
    configuration, so the RRT* leg `HOME→PREPLACE` had an unreachable goal. Fixed by
    seeding preplace from `q_pregrasp` and adding `nearest_joint_branch()` (see LESSONS.md).
  - **Recording**: `media/pick_place_pro.mp4` — 5,050,793 B (5.05 MB), 1280×720, 60 fps,
    23.10 s, H.264/yuv420p. Full 11-phase cycle completed; final frame shows the cube
    resting on the target pad with the arm returned to HOME.
  - **IK evidence** (all four configs, was 1 of 4 broken before):

    | Config | Position error | Collision-free |
    |---|---|---|
    | `q_pregrasp` | 0.090 mm | True |
    | `q_grasp`    | 0.083 mm | True |
    | `q_preplace` | 0.081 mm | True |
    | `q_place`    | 0.083 mm | True |

  - **Self-collision evidence** (`SelfCollisionMonitor`, every contact of every sim step):

    | Metric | Value |
    |---|---|
    | Simulation steps checked | 11050 |
    | Steps with self-collision | **0** |
    | Distinct self-collision pairs | 0 |
    | Max penetration depth | 0.000 mm |
    | Robot↔table contact pairs | 0 |
    | Gripper-internal contact pairs | 0 |

    RESULT: **PASS — no self-collision in any frame**. The script now exits non-zero if
    this check ever fails, so the verification is repeatable rather than a one-off claim.

## Newly Discovered (2026-08-13) — NOT part of Step 5.4

- [ ] Step 6.1: `pick_place_demo.py` runs to DONE but never transports the box
  - Found while regenerating the `media/` plots (previously missing — only mp4s were on disk).
  - `Box final pos: [0.350, 0.200, 0.335]` = still Box **A**. Lateral error **400.0 mm**.
  - `media/ee_trajectory_3d.png`: EE stops ~70 mm short of Box A and ~90 mm short of Box B.
  - `media/gripper_vs_time.png`: fingers close to 0 mm → nothing between them.
  - IK is *not* the culprit — the config summary prints `preplace = [0.350, -0.200, 0.590]`,
    exactly as intended. This is Cartesian-impedance tracking convergence in
    `GraspStateMachine`: DESCEND/DESCEND_PLACE hand off to the gripper before the EE has
    settled onto the commanded pose.
  - Also add a post-condition assert on final box position — the state machine currently
    reaches `DONE` without ever checking that the box moved, so a miss passes silently.
  - Scope note: this is the capstone demo, *not* `record_pro_demo.py`. The pro demo completes
    the full cycle correctly (video ends with the cube on the target pad).

- [ ] Step 6.2: `docs/04_pick_place_results.md` listed four plot filenames that
      `pick_place_demo.py` never writes — corrected 2026-08-13 to the three it actually
      produces. Re-check whether the two dropped plots (joint tracking, state timeline)
      are still wanted.

## Current Focus
> Phase 5 is closed. Next open item is Step 6.1 (capstone box transport failure).

## Blockers
> None

## Review Note (2026-08-13)
Project-wide review found TODO was stale: Steps 5.1 and 5.2 were already implemented
in `src/record_pro_demo.py` but never checked off here, and no LESSONS.md entry was
written for that work. Only Step 5.4 genuinely remains. See `tasks/PROJECT_REVIEW_2026-08-13.md`
at repo root for the full audit.
