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

## Current Focus
> Lab 5 fully complete. All videos verified correct.

## Blockers
> None
