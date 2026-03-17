# Lab 5: Grasping & Manipulation — Implementation Plan

## Phase 1: Gripper Integration

### Step 1.1: Build MJCF scene with UR5e + parallel-jaw gripper ✅
- Create `models/ur5e_gripper.xml`: UR5e arm (torque-controlled) + custom 2-finger
  parallel jaw gripper (position-controlled) attached to tool0
- Create `models/scene_grasp.xml`: includes ur5e_gripper.xml + table + graspable box
  (free joint body) + target marker
- Gripper: two sliding fingers (left/right), equality-mirrored, one position actuator
- **Output:** Scene loads in MuJoCo; gripper visible at arm tip; box sitting on table
- **Completed:** 2026-03-16

### Step 1.2: Create common module (`lab5_common.py`) ✅
- Paths: SCENE_PATH, URDF_PATH (Lab 3 arm-only URDF), MEDIA_DIR
- Constants: NUM_JOINTS=6, GRIPPER_IDX=6, DT=0.001
- Scene geometry: TABLE spec, BOX_A_POS, BOX_B_POS, GRIPPER_TIP_OFFSET
- Model loading helpers: load_mujoco_model, load_pinocchio_model (arm only)
- R_TOPDOWN: target EE orientation for top-down grasps (computed from FK at Q_HOME)
- **Output:** Module importable by all Lab 5 scripts
- **Completed:** 2026-03-16
- **Post-completion fix (2026-03-17):** Corrected `GRIPPER_TIP_OFFSET` from 0.090 → 0.105 m.
  The offset must measure from tool0 to the pad *centre* inside the finger body frame
  (`gripper_base_z 0.020 + finger_body_z 0.060 + pad_z_in_finger 0.025 = 0.105`), not to
  pad half-height. The wrong value placed pads 15 mm too low, causing them to intersect the
  table during descent and blocking the arm short of the box.

### Step 1.3: Implement gripper controller (`gripper_controller.py`) ✅
- `open_gripper(ctrl, open_pos=0.030)`: set ctrl[GRIPPER_IDX] = open_pos
- `close_gripper(ctrl)`: set ctrl[GRIPPER_IDX] = 0.0
- `wait_for_gripper(model, data, timeout_s)`: step sim until finger velocity < threshold
- `is_gripper_in_contact(model, data)`: check if finger pad geoms touch any object
- **Output:** Open/close verified on standalone arm at Q_HOME
- **Completed:** 2026-03-16

### Step 1.4: Write Phase 1 tests ✅
- `test_gripper.py`: scene loads, gripper opens/closes, contact detection works
- **Verify:** All tests pass
- **Completed:** 2026-03-16

## Phase 2: Contact Physics Tuning

### Step 2.1: Tune contact parameters ✅
- Experiment with box condim (3 vs 4), friction, solref, solimp
- Test: arm picks up box, moves, box does NOT slip or fly off
- Document each parameter's effect in LESSONS.md
- Target: box held securely at 30° arm tilt with no slippage
- **Output:** Verified contact params in scene_grasp.xml; documented in LESSONS.md
- **Completed:** 2026-03-16
- **Post-completion fix (2026-03-17):** Gripper position actuator `kp=200` was too weak —
  fingers balanced at the contact boundary (qpos≈0.013 m, 0mm penetration) causing
  intermittent contact. Increased to `kp=1000` so equilibrium squeeze qpos ≈ 0.003 m
  (stable 5mm+ penetration). Updated in `ur5e_gripper.xml`.

### Step 2.2: IK solver for grasp configurations (`grasp_planner.py`) ✅
- `compute_ik(model, data, ee_fid, x_target, R_target, q_init)`: DLS iterative IK
- Returns q_solution or None if IK diverges
- `compute_grasp_configs(box_pos)`: returns dict with keys:
  - `q_pregrasp`: above box (15 cm above, fingers aligned)
  - `q_grasp`: at box center height (fingers at grasp level)
  - `q_preplace`: above target (15 cm above)
  - `q_place`: at target height
  - `q_home`: Q_HOME
- Uses GRIPPER_TIP_OFFSET to correctly offset tool0 target from box center
- **Output:** IK configs verified collision-free via MuJoCo forward step + ncon check
- **Completed:** 2026-03-16
- **Post-completion fixes (2026-03-17):**
  1. Replaced skew-symmetric orientation error with `pin.log3(R_target @ R_cur.T)`. The
     old formula returns zero for any symmetric 180° rotation, silently producing configs
     with the gripper 90–180° misaligned. `log3` is singularity-free.
  2. Changed IK seed for preplace/place from Q_HOME to `q_pregrasp` with shoulder_pan
     negated. Q_HOME cannot converge to the Y=-0.20 side within 300 iterations; the
     mirrored seed places the solver in the correct joint-space branch immediately.

### Step 2.3: Write Phase 2 tests ✅
- `test_grasp_planner.py`: IK convergence, EE position accuracy, configs collision-free
- **Verify:** All tests pass
- **Completed:** 2026-03-16

## Phase 3: Pick and Place Pipeline

### Step 3.1: Implement grasp state machine (`grasp_state_machine.py`) ✅
- Class `GraspStateMachine` with states:
  - IDLE → PLAN_APPROACH → EXEC_APPROACH → DESCEND → CLOSE → LIFT
  - → PLAN_TRANSPORT → EXEC_TRANSPORT → DESCEND_PLACE → RELEASE → RETRACT → DONE
- PLAN_*: calls Lab 4 RRT* + shortcutting + TOPP-RA (collision-free path)
- EXEC_*: runs joint-space impedance controller (Lab 3 pattern: Kp*(q_d-q) + g(q))
- DESCEND/LIFT: Cartesian impedance Z-axis motion (Lab 3 compute_impedance_torque)
- CLOSE/RELEASE: sets ctrl[6] and waits for finger settlement
- `run(box_start_pos, box_target_pos) -> dict`: full simulation loop, returns log
- **Output:** State machine drives simulation step-by-step, box moves from A to B
- **Completed:** 2026-03-16
- **Post-completion fixes (2026-03-17):**
  1. CLOSE state previously applied gravity compensation only (`tau = g`) while the
     gripper closed. Contact reaction forces could push the arm away before contact was
     confirmed. Changed to full joint impedance (`tau = Kp*(q_hold-q) + Kd*(0-qd) + g`)
     to actively hold arm position during close.
  2. Contact success check now tracks any contact across the entire settle window
     (`contact_during_settle` flag), not only the final state after the 1.5s hold. A
     transient contact during squeeze is still a valid grasp.

### Step 3.2: Run pick and place demo ✅
- `record_demo.py`: full pick-and-place cycle with MuJoCo offscreen rendering → MP4
- `record_pro_demo.py`: Menagerie UR5e + Robotiq 2F-85 version → MP4
- Log: arm q, EE position, gripper position, contact forces, state timeline
- Output videos: `media/pick_place_demo.mp4`, `media/pick_place_pro.mp4`
- **Completed:** 2026-03-16; re-recorded with all fixes 2026-03-17
- **Verified:** "Gripper closed — contact: True" for custom gripper demo; full cycle
  completes for both demo scripts.

### Step 3.3: Write Phase 3 tests ✅
- `test_state_machine.py`: state transitions correct, box moves, no crashes
- **Verify:** All tests pass
- **Completed:** 2026-03-16

## Phase 4: Documentation & Blog

### Step 4.1: Write English documentation (`docs/`) ✅
- `01_contact_physics.md`: condim, friction, solref/solimp — what each does, values used
- `02_gripper_design.md`: parallel jaw mechanics, MJCF equality constraint, position control
- `03_grasp_pipeline.md`: state machine, IK flow, approach/grasp/lift/transport/place
- `04_pick_place_results.md`: plots, contact analysis, failure modes
- **Completed:** 2026-03-16

### Step 4.2: Write Turkish documentation (`docs-turkish/`) ✅
- Translate docs/ to Turkish (same 4 files)
- **Completed:** 2026-03-16

### Step 4.3: Write blog post ✅
- "Building a Pick-and-Place Pipeline from Scratch"
- Cover: why grasping is hard, contact modeling, state machines, Lab 3+4 integration
- **Completed:** 2026-03-16

### Step 4.4: Write README.md ✅
- Lab overview, module map, how to run the demo, key results table
- **Completed:** 2026-03-16

---

## Post-Completion Bug Fixes (2026-03-17)

After the lab was declared complete, video review revealed three classes of bugs that were
fixed in a dedicated debugging session. All changes are backward-compatible.

### Summary of all fixes

| Bug | Root cause | Fix |
|-----|-----------|-----|
| Arm stops short / gripper closes on air | `GRIPPER_TIP_OFFSET=0.090` placed pads 15mm too low, into table | Changed to `0.105` in `lab5_common.py` |
| Gripper at 90–180° wrong orientation | IK skew-symmetric ori error = 0 at 180° misalignment | `pin.log3(R_target @ R_cur.T)` in `grasp_planner.py` |
| IK fails for preplace (Y=-0.20) | Q_HOME seed too far from Y=-0.20 solution | Mirror shoulder_pan of pregrasp as preplace seed |
| "Silly movements" in pro demo | IK `% 2π` wrap causes joint discontinuities near ±π | `np.clip(q, -2π, 2π)` in `record_pro_demo.py` |
| Gripper bounces off box, contact=False | `kp=200` too weak; equilibrium at contact boundary | `kp=1000` in `ur5e_gripper.xml` |
| Arm drifts during gripper close | Gravity-comp-only allows contact forces to push arm away | Joint impedance during close settle loop |
| contact=False despite transient contact | Check only ran at end of hold, missed squeeze event | Track contact across full settle window |
