# Lab 5: Lessons Learned

## Bugs & Fixes

### 2026-03-16 — `parameterize_topp_ra` returns 4-tuple, not 3-tuple
**Symptom:** `ValueError: too many values to unpack (expected 3)` in `grasp_state_machine.py:320`
**Root cause:** `trajectory_smoother.parameterize_topp_ra` returns `(times, q, qd, qdd)` — 4 values. The state machine unpacked only 3.
**Fix:** Changed unpacking to `times, q_traj, qd_traj, _ = parameterize_topp_ra(...)`.
**Takeaway:** Always check the actual return signature of cross-lab imports when the calling code assumes a specific tuple length.

### 2026-03-16 — `mpl_toolkits.mplot3d` import fails with system matplotlib
**Symptom:** `ModuleNotFoundError: No module named 'matplotlib.tri.triangulation'` during test collection.
**Root cause:** Two matplotlib versions installed (system Python3 package + pip user package). The system one has a broken `mpl_toolkits.mplot3d`.
**Fix:** Wrapped the import in `rrt_planner.py` with try/except so import failure is non-fatal (Axes3D is visualization-only).
**Takeaway:** Guard any matplotlib 3D import with try/except when the module runs on systems with mixed matplotlib installs.

### 2026-03-16 — Gripper pad geometry: can't contact 40mm box when fully closed
**Symptom:** `is_gripper_in_contact` returned False even after gripper closed on a box placed between the fingers.
**Root cause:** With left/right finger bodies at ±0.020 m from gripper_base center and pad at +0.009 m relative offset, the pad inner face at joint=0 was at 0.024 m — outside the 40mm box edge at 0.020 m. Gap of 4mm meant the gripper could never physically touch the box.
**Fix:** Moved finger body Y positions from ±0.020 m to ±0.015 m. At joint=0, pad inner face = 0.019 m < box edge 0.020 m → 1mm overlap → contact detected.
**Takeaway:** Always prototype gripper geometry in a static scene and verify minimum-gap vs object dimensions before implementing control code.

### 2026-03-16 — `is_gripper_in_contact` too narrow (checked pads only, not finger bodies)
**Symptom:** Even after fixing geometry, contact was still not detected. Debug showed contacts between `left_finger_geom`/`right_finger_geom` and `box_geom` — not the pads.
**Root cause:** The finger body geom is larger than the pad and contacts the box first. The pad geom makes secondary contact only at very close range. `is_gripper_in_contact` checked only pad geom IDs.
**Fix:** Expanded the check to include all finger geoms: `left_pad`, `right_pad`, `left_finger_geom`, `right_finger_geom`.
**Takeaway:** For "is something in the gripper" checks, include all finger geoms, not just friction pads. The large enclosing geom will contact before the small pad tip.

### 2026-03-16 — Contact test checks too late — box falls under gravity
**Symptom:** Contact IS detected at steps 10-200 but the test checked at step 1000. By then the box had fallen to the floor (arm has no gravity compensation in this test).
**Root cause:** No joint torques applied during the test → arm and gripper droop. The box falls once the finger actuator can no longer hold it.
**Fix:** Changed the test to break-and-check during a 200-step window: `if is_gripper_in_contact(...): contact_detected = True; break`.
**Takeaway:** Contact tests that rely on a free-floating box must check during the contact event, not after settling. Or: apply gravity compensation during the test.

### 2026-03-16 — GRIPPER_TIP_OFFSET wrong: pads hit table during descent
**Symptom:** In the recorded demo (record_demo.py) the arm descends toward the box but stops short — it "cannot reach the object." The gripper closes on air slightly above the box.
**Root cause:** `GRIPPER_TIP_OFFSET = 0.090` was computed as `gripper_base_z (0.020) + finger_body_z (0.060) + pad_half_height (0.008) = 0.088 ≈ 0.090`. But the pad _center_ within the finger body is at z=0.025 (not 0), so the correct offset is `0.020 + 0.060 + 0.025 = 0.105 m`. With 0.090, the IK placed the tool0 15 mm too low, pushing pad bottoms 3 mm below the table surface → table contacts blocked the descent.
**Fix:** Changed `GRIPPER_TIP_OFFSET = 0.090` → `0.105` in `lab5_common.py`. Pads now land at box center (0.335 m), 20 mm above the table.
**Takeaway:** The offset must be from tool0 to the pad _center position in finger body frame_ (pos z = 0.025), not just to the finger body origin + pad half-height.

### 2026-03-17 — IK fails for preplace (Y=-0.20 side) when seeded from Q_HOME
**Symptom:** `RuntimeError: IK failed for 'preplace' target at [0.35, -0.2, 0.59]` in `grasp_planner.py`.
**Root cause:** All four IK targets used Q_HOME as seed. The preplace/place targets are on the negative-Y side (Y=-0.20), which is far from Q_HOME in joint space. The DLS IK cannot converge from Q_HOME to this configuration in 300 iterations.
**Fix:** Solved pregrasp first, then built a mirrored seed for preplace: `q_hint_b[0] = -q_pregrasp[0]` (negate shoulder_pan). This places the seed in the correct joint-space branch for the Y=-0.20 side.
**Takeaway:** When box_a and box_b are Y-symmetric, mirror shoulder_pan of the box_a solution to get a valid seed for box_b. Never reuse Q_HOME as seed for configurations far from home.

### 2026-03-17 — IK 180° orientation singularity silently corrupts grasp orientation
**Symptom:** "Gripper closed — contact: False" even after GRIPPER_TIP_OFFSET fix. Diagnostic showed `q_grasp` ee_link Z-axis = [0.137, -0.871, -0.471] instead of [0, 0, -1]. Gripper site was 97mm off in Y.
**Root cause:** Orientation error formula `-0.5*(R_target.T@R_cur - R_cur.T@R_target)` computes the anti-symmetric part of the orientation error. For any rotation that is 180° off, the anti-symmetric part is exactly zero, so the IK reported "converged" with a completely wrong orientation.
**Fix:** Replaced with `pin.log3(R_target @ R_cur.T)` — the Lie algebra logarithm returns `π·axis` at 180° error (no singularity), expressed in the world frame (correct for `LOCAL_WORLD_ALIGNED` Jacobian).
**Takeaway:** Never use the skew-symmetric formula for IK orientation error. Use `pin.log3` — it handles all rotation magnitudes correctly, including 180°.

### 2026-03-17 — IK modular wrapping causes "silly movements" in recorded video
**Symptom:** In record_pro_demo.py, the arm makes large sweeping rotations during transitions — shoulder or wrist appears to spin instead of moving smoothly to the target.
**Root cause:** `q = (q + pi) % (2*pi) - pi` wraps joints to [-π, π] during every IK step. If a joint passes through ±π during iteration, it jumps discontinuously. The resulting IK solution has a joint near ±π that is numerically equivalent to the target but physically far from the seed configuration. Linear interpolation from Q_HOME to this solution then sweeps through ~360° on that joint.
**Fix:** Replaced modular wrap with soft clip `q = np.clip(q, -2*pi, 2*pi)` (actual UR5e hardware limit). This keeps the IK solution near the seed without forcing wrap-around discontinuities.
**Takeaway:** Never use modular arithmetic (`% 2π`) for joint wrapping inside an IK solver seeded from a specific configuration. Use clipping to physical limits instead — it preserves continuity with the seed.

### 2026-03-17 — Gripper kp=200 too weak: fingers bounce off box at contact boundary
**Symptom:** `is_gripper_in_contact` returned False after closing. Diagnostic showed finger qpos settling at 0.013m where the finger_geom inner face was exactly at the box edge (0.020m) — intermittent contact on every other step.
**Root cause:** Position actuator `kp=200` generated only 2.6N at equilibrium (qpos=0.013). The contact reaction force balanced the actuator at the contact boundary, causing the finger to oscillate on/off contact. With the box at its edge, any perturbation broke contact.
**Fix:** Increased `kp=200` → `kp=1000` in `ur5e_gripper.xml`. At 5× higher gain, the equilibrium qpos drops to ≈0.003m, finger_geom inner face penetrates 5mm into box → stable contact.
**Takeaway:** Gripper position actuator kp must be high enough that the equilibrium squeeze position provides clear penetration depth (≥ 2mm). Verify by checking equilibrium qpos vs. expected contact boundary.

### 2026-03-17 — Gravity-comp-only arm during gripper close allows arm to drift off box
**Symptom:** Contact detected at step 22 during close but lost by step 50. The arm drifted away from the box under contact reaction forces.
**Root cause:** `_run_close_gripper` applied only gravity compensation (`tau = g`) to the arm during the settle loop. Contact reaction forces from squeezing the box pushed the arm upward/sideways, breaking finger-box alignment before contact was confirmed.
**Fix:** Changed to full joint impedance during close: `tau = Kp*(q_hold - q) + Kd*(0 - qd) + g`. Also changed contact check to record `True` if contact occurred at ANY point during settle (not just the final state after hold).
**Takeaway:** Always hold arm with joint impedance (not just gravity comp) during gripper close. Gravity comp alone cannot resist contact reaction forces.

## Debug Strategies

### Verify gripper geometry with MuJoCo viewer
Run `python -c "import mujoco; import mujoco.viewer; m=mujoco.MjModel.from_xml_path('models/scene_grasp.xml'); d=mujoco.MjData(m); mujoco.viewer.launch(m, d)"` from models/ to visually inspect the scene before running any control code.

### Print contact pairs for slipping diagnosis
When the box slips: `for i in range(data.ncon): c=data.contact[i]; print(model.geom(c.geom1).name, model.geom(c.geom2).name, c.dist)` to see which geoms are in contact and their penetration depth.

### Step-by-step contact debugging
Print `d.ncon` and all contact pairs at specific step intervals to trace exactly when contact appears and when it breaks. Essential for gripper geometry tuning.

## Key Insights

### Gripper minimum gap must be less than object width
The pad inner face position at GRIPPER_CLOSED must be ≤ object half-width. For a 40mm box (half 0.020 m), the pad face at closed must be ≤ 0.020 m from gripper center. Otherwise the gripper physically misses the object.

### Contact is detected via geom pairs, not force/torque
MuJoCo's `data.contact` list holds geom-geom pairs. For gripper grasping detection, iterate all contacts and check if any gripper geom appears — do NOT limit to just the friction pads.

### Pinocchio arm-only model (no gripper joints) works for FK/IK
Lab 5 reuses the Lab 3 UR5e URDF (6 DOF arm only). This is correct: IK computes arm configurations, and the gripper joint is handled separately by MuJoCo. No need to rebuild the Pinocchio model.
