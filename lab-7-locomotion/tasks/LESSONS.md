# Lab 7: Lessons Learned

## M0: Load G1 and Understand the Robot

### Use the real Menagerie G1, not a simplified stick-figure
- **Symptom:** First attempt used `models/g1_humanoid.xml` — a hand-built capsule/box model with wrong mass (45 kg vs 33 kg), wrong geometry, and no mesh visuals.
- **Root cause:** Previous session created a simplified model instead of using the Menagerie G1.
- **Fix:** Point `lab7_common.py` to `vla_zero_to_hero/third_party/mujoco_menagerie/unitree_g1/scene.xml`. This gives us real STL meshes, calibrated inertias, proper Kp=500 actuators, and the correct 29-DOF model.
- **Takeaway:** Always use the canonical Menagerie model. "Simplified" models introduce wrong physics.

### Position servos prevent ragdoll collapse
- **Symptom:** With `ctrl=0`, robot barely moved. Position servos hold joints at zero = standing pose.
- **Root cause:** Position actuators with Kp=500 act as stiff springs. Even at ctrl=0, servo resists gravity deflections.
- **Fix:** Zero out `m.actuator_gainprm`, `m.actuator_biasprm`, AND `m.dof_damping` at runtime for true ragdoll freefall.
- **Takeaway:** "No control" with MuJoCo position servos is NOT zero torque. Must explicitly disable gains.

### Offscreen renderer framebuffer size
- **Symptom:** `ValueError: Image width 1920 > framebuffer width 640`
- **Fix:** Set `m.vis.global_.offwidth = 1920` and `m.vis.global_.offheight = 1080` at runtime after loading model. No need to modify the XML.
- **Takeaway:** Set offscreen buffer via model API, not by editing Menagerie XML.

### Menagerie G1 model key facts
- 29 actuated DOFs: legs(12) + waist(3) + arms(14)
- All position actuators with uniform Kp=500, dampratio=1
- Pelvis at z=0.793m in keyframe, CoM at ~0.692m
- Total mass: 33.34 kg (pelvis 3.8kg, torso 7.8kg, each leg ~7.2kg, each arm ~3.5kg)
- Foot contact: 4 sphere geoms per foot (r=0.005m), ~17cm toe-to-heel
- Arms neutral pose from keyframe: shoulder_pitch=0.2, shoulder_roll=±0.2, elbow=1.28

### MuJoCo quaternion convention
- MuJoCo freejoint qpos[3:7] = (w, x, y, z)
- Pinocchio FreeFlyer q[3:7] = (x, y, z, w)
- Use `mj_quat_to_pin()` / `pin_quat_to_mj()` from lab7_common

## Debug Strategies

### Check foot site vs foot body
G1 has `left_foot_site` and `right_foot_site` defined in the MJCF.
Use site positions for foot contact targets, not body positions.

### Floating base is fundamentally different
In fixed-base labs (1-6), joint 0 is the robot root. In Lab 7, the robot has a 7-DOF
unactuated freejoint at qpos[0:7] / qvel[0:6]. You cannot directly command pelvis position.

## M1: Standing with Joint PD + Gravity Compensation

### Gravity feedforward through position servos works perfectly
- **Pattern:** `ctrl[i] = q_ref[i] + qfrc_bias[6+i] / Kp` where Kp=500 for all Menagerie actuators.
- **Effect:** Effective torque = Kp*(q_ref - qpos) + qfrc_bias - Kd*qvel = PD + gravity comp.
- **Result:** Pelvis height deviation < 1.6mm over 10s, including a 5N push. The Kp=500 servos are very stiff.
- **Takeaway:** Menagerie position servos with gravity feedforward are trivially stable for standing. Real challenge starts with dynamic motions (M3+).

### Actuator-to-DOF mapping is clean: dof = 6 + actuator_index
- All 29 actuators map linearly to dof indices 6..34.
- `qfrc_bias[6:35]` gives the gravity bias for all actuators in order.
- No need for complex joint-to-actuator lookups.

### qfrc_bias in standing is small for leg joints
- Leg joint biases < 1.3 Nm (legs nearly straight → small gravity torques).
- Waist pitch has the largest bias (~0.79 Nm) due to torso weight offset.
- Arms have moderate bias (~1.4 Nm at shoulder roll) from dangling arm weight.

## M2: CoM Tracking and Support Polygon

### Pinocchio MJCF FreeFlyer Z offset is critical
- **Symptom:** CoM cross-validation error = 793mm (exactly the pelvis MJCF Z).
- **Root cause:** `buildModelFromMJCF(FreeFlyer)` sets joint placement at `p=[0,0,0.793]` from `<body name="pelvis" pos="0 0 0.793">`. So `pelvis_world_z = pin_q[2] + 0.793`. MuJoCo stores world position directly in `qpos[2]`.
- **Fix:** `pin_q[2] = mj_qpos[2] - 0.793` in `mj_qpos_to_pin()`.
- **Result:** 0.000mm cross-validation error after fix.
- **Takeaway:** Always check MJCF body position offsets when using Pinocchio FreeFlyer. This is NOT the same issue as the quaternion convention swap.

### CoM Jacobian corrections must be clamped
- **Symptom:** First attempt with K_COM_P=2.0 caused robot to fall (720mm pelvis drop).
- **Root cause:** `J_com_pinv * com_error * K` produced ~1 rad corrections. With Kp=500 position servos, 1 rad offset = 500 Nm torque spike → immediate instability.
- **Fix:** Reduced gain to K_COM=0.3, clamped `delta_q` to ±0.05 rad, used initial CoM as target (not polygon center).
- **Takeaway:** Position-level corrections through Jacobian IK must be small (< 0.05 rad). The stiff Kp=500 servo amplifies any position error.

### Support polygon from Menagerie foot geoms
- 8 sphere contact geoms (4 per foot, radius=0.005m).
- Identified by: `type==sphere`, `size < 0.01`, parent body name contains "ankle_roll".
- Polygon spans ~17cm toe-to-heel, ~29cm left-right.
- CoM X is near rear of polygon (3mm vs center at 35mm) — robot stands slightly heel-heavy.

## M3/M4 (OLD — DELETED): Open-Loop Hacks

### Previous M3/M4 bypassed Pinocchio entirely — deleted and rebuilding
- **Symptom:** M3 and M4 used open-loop joint offsets (ctrl[1] += 0.20, ctrl[7] += 0.15, etc.) with zero Pinocchio IK, zero frame Jacobians, and no ZMP computation.
- **Root cause:** When CoM Jacobian corrections gave unexpected signs in simulation, the fix was to abandon Pinocchio entirely and hand-tune joint offsets. This is architecturally wrong — it circumvents the Pinocchio-first principle of the lab series.
- **Fix:** Delete m3_single_step.py and m4_walking.py. Rebuild from scratch with Pinocchio IK for all foot/CoM control.
- **Takeaway:** If Pinocchio gives unexpected results, debug the frame convention — don't replace with open-loop hacks. The correct fix was to validate Jacobians with finite differences first (M3a).

### Menagerie G1 actuators have ZERO damping
- **Symptom:** Large ctrl changes cause oscillation and instability.
- **Root cause:** G1 actuator `biasprm = [0, -500, 0]` → Kd = 0. No `dof_damping` either.
- **Fix:** Add explicit velocity damping in control law: `ctrl -= K_vel * d.qvel[6:35] / KP_SERVO` with K_vel ≈ 8-10.
- **Takeaway:** Always check actuator damping. Position servos without damping are critically underdamped.

### Waist_pitch compensation sign: -= is correct (positive waist_pitch = backward lean)
- Verified empirically. Keep this for rebuilt M4.

### Waist_pitch compensation MUST be clamped to ±0.12 rad
- Large corrections destabilize via reaction torque. Keep this for rebuilt M4.

## M3a: Pinocchio Frame Convention Validation (REBUILD)

### Pinocchio Jacobians are CORRECT — validated with central finite differences
- **Test:** For all 12 leg joints, perturbed by ±0.001 rad using `pin.integrate()`, compared (FK_new - FK_old)/(2*eps) against analytical Jacobian columns.
- **Results:** 0/36 failures. Max error ~1e-7 (well below 1e-4 tolerance).
- **Validated Jacobians:**
  - CoM Jacobian: `pin.jacobianCenterOfMass(model, data, q)` — all 12 columns match
  - Left foot: `pin.getFrameJacobian(model, data, 15, pin.LOCAL_WORLD_ALIGNED)` — all match
  - Right foot: `pin.getFrameJacobian(model, data, 28, pin.LOCAL_WORLD_ALIGNED)` — all match
  - OP_FRAME variants (left_foot=16, right_foot=29): identical results
- **Takeaway:** The Jacobians themselves are correct. Previous sign-flip issues in M3 were likely caused by incorrect q conversion or wrong frame convention, NOT by Pinocchio giving wrong answers.

### Use central differences for Jacobian validation, not forward differences
- **Symptom:** Forward differences (f(q+eps) - f(q))/eps gave 3e-4 error for hip_pitch (above 1e-4 tolerance).
- **Root cause:** Forward differences have O(eps) error. For large-lever-arm joints (hip_pitch moves foot 0.65m), second-order term is ~0.65 * 0.001 / 2 ≈ 3e-4.
- **Fix:** Central differences (f(q+eps) - f(q-eps))/(2*eps) → O(eps^2) error ≈ 1e-7.
- **Takeaway:** Always use central differences for numerical Jacobian validation.

### Foot frame positions at standing config
- left_ankle_roll_link (ID=15, BODY): pos=[-0.0000, 0.1185, 0.0331]
- right_ankle_roll_link (ID=28, BODY): pos=[-0.0000, -0.1185, 0.0331]
- left_foot (ID=16, OP_FRAME): identical to ankle_roll_link (zero offset)
- right_foot (ID=29, OP_FRAME): identical to ankle_roll_link
- Both frame types give the same Jacobian and position. Use either for IK.

### Key Jacobian magnitudes for IK design (at standing config)
- hip_pitch → foot x: ±0.654 (largest lever arm — primary sagittal control)
- hip_roll → foot y: ±0.614 (primary lateral control)
- knee → foot x: ±0.318 (secondary sagittal)
- hip_yaw → foot y: ±0.134 (small lateral coupling)
- ankle_pitch → foot x: ±0.018 (fine adjustment only)
- ankle_roll → foot: all zeros at standing (no effect at this config)

## M3b: Foot FK Cross-Validation (Pinocchio vs MuJoCo)

### Pinocchio and MuJoCo FK agree to machine precision
- **Test:** 10 random leg configs (within 60% of joint limits), floating base at standing pose.
- **Result:** 0.0000mm error for all 20 foot comparisons (10 configs x 2 feet) and all 10 CoM comparisons.
- **Key insight:** With the correct `mj_qpos_to_pin()` conversion (quaternion swap + Z offset), there is ZERO disagreement between the two FK engines. This confirms:
  1. The quaternion mapping (MuJoCo w,x,y,z → Pinocchio x,y,z,w) is correct
  2. The Z offset (pin_q[2] = mj_qpos[2] - 0.793) is correct
  3. Joint ordering is identical in both models (no joint permutation needed)
- **Takeaway:** Any future FK/IK disagreement between Pinocchio and MuJoCo is a bug in the conversion code, not a model mismatch. The `mj_qpos_to_pin()` function is verified and trusted.

### MuJoCo foot sites and Pinocchio foot frames have zero local offset
- Both `left_foot`/`right_foot` MuJoCo sites and Pinocchio frames have `local_pos=[0,0,0]` relative to their parent body/joint.
- Direct position comparison is valid without any offset correction.

## M3c: Static Whole-Body IK

### Stacked Jacobian DLS works for floating-base whole-body IK
- **Architecture:** 18-task stack: feet 6D×2 + CoM_XY 2D + pelvis_Z 1D + pelvis_ori 3D.
- **Parameters:** lambda=0.01, alpha=0.3, dq_max=0.1 rad norm clamp.
- **Results:** All 4 tests pass. Identity converges in 0 iterations (trivially). CoM shift 5cm converges in 20 iters with <0.51mm foot slip and <0.13mm CoM error.
- **Takeaway:** Simple stacked Jacobian with DLS is sufficient for quasi-static whole-body IK on the G1. No task hierarchy or explicit weights needed for these displacement magnitudes.

### IK naturally discovers the correct joint couplings
- **Lateral CoM shift (5cm):** IK uses hip_roll (±0.060 rad) and ankle_roll (∓0.061 rad) — the same "safe joints" identified empirically in old M3, but now derived analytically.
- **Forward CoM shift (3cm):** IK uses hip_pitch (+0.044 rad) and ankle_pitch (-0.051 rad) — proper sagittal coupling.
- **Takeaway:** The IK solves correctly what the old M3 hard-coded by hand. This validates that the Pinocchio open-chain Jacobians work for whole-body IK when feet are constrained in the task stack.

### Pelvis Jacobian is only nonzero for floating-base DOFs
- `J_pelvis` (6×35) has nonzero columns only for velocity indices 0-5 (floating base).
- This means pelvis height/orientation tasks only constrain the base motion, while leg joints are "free" for foot/CoM tasks.
- **Implication for IK:** The pelvis constraints act as soft virtual base constraints, not competing with the leg joint DOFs.

## M3d: Weight Shift in Simulation

### Pre-compute IK offline, replay in simulation — NOT online IK feedback
- **Symptom:** Five attempts at online IK (fix_base=False/True, CC Jacobian) all failed. The robot overshot and fell during hold phase, or the CC Jacobian predicted wrong CoM direction.
- **Root cause:** Online IK feedback creates a control loop that's hard to stabilize. The floating-base kinematic Jacobian (open-chain or contact-consistent) predicts different CoM-to-joint sensitivity than MuJoCo dynamics, because servo torque reaction on the pelvis dominates over kinematic chain prediction.
- **Fix:** Pre-compute the entire IK trajectory offline using `whole_body_ik` at 11 waypoints (0%→100% of shift), extract joint angles, cosine-smooth interpolate in simulation. No online IK corrections.
- **Result:** CoM shift 53.5mm, foot drift < 1.4mm, hold phase rock-solid.
- **Takeaway:** For quasi-static motions, pre-computed IK trajectory + PD tracking is far more robust than online IK feedback. Only use online IK when the motion is truly unpredictable (e.g., push recovery).

### Near-critical velocity damping (K_VEL=40, ζ≈0.89) is essential
- **Symptom:** K_VEL=8 (ζ≈0.18) and K_VEL=25 (ζ≈0.56) both caused oscillation and fall.
- **Root cause:** Menagerie G1 actuators have Kd=0. Without explicit damping, the Kp=500 position servos are critically underdamped.
- **Fix:** K_VEL=40, so effective Kd=40, giving ζ = 40 / (2√500) ≈ 0.89. Near-critical damping suppresses oscillation without excessive sluggishness.
- **Takeaway:** Always use K_VEL ≈ 40 (≈ 2√Kp) for the G1 Menagerie actuators. This is the single most important stabilization parameter.

### IK from settled MuJoCo state, not theoretical standing config
- **Pattern:** After 1s settling, convert MuJoCo state to Pinocchio, compute FK, use actual foot/CoM positions as IK references.
- **Benefit:** IK waypoint 0 matches the actual robot state (0 iterations needed). No discontinuity at trajectory start.
- **Takeaway:** Always initialize IK from the actual simulation state, not a theoretical default.

### Foot orientation stays locked with near-zero effort
- Foot orientation errors < 0.001 mrad across all tests, despite no explicit weighting.
- **Reason:** At standing config, foot orientation Jacobian angular rows have large entries for base angular velocity (which is constrained by pelvis orientation task). The two tasks cooperate rather than conflict.

## M3e: LIPM + ZMP Preview Control Study (from rdesarz/biped-walking-controller)

### What is the cart-table model and how does LIPM relate ZMP to CoM?
- The Linear Inverted Pendulum Model (LIPM) treats the robot as a point mass at CoM height z_c, connected to the ground by a massless leg. Under the constant-height constraint, the dynamics decouple into independent X and Y axes: `x_ddot = (g/z_c) * (x_com - x_zmp)`.
- The "cart-table" model reformulates this: the CoM acceleration is proportional to the difference between CoM position and ZMP position. If CoM is ahead of ZMP, it accelerates forward (like an inverted pendulum falling). The ZMP is the point on the ground where the net moment of gravity + inertial forces is zero.
- The key insight: by choosing WHERE the ZMP should be (via footstep placement), we can control WHERE the CoM goes. This is the foundation of all ZMP-based walking.

### What does preview control do that our static approach doesn't?
- Static balance requires CoM to always be above the support polygon. Dynamic walking VIOLATES this — the CoM moves continuously between support regions, using momentum to stay balanced.
- Preview control (Kajita 2003) looks AHEAD at future ZMP references (1.6s lookahead) and computes the optimal CoM trajectory that will track that ZMP plan. It solves a discrete Riccati equation offline to get: (1) integral gain Ki for ZMP error, (2) state feedback gain Gx on [com, com_dot, com_ddot], (3) preview gains Gd[j] for future ZMP refs.
- The control input is jerk (third derivative of position). State is augmented to [x, x_dot, x_ddot] with an integrator on ZMP error. The state-space model: A discretizes the triple integrator, B maps jerk to state, C maps state to ZMP = x - (z_c/g)*x_ddot.
- Critical difference: preview control starts moving the CoM BEFORE the support switches, so the CoM "leads" the ZMP transition. Our static approach waited until after weight shift, which is fundamentally wrong for walking.

### How is the swing foot trajectory generated?
- The reference uses two methods: (1) Quintic Bézier with 6 control points (P0=P1=P2=start, P3=P4=P5=end) for zero vel/acc at endpoints — a "minimum-jerk" profile. (2) Sinusoidal z-profile with linear x-interpolation — simpler but only C1 smooth.
- Vertical profile: sin(π*s) * step_height for sinusoidal, or Bézier shaping for the quintic.
- Horizontal: linear or Bézier interpolation from start to end position.
- The trajectory is parameterized by s ∈ [0,1] over the single-support duration.

### How does their IK combine CoM + foot targets?
- QP-based IK with 4 tasks: (1) Fixed foot = hard equality constraint (6D SE3), (2) Moving foot = soft cost (6D SE3, weight w_mf=100), (3) CoM position = soft cost (3D, weight w_com=10), (4) Torso orientation = soft cost (3D angular, weight w_torso=10). Plus Tikhonov damping on joint velocities (mu=1e-4/1e-5).
- Uses Pinocchio's frame Jacobian in LOCAL frame with Jlog6 for proper SE3 error. The stance foot constraint ensures no slip; other tasks are balanced by weights.
- They lock upper-body joints (arms, head) to reduce the problem size.
- Our stacked DLS IK from M3c is architecturally similar but uses DLS instead of QP. Both should work for the walking pipeline.

---

## Project Review Cleanup (2026-08-13)

Defects found by a repo-wide review after the lab was declared complete. All fixed
in one pass; recorded here because most of them are traps future labs can repeat.

### Renaming a symbol in the common module silently broke every importer
- **Symptom:** `pytest lab-7-locomotion/tests/` failed at COLLECTION with
  `ImportError: cannot import name 'build_pin_q_standing' from 'lab7_common'`.
  Both `test_ik.py` and `test_standing.py` errored out, so the README's "34 tests
  pass" claim had been untrue since the Menagerie rewrite.
- **Root cause:** Commit `c6c1bdb` rewrote `lab7_common.py` for the 29-DOF Menagerie
  G1 and dropped `Q_STAND_JOINTS`, `pelvis_world_to_pin_base()`, and
  `build_pin_q_standing()`. The rewrite never ran the importers, and *`src`* modules
  (`whole_body_ik.py`, `standing_controller.py`) imported those names too — so the
  breakage was not confined to tests. Nothing imported them at module scope during
  the milestone demos that were actually being run, so it went unnoticed.
- **Fix:** Restored all three in `lab7_common.py`, adapted to the Menagerie model.
  `Q_STAND_JOINTS` is now an explicit alias of `CTRL_STAND` (for position servos the
  ctrl target *is* the joint angle, so one array with two names beats two arrays
  that can drift apart).
- **Takeaway:** The lab common module is a published API for every other file in the
  lab. After editing it, run the whole test suite — not just the demo you are on.
  A collection error is invisible if you only ever run `python3 src/mN_*.py`.

### Hard-coded absolute-ish model path pointed outside the repo
- **Symptom:** Every model-loading test raised `FileNotFoundError` on a fresh clone.
- **Root cause:** `_MENAGERIE_G1_DIR` was `PROJECT_ROOT.parent / "vla_zero_to_hero" /
  "third_party" / ...` — a sibling checkout on the original author's machine, not
  anything inside this repository.
- **Fix:** `_resolve_menagerie_g1_dir()` now probes a candidate list, preferring the
  in-repo `PROJECT_ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1"` and
  keeping the old sibling path as a fallback.
- **Takeaway:** CLAUDE.md already says "no hardcoded absolute paths". A path that
  escapes `PROJECT_ROOT` via `.parent` violates the spirit of that rule even though
  it is technically relative.

### Pinocchio 4.1 changed the `buildModelFromMJCF` signature and return type
- **Symptom:** `Boost.Python.ArgumentError: Python argument types ... did not match
  C++ signature` for `buildModelFromMJCF(str, JointModelFreeFlyer)`.
- **Root cause:** Pinocchio >= 4.1 requires an explicit root joint *name* as a third
  argument, and then returns a `(model, constraint_models)` tuple instead of a bare
  model. It also emits a DeprecationWarning pointing at
  `buildModelAndLegacyConstraintsFromMjcf`.
- **Fix:** `load_g1_pinocchio()` calls the three-argument form, falls back to the
  two-argument form on `TypeError`, and unwraps the tuple when it gets one.
- **Takeaway:** Pin the Pinocchio API defensively at the loader boundary so a version
  bump touches one function instead of every milestone script.

### Pinocchio's root frame sits PELVIS_MJCF_Z below the MuJoCo world frame
- **Symptom:** `test_com_at_standing` failed with `Standing CoM z = -0.134`, i.e. the
  centre of mass was apparently 13 cm *underground*.
- **Root cause:** A frame-convention error in the test, not in the physics. The MJCF
  declares `<body name="pelvis" pos="0 0 0.793">`, so when Pinocchio builds the model
  with a FreeFlyer root, its world origin ends up `PELVIS_MJCF_Z` below MuJoCo's.
  *Every* Pinocchio placement carries that offset uniformly. Verified against MuJoCo:
  pin CoM z `-0.1341` + `0.793` = `0.6589` = `mj_data.subtree_com[0][2]` exactly, and
  pin `left_foot` z `-0.7929` + `0.793` ≈ `0.0001` = the foot site on the ground.
- **Fix:** Added `pin_point_to_world()` to `tests/test_ik.py` and applied it to the
  CoM before comparing against `Z_C`. `WholeBodyIK.get_com_position()` and
  `get_foot_positions()` were left alone — both return Pinocchio-frame values, which
  is self-consistent; it was the test that mixed frames.
- **Takeaway:** Per the Pinocchio rules in CLAUDE.md — when Pinocchio "looks wrong",
  debug the frame convention before touching the solver. A clean constant offset
  across *all* frames is the signature of a frame origin mismatch, never a bug in
  the kinematics. Cross-validating one point against MuJoCo localises it in seconds.

### Removed media that asserted a milestone which never shipped
- **Symptom:** `media/m4_walking.mp4` and `media/m4_zmp.png` sat at exactly the paths
  `tasks/PLAN.md` names as the M4 gate artifacts, so anyone browsing the folder would
  conclude M4 (ZMP walking) shipped. It did not — M4 is formally blocked, and those
  files were outputs of the deleted open-loop pipeline this file already disowns
  ("zero Pinocchio, open-loop hacks").
- **Fix:** Deleted five orphaned files — `m4_walking.mp4`, `m4_zmp.png`,
  `m3_single_step.mp4`, `walking_results.png`, `lipm_trajectory.png`. The last three
  had lost their producing scripts. Git history preserves all of them.
  `tasks/PLAN.md` gained a SUPERSEDED banner so its M4 artifact rows can no longer be
  read as delivered evidence.
- **Takeaway:** Media files are claims. An artifact left at a gate path is read as
  "this gate passed" regardless of what the docs say. Delete the evidence when you
  delete the pipeline that produced it.

### Two scripts wrote the same video filename
- **Symptom:** `README.md` cited `media/m3e_zmp_walking.mp4`, but no script wrote that
  name; `m3e_zmp_walking_sim.py` wrote `m3e_single_step.mp4` — the *same* path
  `m3e_single_step.py` writes. Running the two demos in order silently overwrote one
  video with the other, and the README-cited file was unreproducible.
- **Fix:** `m3e_zmp_walking_sim.py` now writes `m3e_zmp_walking.mp4` (matching the
  README and the file already on disk). `m3e_single_step.py`'s plot went from
  `m3_step_analysis.png` to `m3e_step_analysis.png` for the `mN_*` convention; the
  on-disk file was renamed to match. Docstrings and summary prints updated too.
- **Takeaway:** Every media file cited in the README must have exactly one producing
  script writing exactly that name. Grep the write paths against the README citations
  before declaring a lab portfolio-ready.

### Scratch files in src/ can hijack pytest collection
- **Fix:** Deleted `src/test_claude.py`, `src/test2.py`, `src/test3.py`, `src/test4.py`,
  `src/test_write.py` (trivial scaffold junk) and `src/walking_demo.py` (the "Phase 3
  capstone" of the deleted architecture, superseded by `m5_capstone_demo.py`).
- **Takeaway:** `test_*.py` anywhere on the collection path is a pytest target. Never
  park scratch files under that name in `src/`.
