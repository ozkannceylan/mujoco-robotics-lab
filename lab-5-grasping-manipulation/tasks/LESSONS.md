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

### 2026-08-13 — Step 5.4: preplace IK seeded from Q_HOME stalls in the joint-limit clip
**Symptom:** `record_pro_demo.py` could not finish a run. `q_preplace` reported `err=0.1272 m` and `CC q_preplace collision-free: False`; the `HOME→PREPLACE` RRT* leg then had a colliding, unreachable goal and `plan_collision_free` raised `RuntimeError`. The pre-existing `media/pick_place_pro.mp4` therefore could not have come from the current 5.1+5.2 code.
**Root cause:** Two compounding issues. (1) The seed. `q_preplace = ik_at(preplace_pos, Q_HOME, ...)` — from Q_HOME the DLS iteration drives `shoulder_pan` the wrong way round, hits the `np.clip(q, -2π, 2π)` boundary at exactly -6.283 rad and stalls there; the returned `best_q` is 127 mm from target. The code comment asserting Q_HOME was the *right* seed was simply wrong. (2) Even when IK converges, the branch matters: all six UR5e joints are revolute over ±2π, so `q` and `q ± 2π` are the same pose but several radians apart in joint space.
**Fix:** Seed preplace from `q_pregrasp` (converges to 0.081 mm), then normalise with the new `nearest_joint_branch()` helper in `grasp_planner.py`. `ik_at()` now also *validates*: it raises if position error > 2 mm, and raises if a config that RRT* must plan to is in collision. A silently-wrong IK can no longer produce a video.
**Takeaway:** Seed DLS IK from the nearest already-solved configuration, never from a distant home pose — and always assert convergence *and* collision-freedom on IK output before feeding it to a planner. A solver that returns `best_q` after `max_iter` never fails loudly on its own.

### 2026-08-13 — Same 2π-branch bug failed `test_plan_transport_finds_path`
**Symptom:** `pytest lab-5-grasping-manipulation/tests/` → 32 passed, 1 failed. `RuntimeError: RRT* failed to find path from [-2.961 ...] to [2.284 ...]` in `grasp_state_machine.py:316`. (README claimed 33 passed, so this had regressed silently.)
**Root cause:** Identical root cause to the entry above, in the *other* IK path. `grasp_planner.compute_grasp_configs` seeds preplace with `q_hint_b[0] = -q_pregrasp[0]` (mirrored shoulder_pan). That seed converges, but lands on the +2π branch: pan = +2.284 rad while pregrasp is at -2.961 rad. RRT* was therefore asked to sweep 5.245 rad on joint 0 — outside what its sampling bounds and iteration budget can bridge — even though the kinematically identical -4.000 rad solution is only 1.038 rad away.
**Fix:** Applied `nearest_joint_branch(q_solution, q_reference)` to preplace (ref = pregrasp) and place (ref = preplace). Pan sweep dropped 5.245 → 1.038 rad; full joint-space distance 1.468 rad. `test_state_machine.py` now 8/8 passed.
**Takeaway:** After IK, always rewind revolute joints onto the 2π branch nearest the previous waypoint. A planner failure between two *individually valid* configurations is the signature of a branch mismatch, not of a genuinely blocked path.

### 2026-08-13 — Step 5.4 self-collision verification (result)
**What was added:** `SelfCollisionMonitor` in `record_pro_demo.py`. It classifies every geom by parent body into `arm` (6 UR5e links + base, 29 geoms), `grip` (`2f85_*`, 28 geoms) and `env` (floor/table/box/target), then scans `d.contact[:d.ncon]` after **every** `mj_step` — not every rendered frame. A self-collision is `arm↔arm` or `arm↔grip`; `grip↔grip` (the 2F-85's own linkage and pads) is tallied separately and never counted as failure. MuJoCo's default parent filtering already suppresses joint-adjacent bodies, so anything reaching the monitor is a genuine non-adjacent overlap.
**Result:** 11050 sim steps checked, **0** steps with self-collision, 0 distinct pairs, 0.000 mm max penetration, 0 robot↔table contacts. `main()` now returns a non-zero exit code if the check ever fails.
**Takeaway:** Verify "no self-collision" on the contact list at simulation rate, not by eyeballing rendered frames — at 60 fps with a 2 ms timestep only 1 step in 8 is ever drawn, so a brief interpenetration is very likely to fall between frames.

### 2026-08-13 — `pathlib.write_text()` silently converted a CRLF file to LF
**Symptom:** After a scripted bulk edit, `git diff --numstat` reported 947 insertions / 727 deletions on `record_pro_demo.py` — the entire file — for what should have been a ~200-line change.
**Root cause:** The file is stored with CRLF endings. Python text mode reads with universal newlines (`\r\n` → `\n`) and `write_text()` writes plain `\n`, so every line changed.
**Fix:** Re-wrote the file with `read_bytes()` / `b.replace(b'\n', b'\r\n')` / `write_bytes()`. Diff dropped to +245/-25.
**Takeaway:** For scripted edits to repo files, use `read_bytes`/`write_bytes`, or pass `newline=''` — and check `git diff --numstat` afterwards. A whole-file diff on a small edit means line endings changed, and it buries the real change in review.

### 2026-08-13 — Capstone reaches DONE without ever moving the box (found, not yet fixed)
**Symptom:** `pick_place_demo.py` runs all 11 states through to `DONE` and prints "✓ Pick-and-place cycle complete", but `Box final pos: [0.350, 0.200, 0.335]` — still Box A. Lateral error 400.0 mm, i.e. the entire A→B distance.
**Root cause (partial):** Not IK — the config summary prints `preplace = [0.350, -0.200, 0.590]`, exactly the intended target. The regenerated `media/ee_trajectory_3d.png` shows the EE stopping ~70 mm short of Box A and ~90 mm short of Box B, and `media/gripper_vs_time.png` shows the fingers closing to 0 mm (nothing between them). So the Cartesian impedance controller in `GraspStateMachine` never converges onto the commanded pose before the gripper is commanded to close.
**Fix:** Not fixed — logged as TODO Step 6.1. Out of scope for Step 5.4, which covers `record_pro_demo.py` (that script *does* complete the cycle; its recorded video ends with the cube on the target pad).
**Takeaway:** A state machine that advances purely on timers/settling heuristics will happily report success on a total miss. Every manipulation demo needs a post-condition assert on the *object* pose, not just on controller state — this defect survived a 33-test suite because no test checks that the box moved.

### 2026-08-13 — Docs listed four plot files the code never writes
**Symptom:** `docs/04_pick_place_results.md` referenced `ee_trajectory.png`, `joint_tracking.png`, `gripper_contact.png`, `state_timeline.png`; `media/` contained only mp4s, and running `pick_place_demo.py` produced three *differently named* files.
**Root cause:** `plot_results()` writes `ee_trajectory_3d.png`, `ee_position_vs_time.png`, `gripper_vs_time.png`. The docs were written from the plan, not from the code, and no one had run the script to disk since.
**Fix:** Corrected the doc list to the three filenames actually produced, and generated them.
**Takeaway:** Documented artefact names must be verified by running the producer, not copied from the design doc. If `media/` is missing files the docs promise, the docs are the thing that is wrong.

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

---

## Step 6.1 Session — Capstone Box Transport (2026-08-13)

The capstone `pick_place_demo.py` reached DONE without transporting the box
(400 mm lateral error). Fixing it uncovered six independent defects. Final
result: box placed **5.7 mm** from target (30 mm tolerance), 33/33 tests.

### L-6.1a: Friction pads were mounted on the OUTSIDE of the fingers
- **Symptom**: Box crept downward ~4 mm/s in the grip during any arm motion,
  escaping after ~2 s of transport. Grip survived static holds and slow lifts.
- **Root cause**: In `ur5e_gripper.xml` the "inner fingertip" friction pads
  (μ=1.5, condim=4) had their y-offset pointing *away* from the gripper
  centreline (+0.009 on the +y finger). They could never touch the object.
  Grasps ran on the low-friction structural finger geoms, gripping the box's
  top edge asymmetrically (finger qpos 0.009/0.016 — mirror equality yields
  under load).
- **Fix**: Flip both pad y-offsets inward. Pad-box contacts went 0 → 8; grip
  creep 4 mm/s → 0; the whole test suite also got 3× faster (70 s vs 230 s).
- **Takeaway**: Verify *which geoms actually carry the grasp contacts*
  (`mj_contactForce` + geom names), not just that "contact: True".

### L-6.1b: Wrist joints chatter under raw diagonal PD at 1 kHz
- **Symptom**: Constant "61 mrad" settle residual on every state, target-independent.
- **Root cause**: Not steady-state error — an aliased ±60 mrad torque-saturated
  limit cycle. Wrist reflected inertia ~0.015 kg·m² makes the discrete damping
  term unstable (Kd·dt/I = 40·0.001/0.015 > 2).
- **Fix**: Inertia-scale the gains through the mass matrix:
  τ = M(q)(Kp·e + Kd·ė) + g(q) with M from `pin.crba` — uniform critically
  damped error dynamics (ω=20 rad/s, ζ=1) on every joint. Settle residuals
  became exactly the 10 mrad gate.
- **Takeaway**: For torque control at fixed dt, per-joint stability depends on
  reflected inertia; normalise with M(q) instead of hand-tuning six gains.

### L-6.1c: URDF and MJCF described different arms → gravity comp was wrong
- **Symptom**: Cartesian impedance sagged ~20 mm; joint holds off by up to
  15 mrad (elbow), matching g_mj − g_pin ≈ 6 Nm.
- **Root cause**: Lab 3's URDF (wrist_3 carries 1.24 kg) vs this lab's MJCF
  (wrist_3 0.56 kg + 0.13 kg custom jaw gripper).
- **Fix**: `load_pinocchio_model(match_scene_inertias=True)` builds the
  analytical model from `ur5e_gripper.xml` itself via `pin.buildModelFromMJCF`
  (finger joints locked, EE frame = `tool0`). Verified: max|g_mj−g_pin| =
  0.00 mNm and FK parity 0.0000 mm at tool0 across random configs.
- **Takeaway**: The "Pinocchio = analytical brain" rule only works if the
  brain models the body being simulated. Cross-validate g(q) against
  `qfrc_bias`, not just FK.

### L-6.1d: IK collision validation must share the planner's collision truth
- **Symptom**: RRT* "failed to find path" — its goal (q_preplace) was
  collision-flagged, so `plan()` returned None immediately.
- **Root cause**: IK is obstacle-blind (known issue) and its solution *family*
  put the upper arm near the table. Worse, the Lab 4 checker models the
  Menagerie UR5e whose thicker upper arm collides where this scene's slim
  box-geom arm does not, forcing a family 5.4 rad away that RRT* could not
  bridge.
- **Fix**: `SceneCollisionChecker` built from `scene_grasp.xml` (duck-types
  the Lab 4 planner interface) is now the single collision truth for both IK
  validation (`compute_grasp_configs(validate_fn=...)`, with fallback seeds +
  random restarts) and RRT*. A planner must plan for the robot it drives.
- **Takeaway**: Validate IK goals with the *same* checker the planner uses,
  at IK time — never let a colliding config become a planning goal.

### L-6.1e: Convergence gates, not fixed-duration handoffs
- **Symptom**: States handed off with whatever tracking error remained
  (~70 mm EE short of the box in the original report); descend "settle"
  froze the *current* pose, locking the error in.
- **Fix**: Joint states servo the trajectory endpoint until <10 mrad and
  <0.05 rad/s (3 s timeout); Cartesian descends drive to the *absolute*
  FK(q_target) pose (correcting accumulated error) until <3 mm. Cartesian
  gains extended to 6D so orientation is held during descends.
- **Takeaway**: A state machine transition is a contract; gate it on
  measured convergence, and log the residual so a silent miss is impossible.

### L-6.1f: Place-descend must stop at touchdown; retract must ascend first
- **Symptom A**: After touchdown the convergence gate kept pushing toward an
  unreachable in-table target, dragging the box (~40 mm error).
  **Fix**: end DESCEND_PLACE after 30 consecutive box-table contact steps.
- **Symptom B**: After a perfect 6 mm placement, RETRACT's RRT path swept the
  open fingers sideways through the box, dragging it 47 mm.
  **Fix**: ascend vertically (mirror of the approach) before planning home.
- **Takeaway**: Symmetry matters: descend/ascend legs belong on both sides of
  contact events; RRT joint-space paths make no promises about the first
  Cartesian direction of motion.

### Transport timing while carrying
TOPP-RA at full limits (3.14 rad/s, 8 rad/s²) is far too aggressive for a
friction pinch carry. `_plan_and_smooth` now takes vel/acc scales; transport
runs at 0.22/0.15 (cf. Lab 4 capstone's 0.18/0.14). Even with fixed pads,
gentle carry timing is what a real deployment would use.

### Final gate evidence (2026-08-13)
| Metric | Value |
|---|---|
| Box final position | [0.350, −0.194, 0.335] m |
| Place target (Box B) | [0.350, −0.200, 0.335] m |
| **Lateral error** | **5.7 mm** (tolerance 30 mm) |
| Joint settles | 10.0 mrad (= gate) at every state |
| Descend/lift settles | 2.4 / 3.0 / 4.1 mm |
| Grasp contact | True (8 pad-box contacts) |
| Test suite | 33/33 passed |
