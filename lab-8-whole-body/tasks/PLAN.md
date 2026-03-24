# Lab 8: Whole-Body Loco-Manipulation — Implementation Plan

## Phase 1: G1 Whole-Body Setup

### Step 1.1: Create G1 loco-manipulation MuJoCo scene
- Obtain Unitree G1 MJCF from MuJoCo Menagerie (or adapt from Lab 7 model with arms unlocked).
- Build `models/scene_loco_manip.xml` that includes: G1 humanoid (~37 DOF with arms), flat ground plane, a table (height ~0.75 m) placed 1.0 m in front, a graspable box object on the table (~10x6x6 cm, ~0.5 kg, free body with freejoint), lighting, and a tracking camera.
- Configure contact pairs: feet-ground, hand-object, object-table.
- Add weld equality constraint (initially disabled) for rigid grasp between left hand and object.
- **Verify:** Load scene in MuJoCo, confirm G1 renders, table and object are visible, print all joint names and actuator names. Confirm DOF count matches expected (floating base 7q/6v + actuated joints). Object sits stable on table under gravity.

### Step 1.2: Create G1 Pinocchio model with floating base
- Build `src/g1_model.py` with `G1WholeBodyModel` class.
- Load G1 URDF into Pinocchio with `pin.JointModelFreeFlyer` as root joint.
- Parse and store frame IDs for: left_foot, right_foot, left_hand, right_hand, pelvis (base_link).
- Implement FK for all task-relevant frames: `fk_frame(q, frame_name) -> SE3`.
- Implement full-body Jacobian: `jacobian_frame(q, frame_name) -> np.ndarray` (6 x nv).
- Implement CoM computation: `compute_com(q) -> np.ndarray` (3,) and `jacobian_com(q) -> np.ndarray` (3 x nv).
- Implement mass matrix: `mass_matrix(q) -> np.ndarray` (nv x nv) via Pinocchio CRBA.
- Implement nonlinear effects: `nle(q, v) -> np.ndarray` (nv,) via Pinocchio RNEA.
- Implement `compute_com_with_load(q, load_mass, load_position)` for CoM compensation when carrying objects.
- Map joint indices between Pinocchio and MuJoCo explicitly and store as class attributes.
- Build selection matrix S: (na x nv) mapping actuated torques to full generalized forces.
- **Verify:** Cross-validate FK (CoM, foot, hand positions) between Pinocchio and MuJoCo for 5 random configurations with the robot standing. Position error < 1 mm.

### Step 1.3: Create lab8_common.py
- Define paths: LAB_DIR, MODELS_DIR, MEDIA_DIR, SCENE_PATH, URDF_PATH.
- Define constants: DT=0.002 (500 Hz), GRAVITY, total DOFs, actuated DOF count, joint limits, torque limits.
- Define named joint groups: LEG_LEFT_INDICES, LEG_RIGHT_INDICES, ARM_LEFT_INDICES, ARM_RIGHT_INDICES, WAIST_INDICES.
- Define Q_STAND: nominal standing configuration (from Lab 7 or manually tuned).
- Provide `load_mujoco_model()`, `load_pinocchio_model()` helpers.
- Provide quaternion conversion utilities: `pin_quat_to_mj()`, `mj_quat_to_pin()`.
- Provide `get_state_from_mujoco(mj_model, mj_data) -> (q_pin, v_pin)` for state conversion.
- Provide `clip_torques(tau, limits) -> np.ndarray`.
- **Verify:** Import from test script, all paths resolve, models load, state conversion round-trips correctly.

### Step 1.4: Implement standing balance controller (baseline)
- Build `src/balance_controller.py` with `StandingBalanceController`.
- Implement CoM tracking: PD control on CoM position to keep it above the center of the support polygon.
- Use whole-body inverse kinematics (pseudoinverse, not QP yet) to convert CoM velocity command into joint velocity command.
- Add gravity compensation via `nle(q, v)`.
- Convert to torques: `tau = M(q) * qdd_des + nle(q, v)`.
- **Verify:** G1 stands for 10 seconds without falling. Apply 5 N lateral perturbation force to pelvis — robot recovers within 2 seconds. CoM stays within support polygon throughout.

### Step 1.5: Phase 1 tests
- `tests/test_g1_model.py`: FK cross-validation (5 configs), Jacobian numerical check, CoM correctness, mass matrix symmetry and positive-definiteness.
- `tests/test_balance.py`: Standing stability (10s), perturbation recovery, CoM within support polygon.
- **Verify:** All tests pass with `pytest tests/`.


## Phase 2: Whole-Body QP Controller

### Step 2.1: Implement task definition framework
- Build `src/tasks.py` defining the task interface and concrete task types.
- `Task` base class with: `compute_error(q, v) -> np.ndarray`, `compute_jacobian(q) -> np.ndarray`, `dim` property, `name` property.
- `CoMTask`: track desired CoM position (3D). Error = com_desired - com_current. Jacobian = J_com (3 x nv). Weight = 1000.
- `FootPoseTask`: track desired foot SE3 pose (6D). Error = log(T_current.inv() * T_desired). Jacobian = J_foot (6 x nv). Weight = 100.
- `HandPoseTask`: track desired hand SE3 pose (6D). Same pattern as FootPoseTask. Weight = 10.
- `PostureTask`: track desired joint configuration (actuated joints only). Error = q_desired - q_current. Jacobian = [0 | I]. Weight = 1.
- Each task has a PD controller: `a_desired = kp * error + kd * error_dot`.
- **Verify:** Unit test each task — compute error and Jacobian at known configurations, check dimensions, verify Jacobian numerically via finite differences.

### Step 2.2: Implement whole-body QP solver
- Build `src/whole_body_qp.py` with `WholeBodyQP` class.
- QP formulation:
  ```
  min  sum_i  w_i * || J_i * qdd - a_d_i ||^2
  s.t. M * qdd + nle = S^T * tau + J_c^T * f_c    (equation of motion)
       A_friction * f_c <= 0                        (linearized friction cones)
       tau_min <= tau <= tau_max                     (torque limits)
       qdd_min <= qdd <= qdd_max                    (acceleration limits)
  ```
- Decision variables: qdd (nv), tau (na), f_c (n_contacts * 3).
- Use OSQP solver via `osqp` Python package.
- Provide `solve(q, v, tasks, contacts) -> QPResult(qdd, tau, f_contacts, solve_time)`.
- **Verify:** With G1 standing, command a hand position offset. QP produces accelerations that move the hand while keeping CoM stable. Equation of motion constraint residual < 1e-6.

### Step 2.3: Implement contact model
- Build `src/contact_model.py` with `ContactState`, `ContactSchedule`.
- `ContactState` enum: STANCE, SWING.
- Friction cone: linearized with 4 facets for QP inequality constraints.
- `get_active_contacts(t) -> list[ContactInfo]` returns feet in contact and their poses.
- **Verify:** Contact schedule correctly reports double-support and single-support phases. Friction cone constraints are properly formed.

### Step 2.4: Standing + reaching demo
- Build `src/a1_standing_reach.py`.
- G1 stands in double support. Command left hand to reach 20 cm forward and 10 cm to the left.
- QP resolves: CoM shifts to compensate, feet stay planted, hand tracks target.
- Plot: CoM trajectory, hand tracking error, joint torques, ZMP position vs support polygon.
- Record video.
- **Verify:** Hand reaches within 2 cm of target. CoM stays within support polygon. Robot does not fall.

### Step 2.5: Phase 2 tests
- `tests/test_tasks.py`: Task error/Jacobian dimensions, numerical Jacobian verification.
- `tests/test_qp.py`: QP solution satisfies dynamics constraint, torque limits respected, weighted priority ordering.
- `tests/test_contact.py`: Contact schedule correctness, friction cone geometry.
- **Verify:** All tests pass.


## Phase 3: Walking + Arm Motion

### Step 3.1: Integrate gait generator from Lab 7
- Build `src/gait_generator.py` reusing Lab 7's LIPM/ZMP pattern.
- `GaitGenerator` class: given footstep plan, produce time-varying desired CoM and foot swing trajectories.
- Generate contact schedule for the QP controller.
- **Verify:** Plot CoM trajectory, foot trajectories, contact schedule for a 6-step walk. CoM stays within support polygon at all times.

### Step 3.2: Walking with fixed arm pose
- Build `src/a2_walk_fixed_arms.py`.
- G1 walks forward 1.0 m (6 steps) while both arms hold a fixed Cartesian pose.
- QP task stack: CoM (w=1000) > feet (w=100) > hands-fixed-pose (w=10) > posture (w=1).
- **Verify:** G1 completes 6 steps without falling. Arm pose drift < 5 cm during walk. Record video.

### Step 3.3: Walking while reaching
- Build `src/a3_walk_and_reach.py`.
- G1 walks forward 0.8 m, then during the last 2 steps, left hand reaches toward a target point.
- Hand target transitions smoothly using 5th-order polynomial.
- **Verify:** Hand reaches within 5 cm of target while walking remains stable. Record video.

### Step 3.4: Phase 3 tests
- `tests/test_gait.py`: CoM trajectory within support polygon, foot trajectory continuity.
- `tests/test_walking.py`: Walk stability (no fall in 6 steps), arm drift within tolerance.
- **Verify:** All tests pass.


## Phase 4: Loco-Manipulation Sequence

### Step 4.1: Implement loco-manipulation state machine
- Build `src/loco_manip_fsm.py` with `LocoManipStateMachine`.
- States: IDLE -> WALK_TO_TABLE -> STOP_AND_STABILIZE -> REACH -> GRASP -> LIFT -> WALK_WITH_OBJECT -> STOP_AND_STABILIZE_2 -> PLACE -> RELEASE -> RETREAT -> DONE.
- Each state has: entry conditions, active task set, exit conditions.
- **Verify:** State machine transitions correctly in simulation. Log all state transitions with timestamps.

### Step 4.2: Implement CoM compensation for carried object
- When object is grasped, the effective CoM shifts toward the hand.
- Update QP CoM task target: shift desired CoM to keep combined CoM (robot + object) above support polygon.
- **Verify:** With a 0.5 kg object held in left hand, combined CoM is computed correctly. Standing balance is maintained.

### Step 4.3: Build capstone pipeline
- Build `src/b1_loco_manip_pipeline.py` that runs the full sequence:
  1. G1 starts 1.0 m from table in standing pose
  2. Walks forward to the table (4 steps)
  3. Stops and stabilizes (1 second)
  4. Left hand reaches to object on table
  5. Grasps object (weld constraint activated)
  6. Lifts object 10 cm off table
  7. Walks backward 0.5 m carrying object (3 steps)
  8. Stops and stabilizes
  9. Releases and retreats hand to nominal
- Record high-quality video.
- **Verify:** Full pipeline completes without the robot falling or dropping the object.

### Step 4.4: Phase 4 tests
- `tests/test_fsm.py`: State machine transitions, entry/exit conditions.
- `tests/test_com_compensation.py`: Combined CoM correctness with load.
- `tests/test_pipeline.py`: End-to-end pipeline completes.
- **Verify:** All tests pass.


## Phase 5: Documentation & Blog

### Step 5.1: English documentation
- Write `docs/01_whole_body_qp.md` — QP formulation derivation, task hierarchy, OSQP setup, standing + reaching results.
- Write `docs/02_walking_manipulation.md` — Gait integration with QP, arm motion during locomotion, CoM compensation for carried objects.
- Write `docs/03_loco_manip_pipeline.md` — State machine design, full pipeline walkthrough, capstone results.
- **Verify:** All documentation complete, images referenced exist in `media/`.

### Step 5.2: Turkish documentation
- Translate to `docs-turkish/01_tum_vucut_qp.md`.
- Translate to `docs-turkish/02_yuruyus_manipulasyon.md`.
- Translate to `docs-turkish/03_loko_manipulasyon_boru_hatti.md`.

### Step 5.3: Blog post
- Write `blog/lab_08_whole_body.md` — "Walking and Working — Whole-Body Humanoid Control."

### Step 5.4: README
- Write `lab-8-whole-body/README.md` with overview, architecture, setup, demos, results.


## Phase 6: Capstone Demo & Metrics

### Step 6.1: Full capstone demo with visualization
- Build `src/capstone_demo.py` with camera tracking, FSM state overlay, multiple angles.

### Step 6.2: Metrics collection and visualization
- Collect and plot: CoM vs support polygon, ZMP trajectory, hand tracking error, joint torques heatmap, QP solve time histogram, contact forces.

### Step 6.3: Final validation against success criteria
- [ ] Whole-body QP resolves arm + balance tasks without instability
- [ ] G1 reaches for objects while standing without losing balance
- [ ] G1 walks while maintaining arm pose (carrying behavior)
- [ ] Capstone demo: walk -> grasp -> walk with object -> place
- [ ] Documentation complete (English + Turkish)
- [ ] Blog post written
