# Lab 6: Dual-Arm Coordination — Implementation Plan

## Phase 1: Dual-Arm Setup

### Step 1.1: Create dual UR5e MuJoCo scene
- Build `models/ur5e_left.xml` and `models/ur5e_right.xml` by adapting Lab 3's `ur5e.xml`. Each arm has unique joint/actuator/body names prefixed with `left_` or `right_`. Right arm placed facing left at ~1.0 m separation with 180-degree yaw offset.
- Build `models/scene_dual.xml` that includes both arms, a shared table, floor, lighting, and a large box object (graspable by both arms).
- **Verify:** Load scene in MuJoCo, confirm both arms render, print all joint and actuator names, confirm 12 joints (6 per arm) and 12 actuators.

### Step 1.2: Create dual-arm Pinocchio models
- Build `src/dual_arm_model.py` with `DualArmModel` class that loads two separate Pinocchio models (one per arm), each with its own base SE3 transform matching the MuJoCo scene placement.
- Provide FK for each arm in the world frame: `fk_left(q_left) -> SE3`, `fk_right(q_right) -> SE3`.
- Provide Jacobians for each arm in world frame.
- **Verify:** Cross-validate FK between Pinocchio (with base offset) and MuJoCo for 5 random configurations per arm. Position error < 1 mm.

### Step 1.3: Create lab6_common.py
- Define paths, constants (NUM_JOINTS_PER_ARM=6, NUM_JOINTS_TOTAL=12), joint limits, torque limits, home configurations for both arms, quaternion utilities.
- Provide `load_mujoco_model()` and `load_dual_pinocchio_model()` helpers.
- Define named constants for MuJoCo joint/actuator index slicing: `LEFT_JOINT_SLICE = slice(0, 6)`, `RIGHT_JOINT_SLICE = slice(6, 12)`.
- **Verify:** Import from test script, confirm all paths resolve, models load without error.

### Step 1.4: Implement arm-arm collision checking
- Build `src/dual_collision_checker.py` with `DualCollisionChecker` class extending Lab 4's collision checking pattern.
- Add collision pairs: left arm self-collision, right arm self-collision, left-right cross-collision, each arm vs. table/environment.
- Use HPP-FCL geometry objects with correct base transforms for each arm.
- Provide `is_collision_free(q_left, q_right) -> bool` and `get_min_distance(q_left, q_right) -> float`.
- **Verify:** Test with arms at home (no collision), then manually set configurations where arms overlap and confirm collision detected. Unit tests with >= 5 cases.

### Step 1.5: Independent motion test
- Build `src/a1_independent_motion.py` that moves each arm independently to several waypoints using impedance control (imported from Lab 3 pattern).
- Verify no collisions during independent motion using the collision checker.
- Record video of both arms moving independently.
- **Verify:** Arms reach waypoints (position error < 5 mm), no collisions detected, video saved to `media/`.

### Step 1.6: Phase 1 tests
- `tests/test_dual_model.py`: FK cross-validation, Jacobian correctness, base transform application.
- `tests/test_dual_collision.py`: Known collision/free configurations, min-distance correctness.
- **Verify:** All tests pass with `pytest tests/`.


## Phase 2: Coordinated Motion

### Step 2.1: Synchronized trajectory generation
- Build `src/coordinated_planner.py` with trajectory generation that produces time-synchronized joint paths for both arms.
- Implement `SynchronizedTrajectory` dataclass holding left and right trajectories with matching timestamps.
- Implement `plan_synchronized_linear(pose_left_start, pose_left_end, pose_right_start, pose_right_end, duration) -> SynchronizedTrajectory` using task-space linear interpolation + IK.
- The slower arm dictates the total duration so both arms arrive simultaneously.
- **Verify:** Generate a trajectory where both arms move to approach poses. Confirm both arms complete at the same timestep. Plot joint trajectories for both arms.

### Step 2.2: Master-slave coordination mode
- Add `plan_master_slave(master_trajectory, object_pose, grasp_offset_left, grasp_offset_right, master="left") -> SynchronizedTrajectory`.
- Master arm follows a given trajectory. Slave arm computes its target at each timestep by: slave_target = object_pose * grasp_offset_slave, where object_pose is derived from master EE + grasp_offset_master_inv.
- **Verify:** Move the left (master) arm through a sequence of waypoints. Confirm the right (slave) arm maintains correct relative pose to the master. Plot relative EE distance over time — should remain constant.

### Step 2.3: Symmetric coordination mode
- Add `plan_symmetric(object_trajectory, grasp_offset_left, grasp_offset_right) -> SynchronizedTrajectory`.
- Both arms derive targets from an object-centric frame: `arm_target = object_pose * grasp_offset`.
- Object-centric frame abstraction: `ObjectFrame` class with `pose` (SE3), methods to get left/right grasp targets.
- **Verify:** Define an object trajectory (straight line move). Confirm both arms track their respective grasp offsets relative to the object. EE-to-object relative pose error < 2 mm.

### Step 2.4: Simultaneous approach test
- Build `src/a2_coordinated_approach.py` demo: both arms start at home, simultaneously approach a box from opposite sides, stop 5 cm from grasp contact.
- Use the collision checker at each timestep to ensure no arm-arm collision during approach.
- Record video.
- **Verify:** Both arms arrive within 1 timestep of each other, no collisions, video saved.

### Step 2.5: Phase 2 tests
- `tests/test_coordinated_planner.py`: Timing synchronization, master-slave relative pose invariance, symmetric mode accuracy.
- **Verify:** All tests pass.


## Phase 3: Cooperative Manipulation

### Step 3.1: Bimanual grasp state machine
- Build `src/bimanual_grasp.py` with `BimanualGraspStateMachine` following Lab 5's `GraspStateMachine` pattern.
- States: IDLE -> APPROACH -> PRE_GRASP -> GRASP -> LIFT -> CARRY -> PLACE -> RELEASE -> RETREAT -> DONE.
- Each state transition has preconditions (e.g., APPROACH->PRE_GRASP requires both arms within 5 cm of grasp poses).
- MuJoCo weld equality constraint activated at GRASP state to simulate rigid grasp (both EEs welded to object).
- **Verify:** Run state machine in simulation, confirm correct state transitions, log state durations.

### Step 3.2: Dual impedance controller with coupling
- Build `src/cooperative_controller.py` with `DualImpedanceController`.
- Each arm runs its own impedance controller (from Lab 3 pattern): `tau = J^T * (K_p * dx + K_d * dxd) + g(q)`.
- Symmetric impedance: both arms use identical gains so internal forces balance.
- Add internal force term: when grasping, add a squeeze force component along the grasp axis (configurable magnitude, e.g., 10 N) to maintain grip.
- Provide `compute_dual_torques(q_left, qd_left, q_right, qd_right, target_left, target_right, grasping: bool) -> (tau_left, tau_right)`.
- **Verify:** With both arms holding object, perturb object in simulation. Confirm object doesn't slip or rotate excessively. Measure internal force — should be near the setpoint.

### Step 3.3: Cooperative carry pipeline
- Build `src/b1_cooperative_carry.py` that executes the full pipeline: approach -> grasp -> lift -> carry -> place -> release.
- Use the bimanual state machine to orchestrate, coordinated planner for trajectories, dual impedance controller for execution.
- Object trajectory: lift 15 cm, translate 30 cm laterally, lower to table.
- **Verify:** Object successfully carried without dropping. Final placement error < 1 cm. Record video.

### Step 3.4: Rigid grasp constraint handling
- Ensure the weld constraint in MuJoCo properly constrains the object.
- Monitor constraint forces during carry — log maximum constraint force.
- If object rotates in-hand, tune impedance gains or add orientation control.
- **Verify:** Object rotation during carry < 5 degrees. Constraint forces within reasonable bounds.

### Step 3.5: Phase 3 tests
- `tests/test_bimanual_grasp.py`: State machine transitions, precondition checking.
- `tests/test_cooperative_controller.py`: Torque output correctness, internal force balancing.
- **Verify:** All tests pass.


## Phase 4: Documentation & Blog

### Step 4.1: English documentation
- Write `docs/lab6_dual_arm_coordination.md` covering:
  - Dual-arm kinematics theory (object-centric frame, relative EE pose)
  - Coordination modes (master-slave vs. symmetric)
  - Internal force control theory
  - Architecture diagrams
  - Results with plots and metrics
- **Verify:** Documentation is complete, all images referenced exist in `media/`.

### Step 4.2: Turkish documentation
- Translate English docs to `docs-turkish/lab6_cift_kol_koordinasyonu.md`.
- **Verify:** Turkish docs have same structure and content as English.

### Step 4.3: Blog post
- Write `blog/lab_06_dual_arm.md` — "From One Arm to Two: The Coordination Challenge."
- Focus on the conceptual leap: why dual-arm is harder than 2x single-arm.
- Include key results and demo GIFs.
- **Verify:** Blog post reads well, links to demo media.

### Step 4.4: README
- Write `lab-6-dual-arm/README.md` with lab overview, setup instructions, how to run demos, and results summary.
- **Verify:** README is complete and accurate.


## Phase 5: Capstone Demo

### Step 5.1: Multi-scenario capstone demo
- Build `src/capstone_demo.py` that demonstrates the full pipeline:
  1. Both arms start at home.
  2. Coordinated approach to a large box on the table.
  3. Bimanual grasp from opposite sides.
  4. Cooperative lift (15 cm).
  5. Cooperative transport (30 cm lateral).
  6. Cooperative place on target location.
  7. Synchronized release and retreat to home.
- Record high-quality video with camera tracking the object.
- **Verify:** Full pipeline runs end-to-end without failure. Video is smooth and demonstrates all coordination modes.

### Step 5.2: Metrics collection and visualization
- Collect and plot:
  - EE position tracking error (both arms) over time
  - Internal force during grasp phase
  - Object pose error during transport
  - Joint torques for both arms
  - Arm-arm minimum distance over time
- Save all plots to `media/`.
- **Verify:** All plots generated, metrics within success criteria.

### Step 5.3: Final validation against success criteria
- [ ] Dual-arm scene with arm-arm collision avoidance
- [ ] Synchronized approach: both arms arrive at grasp points simultaneously
- [ ] Stable cooperative carry: object doesn't slip, rotate (< 5 deg), or drop
- [ ] Capstone demo: bimanual pick, carry, and place of a large object
- [ ] Documentation complete (English + Turkish)
- [ ] Blog post written
- **Verify:** All criteria checked off.
