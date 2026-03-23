# Lab 7: Locomotion Fundamentals — Implementation Plan

## Phase 1: G1 Setup & Standing Balance

### Step 1.1: G1 Model Creation
- Create a simplified G1 humanoid MJCF model (~23 DOF: 6 per leg, waist, locked arms)
- Build corresponding Pinocchio URDF model with `JointModelFreeFlyer` root joint
- Define joint structure: hip (3-DOF: yaw, roll, pitch), knee (1-DOF), ankle (2-DOF: pitch, roll) per leg + waist
- Configure position-controlled actuators with PD servo gains in MuJoCo
- Build `models/scene_flat.xml` including G1 on flat ground with lighting and tracking camera
- **Verify:** Scene loads in MuJoCo, G1 renders correctly, print all joint/actuator names. Confirm DOF count matches expected (floating base 7q/6v + 13 actuated joints). G1 stands stable under gravity for 5 seconds.

### Step 1.2: Lab 7 Common Module
- Create `src/lab7_common.py` with paths, constants, joint mappings, model loaders
- Define joint index slices for left/right leg, waist, locked arms
- Define constants: DT=0.002 (500 Hz), GRAVITY, total DOFs, actuated DOF count
- Define Q_HOME: nominal standing configuration
- Implement Pinocchio↔MuJoCo quaternion conversion utilities: `pin_quat_to_mj()`, `mj_quat_to_pin()`
- Implement `get_state_from_mujoco(mj_model, mj_data) -> (q_pin, v_pin)` for state conversion
- Implement `load_mujoco_model()`, `load_pinocchio_model()` helpers
- **Verify:** Import from test script, all paths resolve, models load, state conversion round-trips correctly.

### Step 1.3: G1 Model Wrapper
- Create `src/g1_model.py` with `G1Model` class wrapping Pinocchio model
- Implement `compute_com(q)` using `pin.centerOfMass()`
- Implement `compute_com_jacobian(q)` for CoM velocity control (3 x nv)
- Implement `fk_foot(q, side)` returning pin.SE3 for left/right foot frames
- Implement `foot_jacobian(q, side)` returning 6 x nv Jacobian
- Implement `gravity_vector(q)` and `mass_matrix(q)` via Pinocchio RNEA/CRBA
- Implement `whole_body_ik(com_target, lfoot_target, rfoot_target, q_ref)` using damped least squares
- Map joint indices between Pinocchio and MuJoCo explicitly and store as class attributes
- Cross-validate FK between Pinocchio and MuJoCo for 5 random configurations
- **Verify:** FK position error < 1 mm between Pinocchio and MuJoCo. CoM computation matches. Mass matrix is symmetric positive-definite.

### Step 1.4: Standing Balance Controller
- Create `src/balance_controller.py` with `StandingBalanceController`
- Implement CoM tracking: PD control on CoM position to keep it above center of support polygon
- `F_com = Kp*(com_d - com) + Kd*(vcom_d - vcom)`
- Map to joint torques via CoM Jacobian: `tau = J_com^T * F_com + g(q)`
- Add posture regulation term to prevent drift: `tau += Kp_posture * (q_home - q) + Kd_posture * (-qd)`
- Compute support polygon from foot contact positions
- Use gravity feedforward for MuJoCo position servo: `ctrl = q_des + qfrc_bias/Kp + Kd*qd_des/Kp`
- **Verify:** G1 stands for 10 seconds without falling. CoM stays within support polygon.

### Step 1.5: Standing Balance Demo
- Create `src/a1_standing_balance.py` demo script
- Load G1, set to home pose, run standing controller
- Apply perturbation forces (5N lateral pushes at pelvis) at t=3s and t=6s
- Verify recovery within 2 seconds after each perturbation
- Save media: CoM trajectory plot, support polygon with CoM projection, joint torques over time
- Record video of perturbation recovery
- **Verify:** Robot recovers from perturbations. CoM returns to center of support polygon.

### Step 1.6: Phase 1 Tests
- Create `tests/test_g1_model.py`: FK cross-validation (5 configs), Jacobian numerical check, CoM correctness, mass matrix symmetry and positive-definiteness
- Create `tests/test_balance.py`: standing stability (10s), perturbation recovery, CoM within support polygon
- **Verify:** All tests pass with `pytest tests/`

## Phase 2: ZMP Planning

### Step 2.1: LIPM Dynamics
- Create `src/lipm_planner.py` (Linear Inverted Pendulum Model)
- Implement LIPM equations: `x_ddot = (g/z_c) * (x - p)` where p = ZMP
- Implement preview control for ZMP→CoM conversion (Kajita 2003)
- Preview controller: minimize ZMP tracking error with look-ahead window (N_preview = 160 steps)
- Compute preview gains offline using discrete algebraic Riccati equation
- **Verify:** Given a step ZMP reference, CoM trajectory is smooth and tracks ZMP with bounded lag. ZMP error converges to zero in steady state.

### Step 2.2: Footstep Planner
- Create `src/footstep_planner.py` with `FootstepPlanner` class
- Define `GaitParams` dataclass: stride_length=0.15m, step_width=0.18m, step_height=0.04m, step_duration=0.6s, double_support_ratio=0.2
- Implement `plan_footsteps(n_steps)` generating alternating left-right placements
- Compute support polygon for each phase (single/double support)
- **Verify:** Footstep positions are physically reasonable. Support polygons are correct.

### Step 2.3: ZMP Trajectory Generation
- Generate ZMP reference trajectory from footstep plan
- During double support: ZMP transitions linearly from trailing to leading foot
- During single support: ZMP stays at stance foot center
- Combine with LIPM preview control to get CoM trajectory
- **Verify:** ZMP stays inside support polygon at all times. CoM trajectory is smooth and bounded.

### Step 2.4: Foot Trajectory Generation
- Add swing foot trajectory generation to footstep planner
- Use cubic or quintic polynomial for foot lift and land phases
- Parameters: step height (lift clearance), smooth takeoff and landing velocities (zero at contact)
- Generate contact schedule: which foot is in stance/swing at each timestep
- **Verify:** Foot trajectories have zero velocity at takeoff and landing. Maximum height matches step_height parameter. No discontinuities.

### Step 2.5: ZMP Planning Demo
- Create `src/a2_zmp_planning.py` demo script
- Visualize: CoM trajectory (X-Y plane), ZMP trajectory, footstep positions, support polygon over time
- Plot foot swing trajectories in 3D
- Save all plots to `media/`
- **Verify:** All trajectories look physically reasonable. ZMP is inside support polygon at all times.

### Step 2.6: Phase 2 Tests
- Create `tests/test_lipm.py`: LIPM dynamics, preview control convergence, ZMP stability, foot trajectory smoothness
- **Verify:** All tests pass before Phase 3.

## Phase 3: Walking Gait Execution

### Step 3.1: Whole-Body Walking IK
- Create `src/walking_controller.py` with `WalkingController` class
- Given CoM trajectory + foot trajectories → joint angles via whole-body IK
- Task priority: CoM tracking > foot placement > posture regulation
- Use Pinocchio IK with damped least squares (from G1Model)
- IK runs at each control step to compute desired joint positions
- **Verify:** IK converges for all timesteps. Joint positions are within limits.

### Step 3.2: Walking Gait Integration
- Integrate all components: footstep planner → LIPM → foot trajectories → IK → MuJoCo
- PD joint tracking with gravity feedforward: `ctrl = q_des + qfrc_bias/Kp + Kd*qd_des/Kp`
- Control loop at 500 Hz, IK at 500 Hz (or 100 Hz with interpolation)
- Handle contact transitions carefully: blend between double and single support
- **Verify:** G1 takes at least 2-3 steps without falling.

### Step 3.3: Walking Gait Tuning
- Tune parameters iteratively:
  - PD gains (Kp, Kd per joint group: hip, knee, ankle, waist)
  - Step timing (step_duration, double_support_ratio)
  - Step geometry (stride_length, step_height, step_width)
  - CoM height (z_c for LIPM)
  - IK damping and posture regularization
- Target: 10+ stable steps on flat ground
- Log all parameter changes and results in LESSONS.md
- **Verify:** G1 completes 10+ steps without falling.

### Step 3.4: Walking Demo
- Create `src/b1_walking_demo.py`
- Execute 10+ steps on flat ground with visualization
- Record video of walking gait
- Save metrics: steps achieved, max CoM deviation from planned, ZMP margin
- **Verify:** 10+ stable steps. Video looks natural.

### Step 3.5: Capstone Demo
- Create `src/capstone_demo.py`
- Full walking sequence with comprehensive visualization
- Generate media: walking trajectory plot (top-down), CoM vs planned CoM, ZMP vs support polygon, joint torques heatmap, foot clearance over time
- Record high-quality video with tracking camera
- **Verify:** All plots generated. Capstone video is smooth and shows stable walking.

### Step 3.6: Phase 3 Tests
- Create `tests/test_walking.py`: gait stability (10+ steps), foot clearance > 0, CoM deviation bounded, ZMP within support polygon
- **Verify:** All tests pass.

## Phase 4: Documentation & Blog

### Step 4.1: English Documentation
- Write `docs/01_g1_setup_balance.md` — G1 model creation, Pinocchio/MuJoCo setup, standing balance, CoM control theory
- Write `docs/02_zmp_planning.md` — LIPM theory, preview control derivation, ZMP trajectory generation, footstep planning
- Write `docs/03_walking_gait.md` — Whole-body IK, gait execution, parameter tuning, results analysis

### Step 4.2: Turkish Documentation
- Translate to `docs-turkish/01_g1_kurulumu_denge.md`
- Translate to `docs-turkish/02_zmp_planlama.md`
- Translate to `docs-turkish/03_yurume_adimi.md`

### Step 4.3: Blog Post
- Write `blog/lab_07_locomotion.md` — "Making a Humanoid Walk — The ZMP Approach"
- Focus: why walking is hard, LIPM as a simplification, preview control intuition, tuning challenges
- Include capstone GIF and key metrics

### Step 4.4: README
- Write `lab-7-locomotion/README.md` with architecture, setup, running demos, results table
- **Verify:** All documentation complete, all referenced images exist in `media/`.
