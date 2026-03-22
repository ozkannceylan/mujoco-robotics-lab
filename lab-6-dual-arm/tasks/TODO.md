# Lab 6: TODO

## Phase 1: Dual-Arm Setup
- [ ] Step 1.1: Create dual UR5e MuJoCo scene (ur5e_left.xml, ur5e_right.xml, scene_dual.xml)
- [ ] Step 1.2: Create dual-arm Pinocchio models (DualArmModel with base transforms, FK, Jacobians)
- [ ] Step 1.3: Create lab6_common.py (paths, constants, model loaders, index slicing)
- [ ] Step 1.4: Implement arm-arm collision checking (DualCollisionChecker with HPP-FCL)
- [ ] Step 1.5: Independent motion test (a1_independent_motion.py, video)
- [ ] Step 1.6: Phase 1 tests (test_dual_model.py, test_dual_collision.py)

## Phase 2: Coordinated Motion
- [ ] Step 2.1: Synchronized trajectory generation (SynchronizedTrajectory, plan_synchronized_linear)
- [ ] Step 2.2: Master-slave coordination mode (plan_master_slave)
- [ ] Step 2.3: Symmetric coordination mode (plan_symmetric, ObjectFrame)
- [ ] Step 2.4: Simultaneous approach test (a2_coordinated_approach.py, video)
- [ ] Step 2.5: Phase 2 tests (test_coordinated_planner.py)

## Phase 3: Cooperative Manipulation
- [ ] Step 3.1: Bimanual grasp state machine (BimanualGraspStateMachine, weld constraint)
- [ ] Step 3.2: Dual impedance controller with coupling (DualImpedanceController, internal force)
- [ ] Step 3.3: Cooperative carry pipeline (b1_cooperative_carry.py: approach->grasp->lift->carry->place)
- [ ] Step 3.4: Rigid grasp constraint handling (weld tuning, rotation < 5 deg)
- [ ] Step 3.5: Phase 3 tests (test_bimanual_grasp.py, test_cooperative_controller.py)

## Phase 4: Documentation & Blog
- [ ] Step 4.1: English documentation (docs/lab6_dual_arm_coordination.md)
- [ ] Step 4.2: Turkish documentation (docs-turkish/lab6_cift_kol_koordinasyonu.md)
- [ ] Step 4.3: Blog post (blog/lab_06_dual_arm.md)
- [ ] Step 4.4: README (lab-6-dual-arm/README.md)

## Phase 5: Capstone Demo
- [ ] Step 5.1: Multi-scenario capstone demo (capstone_demo.py, video)
- [ ] Step 5.2: Metrics collection and visualization (tracking error, forces, distances)
- [ ] Step 5.3: Final validation against success criteria

## Current Focus
> Step 1.1: Create dual UR5e MuJoCo scene

## Blockers
> None
