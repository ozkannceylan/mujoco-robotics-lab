# Lab 7: TODO

## Phase 1: G1 Setup & Standing Balance
- [ ] Step 1.1: Create G1 MJCF model (~23 DOF, arms locked) and Pinocchio URDF
- [ ] Step 1.2: Create lab7_common.py (paths, constants, joint mappings, state conversion)
- [ ] Step 1.3: Create G1 model wrapper (FK, CoM, Jacobians, IK, dynamics via Pinocchio)
- [ ] Step 1.4: Implement standing balance controller (CoM PD + gravity comp)
- [ ] Step 1.5: Standing balance demo with perturbation recovery (a1_standing_balance.py)
- [ ] Step 1.6: Phase 1 tests (test_g1_model.py, test_balance.py)

## Phase 2: ZMP Planning
- [ ] Step 2.1: Implement LIPM dynamics + preview control (lipm_planner.py)
- [ ] Step 2.2: Implement footstep planner (footstep_planner.py)
- [ ] Step 2.3: Generate ZMP reference trajectory from footstep plan
- [ ] Step 2.4: Add foot swing trajectory generation (quintic polynomial)
- [ ] Step 2.5: ZMP planning demo with visualization (a2_zmp_planning.py)
- [ ] Step 2.6: Phase 2 tests (test_lipm.py)

## Phase 3: Walking Gait Execution
- [ ] Step 3.1: Implement whole-body walking IK (walking_controller.py)
- [ ] Step 3.2: Integrate all components: footstep → LIPM → IK → MuJoCo
- [ ] Step 3.3: Tune walking parameters (gains, timing, geometry) for 10+ steps
- [ ] Step 3.4: Walking demo (b1_walking_demo.py)
- [ ] Step 3.5: Capstone demo with metrics and video (capstone_demo.py)
- [ ] Step 3.6: Phase 3 tests (test_walking.py)

## Phase 4: Documentation & Blog
- [ ] Step 4.1: English documentation (docs/01, docs/02, docs/03)
- [ ] Step 4.2: Turkish documentation (docs-turkish/01, docs-turkish/02, docs-turkish/03)
- [ ] Step 4.3: Blog post (blog/lab_07_locomotion.md)
- [ ] Step 4.4: README (lab-7-locomotion/README.md)

## Current Focus
> Step 1.1: Create G1 MJCF model and Pinocchio URDF

## Blockers
> None
