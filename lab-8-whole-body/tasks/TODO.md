# Lab 8: TODO

## Phase 1: G1 Whole-Body Setup
- [ ] Step 1.1: Create G1 loco-manipulation MuJoCo scene (scene_loco_manip.xml with table + object)
- [ ] Step 1.2: Create G1 Pinocchio model (G1WholeBodyModel with floating base FK, Jacobians, CoM, dynamics)
- [ ] Step 1.3: Create lab8_common.py (paths, constants, joint groups, state conversion, model loaders)
- [ ] Step 1.4: Implement standing balance controller (CoM PD + pseudoinverse, perturbation recovery)
- [ ] Step 1.5: Phase 1 tests (test_g1_model.py, test_balance.py)

## Phase 2: Whole-Body QP Controller
- [ ] Step 2.1: Implement task definition framework (CoMTask, FootPoseTask, HandPoseTask, PostureTask)
- [ ] Step 2.2: Implement whole-body QP solver (WholeBodyQP with OSQP, dynamics + friction + torque constraints)
- [ ] Step 2.3: Implement contact model (ContactState, ContactSchedule, friction cone linearization)
- [ ] Step 2.4: Standing + reaching demo (a1_standing_reach.py)
- [ ] Step 2.5: Phase 2 tests (test_tasks.py, test_qp.py, test_contact.py)

## Phase 3: Walking + Arm Motion
- [ ] Step 3.1: Integrate gait generator from Lab 7 (GaitGenerator with LIPM, foot swing profiles)
- [ ] Step 3.2: Walking with fixed arm pose demo (a2_walk_fixed_arms.py)
- [ ] Step 3.3: Walking while reaching demo (a3_walk_and_reach.py)
- [ ] Step 3.4: Phase 3 tests (test_gait.py, test_walking.py)

## Phase 4: Loco-Manipulation Sequence
- [ ] Step 4.1: Implement loco-manipulation state machine (LocoManipStateMachine with 12 states)
- [ ] Step 4.2: Implement CoM compensation for carried object (compute_com_with_load)
- [ ] Step 4.3: Build capstone pipeline (b1_loco_manip_pipeline.py: walk → grasp → carry → place)
- [ ] Step 4.4: Phase 4 tests (test_fsm.py, test_com_compensation.py, test_pipeline.py)

## Phase 5: Documentation & Blog
- [ ] Step 5.1: English documentation (docs/01, docs/02, docs/03)
- [ ] Step 5.2: Turkish documentation (docs-turkish/01, docs-turkish/02, docs-turkish/03)
- [ ] Step 5.3: Blog post (blog/lab_08_whole_body.md)
- [ ] Step 5.4: README (lab-8-whole-body/README.md)

## Phase 6: Capstone Demo & Metrics
- [ ] Step 6.1: Full capstone demo with visualization (capstone_demo.py)
- [ ] Step 6.2: Metrics collection and visualization (CoM, ZMP, tracking error, torques, solve time)
- [ ] Step 6.3: Final validation against success criteria

## Current Focus
> Step 1.1: Create G1 loco-manipulation MuJoCo scene

## Blockers
> None
