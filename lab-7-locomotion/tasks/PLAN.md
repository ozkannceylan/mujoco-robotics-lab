# Lab 7: Locomotion Fundamentals — Implementation Plan

## Overview
Implement stable bipedal walking on the Unitree G1 humanoid using ZMP-based planning and
whole-body IK. This lab is the paradigm shift from fixed-base (Labs 1–6) to floating-base
dynamics where the robot can fall.

**Platform:** Unitree G1 (29-DOF) from MuJoCo Menagerie at `vla_zero_to_hero/third_party/mujoco_menagerie/unitree_g1/`
**Pinocchio:** Load via `buildModelFromMJCF` (no separate URDF needed)

---

## Phase 1: G1 Setup & Standing Balance

### Step 1.1: Folder structure and lab7_common.py
- Load G1 in MuJoCo and Pinocchio
- Define G1 joint/actuator/velocity index maps
- Define standing keyframe constants (pelvis z=0.757m after settling, CoM z≈0.66m)
- Utilities: `mj_to_pin_state()`, `pin_to_mj_state()` (quaternion convention swap)
- **Expected output:** Both models load without errors, correct nq/nv/nu printed

### Step 1.2: standing_controller.py
- Reset to keyframe, set ctrl = q_stand for 29 actuators
- Run for 5s, apply 5N perturbation force at t=3s for 0.2s
- Record and plot CoM x,y,z over time
- Verify robot recovers to within 5mm of initial CoM after perturbation
- **Expected output:** Plot of CoM trajectory showing recovery, `tests/test_standing.py` passes

---

## Phase 2: ZMP Planning

### Step 2.1: lipm_planner.py — LIPM dynamics and preview controller
- Implement `LIPMPreviewController` (Kajita 2003)
  - Discrete-time LIPM: x_{k+1} = A x_k + b u_k, p = C x_k
  - Augmented system with error integrator for tracking
  - Solve DARE via `scipy.linalg.solve_discrete_are`
  - Compute state feedback gains K_e, K_s and preview gains f(j)
- **Expected output:** Preview gains computed without errors

### Step 2.2: lipm_planner.py — Footstep plan and trajectory generation
- Implement `FootstepPlanner`: alternating left/right steps, fixed stride
  - Step length: 0.10m, step width: 0.10m (G1 foot half-width)
  - 12 steps total (6 left, 6 right)
- Implement `generate_zmp_reference()`: ZMP stays over support foot during single support,
  linearly interpolates during double support
- Implement `swing_trajectory()`: parabolic height, linear horizontal interpolation
- Apply LIPMPreviewController to generate CoM trajectory from ZMP reference
- **Expected output:** Plot of CoM trajectory, ZMP trajectory, support polygon over time

### Step 2.3: Tests
- `tests/test_lipm.py`: verify ZMP stays inside support polygon throughout trajectory
- **Success criterion:** max ZMP deviation from support polygon < 1mm

---

## Phase 3: Walking Gait Execution

### Step 3.1: whole_body_ik.py
- Implement `WholeBodyIK` using Pinocchio:
  - Fix pelvis at desired position and upright orientation
  - Solve left leg IK: Jacobian of `left_foot` frame w.r.t. left leg joints (v[6:12])
  - Solve right leg IK: Jacobian of `right_foot` frame w.r.t. right leg joints (v[12:18])
  - Damped least squares: `J† = Jᵀ(JJᵀ + λI)⁻¹`
  - Integrate with `pin.integrate()` for proper quaternion handling
- **Expected output:** IK converges for typical walking configurations

### Step 3.2: walking_demo.py — Gait execution controller
- Pre-compute full trajectory offline: CoM, left foot, right foot at T=0.01s steps
- Gait phase state machine: INIT → LEFT_SWING → DOUBLE_SUPPORT → RIGHT_SWING → ...
- At each MuJoCo step:
  1. Look up desired CoM and foot positions from pre-computed trajectory
  2. Solve whole-body IK → desired joint angles
  3. Send to MuJoCo position actuators (legs + waist)
  4. Keep arms at home pose
- **Expected output:** Robot walks 10+ steps without falling

### Step 3.3: Tuning and validation
- Tune: step timing, step length, LIPM z_c, IK lambda, perturbation gains
- Validate: 10+ steps, gait visually smooth, ZMP in support polygon
- **Success criterion:** 10+ stable steps on flat ground

---

## Phase 4: Documentation & Blog

### Step 4.1: docs/LAB_07.md
- Floating-base dynamics, ZMP theory, LIPM derivation, results

### Step 4.2: docs-turkish/LAB_07.md
- Turkish translation

### Step 4.3: blog/lab_07_locomotion.md
- Blog post: "Making a Humanoid Walk: The ZMP Approach"

### Step 4.4: README.md
- Lab summary for GitHub

---

## Key Parameters (to be tuned)
| Parameter | Initial Value | Notes |
|-----------|--------------|-------|
| z_c (LIPM height) | 0.66 m | CoM height above ground |
| pelvis_z_stand | 0.757 m | Pelvis height after settling |
| step_length | 0.10 m | Forward displacement per step |
| step_width | 0.10 m | Y distance from center to foot |
| step_height | 0.05 m | Swing foot peak clearance |
| T_single_support | 0.8 s | Single support duration |
| T_double_support | 0.2 s | Double support duration |
| N_preview | 200 | Preview window (at T=0.01s) |
| IK_lambda | 1e-3 | Damped LS regularization |
| IK_step_size | 0.5 | IK integration step |
