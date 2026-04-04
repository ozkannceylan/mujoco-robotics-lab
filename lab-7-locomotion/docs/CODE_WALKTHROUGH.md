# Lab 7: Code Walkthrough

Recommended reading order for understanding the Lab 7 codebase.

## Prerequisites

- Familiarity with MuJoCo position actuators and `mj_step`
- Basic Pinocchio usage (FK, Jacobians)
- Understanding of floating-base conventions (nq != nv, quaternions)

## Reading Order

### 1. `src/lab7_common.py` — Start Here

Central configuration hub. Read this first to understand:
- G1 model dimensions: NQ=36, NV=35, NU=29
- qpos/qvel/ctrl slice definitions (where each body group lives)
- Quaternion conversion functions: `mj_quat_to_pin()`, `pin_quat_to_mj()`
- The critical Z offset: `PELVIS_MJCF_Z = 0.793`
- `mj_qpos_to_pin()`: the master conversion function (quaternion swap + Z offset)
- Model loaders: `load_g1_mujoco()`, `load_g1_pinocchio()`

**Key constants to remember:**
```python
PELVIS_MJCF_Z = 0.793   # pin_q[2] = mj_qpos[2] - 0.793
Z_C = 0.66               # LIPM CoM height
FOOT_Y_OFFSET = 0.064452 # lateral hip offset
```

### 2. `src/m0_explore_g1.py` — Model Exploration

First script. Loads the G1, prints joint info, demonstrates freefall (zeroing actuator gains), and captures T-pose screenshot. Read to understand the model structure.

### 3. `src/m1_standing.py` — Gravity-Compensated PD

The foundational control pattern used everywhere:
```python
ctrl = q_ref + qfrc_bias[6:35] / KP
```

This feedforward pattern adds gravity compensation through position actuators. Read lines 116–127 for the core control loop. Note how `xfrc_applied` is used for the push test.

### 4. `src/m2_com_balance.py` — CoM Tracking

Demonstrates Pinocchio-MuJoCo CoM cross-validation and the support polygon computation. The CoM Jacobian balance controller (`J_com_pinv @ com_error`) shows the first use of Pinocchio for online corrections. Key lesson: corrections must be small (<0.05 rad) because Kp=500 amplifies position errors.

### 5. `src/m3a_pinocchio_validation.py` — Jacobian Validation

Pure math validation — no simulation. Uses central finite differences to verify every Pinocchio Jacobian column for all 12 leg joints. Read this to understand:
- How `pin.integrate()` correctly perturbs quaternion configurations
- Why central differences (O(eps^2)) are necessary over forward differences (O(eps))
- The frame IDs used: left_ankle_roll_link=15, right_ankle_roll_link=28

### 6. `src/m3b_foot_fk_validation.py` — FK Cross-Validation

Tests FK agreement between Pinocchio and MuJoCo at 10 random configurations. Confirms that `mj_qpos_to_pin()` conversion is correct — 0.000mm error for all comparisons. Read to understand the validation methodology.

### 7. `src/m3c_static_ik.py` — Whole-Body IK (Core Algorithm)

**Most important file.** Contains `whole_body_ik()` — the stacked Jacobian DLS solver used by all subsequent code.

Read carefully:
- Lines 79–109: Function signature and docstring
- Lines 126–164: Forward kinematics + task error computation
- Lines 172–214: Stacked Jacobian construction + DLS solve
- Lines 217: `pin.integrate()` for configuration update (NOT `q += dq`)
- The `fix_base` parameter: when True, only optimizes actuated joints (v-indices 6:nv)

Test cases in `main()` demonstrate identity convergence (0 iters), lateral/forward CoM shifts.

### 8. `src/m3d_weight_shift.py` — IK in Simulation (Proven Pattern)

The most reliable dynamic demo. Read to understand the **pre-computed IK** pattern:

1. **Settle** (line 187): 1s of standing PD
2. **Convert** (line 203): MuJoCo state → Pinocchio convention
3. **Pre-compute** (lines 73–121): IK at 11 waypoints, each warm-started from previous
4. **Replay** (lines 253–275): Cosine-smooth interpolation + feedforward + velocity damping

Key control law (line 270–272):
```python
ctrl += bias / KP_SERVO           # gravity feedforward
ctrl -= K_VEL * qvel / KP_SERVO   # velocity damping
```

### 9. LIPM Pipeline (Theory — not successfully executed)

Read these in order to understand the walking planner:

#### `src/zmp_reference.py`
- `Footstep` and `WalkingTiming` data structures
- `generate_footstep_plan()`: alternating L/R steps
- `generate_zmp_reference()`: piecewise-constant/linear ZMP trajectory
- `get_phase_at_time()`: phase lookup for any time t

#### `src/lipm_preview_control.py`
- `PreviewControllerParams` / `PreviewControllerGains` data structures
- `compute_preview_gains()`: solves discrete Riccati equation (offline, once)
- `preview_control_step()`: one-step update with integral error tracking
- `generate_com_trajectory()`: batch CoM generation from ZMP reference

#### `src/swing_trajectory.py`
- `generate_swing_trajectory()`: cubic horizontal + parabolic vertical foot path
- Simple but effective for flat-ground walking

### 10. `src/m3e_zmp_walking_sim.py` — Walking Attempt (Failed)

The most complex script. Read to understand:
- How the LIPM planner, IK, and swing trajectory integrate
- The online IK approach (fix_base=True, max_iter=5)
- Phase-dependent ankle feedback and feedforward
- **Why it fails**: tracking errors at ankles → ZMP drift → CoM divergence

This file is the primary evidence for the execution gap analysis in the architecture doc.

### 11. `src/m5_capstone_demo.py` — Capstone

Combines standing + push recovery + weight shift into a single demonstration. Also generates an LIPM planner overlay plot (theory only). Shows the final, proven control pipeline.

## Files Not in Reading Order

| File | Purpose |
|------|---------|
| `src/lipm_planner.py` | Earlier LIPM implementation (superseded by `lipm_preview_control.py`) |
| `src/standing_controller.py` | Earlier standing controller (superseded by `m1_standing.py`) |
| `src/whole_body_ik.py` | Earlier IK implementation (superseded by `m3c_static_ik.py`) |
| `src/walking_demo.py` | Earlier walking demo (superseded by `m3e_zmp_walking_sim.py`) |
| `src/m3e_single_step.py` | Single step attempt variant |
| `src/m3e_zmp_walking_plan.py` | ZMP plan visualization script |
| `src/test*.py` | Development test scripts |

## Key Patterns

### Pattern 1: Gravity Feedforward
```python
ctrl = q_ref + qfrc_bias[6:35] / KP_SERVO
```
Used in every controller. Compensates gravity through position actuator offsets.

### Pattern 2: Velocity Damping
```python
ctrl -= K_VEL * qvel[6:35] / KP_SERVO
```
Essential because Menagerie G1 actuators have Kd=0. K_VEL=40 gives near-critical damping.

### Pattern 3: Pre-computed IK Replay
```python
# Offline: solve IK at N waypoints
waypoints = [whole_body_ik(..., com_target=com_i) for i in range(N)]
# Online: cosine-smooth interpolation
s = 0.5 * (1 - cos(pi * t / T))
q_target = interp_waypoints(waypoints, s)
```
More robust than online IK feedback for position actuators.

### Pattern 4: MuJoCo → Pinocchio State Conversion
```python
pin_q = mj_qpos_to_pin(mj_data.qpos)  # swap quat + Z offset
pin.forwardKinematics(model, data, pin_q)
pin.updateFramePlacements(model, data)
# Now use pin_data.oMf[frame_id] for SE3 poses
```
