# Lab 7: Locomotion Fundamentals — Architecture

## Module Map

| File | Role |
|------|------|
| `src/lab7_common.py` | Model loading, G1 joint/actuator index maps, quaternion utilities |
| `src/standing_controller.py` | Phase 1: standing balance demo with perturbation test |
| `src/lipm_planner.py` | Phase 2: LIPM preview controller, footstep plan, trajectory generation |
| `src/whole_body_ik.py` | Phase 3: task-space Jacobian IK for leg+waist joints |
| `src/walking_demo.py` | Phase 3: capstone — pre-computes trajectory, executes walking in MuJoCo |

## Data Flow

```
FootstepPlanner (footstep positions)
        │
        ▼ ZMP reference p_ref(t)
LIPMPreviewController
        │
        ▼ CoM trajectory x_c(t), y_c(t)  +  foot trajectories
WholeBodyIK (Pinocchio)
  fix pelvis at (x_c, y_c, z_pelvis)
  solve left_foot → left leg joints
  solve right_foot → right leg joints
        │
        ▼ q_des[7:22] (15 leg+waist joints)
MuJoCo G1 position actuators (kp=500)
        │
        ▼ contact forces, robot state
```

## Key Interfaces

### lab7_common.py
```python
G1_SCENE_PATH: Path           # absolute path to unitree_g1/scene.xml
G1_MJCF_PATH: Path            # absolute path to unitree_g1/g1.xml
Z_C: float = 0.66             # CoM height above ground (LIPM parameter)
PELVIS_Z_STAND: float = 0.757 # Pelvis z height in standing (after settling)

# Joint index maps (in qpos)
Q_BASE   = slice(0, 7)   # xyz + quaternion (MuJoCo: w,x,y,z)
Q_LEFT_LEG  = slice(7, 13)
Q_RIGHT_LEG = slice(13, 19)
Q_WAIST  = slice(19, 22)
Q_LEFT_ARM  = slice(22, 29)
Q_RIGHT_ARM = slice(29, 36)

# Velocity index maps (in qvel, nv=35)
V_BASE      = slice(0, 6)
V_LEFT_LEG  = slice(6, 12)
V_RIGHT_LEG = slice(12, 18)
V_WAIST     = slice(18, 21)

# Actuator index maps (in ctrl, nu=29)
CTRL_LEFT_LEG  = slice(0, 6)
CTRL_RIGHT_LEG = slice(6, 12)
CTRL_WAIST     = slice(12, 15)
CTRL_LEFT_ARM  = slice(15, 22)
CTRL_RIGHT_ARM = slice(22, 29)

def load_g1_mujoco() -> tuple[mujoco.MjModel, mujoco.MjData]
def load_g1_pinocchio() -> tuple[pin.Model, pin.Data]
def mj_qpos_to_pin(mj_qpos: np.ndarray) -> np.ndarray  # swap quaternion
def pin_q_to_mj(pin_q: np.ndarray) -> np.ndarray        # swap quaternion
```

### lipm_planner.py
```python
class LIPMPreviewController:
    """1D preview controller (Kajita 2003). Apply to x and y axes separately."""
    def __init__(self, T, z_c, g, Q_e, R, N_preview)
    def reset(self, x_init, e_init)
    def step(self, p_ref_k, p_ref_preview) -> tuple[np.ndarray, float]
    # Returns: (x_next_state, p_actual_zmp)

class FootstepPlanner:
    """Generate alternating footstep positions."""
    def __init__(self, step_length, step_width, n_steps)
    def generate(self, x0_left, x0_right) -> list[dict]
    # Returns list of {foot: 'left'|'right', pos: np.ndarray, t_start, t_end}

def generate_zmp_reference(footsteps, T_ss, T_ds, dt) -> np.ndarray
    # (N,2) array of [p_ref_x, p_ref_y] at each timestep

def swing_trajectory(t, T_ss, p_start, p_end, height) -> np.ndarray
    # 3D foot position at time t during swing phase

def plan_walking_trajectory(n_steps, ...) -> dict
    # Returns full pre-computed dict: com_x, com_y, lfoot, rfoot, phase, times
```

### whole_body_ik.py
```python
class WholeBodyIK:
    def __init__(self, pin_model, pin_data, left_foot_frame, right_foot_frame)
    def solve(
        self,
        q_current: np.ndarray,      # Pinocchio full qpos (nq=36)
        pelvis_pos_des: np.ndarray,  # (3,) desired pelvis world position
        lfoot_pos_des: np.ndarray,   # (3,) desired left foot world position
        rfoot_pos_des: np.ndarray,   # (3,) desired right foot world position
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> np.ndarray  # Returns updated q (nq=36) with leg joints solved
```

### standing_controller.py
```python
def run_standing_demo(duration_s, perturbation_force_N, perturbation_time_s) -> None
    # Runs MuJoCo simulation, applies perturbation, plots CoM trajectory
```

### walking_demo.py
```python
def main() -> None
    # Full walking capstone demo: plan → execute → plot results
```

## Model Files

| File | Source | Purpose |
|------|--------|---------|
| `unitree_g1/g1.xml` | `/home/ozkan/projects/vla_zero_to_hero/third_party/mujoco_menagerie/unitree_g1/` | G1 MJCF model for MuJoCo |
| `unitree_g1/scene.xml` | same | Scene with ground plane |

Path is resolved in `lab7_common.py` using `Path(__file__).resolve()`.

## G1 Model Properties

| Property | Value |
|----------|-------|
| Total DOF (nq) | 36 (7 base + 29 joints) |
| Velocity DOF (nv) | 35 (6 base + 29 joints) |
| Actuated joints (nu) | 29 |
| Leg joints per side | 6 |
| Waist joints | 3 |
| Arm joints per side | 7 |
| Standing pelvis height | 0.79 m (keyframe), ~0.757 m (settled) |
| Standing CoM height | ~0.659 m |
| Foot separation (y) | ±0.118 m |

## Floating Base Conventions

| Framework | Quaternion Order | Base DOF (q) | Base DOF (v) |
|-----------|-----------------|--------------|--------------|
| MuJoCo | [w, x, y, z] | 7 | 6 |
| Pinocchio (FreeFlyer) | [x, y, z, w] | 7 | 6 |

**Conversion:** `pin_q[3:7] = [mj_q[4], mj_q[5], mj_q[6], mj_q[3]]` (move w from index 3 to index 6)

## Dependencies on Previous Labs

None — Lab 7 uses the G1 model directly from the vla_zero_to_hero menagerie.
The floating-base paradigm is completely new; no code is imported from Labs 1–6.

## Control Architecture Rationale

1. **Position servos (kp=500)**: The G1 in MuJoCo Menagerie uses position actuators.
   Standing just requires `ctrl = q_stand`. Walking requires IK to compute the right targets.

2. **Separate x/y LIPM**: The 2D LIPM separates into two independent 1D problems.
   This works for flat ground with small lateral dynamics.

3. **Decoupled leg IK**: Solve left and right leg IK independently (no coupling).
   Each leg IK is a 3-task (foot position) / 6-DOF problem (overconstrained null-space for
   posture). This is simpler than full whole-body centroidal control (Lab 8 territory).

4. **Pre-computed trajectory**: The full ZMP/CoM/foot trajectory is computed offline before
   the simulation starts. Online control loop just looks up and executes.
