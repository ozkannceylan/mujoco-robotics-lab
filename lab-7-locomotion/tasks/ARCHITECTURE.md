# Lab 7: Locomotion Fundamentals — Architecture

## Module Map

```
lab-7-locomotion/
├── src/
│   ├── lab7_common.py              # Paths, constants, joint mappings, model loaders
│   ├── g1_model.py                 # G1 Pinocchio wrapper: FK, CoM, Jacobians, IK, dynamics
│   ├── balance_controller.py       # Standing balance via CoM PD + gravity comp
│   ├── lipm_planner.py             # Linear Inverted Pendulum Model + preview control
│   ├── footstep_planner.py         # Footstep sequence, ZMP reference, foot trajectories
│   ├── walking_controller.py       # Whole-body IK for walking gait execution
│   ├── a1_standing_balance.py      # Demo: standing with perturbation recovery
│   ├── a2_zmp_planning.py          # Demo: ZMP trajectory visualization
│   ├── b1_walking_demo.py          # Demo: 10+ step walking
│   └── capstone_demo.py            # Capstone: full walking with metrics and video
│
├── models/
│   ├── g1_humanoid.xml             # G1 MJCF model (simplified ~23 DOF, arms locked)
│   ├── g1_humanoid.urdf            # G1 URDF for Pinocchio (free-flyer root)
│   └── scene_flat.xml              # Scene: G1 on flat ground + lighting + camera
│
├── tests/
│   ├── test_g1_model.py            # FK cross-validation, Jacobian, CoM, mass matrix
│   ├── test_balance.py             # Standing stability, perturbation recovery
│   ├── test_lipm.py                # LIPM dynamics, preview control, ZMP stability
│   └── test_walking.py             # Walking stability, foot clearance, step count
│
├── docs/
│   ├── 01_g1_setup_balance.md      # G1 model, standing balance, CoM control
│   ├── 02_zmp_planning.md          # LIPM, preview control, ZMP theory
│   └── 03_walking_gait.md          # IK, gait execution, results
│
├── docs-turkish/
│   ├── 01_g1_kurulumu_denge.md
│   ├── 02_zmp_planlama.md
│   └── 03_yurume_adimi.md
│
├── media/                          # Videos, plots, GIFs
├── tasks/                          # PLAN, ARCHITECTURE, TODO, LESSONS
└── README.md
```

## Data Flow

```
┌──────────────────────────────────────┐
│         FootstepPlanner               │
│  (footstep_planner.py)                │
│                                       │
│  Inputs: n_steps, GaitParams          │
│  Outputs:                             │
│    - footstep_positions [(x,y,z)]     │
│    - zmp_reference(t)                 │
│    - foot_trajectories(t)             │
│    - contact_schedule(t)              │
│    - support_polygons(t)              │
└────────────────┬──────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│         LIPMPlanner                   │
│  (lipm_planner.py)                    │
│                                       │
│  Inputs: zmp_reference(t)             │
│  Preview control (Kajita 2003)        │
│  Outputs: com_trajectory(t)           │
└────────────────┬──────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│         WalkingController             │
│  (walking_controller.py)              │
│                                       │
│  Inputs:                              │
│    - com_trajectory(t)                │
│    - left_foot_trajectory(t)          │
│    - right_foot_trajectory(t)         │
│    - q_current                        │
│  Uses: G1Model (Pinocchio IK)        │
│  Outputs: q_desired(t)               │
└────────────────┬──────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│  MuJoCo Position Servo + Feedforward  │
│                                       │
│  ctrl = q_des + qfrc_bias/Kp         │
│       + Kd*qd_des/Kp                 │
│                                       │
│  (Compensates gravity droop and       │
│   velocity lag in Menagerie servos)   │
└────────────────┬──────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│         MuJoCo Simulation             │
│                                       │
│  mj_step(mj_model, mj_data)          │
│  Read back: qpos, qvel, contacts     │
│  Render cameras                       │
└──────────────────────────────────────┘
                 │
                 └─→ loop at 500 Hz
```

### Data flow summary:
1. **FootstepPlanner** generates the footstep sequence, ZMP reference, foot swing trajectories, and contact schedule from gait parameters.
2. **LIPMPlanner** converts the ZMP reference into a smooth CoM trajectory using preview control.
3. **WalkingController** solves whole-body IK at each timestep: given desired CoM, left foot, and right foot poses, compute joint angles using Pinocchio damped least squares.
4. **MuJoCo position servo** with gravity feedforward sends joint targets to the simulation. The feedforward term `qfrc_bias/Kp` compensates for gravity droop in Menagerie-style actuators.
5. **MuJoCo** steps physics, produces new joint state, and the loop repeats.

## Key Interfaces

### lab7_common.py

```python
from pathlib import Path
import numpy as np

# ---- Paths ----
LAB_DIR: Path                       # lab-7-locomotion/
PROJECT_ROOT: Path                  # mujoco-kinematics-lab/
MODELS_DIR: Path                    # lab-7-locomotion/models/
MEDIA_DIR: Path                     # lab-7-locomotion/media/
SCENE_FLAT_PATH: Path               # lab-7-locomotion/models/scene_flat.xml
G1_URDF_PATH: Path                  # lab-7-locomotion/models/g1_humanoid.urdf

# ---- Constants ----
NUM_LEG_JOINTS: int = 6             # per leg
NUM_WAIST_JOINTS: int = 1
NUM_ACTIVE_JOINTS: int = 13         # 6 + 6 + 1
DT: float = 0.002                   # 500 Hz control
GRAVITY: float = 9.81

# ---- Joint group indices (into actuated joint vector) ----
LEFT_LEG_INDICES: np.ndarray        # [0, 1, 2, 3, 4, 5]
RIGHT_LEG_INDICES: np.ndarray       # [6, 7, 8, 9, 10, 11]
WAIST_INDEX: int                    # 12

# ---- Nominal configuration ----
Q_HOME: np.ndarray                  # standing pose (nq,)

# ---- Model loading ----
def load_mujoco_model(scene_path: Path | None = None) -> tuple:
    """Load MuJoCo model and data. Returns (MjModel, MjData)."""
    ...

def load_pinocchio_model(urdf_path: Path | None = None) -> tuple:
    """Load G1 URDF with free-flyer root. Returns (pin.Model, pin.Data)."""
    ...

# ---- State conversion ----
def get_state_from_mujoco(mj_model, mj_data) -> tuple[np.ndarray, np.ndarray]:
    """Read MuJoCo state, convert to Pinocchio convention.
    Returns (q_pin, v_pin) with quaternion in (x,y,z,w) order.
    """
    ...

# ---- Quaternion conversion ----
def pin_quat_to_mj(quat_xyzw: np.ndarray) -> np.ndarray: ...
def mj_quat_to_pin(quat_wxyz: np.ndarray) -> np.ndarray: ...
```

### g1_model.py

```python
import numpy as np
import pinocchio as pin
from pathlib import Path

class G1Model:
    """Unitree G1 simplified kinematics and dynamics via Pinocchio.

    Wraps a Pinocchio model with free-flyer root joint. Arms are locked;
    only legs and waist are actuated (~13 DOF).
    """

    def __init__(self, urdf_path: Path, mesh_dir: Path | None = None) -> None:
        """Load URDF with pin.JointModelFreeFlyer root."""
        ...

    @property
    def nq(self) -> int: ...            # configuration dimension
    @property
    def nv(self) -> int: ...            # velocity dimension
    @property
    def na(self) -> int: ...            # actuated DOFs = nv - 6

    # ---- Forward kinematics ----
    def compute_com(self, q: np.ndarray) -> np.ndarray:
        """(3,) CoM position in world frame."""
        ...

    def compute_com_jacobian(self, q: np.ndarray) -> np.ndarray:
        """(3 x nv) CoM Jacobian."""
        ...

    def fk_foot(self, q: np.ndarray, side: str) -> pin.SE3:
        """FK for left or right foot frame. side in ('left', 'right')."""
        ...

    def foot_jacobian(self, q: np.ndarray, side: str) -> np.ndarray:
        """(6 x nv) foot Jacobian in LOCAL_WORLD_ALIGNED frame."""
        ...

    # ---- Dynamics ----
    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """(nv,) gravity component of generalized forces."""
        ...

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """(nv x nv) mass matrix via CRBA."""
        ...

    def nle(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """(nv,) nonlinear effects (Coriolis + gravity) via RNEA."""
        ...

    # ---- Inverse kinematics ----
    def whole_body_ik(
        self, com_target: np.ndarray,
        lfoot_target: pin.SE3, rfoot_target: pin.SE3,
        q_ref: np.ndarray, max_iter: int = 50,
        damping: float = 1e-3,
    ) -> np.ndarray | None:
        """Damped least-squares IK for CoM + both feet.
        Returns joint configuration or None if failed.
        """
        ...

    # ---- Joint mapping ----
    def build_joint_map(self, mj_model) -> dict[str, int]:
        """Build Pinocchio-to-MuJoCo joint index mapping."""
        ...
```

### balance_controller.py

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class BalanceGains:
    """PD gains for standing balance."""
    Kp_com: np.ndarray          # (3,) CoM position stiffness
    Kd_com: np.ndarray          # (3,) CoM velocity damping
    Kp_posture: float = 10.0    # posture regulation
    Kd_posture: float = 3.0

class StandingBalanceController:
    """Standing balance using CoM PD + gravity compensation.

    Computes desired joint positions/torques to keep CoM above
    the center of the support polygon.
    """

    def __init__(self, g1: "G1Model", gains: BalanceGains) -> None: ...

    def compute_torques(
        self, q: np.ndarray, qd: np.ndarray,
        com_desired: np.ndarray | None = None,
    ) -> np.ndarray:
        """(na,) actuated joint torques for standing balance.

        tau = J_com^T * F_com + g(q) + posture_regulation
        """
        ...

    def compute_support_polygon(self, q: np.ndarray) -> np.ndarray:
        """(N, 2) vertices of support polygon in XY plane."""
        ...

    def is_com_inside_support(self, q: np.ndarray) -> bool:
        """Check if CoM XY projection is inside support polygon."""
        ...
```

### lipm_planner.py

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class LIPMParams:
    """Parameters for Linear Inverted Pendulum Model."""
    z_c: float = 0.75          # constant CoM height (meters)
    gravity: float = 9.81

class LIPMPlanner:
    """CoM trajectory planning using LIPM + preview control.

    Implements Kajita (2003) preview control: given a ZMP reference
    trajectory, computes a smooth CoM trajectory that tracks the ZMP.
    """

    def __init__(self, params: LIPMParams, dt: float = 0.002) -> None: ...

    def plan_com_trajectory(
        self, zmp_ref: np.ndarray, N_preview: int = 160,
    ) -> np.ndarray:
        """Compute CoM trajectory from ZMP reference using preview control.

        Args:
            zmp_ref: (T, 2) ZMP reference positions (X, Y).
            N_preview: preview horizon length.

        Returns:
            (T, 2) CoM trajectory (X, Y).
        """
        ...

    def compute_preview_gains(self, N: int) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute preview controller gains using discrete Riccati equation.

        Returns: (Gi, Gx, Gd) — integral, state, preview gains.
        """
        ...

    @property
    def omega(self) -> float:
        """Natural frequency: sqrt(g / z_c)."""
        ...
```

### footstep_planner.py

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class GaitParams:
    """Walking gait parameters."""
    stride_length: float = 0.15     # meters
    step_width: float = 0.18        # meters (lateral distance between feet)
    step_height: float = 0.04       # meters (swing foot lift)
    step_duration: float = 0.6      # seconds per step
    double_support_ratio: float = 0.2  # fraction of step in double support

class FootstepPlanner:
    """Plans footstep sequence, ZMP reference, and foot trajectories."""

    def __init__(self, params: GaitParams, dt: float = 0.002) -> None: ...

    def plan_footsteps(self, n_steps: int) -> list[dict]:
        """Generate alternating left-right footstep placements.

        Returns list of dicts with keys: position (3,), side ('left'/'right'),
        timing (start_time, end_time).
        """
        ...

    def generate_zmp_reference(self, footsteps: list[dict]) -> np.ndarray:
        """(T, 2) ZMP reference from footstep sequence.

        Double support: linear transition between feet.
        Single support: ZMP at stance foot center.
        """
        ...

    def generate_foot_trajectory(
        self, start_pos: np.ndarray, end_pos: np.ndarray,
        height: float, duration: float,
    ) -> np.ndarray:
        """(N, 3) swing foot trajectory using quintic polynomial.

        Zero velocity at start and end. Parabolic height profile.
        """
        ...

    def generate_contact_schedule(self, footsteps: list[dict]) -> np.ndarray:
        """(T, 2) contact schedule: [left_in_contact, right_in_contact] per timestep."""
        ...

    def compute_support_polygon(
        self, lfoot_pos: np.ndarray, rfoot_pos: np.ndarray,
        phase: str,
    ) -> np.ndarray:
        """(N, 2) vertices of support polygon for given phase.

        phase: 'double', 'left_stance', 'right_stance'
        """
        ...
```

### walking_controller.py

```python
import numpy as np

class WalkingController:
    """Whole-body walking controller.

    Integrates footstep planner, LIPM planner, and whole-body IK to produce
    joint position targets for the MuJoCo simulation.
    """

    def __init__(
        self, g1: "G1Model",
        balance: "StandingBalanceController",
        lipm: "LIPMPlanner",
        footstep_planner: "FootstepPlanner",
    ) -> None: ...

    def plan_walk(self, n_steps: int) -> None:
        """Pre-plan the full walking trajectory: footsteps, CoM, foot trajectories."""
        ...

    def get_targets(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get (com_target, lfoot_target, rfoot_target) at time t.

        Returns: (com_pos (3,), lfoot SE3, rfoot SE3)
        """
        ...

    def compute_joint_targets(
        self, t: float, q_current: np.ndarray,
    ) -> np.ndarray:
        """Compute desired joint positions at time t via IK.

        Returns: (na,) desired actuated joint positions.
        """
        ...

    def step(
        self, q: np.ndarray, qd: np.ndarray, t: float,
    ) -> np.ndarray:
        """Full control step: IK → feedforward → ctrl.

        Returns: (na,) actuator commands for MuJoCo.
        """
        ...
```

## Model Files

### MuJoCo MJCF

| File | Description | Source |
|------|-------------|--------|
| `g1_humanoid.xml` | Simplified G1 MJCF (~23 DOF: 6/leg + waist, arms locked) | Built from MuJoCo Menagerie G1 with arms locked via equality constraints |
| `scene_flat.xml` | Scene: flat ground + G1 + lighting + tracking camera | New, includes g1_humanoid.xml |

The `g1_humanoid.xml` includes:
- Floating base body (pelvis/torso)
- Left and right legs: hip (yaw, roll, pitch), knee, ankle (pitch, roll)
- Waist joint (1 DOF)
- Arms locked via equality constraints or fixed joints
- Position-controlled actuators with PD servo gains (Menagerie-style general actuators)
- Collision geometries for feet and ground contact

### Pinocchio URDF

| File | Description | Source |
|------|-------------|--------|
| `g1_humanoid.urdf` | G1 URDF with free-flyer root for Pinocchio | Converted from MJCF or from Unitree official |

Loaded with `JointModelFreeFlyer` root joint. Floating base quaternion: Pinocchio uses (x,y,z,w), MuJoCo uses (w,x,y,z) — explicit conversion required.

## Dependencies on Previous Labs

### Lab 3 (Dynamics & Force Control)
- **Pattern:** PD control + gravity compensation, torque clipping
- **Pattern:** `lab3_common.py` structure for common module
- **Pattern:** MuJoCo Menagerie position servo feedforward: `ctrl = q_des + qfrc_bias/Kp + Kd*qd_des/Kp`
- **Import path:** Reimplemented in `lab7_common.py` and `balance_controller.py`

### Cross-validation pattern (all previous labs)
- Pinocchio FK vs MuJoCo body positions
- Always compare at startup and log any discrepancies

### Known Issues (from CLAUDE.md)
- Pinocchio quaternion (x,y,z,w) vs MuJoCo (w,x,y,z) — explicit conversion
- Menagerie position servos have gravity droop — feedforward compensation required
- Frame/body ID mapping differences between Pinocchio and MuJoCo

## Key Design Decisions

1. **Position servo with feedforward, not direct torque control.** The Menagerie G1 model uses position-controlled actuators. Following Lab 3's lesson, we use feedforward compensation: `ctrl = q_des + qfrc_bias/Kp + Kd*qd_des/Kp` for accurate tracking.

2. **Offline trajectory planning.** The entire walk (footsteps, CoM, foot trajectories) is planned before execution. This simplifies the control loop to just IK + tracking. Real-time replanning is not needed for flat ground walking.

3. **Arms locked during locomotion.** Reduces DOF count from ~37 to ~13, making IK faster and more reliable. Arms are unlocked in Lab 8 when manipulation is needed.

4. **LIPM preview control (Kajita 2003).** Standard approach for humanoid walking. Preview control provides smooth CoM trajectories that track the ZMP reference with bounded lag. Well-understood and well-tested.

5. **Damped least-squares IK.** More robust than pseudoinverse near singularities. Damping parameter tuned to balance tracking accuracy vs stability.

6. **500 Hz control loop.** Matches Lab 3's DT=0.002. Sufficient for stable contact dynamics in MuJoCo.
