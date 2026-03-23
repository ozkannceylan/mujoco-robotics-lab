# Lab 8: Whole-Body Loco-Manipulation — Architecture

## Module Map

```
lab-8-whole-body/
├── src/
│   ├── lab8_common.py              # Paths, constants, model loaders, state conversion
│   ├── g1_model.py                 # G1WholeBodyModel: FK, Jacobians, CoM, dynamics
│   ├── tasks.py                    # Task definitions: CoMTask, FootPoseTask, HandPoseTask, PostureTask
│   ├── contact_model.py            # ContactState, ContactSchedule, friction cone
│   ├── whole_body_qp.py            # WholeBodyQP: OSQP-based task-priority controller
│   ├── balance_controller.py       # StandingBalanceController (Phase 1 baseline)
│   ├── gait_generator.py           # GaitGenerator: footstep plan → CoM + foot trajectories
│   ├── loco_manip_fsm.py           # LocoManipStateMachine: full task sequencing
│   ├── a1_standing_reach.py        # Phase 2 demo: stand + reach with QP
│   ├── a2_walk_fixed_arms.py       # Phase 3 demo: walk with fixed arm pose
│   ├── a3_walk_and_reach.py        # Phase 3 demo: walk while reaching
│   ├── b1_loco_manip_pipeline.py   # Phase 4 demo: full loco-manipulation sequence
│   └── capstone_demo.py            # Phase 6: full demo with metrics and video
│
├── models/
│   ├── g1_full.xml                 # G1 MJCF from MuJoCo Menagerie (arms unlocked, ~37 DOF)
│   ├── scene_loco_manip.xml        # Full scene: G1 + ground + table + object + weld
│   └── g1_full.urdf                # G1 URDF for Pinocchio (with floating base)
│
├── tests/
│   ├── test_g1_model.py            # FK cross-validation, Jacobian, CoM, mass matrix
│   ├── test_tasks.py               # Task error/Jacobian dimensions, numerical verification
│   ├── test_qp.py                  # QP constraint satisfaction, torque limits, priority
│   ├── test_contact.py             # Contact schedule, friction cone geometry
│   ├── test_balance.py             # Standing stability, perturbation recovery
│   ├── test_gait.py                # CoM/foot trajectories, contact schedule
│   ├── test_walking.py             # Walk stability, arm drift tolerance
│   ├── test_fsm.py                 # State machine transitions, task activation
│   ├── test_com_compensation.py    # CoM with external load
│   └── test_pipeline.py            # End-to-end pipeline
│
├── docs/
│   ├── 01_whole_body_qp.md
│   ├── 02_walking_manipulation.md
│   └── 03_loco_manip_pipeline.md
│
├── docs-turkish/
│   ├── 01_tum_vucut_qp.md
│   ├── 02_yuruyus_manipulasyon.md
│   └── 03_loko_manipulasyon_boru_hatti.md
│
├── media/                          # Videos, plots, GIFs
├── tasks/                          # PLAN, ARCHITECTURE, TODO, LESSONS
└── README.md
```


## Data Flow

```
┌──────────────────────────────────────────────────┐
│           LocoManipStateMachine                    │
│  (loco_manip_fsm.py)                               │
│                                                    │
│  States: IDLE → WALK_TO_TABLE → STABILIZE →        │
│          REACH → GRASP → LIFT → WALK_WITH_OBJ →   │
│          STABILIZE_2 → PLACE → RELEASE → DONE      │
│                                                    │
│  Outputs: active_tasks[], contact_schedule,        │
│           gait_active, grasp_active                │
└────────────────────┬─────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌──────────────────────┐
│  GaitGenerator   │   │  Task Definitions    │
│  (gait_gen.py)   │   │  (tasks.py)          │
│                  │   │                      │
│  footsteps →     │   │  CoMTask             │
│  com_traj(t)     │   │  FootPoseTask (x2)   │
│  foot_traj(t)    │   │  HandPoseTask (x2)   │
│  contact_sched   │   │  PostureTask         │
└────────┬─────────┘   └──────────┬───────────┘
         │                        │
         └────────┬───────────────┘
                  ▼
┌──────────────────────────────────────────────────┐
│              WholeBodyQP                          │
│  (whole_body_qp.py)                               │
│                                                    │
│  min  Σ w_i ‖J_i·q̈ - (ẍ_d_i - J̇_i·v)‖²       │
│  s.t. M·q̈ + h = S^T·τ + J_c^T·f_c              │
│       friction cone (linearized)                  │
│       torque limits                               │
│       acceleration limits                         │
│                                                    │
│  Decision vars: q̈ (nv), τ (na), f_c (nc*3)      │
│                                                    │
│  Solver: OSQP                                     │
└────────────────────┬─────────────────────────────┘
                     │ τ (actuated joint torques)
                     ▼
┌──────────────────────────────────────────────────┐
│              G1WholeBodyModel                     │
│  (g1_model.py)                                    │
│                                                    │
│  Pinocchio: FK, Jacobians, CoM, M(q), h(q,v)     │
│  Joint mapping: Pinocchio ↔ MuJoCo               │
│  CoM with load compensation                       │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│              MuJoCo Simulation                    │
│                                                    │
│  mj_data.ctrl[actuator_ids] = τ                   │
│  mujoco.mj_step(mj_model, mj_data)               │
│  Read back: qpos, qvel, contact forces, xpos      │
└──────────────────────────────────────────────────┘
```

### Data flow summary:
1. **LocoManipStateMachine** determines the current phase, activates/deactivates tasks, starts/stops the gait generator, and manages grasp state (weld constraint).
2. **GaitGenerator** produces time-varying CoM and foot targets during walking phases. During standing phases, targets are static.
3. **Task definitions** compute errors and Jacobians for each objective using **G1WholeBodyModel**.
4. **WholeBodyQP** assembles all active tasks into a QP, solves for joint accelerations and torques using OSQP, respecting dynamics, contact forces, and joint limits.
5. Torques are sent to **MuJoCo** actuators. MuJoCo steps the physics. Joint state is read back for the next control cycle.
6. **G1WholeBodyModel** provides all analytical computations via Pinocchio. MuJoCo is never used for analytical computation.


## Key Interfaces

### lab8_common.py

```python
from pathlib import Path
import numpy as np

# ---- Paths ----
LAB_DIR: Path
MODELS_DIR: Path
MEDIA_DIR: Path
SCENE_LOCO_MANIP_PATH: Path
G1_URDF_PATH: Path

# ---- Constants ----
DT: float = 0.002                  # 500 Hz control
GRAVITY: np.ndarray                # (3,) [0, 0, -9.81]
NUM_ACTUATED: int                  # ~37 actuated joints
NUM_V: int                         # nv = 6 + NUM_ACTUATED
NUM_Q: int                         # nq = 7 + NUM_ACTUATED

# ---- Joint group indices ----
LEG_LEFT_INDICES: np.ndarray
LEG_RIGHT_INDICES: np.ndarray
WAIST_INDICES: np.ndarray
ARM_LEFT_INDICES: np.ndarray
ARM_RIGHT_INDICES: np.ndarray

# ---- Nominal configuration ----
Q_STAND: np.ndarray                # (nq,) nominal standing config
TORQUE_LIMITS: np.ndarray          # (NUM_ACTUATED,) per-joint

# ---- Model loading ----
def load_mujoco_model() -> tuple: ...
def load_pinocchio_model() -> "G1WholeBodyModel": ...
def get_state_from_mujoco(mj_model, mj_data) -> tuple[np.ndarray, np.ndarray]: ...
def pin_quat_to_mj(quat_xyzw: np.ndarray) -> np.ndarray: ...
def mj_quat_to_pin(quat_wxyz: np.ndarray) -> np.ndarray: ...
def clip_torques(tau: np.ndarray, limits: np.ndarray | None = None) -> np.ndarray: ...
```

### g1_model.py

```python
class G1WholeBodyModel:
    """Unitree G1 whole-body kinematics and dynamics via Pinocchio.
    Free-flyer root joint. Full body including arms.
    """

    def __init__(self, urdf_path: Path) -> None: ...

    @property
    def nq(self) -> int: ...
    @property
    def nv(self) -> int: ...
    @property
    def na(self) -> int: ...

    def fk_frame(self, q: np.ndarray, frame_name: str) -> pin.SE3: ...
    def jacobian_frame(self, q: np.ndarray, frame_name: str) -> np.ndarray: ...
    def jacobian_com(self, q: np.ndarray) -> np.ndarray: ...
    def compute_com(self, q: np.ndarray) -> np.ndarray: ...
    def compute_com_with_load(self, q: np.ndarray, load_mass: float, load_position: np.ndarray) -> np.ndarray: ...
    def mass_matrix(self, q: np.ndarray) -> np.ndarray: ...
    def nle(self, q: np.ndarray, v: np.ndarray) -> np.ndarray: ...
    def selection_matrix(self) -> np.ndarray: ...
    def build_joint_map(self, mj_model) -> dict[str, int]: ...
```

### tasks.py

```python
from abc import ABC, abstractmethod

class Task(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def dim(self) -> int: ...
    @abstractmethod
    def compute_error(self, q: np.ndarray, v: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def compute_jacobian(self, q: np.ndarray) -> np.ndarray: ...
    def compute_desired_acceleration(self, q: np.ndarray, v: np.ndarray) -> np.ndarray: ...

class CoMTask(Task):        # 3D, weight=1000, kp=100, kd=20
class FootPoseTask(Task):   # 6D SE3, weight=100, kp=200, kd=30
class HandPoseTask(Task):   # 6D SE3, weight=10, kp=100, kd=20
class PostureTask(Task):    # na-D, weight=1, kp=10, kd=3
```

### whole_body_qp.py

```python
@dataclass
class QPResult:
    qdd: np.ndarray          # (nv,)
    tau: np.ndarray           # (na,)
    f_contacts: list[np.ndarray]
    solve_time_ms: float
    status: str

class WholeBodyQP:
    """Task-priority whole-body QP controller using OSQP."""
    def __init__(self, model: G1WholeBodyModel, torque_limits: np.ndarray, friction_mu: float = 0.7) -> None: ...
    def solve(self, q: np.ndarray, v: np.ndarray, tasks: list[Task], contacts: list[ContactInfo]) -> QPResult: ...
```

### loco_manip_fsm.py

```python
class LocoManipState(Enum):
    IDLE, WALK_TO_TABLE, STOP_AND_STABILIZE, REACH, GRASP, LIFT,
    WALK_WITH_OBJECT, STOP_AND_STABILIZE_2, PLACE, RELEASE, RETREAT, DONE

class LocoManipStateMachine:
    def __init__(self, config, model, qp, gait_gen, mj_model, mj_data) -> None: ...
    def step(self, q: np.ndarray, v: np.ndarray, t: float) -> np.ndarray: ...
    def is_done(self) -> bool: ...
```


## Model Files

| File | Description | Source |
|------|-------------|--------|
| `g1_full.xml` | G1 MJCF with arms unlocked (~37 DOF) | MuJoCo Menagerie / adapted from Lab 7 |
| `scene_loco_manip.xml` | G1 + ground + table + object + weld constraint | New |
| `g1_full.urdf` | G1 URDF for Pinocchio | Converted from MJCF |


## Dependencies on Previous Labs

| Lab | Pattern reused |
|-----|---------------|
| Lab 3 | Dynamics: M(q), h(q,v), gravity compensation, impedance |
| Lab 5 | GraspStateMachine: weld constraint, state transitions |
| Lab 6 | Coordinated state machine: multi-phase task sequencing |
| Lab 7 | Locomotion: LIPM, ZMP, gait generation, standing balance |

All patterns reimplemented within Lab 8 (no cross-lab imports).


## Key Design Decisions

1. **Weighted QP over strict HQP.** Single OSQP solve with large priority gaps (1000:100:10:1) instead of cascaded QPs. Simpler and fast enough for 500 Hz.
2. **OSQP solver.** Open source, Python bindings, warm-starting, sparse. Alternative: scipy.optimize as fallback.
3. **Explicit contact forces as decision variables.** Enables friction cone constraints for physically consistent solutions.
4. **CoM compensation for carried objects.** Combined CoM target adjusted when object is grasped.
5. **Phase 1 baseline controller before QP.** Ensures model/state conversion work before adding QP complexity.
6. **Simplified grasp (weld constraint).** Focus is on whole-body coordination, not grasp mechanics.
7. **500 Hz control rate.** Sufficient for QP solve time (~1-2 ms) while maintaining stable contact dynamics.
