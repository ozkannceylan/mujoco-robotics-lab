# Lab 6: Dual-Arm Coordination — Architecture

## Module Map

```
lab-6-dual-arm/
├── src/
│   ├── lab6_common.py              # Paths, constants, model loaders, index slicing
│   ├── dual_arm_model.py           # DualArmModel: FK, Jacobians, object-centric frame
│   ├── dual_collision_checker.py   # Arm-arm + arm-env collision checking (HPP-FCL)
│   ├── coordinated_planner.py      # Synchronized trajectory generation (3 modes)
│   ├── cooperative_controller.py   # Dual impedance controller with internal force
│   ├── bimanual_grasp.py           # Bimanual grasp state machine
│   ├── a1_independent_motion.py    # Phase 1 demo: independent arm motion
│   ├── a2_coordinated_approach.py  # Phase 2 demo: synchronized approach
│   ├── b1_cooperative_carry.py     # Phase 3 demo: grasp-lift-carry-place pipeline
│   └── capstone_demo.py            # Phase 5: full multi-scenario demo
│
├── models/
│   ├── ur5e_left.xml               # Left UR5e (prefixed names, base at origin)
│   ├── ur5e_right.xml              # Right UR5e (prefixed names, facing left arm)
│   ├── scene_dual.xml              # Full scene: both arms, table, box object
│   ├── ur5e_left.urdf              # Left arm URDF for Pinocchio (collision geoms)
│   └── ur5e_right.urdf             # Right arm URDF for Pinocchio (collision geoms)
│
├── tests/
│   ├── test_dual_model.py          # FK cross-validation, Jacobians, base transforms
│   ├── test_dual_collision.py      # Collision/free configs, min-distance checks
│   ├── test_coordinated_planner.py # Timing sync, master-slave, symmetric mode
│   ├── test_bimanual_grasp.py      # State transitions, preconditions
│   └── test_cooperative_controller.py  # Torque correctness, force balancing
│
├── docs/
│   └── lab6_dual_arm_coordination.md
├── docs-turkish/
│   └── lab6_cift_kol_koordinasyonu.md
├── media/                          # Videos, plots, GIFs
├── tasks/                          # PLAN, ARCHITECTURE, TODO, LESSONS
└── README.md
```

## Data Flow

```
                        ┌─────────────────────────┐
                        │    BimanualGraspSM       │
                        │  (bimanual_grasp.py)     │
                        │  States: IDLE->APPROACH  │
                        │  ->GRASP->LIFT->CARRY    │
                        │  ->PLACE->RELEASE->DONE  │
                        └─────────┬───────────────┘
                                  │ current state + targets
                                  ▼
┌─────────────────────┐   ┌───────────────────────┐   ┌─────────────────────────┐
│  DualArmModel       │──▶│  CoordinatedPlanner   │──▶│  SynchronizedTrajectory │
│  (dual_arm_model.py)│   │  (coordinated_        │   │  (q_left[T], q_right[T],│
│                     │   │   planner.py)          │   │   timestamps[T])        │
│  - fk_left(q)       │   │                       │   └────────────┬────────────┘
│  - fk_right(q)      │   │  - plan_synchronized  │                │
│  - jacobian_left(q)  │   │  - plan_master_slave  │                │
│  - jacobian_right(q) │   │  - plan_symmetric     │                ▼
│  - object_frame()   │   └───────────────────────┘   ┌─────────────────────────┐
└─────────┬───────────┘                               │  DualImpedanceController│
          │                                           │  (cooperative_           │
          │   ┌───────────────────────┐               │   controller.py)        │
          │   │  DualCollisionChecker │               │                         │
          └──▶│  (dual_collision_     │               │  - compute_dual_torques │
              │   checker.py)         │               │  - internal force term  │
              │                       │               │  - gravity compensation │
              │  - is_collision_free  │               └────────────┬────────────┘
              │  - get_min_distance   │                            │ (tau_left, tau_right)
              └───────────────────────┘                            ▼
                                                      ┌─────────────────────────┐
                                                      │  MuJoCo Simulation      │
                                                      │                         │
                                                      │  mj_data.ctrl[:6]  = L  │
                                                      │  mj_data.ctrl[6:12]= R  │
                                                      │  mujoco.mj_step()       │
                                                      └─────────────────────────┘
```

### Data flow summary:
1. **BimanualGraspSM** determines current phase and target poses (in object-centric frame).
2. **CoordinatedPlanner** uses **DualArmModel** FK/IK to produce time-synchronized joint trajectories for both arms.
3. At each timestep, **DualCollisionChecker** validates the next configuration is collision-free.
4. **DualImpedanceController** computes torques for both arms (impedance + gravity comp + internal force when grasping).
5. Torques are sent to **MuJoCo** as `ctrl` for the 12 actuators.
6. MuJoCo steps the physics; joint states are read back for the next control cycle.


## Key Interfaces

### lab6_common.py

```python
# Paths
LAB_DIR: Path
MODELS_DIR: Path
MEDIA_DIR: Path
SCENE_DUAL_PATH: Path
URDF_LEFT_PATH: Path
URDF_RIGHT_PATH: Path

# Constants
NUM_JOINTS_PER_ARM: int = 6
NUM_JOINTS_TOTAL: int = 12
DT: float = 0.001
LEFT_JOINT_SLICE: slice = slice(0, 6)
RIGHT_JOINT_SLICE: slice = slice(6, 12)
TORQUE_LIMITS: np.ndarray          # (6,) per arm
Q_HOME_LEFT: np.ndarray            # (6,)
Q_HOME_RIGHT: np.ndarray           # (6,) mirrored for facing arm

# Base transforms (SE3) for each arm in world frame
LEFT_BASE_SE3: pin.SE3             # identity or slight offset
RIGHT_BASE_SE3: pin.SE3            # translated + rotated 180 deg about Z

# Model loading
def load_mujoco_model(scene_path: Path | None = None) -> tuple[MjModel, MjData]: ...
def load_dual_pinocchio_models() -> tuple[DualArmModel]: ...

# Quaternion utilities (reused from Lab 3)
def mj_quat_to_pin(quat_wxyz: np.ndarray) -> np.ndarray: ...
def pin_quat_to_mj(quat_xyzw: np.ndarray) -> np.ndarray: ...
```

### dual_arm_model.py

```python
class ObjectFrame:
    """Object-centric frame for bimanual coordination."""
    pose: pin.SE3                                   # object pose in world frame
    grasp_offset_left: pin.SE3                      # left EE relative to object
    grasp_offset_right: pin.SE3                     # right EE relative to object

    def get_left_target(self) -> pin.SE3: ...       # pose * grasp_offset_left
    def get_right_target(self) -> pin.SE3: ...      # pose * grasp_offset_right
    def from_ee_poses(cls, left_ee: pin.SE3, right_ee: pin.SE3) -> "ObjectFrame": ...

class DualArmModel:
    """Dual UR5e kinematics using two Pinocchio models with base offsets."""

    def __init__(self, urdf_left: Path, urdf_right: Path,
                 base_left: pin.SE3, base_right: pin.SE3) -> None: ...

    # Forward kinematics (returns SE3 in world frame)
    def fk_left(self, q_left: np.ndarray) -> pin.SE3: ...
    def fk_right(self, q_right: np.ndarray) -> pin.SE3: ...

    # Jacobians in world frame (6x6)
    def jacobian_left(self, q_left: np.ndarray) -> np.ndarray: ...
    def jacobian_right(self, q_right: np.ndarray) -> np.ndarray: ...

    # Gravity torques
    def gravity_left(self, q_left: np.ndarray) -> np.ndarray: ...
    def gravity_right(self, q_right: np.ndarray) -> np.ndarray: ...

    # IK (numerical, from Lab 2 pattern)
    def ik_left(self, target: pin.SE3, q_init: np.ndarray) -> np.ndarray | None: ...
    def ik_right(self, target: pin.SE3, q_init: np.ndarray) -> np.ndarray | None: ...

    # Relative pose between end-effectors
    def relative_ee_pose(self, q_left: np.ndarray, q_right: np.ndarray) -> pin.SE3: ...
```

### dual_collision_checker.py

```python
class DualCollisionChecker:
    """Collision checking for dual UR5e setup.

    Checks: left self-collision, right self-collision, left-right cross,
    each arm vs. environment (table, objects).
    """

    def __init__(
        self,
        urdf_left: Path, urdf_right: Path,
        base_left: pin.SE3, base_right: pin.SE3,
        environment_specs: list[ObstacleSpec] | None = None,
    ) -> None: ...

    def is_collision_free(self, q_left: np.ndarray, q_right: np.ndarray) -> bool: ...
    def get_min_distance(self, q_left: np.ndarray, q_right: np.ndarray) -> float: ...
    def is_path_free(
        self, q_left_start: np.ndarray, q_left_end: np.ndarray,
        q_right_start: np.ndarray, q_right_end: np.ndarray,
        n_checks: int = 10,
    ) -> bool: ...
```

### coordinated_planner.py

```python
@dataclass
class SynchronizedTrajectory:
    """Time-synchronized joint trajectory for both arms."""
    timestamps: np.ndarray          # (T,)
    q_left: np.ndarray              # (T, 6)
    qd_left: np.ndarray             # (T, 6)
    q_right: np.ndarray             # (T, 6)
    qd_right: np.ndarray            # (T, 6)
    duration: float

class CoordinatedPlanner:
    """Generates synchronized trajectories for dual arms."""

    def __init__(self, dual_model: DualArmModel,
                 collision_checker: DualCollisionChecker) -> None: ...

    def plan_synchronized_linear(
        self,
        target_left: pin.SE3, target_right: pin.SE3,
        q_left_init: np.ndarray, q_right_init: np.ndarray,
        duration: float | None = None,
    ) -> SynchronizedTrajectory: ...

    def plan_master_slave(
        self,
        master_waypoints: list[pin.SE3],
        object_frame: ObjectFrame,
        q_left_init: np.ndarray, q_right_init: np.ndarray,
        master: str = "left",
        duration: float | None = None,
    ) -> SynchronizedTrajectory: ...

    def plan_symmetric(
        self,
        object_trajectory: list[pin.SE3],
        object_frame: ObjectFrame,
        q_left_init: np.ndarray, q_right_init: np.ndarray,
        duration: float | None = None,
    ) -> SynchronizedTrajectory: ...
```

### cooperative_controller.py

```python
@dataclass
class DualImpedanceGains:
    """Impedance gains for dual-arm control."""
    K_p: np.ndarray         # (6, 6) stiffness — same for both arms (symmetric)
    K_d: np.ndarray         # (6, 6) damping — same for both arms
    f_squeeze: float        # Internal squeeze force along grasp axis (N)

class DualImpedanceController:
    """Dual-arm impedance controller with internal force coupling."""

    def __init__(self, dual_model: DualArmModel,
                 gains: DualImpedanceGains | None = None) -> None: ...

    def compute_dual_torques(
        self,
        q_left: np.ndarray, qd_left: np.ndarray,
        q_right: np.ndarray, qd_right: np.ndarray,
        target_left: pin.SE3, target_right: pin.SE3,
        xd_left: np.ndarray | None = None,      # desired EE velocity (6,)
        xd_right: np.ndarray | None = None,
        grasping: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def set_gains(self, gains: DualImpedanceGains) -> None: ...
```

### bimanual_grasp.py

```python
from enum import Enum, auto

class BimanualState(Enum):
    IDLE = auto()
    APPROACH = auto()
    PRE_GRASP = auto()
    GRASP = auto()
    LIFT = auto()
    CARRY = auto()
    PLACE = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()

@dataclass
class BimanualGraspConfig:
    """Configuration for bimanual grasp task."""
    object_pose: pin.SE3               # initial object pose
    grasp_offset_left: pin.SE3         # left EE relative to object
    grasp_offset_right: pin.SE3        # right EE relative to object
    lift_height: float = 0.15          # meters
    carry_displacement: np.ndarray     # (3,) world-frame displacement
    place_pose: pin.SE3                # final object pose
    approach_clearance: float = 0.05   # meters offset before grasp
    position_tolerance: float = 0.005  # meters — when to transition states

class BimanualGraspStateMachine:
    """Orchestrates a bimanual pick-carry-place task."""

    def __init__(
        self,
        config: BimanualGraspConfig,
        dual_model: DualArmModel,
        planner: CoordinatedPlanner,
        controller: DualImpedanceController,
        mj_model, mj_data,
    ) -> None: ...

    @property
    def state(self) -> BimanualState: ...

    def step(
        self, q_left: np.ndarray, qd_left: np.ndarray,
        q_right: np.ndarray, qd_right: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]: ...
    """Returns (tau_left, tau_right) for current timestep."""

    def is_done(self) -> bool: ...

    def activate_weld_constraint(self) -> None: ...
    def deactivate_weld_constraint(self) -> None: ...
```


## Model Files

### MuJoCo MJCF

| File | Description | Source |
|------|-------------|--------|
| `ur5e_left.xml` | Left UR5e, all names prefixed `left_`, base at world origin | Adapted from Lab 3 `ur5e.xml` |
| `ur5e_right.xml` | Right UR5e, all names prefixed `right_`, base at (1.0, 0, 0) rotated 180 deg about Z | Adapted from Lab 3 `ur5e.xml` |
| `scene_dual.xml` | Full scene including both arms, table, large box, lighting, skybox | New, includes both arm XMLs |

The `scene_dual.xml` includes:
- Both arm XMLs via `<include>`
- Shared table (centered between arms)
- Large box object (free body, ~30x15x15 cm, ~2 kg) with contact properties
- Weld equality constraint (initially disabled) for rigid grasp
- Camera positioned to view both arms and workspace

### Pinocchio URDF

| File | Description | Source |
|------|-------------|--------|
| `ur5e_left.urdf` | Left UR5e with collision geometries | Adapted from Lab 4 `ur5e_collision.urdf` |
| `ur5e_right.urdf` | Right UR5e with collision geometries | Adapted from Lab 4 `ur5e_collision.urdf` |

Note: Pinocchio models are loaded with base SE3 transforms applied programmatically (using `pin.SE3` offsets), not baked into the URDF. The URDFs can share the same file if joint names don't need prefixing in Pinocchio (since each model is separate).


## Dependencies on Previous Labs

### Lab 3 (Dynamics & Force Control)
- **Pattern:** `ImpedanceGains` dataclass, `orientation_error()` function, impedance control loop structure
- **Reused concepts:** `tau = J^T * F + g(q)`, gravity compensation, torque clipping
- **Import path:** Not directly imported; patterns reimplemented in `cooperative_controller.py` with dual-arm extensions

### Lab 4 (Motion Planning)
- **Pattern:** `CollisionChecker` class structure (HPP-FCL geometry objects, collision pair registration)
- **Reused concepts:** `is_collision_free()`, `is_path_free()`, `get_min_distance()` interface
- **Pattern:** `shortcut_path()`, `parameterize_topp_ra()` for trajectory post-processing
- **Import path:** Patterns reimplemented in `dual_collision_checker.py` and `coordinated_planner.py`

### Lab 5 (Grasping & Manipulation)
- **Pattern:** `GraspStateMachine` state machine pattern (enum states, step() method, state preconditions)
- **Reused concepts:** Weld constraint activation/deactivation for rigid grasp
- **Import path:** Pattern reimplemented in `bimanual_grasp.py`

### Shared utilities
- **lab3_common.py patterns:** Path layout, model loading, quaternion conversion, torque clipping
- **All reimplemented in lab6_common.py** with dual-arm extensions (no cross-lab imports to keep labs self-contained)


## Key Design Decisions

1. **Two separate Pinocchio models** (not one combined model): Each arm is a standard 6-DOF chain. Combining into a single 12-DOF model would require a custom URDF with a branching kinematic tree, adding complexity with no benefit. Two models with base transforms are simpler and reuse existing code.

2. **Object-centric frame abstraction**: All cooperative motion is planned relative to the object frame, not individual EE frames. This decouples task planning from arm-specific kinematics.

3. **Symmetric impedance for internal force**: Both arms use identical gains. This ensures that when both arms push toward the object with equal stiffness, internal forces balance naturally. No explicit force control loop needed — the impedance framework handles it.

4. **Weld constraint for rigid grasp**: MuJoCo's equality constraint (weld) simplifies grasp simulation. The constraint is disabled during approach and enabled at grasp time. This avoids needing to model friction-limited grasps, focusing the lab on coordination rather than grasp mechanics.

5. **Self-contained lab**: All Lab 3/4/5 patterns are reimplemented within Lab 6, not imported. This keeps each lab independently runnable and avoids cross-lab dependency issues.
