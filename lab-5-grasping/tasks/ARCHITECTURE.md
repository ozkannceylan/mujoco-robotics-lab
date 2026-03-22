# Lab 5: Grasping & Manipulation — Architecture

## Module Map

```
lab-5-grasping/
├── src/
│   ├── lab5_common.py          # Paths, constants, model loaders, utilities
│   ├── gripper_controller.py   # Gripper open/close/detect grasp
│   ├── contact_tuning.py       # Contact parameter experiments
│   ├── grasp_planner.py        # Grasp pose computation + IK
│   ├── grasp_state_machine.py  # State machine for pick-and-place
│   ├── pick_and_place.py       # End-to-end pick-and-place demo
│   └── capstone_demo.py        # Multi-object capstone demonstration
├── models/
│   ├── ur5e_gripper.xml        # UR5e + parallel jaw gripper (included by scene)
│   └── scene_grasp.xml         # Full scene: robot + table + objects
├── tests/
│   ├── test_model_loading.py   # Model loading + FK cross-validation
│   ├── test_gripper.py         # Gripper open/close/detect
│   ├── test_contact.py         # Contact physics validation
│   └── test_pipeline.py        # Pick-and-place pipeline tests
├── docs/
│   ├── 01_gripper_integration.md
│   ├── 02_contact_physics.md
│   └── 03_pick_and_place.md
├── docs-turkish/
│   ├── 01_tutucu_entegrasyonu.md
│   ├── 02_temas_fizigi.md
│   └── 03_al_birak_boru_hatti.md
└── media/                      # Generated plots and figures
```

## Data Flow

```
User Input: pick_pos, place_pos
         │
         ▼
┌──────────────────────┐
│   grasp_planner.py   │  Pinocchio IK → joint configs for approach/grasp/place
│   compute_grasp_poses │
└──────────┬───────────┘
           │ GraspPlan (pre_grasp_q, grasp_q, lift_q, place_q, ...)
           ▼
┌──────────────────────────┐
│  grasp_state_machine.py  │  Orchestrates state transitions
│  GraspStateMachine       │
└──────────┬───────────────┘
           │ Per-state control commands
           ▼
┌──────────────────────────────────────┐
│  Per-state controllers               │
│                                      │
│  APPROACH: joint-space PD + gravity  │ ← Lab 4 trajectory_executor pattern
│  DESCEND:  Cartesian impedance       │ ← Lab 3 b1_impedance_controller
│  GRASP:    gripper close + wait      │ ← gripper_controller.py
│  LIFT:     Cartesian impedance +Z    │ ← Lab 3 b1_impedance_controller
│  TRANSPORT: joint-space PD (direct)  │ ← trajectory_executor pattern
│  PLACE:    Cartesian impedance -Z    │ ← Lab 3 b1_impedance_controller
│  RELEASE:  gripper open + retreat    │ ← gripper_controller.py
└──────────┬───────────────────────────┘
           │ torques + gripper ctrl
           ▼
┌──────────────────────────┐
│  MuJoCo simulation       │  Physics stepping, contact, rendering
│  mj_step()               │
└──────────────────────────┘
```

## Key Interfaces

### lab5_common.py
```python
# Constants
NUM_ARM_JOINTS: int = 6
GRIPPER_ACTUATOR_IDX: int = 6  # ctrl[6] = gripper position
DT: float = 0.001
Q_HOME: np.ndarray  # (6,) arm home config
GRIPPER_OPEN: float = 0.04   # max finger separation
GRIPPER_CLOSED: float = 0.0  # fingers touching

# Model loading
def load_mujoco_model(scene_path=None) -> tuple[MjModel, MjData]
def load_pinocchio_model(urdf_path=None) -> tuple[Model, Data, int]
def get_ee_pose(pin_model, pin_data, ee_fid, q) -> tuple[np.ndarray, np.ndarray]
def clip_torques(tau: np.ndarray) -> np.ndarray
```

### gripper_controller.py
```python
class GripperController:
    def open(mj_data) -> None
    def close(mj_data) -> None
    def set_width(mj_data, width: float) -> None
    def is_grasping(mj_model, mj_data, object_body_id: int) -> bool
    def get_grip_force(mj_model, mj_data) -> float
```

### grasp_planner.py
```python
@dataclass
class GraspPlan:
    pre_grasp_pos: np.ndarray   # (3,) above object
    grasp_pos: np.ndarray       # (3,) at object
    lift_pos: np.ndarray        # (3,) lifted
    place_pos: np.ndarray       # (3,) above target
    place_down_pos: np.ndarray  # (3,) at target surface
    approach_q: np.ndarray      # IK solution for pre-grasp
    grasp_q: np.ndarray         # IK solution for grasp
    R_grasp: np.ndarray         # Gripper orientation (top-down)

def compute_grasp_plan(
    pin_model, pin_data, ee_fid,
    object_pos: np.ndarray,
    target_pos: np.ndarray,
    q_init: np.ndarray,
    approach_height: float = 0.15,
    grasp_height_offset: float = 0.0,
) -> GraspPlan
```

### grasp_state_machine.py
```python
class GraspState(Enum):
    IDLE, APPROACH, DESCEND, GRASP, LIFT, TRANSPORT, PLACE, RELEASE, DONE

class GraspStateMachine:
    def __init__(self, plan: GraspPlan, ...)
    def step(self, mj_model, mj_data, pin_model, pin_data, ee_fid) -> tuple[np.ndarray, float]
        # Returns (arm_torques, gripper_ctrl)
    @property
    def state(self) -> GraspState
    @property
    def is_done(self) -> bool
```

## Model Files

### ur5e_gripper.xml
- Extends Lab 3's `ur5e.xml` base
- Adds parallel-jaw gripper at tool0:
  - Two finger bodies with prismatic joints (symmetric)
  - Single position actuator controlling both fingers
  - Finger geoms: box shapes with high friction
  - Contact: contype=3, conaffinity=3 (contacts with objects)

### scene_grasp.xml
- Includes `ur5e_gripper.xml`
- Table body (box) from Lab 3 pattern
- Graspable box object: free body with freejoint
- Contact parameters tuned for stable grasping

## Dependencies on Previous Labs

| Import | From | Used For |
|--------|------|----------|
| `compute_impedance_torque` | Lab 3 `b1_impedance_controller.py` | Cartesian impedance for descend/lift/place |
| `ImpedanceGains` | Lab 3 `b1_impedance_controller.py` | Gain configuration |
| `orientation_error` | Lab 3 `b1_impedance_controller.py` | 6D error computation |
| Pattern: joint-PD + gravity | Lab 4 `trajectory_executor.py` | Approach/transport motions |
| Pattern: `clip_torques` | Lab 3/4 common | Torque safety |

Note: We copy the impedance controller pattern rather than importing directly, to avoid
cross-lab path dependencies. The Lab 5 common module has its own model loaders pointing
to Lab 5's model files.
