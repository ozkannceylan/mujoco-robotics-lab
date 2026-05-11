# Lab 5: Grasping & Manipulation — Architecture

## Module Map

```
lab-5-grasping-manipulation/
├── models/
│   ├── ur5e_gripper.xml     — UR5e arm + parallel-jaw gripper (MJCF)
│   └── scene_grasp.xml      — Full pick-and-place scene (includes ur5e_gripper.xml)
│
└── src/
    ├── lab5_common.py        — Paths, constants, model loaders, R_TOPDOWN
    ├── gripper_controller.py — Open/close gripper, contact detection
    ├── grasp_planner.py      — DLS IK, grasp config computation, collision-free check
    ├── grasp_state_machine.py — State machine tying Lab 3+4 modules together
    └── pick_place_demo.py    — Capstone demo: full pick-and-place, plots, logging
```

## Data Flow

```
pick_place_demo.py
    │
    ▼
GraspStateMachine.run(box_start, box_target)
    │
    ├── grasp_planner.compute_grasp_configs()
    │       │── DLS IK (Pinocchio FK + Jacobian)
    │       └── returns {q_pregrasp, q_grasp, q_preplace, q_place}
    │
    ├── PLAN_* states ──► Lab 4: CollisionChecker + RRTStarPlanner + trajectory_smoother
    │                          └── returns (times, q_traj, qd_traj)
    │
    ├── EXEC_* states ──► joint-space impedance: Kp*(q_d-q) + Kd*(0-qd) + g(q)
    │                     ctrl[:6] = tau_arm
    │
    ├── DESCEND/LIFT  ──► Lab 3: compute_impedance_torque (Cartesian, 6D)
    │                     ctrl[:6] = tau_cartesian
    │
    └── CLOSE/RELEASE ──► gripper_controller.close/open_gripper()
                          ctrl[6] = 0.0 / 0.030
```

## Key Interfaces

### lab5_common.py
```python
GRIPPER_TIP_OFFSET: float = 0.090        # meters from tool0 to fingertip center
BOX_A_POS: np.ndarray                    # grasp position in world frame
BOX_B_POS: np.ndarray                    # place position in world frame
R_TOPDOWN: np.ndarray                    # 3x3 rotation — EE Z points world -Z

def load_mujoco_model(scene_path=None) -> tuple[mj.MjModel, mj.MjData]
def load_pinocchio_model(urdf_path=None) -> tuple[model, data, ee_fid]
```

### gripper_controller.py
```python
def open_gripper(mj_data, open_pos=0.030) -> None
def close_gripper(mj_data) -> None
def is_gripper_settled(mj_model, mj_data, vel_thresh=0.001) -> bool
def is_gripper_in_contact(mj_model, mj_data) -> bool
```

### grasp_planner.py
```python
@dataclass
class GraspConfigs:
    q_home: np.ndarray
    q_pregrasp: np.ndarray    # tool0 15cm above grasp level
    q_grasp: np.ndarray       # tool0 at fingertip = box center
    q_preplace: np.ndarray
    q_place: np.ndarray

def compute_ik(
    pin_model, pin_data, ee_fid: int,
    x_target: np.ndarray, R_target: np.ndarray,
    q_init: np.ndarray, max_iter=200, tol=1e-4
) -> np.ndarray | None

def compute_grasp_configs(
    pin_model, pin_data, ee_fid: int,
    box_a_pos: np.ndarray, box_b_pos: np.ndarray
) -> GraspConfigs
```

### grasp_state_machine.py
```python
class GraspStateMachine:
    def __init__(self, mj_model, mj_data, pin_model, pin_data, ee_fid, cc, grasp_cfgs)
    def run(self) -> dict   # returns log: time, q, ee_pos, gripper_pos, state
```

## Model Files

### ur5e_gripper.xml
- Arm kinematic chain identical to Lab 3 `ur5e.xml` (6 torque motors)
- Gripper added as children of `tool0`:
  - `gripper_base` body (adapter plate)
  - `left_finger` / `right_finger` bodies with slide joints (axis Y / -Y)
  - Equality constraint: `right_finger_joint = left_finger_joint` (mirror symmetry)
  - Position actuator `gripper` (ctrl index 6) on `left_finger_joint`, kp=200
- `gripper_site`: site at fingertip center for IK reference

### scene_grasp.xml
- Includes `ur5e_gripper.xml`
- Table: box body at x=0.45, y=0.0, z=0.3, half-extents [0.35, 0.45, 0.015]
- `grasp_box`: free body (freejoint) at BOX_A_POS; box geom 0.040×0.040×0.040
- `target_pad`: visual-only body marking BOX_B_POS on table surface

## Dependencies on Previous Labs

| Feature needed | Source | How imported |
|---|---|---|
| Cartesian impedance controller | Lab 3 `b1_impedance_controller.py` | `sys.path` + direct import |
| CollisionChecker | Lab 4 `collision_checker.py` | `sys.path` + direct import |
| RRTStarPlanner | Lab 4 `rrt_planner.py` | `sys.path` + direct import |
| shortcut_path + parameterize_topp_ra | Lab 4 `trajectory_smoother.py` | `sys.path` + direct import |
| UR5e URDF (Pinocchio) | Lab 3 `models/ur5e.urdf` | absolute Path reference |

## Contact Physics Design

| Parameter | Value | Rationale |
|---|---|---|
| condim | 4 | Elliptic friction cone — 2D tangential friction |
| friction | 1.5 0.005 0.0001 | High tangential (grip), realistic torsional |
| solimp | 0.99 0.99 0.001 | Stiff contacts; 1mm penetration depth |
| solref | 0.002 1 | Fast contact resolution (2ms time constant) |
| Box mass | 0.15 kg | Light enough for gripper force; heavy enough to not fly off |

## Actuator Layout

| Index | Name | Type | Joint | Range |
|---|---|---|---|---|
| 0–5 | motor1–6 | torque | shoulder_pan … wrist_3 | ±180 Nm |
| 6 | gripper | position | left_finger_joint | 0–0.030 m |

ctrl[0:6] = arm torques (set by impedance / joint-space controllers)
ctrl[6] = gripper position setpoint (0.0 = closed, 0.030 = open)
