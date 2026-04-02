# Lab 6: Dual-Arm Coordination -- Architecture Document

This document is the technical reference for Lab 6 of the MuJoCo Robotics Lab
portfolio project. It covers the full dual-arm cooperative manipulation system:
two UR5e arms that approach a box from opposite sides, grasp it with weld
constraints, lift it, carry it laterally, and place it at a new location.

The system uses Pinocchio for all analytical computation (FK, Jacobian, IK) and
MuJoCo for physics simulation, rendering, and contact dynamics. No computation
is duplicated between the two -- Pinocchio is the brain, MuJoCo is the body.

---

## Table of Contents

1. [System Overview Diagram](#1-system-overview-diagram)
2. [Module-by-Module Breakdown](#2-module-by-module-breakdown)
3. [Controller Design Deep-Dive](#3-controller-design-deep-dive)
4. [IK Pipeline](#4-ik-pipeline)
5. [Grasp Pose Computation](#5-grasp-pose-computation)
6. [State Machine Walkthrough](#6-state-machine-walkthrough)
7. [Lessons Learned](#7-lessons-learned)

---

## 1. System Overview Diagram

### Module Dependency Graph

```
                        +-------------------------------+
                        |       scene_dual.xml          |
                        |  (MJCF: two UR5e arms, table, |
                        |   box, weld constraints)      |
                        +-------------------------------+
                          |                           |
                   includes                     includes
                          v                           v
                  +---------------+          +----------------+
                  | ur5e_left.xml |          | ur5e_right.xml |
                  | (base @ 0,0,0|          | (base @ 1,0,0) |
                  |  6 motors)    |          |  6 motors)     |
                  +---------------+          +----------------+
                          |                           |
                          +--------- loaded by -------+
                                       |
                                       v
                        +-------------------------------+
                        |       lab6_common.py          |
                        | Constants, paths, loaders,    |
                        | quaternion converters          |
                        +-------------------------------+
                          ^       ^       ^        ^
                          |       |       |        |
              +-----------+   +---+---+   |   +----+----------+
              |               |       |   |   |               |
  +-----------+--+   +--------+---+   |   |   |  +------------+------+
  | dual_arm_    |   | joint_pd_  |   |   |   |  | bimanual_state_   |
  | model.py     |   | controller |   |   |   |  | machine.py        |
  | (Pinocchio   |   | .py        |   |   |   |  | (6-state pipeline,|
  |  FK/J/IK)    |   | (PD+grav   |   |   |   |  |  weld mgmt,       |
  |              |   |  comp)     |   |   |   |  |  collision-free IK)|
  +-----------+--+   +--------+---+   |   |   |  +---+----------+----+
              |               |       |   |   |      |          |
              |               |       |   |   |      |          |
              |               |       |   |   |      |          |
              v               v       |   |   v      v          v
  +--------+  +--------+     |   +----+------+  +----------+
  |Pinocchio|  |MuJoCo  |     |   | grasp_   |  | ur5e.urdf|
  | (pin)   |  |mj_step |     |   | pose_    |  | (Lab 3   |
  | FK, J,  |  |mj_fwd  |     |   | calc.py  |  | Menagerie|
  | IK solve|  |ctrl[]   |     |   |          |  | match)   |
  +--------+  +--------+     |   +----------+  +----------+
                              |
                  +-----------+-----------+
                  | m0..m5 milestone       |
                  | scripts (runners)      |
                  +-----------+-----------+
```

### Data Flow: One Simulation Step

```
  +-----------------+     q_target_left      +-------------------+
  | State Machine   | ----q_target_right---> | DualArmJointPD    |
  | (decides what   |                        | .compute()        |
  |  targets to set |                        |                   |
  |  based on state)|     +--------+         | For each arm:     |
  +--------+--------+     |MuJoCo  | <-----> | tau = Kp*(qt-q)   |
           |              |mj_data |         |   + Kd*(0-qvel)   |
           |              |        |         |   + qfrc_bias     |
           |              | .qpos  |         | clip(tau, limits) |
           |              | .qvel  |         | -> mj_data.ctrl[] |
           |              | .ctrl  |         +-------------------+
           |              | .xpos  |
           |              | .xmat  |         +-------------------+
           |              | .ncon  |         | mujoco.mj_step()  |
           |              | .eq_*  |         | Advances physics   |
           |              +--------+         | 1 ms (DT=0.001)   |
           |                                 +-------------------+
           |
           |  (IK calls)
           v
  +-----------------+     target_pos (world)  +-------------------+
  | DualArmModel    | <----target_rot-------> | Pinocchio         |
  | .ik(arm, pos,   |                        | pin.forwardKinem. |
  |   rot)          |     q_solution         | pin.getFrameJac.  |
  |                 | ----------------------> | pin.log3()        |
  | base offsets:   |                        |                   |
  |  L=[0,0,0]     |                        | DLS solve loop    |
  |  R=[1,0,0]     |                        +-------------------+
  +-----------------+
```

### Import Graph (source-level)

```
lab6_common.py          <-- imported by everything
    |
    +--- dual_arm_model.py      imports: lab6_common, pinocchio, numpy
    |
    +--- joint_pd_controller.py imports: lab6_common, numpy
    |
    +--- grasp_pose_calculator.py imports: lab6_common, mujoco, pinocchio, numpy
    |
    +--- bimanual_state_machine.py imports: lab6_common, dual_arm_model,
    |                                       grasp_pose_calculator,
    |                                       joint_pd_controller,
    |                                       mujoco, pinocchio, numpy
    |
    +--- m0_validate_scene.py    imports: lab6_common, mujoco
    +--- m1_independent_motion.py imports: lab6_common, dual_arm_model,
    |                                      joint_pd_controller, mujoco
    +--- m2_fk_validation.py     imports: lab6_common, dual_arm_model, mujoco
    +--- m2_ik_validation.py     imports: lab6_common, dual_arm_model
    +--- m3_coordinated_approach.py imports: lab6_common, dual_arm_model,
    |                                        grasp_pose_calculator,
    |                                        joint_pd_controller, mujoco
    +--- m4_cooperative_carry.py imports: lab6_common, dual_arm_model,
    |                                     bimanual_state_machine,
    |                                     joint_pd_controller, mujoco
    +--- m5_capstone_demo.py     imports: (same as m4 + rendering)
```

### Cross-Lab Dependencies

Lab 6 depends on one artifact from Lab 3:

```
lab-3-dynamics-force-control/models/ur5e.urdf
    |
    | (copied to)
    v
lab-6-dual-arm/models/ur5e.urdf
```

This is the Menagerie-matching URDF with 180-degree Z in `world_joint` and
`shoulder_lift rpy="0 pi/2 0"`. It is NOT the standard DH-convention URDF from
`universal_robots_description`. See Lesson L4 for why this matters.

---

## 2. Module-by-Module Breakdown

### 2.1 `lab6_common.py` (115 lines)

**Purpose:** Central configuration hub for all Lab 6 code. Every source file
imports from here. It defines physical constants, file paths, array slicing
helpers, and utility functions that would otherwise be duplicated across modules.

**Key constants:**

```python
DT = 0.001                          # 1 kHz simulation timestep
NUM_JOINTS_PER_ARM = 6              # UR5e has 6 revolute joints
NUM_JOINTS_TOTAL = 12               # Two arms combined
LEFT_JOINT_SLICE = slice(0, 6)      # MuJoCo qpos/qvel indices for left arm
RIGHT_JOINT_SLICE = slice(6, 12)    # MuJoCo qpos/qvel indices for right arm
LEFT_CTRL_SLICE = slice(0, 6)       # MuJoCo ctrl indices for left actuators
RIGHT_CTRL_SLICE = slice(6, 12)     # MuJoCo ctrl indices for right actuators

TORQUE_LIMITS = [150, 150, 150, 28, 28, 28]  # Nm per joint

Q_HOME_LEFT = [-pi/2, -pi/2, pi/2, -pi/2, -pi/2, 0]
Q_HOME_RIGHT = Q_HOME_LEFT.copy()   # Identical -- no base yaw difference

TABLE_SURFACE_Z = 0.17              # Table top z-coordinate
BOX_HALF_EXTENTS = [0.15, 0.075, 0.075]  # 30x15x15 cm box
```

**Key functions:**

```python
def mj_quat_to_pin(quat_wxyz: np.ndarray) -> np.ndarray
    # (w,x,y,z) -> (x,y,z,w)

def pin_quat_to_mj(quat_xyzw: np.ndarray) -> np.ndarray
    # (x,y,z,w) -> (w,x,y,z)

def load_mujoco_model(scene_path: Path | None = None) -> tuple[MjModel, MjData]
    # Loads scene_dual.xml, returns (model, data)

def clip_torques(tau: np.ndarray) -> np.ndarray
    # Clips 6-element torque vector to TORQUE_LIMITS

def get_mj_body_id(mj_model, name: str) -> int
def get_mj_site_id(mj_model, name: str) -> int
```

**Why it exists:** Without this module, every file would independently define
DT, joint slices, torque limits, and file paths. A single change (e.g., moving
the table) would require editing multiple files. The common module ensures
one source of truth.

**What breaks if removed:** Every other module fails to import. The entire lab
is non-functional.

---

### 2.2 `dual_arm_model.py` (312 lines)

**Purpose:** Wraps two independent Pinocchio UR5e models with world-frame base
offsets that match the MuJoCo scene. Provides FK, Jacobian, and Damped
Least-Squares IK for each arm. This is the "analytical brain" of the system --
all kinematic computation passes through this class.

**Key class:**

```python
class DualArmModel:
    def __init__(self, urdf_path: Path | None = None) -> None
    def fk_left(self, q: np.ndarray) -> pin.SE3
    def fk_right(self, q: np.ndarray) -> pin.SE3
    def fk(self, arm: str, q: np.ndarray) -> pin.SE3
    def jacobian_left(self, q: np.ndarray) -> np.ndarray   # (6, 6)
    def jacobian_right(self, q: np.ndarray) -> np.ndarray   # (6, 6)
    def jacobian(self, arm: str, q: np.ndarray) -> np.ndarray
    def ik(
        self,
        arm: str,
        target_pos: np.ndarray,          # world frame
        target_rot: np.ndarray | None,   # None = position-only
        q_init: np.ndarray | None,
        *,
        damping: float = 0.01,
        max_iter: int = 200,
        tol_pos: float = 1e-4,           # 0.1 mm
        tol_rot: float = 1e-3,           # ~0.06 deg
        dq_max: float = 0.5,             # rad per iteration
        n_restarts: int = 8,
    ) -> tuple[np.ndarray, bool, dict]
```

**Internal method:**

```python
def _ik_single(
    self, model, data, ee_fid,
    target_pos_local, target_rot, q_init,
    damping, max_iter, tol_pos, tol_rot, dq_max,
) -> tuple[np.ndarray, bool, dict]
```

**Design decisions:**

- Two independent Pinocchio models (not one 12-DOF model). This is simpler,
  matches the physical reality of two separate robots sharing no kinematic
  chain, and avoids the complexity of a branching URDF. The cost is that
  there is no cross-arm Jacobian, but dual-arm coordination does not need one
  at the joint-PD control level used here.
- Base offsets are added to FK output rather than modifying the URDF. The URDF
  already contains the Menagerie 180-degree Z base rotation. The world-frame
  translation (LEFT=[0,0,0], RIGHT=[1,0,0]) is a pure offset.
- IK subtracts the base offset from the world-frame target before solving
  in the arm's local frame.

**What breaks if removed:** No FK, no Jacobian, no IK. The state machine cannot
compute any joint targets. The system reduces to a scene with no motion.

---

### 2.3 `joint_pd_controller.py` (70 lines)

**Purpose:** Joint-space PD controller with MuJoCo gravity compensation for
both arms. This is the only module that writes to `mj_data.ctrl[]`. All motion
in the system passes through this controller.

**Key class:**

```python
class DualArmJointPD:
    def __init__(self, kp: float | np.ndarray = 100.0,
                 kd: float | np.ndarray = 10.0) -> None
    @property
    def saturated(self) -> bool
    def compute(
        self,
        mj_data,
        q_target_left: np.ndarray,   # (6,)
        q_target_right: np.ndarray,  # (6,)
    ) -> None
```

**Control law:**

```
tau = Kp * (q_target - q) + Kd * (0 - qvel) + qfrc_bias
tau_clipped = clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
mj_data.ctrl[arm_slice] = tau_clipped
```

**Why it exists:** Separates the control computation from the state machine
logic. The state machine decides WHERE to go (joint targets); the controller
decides HOW to get there (torque computation). This separation means you can
swap controllers (e.g., impedance control) without touching the state machine.

**What breaks if removed:** No torques are applied to any actuator. Both arms
collapse under gravity.

---

### 2.4 `grasp_pose_calculator.py` (95 lines)

**Purpose:** Computes approach and grasp SE3 poses from the current box state
in MuJoCo. All targets are derived from the live box body position and
orientation -- nothing is hardcoded. This module answers the question: "Given
where the box is right now, where should each arm's end-effector be?"

**Key constants:**

```python
APPROACH_STANDOFF = 0.10   # 10 cm from box surface
GRASP_STANDOFF = 0.05      # 5 cm from box surface
```

**Key functions:**

```python
def _rotation_facing(direction: np.ndarray) -> np.ndarray
    # Builds a 3x3 rotation matrix where:
    #   z-axis = direction (normalized)
    #   y-axis ~ world +z (or +y if z is near-vertical)
    #   x-axis = cross(y, z), re-orthogonalized

def compute_grasp_poses(
    mj_model: MjModel,
    mj_data: MjData,
    box_body_name: str = "box",
) -> dict[str, pin.SE3]
    # Returns {
    #   "left_approach":  SE3 @ 10cm from -x side of box,
    #   "left_grasp":     SE3 @ 5cm from -x side of box,
    #   "right_approach": SE3 @ 10cm from +x side of box,
    #   "right_grasp":    SE3 @ 5cm from +x side of box,
    # }
```

**Design decisions:**

- The left arm always approaches from the -x side of the box (in the box's
  local frame), the right arm from the +x side. This aligns with the physical
  setup: left base at x=0, right base at x=1, box at x=0.5.
- The EE orientation is constructed so that the EE z-axis (approach direction)
  points toward the box center. This matches the EE site frame convention
  established by the `R_x(-90deg)` rotation in the MJCF files (see Lesson L2).
- Standoff distances are measured from the box surface, not the box center.

**What breaks if removed:** The state machine has no grasp targets. It uses a
local copy of `_compute_ee_targets_from_box()` (derived from this module's
logic) for the CLOSE/LIFT/CARRY/PLACE states, but APPROACH still needs the
exported constants and `_rotation_facing`.

---

### 2.5 `bimanual_state_machine.py` (638 lines)

**Purpose:** The central orchestrator. Implements a 6-state pipeline
(APPROACH, CLOSE, GRASP, LIFT, CARRY, PLACE) that coordinates both arms
through a cooperative manipulation task. Contains collision-free IK search,
weld constraint management, smooth-step trajectory ramping, and convergence
detection.

**Key enums and constants:**

```python
class State(enum.Enum):
    APPROACH = 1    # Both arms to grasp standoff
    CLOSE = 2       # Push EEs 2cm inside box surface
    GRASP = 3       # Activate weld constraints
    LIFT = 4        # Raise box 15cm
    CARRY = 5       # Translate box 20cm in +y
    PLACE = 6       # Lower box to table, release welds
    DONE = 7        # Terminal state

CONTACT_PENETRATION = 0.02   # 2cm inside box surface
LIFT_DZ = 0.15               # 15cm vertical lift
CARRY_DY = 0.20              # 20cm lateral carry (in +y, NOT +x)
SETTLE_THRESHOLD = 0.005     # rad -- position convergence
SETTLE_DURATION = 0.15       # seconds of sustained convergence
```

**Key module-level functions:**

```python
def _wrap_joints(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray
    # Wraps each joint to the equivalent angle closest to q_ref
    # Prevents 200deg motions when a 160deg motion in the opposite
    # direction reaches the same configuration

def _find_collision_free_ik(
    dual: DualArmModel,
    arm: str,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q_ref: np.ndarray,
    mj_model, mj_data,
    q_init: np.ndarray | None = None,
    n_trials: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray | None, bool]
    # Searches for collision-free IK solution closest to q_init
    # Evaluates up to 300 random starts, checks MuJoCo contacts

def _compute_ee_targets_from_box(
    box_pos: np.ndarray,
    box_rot: np.ndarray,
    standoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    # Returns (left_pos, left_rot, right_pos, right_rot)
    # Positive standoff = outside box, negative = inside box surface

def _solve_ik_pair(
    dual, left_pos, left_rot, right_pos, right_rot,
    q_prev_left, q_prev_right,
    mj_model, mj_data,
    collision_free: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, bool]
    # Solves IK for both arms, optionally with collision checking
```

**Key class:**

```python
class BimanualStateMachine:
    def __init__(
        self,
        mj_model, mj_data,
        dual: DualArmModel,
        controller: DualArmJointPD,
        renderer: mujoco.Renderer | None = None,
        frame_skip: int = 33,
    ) -> None

    def run(self) -> bool
        # Executes the full state machine, returns True if all gates pass

    # Internal methods:
    def _set_weld_relpose(self, weld_id, body1_name, body2_name) -> None
    def _run_pd_until_settle(self, max_time, ramp_time) -> dict
    def _hold(self, duration: float) -> None
    def _step(self) -> None
    def _box_pos(self) -> np.ndarray
    def _box_rot(self) -> np.ndarray
```

**Why it exists:** Without this module, there is no task-level intelligence.
The other modules provide capabilities (kinematics, control, pose
computation), but this module sequences them into a coherent manipulation task.

**What breaks if removed:** The cooperative carry demo cannot run. Individual
milestones (m0-m3) still work since they do not import this module, but m4 and
m5 (the full demo) depend on it entirely.

---

### 2.6 Model Files

#### `scene_dual.xml`

The top-level MJCF scene file. Key elements:

| Element | Details |
|---------|---------|
| Timestep | 0.001s (1 kHz), implicit-fast integrator |
| Left arm | Included via `ur5e_left.xml`, base at origin |
| Right arm | Included via `ur5e_right.xml`, base at (1, 0, 0) |
| Table | Body at (0.5, 0, 0.15), half-extents (0.20, 0.30, 0.02) |
| Box | Free body at (0.5, 0, 0.245), half-extents (0.15, 0.075, 0.075), density=296 |
| Weld constraints | `left_grasp` and `right_grasp`, body1=wrist_3_link, body2=box, initially inactive |

The table x half-extent is 0.20 (not the original 0.40). This was narrowed
after Lesson L3 revealed that a wider table collided with the right arm's
shoulder link.

The box density of 296 kg/m^3 produces a mass of approximately 1 kg
(0.15 * 2 * 0.075 * 2 * 0.075 * 2 * 296 = 1.0 kg). This is within the
UR5e's payload capacity but heavy enough to require gravity compensation.

#### `ur5e_left.xml` / `ur5e_right.xml`

Identical Menagerie body hierarchies with prefixed names (`left_`, `right_`).

Key details:

- **Base quaternion:** `quat="0 0 0 -1"` -- this is the Menagerie mesh-alignment
  rotation (180 deg about Z). Both arms have the same base orientation.
  There is no yaw rotation to make them "face each other" (see Lesson L1).
- **EE site:** `quat="0.7071 -0.7071 0 0"` applies R_x(-90 deg) so the site
  z-axis aligns with the tool approach direction (body +Y). This was added
  after Lesson L2.
- **Actuators:** Direct torque motors (`<motor>`) with ctrl limits matching
  TORQUE_LIMITS. Joints 1-3 (shoulder, upper arm, elbow): +/-150 Nm. Joints
  4-6 (wrist): +/-28 Nm.
- **Joint axes:** All joints use `axis="0 1 0"` except shoulder_pan (`0 0 1`)
  and wrist_2 (`0 0 1`). This matches the Menagerie convention.

#### `ur5e.urdf`

Lab 3's Menagerie-matching URDF. Key differences from the "standard" DH URDF:

| Property | Standard DH URDF | Lab 3 Menagerie URDF |
|----------|-----------------|---------------------|
| world_joint | identity | 180 deg Z rotation |
| shoulder_lift origin | `rpy="pi/2 0 0"` | `rpy="0 pi/2 0"` |
| Joint axes | `0 0 1` | `0 1 0` |
| FK match to MuJoCo | ~99 mm offset | 0.000 mm (exact) |

---

### 2.7 Milestone Scripts

| Script | Milestone | Purpose |
|--------|-----------|---------|
| `m0_validate_scene.py` | M0 | Loads scene, prints body/joint names, validates structure |
| `m1_independent_motion.py` | M1 | PD control of each arm independently, convergence test |
| `m2_fk_validation.py` | M2 | Cross-validates Pinocchio FK against MuJoCo xpos |
| `m2_ik_validation.py` | M2 | Tests DLS IK convergence on random targets |
| `m2_ik_visual.py` | M2 | Visual IK validation with MuJoCo rendering |
| `m3_coordinated_approach.py` | M3 | Both arms approach box simultaneously |
| `m4_cooperative_carry.py` | M4 | Full state machine: approach, grasp, lift, carry, place |
| `m5_capstone_demo.py` | M5 | Final demo with rendering and metric plots |

---

## 3. Controller Design Deep-Dive

### Why Joint PD, Not Impedance Control

Lab 3 implemented Cartesian impedance control for the UR5e. Lab 6 deliberately
does NOT use it. The reasons are pragmatic:

1. **Large reconfigurations dominate the task.** The HOME-to-approach motion
   requires up to 150 degrees of joint travel. Impedance control is designed
   for compliant behavior near a target, not for driving an arm across its
   workspace. A joint PD controller with gravity compensation tracks large
   joint-space trajectories directly and reliably.

2. **Weld constraints handle contact, not the controller.** In a single-arm
   grasp (Lab 5), the controller must produce contact forces. In Lab 6, the
   weld constraint rigidly attaches the EE to the box. The controller only
   needs to move the arm to configurations that are consistent with the weld.
   Impedance compliance would fight the weld constraint.

3. **Simplicity.** Joint PD with gravity compensation is a single equation:
   `tau = Kp*(q_target-q) + Kd*(0-qvel) + qfrc_bias`. No Jacobian transpose,
   no operational space dynamics, no null-space projection. For a task where
   all targets are pre-computed joint angles from IK, this is sufficient.

The CLAUDE.md project rules encode this explicitly: "No impedance control for
large motions. Joint PD only until within 10cm of target."

### Gravity Compensation Strategy

The controller uses MuJoCo's `qfrc_bias` as the gravity+Coriolis feedforward
term:

```
tau = Kp * (q_target - q) + Kd * (0 - qvel) + qfrc_bias
```

`mj_data.qfrc_bias` is computed by MuJoCo during `mj_step` (or `mj_forward`).
It contains the sum of gravity, Coriolis, and centrifugal forces -- everything
needed to hold the arm stationary at the current configuration with zero
acceleration.

**Why use MuJoCo's qfrc_bias instead of Pinocchio's gravity vector?**

In earlier labs, Pinocchio computed `g(q)` via RNEA and this was used as
feedforward. That approach requires a Pinocchio `model.createData()` call and
`pin.computeGeneralizedGravity()` every timestep. Since MuJoCo already computes
`qfrc_bias` as part of the simulation step, using it is (a) free (already
computed), (b) guaranteed to match the simulation dynamics exactly, and
(c) includes Coriolis terms that Pinocchio's gravity-only computation omits.

The tradeoff is coupling: the controller now depends on MuJoCo internals
rather than being a pure function of (q, qvel, q_target). For this lab, that
coupling is acceptable because the controller only ever runs inside MuJoCo.

### Gain Selection: Kp and Kd

The lab uses two gain regimes:

| Context | Kp | Kd | Rationale |
|---------|----|----|-----------|
| M1 (small motions, validation) | 100 | 10 | Low gains for gentle regulation near home |
| M3 (approach, large reconfig) | 500 | 50 | High gains for fast tracking of 150-deg motions |
| M4 (state machine with ramp) | 300 | 40 | Moderate gains + smooth-step ramp for smooth motion |

**What happens if Kp is too low (e.g., 50)?**

For a 2.5 rad error (typical HOME-to-approach), the proportional torque is
50 * 2.5 = 125 Nm. After gravity compensation (~30-80 Nm for the UR5e shoulder
in horizontal configurations), the net tracking torque is only 45-95 Nm. With
joint friction and the 150 Nm clamp, the arm moves sluggishly and may not
converge within the 5-second phase timeout.

**What happens if Kp is too high (e.g., 1000)?**

The proportional term dominates and the clipping at 150 Nm / 28 Nm engages
for any error above 0.15 rad (shoulder) or 0.028 rad (wrist). The arm
accelerates at maximum torque, overshoots the target, and oscillates. The
derivative term Kd would need to be proportionally large to damp the
oscillations, but high Kd also amplifies sensor noise in qvel.

**What happens if Kd is too low relative to Kp?**

Underdamped response: the arm overshoots and rings. For Kp=500 and Kd=10,
the damping ratio zeta = Kd / (2*sqrt(Kp*I)) is well below 1 for any
reasonable link inertia. The ringing can cause oscillating contact forces
that destabilize the grasp.

**What happens if Kd is too high relative to Kp?**

Overdamped response: the arm moves slowly and takes many seconds to converge.
For Kp=100 and Kd=100, the arm barely moves -- the derivative term kills any
velocity before the proportional term can drive the arm to the target.

The chosen ratios (Kp/Kd ~ 7-10) produce near-critically-damped responses for
the UR5e's link inertias, confirmed empirically by observing convergence times
(0.5-2.5 seconds for typical motions).

### Why No Direct Cartesian-Space Control

The architecture deliberately keeps control in joint space:

```
Task space target (SE3)
    |
    v
IK solver (Pinocchio) ---> joint angle target (6,)
    |
    v
Joint PD controller (MuJoCo) ---> torques (6,)
```

An alternative would be Cartesian-space control:

```
Task space target (SE3)
    |
    v
Cartesian controller ---> wrench (6,)
    |
    v
J^T * wrench ---> torques (6,)
```

The joint-space approach is preferred because:

1. **IK pre-validates reachability.** If IK fails, the system knows before
   commanding any motion. Cartesian control can drive the arm into singularities
   or joint limits without warning.
2. **Collision checking happens at IK time.** The collision-free IK search
   evaluates MuJoCo contacts at candidate configurations. Cartesian control
   would need online collision avoidance, which is more complex.
3. **Weld-loaded motion.** During LIFT/CARRY, the weld constraint applies
   external forces to the EE. A Cartesian controller would need to account
   for these forces in its dynamics model. Joint PD with qfrc_bias handles
   them implicitly.

---

## 4. IK Pipeline

### Damped Least-Squares Formulation

The IK solver minimizes the task-space error using the Damped Least-Squares
(DLS) method, also known as Levenberg-Marquardt regularization of the
pseudoinverse.

**Problem statement:** Given a desired end-effector pose `T_des` and current
joint configuration `q`, find `dq` that reduces the task-space error.

**Error computation:**

For 6-DOF (position + orientation):

```
e_pos = p_des - p_current           (3,)  -- position error
R_err = R_des @ R_current^T         (3,3) -- rotation error
e_rot = log3(R_err)                 (3,)  -- axis-angle representation
e = [e_pos; e_rot]                  (6,)  -- full task-space error
```

For position-only (3-DOF):

```
e = e_pos = p_des - p_current       (3,)
```

**Jacobian:**

```
J = getFrameJacobian(model, data, ee_fid, WORLD)   (6, 6) for full
J = J[:3, :]                                        (3, 6) for position-only
```

The Jacobian is computed in the WORLD reference frame. Since each arm has its
own Pinocchio model with 6 joints, J is always square (6x6) for full IK or
wide (3x6) for position-only.

**DLS update:**

```
dq = J^T @ (J @ J^T + lambda^2 @ I)^{-1} @ e
```

This is equivalent to the regularized pseudoinverse `dq = (J^T J + lambda^2 I)^{-1} J^T e`
but is computationally preferable when `m < n` (task dimension < joint dimension),
which is always the case for position-only IK (m=3, n=6).

The key insight: `J @ J^T` is `m x m` (3x3 or 6x6), while `J^T @ J` is `n x n`
(6x6 always). For position-only IK, the DLS formulation inverts a 3x3 matrix
instead of a 6x6 matrix.

### Why lambda = 0.01

The damping factor lambda controls the tradeoff between tracking accuracy and
joint velocity magnitude near singularities.

- **lambda = 0:** Pure pseudoinverse. Near singularities, `dq` magnitudes
  explode because the smallest singular value of J approaches zero, and the
  pseudoinverse amplifies the error component in that direction without bound.

- **lambda >> 0 (e.g., 0.1):** Heavy damping. The solver converges slowly
  because `dq` is suppressed even when far from singularities. The effective
  step size is reduced by a factor of `sigma^2 / (sigma^2 + lambda^2)` for
  each singular value sigma.

- **lambda = 0.01:** Light damping. For the UR5e, typical singular values
  range from 0.01 to 1.0. At sigma=0.01 (near-singular), the damping factor
  reduces the contribution to `0.01^2 / (0.01^2 + 0.01^2) = 0.5` -- a 50%
  reduction, preventing explosion. At sigma=0.1 (healthy), the reduction is
  `0.1^2 / (0.1^2 + 0.01^2) = 0.99` -- negligible impact. This provides
  singularity robustness without sacrificing convergence speed.

### Multi-Start Strategy

The UR5e has multiple IK solutions for most targets (up to 8 for a 6R
manipulator). The DLS solver is iterative and converges to the solution nearest
to the initial guess `q_init`. A single start from `Q_HOME` may converge to a
solution that:

1. Collides with the table or the other arm.
2. Requires joint wrapping that crosses the joint limit boundary.
3. Is far from the previous configuration, causing large PD transitions.

The multi-start approach:

```
1. Try q_init (or Q_HOME) first
2. If converged, return immediately
3. Otherwise, for i in 1..n_restarts:
   a. q_rand = Q_HOME + uniform(-1.0, +1.0) for each joint
   b. Run _ik_single from q_rand
   c. If converged and pos_err < best_err, update best
4. Return best solution found
```

Default `n_restarts=8` was determined empirically: 6-DOF targets converge
in 1-3 restarts for most poses in the workspace. Position-only targets
(underconstrained) need more restarts because the 3-dimensional null space
makes convergence from a random start less likely.

### Step Clamping

Each DLS iteration produces a joint update `dq`. Without clamping, `dq` can
be arbitrarily large when the error is large and the Jacobian is well-conditioned.
Large steps overshoot, especially when the Jacobian is linearized at a
configuration far from the solution.

The clamping rule:

```python
dq_norm = np.linalg.norm(dq)
if dq_norm > dq_max:
    dq = dq * (dq_max / dq_norm)
```

With `dq_max = 0.5` rad, the maximum joint motion per iteration is 28.6 degrees.
This is small enough to keep the Jacobian linearization valid over the step,
but large enough that convergence typically occurs within 20-50 iterations
(well under the 200 iteration limit).

### How Base Transforms Are Applied

Pinocchio models the arm in its local frame (base at origin). MuJoCo places
the arms at different world positions. The bridge:

```python
# In ik():
target_pos_local = target_pos_world - base_offset   # Subtract base
q_solution = _ik_single(..., target_pos_local, ...)  # Solve in local frame

# In fk():
oMf = pin_fk(q)                                     # Local frame result
world_pose = SE3(oMf.rotation, oMf.translation + base_offset)  # Add base
```

The rotation is NOT affected by the base offset (both arms have the same
base orientation). Only translation is shifted.

---

## 5. Grasp Pose Computation

### How Targets Derive from Box Pose

The box is a free body with 6 DOF (3 position + 3 orientation via freejoint).
At any simulation time, its world-frame state is:

```python
box_pos = mj_data.xpos[box_id]           # (3,) center position
box_rot = mj_data.xmat[box_id].reshape(3, 3)  # (3,3) rotation matrix
```

The box's local x-axis (longest dimension, half-extent 0.15m) is extracted as:

```python
box_x_axis = box_rot[:, 0]     # First column of rotation matrix
```

End-effector positions are computed as offsets from the box center along this axis:

```
left_pos  = box_pos - box_x_axis * (box_half_x + standoff)
right_pos = box_pos + box_x_axis * (box_half_x + standoff)
```

Where `standoff` is the distance from the box surface (not center) to the EE.
At the initial configuration (box axis aligned with world x), this places the
left EE to the left and the right EE to the right, with the box between them.

### Why 2cm Inside Box Surface (CONTACT_PENETRATION)

During the CLOSE state, the EE targets are set 2cm INSIDE the box surface
(`standoff = -0.02`). This serves two purposes:

1. **Ensures contact.** The IK solution places the EE nominally at the target,
   but PD tracking error, gravity droop, and box compliance mean the actual
   EE may be 2-5mm away from the nominal position. By targeting 2cm inside
   the surface, the actual contact force is guaranteed -- the MuJoCo contact
   solver prevents physical penetration and produces the normal force needed
   for the weld constraint activation to be meaningful.

2. **Sets the weld relative pose.** When the weld constraint is activated in
   the GRASP state, it locks the current relative transform between the
   wrist link and the box. If the EE is not in contact (floating 5mm above
   the surface), the weld locks a non-contact relative pose. During LIFT,
   the weld would need to produce adhesive forces to maintain this gap,
   which is physically nonsensical. By ensuring contact penetration, the
   locked relative pose is one where the arm is pressing against the box.

The 2cm value was chosen as a balance: large enough to guarantee contact
despite tracking error, small enough that the contact forces do not significantly
disturb the box position before the weld activates.

### Orientation: lookAt Toward Box

The EE orientation is constructed by `_rotation_facing(direction)`:

```python
def _rotation_facing(direction):
    z = normalize(direction)           # Approach direction
    up = [0, 0, 1]                     # World z preferred for y
    if abs(dot(z, up)) > 0.99:         # Near-vertical: use world y
        up = [0, 1, 0]
    x = normalize(cross(up, z))        # Right-hand rule
    y = cross(z, x)                    # Orthogonal completion
    return [x, y, z]                   # Column-stack to 3x3
```

For the left arm, `direction = +box_x_axis` (EE approaches from -x, pointing
toward +x). For the right arm, `direction = -box_x_axis`. This makes both
EEs "look at" the box center.

The EE z-axis convention matches the MJCF site frame: the `R_x(-90deg)`
rotation on `left_ee_site` / `right_ee_site` makes the site z-axis equal to
the tool approach direction (body +Y). The Pinocchio IK solves for this same
z-axis alignment.

### Standoff/Approach/Contact Progression

The approach sequence uses three standoff distances:

```
APPROACH (10cm)  -->  GRASP_STANDOFF (5cm)  -->  CLOSE (-2cm)
     |                       |                       |
     |     Phase 1           |     Phase 2           |     Phase 3
     |  HOME -> 10cm         |  10cm -> 5cm          |  5cm -> contact
     |  (large reconfig)     |  (small adjustment)   |  (push into box)
```

Phase 1 (HOME to APPROACH) is the largest motion -- the arm goes from pointing
down to pointing sideways. This is where collision-free IK is critical.

Phase 2 (APPROACH to GRASP_STANDOFF) is a small 5cm inward motion. It uses
the Phase 1 solution as the IK seed, so the joint change is minimal (~0.68 rad
max joint delta).

Phase 3 (GRASP_STANDOFF to CLOSE) pushes the EE 7cm further inward, through
the box surface. This uses non-collision-free IK because contact is the goal.

---

## 6. State Machine Walkthrough

### State Transition Diagram

```
    +----------+          +-------+          +-------+
    | APPROACH |--settle->| CLOSE |--settle->| GRASP |
    | (2 phase)|          |(push) |          |(weld) |
    +----------+          +-------+          +-------+
                                                 |
                                              hold 0.15s
                                                 |
                                                 v
    +-------+          +-------+          +------+
    | PLACE |<--settle-| CARRY |<--settle-| LIFT |
    |(lower)|          | (+y)  |          |(+z)  |
    +-------+          +-------+          +------+
        |
        v
    (release welds)
        |
        v
    (retract arms)
        |
        v
    +------+
    | DONE |
    +------+
```

### State 1: APPROACH

**Entry condition:** System starts in this state. Both arms at Q_HOME. Box at
initial position (0.5, 0, 0.245).

**What happens:**

1. Read current box position and rotation from MuJoCo.
2. Compute APPROACH poses (10cm standoff from each side of box).
3. Run `_find_collision_free_ik()` for both arms (300 trials each).
   This evaluates random IK starts, wraps joints to minimize distance from
   Q_HOME, checks each solution for MuJoCo contact penetration, and keeps
   the closest collision-free solution.
4. Compute GRASP_STANDOFF poses (5cm standoff) seeded from the APPROACH
   solutions. This ensures the joint-space transition from approach to
   grasp standoff is small.
5. Phase 1: Command both arms to APPROACH targets. Run `_run_pd_until_settle()`
   which smooth-step ramps the target over 2 seconds and waits for convergence.
6. Phase 2: Command both arms to GRASP_STANDOFF targets. Same settle procedure.

**Exit condition:** Both arms have settled within SETTLE_THRESHOLD (0.005 rad)
for SETTLE_DURATION (0.15s), or velocity has dropped below 0.02 rad/s after
the ramp completes.

**Controller:** Joint PD with Kp=300, Kd=40 (set by the milestone script,
not the state machine).

### State 2: CLOSE

**Entry condition:** Both arms at grasp standoff (5cm from box surface).

**What happens:**

1. Compute CLOSE targets: EE positions 2cm inside box surface
   (`standoff = -CONTACT_PENETRATION`).
2. Solve IK for both arms WITHOUT collision checking (`collision_free=False`).
   Contact is intentional in this state.
3. Command both arms to CLOSE targets with `_run_pd_until_settle()`.
4. Hold for 0.15 seconds to stabilize contact forces.
5. Gate check: verify each EE is within 3cm of the box surface.

**Exit condition:** PD convergence or velocity settle.

**Why no collision-free IK:** The whole point of CLOSE is to establish contact.
Collision-free IK would reject any solution where the arm touches the box.

### State 3: GRASP

**Entry condition:** Both arms in contact with box. EEs 2cm inside box surface.

**What happens:**

1. Call `_set_weld_relpose()` for both weld constraints. This reads the current
   world-frame positions and rotations of `left_wrist_3_link` and `box`,
   computes the relative transform (position and quaternion of box in the
   wrist link's local frame), and writes it into `mj_model.eq_data[weld_id]`.

   The eq_data layout for a weld constraint (11 floats):

   ```
   [0:3]   anchor point on body1 (wrist link local frame) = [0, 0, 0]
   [3:6]   relative position of body2 (box) in body1 frame
   [6:10]  relative quaternion (w, x, y, z) of body2 in body1 frame
   [10]    torquescale = 1.0
   ```

2. Activate both weld constraints: `mj_data.eq_active[weld_id] = 1`.

3. Hold for GRASP_HOLD_TIME (0.15s) to let the constraint solver stabilize.

4. Gate check: verify box drift from initial position is less than 1cm.

**Exit condition:** Hold duration elapsed.

**Why weld constraints instead of friction-based grasping:**

Friction-based grasping requires:
- Accurate contact force control (impedance or force controller)
- Sufficient friction coefficient between EE and box
- Grip force maintenance throughout LIFT/CARRY (the UR5e EE is a flat
  flange, not a gripper)
- Handling of force/position coupling in the controller

Weld constraints bypass all of this. They create a rigid attachment between the
wrist link and the box, which is physically equivalent to "the robot is holding
the box with a perfect gripper." This is a deliberate simplification: Lab 5
already demonstrated friction-based grasping with a Robotiq 2F-85 gripper.
Lab 6 focuses on dual-arm coordination, not grasp quality.

**How weld body names resolve:**

The MJCF declares:

```xml
<weld name="left_grasp" body1="left_wrist_3_link" body2="box" active="false"/>
```

At runtime, `mujoco.mj_name2id(model, mjOBJ_EQUALITY, "left_grasp")` returns
the constraint index. The body names `left_wrist_3_link` and `box` are resolved
at model compilation time. The `_set_weld_relpose()` method looks up these body
IDs dynamically to compute the current relative pose.

**Why set eq_data before activation:**

The eq_data is initialized at model compilation from the initial body positions
in the MJCF. At compile time, `left_wrist_3_link` is at Q_HOME and `box` is at
(0.5, 0, 0.245). The relative pose is meaningless for the grasped
configuration. If you activate the weld without updating eq_data, MuJoCo's
constraint solver tries to restore the compile-time relative pose, which
launches the box across the scene. See Lesson L8.

### State 4: LIFT

**Entry condition:** Weld constraints active. Box rigidly attached to both wrists.

**What happens:**

1. Compute lifted box position: current box pos + [0, 0, LIFT_DZ] where
   LIFT_DZ = 0.15m.
2. Compute EE targets using `standoff = -CONTACT_PENETRATION` (matching the
   locked weld offset). This is critical -- see Lesson L10.
3. Solve IK for both arms (no collision check needed -- welds prevent separation).
4. Command both arms with `_run_pd_until_settle()`.

**Exit condition:** PD convergence (position or velocity based).

**Why -CONTACT_PENETRATION standoff:** The weld locked the EE at 2cm inside
the box surface. If the IK target uses a different standoff (e.g., 5cm
outside), the PD controller and weld constraint fight: the PD tries to push
the arm to 5cm standoff while the weld holds it at -2cm. The result is internal
forces that waste torque and may not converge.

### State 5: CARRY

**Entry condition:** Box lifted 15cm above table.

**What happens:**

1. Compute carried box position: current box pos + [0, CARRY_DY, 0] where
   CARRY_DY = 0.20m (in +y direction).
2. Same IK and PD procedure as LIFT.

**Exit condition:** PD convergence.

**Why +y and not +x:** The two arm bases are separated along x (0, 0, 0) and
(1, 0, 0). The box starts at x=0.5 (midpoint). Carrying in +x moves the box
toward the right arm base. At x=0.7, the right EE needs to reach
x = 0.7 + 0.15 + 0.02 = 0.87 (box_half_x + penetration) with the wrist
pointing in -x. This is near the workspace boundary for the right arm. Carrying
in +y maintains symmetric arm configurations because both arms have the same
y-offset (0). See Lesson L9.

### State 6: PLACE

**Entry condition:** Box carried to new location (displaced 20cm in +y).

**What happens:**

1. Compute placement box position with absolute z:
   `z = TABLE_SURFACE_Z + BOX_HALF_EXTENTS[2] = 0.17 + 0.075 = 0.245`.
   The absolute z is used instead of a relative delta to avoid accumulating
   errors from LIFT overshoot (Lesson L11).
2. Same IK and PD procedure as LIFT/CARRY.
3. After settle, deactivate both weld constraints: `eq_active = 0`.
4. Immediately solve IK for retraction poses (APPROACH_STANDOFF = 10cm from
   box) to prevent the arms from pushing the now-released box.
5. Command retraction with `_run_pd_until_settle()`.
6. Hold for PLACE_HOLD_TIME (0.15s).
7. Transition to DONE.

**Exit condition:** Always transitions to DONE (no gate check on placement
accuracy in the current implementation).

**Why retract after weld release:** When the welds deactivate, the arms are
still at -CONTACT_PENETRATION (2cm inside the box surface). The contact forces
from the arm pushing against the box surface immediately push the box sideways
or off the table. Retracting to 10cm standoff breaks contact and lets the box
settle under gravity onto the table surface. See Lesson L11.

### Settle Detection: `_run_pd_until_settle()`

This is the core motion execution loop, used by every state except GRASP
(which uses `_hold()`).

```python
def _run_pd_until_settle(self, max_time=5.0, ramp_time=2.0) -> dict:
```

**Smooth-step ramp:**

Instead of commanding the final joint target instantly, the method interpolates
between the current and target configurations using a smooth-step function:

```python
alpha = min(1.0, (step + 1) / n_ramp)                 # Linear 0->1 over ramp_time
alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)     # Hermite smoothstep
q_target = q_start + alpha_smooth * (q_final - q_start)
```

The smooth-step `3*alpha^2 - 2*alpha^3` has zero derivative at both endpoints
(alpha=0 and alpha=1), which means zero velocity at the start and end of the
ramp. This prevents jerk at transition boundaries.

**Dual convergence criterion:**

The loop exits when EITHER of two conditions is met:

1. **Position-based:** Max joint error < SETTLE_THRESHOLD (0.005 rad = 0.29 deg)
   sustained for SETTLE_DURATION (0.15s = 150 timesteps) on BOTH arms.

2. **Velocity-based:** Max joint velocity < 0.02 rad/s sustained for
   SETTLE_DURATION on BOTH arms, but ONLY after the ramp has completed
   (alpha >= 1.0).

The velocity criterion handles weld-loaded phases (LIFT, CARRY, PLACE) where
the weld constraint creates a residual position error that the PD controller
cannot eliminate. The arm reaches a static equilibrium with the weld force,
so velocity drops to zero even though position error remains. Without the
velocity criterion, these phases would always time out.

**Post-settle hold:**

After either criterion is met, the method holds the final target for 0.1
seconds (100 timesteps) to ensure complete settling before transitioning states.

---

## 7. Lessons Learned

This section documents all 12 engineering lessons discovered during Lab 6
development. Each lesson includes the symptom that triggered investigation,
the root cause, the fix applied, and the general takeaway.

### L1: Don't use base yaw to make arms face each other

**Symptom:** A 180-degree yaw mount on the right arm base made Q_HOME mirroring
intractable. Three iterations of joint sign-flipping and grid search could not
reliably produce symmetric downward-pointing EEs.

**Root cause:** A base yaw rotation changes the relationship between joint space
and Cartesian space. Mirroring joint values does not mirror Cartesian poses
when base orientations differ. The mount R_z(pi) composed with Menagerie's
R_z(pi) base quaternion produced confusing effective orientations -- the two
rotations partially cancelled, leaving the arm in an unexpected frame.

**Fix:** Removed the 180-degree yaw from the right arm base entirely. Both arms
now have identical base orientation (Menagerie `quat="0 0 0 -1"` only).
Q_HOME_RIGHT = Q_HOME_LEFT. "Arms facing each other" is deferred to the IK
targets in the APPROACH state.

**Takeaway:** Keep base frames identical for both arms. Handle facing direction
through IK/trajectory planning, not base rotation. Simpler base = simpler
everything downstream. If you need arms to face each other, let the IK solver
figure out the joint angles for the desired EE orientation.

---

### L2: UR5e EE site frame needs explicit rotation for approach direction

**Symptom:** The EE site z-axis pointed along the body Z direction (sideways),
not along the tool approach direction. The gate criterion `dot(z, -z_world)`
always measured approximately 0, indicating the site was not pointing downward
as expected.

**Root cause:** MuJoCo sites inherit the body frame by default. The UR5e's tool
approach direction is along body +Y, not +Z. The wrist_3_link body frame has
its Y axis along the tool axis. A bare site with no explicit orientation
inherits the body frame's Z axis, which is perpendicular to the tool.

**Fix:** Added `quat="0.7071 -0.7071 0 0"` (which encodes R_x(-90 deg)) to both
EE sites. This rotates the site frame so that the site z-axis aligns with
body +Y, which is the tool approach direction.

**Takeaway:** Always define EE site orientation explicitly. Do not assume body
frame axes match task-space conventions. Print the site frame axes at a known
configuration to verify before using them in control or metric computation.

---

### L3: Table collision blocking right arm PD convergence

**Symptom:** Right arm steady-state joint error was approximately 0.6 rad at the
shoulder_pan joint while the left arm converged to less than 0.001 rad. Same
Kp/Kd for both. The right arm appeared "stuck."

**Root cause:** The table x half-extent was originally 0.40, placing the table
edge at x=0.9. The right arm base at x=1.0 meant the UR5e's shoulder and
upper-arm collision capsules (radius 0.06m) were physically inside the table
geometry. MuJoCo's contact solver generated persistent contact forces that
prevented the shoulder_pan joint from reaching its target.

**Fix:** Narrowed the table x half-extent from 0.40 to 0.20 (table now spans
x=0.3 to x=0.7). This provides 0.3m clearance from each arm base to the
nearest table edge.

**Takeaway:** When a joint-space controller fails to converge on one arm but
not the other, check `data.ncon` and inspect contact geom pairs. Collision
with scene geometry is the most common cause of asymmetric PD failures. The
debugging sequence: (1) print `data.ncon`, (2) iterate over `data.contact`,
(3) map geom IDs to body names, (4) identify unexpected contact pairs.

---

### L4: Lab 4 URDF uses different kinematic convention than MuJoCo Menagerie

**Symptom:** Pinocchio FK with the Lab 4 URDF gave approximately 99mm offset
from MuJoCo EE positions. The offset rotated with shoulder_pan, indicating it
was constant in the arm's local frame.

**Root cause:** Lab 4's URDF uses the standard DH convention from
`universal_robots_description` (shoulder_lift: `rpy="pi/2 0 0" xyz="0 0 0"`,
elbow: `xyz="-0.425 0 0"`). MuJoCo Menagerie uses a different body layout
(shoulder_lift: `pos="0 0.138 0" quat="1 0 1 0"`). These are different
parameterizations of the same physical robot -- they produce identical EE
trajectories in their own reference frames, but the intermediate link frames
differ.

Lab 3's URDF was hand-tuned to match Menagerie exactly, including: (1) a
180-degree Z rotation in `world_joint`, (2) `shoulder_lift rpy="0 pi/2 0"
xyz="0 0.138 0"`, (3) joint axes as `0 1 0` instead of `0 0 1`.

**Fix:** Replaced Lab 6's URDF with Lab 3's Menagerie-matching kinematic chain
(minus the gripper payload link). FK error dropped from 99mm to 0.000mm
(exact match within numerical precision).

**Takeaway:** Always verify which URDF matches your MuJoCo model. The
"standard" DH-convention URDF from ROS packages does NOT match MuJoCo
Menagerie. When Pinocchio FK disagrees with MuJoCo by a constant rotated
offset, the URDF kinematic chain is the first thing to check. The cross-
validation pattern (`assert np.allclose(pin_pos, mj_pos, atol=1e-3)`) should
run on the first forward kinematics test, not after building the full pipeline.

---

### L5: DLS IK needs step clamping and multi-start for reliability

**Symptom:** Basic DLS IK (damping=0.01, max_iter=100) failed on 4 out of 20
6-DOF targets and all position-only targets. The solver got stuck in local
minima with 500-990mm residual error.

**Root cause:** Without step clamping, large Jacobian pseudoinverse steps
overshoot, especially near singularities where one direction of dq is
amplified. The solver oscillates across the target without converging. A single
initial guess (Q_HOME) is far from some targets in joint space, and the DLS
gradient path from Q_HOME may pass through a singularity.

**Fix:** Two changes: (1) Added `dq_max=0.5` rad step clamping per iteration.
This limits joint motion to 28.6 degrees per step, keeping the Jacobian
linearization valid. (2) Added multi-start with `n_restarts=8` random
perturbations of +/-1 rad around Q_HOME. Results: 6-DOF targets: 20/20
converge. Position-only with n_restarts=20: 5/5 converge.

**Takeaway:** DLS IK for 6-DOF arms should always include both step clamping
and multi-start. Position-only IK (underconstrained, 3 equations in 6
unknowns) needs more restarts than full 6-DOF because the 3D null space
creates many local minima.

---

### L6: IK solutions must be collision-checked and joint-wrapped for dual-arm

**Symptom:** Arm PD controller failed to converge to IK targets (error stuck at
1.0+ rad). The left arm collided with the box during the swing from HOME to
approach. The right arm's upper_arm_link rested on the table at the final
configuration. MuJoCo reported 9-11 contact pairs.

**Root cause:** Three compounding issues: (1) The IK solver finds kinematically
valid but physically colliding configurations -- it has no knowledge of scene
geometry. (2) Without joint wrapping, IK finds solutions that require 200+
degrees of joint motion, going "the long way around" via 2*pi-equivalent
angles. The longer path may sweep through the box or table. (3) The
HOME-to-approach reconfiguration is genuinely large (approximately 150-degree
max joint change) because the EE must rotate from pointing down to pointing
sideways.

**Fix:** Three-part solution:

1. **Joint wrapping:** After IK converges, wrap each joint angle to the
   equivalent value closest to `q_ref` (typically Q_HOME). This ensures the
   shortest-path joint motion.

2. **Collision-free IK search:** Evaluate 300 random IK starts. For each
   converged solution, set MuJoCo qpos to the candidate, call `mj_forward`,
   check all contact pairs for arm-scene penetration (ignoring box-table
   contact), and keep the closest collision-free solution to the center
   configuration.

3. **Chained IK:** Use the APPROACH solution as the seed for GRASP_STANDOFF
   IK. This makes the Phase 2 transition only approximately 0.68 rad (max
   joint delta) instead of a completely different configuration.

**Takeaway:** For dual-arm scenes with obstacles, IK must be followed by
collision checking. Joint wrapping is essential for PD control -- a 200-degree
motion that could be a 160-degree motion in the opposite direction may avoid
collisions entirely. Always chain sequential IK targets (approach then grasp)
so transitions are small.

---

### L7: Large reconfigurations need higher PD gains

**Symptom:** Kp=100, Kd=10 (the gains from M1 small-motion validation) failed
to drive the arm through a 150-degree reconfiguration from HOME to the approach
pose. The arms barely moved in 8 seconds, making only small oscillations around
HOME.

**Root cause:** For small motions near equilibrium, Kp=100 produces
100*0.005=0.5 Nm at threshold -- sufficient for fine regulation. For large
motions (2.5 rad error), it produces 100*2.5=250 Nm, which clips to 150 Nm for
the big joints. But the real issue is Kd=10 relative to Kp=100: the
damping ratio is too high for fast tracking. The critically-damped natural
frequency is too low, producing sluggish response for these distances. The
arm effectively cannot accelerate fast enough because the derivative term
absorbs most of the proportional torque.

**Fix:** Increased to Kp=500, Kd=50 for M3. Both arms settled in 0.62 seconds
with 2ms synchronization error. For M4 (with smooth-step ramping), Kp=300,
Kd=40 proved sufficient because the ramp prevents the full position error
from appearing instantaneously.

**Takeaway:** PD gains should be scaled to the task. Small-error regulation
(M1: 0.001 rad reference tracking) works with low gains. Large reconfigurations
(M3: 2.5 rad traversal) need higher Kp and proportionally higher Kd. The
Kp/Kd ratio should stay around 7-10 for near-critical damping.

---

### L8: Weld constraints must have eq_data set to current relative pose before activation

**Symptom:** Activating weld constraints (`eq_active=1`) at runtime launched the
box 90+ cm away. The box position went from [0.5, 0, 0.245] to
[1.12, 0.65, 0.37] in 0.5 seconds.

**Root cause:** MuJoCo weld constraints enforce the relative pose stored in
`mj_model.eq_data[weld_id]`. This data is initialized at model compilation time
from the initial body positions in the MJCF. At compile time, `left_wrist_3_link`
is at the Q_HOME configuration and `box` is at (0.5, 0, 0.245). The relative
pose between these two bodies at compile time is completely different from the
relative pose at grasp time (when the wrist is touching the box). Activating
the weld makes the constraint solver try to restore the compile-time relative
pose, which requires moving the box rapidly.

**Fix:** Before activating, compute the current relative transform (position +
quaternion) between body1 (wrist link) and body2 (box) and write it into
`mj_model.eq_data[weld_id]`. The computation:

```python
rel_pos = mat1.T @ (pos2 - pos1)          # body2 position in body1 frame
rel_mat = mat1.T @ mat2                    # body2 rotation in body1 frame
mju_mat2Quat(rel_quat, rel_mat.flatten()) # Convert to (w,x,y,z) quaternion
eq_data[0:3] = [0, 0, 0]                  # anchor at body1 origin
eq_data[3:6] = rel_pos
eq_data[6:10] = rel_quat
eq_data[10] = 1.0                         # torquescale
```

**Takeaway:** Never activate runtime weld constraints without first setting
`eq_data` to the current relative pose. The default eq_data from compilation
is almost never what you want at runtime. This is a common MuJoCo pitfall that
is not well-documented -- the weld constraint reference documentation describes
the eq_data layout but does not warn about the compilation-time initialization.

---

### L9: Carry direction constrained by dual-arm workspace geometry

**Symptom:** CARRY IK failed when attempting a 20cm carry in the +x direction.
The right arm IK returned no solution.

**Root cause:** Both arm bases are at y=0, separated by 1.0m in x. The box
center starts at x=0.5. Carrying in +x moves the box to x=0.7. The right EE
position would be at `x = 0.7 + 0.15 + 0.02 = 0.87` (box_half_x +
penetration). With the right arm base at x=1.0, the arm needs to reach
a point only 0.13m in front of the base while maintaining a -x pointing
orientation. At the lifted z=0.41, this is beyond the kinematic workspace
of the UR5e with the required orientation.

**Fix:** Changed carry direction from +x to +y. Lateral carry is symmetric for
both arms (both at y=0), so a 20cm +y displacement keeps both arms comfortably
within their workspace envelopes.

**Takeaway:** For dual-arm setups where the bases are separated along one axis,
carrying along that axis is severely constrained because it moves the object
asymmetrically toward one base. Carry along the perpendicular axis preserves
symmetric reachability. Plan the workspace and carry direction during the
architecture phase, not after IK failures during implementation.

---

### L10: IK standoff during weld-active phases must match locked EE-to-box offset

**Symptom:** During LIFT and CARRY, the arms produced high torques and vibrated.
The PD controller was saturating (torques clipped to limits) even though the
box was moving in the correct direction.

**Root cause:** The LIFT/CARRY IK targets were computed with
`standoff=GRASP_STANDOFF` (5cm outside box surface). But the weld constraints
locked the EE at `standoff=-CONTACT_PENETRATION` (-2cm, inside box surface).
This 7cm offset between the PD target and the weld-enforced position created
internal forces: the PD controller tried to pull the arm 7cm outward while
the weld held it in place. The PD wasted its full torque budget fighting the
weld.

**Fix:** Changed all weld-active phases (LIFT, CARRY, PLACE) to use
`-CONTACT_PENETRATION` as the standoff when computing EE targets from the
desired box position. Now the IK target matches the weld-enforced EE position,
and the PD controller only needs to produce the torques for moving the arm to
the new configuration.

**Takeaway:** Once a weld locks a relative pose, all subsequent IK targets must
be computed using that same relative offset. The general principle: if an
external constraint fixes some relationship, the controller's reference must
be consistent with that constraint. Internal forces between the controller and
the constraint are pure waste.

---

### L11: Box placement needs absolute z target and arm retraction

**Symptom:** After weld release, the box slid off the table. The z position went
from 0.26 to 0.10, and x/y drifted 10cm. Rotation error was 120 degrees.

**Root cause:** Two compounding issues:

1. PLACE used a relative delta (`-LIFT_DZ`) to compute the lowering target.
   But LIFT overshooting by 1.5cm (a common PD tracking error) meant the box
   was lowered to 1.5cm above the table surface, not resting on it. When the
   welds released, the box fell the remaining 1.5cm and bounced.

2. After weld release, the arms were still at -CONTACT_PENETRATION (2cm inside
   the box surface). With the welds gone, the physical contact between the
   flat EE and the box surface pushed the box sideways. The arm was
   effectively shoving the box off the table.

**Fix:** Two changes: (1) Compute absolute placement z as
`TABLE_SURFACE_Z + BOX_HALF_EXTENTS[2]` (= 0.245m), the exact z where the
box center rests on the table. This eliminates cumulative errors from LIFT/CARRY
overshoot. (2) After weld release, immediately solve IK for retraction to
APPROACH_STANDOFF (10cm from box) and command the PD controller. This breaks
contact before the arms can push the box.

**Takeaway:** Always use absolute target positions for critical placements (do
not accumulate relative deltas through a chain of motions). After releasing
rigid attachment constraints, retract immediately to break contact. The arms
at contact distance will push the object.

---

### L12: Step-command PD produces jerky motion -- use ramped interpolation

**Symptom:** Commanding the final joint target instantly to the PD controller
produced fast, jerky arm movements with visible oscillation. The arms
accelerated aggressively, overshot, and rang before settling.

**Root cause:** A step command at high Kp (500) produces maximum torque
instantaneously. At t=0, the full position error (e.g., 2.5 rad) generates
500*2.5 = 1250 Nm, which clips to 150 Nm. The arm accelerates at the torque
limit, builds up velocity, then overshoots when the error sign reverses. The
Kd term eventually damps the oscillation, but the transient is visually
unpleasant and creates large contact forces if the arm hits anything.

**Fix:** Added 2-second smooth-step (3*alpha^2 - 2*alpha^3) interpolation
between the current and target joint configurations. Lower gains (Kp=300,
Kd=40) combined with the ramp produce slow, smooth motion. The smooth-step
has zero derivative at both endpoints, so the arm starts and stops gently.

The implementation in `_run_pd_until_settle()`:

```python
alpha = min(1.0, (step + 1) / n_ramp)
alpha_smooth = alpha**2 * (3 - 2*alpha)
q_target = q_start + alpha_smooth * (q_final - q_start)
```

**Takeaway:** For visually appealing and physically safe robot motion, always
ramp the PD target instead of step-commanding it. Smooth-step interpolation is
a simple, cheap, and effective choice. The combination of ramped targets and
moderate PD gains produces motion that looks natural and avoids the mechanical
stress of bang-bang-style torque profiles.

---

## Appendix: File Path Quick Reference

| File | Absolute Path |
|------|--------------|
| Scene MJCF | `lab-6-dual-arm/models/scene_dual.xml` |
| Left arm MJCF | `lab-6-dual-arm/models/ur5e_left.xml` |
| Right arm MJCF | `lab-6-dual-arm/models/ur5e_right.xml` |
| UR5e URDF | `lab-6-dual-arm/models/ur5e.urdf` |
| Common module | `lab-6-dual-arm/src/lab6_common.py` |
| Dual arm model | `lab-6-dual-arm/src/dual_arm_model.py` |
| PD controller | `lab-6-dual-arm/src/joint_pd_controller.py` |
| Grasp calculator | `lab-6-dual-arm/src/grasp_pose_calculator.py` |
| State machine | `lab-6-dual-arm/src/bimanual_state_machine.py` |
| Capstone demo | `lab-6-dual-arm/src/m5_capstone_demo.py` |
| Lessons learned | `lab-6-dual-arm/tasks/LESSONS.md` |
