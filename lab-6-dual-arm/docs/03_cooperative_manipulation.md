# Cooperative Manipulation: Impedance Control, Internal Forces, and State Machine

## Overview

Grasping and transporting a rigid object with two arms introduces a constraint
that is absent in single-arm manipulation: the internal force problem. When
both arms hold the same object, the difference between their commanded forces
creates an internal wrench that can crush or stretch the object without
producing any net motion. This document explains how dual impedance control
manages this, how a squeeze force maintains grip, and how the
`BimanualGraspStateMachine` orchestrates the full pick-carry-place pipeline.

---

## Dual Impedance Control

Each arm runs an independent impedance controller of the form:

```
tau = J^T * F + g(q)

where  F = K_p * e_x + K_d * (xd_des - J * qd)
       e_x = [pos_error (3); ori_error (3)]
```

The gains `K_p` (stiffness) and `K_d` (damping) are 6×6 diagonal matrices
covering three translational and three rotational degrees of freedom:

```python
@dataclass
class DualImpedanceGains:
    K_p: np.ndarray = field(
        default_factory=lambda: np.diag([400.0, 400.0, 400.0, 40.0, 40.0, 40.0])
    )
    K_d: np.ndarray = field(
        default_factory=lambda: np.diag([60.0, 60.0, 60.0, 6.0, 6.0, 6.0])
    )
    f_squeeze: float = 10.0   # N
```

Translational stiffness is 400 N/m; rotational stiffness is 40 Nm/rad. The
damping is tuned to give a damping ratio near critical for the closed-loop
system at typical UR5e link inertias.

### Orientation error

The orientation error is extracted from the skew-symmetric part of the
rotation error matrix, avoiding gimbal lock issues inherent in Euler angles:

```python
def orientation_error(R_des, R_cur) -> np.ndarray:
    R_err = R_des.T @ R_cur - R_cur.T @ R_des
    e = 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])
    return e
```

This is the vee-map applied to the skew-symmetric residual, giving a 3-vector
whose magnitude is zero at perfect alignment.

### Per-arm torque computation

```python
def _compute_arm_torque(self, q, qd, target, xd_des, side) -> np.ndarray:
    ee_pose = self.model.fk_left(q)         # or fk_right
    J       = self.model.jacobian_left(q)   # (6, 6), world frame
    g       = self.model.gravity_left(q)    # (6,)

    pos_err = target.translation - ee_pose.translation
    ori_err = orientation_error(target.rotation, ee_pose.rotation)
    error   = np.concatenate([pos_err, ori_err])   # (6,)

    v_cur   = J @ qd
    vel_err = (xd_des or np.zeros(6)) - v_cur

    F   = K_p @ error + K_d @ vel_err
    tau = J.T @ F + g
    return tau
```

Gravity compensation `g(q)` ensures the arms hold their pose even when the
impedance wrench is near zero, preventing gravity droop at low stiffness.

---

## Internal Force Control: Symmetric Gains and Squeeze Force

### Why symmetric gains prevent unintended internal forces

When both arms grip a rigid object and both are commanded toward the same
object-centric targets with **equal** gains, the impedance wrenches are
equal and opposite along the grasp axis. Their resultant net force on the
object is zero; only the symmetric (internal) component remains. This is the
key insight behind symmetric impedance control:

```
F_net      = F_left + F_right   →  object accelerates (desired)
F_internal = F_left - F_right   →  internal squeeze/stretch (controlled)
```

With identical `K_p` and `K_d` on both arms, any positional tracking error on
one arm generates exactly the same restoring force as the same error on the
other arm, keeping `F_internal` bounded without an explicit force control loop.

### Squeeze force along the grasp axis

To maintain a positive contact force during transport (preventing the object
from slipping out), a configurable squeeze force is added along the line
connecting the two end-effectors:

```python
def _compute_squeeze_torques(self, q_left, q_right):
    pos_left  = self.model.fk_left_pos(q_left)
    pos_right = self.model.fk_right_pos(q_right)

    diff      = pos_right - pos_left
    grasp_axis = diff / np.linalg.norm(diff)   # unit vector left→right

    F_squeeze_left  = np.zeros(6)
    F_squeeze_left[:3]  = +f_squeeze * grasp_axis   # left pushes toward right
    F_squeeze_right = np.zeros(6)
    F_squeeze_right[:3] = -f_squeeze * grasp_axis   # right pushes toward left

    tau_left  = J_left.T  @ F_squeeze_left
    tau_right = J_right.T @ F_squeeze_right
    return tau_left, tau_right
```

With `f_squeeze = 10 N` the squeeze torques are small relative to the main
impedance torques (~150–200 Nm peak) and are added only when `grasping=True`
is passed to `compute_dual_torques`.

### Complete torque computation

```python
def compute_dual_torques(
    self,
    q_left, qd_left, q_right, qd_right,
    target_left, target_right,
    xd_left=None, xd_right=None,
    grasping: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    tau_left  = self._compute_arm_torque(q_left,  qd_left,  target_left,  xd_left,  "left")
    tau_right = self._compute_arm_torque(q_right, qd_right, target_right, xd_right, "right")

    if grasping and self.gains.f_squeeze > 0:
        sq_l, sq_r = self._compute_squeeze_torques(q_left, q_right)
        tau_left  += sq_l
        tau_right += sq_r

    return clip_torques(tau_left), clip_torques(tau_right)
```

Torques are clipped to hardware limits (defined in `lab6_common.TORQUE_LIMITS`)
before being sent to MuJoCo actuators.

---

## BimanualGraspStateMachine

The state machine orchestrates the full cooperative pick-carry-place task in
ten states. It owns a `DualImpedanceController` internally and is the sole
source of joint torques throughout the task.

### States and transitions

```
IDLE
  │ (immediate on first step)
  ▼
APPROACH  ── both EEs within tolerance of pre-grasp poses ──▶ PRE_GRASP
  │
PRE_GRASP  ── both EEs at grasp contact positions ──▶ (activate welds) ──▶ GRASP
  │
GRASP  ── elapsed ≥ settle_time (0.5 s) ──▶ LIFT
  │
LIFT  ── both EEs at lift height ──▶ CARRY
  │
CARRY  ── both EEs at carry (transport) positions ──▶ LOWER
  │
LOWER  ── both EEs at place height ──▶ (deactivate welds) ──▶ RELEASE
  │
RELEASE  ── elapsed ≥ settle_time ──▶ RETREAT
  │
RETREAT  ── both EEs at retreat positions ──▶ DONE
```

The `grasping` flag passed to the impedance controller is `True` during GRASP,
LIFT, CARRY, and LOWER states, enabling the squeeze force. It is `False`
during all other states.

### Weld constraint management

MuJoCo equality (weld) constraints `left_grasp` and `right_grasp` are toggled
via `mj_model.eq_active`:

```python
def _activate_welds(self, mj_model, mj_data) -> None:
    for name in ("left_grasp", "right_grasp"):
        eq_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id >= 0:
            mj_model.eq_active[eq_id] = 1

def _deactivate_welds(self, mj_model, mj_data) -> None:
    for name in ("left_grasp", "right_grasp"):
        eq_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id >= 0:
            mj_model.eq_active[eq_id] = 0
```

Welds are activated at the PRE_GRASP → GRASP transition (after both EEs reach
contact positions) and deactivated at the LOWER → RELEASE transition (after
the object has been placed on the table). The `settle_time` pauses at GRASP
and RELEASE give MuJoCo's constraint solver time to settle before the next
motion begins.

### Target geometry

The state machine precomputes all fixed SE3 targets from the task configuration
at construction time. Grasp geometry uses side-approach along world X:

```
Left arm (from -X side):
  approach_pos_left = [obj_x - grasp_offset_x - clearance,  obj_y, obj_z]
  grasp_pos_left    = [obj_x - grasp_offset_x,              obj_y, obj_z]

Right arm (from +X side):
  approach_pos_right = [obj_x + grasp_offset_x + clearance, obj_y, obj_z]
  grasp_pos_right    = [obj_x + grasp_offset_x,             obj_y, obj_z]
```

EE orientations are fixed: the left EE's Z-axis points +X (toward the object
centre from the left), and the right EE's Z-axis points -X. These are constant
for all states so the approach remains straight.

### Step interface

The caller invokes `step()` at each simulation timestep with the current joint
states. The method evaluates transition conditions first, then computes torques:

```python
def step(
    self,
    q_left, qd_left, q_right, qd_right,
    t: float,
    mj_model=None, mj_data=None,
) -> tuple[np.ndarray, np.ndarray]:
    self._maybe_transition(q_left, q_right, t, mj_model, mj_data)
    tgt_l, tgt_r, grasping = self._targets_for_state(self.state, q_left, q_right)
    return self.controller.compute_dual_torques(
        q_left, qd_left, q_right, qd_right,
        tgt_l, tgt_r,
        grasping=grasping,
    )
```

---

## The Cooperative Carry Pipeline

A complete pick-carry-place task uses the following sequence:

| Phase | Duration | Key action |
|-------|----------|------------|
| APPROACH | ~2–3 s | Both arms move from home to pre-grasp clearance |
| PRE_GRASP | ~1–2 s | Arms close in to contact positions |
| GRASP | 0.5 s | Welds activated; impedance holds object firmly |
| LIFT | ~1.5 s | Object lifted 15 cm above table |
| CARRY | ~2 s | Object transported 30 cm laterally |
| LOWER | ~1.5 s | Object lowered to target table position |
| RELEASE | 0.5 s | Welds deactivated; object rests on table |
| RETREAT | ~2 s | Arms pull back to retreat clearance |

State transitions are driven by position tolerance checks (default 1 cm) so
actual durations depend on controller convergence speed.

### Force balancing during transport

During LIFT/CARRY/LOWER, the object experiences:

```
F_gravity    = m * g = 2 kg * 9.81 = ~19.6 N (downward)
F_left_Z     = impedance term resisting downward object displacement
F_right_Z    = impedance term (symmetric to left)
F_constraint = weld constraint force (handles residual error)
```

With symmetric gains, each arm contributes approximately half of the vertical
load (`~9.8 N` each). The weld constraint absorbs any residual imbalance during
transient phases.

---

## Usage Example

```python
from dual_arm_model import DualArmModel
from bimanual_grasp import BimanualGraspStateMachine, BimanualTaskConfig
import numpy as np

dual_model = DualArmModel()

config = BimanualTaskConfig(
    object_pos   = np.array([0.5,  0.0, 0.85]),
    target_pos   = np.array([0.5,  0.3, 0.85]),
    lift_height  = 0.15,
    approach_clearance = 0.08,
    grasp_offset_x     = 0.18,
)

sm = BimanualGraspStateMachine(dual_model, config)

# Simulation loop
while not sm.is_done:
    q_left  = mj_data.qpos[LEFT_JOINT_SLICE]
    qd_left = mj_data.qvel[LEFT_JOINT_SLICE]
    q_right  = mj_data.qpos[RIGHT_JOINT_SLICE]
    qd_right = mj_data.qvel[RIGHT_JOINT_SLICE]

    tau_left, tau_right = sm.step(
        q_left, qd_left, q_right, qd_right,
        t=mj_data.time,
        mj_model=mj_model,
        mj_data=mj_data,
    )

    mj_data.ctrl[LEFT_CTRL_SLICE]  = tau_left
    mj_data.ctrl[RIGHT_CTRL_SLICE] = tau_right
    mujoco.mj_step(mj_model, mj_data)

print("State log:", sm.get_state_log())
```

The state log records `(time, state_name)` for every transition, which is
useful for debugging and plotting task timing.
