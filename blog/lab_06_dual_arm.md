# Lab 6: From One Arm to Two — The Coordination Challenge

## The Naive Mental Model Is Wrong

When I started planning Lab 6, my first instinct was: "dual-arm is just two single-arm problems running in parallel." The kinematics are the same. The controller is the same. Just spin up two instances and let them run.

That mental model is wrong, and understanding why it's wrong is the entire point of this lab.

Two arms operating independently in a shared workspace are not twice as capable — they're a collision waiting to happen. Synchronization is undefined. If one arm is slower, the other has already finished and is holding its side of the object at the wrong height. Internal forces are unconstrained. If both arms grip the same object but disagree by even a few millimeters on the grasp pose, the object experiences a large internal force that either breaks the grasp or deforms the object. None of these problems exist when each arm operates alone.

Lab 6 is about solving all of them.

## The Architecture: Five Modules

Before diving into the problems, it's worth grounding the structure. The lab builds five modules that stack on top of each other:

```
BimanualGraspStateMachine   ← orchestrates the full pipeline
       |
CoordinatedPlanner          ← generates time-synchronized trajectories
       |
DualArmModel                ← FK, Jacobians, IK for both arms in world frame
       |
DualCollisionChecker        ← arm-arm + arm-environment collision queries
       |
DualImpedanceController     ← computes torques for both arms simultaneously
```

The data flow is: the state machine decides what phase the robot is in, the planner produces synchronized joint trajectories for both arms, the collision checker validates them, and the impedance controller tracks them at 1 kHz in MuJoCo. Torques are written to `mj_data.ctrl[:6]` for the left arm and `mj_data.ctrl[6:12]` for the right.

## Problem 1: Shared Workspace and Arm-Arm Collision

The two UR5e arms face each other across a shared table, separated by about one meter with the right arm rotated 180 degrees about the world Z axis. This placement puts both arms in constant reach of the same central workspace — exactly where you want for bimanual tasks, and exactly where collisions happen.

Self-collision checking for each arm in isolation was already solved in Lab 4. The new problem is cross-arm collision: every geometry on the left arm can potentially hit every geometry on the right arm. The `DualCollisionChecker` handles four categories:

1. Left arm self-collision (adjacency-filtered)
2. Right arm self-collision (adjacency-filtered)
3. Left vs. right cross-arm collision
4. Each arm vs. the environment (table surface)

The cross-arm check is the tricky one. Each arm has its own Pinocchio collision model, and those models don't know about each other. The solution is to use HPP-FCL directly: after running each arm's FK in its own model, extract the world-frame geometry placements and query HPP-FCL collision pairs across the two models. The key utility is a simple conversion function:

```python
def _se3_to_hppfcl(tf: pin.SE3) -> hppfcl.Transform3f:
    return hppfcl.Transform3f(tf.rotation, tf.translation)
```

With both arm geometries expressed in the same world frame, HPP-FCL's GJK/EPA distance queries work correctly across models. The `get_min_distance(q_left, q_right)` method returns the closest approach distance between any left-right geometry pair, which is monitored continuously during motion.

The base transforms are critical here. The left arm's base is at the world origin; the right arm's base is at `[1.0, 0, 0]` with a 180-degree yaw rotation. These offsets are encoded as Pinocchio `SE3` transforms applied programmatically rather than baked into the URDFs, which means both arms can share the same URDF file with only the base transform distinguishing them.

## Problem 2: The Object-Centric Frame Insight

Here is the central conceptual shift of Lab 6.

When two arms carry an object together, the natural planning frame is the *object*, not either arm. If you plan for each arm independently — "left arm go to this pose, right arm go to that pose" — you're leaving the coordination implicit. The consistency of the grasp is not guaranteed; it depends on both IK solutions being correct and the object being exactly where you modeled it.

The better approach: define all coordination in terms of what the *object* should do, then derive arm targets from that.

The `ObjectFrame` class captures this abstraction:

```python
@dataclass
class ObjectFrame:
    pose: pin.SE3                   # object pose in world frame
    grasp_offset_left: pin.SE3      # left EE relative to object
    grasp_offset_right: pin.SE3     # right EE relative to object

    def get_left_target(self) -> pin.SE3:
        return self.pose * self.grasp_offset_left

    def get_right_target(self) -> pin.SE3:
        return self.pose * self.grasp_offset_right
```

The grasp offsets are fixed at grasp time and encode the desired relative pose between each end-effector and the object. Once those offsets are established, moving the object frame — up for a lift, sideways for a carry — automatically produces correct targets for both arms without any further coordination logic. The arms stay in formation by construction.

This pattern also makes the code robust to scene changes. If you want the arms to carry the object to a different destination, you only change the object trajectory. You don't touch the grasp geometry or the IK calls.

## Problem 3: Synchronized Timing

Two arms moving toward independent goals have independent motion times. If you let each arm plan its own trajectory, the faster arm arrives first and either waits at the goal (which causes it to drift under gravity) or keeps moving past the goal (which destroys the grasp geometry).

The `CoordinatedPlanner` enforces synchronized arrival with a simple rule: **the slower arm dictates the duration**. When `plan_synchronized_linear()` is called with targets for both arms, it computes the minimum time needed for each arm to reach its target under velocity limits, then sets the trajectory duration to the maximum of the two. Both trajectories are then parameterized to use that shared duration, so they arrive at exactly the same timestep.

The `SynchronizedTrajectory` dataclass makes the constraint explicit:

```python
@dataclass
class SynchronizedTrajectory:
    timestamps: np.ndarray    # (T,) — shared by both arms
    q_left: np.ndarray        # (T, 6)
    qd_left: np.ndarray       # (T, 6)
    q_right: np.ndarray       # (T, 6)
    qd_right: np.ndarray      # (T, 6)
```

There is only one `timestamps` array. Both arms index into it together. You cannot accidentally run one arm on a different time base.

The planner supports three coordination modes:

**Synchronized linear**: Both arms interpolate independently from start to goal, with shared duration. Use this for approach and retreat where arms aren't yet holding the object.

**Master-slave**: The master arm follows a given trajectory. The slave derives its target at each timestep by computing `slave_target = object_pose * grasp_offset_slave`, where `object_pose` comes from the master's current EE pose. The slave tracks the master's motion implicitly. Use this when one arm is the "leader" — for instance, when the left arm is executing a pre-planned path and the right arm needs to follow.

**Symmetric**: Both arms derive targets from a planned object trajectory. The object frame moves along a specified path, and both arm targets follow. Use this for all cooperative carry phases.

## Problem 4: Internal Force Control

When two arms hold a rigid object, they form a closed kinematic chain. Any discrepancy in the targets — even a fraction of a millimeter — creates an internal force that the object must bear. Too little internal force and the grasp is loose. Too much and you're fighting the physics simulator.

The elegant solution is symmetric impedance: give both arms identical gains. The `DualImpedanceGains` dataclass uses the same `K_p` and `K_d` for both arms:

```python
@dataclass
class DualImpedanceGains:
    K_p: np.ndarray = field(
        default_factory=lambda: np.diag([400.0, 400.0, 400.0, 40.0, 40.0, 40.0])
    )
    K_d: np.ndarray = field(
        default_factory=lambda: np.diag([60.0, 60.0, 60.0, 6.0, 6.0, 6.0])
    )
    f_squeeze: float = 10.0  # N, internal grasp force
```

With symmetric gains, if both arms are displaced from their targets by the same error (which is what happens when the object moves), both arms produce the same restoring force. The net force on the object is zero. Internal forces balance naturally without any explicit force sensor or force control loop. This is the primary reason for symmetric gains — it's not just elegance, it's stability.

When actively grasping, the controller adds a small squeeze force along the axis connecting the two end-effectors. This is the `f_squeeze` term. The grasp axis is recomputed at every timestep from the current EE positions:

```python
grasp_axis = (pos_right - pos_left) / np.linalg.norm(pos_right - pos_left)
# Left arm pushes toward right
F_squeeze_left[:3] = f_squeeze * grasp_axis
# Right arm pushes toward left
F_squeeze_right[:3] = -f_squeeze * grasp_axis
```

The squeeze is applied through `J^T @ F_squeeze`, so it maps naturally to joint torques via the Jacobian transpose. A 10 N squeeze at the end-effector is small enough not to deform the simulated object but large enough to keep the grasp stable during transport.

The full per-arm impedance law is `tau = J^T * (K_p * error + K_d * vel_error) + g(q)`. The gravity term `g(q)` comes from Pinocchio's RNEA and is essential — without it, both arms droop under their own weight and the tracking error feeds back into the internal force calculation.

## Problem 5: The Bimanual State Machine

Coordinating two arms through a full pick-carry-place pipeline requires managing ten distinct phases, each with its own preconditions, targets, and controller configuration. The `BimanualGraspStateMachine` encodes this as an enum state machine:

```
IDLE → APPROACH → PRE_GRASP → GRASP → LIFT → CARRY → LOWER → RELEASE → RETREAT → DONE
```

Each transition has a concrete precondition:

- `APPROACH → PRE_GRASP`: Both EEs within 1 cm of the pre-grasp positions (the "clearance" positions 8 cm out from the object sides)
- `PRE_GRASP → GRASP`: Both EEs within 1 cm of the actual grasp contact points; MuJoCo weld constraints activated
- `GRASP → LIFT`: 500 ms settle time after weld activation (lets the physics constraint stabilize)
- `LIFT → CARRY`: Both EEs within 1 cm of lift height targets
- `CARRY → LOWER`: Both EEs within 1 cm of lateral transport targets
- `LOWER → RELEASE`: Both EEs within 1 cm of placement height; weld constraints deactivated
- `RELEASE → RETREAT`: 500 ms settle time after release
- `RETREAT → DONE`: Both EEs within 1 cm of retreat clearance positions

The `grasping` flag — passed from `_targets_for_state()` to the impedance controller — is `True` only during `GRASP`, `LIFT`, `CARRY`, and `LOWER`. This enables the squeeze force precisely when the arms are holding the object and disables it during approach and retreat, where the squeeze would push the arms into the table or each other.

The main loop is clean because all the complexity is encapsulated:

```python
tau_left, tau_right = state_machine.step(
    q_left, qd_left, q_right, qd_right, t, mj_model, mj_data
)
mj_data.ctrl[LEFT_CTRL_SLICE] = tau_left
mj_data.ctrl[RIGHT_CTRL_SLICE] = tau_right
mujoco.mj_step(mj_model, mj_data)
```

## Problem 6: Rigid Grasp Simulation

Lab 5 introduced MuJoCo's weld equality constraint for simulating a rigid grasp without modeling friction-limited contact mechanics. Lab 6 reuses this pattern for both arms simultaneously.

The scene XML defines two weld constraints, initially disabled:

```xml
<equality>
  <weld name="left_grasp"  body1="left_robotiq_85" body2="box" active="false"/>
  <weld name="right_grasp" body1="right_robotiq_85" body2="box" active="false"/>
</equality>
```

At the `PRE_GRASP → GRASP` transition, the state machine activates both constraints by name:

```python
for name in ("left_grasp", "right_grasp"):
    eq_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    mj_model.eq_active[eq_id] = 1
```

This is sufficient to rigidly attach the box to both end-effectors simultaneously. From this point, moving the arms moves the box. The weld encodes the relative pose between the EE and the box at activation time, so any offset from imperfect grasp alignment is automatically compensated.

At the `LOWER → RELEASE` transition, both welds are deactivated and the box becomes a free body resting on the table. The arms then retreat without disturbing the placed object.

## Results

The full cooperative carry pipeline executes reliably across the ten-state sequence. Key metrics:

- **EE tracking error**: RMS < 5 mm for both arms throughout the task
- **Object rotation during carry**: < 5 degrees (within the success criterion)
- **Arm-arm minimum distance**: Monitored at every timestep; remains above 8 cm during all phases
- **Squeeze force**: Steady at ~10 N during grasp/lift/carry; drops to zero at release

Plots of EE tracking error, internal force, object pose error, and arm-arm minimum clearance are saved to `media/`.

![Cooperative carry: both arms lifting and transporting the box](../lab-6-dual-arm/media/cooperative_carry.gif)

The most revealing plot is the arm-arm minimum distance over time. During the approach phase, both arms converge from home positions on opposite sides of the workspace, and the clearance drops to its minimum around the time of contact. The collision checker would flag any configuration below 2 cm; in practice the minimum observed was well above that threshold.

## What Failed First

**Grasp target geometry took iteration.** The initial plan had the left arm approach from the -X side with an identity orientation at the contact point. The IK solver found configurations that put the arm directly above the box, not beside it, because the identity orientation was ambiguous. The fix was explicit rotation matrices for the two grasp directions — `_R_LEFT_GRASP` for the arm approaching in the +X direction, `_R_RIGHT_GRASP` for the arm approaching in the -X direction. These are precomputed once and reused throughout.

**Simultaneous weld activation causes a transient.** Activating two weld constraints at the same instant creates a brief physics transient as the solver reconciles the new constraints. The 500 ms settle time in the GRASP state absorbs this. Skipping the settle time causes the box to jerk at lift initiation.

**The right arm's mirrored home configuration.** The left arm's home configuration puts it in a comfortable extended pose on the left side. The right arm's home is the mirror image — joints at negated angles — because the arm is physically rotated 180 degrees. Getting this wrong means the "home" pose is actually a folded-in configuration on the wrong side of the workspace. The `Q_HOME_RIGHT` constant in `lab6_common.py` encodes the correct mirrored values.

## Dependencies on Previous Labs

Lab 6 is self-contained but draws on four prior patterns:

- **Lab 3** (Dynamics): The per-arm impedance law `tau = J^T * F + g(q)`, gravity compensation via RNEA, and the orientation error function using `vee(R_d^T R - R^T R_d)`
- **Lab 4** (Planning): The HPP-FCL collision checking infrastructure, adjacency gap filtering for self-collision, and the `is_path_free()` interpolated check pattern
- **Lab 5** (Grasping): The enum-based state machine pattern, weld constraint activation/deactivation, and the general structure of `step()` returning torques

None of these are imported directly — each pattern is reimplemented with dual-arm extensions to keep the lab self-contained and independently runnable.

## What's Next

Lab 7 is locomotion. The challenge there is fundamentally different: instead of two arms that must coordinate, we have a floating base that must stay balanced while the legs execute a gait. The tools carry over — Pinocchio for RNEA and centroidal dynamics, MuJoCo for contact simulation — but the problem framing shifts from end-effector coordination to whole-body stability.

The dual-arm lab is worth revisiting later when we reach Lab 8 (whole-body loco-manipulation), where the arms coordinate with the legs and the base all at once. The object-centric frame abstraction and the bimanual state machine will be directly reusable. The internal force balancing problem doesn't go away — it just gets embedded in a larger coordination problem that also includes a moving base.
