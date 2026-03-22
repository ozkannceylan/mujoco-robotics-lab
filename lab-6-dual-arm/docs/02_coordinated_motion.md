# Coordinated Motion: Object-Centric Planning and Synchronized Trajectories

## Overview

Moving two arms at once requires more than running two independent trajectory
planners in parallel. The arms share a workspace, manipulate a common object,
and must arrive at their targets at exactly the same time. This document
covers the object-centric frame abstraction, the three coordination modes
implemented in `CoordinatedPlanner`, SE3 interpolation, and how IK is
integrated into the planning pipeline.

---

## Object-Centric Frame Abstraction

All cooperative motion is planned relative to the object, not relative to
individual arm bases. The `ObjectFrame` dataclass encapsulates this:

```python
@dataclass
class ObjectFrame:
    pose: pin.SE3             # object pose in world frame
    grasp_offset_left: pin.SE3   # left EE pose expressed in object frame
    grasp_offset_right: pin.SE3  # right EE pose expressed in object frame
```

Given any new object pose, each arm's target in the world frame is:

```
arm_target_world = object_pose_world * grasp_offset_arm
```

This is implemented as:

```python
def get_left_target(self) -> pin.SE3:
    return self.pose * self.grasp_offset_left

def get_right_target(self) -> pin.SE3:
    return self.pose * self.grasp_offset_right
```

Grasp offsets are captured from measured EE poses at grasp time:

```python
@classmethod
def from_ee_poses(cls, object_pose, left_ee, right_ee) -> "ObjectFrame":
    offset_left  = object_pose.inverse() * left_ee
    offset_right = object_pose.inverse() * right_ee
    return cls(pose=object_pose,
               grasp_offset_left=offset_left,
               grasp_offset_right=offset_right)
```

The `moved_to()` method produces a new `ObjectFrame` at a different object
pose while keeping the grasp offsets unchanged — the key operation used during
transport:

```python
def moved_to(self, new_pose: pin.SE3) -> "ObjectFrame":
    return ObjectFrame(
        pose=new_pose,
        grasp_offset_left=self.grasp_offset_left,
        grasp_offset_right=self.grasp_offset_right,
    )
```

---

## SynchronizedTrajectory

All three planning modes produce a `SynchronizedTrajectory` — a dataclass that
guarantees both arms share the same time axis:

```python
@dataclass
class SynchronizedTrajectory:
    timestamps: np.ndarray   # (T,)   — seconds
    q_left:     np.ndarray   # (T, 6) — left joint positions
    qd_left:    np.ndarray   # (T, 6) — left joint velocities
    q_right:    np.ndarray   # (T, 6)
    qd_right:   np.ndarray   # (T, 6)
```

Shape validation runs at construction time (via `__post_init__`). Both arms
are guaranteed to have the same number of timesteps so they can be indexed
together in the control loop:

```python
for i in range(traj.n_steps):
    tau_l, tau_r = controller.compute_dual_torques(
        traj.q_left[i], traj.qd_left[i],
        traj.q_right[i], traj.qd_right[i],
        ...
    )
```

---

## SE3 Interpolation: Linear Position + Quaternion SLERP

Task-space planning requires smoothly interpolating between two SE3 poses.
Translation is linearly interpolated; rotation uses spherical linear
interpolation (SLERP) on unit quaternions:

```python
@staticmethod
def _interpolate_se3(start: pin.SE3, end: pin.SE3, alpha: float) -> pin.SE3:
    # Linear position
    pos = (1.0 - alpha) * start.translation + alpha * end.translation

    # Quaternion SLERP (shortest-path)
    q1 = pin.Quaternion(start.rotation)
    q2 = pin.Quaternion(end.rotation)
    if q1.dot(q2) < 0.0:
        q2 = pin.Quaternion(-q2.coeffs())   # ensure shortest arc
    q_interp = q1.slerp(alpha, q2)
    R_interp  = q_interp.toRotationMatrix()

    return pin.SE3(R_interp, pos)
```

The dot-product sign check before SLERP prevents the algorithm from taking the
long way around the sphere. With `alpha` in `[0, 1]`, the interpolated pose at
`alpha=0` equals `start` exactly and at `alpha=1` equals `end` exactly.

---

## Coordination Mode 1: Synchronized Linear

Both arms move simultaneously from their current poses to target poses in
task space. IK is solved at 20 sparse waypoints along the interpolated SE3
path; the results are then upsampled to the simulation timestep `DT`:

```python
def plan_synchronized_linear(
    self,
    target_left: pin.SE3,
    target_right: pin.SE3,
    q_left_init: np.ndarray,
    q_right_init: np.ndarray,
    duration: float | None = None,
) -> SynchronizedTrajectory:
```

Planning steps:

1. Compute start poses via `fk_left(q_left_init)` and `fk_right(q_right_init)`.
2. Interpolate 20 SE3 waypoints along the paths for both arms simultaneously.
3. Solve IK at each waypoint using sequential warm-starting (each waypoint's
   IK seeds from the previous solution).
4. Auto-compute `duration` from the accumulated joint-space path length divided
   by 70% of the joint velocity limits (safety margin).
5. Upsample the sparse waypoints to a dense DT-rate trajectory via
   `np.interp` per joint.
6. Compute velocities via `np.gradient` over the dense positions.

Auto-duration formula:

```
duration = max_over_joints( total_displacement / (VEL_LIMITS * 0.7) )
```

This ensures no joint exceeds 70% of its velocity limit at any point in the
trajectory.

---

## Coordination Mode 2: Master-Slave

One arm (the master) follows a prescribed sequence of waypoints. The other arm
(the slave) automatically tracks the rigid relationship between master and
object:

```
object_pose_new  = master_waypoint * offset_master^{-1}
slave_target     = object_pose_new * offset_slave
```

```python
def plan_master_slave(
    self,
    master_waypoints_se3: list[pin.SE3],
    object_frame: ObjectFrame,
    q_left_init: np.ndarray,
    q_right_init: np.ndarray,
    master: str = "left",
    duration: float | None = None,
) -> SynchronizedTrajectory:
```

The planner prepends the current FK pose of the master arm as waypoint[0] so
the generated trajectory is always smooth from the current configuration. At
each waypoint:

1. Solve master IK toward `master_waypoints_se3[i]`.
2. Reconstruct the object pose: `obj_pose = master_wp * offset_master.inverse()`.
3. Derive the slave target: `slave_target = obj_pose * offset_slave`.
4. Solve slave IK toward `slave_target`.

If IK fails at any waypoint, the previous solution is held and a warning is
issued. A `RuntimeError` is raised if more than half of all waypoints fail.

**Invariant verified by tests:** the relative pose between master EE and object
remains constant across all waypoints (error < 2 mm).

---

## Coordination Mode 3: Symmetric

Both arms simultaneously derive their targets from an object trajectory.
Neither arm is the master — both are equal participants tracking the moving
object frame:

```python
def plan_symmetric(
    self,
    object_waypoints_se3: list[pin.SE3],
    object_frame: ObjectFrame,
    q_left_init: np.ndarray,
    q_right_init: np.ndarray,
    duration: float | None = None,
) -> SynchronizedTrajectory:
```

At each object pose along the interpolated path:

```python
obj_pose     = _interpolate_se3(obj_start, obj_end, alpha)
left_target  = obj_pose * object_frame.grasp_offset_left
right_target = obj_pose * object_frame.grasp_offset_right
```

IK is solved independently for both arms. The object path is split into
segments; 20 waypoints are distributed evenly across all segments. Waypoints
at segment boundaries are not duplicated (the inner loop starts at index 1 for
subsequent segments).

This mode is used for the LIFT, CARRY, and LOWER phases of the cooperative
carry pipeline, where the object trajectory is the authoritative source of truth
and both arms follow.

---

## IK Integration

All three modes solve IK via the damped least-squares (Levenberg-Marquardt)
solver in `DualArmModel`. The critical implementation detail is sequential
warm-starting:

```python
# Each waypoint's IK is seeded from the previous solution
q_l = self.model.ik_left(left_target, q_left_wp[i - 1])
```

This dramatically improves convergence along smooth trajectories because
consecutive poses are close in configuration space. Without warm-starting,
the solver would have to converge from the initial configuration at every
waypoint and would fail more often near singularities or joint limits.

The IK solver itself uses damped least squares:

```
J^T J dq = J^T e     →    (J^T J + λI) dq = J^T e

q ← q + α * dq
q ← clip(q, q_min, q_max)
```

With damping factor `λ = 1e-6` and step size `α = 0.5`. The 6D error vector
`e` is computed from `pin.log6(current.inverse() * target).vector`, which
captures both translational and rotational residuals in a single vector.

---

## Auto-Duration and Velocity Safety

When `duration=None`, the planner computes the minimum safe duration:

```python
@staticmethod
def _auto_duration(q_left_disp, q_right_disp) -> float:
    max_left  = np.max(np.abs(q_left_disp)  / (VEL_LIMITS * 0.7))
    max_right = np.max(np.abs(q_right_disp) / (VEL_LIMITS * 0.7))
    return float(max(max_left, max_right, 0.1))
```

Both arms are considered: the slower arm sets the duration so both can
complete their motion simultaneously. The 0.1 s floor prevents degenerate
near-zero durations for very small movements.

---

## Usage Example

```python
from dual_arm_model import DualArmModel, ObjectFrame
from coordinated_planner import CoordinatedPlanner
import pinocchio as pin
import numpy as np

model   = DualArmModel()
planner = CoordinatedPlanner(model)

# Symmetric mode: carry object from start to goal
object_start = pin.SE3(np.eye(3), np.array([0.5, 0.0, 0.85]))
object_goal  = pin.SE3(np.eye(3), np.array([0.5, 0.3, 0.85]))

obj_frame = ObjectFrame.from_ee_poses(
    object_start,
    left_ee  = model.fk_left(q_left_current),
    right_ee = model.fk_right(q_right_current),
)

traj = planner.plan_symmetric(
    object_waypoints_se3=[object_goal],
    object_frame=obj_frame,
    q_left_init=q_left_current,
    q_right_init=q_right_current,
)

print(f"Trajectory duration: {traj.duration:.2f} s, steps: {traj.n_steps}")
```
