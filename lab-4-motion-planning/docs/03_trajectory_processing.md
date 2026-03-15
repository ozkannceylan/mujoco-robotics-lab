# Lab 4: Trajectory Post-Processing

## Overview

The raw path produced by RRT or RRT* is a sequence of waypoints connected by
straight-line segments in C-space. This path has two problems:

1. **Too many waypoints.** The tree-growth process introduces unnecessary
   detours. Many intermediate waypoints can be removed if the direct segment
   between their neighbors is collision-free.
2. **No timing information.** The path is purely geometric -- it says nothing
   about how fast the robot should move. Executing it requires assigning
   velocities and accelerations that respect the robot's physical limits.

Lab 4 addresses these with a two-stage pipeline:

    Raw RRT path --> Shortcutting --> TOPP-RA --> Timed trajectory

---

## Stage 1: Path Shortcutting

### Algorithm

The shortcutting algorithm is a randomized greedy approach:

1. Copy the raw path into a working list.
2. For `max_iter` iterations:
   a. Pick two random indices `i` and `j` with `j >= i + 2` (at least one
      waypoint between them).
   b. Check if the straight-line segment from waypoint `i` to waypoint `j` is
      collision-free using `CollisionChecker.is_path_free`.
   c. If free, remove all waypoints between `i` and `j`.
3. Return the shortened path.

This is simple but effective. In practice, a path with 12 waypoints is often
reduced to just 2 (start and goal) when the direct connection happens to be
collision-free after a few attempts from different angles.

### API

```python
from trajectory_smoother import shortcut_path

shortened = shortcut_path(
    path=raw_path,
    collision_checker=cc,
    max_iter=200,     # Number of random shortcut attempts
    seed=42,          # For reproducibility
)
```

### Properties

- **Preserves endpoints:** The first and last waypoints are never removed.
- **Collision-safe:** Every edge in the shortened path passes `is_path_free`.
- **Monotonically shorter:** Each successful shortcut reduces waypoint count;
  the path never gets longer.
- **Idempotent for 2-point paths:** A path with only start and goal is
  returned unchanged.

### Typical Results

From the capstone demo with seed=42:

| Path          | Before | After | Reduction |
|---------------|--------|-------|-----------|
| RRT shortcut  | 12     | 2     | 83%       |
| RRT* shortcut | 8      | 2     | 75%       |

---

## Stage 2: TOPP-RA Time-Optimal Parameterization

### Background

Given a geometric path (a curve in C-space), Time-Optimal Path Parameterization
(TOPP) finds the fastest way to traverse that curve subject to constraints on
joint velocities and accelerations. The key insight is that for a fixed path
shape, the problem reduces to a 1D optimization over the path parameter `s`:

    q(t) = path(s(t))
    dq/dt = path'(s) * ds/dt
    d2q/dt2 = path''(s) * (ds/dt)^2 + path'(s) * d2s/dt2

TOPP-RA (Reachability Analysis) solves this efficiently by discretizing the
path parameter and computing reachable velocity intervals at each grid point.

### Implementation

Lab 4 uses the `toppra` Python library. The pipeline:

1. **Duplicate filtering:** Remove consecutive waypoints that are closer than
   1e-8 in L2 norm. This prevents degenerate spline knots.

2. **Arc-length parameterization:** Assign a path parameter `s` based on
   cumulative C-space arc length, normalized to [0, 1]:

       s_0 = 0
       s_i = s_{i-1} + ||q_i - q_{i-1}||
       s_i = s_i / s_total

3. **Spline interpolation:** Fit a cubic spline through the waypoints using
   `toppra.SplineInterpolator(ss, waypoints)`.

4. **Constraint setup:**

   ```python
   vlim = constraint.JointVelocityConstraint(vel_limits)
   alim = constraint.JointAccelerationConstraint(acc_limits)
   ```

   The UR5e limits used in Lab 4:
   - Velocity: [3.14, 3.14, 3.14, 6.28, 6.28, 6.28] rad/s
   - Acceleration: [8.0, 8.0, 8.0, 16.0, 16.0, 16.0] rad/s^2

5. **Solve:** Run the TOPP-RA algorithm with constant-acceleration
   parameterization:

   ```python
   instance = ta.algorithm.TOPPRA(
       [vlim, alim], path_spline, parametrizer="ParametrizeConstAccel"
   )
   traj = instance.compute_trajectory()
   ```

6. **Sample:** Evaluate the trajectory at 1 kHz (dt=0.001s) to produce arrays
   of times, positions, velocities, and accelerations.

### API

```python
from trajectory_smoother import parameterize_topp_ra
from lab4_common import VEL_LIMITS, ACC_LIMITS

times, positions, velocities, accelerations = parameterize_topp_ra(
    path=shortened_path,
    vel_limits=VEL_LIMITS,
    acc_limits=ACC_LIMITS,
    dt=0.001,
)
# times: (N,) seconds
# positions: (N, 6) joint angles
# velocities: (N, 6) joint velocities
# accelerations: (N, 6) joint accelerations
```

### Guarantees

The TOPP-RA output satisfies:
- `positions[0]` matches the first waypoint (atol 1e-4)
- `positions[-1]` matches the last waypoint (atol 1e-4)
- `|velocities| <= vel_limits` at every sample (within 0.05 rad/s tolerance)
- `|accelerations| <= acc_limits` at every sample (within 0.5 rad/s^2 tolerance)
- `times` is monotonically non-decreasing

These properties are enforced by the test suite in `tests/test_trajectory.py`.

### Edge Case: Degenerate Paths

If duplicate filtering reduces the path to a single waypoint, the function
returns a single-sample trajectory with zero velocity and acceleration. This
avoids crashes from degenerate splines.

---

## Full Pipeline Example

```python
from collision_checker import CollisionChecker
from rrt_planner import RRTStarPlanner
from trajectory_smoother import shortcut_path, parameterize_topp_ra
from lab4_common import Q_HOME, VEL_LIMITS, ACC_LIMITS

# Plan
cc = CollisionChecker()
planner = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.1, rewire_radius=1.0)
raw_path = planner.plan(Q_HOME, q_goal, max_iter=5000, rrt_star=True, seed=42)

# Shortcut
short_path = shortcut_path(raw_path, cc, max_iter=300, seed=42)

# Time-parameterize
times, q_traj, qd_traj, qdd_traj = parameterize_topp_ra(
    short_path, VEL_LIMITS, ACC_LIMITS
)

print(f"Raw: {len(raw_path)} waypoints")
print(f"Shortcut: {len(short_path)} waypoints")
print(f"Trajectory: {times[-1]:.3f}s, {len(times)} samples at 1 kHz")
```

---

## Test Coverage

Trajectory processing is tested in `tests/test_trajectory.py` with 14 tests
across three classes:

- `TestShortcutting` -- path shortening, endpoint preservation, collision
  freedom, idempotency on 2-point paths, seed reproducibility
- `TestTOPPRA` -- valid trajectory shape, endpoint accuracy, velocity and
  acceleration limit compliance, monotonic time
- `TestExecution` -- tracking error and final position accuracy (covered in
  the next document)
