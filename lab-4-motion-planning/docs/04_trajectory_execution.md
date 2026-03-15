# Lab 4: Trajectory Execution

## Overview

The final stage of the motion planning pipeline takes the timed trajectory from
TOPP-RA and executes it on the UR5e in MuJoCo simulation. The robot is
torque-controlled, meaning we must compute joint torques at every timestep.
Lab 4 uses joint-space impedance control with gravity compensation from
Pinocchio.

---

## Control Law

The torque applied to each joint at every simulation step is:

    tau = Kp * (q_d - q) + Kd * (qd_d - qd) + g(q)

Where:

| Symbol   | Description                                    |
|----------|------------------------------------------------|
| `q_d`    | Desired joint position (from trajectory)       |
| `q`      | Actual joint position (from MuJoCo)            |
| `qd_d`   | Desired joint velocity (from trajectory)       |
| `qd`     | Actual joint velocity (from MuJoCo)            |
| `Kp`     | Proportional gain (stiffness), default 400     |
| `Kd`     | Derivative gain (damping), default 40          |
| `g(q)`   | Gravity compensation torque from Pinocchio     |

This is a PD controller with feedforward gravity compensation. The Kp and Kd
gains form a critically damped system (Kd = 2 * sqrt(Kp) = 40 for Kp = 400),
which provides fast tracking without oscillation.

### Gravity Compensation

The gravity term `g(q)` is computed via Pinocchio's RNEA (Recursive
Newton-Euler Algorithm) with zero velocity and zero acceleration:

```python
pin.computeGeneralizedGravity(pin_model, pin_data, q)
g = pin_data.g.copy()
```

This is equivalent to calling `pin.rnea(model, data, q, zero_vel, zero_acc)`.
Without this term, the controller would need to "fight" gravity through the
proportional gain alone, causing steady-state droop proportional to
`m*g*l / Kp`.

### Torque Clipping

After computing the control torque, it is clipped to the UR5e joint torque
limits before being applied:

    Joints 1-3: +/- 150 Nm
    Joints 4-6: +/- 28 Nm

```python
tau = clip_torques(tau)  # np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
mj_data.ctrl[:NUM_JOINTS] = tau
```

---

## Trajectory Interpolation

The TOPP-RA trajectory is sampled at 1 kHz, matching the MuJoCo simulation
timestep (dt = 0.001s). At each simulation step, the desired position and
velocity are obtained by linear interpolation:

```python
q_d = np.array([np.interp(t, times, q_traj[:, j]) for j in range(6)])
qd_d = np.array([np.interp(t, times, qd_traj[:, j]) for j in range(6)])
```

After the trajectory ends, the controller holds the final configuration with
zero desired velocity for an additional 0.5 seconds of settling time. This
allows the PD controller to damp out any residual tracking error.

---

## Execution Pipeline

```python
from trajectory_executor import execute_trajectory

result = execute_trajectory(
    times=times,        # (N,) time array from TOPP-RA
    q_traj=positions,   # (N, 6) joint positions
    qd_traj=velocities, # (N, 6) joint velocities
    Kp=400.0,
    Kd=40.0,
)
```

The function returns a dictionary with logged data:

| Key          | Shape      | Description                            |
|--------------|------------|----------------------------------------|
| `time`       | (M,)       | Simulation timestamps                  |
| `q_actual`   | (M, 6)     | Measured joint positions from MuJoCo   |
| `q_desired`  | (M, 6)     | Commanded joint positions              |
| `tau`        | (M, 6)     | Applied joint torques                  |
| `ee_pos`     | (M, 3)     | End-effector Cartesian position        |

Here M = total simulation steps = (trajectory_duration + 0.5s) / dt.

---

## Architecture: Pinocchio Brain, MuJoCo Body

This execution loop demonstrates the dual-library architecture used throughout
the lab series:

- **Pinocchio** computes the analytical gravity vector `g(q)` via RNEA. This
  is a pure computation with no simulation state.
- **MuJoCo** provides the physics simulation: it steps the dynamics, reports
  `qpos` and `qvel`, handles contacts, and renders.

The two never duplicate work. Pinocchio handles the math; MuJoCo handles
the physics.

```
Each timestep:
  q, qd  <--  MuJoCo (read state)
  g(q)   <--  Pinocchio RNEA (compute gravity)
  tau    =   Kp*(q_d - q) + Kd*(qd_d - qd) + g(q)
  tau    -->  MuJoCo ctrl (apply torques)
  mj_step -> MuJoCo (advance simulation)
```

---

## Tracking Performance

### Metrics

Two metrics characterize execution quality:

- **RMS tracking error:** `sqrt(mean((q_actual - q_desired)^2))` over all
  joints and all timesteps. This captures the average deviation during motion.
- **Final position error:** `||q_actual[-1] - q_goal||_2` after settling.
  This captures the steady-state accuracy.

### Results

The test suite enforces:
- RMS tracking error < 0.05 rad
- Final position error < 0.1 rad

In practice, the capstone demo achieves tighter performance:
- RMS < 0.01 rad
- Final error < 0.02 rad

### Raw vs. Smoothed Trajectory Comparison

The capstone demo executes both the RRT and RRT* smoothed trajectories and
compares them. Key observations:

| Metric              | Raw RRT path    | Smoothed RRT* path |
|---------------------|-----------------|--------------------|
| Trajectory duration | Longer (~2.3x)  | Shorter            |
| Tracking RMS        | Higher (~2.5x)  | Lower              |
| Final error         | Comparable      | Comparable         |

The smoothed path is faster because shortcutting removes detours, and TOPP-RA
finds the time-optimal parameterization of the shorter geometric path. The
tracking error is lower because the smoothed trajectory has gentler curvature,
requiring less aggressive torques.

---

## Gain Selection: Kp=400, Kd=40

The default gains were chosen based on critical damping. For a second-order
system:

    M * qdd + Kd * qd + Kp * q = 0

Critical damping occurs when `Kd = 2 * sqrt(Kp * M)`. With effective inertia
roughly 1 kg*m^2 for the UR5e wrist joints:

    Kd = 2 * sqrt(400 * 1) = 40

Higher Kp improves tracking stiffness but amplifies sensor noise and can
excite high-frequency dynamics. Lower Kp causes sluggish tracking and larger
steady-state errors. The 400/40 combination provides a good balance for the
UR5e across its workspace.

---

## Visualization

The capstone demo generates two comparison plots:

1. **Execution comparison** (`capstone_execution_comparison.png`): 2x2 grid
   showing per-joint tracking error and applied torques for both RRT and RRT*
   trajectories.

2. **End-effector trajectory** (`capstone_ee_trajectory.png`): 3D plot of the
   actual end-effector path traced during execution, comparing the two
   planners.

---

## Full Pipeline Summary

```
q_start, q_goal
      |
      v
  RRT* Planner  -->  Raw path (12 waypoints)
      |
      v
  Shortcutting  -->  Short path (2 waypoints)
      |
      v
  TOPP-RA       -->  Timed trajectory (positions, velocities, accelerations)
      |
      v
  Impedance     -->  Torque commands at 1 kHz
  Controller         tau = Kp*(q_d - q) + Kd*(qd_d - qd) + g(q)
      |
      v
  MuJoCo Step   -->  Simulated execution with logged data
```

---

## Test Coverage

Execution is tested in `tests/test_trajectory.py` within the `TestExecution`
class:

- `test_tracking_error_below_threshold` -- RMS error < 0.05 rad
- `test_final_position_close_to_goal` -- final error < 0.1 rad
- `test_result_dict_keys` -- all expected keys present in output

Combined with the collision and planner tests, Lab 4 has 45 total tests
passing across 3 test files.
