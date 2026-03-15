# Lab 4: Motion Planning & Collision Avoidance

Sampling-based motion planning for the UR5e 6-DOF arm with collision avoidance, trajectory optimization, and torque-controlled execution.

## Overview

This lab implements a complete motion planning pipeline:

1. **Collision Infrastructure** — Pinocchio + HPP-FCL for self-collision and environment obstacle checking
2. **RRT / RRT\*** — Sampling-based planners in configuration space with goal bias and rewiring
3. **Trajectory Post-Processing** — Path shortcutting + TOPP-RA time-optimal parameterization
4. **Trajectory Execution** — Joint-space impedance control on torque-controlled UR5e in MuJoCo

## Key Results

| Metric | Value |
|---|---|
| Pinocchio/MuJoCo collision agreement | 92.8% |
| RRT planning time | 0.02s |
| RRT* planning time | 29.4s (5000 iterations) |
| Shortcutting reduction | 12 to 2 waypoints |
| TOPP-RA trajectory duration | 0.87-0.99s |
| Tracking RMS error | < 0.01 rad |
| Final position error | < 0.02 rad |
| Total tests | 45 passing |

## Architecture

```
collision_checker.py    Pinocchio + HPP-FCL collision queries
        |
rrt_planner.py          RRT/RRT* sampling-based planning
        |
trajectory_smoother.py  Shortcutting + TOPP-RA timing
        |
trajectory_executor.py  Joint-space impedance on MuJoCo
        |
capstone_demo.py        Full pipeline: plan -> smooth -> execute
```

## Quick Start

```bash
cd lab-4-motion-planning/src

# Run the capstone demo
python capstone_demo.py

# Run all tests
cd .. && python -m pytest tests/ -v
```

## Scene Setup

The scene contains a UR5e arm with 4 box obstacles on a table, creating narrow passages:

- **Table**: 0.60m x 0.90m at height 0.30m
- **Obstacles**: 4 red boxes at various heights creating planning challenges
- **Robot**: UR5e with torque-mode actuators (6 DOF)

## Collision Checking

The `CollisionChecker` class wraps Pinocchio's GeometryModel with HPP-FCL:

- Self-collision pairs with adjacent-link filtering (adjacency gap = 1)
- Environment obstacles added as fixed boxes in the world frame
- Cross-validated against MuJoCo contacts (92.8% agreement)

```python
from collision_checker import CollisionChecker

cc = CollisionChecker()
free = cc.is_collision_free(q)           # single config
path_ok = cc.is_path_free(q1, q2)       # edge check
dist = cc.compute_min_distance(q)        # penetration depth
```

## Planning

RRT and RRT* with configurable parameters:

```python
from rrt_planner import RRTStarPlanner

planner = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.1, rewire_radius=1.0)
path = planner.plan(q_start, q_goal, max_iter=5000, rrt_star=True, seed=42)
```

## Trajectory Processing

```python
from trajectory_smoother import shortcut_path, parameterize_topp_ra

short = shortcut_path(path, cc, max_iter=200)
times, pos, vel, acc = parameterize_topp_ra(short, VEL_LIMITS, ACC_LIMITS)
```

## Execution

```python
from trajectory_executor import execute_trajectory

result = execute_trajectory(times, pos, vel, Kp=400.0, Kd=40.0)
# result: {time, q_actual, q_desired, tau, ee_pos}
```

## Dependencies

- Python 3.10+
- MuJoCo
- Pinocchio (with HPP-FCL)
- NumPy
- Matplotlib
- toppra

## File Structure

```
lab-4-motion-planning/
├── src/
│   ├── lab4_common.py           # Constants, paths, model loading
│   ├── collision_checker.py     # Pinocchio collision infrastructure
│   ├── rrt_planner.py           # RRT/RRT* planner + visualization
│   ├── trajectory_smoother.py   # Shortcutting + TOPP-RA
│   ├── trajectory_executor.py   # Impedance control execution
│   └── capstone_demo.py         # Full pipeline demo
├── models/
│   ├── ur5e.xml                 # Torque-mode UR5e (MJCF)
│   ├── ur5e_collision.urdf      # UR5e with collision geometries
│   └── scene_obstacles.xml      # Cluttered scene with obstacles
├── tests/
│   ├── test_collision.py        # 21 tests
│   ├── test_planner.py          # 11 tests
│   └── test_trajectory.py       # 13 tests
├── docs/                        # English documentation
├── docs-turkish/                # Turkish documentation
├── media/                       # Plots and visualizations
└── tasks/                       # PLAN, ARCHITECTURE, TODO, LESSONS
```
