# Lab 4: Motion Planning & Collision Avoidance — Architecture

Completion date: 2026-03-17

## Architecture Status

The canonical Lab 4 architecture is now implemented.

- MuJoCo executes the Menagerie UR5e with mounted Robotiq 2F-85
- Collision truth comes from that same executed MuJoCo geometry
- Pinocchio is retained for FK and gravity terms, not as the primary collision truth source

## Module Map

```
lab-4-motion-planning/
├── src/
│   ├── lab4_common.py             # Canonical stack loading, obstacle specs, actuator helpers
│   ├── collision_checker.py       # MuJoCo-geometry collision checking
│   ├── rrt_planner.py             # RRT and RRT* planner
│   ├── trajectory_smoother.py     # Shortcutting + time parameterization
│   ├── trajectory_executor.py     # PD + gravity execution on the real stack
│   ├── capstone_demo.py           # Stable comparison demo
│   └── record_lab4_validation.py  # Blocked-path validation video
├── models/
├── tests/
├── docs/
├── docs-turkish/
├── media/
└── README.md
```

## Data Flow

```
Q_START, Q_GOAL
        │
        ▼
collision_checker.py
        │
        ├─ uses canonical MuJoCo robot + obstacles
        ├─ is_collision_free(q)
        ├─ is_path_free(q1, q2)
        └─ compute_min_distance(q)
        │
        ▼
rrt_planner.py
        │
        └─ collision-free joint-space path
        │
        ▼
trajectory_smoother.py
        │
        ├─ shortcutting
        └─ timed trajectory
        │
        ▼
trajectory_executor.py
        │
        ├─ q, qd from MuJoCo
        ├─ g(q) from Pinocchio
        └─ torque mapped into Menagerie arm actuators
        │
        ▼
MuJoCo execution + logged telemetry + validation media
```

## Key Interfaces

### `lab4_common.py`

```python
def load_mujoco_model(...): ...
def load_pinocchio_model(...): ...
def apply_arm_torques(...): ...
def get_ee_pos(...): ...
def get_mj_ee_pos(...): ...
```

Responsibilities:

- load the canonical UR5e + Robotiq scene with configured obstacles
- expose stable obstacle specs and actuator helpers
- keep FK utilities aligned with the executed stack

### `collision_checker.py`

Responsibilities:

- evaluate collisions on the executed MuJoCo geometry
- preserve the Lab 4 collision API used by planner/tests
- ignore internal Robotiq linkage proximity that is not meaningful planning collision

### `rrt_planner.py`

Responsibilities:

- plan collision-free C-space paths
- support both RRT and RRT*
- visualize planner output using the canonical obstacle layout

### `trajectory_smoother.py`

Responsibilities:

- shorten paths with shortcutting
- keep the `parameterize_topp_ra(...)` API stable
- use TOPP-RA when available, else a conservative quintic fallback

### `trajectory_executor.py`

Responsibilities:

- execute trajectories on the canonical MuJoCo stack
- add gravity compensation from Pinocchio
- map desired torques through the Menagerie actuator model

### `record_lab4_validation.py`

Responsibilities:

- run the blocked-path validation scenario
- save the sign-off video to `media/`
- overlay the MuJoCo scene with execution/path metrics

## Video Production

All demo videos use the shared `tools/video_producer.py` three-phase pipeline.
Lab 4 demo scripts live in `src/` alongside the other modules. All output goes to `media/`.

- `src/generate_lab4_demo.py` — slalom demo generator using the shared video producer
- `src/slalom_demo.py` — slalom planning pipeline, metrics, camera schedule
- `src/record_lab4_validation.py` — blocked-path validation video recorder
