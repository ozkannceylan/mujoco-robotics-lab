# Lab 4: Motion Planning & Collision Avoidance — Architecture

## Module Map

```
lab-4-motion-planning/
├── src/
│   ├── lab4_common.py           # Paths, constants, model loading
│   ├── collision_checker.py     # Pinocchio collision infrastructure
│   ├── rrt_planner.py           # RRT and RRT* planner
│   ├── trajectory_smoother.py   # Shortcutting + TOPP-RA integration
│   ├── trajectory_executor.py   # Execute trajectory on torque-controlled arm
│   └── capstone_demo.py         # Full pipeline: plan → smooth → execute
├── models/
│   ├── ur5e.xml                 # Reuse from Lab 3 (torque-mode actuators)
│   ├── ur5e_collision.urdf      # UR5e URDF with collision geometries
│   └── scene_obstacles.xml      # Cluttered scene: table + box obstacles
├── tests/
│   ├── test_collision.py        # Collision checker correctness
│   ├── test_planner.py          # RRT/RRT* path validity
│   └── test_trajectory.py       # Smoothing + execution tests
├── docs/                        # English documentation
├── docs-turkish/                # Turkish documentation
├── media/                       # Plots, visualizations
└── tasks/                       # PLAN, ARCHITECTURE, TODO, LESSONS
```

## Data Flow

```
                     ┌──────────────┐
  Q_START, Q_GOAL ──▶│  RRT* Planner │
                     │  (rrt_planner)│
                     └──────┬───────┘
                            │ waypoints: list[np.ndarray]
                            │ (collision-free C-space path)
                            ▼
                     ┌──────────────────┐
                     │ Trajectory Smoother│
                     │ (trajectory_smoother)│
                     │                    │
                     │ 1. Shortcutting    │
                     │ 2. TOPP-RA timing  │
                     └──────┬─────────────┘
                            │ TimedTrajectory: (t, q, qd, qdd)
                            ▼
                     ┌──────────────────┐
                     │ Trajectory Executor│
                     │ (trajectory_executor)│
                     │                    │
                     │ Joint impedance:   │
                     │ τ = Kp(q_d-q)     │
                     │   + Kd(qd_d-qd)  │
                     │   + g(q)          │
                     └──────────────────┘
                            │
                            ▼
                        MuJoCo sim
```

All modules use `collision_checker.py` for collision queries:

```
collision_checker.py
├── CollisionChecker class
│   ├── __init__(urdf_path, obstacle_specs)
│   │   → loads Pinocchio model + collision model
│   │   → adds obstacle boxes to GeometryModel
│   │   → registers collision pairs
│   ├── is_collision_free(q) → bool
│   ├── is_path_free(q1, q2, resolution) → bool
│   └── compute_min_distance(q) → float
│
└── Used by: rrt_planner, trajectory_smoother, tests
```

## Key Interfaces

### collision_checker.py

```python
class CollisionChecker:
    def __init__(
        self,
        urdf_path: Path,
        obstacle_specs: list[ObstacleSpec] | None = None,
    ) -> None:
        """Load Pinocchio model and add obstacles to collision model."""

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Check if configuration q is collision-free."""

    def is_path_free(
        self, q1: np.ndarray, q2: np.ndarray, resolution: float = 0.05
    ) -> bool:
        """Check if linear path from q1 to q2 is collision-free."""

    def compute_min_distance(self, q: np.ndarray) -> float:
        """Compute minimum distance between any collision pair at q."""

@dataclass
class ObstacleSpec:
    name: str
    position: np.ndarray   # (3,) world position
    half_extents: np.ndarray  # (3,) box half-sizes
```

### rrt_planner.py

```python
@dataclass
class RRTNode:
    q: np.ndarray          # configuration (6,)
    parent: int | None     # index of parent in tree
    cost: float            # path cost from root (RRT* only)

class RRTStarPlanner:
    def __init__(
        self,
        collision_checker: CollisionChecker,
        joint_limits_lower: np.ndarray,
        joint_limits_upper: np.ndarray,
        step_size: float = 0.3,
        goal_bias: float = 0.1,
        rewire_radius: float = 1.0,
    ) -> None: ...

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        max_iter: int = 5000,
        rrt_star: bool = True,
    ) -> list[np.ndarray] | None:
        """Plan collision-free path. Returns waypoints or None."""

    @property
    def tree(self) -> list[RRTNode]:
        """Access the search tree for visualization."""
```

### trajectory_smoother.py

```python
def shortcut_path(
    path: list[np.ndarray],
    collision_checker: CollisionChecker,
    max_iter: int = 200,
) -> list[np.ndarray]:
    """Shortcut a C-space path by removing unnecessary waypoints."""

def parameterize_topp_ra(
    path: list[np.ndarray],
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-optimal path parameterization via TOPP-RA.

    Returns: (times, positions, velocities, accelerations)
    """
```

### trajectory_executor.py

```python
def execute_trajectory(
    times: np.ndarray,
    q_traj: np.ndarray,
    qd_traj: np.ndarray,
    scene_path: Path,
    Kp: float = 400.0,
    Kd: float = 40.0,
) -> dict[str, np.ndarray]:
    """Execute timed trajectory on torque-controlled UR5e.

    Uses joint-space impedance: τ = Kp(q_d-q) + Kd(qd_d-qd) + g(q).

    Returns: dict with time, q_actual, q_desired, tau, ee_pos.
    """
```

## Model Files

### ur5e.xml
- Reuse Lab 3's torque-mode model (motor actuators, armature=0.01)
- Symlink or copy from `lab-3-dynamics-force-control/models/`

### ur5e_collision.urdf
- Extend Lab 3's `ur5e.urdf` by adding `<collision>` elements to each link
- Collision shapes match MJCF geom shapes:
  - base: cylinder (radius=0.09, half_length=0.06)
  - shoulder: box (0.06, 0.06, 0.09)
  - upper_arm: capsule (0→-0.425, radius=0.055)
  - forearm: capsule (0→(-0.3922, 0, 0.1333), radius=0.05)
  - wrist_1: capsule (0→(0, 0, 0.0997), radius=0.045)
  - wrist_2: capsule (0→(0, 0, 0.0996), radius=0.04)
  - wrist_3: capsule (0→(0, 0, 0.135), radius=0.035)
  - tool0: box (0.015, 0.015, 0.0675)

### scene_obstacles.xml
- Includes ur5e.xml
- Table (reuse from Lab 3)
- 3–5 box obstacles on table at known positions
- Each obstacle: `<body>` with `<geom type="box">` at specified position

## Dependencies on Previous Labs

- **Lab 3 `lab3_common.py`**: NUM_JOINTS, DT, Q_HOME, TORQUE_LIMITS, load functions
- **Lab 3 `ur5e.xml`**: torque-mode actuators
- **Lab 3 `ur5e.urdf`**: kinematic chain (extended with collision geometries)
- **Lab 3 impedance controller**: reuse `compute_impedance_torque` or re-implement joint-space PD + g(q)

## Simulation Parameters

- Timestep: 0.001s (1 kHz, matching Lab 3)
- Joint limits: [-2π, 2π] for all joints
- RRT step_size: 0.3 rad
- RRT goal_bias: 0.1 (10% of samples directed at goal)
- RRT max_iter: 5000
- RRT* rewire_radius: 1.0 rad
- Shortcutting iterations: 200
- Collision resolution: 0.05 rad (interpolation step for edge checking)
- Execution gains: Kp=400, Kd=40 (joint-space impedance)
- Velocity limits: [3.14, 3.14, 3.14, 6.28, 6.28, 6.28] rad/s (UR5e datasheet)
- Acceleration limits: [8.0, 8.0, 8.0, 16.0, 16.0, 16.0] rad/s²

## Obstacle Layout (Capstone)

```
  Top view (XY plane, Z up):

        ┌──────────────────────┐
        │       TABLE          │
        │                      │
        │   ┌──┐       ┌──┐   │
        │   │O1│       │O2│   │
        │   └──┘       └──┘   │
        │         ┌──┐         │
        │         │O3│         │
        │   ┌──┐  └──┘  ┌──┐  │
        │   │O4│        │O5│  │
        │   └──┘        └──┘  │
        │                      │
        └──────────────────────┘
             ▲ UR5e base
```

Obstacle positions and sizes will be tuned to create:
- One narrow passage the arm must navigate through
- At least one configuration where naive straight-line IK would collide
- Both start and goal configs must be collision-free
