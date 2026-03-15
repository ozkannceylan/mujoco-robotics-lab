# Lab 4: Motion Planning & Collision Avoidance — Implementation Plan

## Phase 1: Collision Infrastructure

### Step 1.1: Create cluttered MJCF scene
- Create `models/scene_obstacles.xml` extending Lab 3's UR5e torque model
- Add 3–5 box obstacles on/around the table at known positions
- Obstacles: different sizes and positions to create narrow passages
- Verify scene loads and renders in MuJoCo
- **Output:** Scene file with obstacles visible in simulation

### Step 1.2: Create common module (`lab4_common.py`)
- Define paths (SCENE_PATH, URDF_PATH, MEDIA_DIR, etc.)
- Import/reuse Lab 3 constants (NUM_JOINTS, DT, Q_HOME, TORQUE_LIMITS)
- Define key configurations: Q_START, Q_GOAL for the capstone demo
- Load functions for Pinocchio and MuJoCo models
- **Output:** Common module importable by all Phase 1–4 scripts

### Step 1.3: Set up Pinocchio collision model with obstacles
- Load UR5e URDF into Pinocchio with collision geometries (capsules/boxes matching MJCF)
- Add environment obstacles (boxes) as collision objects in Pinocchio's GeometryModel
- Register collision pairs: robot links vs environment, robot links vs robot links (self-collision)
- Implement `is_collision_free(q)` → bool
- Implement `is_path_segment_free(q1, q2, resolution)` → bool (interpolation check)
- **Output:** Collision checker with tests

### Step 1.4: Cross-validate collision between Pinocchio and MuJoCo
- For several test configs (known collision, known free), check both engines agree
- MuJoCo: set qpos, mj_forward, check data.ncon
- Pinocchio: computeCollisions, check result
- **Output:** Cross-validation passing, any discrepancies documented

### Step 1.5: Write Phase 1 tests
- `test_collision.py`: collision at known configs, free at known configs, path segment checks
- **Verify:** All tests pass

## Phase 2: RRT / RRT* Implementation

### Step 2.1: Implement basic RRT planner
- `rrt_planner.py` with class `RRTPlanner`
- C-space: 6D joint space with joint limits as bounds
- `sample_random()`: uniform random in joint limits (with goal bias)
- `nearest(tree, q)`: find nearest node by L2 distance
- `steer(q_near, q_rand, step_size)`: extend toward sample with max step
- `plan(q_start, q_goal, max_iter)` → list of configs or None
- Tune: step_size=0.3 rad, goal_bias=0.1, max_iter=5000
- **Output:** RRT finds paths in obstacle-free and simple-obstacle scenes

### Step 2.2: Extend to RRT* with rewiring
- Add cost tracking (path length from start)
- `near_neighbors(tree, q, radius)`: find all nodes within rewiring radius
- Rewire: if new node provides shorter path to neighbors, update parent
- Choose parent: pick lowest-cost neighbor as parent
- **Output:** RRT* produces shorter paths than RRT

### Step 2.3: Visualize planned paths
- Plot RRT tree growth in 2D projection (joint 1 vs joint 2)
- Plot planned path in MuJoCo: render arm at waypoints
- Compare RRT vs RRT* path lengths
- **Output:** Visualization plots saved to media/

### Step 2.4: Write Phase 2 tests
- `test_planner.py`: path validity, collision-free, start/goal match
- Test on easy (no obstacles) and medium (few obstacles) scenes
- **Verify:** All tests pass

## Phase 3: Trajectory Post-Processing & Execution

### Step 3.1: Implement path shortcutting
- Iterative shortcutting: pick two random waypoints, if collision-free straight line between them, replace intermediate waypoints
- Configurable: max_iterations, min_segment_length
- **Output:** Shorter, smoother paths from shortcutting

### Step 3.2: Integrate TOPP-RA for time-optimal parameterization
- Install `toppra` package
- Given waypoints in C-space, compute time-optimal velocity profile
- Respect joint velocity and acceleration limits
- Output: time-stamped trajectory (t, q(t), qd(t), qdd(t))
- **Output:** Smooth, time-parameterized trajectories

### Step 3.3: Execute trajectory on torque-controlled UR5e
- Reuse Lab 3's impedance controller for joint-space trajectory tracking
- Feed q_des(t), qd_des(t) from TOPP-RA to impedance controller
- Joint-space impedance: τ = K_p·(q_des - q) + K_d·(qd_des - qd) + g(q)
- Record actual vs desired joint positions
- **Output:** Executed trajectory with tracking error analysis

### Step 3.4: Compare raw vs smoothed execution
- Execute raw RRT* path (linear interpolation between waypoints)
- Execute shortcut + TOPP-RA smoothed path
- Compare: execution time, max jerk, tracking error
- **Output:** Comparison plots in media/

### Step 3.5: Write Phase 3 tests
- `test_trajectory.py`: shortcutting reduces length, TOPP-RA respects limits
- **Verify:** All tests pass

## Phase 4: Capstone & Documentation

### Step 4.1: Design capstone cluttered scene
- Table with 3–5 box obstacles creating narrow passages
- Define start config (arm to one side) and goal config (arm on other side, reaching between obstacles)
- Verify RRT* can solve the scene

### Step 4.2: Run capstone demo
- Plan collision-free path with RRT*
- Smooth with shortcutting + TOPP-RA
- Execute on torque-controlled UR5e
- Record trajectory: joint positions, EE path, obstacle clearance
- Save plots to media/

### Step 4.3: Write English documentation (`docs/`)
- Theory: C-space, RRT*, collision checking, TOPP-RA
- Architecture and results
- Include algorithm visualizations

### Step 4.4: Write Turkish documentation (`docs-turkish/`)
- Translate docs/ to Turkish

### Step 4.5: Write blog post
- "From Free Space to Cluttered Environments"
- Cover: why planning matters, RRT* intuition, smoothing, demo results

### Step 4.6: Write README.md
- Lab overview, module map, how to run, key results
