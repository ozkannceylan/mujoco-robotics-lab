# UR5e Robotics Lab: From Fundamentals to Portfolio Demo

**Engineer:** Ozkan — AI/Robotics Engineer, mechatronics background, RL for mobile robotics
**Builds on:** 2-link planar MuJoCo crash course (completed)
**Stack:** MuJoCo + Pinocchio + UR5e
**Duration:** 12 days (~4–6 hours/day)
**Output:** Portfolio-ready manipulation demo + VLA bridge document

---

## Architecture Overview

```text
PHASE 1: FOUNDATIONS (Days 1–2)
┌──────────────────────────────────────────────────────────┐
│  1.1  Environment setup    (MuJoCo + Pinocchio + UR5e)   │
│  1.2  FK + DH parameters   (6-DOF forward kinematics)    │
└──────────────────────────────────────────────────────────┘
                         ↓
PHASE 2: CORE KINEMATICS (Days 3–5)
┌──────────────────────────────────────────────────────────┐
│  2.1  Jacobian (6×6)       Singularity + manipulability  │
│  2.2  Inverse kinematics   Analytical + numerical        │
│  2.3  Dynamics             Pinocchio RNEA, ABA, CRBA     │
└──────────────────────────────────────────────────────────┘
                         ↓
PHASE 3: MOTION + CONTROL (Days 6–8)
┌──────────────────────────────────────────────────────────┐
│  3.1  Trajectory generation    Joint + Cartesian space   │
│  3.2  Control hierarchy        PD → task-space → OSC     │
│  3.3  Constraints              Joint limits + collision  │
└──────────────────────────────────────────────────────────┘
                         ↓
PHASE 4: INTEGRATION DEMOS (Days 9–10)
┌──────────────────────────────────────────────────────────┐
│  4.1  Full pick-and-place pipeline                       │
│  4.2  ROS2 + MoveIt2 bridge                              │
└──────────────────────────────────────────────────────────┘
                         ↓
PHASE 5: PORTFOLIO + VLA BRIDGE (Days 11–12)
┌──────────────────────────────────────────────────────────┐
│  5.1  VLA bridge document                                │
│  5.2  Portfolio package (README + video + metrics)        │
└──────────────────────────────────────────────────────────┘
```

---

## Why This Stack

### UR5e

- Industry standard 6-DOF arm — most documentation, most interview questions reference it
- Known DH parameters, well-studied singularity configurations
- MuJoCo Menagerie ships a validated MJCF model (or use `mujoco_menagerie/universal_robots_ur5e`)
- Has a closed-form analytical IK solution (rare for 6-DOF — huge learning opportunity)

### Pinocchio

- Gold-standard C++/Python rigid-body dynamics library (LAAS-CNRS)
- Computes FK, Jacobians, RNEA (inverse dynamics), ABA (forward dynamics), CRBA (mass matrix) analytically
- Loads URDF directly — same model file ecosystem as ROS2
- Where MuJoCo is a black-box simulator, Pinocchio lets you inspect every matrix — critical for understanding, interviews, and debugging controllers

### MuJoCo + Pinocchio Together

```text
Pinocchio                          MuJoCo
┌────────────────────┐            ┌────────────────────┐
│ Analytical engine   │            │ Physics simulator   │
│                     │            │                     │
│ • FK/IK compute     │───────────→│ • Contact dynamics  │
│ • Jacobian          │  θ_target  │ • Rendering         │
│ • M(q), C(q,v), g  │            │ • Sensor readout    │
│ • Gravity comp.     │←───────────│ • State feedback    │
│ • Collision model   │  q, v, τ   │                     │
└────────────────────┘            └────────────────────┘
```

- Use Pinocchio for all analytical computations (FK, Jacobian, dynamics, IK)
- Use MuJoCo for simulation, rendering, and contact physics
- This mirrors real-world robotics: analytical model for control, simulator/hardware for execution

---

## Project Structure

```text
ur5e-robotics-lab/
├── models/
│   ├── ur5e.urdf                    ← from mujoco_menagerie or official UR
│   ├── ur5e.xml                     ← MJCF for MuJoCo
│   └── scene.xml                    ← table + objects
├── src/
│   ├── core/
│   │   ├── fk.py                    ← DH-based FK + Pinocchio FK
│   │   ├── jacobian.py              ← analytical + Pinocchio Jacobian
│   │   ├── ik_analytical.py         ← UR5e closed-form IK
│   │   ├── ik_numerical.py          ← Jacobian-based + damped LS
│   │   └── dynamics.py              ← Pinocchio dynamics wrappers
│   ├── motion/
│   │   ├── trajectory.py            ← cubic, quintic, Cartesian
│   │   ├── task_space_control.py    ← OSC, impedance control
│   │   └── constraints.py           ← joint limits, self-collision
│   ├── pipeline/
│   │   ├── pick_and_place.py        ← full demo pipeline
│   │   └── mujoco_sim.py            ← MuJoCo simulation wrapper
│   └── ros2_bridge/
│       ├── mujoco_bridge_node.py
│       └── moveit_interface.py
├── tests/
│   ├── test_fk.py
│   ├── test_jacobian.py
│   ├── test_ik.py
│   ├── test_dynamics.py
│   └── test_trajectory.py
├── demos/
│   ├── demo_pick_place.py
│   ├── demo_circle_tracking.py
│   ├── demo_singularity.py
│   └── demo_impedance.py
├── notebooks/                       ← optional Jupyter for visualization
├── docs/
│   ├── vla_bridge.md
│   └── interview_cheatsheet.md
├── CLAUDE.md
└── README.md
```

---

# PHASE 1 — FOUNDATIONS (Days 1–2)

---

## 1.1 Environment Setup (Day 1, ~4 hours)

### Learning Goal

Get MuJoCo and Pinocchio running with the same UR5e model, verify they agree on basic state.

### Concepts

**UR5e Specifications You Need to Know**

```text
DOF:          6 revolute joints
Reach:        850 mm
Payload:      5 kg
Joint order:  shoulder_pan → shoulder_lift → elbow →
              wrist_1 → wrist_2 → wrist_3

Joint limits (approximate):
  J1 (shoulder_pan):   ±360°
  J2 (shoulder_lift):  ±360°
  J3 (elbow):          ±360°
  J4 (wrist_1):        ±360°
  J5 (wrist_2):        ±360°
  J6 (wrist_3):        ±360°
```

**URDF vs. MJCF — Mental Model**

```text
URDF (ROS ecosystem)              MJCF (MuJoCo native)
├── <link>    ← body + visual     ├── <body>    ← frame
├── <joint>   ← DOF               ├── <joint>   ← DOF
├── <inertial>                     ├── <geom>    ← shape + collision
└── <gazebo>  ← sim-specific      ├── <actuator>← motors
                                   └── <sensor>  ← measurements

Pinocchio loads URDF natively.
MuJoCo can load URDF (with mujoco.MjModel.from_xml_path)
or use pre-converted MJCF from mujoco_menagerie.
```

**Pinocchio Model Loading**

```text
pinocchio.buildModelFromUrdf(urdf_path)
  → model: kinematic/dynamic parameters
  → data:  runtime computation buffers

model = the robot's blueprint (never changes)
data  = the robot's current state (changes every step)
```

### Tasks

1. Install dependencies: `mujoco`, `pinocchio` (pin), `meshcat-python` (optional visualization)
2. Download UR5e model — either from `mujoco_menagerie` (MJCF) or official UR URDF
3. Load the model in MuJoCo — run 100 steps, read `data.qpos` (6-element array)
4. Load the same model (URDF) in Pinocchio — call `pinocchio.forwardKinematics(model, data, q)` for the same joint angles
5. Compare the end-effector positions from both — they should agree within < 0.001 m
6. Build a minimal scene with a table and a target sphere
7. Create a `mujoco_sim.py` wrapper class that encapsulates: load model, step, read state, apply control, render

### Verification

- [ ] MuJoCo loads and steps the UR5e model without errors
- [ ] `data.qpos` returns a 6-element array
- [ ] Pinocchio loads the same robot from URDF
- [ ] FK results from both agree (< 0.001 m error)
- [ ] Scene renders with a table and target object

---

## 1.2 FK + DH Parameters (Day 2, ~5 hours)

### Learning Goal

Understand how DH parameters define a 6-DOF kinematic chain and compute FK manually, then verify against Pinocchio.

### Concepts

**Why DH Parameters Now (vs. Geometric FK Before)**

```text
2-link planar → geometric FK was simpler
6-DOF spatial → DH parameters are the systematic approach

DH gives you a mechanical recipe:
  For each joint i, four parameters define the transform T(i-1 → i)
```

**DH Convention (Modified DH for UR5e)**

```text
T(i-1 → i) = Rot_x(α) · Trans_x(a) · Trans_z(d) · Rot_z(θ)

Parameters per joint:
  α (alpha) : twist angle — rotation around x(i-1)
  a         : link length — translation along x(i-1)
  d         : link offset — translation along z(i)
  θ (theta) : joint angle — rotation around z(i) ← this is the variable
```

**UR5e DH Table (Modified DH)**

```text
Joint │  α (rad)  │  a (m)   │  d (m)   │  θ
──────┼───────────┼──────────┼──────────┼────────
  1   │  0        │  0       │  0.1625  │  θ₁
  2   │  π/2      │  0       │  0       │  θ₂
  3   │  0        │ -0.4250  │  0       │  θ₃
  4   │  0        │ -0.3922  │  0.1333  │  θ₄
  5   │  π/2      │  0       │  0.0997  │  θ₅
  6   │ -π/2      │  0       │  0.0996  │  θ₆
```

**FK Chain — The Big Picture**

```text
T_base_to_ee = T₀₁ · T₁₂ · T₂₃ · T₃₄ · T₄₅ · T₅₆

This gives you a 4×4 homogeneous transform:
┌         ┐
│ R₃ₓ₃  p │   R = orientation (rotation matrix)
│ 0    1  │   p = position (x, y, z)
└         ┘

From the 2-link course you know T = Rot · Trans.
Now you're chaining 6 of them — same idea, more matrices.
```

**How Pinocchio Does FK**

```text
pinocchio.forwardKinematics(model, data, q)
  → data.oMi[joint_id]   gives T for each joint frame
  → data.oMf[frame_id]   gives T for each named frame

The "ee" frame is usually the last operational frame.
You can get both position and orientation from it.
```

### Tasks

1. Write out the DH parameter table for UR5e on paper — understand what each parameter physically means
2. Implement the 4×4 DH transform function: given (α, a, d, θ), return T
3. Chain 6 transforms to compute full FK: T_base_to_ee = T₀₁ · ... · T₅₆
4. Compute FK for the "home" configuration (all zeros) and 5 other joint configurations
5. Compare your DH-based FK against Pinocchio's FK for each configuration
6. Extract both position (translation) AND orientation (rotation matrix → RPY or quaternion)
7. Create a comparison table: DH FK vs. Pinocchio FK — both position and orientation error
8. Visualize the robot in 3–4 different configurations (Matplotlib 3D or MuJoCo render)

### Verification

- [ ] DH FK matches Pinocchio FK (< 0.001 m position, < 0.01 rad orientation)
- [ ] You can explain what each DH parameter physically means for the UR5e
- [ ] FK works correctly for at least 6 joint configurations including edge cases
- [ ] 3D visualization shows the robot in different poses

---

# PHASE 2 — CORE KINEMATICS (Days 3–5)

---

## 2.1 Jacobian — 6×6 (Day 3, ~5 hours)

### Learning Goal

Compute and understand the full 6×6 Jacobian for a spatial manipulator. Connect it to singularity analysis and manipulability.

### Concepts

**Scaling Up from 2×2 to 6×6**

```text
2-link planar:  ẋ = J(2×2) · θ̇     (vx, vy)
6-DOF spatial:  ẋ = J(6×6) · θ̇     (vx, vy, vz, ωx, ωy, ωz)

The top 3 rows = linear velocity Jacobian (same idea as 2-link)
The bottom 3 rows = angular velocity Jacobian (new for 3D)
```

**Geometric Jacobian — Construction**

```text
For each revolute joint i:

  J_linear[:,i]  = z(i-1) × (p_ee - p(i-1))
  J_angular[:,i] = z(i-1)

where:
  z(i-1) = z-axis of frame (i-1) — the joint rotation axis
  p(i-1) = origin of frame (i-1)
  p_ee   = end-effector position
  ×      = cross product

Stack them:
  J = | J_linear  |   (3×6)
      | J_angular |   (3×6)
```

**Pinocchio Jacobian Computation**

```text
pinocchio.computeJointJacobians(model, data, q)
J = pinocchio.getJointJacobian(model, data, joint_id, reference_frame)

reference_frame options:
  LOCAL        — expressed in the joint's own frame
  WORLD        — expressed in world frame
  LOCAL_WORLD_ALIGNED — origin at joint, axes aligned with world
```

**Singularity Analysis — UR5e Specific**

```text
Singularity types for UR5e:

1. WRIST SINGULARITY
   → J5 ≈ 0  (wrist axes align)
   → The robot loses one DOF of wrist rotation
   → Most common in practice

2. SHOULDER SINGULARITY
   → End-effector on the J1 axis (directly above/below base)
   → det(J) → 0

3. ELBOW SINGULARITY
   → Arm fully extended (J3 = 0 or π)
   → Similar to the 2-link case you already know

Manipulability measure:
  w = sqrt(det(J · Jᵀ))
  High w = good dexterity, w → 0 = approaching singularity
```

**Why This Matters**

```text
In a VLA pipeline:
  - Jacobian maps action-space velocities to joint velocities
  - Singularity avoidance keeps the robot safe
  - Manipulability weighting can bias the policy toward "good" configurations
  - Task-space control (OSC) is built directly on the Jacobian

In interviews:
  - "Explain the robot Jacobian" is a standard question
  - "What happens at a singularity?" is the follow-up
```

### Tasks

1. Implement the geometric Jacobian construction from DH frames
2. Compute the same Jacobian using Pinocchio and compare (< 0.0001 error)
3. Compute the numerical Jacobian (finite differences) as a third check
4. Implement manipulability computation: w = sqrt(det(J · Jᵀ))
5. Map manipulability across the workspace — vary q2 and q3, plot w as a heatmap
6. Find and verify each singularity type: move the robot to a wrist singularity (q5 ≈ 0), shoulder singularity, and elbow singularity — observe det(J) → 0
7. Compute and visualize the velocity ellipsoid at several configurations

### Verification

- [ ] Your Jacobian matches Pinocchio (< 0.0001 error)
- [ ] Manipulability heatmap clearly shows high/low dexterity regions
- [ ] All three singularity types identified and verified (det(J) → 0)
- [ ] Velocity ellipsoid visualization shows the "easy" and "hard" motion directions
- [ ] You can explain "What is a singularity?" in an interview context

---

## 2.2 Inverse Kinematics (Day 4, ~5 hours)

### Learning Goal

Solve IK for a 6-DOF arm using both analytical and numerical approaches, understand multiple solutions and selection strategies.

### Concepts

**Why UR5e Has an Analytical IK (and Most Robots Don't)**

```text
The UR5e has a special kinematic structure:
  - Three consecutive wrist axes intersect at a point (spherical wrist)
  - This decouples position and orientation IK

Position IK (solve for q1, q2, q3):
  → Determines the wrist center position

Orientation IK (solve for q4, q5, q6):
  → Determines the end-effector orientation at that wrist center

Most 6-DOF arms don't have intersecting wrist axes
→ No decoupling → no closed-form solution → numerical only

The UR5e analytical IK produces UP TO 8 SOLUTIONS
(2 shoulder × 2 elbow × 2 wrist configurations)
```

**Numerical IK — Scaling from 2-Link**

```text
Same algorithm you used for 2-link, but now in SE(3):

repeat:
  1. p_current, R_current = FK(θ)
  2. Δx = [p_target - p_current; orientation_error(R_target, R_current)]
  3. Δθ = J⁺ · Δx     ← 6×6 pseudo-inverse
  4. θ = θ + α · Δθ
until ||Δx|| < tolerance

Orientation error (log map):
  e_orient = 0.5 · (R_target · R_current^T - I)_vee
  
  The _vee operator extracts the 3-vector from a skew-symmetric matrix.
  Pinocchio has pinocchio.log3() for this.
```

**Damped Least Squares (6-DOF Version)**

```text
Δθ = Jᵀ · (J·Jᵀ + λ²·I)⁻¹ · Δx

Same as 2-link, but now:
  - λ can be adaptive: increase near singularities
  - Selectively damp: weight different directions differently
```

**IK Solution Selection Strategy**

```text
With up to 8 analytical solutions, you need a selection policy:

1. CLOSEST TO CURRENT — minimize ||q_new - q_current||
   → Smoothest motion, used in real-time control

2. MANIPULABILITY — pick the solution with highest w(q)
   → Stays away from singularities

3. JOINT LIMIT MARGIN — pick the solution furthest from limits
   → Safety-first approach

4. WEIGHTED COMBINATION
   → score = α · distance + β · manipulability + γ · limit_margin
```

### Tasks

1. Implement the UR5e analytical IK step by step:
   - Wrist center computation from target pose
   - q1 solution (2 options: shoulder left/right)
   - q5, q6 from wrist center orientation
   - q3 from elbow geometry (2 options: elbow up/down)
   - q2, q4 from remaining constraints
2. Verify all 8 solutions by running FK on each — they should all reach the same target
3. Implement numerical IK with the Jacobian pseudo-inverse (full 6-DOF, position + orientation)
4. Implement damped least squares with adaptive λ
5. Implement a solution selection policy (closest-to-current + manipulability weighting)
6. Test both IK methods on 50 random reachable targets — compare success rate, iteration count, and computation time
7. Apply IK solutions in MuJoCo — the robot should reach the target pose

### Verification

- [ ] Analytical IK returns correct solutions (FK verification < 0.001 m, 0.01 rad)
- [ ] Multiple solutions (up to 8) are found and can be enumerated
- [ ] Numerical IK reaches 95%+ success on reachable targets
- [ ] Damped LS remains stable near singularities
- [ ] Solution selection consistently picks reasonable configurations
- [ ] MuJoCo robot reaches commanded target

---

## 2.3 Dynamics with Pinocchio (Day 5, ~4 hours)

### Learning Goal

Use Pinocchio to compute all terms of the robot dynamics equation and understand their role in control.

### Concepts

**Robot Dynamics Equation (Revisited)**

```text
τ = M(q)·q̈ + C(q, q̇)·q̇ + g(q)

Now with 6 joints:
  τ   : 6×1 joint torques
  M(q): 6×6 mass/inertia matrix (symmetric, positive definite)
  C   : 6×6 Coriolis + centrifugal matrix
  g   : 6×1 gravity vector
```

**Pinocchio Algorithms You Need to Know**

```text
RNEA — Recursive Newton-Euler Algorithm
  Input:  q, q̇, q̈
  Output: τ = M·q̈ + C·q̇ + g
  Use:    "What torques are needed for this motion?"
  → pinocchio.rnea(model, data, q, v, a)

ABA — Articulated Body Algorithm
  Input:  q, q̇, τ
  Output: q̈ (forward dynamics)
  Use:    "What acceleration results from these torques?"
  → pinocchio.aba(model, data, q, v, tau)

CRBA — Composite Rigid-Body Algorithm
  Input:  q
  Output: M(q) — the mass matrix
  Use:    "What is the robot's inertia at this configuration?"
  → pinocchio.crba(model, data, q)

Gravity vector:
  → pinocchio.computeGeneralizedGravity(model, data, q)
  → data.g  (6×1)

Coriolis:
  → pinocchio.computeCoriolisMatrix(model, data, q, v)
  → data.C  (6×6)
```

**Why Dynamics Matter for Control**

```text
PD control (Phase 3 preview):
  τ = Kp·(q_d - q) + Kd·(q̇_d - q̇) + g(q)
                                         ↑
                        Pinocchio gives you this for free

Computed Torque control:
  τ = M(q)·(q̈_d + Kp·e + Kd·ė) + C(q,q̇)·q̇ + g(q)
      ↑                             ↑          ↑
      All from Pinocchio — this is why we use it
```

### Tasks

1. Compute M(q) using `pinocchio.crba()` at the home configuration — observe it's 6×6, symmetric, positive definite
2. Compute g(q) for several configurations — observe how gravity load changes with arm pose
3. Use RNEA to compute the torques needed to hold the robot still at various configurations (q̈ = 0, q̇ = 0 → τ = g(q))
4. Use ABA to compute forward dynamics: given τ = 0 and gravity, see how the robot would fall
5. Compare: Pinocchio's RNEA output vs. MuJoCo's `data.qfrc_bias` — they should agree
6. Compute the condition number of M(q) at different configurations — high condition number = hard to control
7. Answer: "Why does M(q) depend on configuration?" with a concrete example from your computations

### Verification

- [ ] M(q) is symmetric and positive definite at all tested configurations
- [ ] g(q) changes significantly between different arm poses (e.g., arm stretched out vs. folded)
- [ ] RNEA matches MuJoCo's `qfrc_bias` (< 0.01 Nm error)
- [ ] You can explain the role of M, C, g in the dynamics equation
- [ ] Condition number of M varies across configurations — you know where control is harder

---

# PHASE 3 — MOTION PLANNING + CONTROL (Days 6–8)

---

## 3.1 Trajectory Generation (Day 6, ~5 hours)

### Learning Goal

Plan smooth, physically realizable motions in both joint space and Cartesian space for a 6-DOF arm.

### Concepts

**Joint-Space vs. Cartesian-Space (Revisited for 6-DOF)**

```text
JOINT SPACE                          CARTESIAN SPACE
q(t): interpolate 6 joint angles    x(t): interpolate position + orientation
                                     → requires IK at every timestep
                                     → orientation interpolation is non-trivial

New complexity vs. 2-link:
  - 6 joints need synchronized timing
  - Orientation requires SLERP (not just position lerp)
  - Via-point trajectories for multi-segment paths
```

**Cubic + Quintic (Same as 2-link, but per joint)**

```text
Apply the same polynomial formulas independently to each joint.
The key is: all 6 joints must have the SAME duration T.

Multi-segment (via-points):
  q₁ → q₂ → q₃ → ... → qN
  Each segment gets its own polynomial
  Boundary conditions enforce continuity of velocity (and acceleration for quintic)
```

**Cartesian Trajectory — Orientation Interpolation**

```text
Position: simple linear interpolation
  p(t) = (1-s)·p_start + s·p_end,  s = t/T

Orientation: SLERP (Spherical Linear Interpolation)
  R(t) = R_start · exp(s · log(R_start^T · R_end))

  Or in quaternion form:
  q(t) = slerp(q_start, q_end, s)

Pinocchio provides:
  pinocchio.interpolate(model, q_start, q_end, alpha)
  — works in configuration space, handles SO(3) properly
```

**Minimum Jerk Trajectory (Smooth for Task Space)**

```text
s(t) = 10·(t/T)³ - 15·(t/T)⁴ + 6·(t/T)⁵

Properties:
  s(0)=0, s(T)=1
  ṡ(0)=0, ṡ(T)=0
  s̈(0)=0, s̈(T)=0

This gives the smoothest possible motion — minimum jerk.
Used in human motor neuroscience as a model of natural movement.
```

**Trapezoidal Velocity Profile (Industrial Standard)**

```text
Three phases:
  1. Acceleration (constant q̈ = q̈_max)
  2. Cruise (constant q̇ = q̇_max)
  3. Deceleration (constant q̈ = -q̈_max)

This is what real industrial robots use.
It respects velocity and acceleration limits explicitly.
```

### Tasks

1. Implement joint-space cubic and quintic trajectory for all 6 joints simultaneously
2. Implement a trapezoidal velocity profile with configurable q̇_max and q̈_max per joint
3. Implement Cartesian trajectory with SLERP for orientation (use Pinocchio's `interpolate` or quaternion slerp)
4. Implement minimum jerk trajectory in Cartesian space
5. Implement multi-segment (via-point) trajectory: plan a path through 3–4 waypoints
6. Compare all trajectory types by plotting: joint positions, velocities, accelerations, and the end-effector path in 3D
7. Check feasibility: do any joints exceed UR5e velocity/acceleration limits?

### Verification

- [ ] Cubic and quintic trajectories have zero start/end velocity
- [ ] Quintic also has zero start/end acceleration
- [ ] Trapezoidal profile respects configured velocity/acceleration limits
- [ ] Cartesian trajectory produces a straight-line end-effector path with smooth orientation change
- [ ] Multi-segment trajectory is continuous in velocity at via-points
- [ ] 3D visualization shows clearly different end-effector paths for each method

---

## 3.2 Control Hierarchy (Day 7, ~5 hours)

### Learning Goal

Build a control hierarchy from simple PD to task-space Operational Space Control (OSC).

### Concepts

**Control Levels — Bottom to Top**

```text
Level 0: JOINT PD (you already know this)
  τ = Kp·(q_d - q) + Kd·(q̇_d - q̇)
  + gravity compensation: τ += g(q)

Level 1: COMPUTED TORQUE (inverse dynamics control)
  τ = M(q)·(q̈_d + Kp·e + Kd·ė) + C·q̇ + g(q)
  
  This "cancels out" the robot dynamics.
  The closed-loop becomes: ë + Kd·ė + Kp·e = 0
  → Linear, decoupled, easy to tune.
  → Requires accurate M, C, g — that's where Pinocchio shines.

Level 2: TASK-SPACE CONTROL (Operational Space Control)
  Work directly in Cartesian space — no IK needed!
  
  τ = Jᵀ · Λ · (ẍ_d + Kp·e_x + Kd·ė_x) + Jᵀ · μ + g(q)
  
  where:
    Λ = (J · M⁻¹ · Jᵀ)⁻¹     ← task-space inertia matrix
    μ = Λ · J · M⁻¹ · C·q̇ - Λ · J̇ · q̇  ← task-space Coriolis
    e_x = x_target - x_current  ← Cartesian error
  
  This is Operational Space Control (Khatib, 1987).
  The gold standard for Cartesian control.
```

**Why OSC Matters**

```text
For humanoid robotics and VLA:
  - The policy outputs Cartesian targets, not joint angles
  - OSC converts Cartesian commands to joint torques directly
  - No IK solve needed in the control loop → faster
  - Natural impedance behavior → safe contact with environment
  - This is what most modern humanoid controllers use
```

**Impedance Control (Preview)**

```text
Instead of tracking a trajectory rigidly:
  F = Kp·(x_d - x) + Kd·(ẋ_d - ẋ)

  → The end-effector behaves like a spring-damper system
  → Compliant: adapts to contact forces
  → Stiffness Kp controls how "rigid" the tracking is

τ = Jᵀ · F + g(q)

This is a simplified version of OSC — no inertia shaping.
Good enough for many practical applications.
```

### Tasks

1. Implement joint-space PD + gravity compensation (upgrade from 2-link PD — now using Pinocchio for g(q))
2. Implement computed torque control using Pinocchio's M, C, g
3. Compare PD vs. computed torque on a trajectory tracking task — plot tracking error
4. Implement task-space impedance control: τ = Jᵀ · (Kp·e_x + Kd·ė_x) + g(q)
5. Implement full Operational Space Control with inertia shaping (Λ, μ terms from Pinocchio)
6. Compare all controllers on the same trajectory — tracking error, torque magnitude, smoothness
7. Demonstrate impedance behavior: push the end-effector (apply external force in MuJoCo) and show compliant response

### Verification

- [ ] PD + gravity compensation tracks a static target (< 0.01 rad steady-state error)
- [ ] Computed torque shows significantly lower tracking error than PD
- [ ] OSC tracks a Cartesian trajectory without an IK solver in the loop
- [ ] Impedance control shows compliant behavior when external forces are applied
- [ ] You can explain the difference between PD, computed torque, and OSC

---

## 3.3 Constraints (Day 8, ~4 hours)

### Learning Goal

Handle real-world constraints: joint limits, velocity limits, and self-collision avoidance.

### Concepts

**Joint Limit Handling**

```text
Approach 1: CLAMPING (simple but discontinuous)
  q_cmd = clamp(q_cmd, q_min, q_max)

Approach 2: REPULSIVE POTENTIAL FIELD
  τ_limit = -k · ∂V/∂q
  V(q) = penalty that increases as q approaches limits
  → Smoothly pushes joints away from limits

Approach 3: NULL-SPACE OPTIMIZATION (for redundant systems)
  Not applicable to UR5e (6 DOF, 6 task DOF → no redundancy)
  But important to understand for 7-DOF arms like Panda/iiwa and humanoids
```

**Self-Collision Checking**

```text
Pinocchio + HPP-FCL (bundled with Pinocchio):
  pinocchio.computeCollisions(model, data, geometry_model, geometry_data, q)
  → Returns True/False for each collision pair

Use this to:
  1. Check IK solutions before commanding them
  2. Add a collision penalty to trajectory optimization
  3. Reject configurations in the data generation pipeline
```

**Velocity + Acceleration Limits**

```text
UR5e limits (approximate):
  q̇_max  ≈ 3.14 rad/s (180°/s) for large joints
  q̈_max  ≈ varies per joint

In the control loop:
  Δq = q_target - q_current
  q̇_cmd = Δq / dt
  if |q̇_cmd| > q̇_max:
    scale Δq to respect the limit
```

### Tasks

1. Implement joint limit clamping — test with a trajectory that would exceed limits
2. Implement a repulsive potential field for joint limits — visualize the repulsive torques
3. Set up Pinocchio's collision model for UR5e — verify known collision-free and colliding configurations
4. Add collision checking to your IK solver — reject solutions that self-collide
5. Implement velocity scaling: given a desired Δq, scale it to respect velocity limits while preserving direction
6. Create a "stress test" scenario: command the robot through a series of targets that challenge limits, singularities, and collisions simultaneously
7. Log and plot constraint violations over time

### Verification

- [ ] Joint limits are never exceeded in the simulation
- [ ] Self-collision is detected correctly for known test configurations
- [ ] IK solver rejects colliding solutions
- [ ] Velocity limits are respected — motion slows down rather than jumping
- [ ] Stress test runs without any constraint violations

---

# PHASE 4 — INTEGRATION DEMOS (Days 9–10)

---

## 4.1 Full Pick-and-Place Pipeline (Day 9, ~5 hours)

### Learning Goal

Integrate every component into a working manipulation pipeline.

### Pipeline Architecture

```text
                    ┌──────────────────┐
                    │  Target poses    │
                    │  (from task)     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  IK solver       │
                    │  analytical +    │
                    │  solution select │
                    │  + collision     │
                    │  check           │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Trajectory      │
                    │  generator       │
                    │  (trapezoidal +  │
                    │  velocity limit) │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Controller      │
                    │  (OSC or PD+g)   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  MuJoCo sim      │
                    │  + contact       │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Metrics + video │
                    └──────────────────┘
```

### Demo Scenarios

```text
Demo 1: PICK AND PLACE
  Approach → grasp (close gripper) → lift → move → place → release
  Tests: IK, trajectory, control, gripper actuation

Demo 2: MULTI-WAYPOINT TOUR
  Move through 4 waypoints forming a square in Cartesian space
  Tests: multi-segment trajectory, smooth transitions

Demo 3: CIRCLE TRACKING (Cartesian)
  End-effector traces a circle in a plane
  Tests: continuous Cartesian trajectory + IK/OSC at high frequency

Demo 4: SINGULARITY STRESS TEST
  Push the robot near each singularity type while tracking a trajectory
  Tests: damped LS, adaptive λ, graceful degradation
```

### Tasks

1. Integrate the full pipeline: target_poses → IK → trajectory → controller → MuJoCo → render
2. Add a simple gripper to the UR5e model (parallel jaw — just two prismatic joints)
3. Implement and run Demo 1: pick and place with grasp/release sequencing
4. Implement and run Demo 2: multi-waypoint tour
5. Implement and run Demo 3: circle tracking
6. Implement and run Demo 4: singularity stress test
7. Record a video for each demo (MuJoCo rendering or Matplotlib 3D animation)
8. Create a metrics dashboard: tracking error, joint torques, manipulability, joint limit proximity, computation time per step

### Verification

- [ ] Pick-and-place completes reliably (object reaches target location)
- [ ] Multi-waypoint path is smooth (no jerky transitions)
- [ ] Circle tracking produces an actual circle (< 5 mm RMS tracking error)
- [ ] Singularity test degrades gracefully (no joint velocity explosion)
- [ ] At least 2 demos have recorded videos
- [ ] Metrics dashboard exists with 4+ metrics plotted

---

## 4.2 ROS2 + MoveIt2 Bridge (Day 10, ~5 hours)

### Learning Goal

Connect your MuJoCo simulation to the ROS2 ecosystem via a bridge node, and use MoveIt2 for motion planning.

### Architecture

```text
┌──────────────────────┐
│  MoveIt2              │
│  (motion planning)    │
│                       │
│  • OMPL planners      │
│  • Collision checking │
│  • Trajectory exec    │
└──────────┬───────────┘
           │ /joint_trajectory
           │ (FollowJointTrajectory action)
┌──────────▼───────────┐
│  MuJoCo Bridge Node  │
│                       │
│  Subscribes:          │
│    /joint_trajectory  │
│  Publishes:           │
│    /joint_states      │
│    /ee_pose           │
│  Services:            │
│    /get_robot_state   │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  MuJoCo Simulation   │
│  (physics + render)   │
└──────────────────────┘
```

**What This Enables**

```text
With this bridge:
  1. You can use MoveIt2's motion planners (RRT, PRM, etc.)
  2. MoveIt2's collision checking layer sits on top of your sim
  3. You can use rviz2 to visualize and command the robot
  4. Your MuJoCo sim becomes a "hardware" backend for the whole ROS2 stack
  5. Same interface you'd use for a real UR5e robot
```

### Tasks

1. Create a ROS2 node that wraps the MuJoCo simulation:
   - Subscribes to `/joint_trajectory` (trajectory commands)
   - Publishes `/joint_states` at simulation rate
   - Publishes `/ee_pose` (end-effector transform)
2. Write a `ur5e_moveit_config` package (or adapt from existing):
   - URDF with collision meshes
   - SRDF with planning groups
   - kinematics.yaml (KDL or your analytical solver)
   - joint_limits.yaml
3. Connect MoveIt2 to your bridge node via the FollowJointTrajectory action interface
4. Plan and execute a motion in rviz2 — verify the MuJoCo robot follows
5. Run the pick-and-place demo through MoveIt2 instead of your custom planner
6. Compare: your custom planner vs. MoveIt2's OMPL planner — path quality and planning time

### Verification

- [ ] `ros2 topic echo /joint_states` shows correct values from MuJoCo
- [ ] MoveIt2 plans a motion in rviz2 and the MuJoCo robot executes it
- [ ] Collision avoidance works through MoveIt2
- [ ] Pick-and-place succeeds through the MoveIt2 pipeline
- [ ] Bridge node handles real-time simulation loop without stalling

---

# PHASE 5 — PORTFOLIO + VLA BRIDGE (Days 11–12)

---

## 5.1 VLA Bridge Document (Day 11, ~4 hours)

### Learning Goal

Explicitly connect every concept from this lab to your humanoid VLA project.

### Document Structure

```text
VLA BRIDGE: How This Lab Maps to Humanoid Robotics
├── FK/DH → Action space definition
│   How the VLA model's output actions map to joint configurations
│   Why the action head needs to respect the kinematic structure
│
├── Jacobian → Velocity constraints + safety
│   Joint velocity limits in the data generation pipeline
│   Manipulability weighting for configuration selection
│   Singularity avoidance in the IK-based data generator
│
├── IK → Data generation backbone
│   Your current humanoid_vla uses IK to generate training data
│   Analytical vs. numerical tradeoffs for Unitree G1
│   Solution selection strategy for training data quality
│
├── Dynamics → Torque-aware policy
│   Why dynamics matter for sim-to-real transfer
│   Gravity compensation in the action head
│   Feasibility checking for generated actions
│
├── Trajectory → Motion primitives
│   How trajectory generation connects to the ACT model's output
│   Temporal action chunking in context of polynomial trajectories
│   Cartesian vs. joint space action representation tradeoffs
│
├── Control → Sim-to-real gap
│   Controller design affects what the policy needs to learn
│   OSC as the low-level controller beneath the VLA policy
│   Impedance control for safe human-robot interaction
│
└── ROS2/MoveIt2 → Deployment infrastructure
    How the trained policy integrates with the robot stack
    Bridge pattern: policy output → controller → hardware/sim
```

### Tasks

1. Write the VLA bridge document following the structure above
2. For each section, provide a concrete example from your humanoid_vla project
3. Identify 3 things in your current humanoid_vla pipeline that you would now change or improve
4. Create a "lessons learned" section with specific numerical results from the lab

---

## 5.2 Portfolio Package (Day 12, ~5 hours)

### Learning Goal

Package everything into a portfolio-ready repository.

### README Structure

```text
README.md
├── Overview (what + why + who)
├── Architecture diagram (the pipeline visual)
├── Tech stack (MuJoCo, Pinocchio, ROS2, Python)
├── Key results
│   ├── FK validation: DH vs. Pinocchio (< 0.001m)
│   ├── IK success rate: X% on Y targets
│   ├── OSC tracking error: X mm RMS
│   └── Singularity handling: graceful degradation demo
├── Demo videos (embedded or linked)
├── How to run
├── Project structure
└── Connection to broader work (link to humanoid_vla)
```

### Tasks

1. Write a polished README.md with all sections above
2. Create a 60-second demo reel combining clips from the best demos
3. Compile a results table with quantitative metrics from all phases
4. Write the interview cheatsheet: `docs/interview_cheatsheet.md`
   - "Explain FK vs. IK" — your answer with UR5e examples
   - "What is a Jacobian?" — your 3-sentence answer
   - "How do you handle singularities?" — damped LS, adaptive λ, manipulability
   - "Explain OSC" — task-space control with inertia shaping
   - "How does sim-to-real transfer work?" — dynamics, controller, domain randomization
5. Clean up code: docstrings, type hints, consistent naming
6. Push to GitHub — make it a public portfolio piece

### Verification

- [ ] README is clear enough for a hiring manager to understand the project in 2 minutes
- [ ] At least one demo video is embedded/linked
- [ ] Interview cheatsheet covers 5+ common robotics questions
- [ ] Code is clean and well-documented
- [ ] Repository is public on GitHub

---

# CLAUDE CODE PROMPT TEMPLATE

Use this template for each step when working with Claude Code:

```text
CONTEXT:
I am building a 6-DOF robotics lab with UR5e, MuJoCo, and Pinocchio.
I have completed Phase X and I am now on step Y.Z.

Previous work:
- [List relevant completed steps and their output files]

TASK:
[Copy the task list from the relevant section]

CONSTRAINTS:
- Python 3.10+, numpy, mujoco, pinocchio, matplotlib, meshcat (optional)
- Use Pinocchio for all analytical computations (FK, Jacobian, dynamics, IK)
- Use MuJoCo for simulation and rendering
- Code should be well-commented in English
- Every function must include docstrings and type hints
- Test code in separate files under tests/
- Model files under models/

VERIFICATION:
[Copy verification criteria from the relevant section]

OUTPUT FORMAT:
- Write and run the code
- Show results (plot, table, or print output)
- If there is an error, fix it
- Save outputs to the appropriate directory
```

---

# QUICK REFERENCE

## DH Transform

```text
T(α, a, d, θ) = Rot_x(α) · Trans_x(a) · Trans_z(d) · Rot_z(θ)
```

## 6×6 Jacobian (Geometric)

```text
J[:,i] = | z(i-1) × (p_ee - p(i-1)) |   ← linear
         | z(i-1)                     |   ← angular
```

## Analytical IK (UR5e key insight)

```text
Spherical wrist → decouple position (q1,q2,q3) and orientation (q4,q5,q6)
Up to 8 solutions → selection policy needed
```

## Dynamics

```text
τ = M(q)·q̈ + C(q,q̇)·q̇ + g(q)
Pinocchio: rnea(), aba(), crba(), computeGeneralizedGravity()
```

## OSC (Operational Space Control)

```text
τ = Jᵀ · Λ · (ẍ_d + Kp·e_x + Kd·ė_x) + Jᵀ·μ + g(q)
Λ = (J·M⁻¹·Jᵀ)⁻¹
```

## Trajectory Types

```text
Cubic:       smooth, simple, zero velocity at endpoints
Quintic:     smoother, zero velocity + acceleration at endpoints
Trapezoidal: industrial standard, explicit velocity/accel limits
Min-jerk:    s(t) = 10(t/T)³ - 15(t/T)⁴ + 6(t/T)⁵
SLERP:       orientation interpolation on SO(3)
```