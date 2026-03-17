# MuJoCo + Robotics Fundamentals: 2-Day Crash Course

**Person:** Ozkan — AI Engineer with a mechatronics background, experienced in VLA/IK pipelines
**Goal:** Refresh and deepen kinematics fundamentals, then turn them into hands-on MuJoCo practice
**Dates:** March 9–10, 2026
**Method:** Each step is self-contained and can be given directly as a prompt to Claude Code, Codex, or Gemini CLI

---

## Architecture Overview

```text
DAY 1: THEORY → SIMULATION
┌─────────────────────────────────────────────────┐
│  A1: MuJoCo Environment       (1 hour)          │
│  A2: FK — Forward Kinematics  (2 hours)         │
│  A3: Jacobian                 (2 hours)         │
│  A4: IK — Inverse Kinematics  (2.5 hours)       │
│  -- Break + Review --                            │
│  A5: Dynamics Fundamentals    (1 hour)          │
└─────────────────────────────────────────────────┘
                      ↓
DAY 2: INTEGRATION → DEMO
┌─────────────────────────────────────────────────┐
│  B1: Trajectory Generation     (2 hours)        │
│  B2: PD/PID Controller         (1.5 hours)      │
│  B3: MuJoCo Full Pipeline      (2.5 hours)      │
│  B4: ROS2 Bridge (optional)    (2 hours)        │
│  B5: Wrap-up + Documentation   (1 hour)         │
└─────────────────────────────────────────────────┘
```

---

# DAY 1 — THEORY + SIMULATION FOUNDATIONS (~8.5 hours)

---

## A1: MuJoCo Environment Setup (1 hour)

### Learning Goal

Understand how MuJoCo works as a physics engine and become familiar with the MJCF format.

### Concepts You Need to Know

**MuJoCo vs. Other Simulators**

* MuJoCo = *Multi-Joint dynamics with Contact*, optimized for contact-rich simulation.
* Compared with Gazebo and Isaac Sim, it is lighter, faster, and more ML-friendly through its Python API.
* Since you have already explored Isaac Sim and Gazebo, MuJoCo’s main advantage for you is fast headless batch simulation.

**MJCF Structure (Mental Model)**

```xml
<mujoco>
  ├── <option>        ← physics parameters (gravity, timestep)
  ├── <worldbody>     ← scene hierarchy (body → joint → geom)
  ├── <actuator>      ← motors connected to joints
  └── <sensor>        ← measurements (position, velocity, force)
```

**Critical Concept — Body vs. Geom vs. Joint**

* **Body:** A coordinate frame. Invisible reference point.
* **Geom:** The physical shape used for collision and visuals. Attached to a body.
* **Joint:** The degree of freedom between bodies. Defined inside a body.

### Tasks

1. Run `pip install mujoco` and verify with `import mujoco; print(mujoco.__version__)`.
2. Write an MJCF file for a 2-link planar robot using the specification below.
3. Launch the simulation, run 100 steps, and read the joint positions.

### Robot Specification

```text
Robot: 2-Link Planar Manipulator
- Link 1: length 0.3 m, rotates around z-axis (hinge)
- Link 2: length 0.3 m, rotates around z-axis (hinge)
- 2 motors (one per joint, ctrl range: [-10, 10])
- 2 position sensors (one per joint)
- Plane: XY plane (gravity off or along z, robot operates horizontally)
```

### Verification Criteria

* [ ] `mj_step()` runs without errors
* [ ] `data.qpos` returns a 2-element array
* [ ] Joint angles change when motor commands are applied

---

## A2: Forward Kinematics (2 hours)

### Learning Goal

Answer the question: *Given the joint angles, where is the end-effector?*

### Theory You Need to Know

**Why Not DH Parameters Yet**
For a 2-link planar robot, DH parameters are unnecessary overhead. A geometric approach is enough and more intuitive.
You can learn DH later when moving to 6+ DOF robots.

**Geometric FK — 2-Link Planar**

```text
Input: θ₁, θ₂ (joint angles)
       L₁, L₂ (link lengths)

Joint 1 position (base): (0, 0)

Joint 2 position:
  x₁ = L₁ · cos(θ₁)
  y₁ = L₁ · sin(θ₁)

End-effector position:
  x₂ = L₁ · cos(θ₁) + L₂ · cos(θ₁ + θ₂)
  y₂ = L₁ · sin(θ₁) + L₂ · sin(θ₁ + θ₂)
```

**Homogeneous Transformation Refresher**

```text
T = | R  p |    R: 2x2 rotation matrix
    | 0  1 |    p: 2x1 position vector

T₀₁ = transform base → joint 2
T₀₂ = T₀₁ · T₁₂ = transform base → end-effector

Remember this chain rule — the same idea scales to 6DOF.
```

**Why This Matters**

* FK is the foundation of IK.
* The Jacobian is the derivative of FK.
* In a VLA pipeline, action → joint-space mapping is built on this.

### Tasks

1. On paper, manually compute the end-effector position for θ₁ = 30° and θ₂ = 45°.
2. Write an FK function in Python using NumPy and trigonometry.
3. Read `data.xpos` from MuJoCo and compare it with your FK result.
4. Build a comparison table for 10 angle combinations: FK vs. MuJoCo.

### Verification Criteria

* [ ] Hand calculation matches Python result (< 0.001 error)
* [ ] Python FK matches MuJoCo `xpos` (< 0.01 error)
* [ ] You can draw the robot configuration with Matplotlib as a stick figure

---

## A3: Jacobian (2 hours)

### Learning Goal

Answer the question: *What is the relationship between joint velocities and end-effector velocity?*

### Theory You Need to Know

**What Is the Jacobian — Intuition**

```text
ẋ = J(θ) · θ̇

ẋ  = end-effector velocity vector [vx, vy]   (2x1)
θ̇  = joint velocity vector [θ̇₁, θ̇₂]         (2x1)
J  = Jacobian matrix                         (2x2)
```

The Jacobian is a velocity transformer from joint space to Cartesian space.

**2-Link Planar Robot Jacobian**

```text
J = | -L₁·sin(θ₁) - L₂·sin(θ₁+θ₂)    -L₂·sin(θ₁+θ₂) |
    |  L₁·cos(θ₁) + L₂·cos(θ₁+θ₂)     L₂·cos(θ₁+θ₂) |
```

These are just the partial derivatives of FK:

```text
J[0,0] = ∂x/∂θ₁,  J[0,1] = ∂x/∂θ₂
J[1,0] = ∂y/∂θ₁,  J[1,1] = ∂y/∂θ₂
```

**Critical Concepts**

1. **Singularity:** when `det(J) = 0`, the robot loses motion capability.

   * For the 2-link case: θ₂ = 0 or θ₂ = π (arm fully stretched or folded flat)
   * Near singularities, IK becomes ill-conditioned or ambiguous.
   * `det(J)` also acts as a manipulability measure.

2. **Jacobian Transpose vs. Pseudo-inverse**

```text
J^T  : simple, stable, slower convergence → useful in controllers
J^+  : faster convergence, sensitive near singularities → useful in IK solvers
J^+ = J^T · (J · J^T)^(-1)
```

3. **Why This Matters for You**

* Task-space control for humanoids such as Unitree G1 depends on Jacobians.
* Joint velocity limits and safety constraints in IK-based pipelines are tied to the Jacobian.

### Tasks

1. Derive the analytical Jacobian from the FK equations, first on paper, then in code.
2. Compute a numerical Jacobian with finite differences and compare it to the analytical one.
3. Compute `det(J)` for different θ₂ values and observe the singularity behavior.
4. Compare your Jacobian with MuJoCo’s `mj_jac()`.

### Verification Criteria

* [ ] Analytical J matches numerical J (< 0.0001 error)
* [ ] Your Jacobian matches MuJoCo’s Jacobian
* [ ] As θ₂ → 0, `det(J)` → 0 in your plot
* [ ] You can explain “What is a Jacobian?” in 3 sentences

---

## A4: Inverse Kinematics (2.5 hours)

### Learning Goal

Answer the question: *What joint angles are needed to move the end-effector to a target position?*

### Theory You Need to Know

**Two Main Approaches**

```text
┌─────────────────────────────────────────────┐
│             IK Solution Methods             │
├──────────────────┬──────────────────────────┤
│  ANALYTICAL      │  NUMERICAL               │
│  - Closed form   │  - Iterative             │
│  - Fast          │  - General purpose       │
│  - Only for      │  - Works for any robot   │
│    simple robots │  - Singularity risk      │
└──────────────────┴──────────────────────────┘
```

**Analytical IK — 2-Link Case (Cosine Rule)**

```text
Input: (x_target, y_target), L₁, L₂

1. Reachability check:
   d = sqrt(x² + y²)
   Reachable ↔ |L₁ - L₂| ≤ d ≤ L₁ + L₂

2. Compute θ₂ (cosine rule):
   cos(θ₂) = (x² + y² - L₁² - L₂²) / (2·L₁·L₂)
   θ₂ = ±acos(cos(θ₂))     ← TWO SOLUTIONS! (elbow up / elbow down)

3. Compute θ₁:
   θ₁ = atan2(y, x) - atan2(L₂·sin(θ₂), L₁ + L₂·cos(θ₂))
```

**Numerical IK — Jacobian Pseudo-inverse**

```text
repeat:
  1. Compute current position: p_current = FK(θ)
  2. Compute error: Δx = p_target - p_current
  3. Compute joint correction: Δθ = J⁺(θ) · Δx
  4. Update: θ = θ + α · Δθ
until ||Δx|| < tolerance
```

**Damped Least Squares (More Stable)**

```text
Δθ = J^T · (J·J^T + λ²·I)^(-1) · Δx
λ = damping factor
```

**Why You Should Know Both**

* Analytical IK is fast and exact for simple robots.
* Numerical IK is the only realistic option for high-DOF systems such as Unitree G1.
* Your existing humanoid IK pipeline already uses numerical IK — now you understand the mechanics behind it.

### Tasks

1. Implement analytical IK and return both elbow-up and elbow-down solutions.
2. Implement numerical IK using the Jacobian pseudo-inverse.
3. Implement a Damped Least Squares version.
4. Compare all three methods on 20 random target points: success rate and iteration count.
5. Apply the IK solutions in MuJoCo and verify that the robot reaches the target.

### Verification Criteria

* [ ] Analytical IK returns correct elbow-up and elbow-down solutions
* [ ] Numerical IK reaches 95%+ success on reachable targets
* [ ] Damped LS remains stable near singularities
* [ ] The MuJoCo robot reaches the commanded target

---

## A5: Dynamics Fundamentals (1 hour)

### Learning Goal

Understand the basis of the question: *How much torque should the motors apply?*

### Theory You Need to Know

**Robot Dynamics Equation**

```text
τ = M(θ)·θ̈ + C(θ,θ̇)·θ̇ + g(θ)

τ      : joint torques
M(θ)   : inertia matrix
C(θ,θ̇): Coriolis + centrifugal terms
g(θ)   : gravity terms
```

**Why This Matters — Without Going Too Deep Yet**

* MuJoCo solves this equation internally every time you call `mj_step()`.
* But when you write a PD controller, gravity compensation may matter.
* `data.qfrc_bias` exposes MuJoCo’s estimate of `C·θ̇ + g`.

**What Is Enough for Now**

* Know conceptually what M, C, and g mean.
* Learn how MuJoCo exposes these terms through the API.
* If needed, deeper dynamics can become a later sprint after Day 2.

### Tasks

1. Read `data.qfrc_bias` and `data.qM` from the MuJoCo API.
2. Toggle gravity on/off and observe how `qfrc_bias` changes.
3. Answer the conceptual question: *Why does M(θ) depend on configuration?*

### Verification Criteria

* [ ] You can explain `τ = M·θ̈ + C·θ̇ + g`
* [ ] You can read dynamics-related values from MuJoCo

---

# DAY 2 — INTEGRATION + DEMO (~9 hours)

---

## B1: Trajectory Generation (2 hours)

### Learning Goal

Plan *how* the robot should move from point A to point B.

### Theory You Need to Know

**Two Planning Spaces**

```text
┌──────────────────────────────────────────────┐
│         Where Is the Trajectory Planned?     │
├─────────────────────┬────────────────────────┤
│  JOINT SPACE        │  CARTESIAN SPACE       │
│  θ(t): angle interp │  x(t): position interp │
│  Simple, fast       │  Straight EE path      │
│  EE path may curve  │  More expensive        │
│                     │  (requires IK each step)│
└─────────────────────┴────────────────────────┘
```

**Cubic Polynomial Trajectory (Joint Space)**

```text
θ(t) = a₀ + a₁·t + a₂·t² + a₃·t³

Boundary conditions:
  θ(0)  = θ_start
  θ(T)  = θ_end
  θ̇(0) = 0
  θ̇(T) = 0

Solution:
  a₀ = θ_start
  a₁ = 0
  a₂ = 3(θ_end - θ_start) / T²
  a₃ = -2(θ_end - θ_start) / T³
```

**Quintic Polynomial — Smoother Motion**

```text
θ(t) = a₀ + a₁·t + a₂·t² + a₃·t³ + a₄·t⁴ + a₅·t⁵
```

With 6 boundary conditions, you constrain position, velocity, and acceleration at both ends.
That gives smoother motion and less stress on the actuators.

**Cartesian Trajectory + IK**

```text
for t in timeline:
  x_desired(t) = interpolate(x_start, x_end, t)
  θ(t) = IK(x_desired(t))
  send θ(t) to robot
```

### Tasks

1. Implement a cubic polynomial trajectory generator in joint space.
2. Compare cubic and quintic trajectories by plotting velocity and acceleration profiles.
3. Implement a Cartesian straight-line trajectory using linear interpolation + IK.
4. Plot and compare the end-effector paths for both approaches.

### Verification Criteria

* [ ] Cubic trajectory has zero start/end velocity
* [ ] Quintic trajectory has zero start/end velocity and acceleration
* [ ] Cartesian trajectory produces a straight end-effector path
* [ ] Joint-space trajectory produces a curved end-effector path, as expected

---

## B2: PD/PID Controller (1.5 hours)

### Learning Goal

Design a control law that tracks the trajectory.

### Theory You Need to Know

**PD Controller (Robotics Standard)**

```text
τ = Kp · (θ_desired - θ_actual) + Kd · (θ̇_desired - θ̇_actual)
```

* `Kp`: position error gain
* `Kd`: velocity error gain, adds damping

**PD + Gravity Compensation**

```text
τ = Kp · e + Kd · ė + g(θ)
```

* `g(θ)` can be approximated from MuJoCo through `data.qfrc_bias`.
* This prevents the motor from wasting effort only to fight gravity.

**Tuning Strategy**

```text
1. Start with Kd = 0 and increase Kp until oscillation starts
2. Reduce Kp slightly
3. Add Kd until oscillation is damped out
4. Increase both gradually to balance speed and stability

Typical starting point: Kp = 100, Kd = 10
```

**Why PD Instead of PID**

* In robot dynamics, the integral term is often unnecessary.
* Gravity compensation already reduces steady-state error.
* The integral term can introduce windup.

### Tasks

1. Implement a PD controller for a single target angle.
2. Add gravity compensation.
3. Use PD to track the cubic trajectory from B1.
4. Run Kp/Kd tuning experiments and plot tracking error.
5. Visualize the result in MuJoCo.

### Verification Criteria

* [ ] The robot reaches the target and settles (overshoot < 5%)
* [ ] Trajectory tracking error < 0.05 rad
* [ ] You observed the difference with and without gravity compensation

---

## B3: MuJoCo Full Pipeline (2.5 hours)

### Learning Goal

Combine all pieces into a working demo.

### Pipeline Architecture

```text
                    ┌──────────────┐
                    │   TARGET     │
                    │   (x, y)     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   IK SOLVER  │
                    │   analytical │
                    │   or         │
                    │   numerical  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  TRAJECTORY  │
                    │  GENERATOR   │
                    │  cubic /     │
                    │  quintic     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  PD          │
                    │  CONTROLLER  │
                    │  + gravity   │
                    │  comp.       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  MuJoCo      │
                    │  mj_step()   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ VIDEO / PLOT │
                    │ OUTPUT       │
                    └──────────────┘
```

### Demo Scenarios

```text
Demo 1: Pick and Place
  Move end-effector from (0.2, 0.3) → (0.4, 0.1) → (0.2, 0.3)

Demo 2: Circle Tracking
  Make the end-effector trace a circle using Cartesian trajectory + IK

Demo 3: Singularity Edge
  Push the robot near a singularity and compare damped LS vs. naive IK
```

### Tasks

1. Integrate the full pipeline: target → IK → trajectory → PD → MuJoCo → render.
2. Run Demo 1, Demo 2, and Demo 3.
3. Record a video for each demo using MuJoCo rendering or Matplotlib animation.
4. Log and plot metrics such as tracking error and joint torque.

### Verification Criteria

* [ ] Pick-and-place works reliably
* [ ] Circle tracking produces an actual circular path
* [ ] At least one demo has a recorded video
* [ ] A metrics dashboard exists for tracking error, torque, and joint angles

---

## B4: ROS2 Bridge — OPTIONAL (2 hours)

### Why Optional?

You already know ROS2. This step is less about learning and more about integration engineering, so it can be delegated to Claude Code or Codex if needed.

### If You Do It — Architecture

```text
┌─────────────────┐     /joint_command      ┌──────────────┐
│   ROS2 Node     │ ──────────────────────→ │  MuJoCo      │
│   (commander)   │                          │  Bridge Node │
│                 │ ←────────────────────── │              │
└─────────────────┘     /joint_state         └──────────────┘

Topics:
  /joint_command  (Float64MultiArray)  → target joint angles
  /joint_state    (JointState)         → current joint state
  /ee_pose        (Pose)               → end-effector position
```

### Tasks

1. Wrap the MuJoCo simulation as a ROS2 node.
2. Add a `/joint_command` subscriber and connect it to the PD controller.
3. Publish `/joint_state`.
4. Send commands from a second node and move the robot.

### Verification Criteria

* [ ] `ros2 topic pub` moves the robot
* [ ] `ros2 topic echo /joint_state` shows correct values

---

## B5: Wrap-up + Documentation (1 hour)

### Tasks

1. Write a `README.md` explaining what you learned and what each file does.
2. Save the test results for each module.
3. Write a short 3-point note on how this connects back to your humanoid VLA work:

   * FK/IK → VLA action-space mapping
   * Jacobian → joint velocity limits and safety constraints
   * Trajectory generation → motion planning in the data generation pipeline

---

# CLAUDE CODE / CODEX USAGE GUIDE

## Prompt Template for Each Step

```text
CONTEXT:
I am learning kinematics with a 2-link planar robot.
I am using MuJoCo simulation.
The goal of this step is: [STEP NAME]

TASK:
[Copy the task list here]

CONSTRAINTS:
- Python 3.10+, numpy, mujoco, matplotlib
- Code should be well-commented (Turkish or English)
- Every function must include docstrings and type hints
- Test code should be in a separate file
- MJCF file should be a separate .xml file

VERIFICATION:
[Copy the verification criteria here]

OUTPUT FORMAT:
- Write and run the code
- Show results (plot, table, or print output)
- If there is an error, fix it
```

## Suggested Project Structure

```text
mujoco-robotics-crashcourse/
├── models/
│   └── two_link.xml
├── src/
│   ├── forward_kinematics.py
│   ├── jacobian.py
│   ├── inverse_kinematics.py
│   ├── trajectory.py
│   ├── pd_controller.py
│   └── full_pipeline.py
├── tests/
│   ├── test_fk.py
│   ├── test_jacobian.py
│   ├── test_ik.py
│   └── test_trajectory.py
├── demos/
│   ├── demo_pick_place.py
│   ├── demo_circle.py
│   └── demo_singularity.py
├── ros2_bridge/
│   ├── mujoco_bridge.py
│   └── commander.py
└── README.md
```

---

# QUICK REFERENCE CARDS

## FK

```text
x = L₁·cos(θ₁) + L₂·cos(θ₁+θ₂)
y = L₁·sin(θ₁) + L₂·sin(θ₁+θ₂)
```

## Jacobian

```text
J = | ∂x/∂θ₁  ∂x/∂θ₂ |
    | ∂y/∂θ₁  ∂y/∂θ₂ |

Singularity: det(J) = 0 → θ₂ = 0 or π
```

## Analytical IK

```text
cos(θ₂) = (x²+y²-L₁²-L₂²) / (2·L₁·L₂)
θ₂ = ±acos(...)
θ₁ = atan2(y,x) - atan2(L₂·sin(θ₂), L₁+L₂·cos(θ₂))
```

## Numerical IK

```text
repeat: Δθ = J⁺ · (x_target - FK(θ))
        θ += α · Δθ
```

## PD Control

```text
τ = Kp·(θ_d - θ) + Kd·(θ̇_d - θ̇) + g(θ)
```

## Cubic Trajectory

```text
θ(t) = θ_start + (3/T²)·Δθ·t² - (2/T³)·Δθ·t³
```
