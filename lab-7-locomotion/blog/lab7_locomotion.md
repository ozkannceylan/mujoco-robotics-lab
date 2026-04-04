# Why Making a Humanoid Walk is Harder Than It Looks

*Lab 7 of a robotics portfolio series using MuJoCo and Pinocchio*

---

You know that feeling when you watch a toddler stumble across a room and think, "How hard can it be?" After spending weeks trying to make a simulated humanoid take a single step forward, I can tell you: it is extraordinarily hard. And the reasons why are both humbling and illuminating.

This post covers Lab 7 of my robotics lab series — the transition from fixed-base manipulation to floating-base locomotion. I moved from a UR5e industrial arm bolted to a table (Labs 1–6) to a Unitree G1 humanoid standing on its own two feet. The goal was simple: make it walk. The outcome was more nuanced than I expected.

## The Setup

The Unitree G1 is a 29-DOF humanoid weighing 33.34 kg, standing about 0.79 meters tall. I used the MuJoCo Menagerie model — real STL meshes, calibrated inertias, proper actuator gains. For analytical computation (forward kinematics, Jacobians, inverse kinematics), I used Pinocchio, the same library that powered Labs 2–6. MuJoCo handles the physics simulation.

The first challenge hit before any control code was written: **floating-base robots are fundamentally different from fixed-base ones.**

In Labs 1–6, the UR5e was bolted to a table. Joint 0 was the robot's root. Every joint angle mapped directly to a velocity. The configuration space was simple: 6 joints, 6 velocities.

With the G1, the root is a *freejoint* — the robot can translate and rotate freely in space. This freejoint uses a quaternion for orientation, which has 4 components but only 3 degrees of freedom (it lives on the unit sphere S3). So the configuration has 36 dimensions (7 for the base + 29 joints) but the velocity space has only 35 (6 for the base + 29 joints). This means you can't just add velocities to configurations — you need the exponential map on the quaternion manifold. In Pinocchio, this is `pin.integrate(model, q, dq)`. Getting this wrong produces subtle, maddening bugs.

Then there's the quaternion convention mismatch: MuJoCo stores (w, x, y, z), Pinocchio expects (x, y, z, w). And the MJCF model defines the pelvis at z=0.793m, which Pinocchio bakes into the joint placement, requiring a Z offset in every conversion. These are the kinds of details that consume days of debugging if you're not careful.

## What Worked Beautifully

### Standing Balance (Milestone 1)

The first real win was standing. The G1 uses position servos: `tau = Kp * (ctrl - qpos)`, with Kp=500. To add gravity compensation, you set:

```
ctrl = q_ref + qfrc_bias / Kp
```

This gives you PD control plus gravity compensation through position actuator offsets. The result? The robot stands for 10 seconds with only 1.6mm of pelvis height deviation. It recovers from a 5N lateral push without breaking a sweat.

This was deceptively easy. The stiff Kp=500 servos act as strong springs, and with gravity compensation baked in, the standing pose is rock-solid. I should have been more suspicious of how easy it was.

### Pinocchio-MuJoCo Cross-Validation (Milestones 2, 3a, 3b)

One thing I'm proud of: the validation infrastructure. I verified every analytical quantity against the physics simulator:

- **CoM position**: Pinocchio vs MuJoCo agrees to 0.000mm (after fixing the Z offset bug)
- **Foot FK at 10 random configurations**: 0.000mm error
- **Jacobian columns**: Validated all 12 leg joints via central finite differences — max error 1.09e-07

This was worth every minute invested. When the walking pipeline later failed, I knew with certainty that the kinematic computations were correct. The bug had to be in the dynamics, not the math.

### Whole-Body Inverse Kinematics (Milestone 3c)

I built a stacked Jacobian solver with damped least squares (DLS) that handles 18 simultaneous task dimensions: both feet (6D each), CoM XY, pelvis height, and pelvis orientation. Feed it a CoM target, and it computes the joint angles that achieve that CoM position while keeping both feet planted and the pelvis upright.

The IK naturally discovers the correct joint couplings — for a lateral CoM shift, it uses hip roll and ankle roll; for a forward shift, hip pitch and ankle pitch. These are exactly the joints a human would expect, but the IK derives them analytically from the Jacobian structure. No hand-tuning required.

### Weight Shift in Simulation (Milestone 3d)

The crown jewel of Lab 7. I shift the robot's center of mass 53.5mm laterally (over the left foot) while both feet stay planted with less than 1.4mm of drift. The control pipeline:

1. Settle the robot for 1 second
2. Convert MuJoCo state to Pinocchio coordinates
3. Pre-compute IK at 11 waypoints along the shift path
4. Cosine-smooth interpolation through the waypoints in simulation
5. Gravity feedforward + velocity damping (K_VEL=40)

That velocity damping constant — K_VEL=40 — is the single most important parameter. The Menagerie G1 actuators have zero built-in damping (Kd=0). Without explicit velocity damping, the Kp=500 servos are critically underdamped. K_VEL=40 gives a damping ratio of about 0.89 — near-critical, which suppresses oscillation without excessive sluggishness.

## Where It All Fell Apart

With standing, cross-validation, IK, and weight shift all working, the walking pipeline should have been straightforward. I had every piece:

- **LIPM preview control** (Kajita 2003): Generates optimal CoM trajectories that anticipate future footstep transitions
- **ZMP reference generator**: Piecewise-constant during single support, linearly interpolated during double support
- **Swing foot trajectory**: Cubic horizontal interpolation with parabolic vertical clearance
- **Whole-body IK**: Converts CoM + foot targets to joint angles

I tried six different approaches to execute a single forward step. Every single one ended with the robot on the ground.

| Attempt | What I Changed | When It Fell |
|---------|---------------|-------------|
| 1 | Basic pipeline (Kp=400, 1.5s single support) | 3.64s |
| 2 | Added gravity feedforward + ankle feedback | 3.79s |
| 3 | Higher gains (Kp=800), shorter single support (0.5s) | 3.78s (survived first step, fell on second) |
| 4 | Skip alignment step, keep both feet planted | 4.04s |
| 5 | Custom manual ZMP reference, no alignment | 2.84s (worse!) |
| 6 | Online IK with measured base state | 3.04s |

### The Root Cause

The failure isn't in any single component. Each module works correctly in isolation. The problem is the **execution gap** — the chasm between kinematic planning (which assumes perfect joint tracking) and dynamic simulation (where tracking errors compound and destabilize the system).

Here's the core issue: **position actuators cannot provide ZMP control.**

ZMP-based walking requires precise control of ankle torques — that's how you keep the zero-moment point inside the stance foot. But position actuators track joint angles, not torques. When the robot enters single support:

1. The IK plans joint angles assuming they'll be tracked perfectly
2. The servos track those angles with some lag
3. The tracking errors create unintended ankle torques
4. The unintended torques shift the actual ZMP away from the planned ZMP
5. The CoM accelerates in the wrong direction
6. The error compounds exponentially — within a second, the robot falls

This is a structural limitation, not a tuning problem. I could have spent another month adjusting gains and timing, and it would not have worked. The control architecture fundamentally cannot command the torques needed for single-support balance.

## What I Learned

### The Textbook Pipeline Has a Hidden Assumption

Every robotics textbook teaches ZMP + LIPM + IK as the walking pipeline. What they don't emphasize is that this pipeline assumes **perfect torque control**. When you're writing equations on a whiteboard, this assumption is invisible. When you're running a simulation with position actuators, it's fatal.

The textbooks aren't wrong — ZMP walking works brilliantly on robots with torque-controlled joints (Honda's ASIMO, Boston Dynamics' Atlas). But the gap between "here's the theory" and "here's how to actually implement it on your specific actuator model" is enormous and rarely discussed.

### Why RL Has Won

This experience crystallized for me why reinforcement learning has become the dominant approach for humanoid locomotion. RL doesn't need to model the ZMP or solve Riccati equations. It learns a direct mapping from state to actuator commands through millions of simulation episodes, implicitly compensating for actuator dynamics, contact friction, and all the other factors that make analytical approaches fragile.

The RL approach doesn't care whether your actuators are position-controlled, torque-controlled, or somewhere in between. It learns the mapping that works for your specific robot, in your specific simulator, with your specific terrain. This generality is why projects like Isaac Lab train humanoids on GPU-parallelized environments — brute-force learning sidesteps the execution gap entirely.

### Pre-Computed Beats Online for Position Actuators

One tactical lesson: for quasi-static motions on position-actuated robots, pre-compute the entire trajectory offline and replay it. Five attempts at online IK feedback (updating targets every timestep based on measured state) all failed. The kinematic Jacobian predicts different CoM-to-joint sensitivity than what the dynamics produce, because servo torque reactions dominate over kinematic chain predictions.

Pre-computing IK at discrete waypoints and cosine-smoothing between them works reliably. The PD controller handles small tracking errors, and gravity compensation keeps the robot upright. No feedback loop, no instability.

### Damping is Everything

The Menagerie G1 has Kd=0 on all actuators. Adding explicit velocity damping (`ctrl -= K_VEL * qvel / Kp`) with K_VEL=40 was the single most important change in the entire lab. Without it, nothing works — not standing, not weight shift, nothing. K_VEL = 2*sqrt(Kp) gives near-critical damping. Remember this if you ever work with the G1 in MuJoCo.

## What's Next

Lab 7 ends at the boundary between static and dynamic balance. The robot can stand, recover from pushes, and shift its weight — all reliably. It cannot walk.

Lab 8 (Whole-Body Loco-Manipulation) will need to cross that boundary. The path forward likely involves either:
- Switching to torque-controlled actuators with a QP-based whole-body controller
- Using RL to learn a walking policy that works with position actuators
- Hybrid approaches that combine model-based planning with learned residuals

Whatever approach Lab 8 takes, Lab 7 provides the foundation: validated kinematics, proven IK, reliable standing balance, and — perhaps most importantly — a deep understanding of exactly why classical ZMP walking fails with position actuators. Understanding the failure is the first step toward the fix.

---

*Lab 7 code and documentation: [mujoco-robotics-lab/lab-7-locomotion](../)*

*This is part of a robotics portfolio series progressing from simple planar arms to VLA-controlled humanoid manipulation. Previous labs cover FK, Jacobians, IK, dynamics, force control, motion planning, grasping, and dual-arm coordination.*
