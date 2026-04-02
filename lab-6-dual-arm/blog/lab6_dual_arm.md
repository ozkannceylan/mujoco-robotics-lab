# From One Arm to Two: Why Dual-Arm Coordination is Harder Than You Think

Two arms should be twice as easy as one arm, right? You already have the IK solver, the PD controller, the motion planner. Just instantiate everything twice, mirror the targets, and you are done.

I believed this for about fifteen minutes. Then I spent the next several days debugging problems that do not exist in single-arm manipulation: arms colliding with each other during large reconfigurations, weld constraints launching a box across the room, PD gains that worked perfectly for one arm failing catastrophically when a rigid object couples two kinematic chains together. This is the story of Lab 6 in my MuJoCo robotics series, where I built a dual-arm system with two UR5e robots that cooperatively grasp, lift, carry, and place a box -- and where almost every assumption I carried over from five labs of single-arm work turned out to be wrong.

## The three tangled problems

At a high level, dual-arm cooperative manipulation tangles three problems that are mostly independent in single-arm work.

### 1. Large motion planning in a cluttered shared workspace

In single-arm grasping (Lab 5), the arm starts at a home configuration and moves to a pre-grasp pose. The reconfiguration is modest -- maybe 30-40 degrees of joint motion. In the dual-arm setup, both arms start in a home configuration with end-effectors pointing straight down, and they need to rotate to face each other horizontally across a 30cm box sitting on a table between them. That is a 150-degree reconfiguration of the shoulder and elbow joints, and the swept volume of each arm passes through the space occupied by the table, the box, and the other arm.

This is not a theoretical concern. In my first attempt, the left arm's upper-arm link swept directly through the box on its way to the approach pose. The right arm's shoulder capsule (radius 6cm) physically rested on the table because I had sized the table too wide. IK gives you a kinematically valid endpoint, but it says nothing about whether the path to get there is collision-free.

### 2. Contact geometry for bimanual grasping

Two end-effectors must approach from opposite sides of a box (30cm x 15cm x 15cm), push 2cm into the surface to establish firm contact, then lock onto the box with weld constraints. The approach direction, standoff distance, and penetration depth must be computed from the box's current pose in the world frame -- nothing can be hardcoded, because the box is a freejoint body that moves.

The contact geometry cascades into IK: each EE needs a specific orientation (z-axis pointing toward the box center) at a specific position (box surface minus penetration offset). The left arm approaches from -x, the right from +x. These are fully constrained 6-DOF IK targets with orientations that are roughly perpendicular to the home configuration. The solver must find solutions that (a) converge, (b) are reachable without joint wrapping artifacts, and (c) do not collide with anything.

### 3. Force coordination through a rigid coupling

Once both arms grasp the box, they are no longer independent. The box acts as a rigid mechanical coupling between two 6-DOF kinematic chains, creating a closed-loop mechanism. If the PD controller on the left arm wants to move the box 1mm to the left while the right arm wants to hold position, something has to give. Internal forces build up. Weld constraints fight the controller. The box vibrates.

In single-arm work, you tune Kp and Kd once and forget about it. In dual-arm work, the gains must be balanced between arms and appropriate for the phase of manipulation. Kp=500 works for the approach phase where you need to drive through a 150-degree reconfiguration in under a second. But that same gain, applied during the carry phase when both arms are rigidly coupled through the box, produces oscillations and internal stress that can destabilize the whole system. The capstone demo uses Kp=300, Kd=40 with smooth-step interpolation to produce controlled, jerk-free motion.

## Why sequential milestones

I structured Lab 6 as five sequential milestones (M0 through M5), each with explicit gate criteria that must pass before proceeding. This was not my first approach -- more on that shortly.

**M0: Scene validation.** Load the MuJoCo scene, confirm 12 hinge joints and 12 actuators (6 per arm), verify both EE sites point downward at home configuration (dot product with -z > 0.9), check table clearance from arm bases. This caught the table collision immediately -- the original x half-extent of 0.40m put the table edge at x=0.9m, only 10cm from the right arm base at x=1.0m. The UR5e shoulder capsule (radius 6cm) was physically inside the table. Fix: narrowed to 0.20m, giving 30cm clearance.

**M1: Independent PD control.** Prove that each arm can track joint targets independently with a PD controller plus MuJoCo gravity compensation (`qfrc_bias`). Gate: steady-state error below 0.001 rad for both arms simultaneously. This validated the basic control loop before adding any coordination.

**M2: FK/IK cross-validation.** Load the same UR5e URDF into Pinocchio and compare FK outputs against MuJoCo site positions across 20 joint configurations (5 hand-picked + 15 random). This is where things got interesting. The initial URDF -- the standard one from `universal_robots_description` that I had used in Lab 4 -- gave a 99mm FK error. Not 0.99mm. Ninety-nine millimeters. The error was constant in the arm-local frame and rotated with the shoulder pan joint, which is the signature of a kinematic chain mismatch. The standard DH-convention URDF parameterizes the shoulder lift joint as `rpy="pi/2 0 0" xyz="0 0 0"` with elbow at `xyz="-0.425 0 0"`. MuJoCo Menagerie uses `pos="0 0.138 0" quat="1 0 1 0"` -- a different (but equivalent) parameterization. Lab 3 had already hand-tuned a URDF to match Menagerie exactly. Swapping it in dropped the error from 99mm to 0.000mm. Exact match across all 20 configurations.

**M3: Coordinated approach.** Both arms simultaneously move from home to grasp standoff poses (5cm from box surface). This required the collision-free IK search: 300 random starts per arm, each converged solution checked for MuJoCo contacts, the closest collision-free configuration to home selected. Joint wrapping was essential -- without it, the IK solver would find solutions requiring 200+ degrees of joint motion when 160 degrees in the opposite direction would have worked and avoided collisions entirely. Gate: Cartesian error < 5mm, arrival time difference < 50ms, EE z-axis dot with toward-box direction > 0.9.

**M4: Full pick-and-place state machine.** Six states: APPROACH, CLOSE, GRASP, LIFT, CARRY, PLACE. Each transition computes new IK targets from the current box pose, solves for both arms, runs the PD controller with smooth-step interpolation until convergence. The GRASP state activates weld constraints after writing the current relative transform into `eq_data`. The PLACE state computes absolute z from table surface height, deactivates welds, and immediately retracts to avoid pushing the box with contact forces.

**M5: Capstone demo.** Run the full pipeline with video recording and trajectory plotting. State overlay burned into each frame. Box trajectory plotted as x/y/z vs time with state boundaries marked.

## How the milestone-gated approach prevented chaos

I can say with confidence that this approach works because I tried the alternative first. The original attempt lived in a directory called `lab-6-dual-arm-coordination/` (since deleted). In that version, I tried to build everything at once: scene, controller, IK, state machine, all in parallel. The result was three iterations of joint sign-flipping trying to make the arms face each other (I had added a 180-degree yaw mount to the right arm base, which made joint-space mirroring intractable), FK mismatches masked by other bugs (the 99mm URDF error was invisible because the state machine was also broken), and a completely non-functional state machine where I could not tell if failures were from IK, control, collision, or weld constraint issues because all four were wrong simultaneously.

The gated approach changed everything. The URDF mismatch was caught at M2 -- before it could corrupt M3's IK search or M4's state machine. The table collision was caught at M0 -- before controllers were involved, so I knew it was a geometry problem, not a gain tuning problem. The weld constraint launch (box flying 90cm across the room when welds were activated) was caught at M4 -- cleanly isolated from IK issues because M2 and M3 had already proven that FK and IK were correct.

Each milestone peeled away one layer of complexity. By the time I reached M4, I knew with certainty that the scene geometry was correct (M0), the PD controller converged (M1), FK/IK matched between Pinocchio and MuJoCo (M2), and both arms could reach grasp poses without collision (M3). When the weld constraint launched the box, I knew the problem was specifically in how I was activating the constraint, not in any of the upstream systems.

## Twelve lessons, twelve scars

Lab 6 produced twelve entries in the lessons-learned log. Here are the ones that surprised me most:

**The weld constraint relative pose (L8).** MuJoCo weld constraints enforce the relative pose stored in `eq_data`, which is computed at model compilation time from the initial body positions. At grasp time, the wrists are in completely different positions relative to the box. Activating the weld without updating `eq_data` first makes the solver try to teleport the box to the initial relative pose, launching it 90cm away. The fix: before activating, compute the current relative transform (`R_body1^T * (pos_body2 - pos_body1)` for position, `R_body1^T * R_body2` for rotation, converted to quaternion via `mju_mat2Quat`) and write it into the 11-float `eq_data` layout: `[anchor(3), rel_pos(3), rel_quat(4), torquescale(1)]`.

**Carry direction is constrained by workspace geometry (L9).** Both arm bases sit on the x-axis (x=0.0 and x=1.0), separated by 1m. My first attempt carried the box 20cm in +x. The right arm could not reach -- the EE position at `box_x + half_x + standoff` approached x=0.85, which is too close to the right arm base at x=1.0 for the required sideways-pointing orientation. The fix: carry in +y instead, which is symmetric for both arms.

**Step commands produce jerky motion (L12).** Commanding the final joint target instantly at Kp=500 produces maximum torque on the first timestep. The arm accelerates at maximum rate, overshoots, oscillates, and settles. The fix was a smooth-step interpolation (`3a^2 - 2a^3`) ramped over 2 seconds, combined with lower gains (Kp=300, Kd=40). Visually, the difference is dramatic.

## Key results

- **FK cross-validation:** 0.000mm error across 20 configurations (exact match between Pinocchio and MuJoCo after switching to the correct URDF)
- **IK convergence:** 20/20 targets solved, collision-free search with 300 random starts per arm
- **Coordinated approach:** 2ms synchronization between arm arrivals, 0.1mm Cartesian error at grasp standoff
- **Full pick-and-place:** lift 15cm, carry 22cm in +y, place with 0.0cm z-error and 3-degree rotation error
- **State machine:** 6 states executed in sequence, weld constraints activated/deactivated cleanly, arms retracted after release

The demo videos (`m3_approach.mp4`, `m4_carry.mp4`, `m5_capstone.mp4`) and trajectory plot (`m5_trajectory.png`) are in the `media/` directory. The capstone trajectory plot shows box x/y/z over time with state boundaries marked -- the z-channel clearly shows the lift plateau, and the y-channel shows the carry displacement.

## What I would do differently

**Start with the correct URDF from day one.** The 99mm FK mismatch cost hours of debugging. In hindsight, FK cross-validation should be step zero of any new lab that uses Pinocchio alongside MuJoCo. I got lucky that this was "only" a 99mm offset -- a smaller error might have been masked by other tolerances and surfaced much later as mysterious IK failures.

**Design arm spacing from workspace analysis.** I picked 1m separation because it "looked right" in the viewer. This worked, but the +x carry failure showed that the usable workspace for coordinated tasks is much smaller than the individual arm workspaces. A proper reachability analysis -- computing the intersection of both arms' workspaces at the required orientations -- would have revealed the carry direction constraint before I spent time debugging IK failures.

**Use impedance control for the contact phase.** The current approach jams the end-effectors 2cm into the box surface using position control, then locks with weld constraints. This works but is not how you would do it on real hardware. A Cartesian impedance controller (which I built in Lab 3) would allow compliant approach, force-limited contact establishment, and graceful handling of pose uncertainty. The weld constraints are a simulation convenience that would not exist on a real robot.

**Add trajectory interpolation from the start.** The smooth-step interpolation was a late addition after seeing jerky motion in the capstone demo. Every phase transition should have been interpolated from the beginning. On reflection, this is obvious -- step commands to a PD controller are always jerky. I just never noticed in single-arm work because the motions were smaller.

**Consider bilateral teleoperation for natural motion.** The current system plans all targets offline (IK for each state), then executes them open-loop. A more natural approach would be to teleoperate one arm and have the other mirror its motion through the box's rigid coupling. This would produce more organic-looking coordinated motion and is a stepping stone toward learning-based bimanual policies.

## The real lesson

The fundamental insight from Lab 6 is that dual-arm coordination is not twice as hard as single-arm manipulation. It is a qualitatively different problem. The shared workspace introduces collision constraints that do not exist with a single arm. The rigid coupling through the grasped object creates a closed-loop mechanism that changes the dynamics. The symmetry between arms means that every bug is either doubled (both arms wrong in the same way) or masked (one arm compensating for the other's error in a way that looks correct until it doesn't).

The milestone-gated approach was the single most important decision I made. Not because the milestones themselves were clever -- M0 through M4 are straightforward engineering steps -- but because they forced me to prove each layer was correct before building on top of it. In the previous failed attempt, every bug was entangled with every other bug. In the gated approach, each bug was isolated to exactly one milestone and could be diagnosed in minutes instead of hours.

If you are planning to move from single-arm to dual-arm manipulation, my advice is: do not trust anything you built for one arm to work unchanged for two. Validate everything from the ground up. And above all, do not try to build the whole pipeline at once.

---

*This is Lab 6 of a 9-lab MuJoCo robotics series. Previous labs covered 2-link planar arms (Lab 1), UR5e 6-DOF kinematics (Lab 2), dynamics and force control (Lab 3), motion planning with RRT* and TOPP-RA (Lab 4), and grasping with a custom parallel-jaw gripper (Lab 5). Code and demos are available in the [mujoco-robotics-lab](https://github.com/ozkannceylan/mujoco-robotics-lab) repository.*
