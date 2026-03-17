# Building a Pick-and-Place Pipeline from Scratch

*Lab 5 of the MuJoCo Robotics Lab series*

---

When people talk about robot manipulation, they usually jump straight to the exciting part: the robot picking up an object. But before that first successful grasp, there are a dozen things that silently fail. Contact geometry doesn't match, the IK solver diverges, the planner times out, or the box slips through fingers that were never close enough to touch it.

This post is about building a pick-and-place pipeline bottom-up — from a blank MJCF scene to a UR5e that reliably moves a 40mm cube from one table position to another. No pre-built grasping libraries, no learned policies. Just physics, math, and a lot of debugging.

---

## Why Grasping Is Hard (Even in Simulation)

In pure motion planning, success is binary: did the arm reach the goal configuration without collision? In grasping, you add contact mechanics into the mix. The same geometric constraints that make a trajectory collision-free can also make a grasp impossible — an arm that approaches without hitting the table might not have the right approach angle to close on the box.

You also introduce a new failure mode that didn't exist in Labs 1–4: **the object can move**. A free joint body responds to contact forces, gravity, and friction. If your gripper geometry is off by 4mm, the box just drops to the floor, and your controller never knows why.

The central design challenge is: **how do you decompose a grasping task into sub-problems that are individually testable?**

---

## The Architecture

I split the pipeline into four layers, each testable in isolation:

**Layer 1: Contact model** — gripper MJCF and contact parameters. Testable with a static scene: close the gripper on a box, verify contact is detected.

**Layer 2: IK configurations** — Pinocchio DLS IK computes arm joint configs for each phase (pregrasp, grasp, preplace, place). Testable by checking FK accuracy vs. target.

**Layer 3: Motion planning** — Lab 4's RRT* + TOPP-RA plans collision-free trajectories between configs. Testable by checking path validity and velocity/acceleration bounds.

**Layer 4: State machine** — ties everything together. Testable by checking state transitions in isolation with mocked sub-steps.

The architecture principle from CLAUDE.md applies perfectly here: Pinocchio computes, MuJoCo simulates. Pinocchio's FK/IK never touches the simulator; MuJoCo's contact solver never touches Pinocchio's math.

---

## Building the Gripper

The hardest part of a custom parallel-jaw gripper is getting the geometry right before writing a single line of control code.

My first gripper design had the finger bodies at ±0.020 m from the gripper center. The pads added +0.009 m of offset, putting the inner pad face at 0.024 m. The box half-width was 0.020 m. Gap: 4mm. **The gripper could never physically touch the box even when fully closed.**

This is easy to miss because nothing crashes. The simulation runs happily, the fingers close to their minimum position, and the box just sits there untouched. The only indication is `data.ncon == 0` after closing — silence from the contact solver.

The fix: move the finger bodies inward to ±0.015 m. Now the inner face at `joint=0` is at 0.019 m — 1 mm inside the box edge. One millimeter is enough for MuJoCo's stiff contact (`solimp="0.99 0.99 0.001"`) to generate holding force.

**Lesson:** Prototype gripper geometry statically. Compute `pad_inner_face` explicitly and compare to `object_half_width` before running any control code.

---

## Contact Parameters

MuJoCo gives you five levers for contact behavior. For grasping, three matter most:

**`condim=4`** — the fourth dimension is torsional friction. Without it, a gripper that contacts a box symmetrically can still rotate the box during transport. With it, the box stays aligned.

**`solimp="0.99 0.99 0.001"`** — near-rigid contacts. This keeps the box from visually compressing under the grip and from bouncing when the gripper first contacts it.

**`friction="1.5 0.005 0.0001"`** — μ_slide of 1.5 is enough to hold a 150g box against gravity with some margin. I tested by slowly tilting the arm to 30° and watching for slip.

The interesting interaction: `solref` (contact rise time) and `solimp` interact with `mass`. A very heavy box with soft contacts can make the gripper "bounce" when it first touches. Keep contacts stiff and box mass realistic.

---

## DLS IK and the Tip Offset

The IK solver computes arm configurations from Cartesian targets. For grasping, the target is NOT the box position — it's the `tool0` origin position that puts the fingertip center at the box.

```
tool0_target = box_pos + [0, 0, GRIPPER_TIP_OFFSET]
```

With `GRIPPER_TIP_OFFSET = 0.090 m` (the distance from tool0 to the gripper_site). Get this offset wrong and the arm reaches the right configuration but the fingers are 9 cm above or below the box.

The DLS method (`Δq = α Jᵀ (J Jᵀ + λ²I)⁻¹ e`) handles the 6-DOF arm cleanly. With λ²=1e-4, it avoids singularity blow-up near workspace boundaries. Convergence to < 0.1mm in 300 iterations for all four grasp configurations.

I compute all configurations **offline** before the pick-and-place cycle starts. There's no reason to solve IK inside the control loop when the scene geometry is known in advance. This also makes it easy to verify each config before running the full pipeline.

---

## Integrating Labs 3 and 4

Lab 5 is where the previous labs start paying off. The state machine imports:

- **Lab 4 `RRTStarPlanner`**: finds collision-free paths in joint space with the table obstacle
- **Lab 4 `parameterize_topp_ra`**: converts waypoints to a time-optimal trajectory with velocity and acceleration bounds
- **Lab 3 `compute_impedance_torque`**: executes the trajectory with gravity compensation

The one integration bug I hit: `parameterize_topp_ra` returns **4 values** `(times, q, qd, qdd)`, but the state machine was unpacking 3. Python's tuple unpacking error (`ValueError: too many values to unpack`) is cryptic until you check the actual function signature.

Cross-lab imports require manually adding the foreign `src/` to `sys.path`:
```python
add_lab_src_to_path("lab-3-dynamics-force-control")
add_lab_src_to_path("lab-4-motion-planning")
```

This is the right pattern for a monorepo without a proper package manager. Each lab is self-contained but can pull from earlier labs via explicit path manipulation.

---

## The State Machine

Pick-and-place decomposes naturally into a linear state machine:

```
IDLE → PLAN_APPROACH → EXEC_APPROACH → DESCEND → CLOSE → LIFT
     → PLAN_TRANSPORT → EXEC_TRANSPORT → DESCEND_PLACE → RELEASE → RETRACT → DONE
```

Each `PLAN_*` state runs RRT* + TOPP-RA offline, then hands a pre-computed trajectory to the `EXEC_*` state. Each `EXEC_*` state just reads the next timestep from the trajectory and computes joint torques. Decoupling planning from execution is critical: it means you can test each layer independently.

The descent/lift states use Cartesian impedance (Lab 3) rather than another planning call. Moving 15cm along the Z axis doesn't need RRT* — a simple Cartesian Z-direction command is more efficient and more reliable for a straight vertical motion.

---

## What I Learned

**1. Contact geometry must be verified before control code.** A 4mm gap killed grasping. Static debugging (checking `data.ncon` manually) reveals this in seconds.

**2. Check contact geoms, not just pad geoms.** The finger body geoms (`left_finger_geom`) contact the box before the smaller friction pads. `is_gripper_in_contact` initially checked only `left_pad`/`right_pad` and missed these contacts entirely.

**3. Test contact during closing, not after settling.** A box placed between the fingers with no arm gravity compensation will fall to the floor in ~1 second. Contact detection tests need to check during the event, not 1000 steps later.

**4. Cross-lab integration surfaces API mismatches.** Return signatures of functions written in earlier labs may not match what the current lab expects. Always verify actual return types, not assumed ones.

**5. Lab 3+4 investment pays off immediately.** The state machine's planning and execution layers were written in <200 lines total by composing Lab 3 and Lab 4 components. The hardest part of Lab 5 was contact modeling — not the control architecture.

---

## What's Next

Lab 6 moves to dual-arm coordination: two UR5e arms coordinating on a shared task. The challenge shifts from single-arm grasping geometry to relative motion synchronization and avoiding inter-arm collisions. The state machine pattern from Lab 5 will extend naturally to a dual-arm coordinator.

Lab 9 is the end goal: a VLA-controlled humanoid. Every lab in this series is building toward that. Lab 5's contribution is understanding that grasping is a geometry and contact problem first, and a planning/control problem second.

---

*Code: [github.com/ozkannceylan/mujoco-robotics-lab](https://github.com/ozkannceylan/mujoco-robotics-lab)*
