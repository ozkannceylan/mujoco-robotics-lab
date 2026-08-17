# The Humanoid Walked Once I Stopped Telling It Where to Stand

*Lab 8 of a robotics portfolio series using MuJoCo and Pinocchio*

---

Lab 7 ended in a failure I was fairly proud of. I had a Unitree G1 in MuJoCo that
could stand, shift its weight, and hold a pose to within a millimetre — and could
not take a single walking step. Six attempts. The inverse kinematics converged
every time; the robot fell over every time. I wrote down a diagnosis (*the
actuators are the problem*), closed the lab, and moved on.

Lab 8 is the test of that diagnosis. It replaces the G1's position servos with
torque motors, rebuilds the control stack around a whole-body optimisation, and
asks the robot to walk and use its hands at the same time. It walks 1.18 m in
twelve steps, walks while carrying, and finishes by walking to a pedestal,
picking up a box, carrying it, and putting it down 11.8 mm from where it was
asked to.

The diagnosis was right. It was also nowhere near the whole story.

## Why position servos can't walk

MuJoCo's Menagerie G1 ships with 29 `<position kp="500">` actuators. Each one
runs `τ = Kp(ctrl − qpos) − Kd·qvel` *inside* the simulator. The only thing you
can send is a joint angle.

That's fine for standing — a stiff spring at the right angle is a good standing
controller, which is exactly why Lab 7's standing worked so easily and why I
should have been more suspicious of it. It is fatal for walking, because walking
is a statement about *forces*: the ground pushes the robot, and the whole art is
deciding where under the foot that push should act. A joint-angle interface has
nowhere to say that.

So M0 rebuilt the model with `<motor>` actuators, taking each one's limit from
the joint's own `actuatorfrcrange` rather than inventing numbers. Then it
re-established standing from outside the simulator — and immediately produced my
first useful finding, which is that **gravity compensation alone cannot stand**.

`τ = g(q)` cancels the robot's weight exactly. It also adds no posture stiffness
whatsoever, and a standing humanoid is an inverted pendulum: mine collapsed to a
0.097 m squat in about two seconds. The position servos had been quietly
supplying the stabilising term all along. When you port a robot from position
control to torque control, you have to enumerate everything the servo was doing
for you and re-supply it explicitly.

The second M0 finding was more uncomfortable, because the bug was a fix I had
written myself. In Lab 5 I had learned to shape PD gains with the mass matrix —
`τ = M(q)(Kp·e + Kd·ė) + g` — and it genuinely improved a UR5e. Applied to the
G1, it made the robot fall at every gain setting. `M(q)[6:,6:]` on a *floating*
base isn't the inertia a standing robot feels through its closed leg chains;
multiplying gains by it just saturates the actuators. Raw joint gains stand with
0.18 mm of drift. **An inherited fix is only valid inside the assumptions that
produced it**, and I now write those assumptions down next to the fix.

## The QP that couldn't balance

With torque control available, the plan was a whole-body quadratic program: one
optimisation per control tick that resolves balance, foot contacts, hand targets
and posture together. My lab brief specified it at the velocity level —
`min ‖J q̇ − ẋ_des‖²` — which is the standard formulation for redundant arms and
which I built first.

It fell over during every reach. And it failed with a symptom that turned out to
be the most useful diagnostic in the whole lab: **making the hand task stronger
made the robot fall sooner**. Weights from 1e2 to 1e4 all fell. The only version
that stayed up was one where the hand tracked so badly it barely moved.

A controller that degrades as you ask it to do its job is optimising the wrong
variable. The reason here is physical rather than numerical: a velocity-level QP
can satisfy `J_com q̇ = 0` *exactly* — the centre of mass has zero commanded
velocity — while the robot rotates about its ankles and topples. Centre-of-mass
motion is produced by contact forces, and contact forces are not in that
formulation at all.

So the QP moved to the acceleration level, with the contact wrenches as decision
variables alongside the joint accelerations:

```
min_{q̈,f}  Σ w‖J q̈ + J̇q̇ − ẍ_des‖² + λ_a‖q̈‖² + λ_f‖f‖²
s.t.  M[:6] q̈ + h[:6] = J_cᵀ[:6] f      (unactuated floating base)
      J_c q̈ + J̇_c q̇ = 0                (stance feet don't accelerate)
      friction pyramid · CoP inside foot · f_z ≥ f_min · |τ| ≤ τ_max
τ  =  M[6:] q̈ + h[6:] − J_cᵀ[6:] f
```

47 variables, 0.11 ms per solve. Now strengthening the hand task *improves*
tracking, which is the sanity check I should have insisted on before tuning
anything. The right hand traced a 10 cm circle at 7.08 mm RMS with both feet
planted.

## "Where the CoM is" is the wrong thing to command

Stepping in place worked next, with a strategy anyone would write first: move the
CoM over whichever foot is about to take the load, then swing the other one.
Four steps, ZMP inside the support polygon 98.7% of the time.

Then I asked for forward walking and it collapsed to three steps out of ten and
22 cm — at *every* stride length, every double-support duration, every CoM bias I
tried. That flatness is the tell. When a whole family of parameters gives you the
same failure, you are not badly tuned; you are solving the wrong problem.

Quasi-static balance needs a moment of rest over each foot. Forward walking never
grants one. Under the linear inverted pendulum the CoM obeys `c̈ = ω²(c − p)`
with `ω = √(g/z_c)`, and that splits cleanly into two parts:

```
ξ = c + ċ/ω     divergent — this is the one that can run away
η = c − ċ/ω     convergent — needs no control at all
```

Only `ξ` is unstable, and `ξ̇ = ω(ξ − p)` says it is steered entirely by the ZMP —
which the whole-body QP can already place anywhere inside the feet. So instead of
commanding where the body should *be*, I plan a piecewise-linear ZMP through the
footsteps, back-integrate the DCM from a terminal rest condition (backwards is
the only stable direction — forward integration of an unstable system amplifies
the boundary error by `e^{ωT}`), and command

```
p_cmd = ξ − ξ̇_ref/ω + (k/ω)(ξ − ξ_ref)      →      c̈_des = ω²(c − p_cmd)
```

There is no CoM position task on the control path at all. The body travels
freely; only its divergent component is regulated.

The first run of this was **worse** than what it replaced. Two steps out of
twelve, 0.32 m *backwards*, with the commanded ZMP pinned against a foot edge on
53% of ticks. What saved the day was reading the saturation instead of the
tracking error — the controller was not wrong, it was being refused.

Three things were wrong, and none of them was the control law:

**The foot was a guess.** My contact model described the sole as a symmetric
±0.08 m box centred on the ankle frame. The real Menagerie foot spans
x ∈ [−0.05, 0.12], sits 35 mm *below* that frame, and is 25 mm wide. The guess
simultaneously claimed 30 mm of rearward CoP that doesn't exist — so the QP wrote
wrenches MuJoCo simply refused to produce — and threw away 40 mm of forward CoP,
which is precisely the authority that decelerates the body before touchdown. It
also ignored the shear term: because the wrench is expressed about a frame above
the ground, `CoP_x = (−m_y − h·f_x)/f_z`, not `−m_y/f_z`. Standing cannot tell
these models apart. Walking uses both ends of the foot every single step.

**A tighter solver tolerance was buying inaccuracy.** 38% of control ticks were
returning OSQP's `maximum iterations reached` at 12.6 ms per solve. `eps = 1e-6`
is far below what a cost spanning task weights 1e4 to 1e1 against a 1e-4
regularisation can actually deliver, so the solver was spending its entire budget
not converging. At `1e-4`, every tick converges in ~25 iterations and 0.073 ms —
and the constraint residual *fell*, from 0.021 to 8.5e-5 N·m. Asking for less
accuracy produced a more accurate answer. A hit iteration cap is a correctness
warning, not a performance note: what comes back is wherever the solver happened
to be when the clock ran out.

**Stance width, not stride length, sets the difficulty.** The ZMP has to cross
from one foot to the other every step, and the lateral DCM swings with the same
amplitude, so the cost of lateral balance is set by how far apart the feet are.
The G1's natural 0.237 m rest stance gave 7 steps and a fall. 0.18 m gave 12 out
of 12, DCM RMS 6.2 mm, peak torque down from a saturated 139 N·m to 56.

Together those took commanded-versus-realised CoM acceleration from slope 0.78
with a −0.09 m/s² bias to slope 0.95 with correlation 0.995. Then it walked.

Two more things I'd have got wrong by pure reasoning. I added a leaky integrator
to cancel that acceleration bias — textbook, and it worked. After the contact and
solver fixes removed the bias at its source, the same integrator turned a passing
gait into a falling one (12/12 → 8/12). **Re-measure a compensator after fixing
what it was compensating for.** And I "fixed" the plan's initial condition, where
the DCM leads a ramping ZMP by `k/ω` and so starts 30 mm off-centre while the
robot stands still. That is an initial-condition mismatch by every textbook
argument, removing it is clean, and it halved the gait (12/12 → 6/12). The lead
isn't an error. It's the lateral momentum the first step needs.

## Arms and legs are the same problem

Walking while using the hands is where the lab's two halves collide, and my first
instinct — find a hand-task weight that both walks and tracks — burned a lot of
compute for nothing. What eventually told me it was hopeless wasn't the failures
but their *ordering*: weight 1e1 walked with a 46 mm droop, 2e1 fell at step 5,
1e2 fell at step 7, 3e2 fell at step 5. When the failures are non-monotonic, you
are not mis-tuned; a term is missing.

The term is centroidal angular momentum. The CoM Jacobian includes the arms, so a
hand task is not an independent addition to a walking controller — every kilogram
the QP accelerates to satisfy a hand target lands in the quantity keeping the
robot upright. Regulating `L = A_g(q) q̇` lets the QP say "the arms may move, but
they may not spin the body", instead of restraining them through a joint-space
posture pull that was doubling, badly, as a momentum damper.

The hand task that fell on step 7 then walked the full distance with three times
better tracking: 12/12 steps, 14.5 mm RMS. My lab brief had listed "regulate
centroidal momentum while performing arm tasks" from the very beginning. It took
a wall of failed tuning before I believed it.

Two scoping rules came out of that, both learned by breaking something. The
momentum task is an arm-task *companion*, not a global stabiliser — left enabled
across a bare walk it cancels the angular momentum walking itself generates (the
gait runs ±2 kg·m²/s of roll) and put the robot down on step 2. And its reference
is zero only for a *held* pose; a task that deliberately moves mass needs an
`L_ref`, or the term fights the very trajectory it was added to enable.

## The result I'm reporting as a failure

One sub-task didn't make it: walking while the right hand traces a circle. It
walks the distance and tracks to 37.6 mm RMS, which is inside my 50 mm gate. I
am still reporting it as exploratory, because of one check.

I shifted the circle's *starting phase*. That changes nothing about the task's
difficulty — same radius, same speed, same everything. 12/12 steps became 9/12 at
0.3 rad and 3/12 at 1.0 rad.

A result that a no-op perturbation can destroy is a draw from a distribution, not
a property of the controller. The same test on the two-handed carry came back
flat — 12/12 at nominal, at a shorter stride, at a longer double support,
15.2–15.4 mm every time — and that's what made the carry gate trustworthy.

The control that settled the diagnosis was running the reach with the circle
radius set to **zero**. It still fell, with the same lateral signature. The
circle was never the cause; an asymmetric upper body on a walking robot is, and
lateral balance is the axis with no margin — which is the same finding that had
already decided the walking gait. Sixteen parameter families ruled out, written
up, deferred.

## The capstone: everything, in one take

The final demo is 25 seconds with no cuts: walk to a pedestal, stop, reach,
grasp, lift, bring the load to the chest and secure it with the second hand, walk
carrying it, stop, place it, let go. The payload ends 11.8 mm from its target
after being transported 0.384 m.

The interesting part is that the capstone invents no new control at all. Every
phase is a regime an earlier milestone validated — the approach walk is the DCM
gait, the standing reach is the M1 controller, the carry walk is M4's. All ten of
its defects were in the *transitions*, and three were in code that four
milestones had already exercised without revealing them: the momentum task
scoping above; a gait that always swung the left foot first, so a walk resumed
after an odd number of steps re-stepped its own leading foot into the place it
already stood; and a walk that ended mid-stride, handing the next one a stance no
milestone had seen. All latent in M3 and M4, because each of those walks exactly
once.

Two more were geometry the controller cannot perceive. The identical M3
controller walks 12 steps on the bare model and fell on step 4 in the capstone
scene — not a balance problem, and two hours of balance hypotheses said so.
Logging every contact involving a scene prop named it in one line:
`pick_pedestal ↔ right_hip_roll_link`. Scene furniture at limb height is a
collision the balance controller cannot anticipate. And at the very end, an
accurate release still put the box on the floor, because the target sat 0.09 m in
from a pedestal with a 0.10 m half-extent: the box overhung its own edge and
tipped off.

Three findings from the manipulation side worth keeping:

**A MuJoCo weld holds its compile-time relative pose.** `eq_active` is a switch,
not a "grasp here" instruction. Activating it where the hand happens to be
commands a snap back to the rest configuration — I measured a 0.42 m lurch that
threw the robot down. Write the live relative pose into `eq_data` first, and
refuse to close a weld across a gap the hand hasn't actually crossed.

**Symmetric arms are not a symmetric load.** Carrying on one arm walks 0.64 m and
falls, even with both arms held in mirror-image poses, because what the balance
controller answers to is where the *mass* is, not where the limbs are. The
sequence picks one-handed (only the right hand reaches the pedestal), brings the
load to the chest mid-line, and lets the left hand join it there. Carry distance
went to 0.95 m. Placing then has to *release* the left weld again, because two
welds form a closed kinematic chain the QP doesn't model and the left arm drags
against the placing motion.

**Servo the object, not the gripper.** The last 55 mm of error was a stale
transform: the hand target came from a hand→payload offset measured once before
the motion, and a compliant grip lets the load settle over 25 seconds.
Recomputing the target every tick from the *live* offset took the release from
65 mm to 18.9 mm.

## What I'd tell myself at the start

Three habits earned their keep, and none of them is about control theory.

**Read the saturation, not the error.** The DCM controller was right the first
time; it was being refused by a foot model I'd guessed at. The tracking error
told me nothing useful. The 53% ZMP clamp rate told me everything.

**Perturb what should not matter before believing a pass.** A gate that survives
a no-op perturbation is a controller property. One that doesn't is a lucky seed,
and the difference between those two is the difference between a result and a
screenshot.

**A model that a passing test cannot falsify is still worth checking.** The wrong
foot geometry was invisible to every standing test I had, and it was the single
largest error in the lab. "All tests pass" is a statement about the tests.

Lab 9 is the VLA integration, and these controllers are what will generate its
demonstrations. Which means the honest thing to document isn't what I intended
them to do — it's what I measured them doing, including the circle they can't
trace.

---

*Code, gates and full measurement tables:
[`lab-8-loco-manipulation/`](../README.md). Every failure summarised here has a
long-form entry in [`tasks/LESSONS.md`](../tasks/LESSONS.md).*
