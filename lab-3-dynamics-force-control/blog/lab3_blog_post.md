# Position Control Pushes Through Walls

*Lab 3 of the MuJoCo Robotics Lab series*

---

Labs 1 and 2 were about getting to a pose. Forward kinematics, Jacobians, inverse kinematics, a PD loop tight enough to drag a UR5e end-effector around the edges of a cube. By the end of Lab 2 I could put the tool tip within 0.088 mm RMS of any commanded point in the workspace.

Then I put a table in the scene.

A position controller does not know what a table is. It knows a setpoint and an error, and its entire theory of the world is that the error should be zero. Command a point 5 mm below the tabletop and it will spend torque — all of it, up to whatever the actuator can deliver — trying to get there. In simulation you get contact forces in the hundreds of newtons and a solver that eventually throws the arm somewhere unphysical. On real hardware you get a bent tool or a tripped safety stop.

This is not a tuning problem. A stiffer controller pushes harder; a softer one droops under gravity and misses the table entirely. Position is simply the wrong thing to command when the robot is touching something.

Lab 3 is the leap from commanding positions to commanding forces. It is, by a wide margin, the most important conceptual step in this series — everything after it (grasping in Lab 5, cooperative carry in Lab 6, a humanoid holding its own weight in Lab 7) is downstream of getting this right.

---

## The Equation You Have to Believe

Everything in this lab hangs off one equation:

```
M(q)q̈ + C(q,q̇)q̇ + g(q) = τ
```

The inertia matrix `M(q)`, the Coriolis/centrifugal term `C(q,q̇)q̇`, the gravity vector `g(q)`. Once you have those, you can stop asking "what position do I want" and start asking "what torque produces the interaction I want."

The problem is that you have to *believe* them. Every controller in this lab is a linear combination of `g(q)`, `J(q)`, and `M(q)`, computed by Pinocchio and applied to a robot simulated by MuJoCo. If Pinocchio's model of the robot differs from MuJoCo's model of the robot, every controller downstream is quietly compensating for the wrong robot. The failure mode is not a crash — it is a slow, plausible-looking droop that you spend two days blaming on your gains.

So the first thing I built was not a controller. It was a parity check.

Pinocchio computes `g(q)` with `computeGeneralizedGravity()` (an RNEA call with velocity and acceleration zeroed) and `M(q)` with `crba()` — the Composite Rigid Body Algorithm, which fills only the upper triangle, so you symmetrize it yourself. MuJoCo exposes the same quantities: `qfrc_bias` at zero velocity is the gravity vector, and `mj_fullM` gives you the dense inertia matrix. Same robot, two independent implementations, no shared code path. Any disagreement is a modeling bug.

Across `Q_HOME`, `Q_ZEROS`, and random configurations:

| Quantity | Method | Max error |
|---|---|---|
| `g(q)` | Pinocchio RNEA vs MuJoCo `qfrc_bias` | 8.01e-06 |
| `M(q)` | Pinocchio CRBA vs MuJoCo `mj_fullM` | 3.34e-05 |

Sub-1e-4 on both. That is the license to build controllers.

Getting there was not free. Two things had to be true that were not:

**The armature has to be mirrored.** MuJoCo's `armature` (rotor inertia reflected through the gear ratio) adds to the diagonal of `M`. Pinocchio has a `model.armature` field for exactly this — but the obvious-looking `rotorInertia` attribute is *not* it, and setting the wrong one leaves you with a diagonal that is consistently light.

**The payload has to match.** The canonical stack is a Menagerie UR5e with a Robotiq 2F-85 mounted on the flange. MuJoCo knows about that gripper because it is in the compiled scene. Pinocchio only knows what the URDF tells it. Until I rewrote the `ee_link` inertial in `models/ur5e.urdf` with the compiled scene's actual mass, centre of mass, and inertia tensor, parity failed and gravity compensation drooped — because Pinocchio was holding up a robot that weighed less than the one MuJoCo was simulating.

The lesson generalizes past this lab: for cross-engine parity, the mounted payload is not bookkeeping. It is a load-bearing part of the model.

---

## Making the Robot Float

With `g(q)` trusted, the simplest possible dynamics controller is one line:

```
τ = g(q)
```

Cancel gravity, command nothing else. The equations of motion collapse to `M(q)q̈ + C(q,q̇)q̇ = 0` and the arm just... hangs there. No setpoint, no error signal, no stiffness. Push it and it moves; let go and it stays where you left it.

Measured on the canonical stack: max joint error while holding `Q_HOME` for the full run is **8.91e-06 rad**. Then I hit it with a 20 Nm / 10 Nm torque pulse on shoulder-lift and elbow and let it recover — final speed **0.0134 rad/s**, against a 0.1 rad/s criterion.

Worth being precise about what that second number means, because it is easy to oversell. Gravity compensation is not a stabilizing controller. It has no restoring term, so it does not pull the arm *back* to where it was — the perturbation test measures residual drift, not tracking. Hold-in-place and return-to-setpoint are different claims, and `τ = g(q)` only makes the first one.

But it is the right foundation. Every controller after this is `g(q)` plus something.

---

## The Actuator Problem, or: What Lab 2 Was Really Teaching Me

The lab brief was unambiguous on this point: *"Torque mode is critical. The MJCF actuator tags must change from `position` to `torque`. Without this, MuJoCo's internal PD controller masks the dynamics."*

I did not do that, and I think not doing it was the right call.

Swapping the actuator tags means editing the Menagerie model. The moment you do that, you are no longer simulating a UR5e — you are simulating an idealized 6-DOF arm with the UR5e's geometry and none of its actuation. Every torque you compute is delivered perfectly. That is a fine pedagogical robot and a bad rehearsal for hardware, where you never get to bypass the drive.

The alternative is to keep the Menagerie actuators and *invert* them. Each arm actuator is a `general` actuator with a fixed gain and an affine bias:

```
τ_actuator = gain·ctrl + bias0 + bias1·length + bias2·velocity
```

For the UR5e that is `gain = 2000, bias = (0, -2000, -400)` on shoulder-pan / shoulder-lift / elbow, and `gain = 500, bias = (0, -500, -100)` on the three wrist joints — a position servo with `Kp = 2000, Kd = 400` written in MuJoCo's generic actuator language. It is affine in `ctrl`, so it inverts in closed form:

```python
ctrl_i = (tau[i] - (bias0 + bias1 * length + bias2 * velocity)) / gain
```

That is the whole of `arm_torques_to_ctrl()` in `lab3_common.py`. Give it the torque you want, it hands you the control signal that produces it, clipped to the actuator's range.

Here is the part I enjoyed: this is the same insight Lab 2 handed me, generalized. In Lab 2 the Menagerie position servos drooped under gravity and lagged during motion, and the fix was a feedforward term:

```
ctrl = q_des + qfrc_bias/Kp + Kd·q̇_des/Kp
```

That took tracking from 133 mm to 0.088 mm RMS — a factor of 1500. At the time it looked like a trick for making position control work better. It is not a trick. It is a partial inversion of the actuator's bias model, in the special case where the torque you want happens to be the gravity torque. Lab 3 just does the general case: solve the same affine equation for arbitrary `τ`, not only for `τ = g(q)`.

The takeaway I actually keep: when a simulator's actuator model is in your way, read it before you replace it. Menagerie's actuators are documented, affine, and invertible. Deleting them would have been faster and would have taught me nothing.

---

## Impedance: A Spring You Can Design

Gravity compensation makes the arm neutral. Impedance control gives it a personality:

```
F = K_p·(x_d - x) + K_d·(ẋ_d - ẋ)
τ = Jᵀ·F + g(q)
```

You are not commanding a position. You are declaring that the end-effector should behave like a spring-damper anchored at `x_d`, and then letting `Jᵀ` translate that desired Cartesian wrench into joint torques. The robot goes toward the target, but if something is in the way it yields, with a compliance you chose on purpose.

`K_p` is the design knob, and its units are honest — newtons per metre, so you can predict the behaviour before you run anything. Under a 40 N load:

| `K_p` (N/m) | Feel | Deflection |
|---|---|---|
| 100 | soft | 104 mm |
| 500 | medium | 43 mm |
| 2000 | stiff | 17 mm |

Two implementation details that matter more than they look:

**Use `LOCAL_WORLD_ALIGNED` Jacobians.** Pinocchio offers `LOCAL`, `WORLD`, and `LOCAL_WORLD_ALIGNED`, and they are all "the Jacobian." If your stiffness matrix is expressed in world axes — which it is, because you think about stiffness as "stiff in Z, soft in XY" — then the Jacobian's linear rows had better be in world axes too. This is a whole-project rule now, not just a Lab 3 one.

**Use the Lie log for orientation error.** The 6-D controller originally took orientation error from the skew-symmetric part of `R_d·Rᵀ` — a small-angle approximation dressed up as a formula, fine near the setpoint and progressively wrong away from it. `pin.log3(R_d·Rᵀ)`, the actual logarithmic map on SO(3), is valid for large rotations at essentially no cost.

---

## Touching the Table on Purpose

Now the part the lab exists for. Not "avoid the table" — Lab 4 handles that. *Touch* the table, deliberately, at a controlled force, and slide along it.

The decomposition is classic hybrid position-force control. Partition the task space with selection matrices: XY is position-controlled, Z is force-controlled. Nothing is controlled twice, and every direction is controlled once.

```
S_p = diag(1, 1, 0)   position control in XY
S_f = diag(0, 0, 1)   force control in Z
```

XY runs an impedance law. Z runs PI on force error plus velocity damping:

```
e_f = F_desired - F_measured
F_f = -(K_fp·e_f + K_fi·∫e_f dt) - K_dz·ż
τ   = Jᵀ·(F_p + F_f) + g(q)
```

The integral term is what buys zero steady-state force error. The `K_dz = 30` velocity-damping term is what keeps the whole thing from chattering, and it is not optional: stiff contact plus integral action is an oscillator waiting for an excuse. You need both measurement smoothing and damping in the contact direction.

**Measuring the force was the sharpest trap in the lab.** The obvious move is a `<force>` sensor on the tool site. I did that, and it reported force when the arm was nowhere near the table. MuJoCo's `<force>` sensor measures the constraint forces on a body — which includes articulation forces from the arm holding itself up, not just contact. It is answering a different question than the one you asked.

The right instrument is `mj_contactForce()`, iterated over the actual contacts in `data.contact`, filtered to the pairs you care about. Contacts only exist when things are touching, so silence means "not in contact" and a reading means "in contact," which is the semantics a force controller needs. Raw per-contact forces are noisy, so they go through an EMA low-pass (α = 0.2) before reaching the PI loop.

And the filter has to cover the *real* contact set. My first version watched only the terminal tool body, on the reasonable-sounding theory that the tip touches first. It does not. With a Robotiq 2F-85 hanging off the flange, the first geom to reach the table can be `wrist_3_link` or any of the mounted gripper bodies depending on approach angle. Contact logic has to reflect the geometry you are actually simulating, not the idealized point-tip cartoon in your head.

---

## The Numbers

Static hold: descend, establish contact, regulate 5 N while holding XY.

| Metric | Value |
|---|---|
| Mean force | 4.89 N |
| Within 5 ± 1 N | 99.96 % |
| Max XY error | 3.60 mm |

Capstone: hold 5 N while tracing a 50 mm straight line across the table.

| Metric | Value |
|---|---|
| Within 5 ± 1 N | 94.07 % |
| Max XY error | 1.70 mm |

Sub-2 mm position error while regulating contact force to within a newton for 94% of a moving trace. The full suite behind those claims is 34 tests across dynamics, gravity compensation, impedance, and force control.

The honest caveat: **the line is 50 mm, and it lives near (0.4, 0.0) for a reason.** In the vertical tool configuration the arm uses for table contact, the X row of the Jacobian is tiny — on the order of 0.01 — which means a large joint motion buys a small Cartesian X motion. XY tracking bandwidth during contact is limited by the arm's own manipulability at that pose, not by the controller. Longer traces need slower trajectories, and traces further from that sweet spot degrade. That is a workspace-geometry limit, and the fix is a better pose or a redundant arm, not better gains.

Notice also that the static hold has *worse* XY error (3.60 mm) than the moving trace (1.70 mm). That is not a typo. The static number is dominated by the transient while contact is being established and the force integrator is winding up; the trace number is measured over a motion that begins already in stable contact. Different phases, different error sources. Reporting only the flattering one would have been easy and dishonest.

---

## What I Learned

**1. Cross-validate before you control.** Two independent implementations of the same physics, agreeing to 1e-5, is the cheapest bug insurance in robotics.

**2. Model mismatch looks like bad tuning.** A gripper payload missing from the URDF does not raise an exception. It shows up as droop, droop looks like insufficient stiffness, and chasing it with stiffness makes everything worse.

**3. Read the actuator model before replacing it.** Menagerie's position servos are affine and invertible. Inverting them keeps the real actuation in the loop; deleting them would have made the lab easier and less true.

**4. Choose instruments by what they actually measure.** `<force>` sensors and `mj_contactForce()` both return newtons. Only one of them returns *contact* newtons.

**5. Force control needs damping and filtering, not just PI.** Stiff contact plus integral action oscillates. EMA on the measurement, velocity damping in the contact direction, then tune.

**6. Simulator APIs are a dependency, not a constant.** Long after sign-off, MuJoCo 3.11 removed `MjData.qM` and re-signatured `mj_fullM` to `(model, data, dst)`. Three parity tests failed instantly. The fix was one helper in `lab3_common.py` probing `getattr(mj_data, "qM", None)` and dispatching to the old or new call — a one-line change because every raw simulator field this lab touches is already wrapped in the shared module. That habit was worth more than the fix.

---

## What's Next

Lab 4 goes back to avoiding contact — RRT*, real-geometry collision checking, TOPP-RA time parameterization — but Lab 3 is the one that makes Lab 5 possible. A gripper closing on a box is a force-control problem wearing a geometry costume. The `compute_impedance_torque` written here is imported directly by Lab 5's pick-and-place state machine and again by Lab 6's cooperative carry, where two arms welded to the same box have to negotiate internal forces they can only feel through contact.

And it is the reason a humanoid can eventually stand up in Lab 7. A robot that only knows positions cannot hold its own weight; it can only be told where its joints should be and hope the numbers were right. A robot that knows `M`, `C`, and `g` can be told what forces to make, and forces are the language the physical world actually speaks.

---

*Code: [github.com/ozkannceylan/mujoco-robotics-lab](https://github.com/ozkannceylan/mujoco-robotics-lab)*
