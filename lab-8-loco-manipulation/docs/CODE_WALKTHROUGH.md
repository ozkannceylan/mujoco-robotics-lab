# Lab 8 — Code Walkthrough

A reading order for the source, with the reasoning that shaped each file. Start
at [`ARCHITECTURE.md`](ARCHITECTURE.md) for the shape of the whole thing; this
document walks the code in the order it was built, which is also the order it
makes sense in.

Every claim with a number in it was measured. Where a number contradicts the
obvious design choice, that is the interesting part, and it is called out.

---

## 1. `g1_torque_model.py` — making the robot commandable

Menagerie ships the G1 with 29 `<position kp="500">` servos. Those compute a PD
law *inside* MuJoCo, so the only available command is a joint angle — there is
nowhere to inject a torque from an inverse-dynamics pipeline.

```python
actuator.set_to_motor()
actuator.inheritrange = 0
actuator.biasprm = [0.0] * len(actuator.biasprm)   # clear kp/kv leftovers
actuator.ctrlrange = frc                            # from actuatorfrcrange
```

Three details that are easy to skip and expensive to miss:

- **Torque limits come from the model**, each joint's `actuatorfrcrange`
  (5–139 N·m). The builder raises rather than silently produce an
  unlimited-torque robot.
- **The stand keyframe's `ctrl` is zeroed.** Those were position targets;
  reinterpreted as torques they are meaningless and dangerous.
- The model is built **at runtime via `MjSpec`**, not committed as XML, so
  Menagerie stays the single source of truth. `export_xml()` emits a snapshot
  when you want to read one.

The port from position to torque control removes every stabilising term the
servo was quietly providing. M0's ablation is the receipt:

| gravity mode | result | CoM drift | steady joint error |
|---|---|---|---|
| none (pure PD) | stands | 0.18 mm | 2.77 mrad |
| free-space `g(q)` | stands | 0.96 mm | 1.40 mrad |
| contact-consistent | stands | 0.62 mm | **0.00 mrad** |
| `g(q)` alone, no PD | **falls** | — | collapses in ~2 s |

Gravity compensation alone cannot hold a standing humanoid: it cancels weight
but adds no posture stiffness, and a standing robot is an inverted pendulum.

---

## 2. `lab8_common.py` — the frame contract

Everything that crosses the MuJoCo↔Pinocchio boundary lives here, so no call
site has to think about it.

```python
def mj_state_to_pin(mj_data):
    q = mj_qpos_to_pin(mj_data.qpos[:NQ])     # slice: scenes may append bodies
    v = mj_data.qvel[:NV].copy()
    ...
    v[0:3] = R.T @ mj_data.qvel[0:3]          # world → body base twist
```

The base-twist rotation is the kind of bug that stays invisible for a long time:
MuJoCo reports the floating-base linear velocity in the **world** frame,
Pinocchio's FreeFlyer expects it in the **local body** frame. Get it wrong and
nothing happens until the base actually moves, at which point the Coriolis terms
are silently wrong.

The slicing to `[:NQ]`/`[:NV]` is M5's contribution: once a scene appends a
freejoint payload, `qpos` is longer than the robot. Same story for
`robot_com()` — `subtree_com[0]` is the *world* subtree, which quietly becomes
"the scene's centre of mass" the moment props exist. The pelvis subtree is the
robot and nothing but the robot.

LIPM primitives live here too:

```python
def lipm_omega(com_height):          return sqrt(GRAVITY / com_height)
def divergent_component(c, ċ, ω):    return c + ċ / ω
```

`com_height` is the CoM height **above the contact plane**, and it is measured
on the settled robot rather than assumed — ω is the one number coupling the plan
to the machine, and the G1 settles ~15 mm below its keyframe.

---

## 3. `wb_tasks.py` — what a task is

A task supplies a Jacobian, an error, and a drift term, and the QP asks it for a
desired acceleration:

```python
ẍ_des = ẍ_ref + k_p·e + k_d·(ẋ_ref − ẋ)
```

The feedforward terms are not decoration. On M1's hand circle, supplying the
trajectory's own ẋ and ẍ took tracking from 18.63 mm RMS to **7.08 mm** — most
of the residual was pure lag.

`TaskStack.update_dynamics` runs forward kinematics with **zero** acceleration,
which makes every frame/CoM acceleration Pinocchio reports equal to the pure
`J̇q̇` drift — exactly the term the QP must cancel. It also computes the
centroidal momentum matrix and its time variation, cheaply enough to do
unconditionally rather than duplicate the kinematics pass.

### `DCMTask`

Same Jacobian as `CoMTask` (the CoM one, horizontal rows). What changes is the
desired acceleration:

```python
def desired_acceleration(self, model, data, q, v, damping_gain=None):
    com = self.current_com(data)[:2]
    return self.omega**2 * (com - self.commanded_vrp(data))
```

and `commanded_vrp` is the DCM control law, clamped into the current support
polygon. The clamp matters: commanding a ZMP outside the feet asks for a wrench
the QP's CoP rows forbid, so without it the request is quietly traded away in
the cost and the controller believes it is still in charge. With it, saturation
is visible in `vrp_saturated` — and reading that flag is what diagnosed M3.

A leaky integrator is present and **defaults to off**. It was added to cancel a
−0.09 m/s² acceleration bias, worked, and then became pure phase lag once the
bias was fixed at its source (the contact model and the solver tolerance):
12/12 steps → 8/12. An integrator is a way of not knowing what your error is.

### `CentroidalAngularMomentumTask`

```python
def desired_acceleration(self, model, data, q, v, damping_gain=None):
    momentum = self.momentum(data, v)             # A_g[3:6] @ v
    return self.axis_scale * (-self.gain * (momentum - self.reference))
```

`reference` is zero for a held pose and `L_ref` for a task that deliberately
moves mass. Per-axis weighting exists and measured as *not* the answer to M4's
reach problem; it is kept because the capability is real and the negative result
is worth being able to reproduce.

---

## 4. `wb_id_qp.py` — the control path

47 decision variables in double support: 35 joint accelerations plus a 6D wrench
per stance foot.

```
equalities    M[:6] q̈ − J_cᵀ[:6] f = −h[:6]        unactuated base
              J_c q̈ = −J̇_c q̇                       stance feet hold
inequalities  friction pyramid, CoP box, f_z ≥ f_min, |τ| ≤ τ_max
readout       τ = M[6:] q̈ + h[6:] − J_cᵀ[6:] f
```

The CoP rows are where the foot's actual geometry enters:

```python
([ h, 0, cx + hl, 0,  1, 0], 0.0, inf),    # CoP_x ≤ cx + hl
([-h, 0, hl - cx, 0, -1, 0], 0.0, inf),    # CoP_x ≥ cx − hl
```

The `h` terms carry the shear correction — the wrench is expressed about a frame
0.035 m above the ground, so `CoP_x = (−m_y − h·f_x)/f_z`, not `−m_y/f_z`. At
the shear a walking step uses, the difference is ~12 mm on an axis with 85 mm of
total travel.

**Solver settings deserve their own paragraph** because the intuitive choice is
backwards. At `eps = 1e-6`, 38 % of ticks returned `maximum iterations reached`
at 12.6 ms per solve against a 1 ms budget. At `1e-4` every tick converges in
~25 iterations and 0.073 ms — *and the constraint residual drops* from 0.021 to
8.5e-5 N·m. The tolerance was below what a cost spanning weights 1e4…1e1 against
a 1e-4 regularisation can deliver, so the solver spent its whole budget not
converging.

`set_contacts` discards the OSQP instance rather than hot-updating it: the
factorisation is built for a fixed dimension, and at 0.07 ms a rebuild on two
phase transitions per step is free. Correctness beats reusing a stale
factorisation.

---

## 5. `gait_planner.py` + `dcm_planner.py` — where to put the feet, and where the body is going

`GaitSchedule` owns the timeline. Footstep placement puts the swing foot
`step_length` **ahead of the stance foot**, not ahead of its own previous
position — stepping ahead of your own footprint advances the body by half a
stride and is a shuffle; passing the stance foot is a walk.

`DCMPlan` builds a piecewise-linear ZMP and solves the DCM backwards. For a
segment with `p(τ) = p₀ + kτ`:

```
ξ(τ)  = A e^{ωτ} + p(τ) + k/ω,     A = ξ₀ − p₀ − k/ω
ξ̇(τ)  = ω A e^{ωτ} + k
backward step:  A = (ξ_T − p_T − k/ω) e^{−ωT}
```

Constant-ZMP segments are the `k = 0` case of the same formula, which is why one
class handles both.

One subtlety worth the space, because the obvious fix is wrong. A DCM tracking a
ramping ZMP leads it by `k/ω` in steady state, so the plan starts with ξ about
30 mm off-centre while the robot stands still — a textbook initial-condition
mismatch. Splitting the settle into a hold plus a short sweep removes it
cleanly, and takes the gait from **12/12 steps to 6/12**. The lead is not an
error: it is the lateral momentum the first step needs, and the settle is the
robot acquiring it. `settle_sweep` defaults to 1.0 (no hold) and the split path
is kept only so the reasoning can be re-run.

---

## 6. `locomotion_controller.py` — gait meets QP

Three guards make contact switching survivable:

**The stance set is scheduled intent ∩ measured contact.**

```python
confirmed = tuple(name for name in scheduled if measured.get(name, False))
```

Listing a foot that is actually in the air hands the controller an imaginary
support polygon and imaginary forces. Measured, in M2: the schedule declared
double support while the landing foot was still 60 mm up, the QP planned against
two feet, and the robot launched itself — foot 0.66 m in the air, torques
saturated, fall at 4.0 s.

**The swing task fades in** over the first fraction of the swing, so lift-off
does not begin with a step change in acceleration demand at the same instant
support is halved.

**The commanded ZMP is clamped** to the union of the stance feet's contact
patches, using the *same* asymmetric patch the QP's CoP rows use. A clamp that
disagreed with the constraint it protects would be worse than none.

---

## 7. `capstone_scene.py` + `m5_capstone.py` — sequencing proven regimes

The capstone invents no new control. Its phases are configurations earlier
milestones validated — approach walk = M3, standing reach = M1, carry walk =
M4's carry — and its job is to survive the *transitions*. All ten of its defects
were in transitions, and only three were control problems.

The grasp:

```python
def set_weld(self, active, which="right"):
    for weld_id in self._weld_ids(which):
        if active:
            self._capture_relative_pose(weld_id)   # ← the whole point
        self.data.eq_active[weld_id] = int(active)
```

`mjEQ_WELD` holds body2 at the relpose stored in `model.eq_data`, and that field
is baked at **compile time** — from the rest pose, where the hand is at
x = −0.02 and the payload at x = 0.40. Activating without refreshing it does not
grasp the payload; it commands a 0.42 m snap, and the simulator delivers exactly
that: the payload leapt 0.115 m and took the robot down.

The payload entering the model:

```python
inertia   = pin.Inertia.FromBox(mass, 2a, 2a, 2a)
placement = frame.placement * pin.SE3(np.eye(3), offset)
model.appendBodyToJoint(frame.parentJoint, inertia, placement)
return model.createData()     # every holder of the old data must follow
```

`nq`, `nv` and every frame id are unchanged — the payload adds inertia to an
existing joint, not a degree of freedom — so tasks and QP dimensions survive
untouched. Only `M`, `J_com` and `A_g` move. The gait is then replanned on the
robot that now exists.

Placing, finally, is about the object rather than the hand:

```python
target = self.payload_goal_to_hand(waypoint)   # recomputed every tick
```

The grip is compliant by design, so the load settles over a 25 s sequence and a
hand target derived from one pre-motion measurement bakes that drift into the
result — a systematic 55 mm placement error. Servoing the payload took release
accuracy to 18.9 mm. And the last 20 mm turned out to be a statics problem about
the shelf, not a controls problem about the arm: a target 0.09 m in from a
0.10 m half-extent pedestal leaves a 30 mm box overhanging, and it tips off.

---

## 8. Reading the tests

`tests/` is 97 tests and they are organised by what could silently be wrong:

- `test_torque_model.py` — actuator semantics, torque limits, ctrl→force
  mapping, keyframe hygiene, and **model parity**: `g(q)` and `M(q)` against
  MuJoCo's own, to 1e-16 relative. If Pinocchio and MuJoCo disagree, nothing
  above them means anything.
- `test_wb_tasks.py` — every task Jacobian against finite differences, drift
  terms, frame conventions, and QP force balance / friction / CoP / torque
  limits.
- `test_gait.py` — phase sequencing, contact sets never empty, swing continuity
  and zero touchdown velocity, feedforward vs finite differences.
- `test_dcm.py` — the DCM plan against its own defining ODE, continuity across
  phase boundaries, terminal rest, the control law reducing to the planned ZMP
  under perfect tracking, and the CoP box against the real foot geometry.

That last group is the pattern worth copying: a planner is tested against the
differential equation it claims to solve, not against a golden trajectory.

---

## Running things

```bash
./tools/setup_env.sh          # deps + Menagerie clone (a bare clone is not enough)
export MUJOCO_GL=egl          # headless rendering

python3 lab-8-loco-manipulation/src/m0_torque_standing.py
python3 lab-8-loco-manipulation/src/m1_standing_reach.py
python3 lab-8-loco-manipulation/src/m2_stepping.py
python3 lab-8-loco-manipulation/src/m3_walking.py            # --in-place re-runs M2's gate
python3 lab-8-loco-manipulation/src/m4_walk_reach.py         # --mode carry|reach|both
python3 lab-8-loco-manipulation/src/m5_capstone.py           # --no-video for speed

pytest lab-8-loco-manipulation/tests/                        # 97 tests, ~5 s
```

Each demo prints its own gate table and writes its evidence to `media/`. A gate
that fails exits non-zero.
