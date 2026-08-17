# Lab 8 — Architecture

Whole-body loco-manipulation on a torque-actuated Unitree G1: how the pieces
fit, and why each one is shaped the way it is. For the milestone-by-milestone
narrative see [`../tasks/LESSONS.md`](../tasks/LESSONS.md); for a guided read of
the code itself see [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md).

---

## The one-paragraph version

Pinocchio computes the model — mass matrix, CoM Jacobian, centroidal momentum
matrix, frame Jacobians. A quadratic program solves, once per millisecond, for
joint accelerations *and* contact wrenches together, subject to the unactuated
base dynamics, the stance feet, friction, centre-of-pressure limits and torque
limits. Joint torques are read out of the QP's actuated rows and applied to
MuJoCo. Everything above that — walking, reaching, carrying — is a matter of
which *tasks* are in the QP's cost and what they are asked for.

```
gait plan (DCM)   hand targets   payload
      │                │            │
      ▼                ▼            ▼
   task stack:  DCM · swing foot · pelvis · momentum · hands · posture
      │            (Pinocchio, LOCAL_WORLD_ALIGNED, J̇q̇ drift + feedforward)
      ▼
   whole-body inverse-dynamics QP  (OSQP, ~0.07 ms)
      variables:    q̈ (35)  +  contact wrenches f (6 per stance foot)
      equalities:   unactuated base rows · stance feet hold
      inequalities: friction pyramid · CoP inside foot · f_z ≥ f_min · |τ| ≤ τ_max
      ▼
   τ = M[6:]q̈ + h[6:] − J_cᵀ[6:] f     →  MuJoCo, torque actuators, 1 kHz
```

---

## Why a QP at the acceleration level

This is the single most important structural decision in the lab, and it was
made by measurement rather than by preference.

A velocity-level ("kinematic") QP — `min ‖J q̇ − ẋ_des‖²` — was written first
and is kept in `wb_qp.py`, labelled unfit for balance. It cannot balance a
floating-base robot, and the reason is not tuning: **CoM motion is produced by
contact forces, which a kinematic QP does not represent.** Such a QP can
satisfy `J_com q̇ = 0` exactly while the robot topples. The tell-tale symptom,
measured on the G1: making the *hand* task stronger made the robot fall
*sooner*.

Solving at the acceleration level with contact wrenches as decision variables
turns "keep the CoM over the feet" from a wish into a constraint the solver can
enforce, because the forces that would do it are variables it owns.

Two consequences follow throughout the lab:

- **The base rows are a constraint, not a command.** The first six rows of
  `M q̈ + h = Sᵀτ + J_cᵀ f` involve no actuator. They enter as an equality on
  `(q̈, f)`; the remaining 29 rows *define* τ, which is read out afterwards.
- **Contact geometry is part of the controller.** The CoP inequality is where
  the robot's feet enter the mathematics, and getting that geometry wrong is
  indistinguishable from a control bug — see "Contact model" below.

## Why DCM and not a CoM trajectory

Under the linear inverted pendulum the CoM obeys `c̈ = ω²(c − p)` with
`ω = √(g/z_c)` and `p` the ZMP. That splits into

```
ξ = c + ċ/ω        divergent      ξ̇ = ω(ξ − p)      ← unstable, must be steered
η = c − ċ/ω        convergent                        ← stable, needs no control
```

Only `ξ` can run away, and it is steered by the ZMP, which the QP can place
anywhere inside the support polygon. So the controller commands the divergent
component and lets the CoM travel freely.

The alternative — commanding CoM *position* — works for stepping in place and
provably does not walk: it needs a moment of rest over each foot that forward
walking never provides. Measured, the quasi-static reference reached 3 of 10
steps at every stride length, double-support duration and CoM bias tried.

`dcm_planner.py` plans a piecewise-linear ZMP through the footsteps and
**back-integrates** `ξ̇ = ω(ξ − p)` from a terminal rest condition. Backwards is
the only stable direction: forward integration of an unstable system amplifies
the boundary error by `e^{ωT}`, the backward recursion contracts it by
`e^{−ωT}`.

`wb_tasks.DCMTask` then commands

```
p_cmd  = ξ − ξ̇_ref/ω + (k/ω)(ξ − ξ_ref)          clamped into the stance feet
c̈_des = ω²(c − p_cmd)  =  −ω·ċ + ω·ξ̇_ref − ω·k·(ξ − ξ_ref)
```

Note what is absent: any term pulling the CoM toward a commanded position.

## Why a centroidal momentum task exists

Locomotion is a centroidal problem and the CoM Jacobian includes the arms, so a
hand task is not an independent addition — every kilogram the QP accelerates to
satisfy a hand target lands in the quantity keeping the robot upright.

No hand-task weight both walked and tracked. The failures were **non-monotonic**
(weight 1e1 walked with a 46 mm droop, 2e1 fell at step 5, 1e2 at step 7, 3e2 at
step 5), which is the signature of a missing term rather than a mis-tuned one.

`CentroidalAngularMomentumTask` regulates `L = A_g(q) q̇`, letting the QP say
"the arms may move, but they may not spin the body" instead of restraining them
through a joint-space posture pull doubling as a momentum damper. The same hand
task that fell on step 7 then walked the full distance with three times better
tracking.

Two scoping rules were learned the hard way:

- It is an **arm-task companion, not a global stabiliser**. Enabled across a
  bare walk it cancels the angular momentum walking itself generates (the gait
  runs ±2 kg·m²/s of roll) and fells the robot on step 2.
- Its reference is zero for a *held* pose and `L_ref` for a task that
  deliberately moves mass (resolved momentum control, Kajita et al. 2003) —
  otherwise it fights the trajectory it was added to enable.

## Contact model

The QP's CoP rows describe the foot. Menagerie's G1 sole is four contact spheres
at x ∈ {−0.05, 0.12}, y ∈ {±0.025, ±0.03}, z = −0.03 in the ankle-roll frame,
so the honest description is a patch of half-length 0.085 centred **0.035 m
ahead of the frame**, sitting 0.035 m below it.

An earlier symmetric ±0.08 m box centred on the ankle was wrong in three ways at
once, and each mattered:

| error | consequence |
|---|---|
| over-claimed 30 mm of rearward CoP | the QP planned wrenches MuJoCo refused to produce — a constant force error |
| discarded 40 mm of forward CoP | threw away the authority that decelerates the CoM before touchdown |
| ignored the frame's height above ground | `CoP = −m_y/f_z` is only right at zero shear; walking is exactly when it is not |

With the patch offset and the `h·f` shear term
(`CoP_x = (−m_y − h·f_x)/f_z`), commanded-versus-realised CoM acceleration went
from slope 0.78 with a −0.09 m/s² bias to slope 0.95 with correlation 0.995.

Standing cannot distinguish these models. Walking uses both ends of the foot
every step.

---

## Modules

| File | Role |
|---|---|
| `g1_torque_model.py` | Builds the torque-actuated G1 from Menagerie via `MjSpec`: 29 `<position>` servos → `<motor>`, ctrlrange from each joint's `actuatorfrcrange`, keyframe ctrl zeroed, floor + light added |
| `lab8_common.py` | Paths, constants, loaders, MuJoCo↔Pinocchio state conversion, LIPM primitives, CoM/contact/support-polygon helpers, payload attachment |
| `standing_controller.py` | Joint PD + selectable gravity mode; used only to settle the robot before the QP takes over |
| `wb_tasks.py` | Task definitions and the stack: frame position/pose/orientation, CoM, **DCM**, **centroidal angular momentum**, posture |
| `wb_id_qp.py` | The control path: acceleration-level inverse-dynamics QP with contact wrenches |
| `wb_qp.py` | Velocity-level QP — kinematic sub-problems only, **not** balance |
| `gait_planner.py` | Phase timeline, contact sets, swing references with feedforward, footstep placement (`step_length`, `step_width`, `first_swing`, `close_stance`) |
| `dcm_planner.py` | ZMP through the footsteps + the DCM trajectory it generates |
| `locomotion_controller.py` | Gait → QP wiring: measured-contact stance set, swing ramp-in, ZMP clamp, telemetry |
| `capstone_scene.py` | M5 scene: pedestals, freejoint payload, two weld grasps with live relative-pose capture |
| `mN_*.py` | One runnable gate demo per milestone, each writing its own evidence to `media/` |

The torque model is **generated at runtime, not committed**: Menagerie stays the
single source of truth, and `export_xml()` can emit a snapshot for inspection.

---

## Data flow, one control tick

```
mj_data.qpos, qvel
   │  mj_state_to_pin — pelvis z-offset, quaternion order, base twist world→body,
   │                     sliced to the robot (a scene may append free bodies)
   ▼
q (36), v (35)
   │  TaskStack.update_dynamics — FK with zero acceleration, so every reported
   │                               frame/CoM acceleration *is* the J̇q̇ drift
   ▼
Jacobians · drifts · A_g · J_com
   │  each task: desired_acceleration = ẍ_ref + k_p·e + k_d·(ẋ_ref − ẋ)
   ▼
WholeBodyIDQP.solve  →  q̈, f, τ
   │
   ▼
mj_data.ctrl = τ   →   mujoco.mj_step
```

Kinematics are evaluated **once per tick**, not once per task: with six tasks
over a 35-DOF model the redundant passes would dominate the tick.

### Conventions that are not negotiable

- Jacobians are `pin.LOCAL_WORLD_ALIGNED` — translation rows are world-aligned,
  so a world-frame positional error maps straight onto them.
- Pinocchio's world sits `PELVIS_MJCF_Z` = 0.793 m below MuJoCo's. Task targets
  are given in **MuJoCo world coordinates** and converted internally, so callers
  never juggle two frames.
- `nq ≠ nv` on a floating base (36 vs 35). Configuration updates go through
  `pin.integrate`, never `q += dq`.
- Every task Jacobian is validated against finite differences in
  `tests/test_wb_tasks.py`.

---

## The gait plan

`GaitSchedule` lays out `DS → SS → DS → SS …` and, for any `t`, returns the
contact set, the swing-foot reference (position, velocity, acceleration) and the
phase index. `DCMPlan` turns the same timeline into a ZMP and the DCM arc it
generates.

Three parameters carry most of the behaviour:

- **`step_width`** — the dominant gait parameter. The ZMP crosses between the
  feet every step and the lateral DCM swings with it, so the cost of lateral
  balance is set by how far apart the feet are. The G1's 0.237 m rest stance
  gives 7 of 12 steps and a fall; 0.18 m gives 12 of 12.
- **`first_swing`** — step with the *trailing* foot. From a level stance this is
  equivalent to always-left; resuming a gait mid-stride it is the difference
  between walking and re-stepping the leading foot into the place it already
  stands.
- **`close_stance`** — end a walk with the feet together, the way a real walk
  ends, so the *next* walk starts from a stance the controller has seen.

The last two are only observable in a sequence that walks more than once — they
were latent through M3 and M4 and surfaced in the capstone.

---

## Manipulation and the payload

The grasp is a MuJoCo `mjEQ_WELD`, which the brief permits ("grasp stays
SIMPLE"): the G1 model here has no hand, and Lab 5 already validated a real
parallel-jaw grasp. What Lab 8 tests is whether the *whole-body* controller
survives acquiring, carrying and releasing mass.

Two things about welds are easy to get wrong and both cost a fall:

- A weld holds its **compile-time** relative pose. `eq_active` is a switch, not
  a "grasp here" instruction — activating it naively commanded a 0.42 m snap.
  `CapstoneScene.set_weld` writes the live hand→payload transform into
  `eq_data[3:10]` first, and refuses to close across a gap.
- Two welds make a **closed kinematic chain** the QP does not model. The
  capstone therefore picks one-handed, carries two-handed, and opens the second
  weld before placing.

At the grasp, `attach_payload_to_pinocchio` folds the payload's inertia into the
wrist's parent joint and returns fresh `pin.Data`. Frame ids, `nq` and `nv` are
unchanged — the payload adds inertia to an existing joint rather than a degree
of freedom — so tasks and QP dimensions survive untouched; only `M`, `J_com` and
`A_g` move. The gait is then **replanned**, because a plan describes a specific
configuration and picking mass up is a change to it.

---

## Solver settings

OSQP, warm-started between ticks, rebuilt when the contact set changes shape.

`tolerance = 1e-4`, `max_iterations = 2000`. The earlier `1e-6` was below what
the problem's conditioning can deliver — a cost spanning task weights 1e4…1e1
against a 1e-4 regularisation — and 38 % of control ticks returned `maximum
iterations reached` at 12.6 ms per solve. At `1e-4` every tick converges in ~25
iterations and **0.073 ms**, and the base-dynamics constraint residual *falls*
from 0.021 to 8.5e-5 N·m. Asking for less accuracy produced a more accurate
answer.

A hit iteration cap is a correctness warning, not a performance note: the
returned point is wherever the solver happened to be.

---

## Cross-lab dependencies

Lab 7 supplies the G1 conventions — joint ordering, qpos/qvel slices, the pelvis
MJCF offset, quaternion conversions — re-exported through `lab8_common` so
downstream modules import one namespace.

Foreign labs are added to `sys.path` with **`append`, never `insert(0)`**. Labs
share module names (`standing_controller`, `record_demo`), and putting a foreign
`src/` ahead of this lab's own silently shadows local modules with another lab's
implementation.

Lab 9's data pipeline depends on these controllers, which is why the walking and
carrying regimes are documented here in terms of *what was measured*, not what
was intended.
