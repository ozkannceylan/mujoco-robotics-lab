# Lab 8 — Lessons

> Live journal. Log bugs/fixes/insights AS THEY HAPPEN (Symptom / Root cause / Fix /
> Takeaway). Seeded at kickoff with the inherited constraints this lab exists to answer.

## Inherited at kickoff (2026-08-14)

### I-1: Position actuators cannot walk (Lab 7's terminal finding)
Menagerie G1 `<position>` servos track quasi-static references only; dynamic ZMP
replay fails structurally (Lab 7 M3e, 6 attempts — IK converges, PD replay diverges).
Lab 8's entire M0 exists because of this: torque actuators + RNEA inverse dynamics
are the unblock. If M3 walking fails here too, the diagnosis to revisit is the
*controller*, not the actuator model — that hypothesis has already been spent.

### I-2: The analytical model must model the simulated body (Lab 5 L-6.1c)
Build the Pinocchio model from the SAME MJCF MuJoCo steps (`g1_torque.xml`), and
gate M0 on g(q)/M(q) cross-validation — not just FK.

### I-3: Raw diagonal PD chatters on small-inertia joints at 1 kHz (Lab 5 L-6.1b)
Kd·dt/I > 2 is discretely unstable. Shape gains through M(q) from the start; G1
wrist/ankle joints are exactly the risk class.

### I-4: State machines need convergence gates and post-conditions (Lab 5 L-6.1e/f)
Transitions gate on measured convergence with logged residuals; the capstone asserts
the payload's final pose. A run must not be able to claim success silently.

### I-5: Evidence discipline (Lab 7 cleanup lesson)
Every `media/` file has exactly one producing script writing exactly that name;
delete outputs of deleted pipelines in the same commit.

## Session log

### M0 — Torque-Actuated G1 Bring-Up (2026-08-15) — GATE PASSED

Four findings, two of which contradict what this lab inherited.

#### L-M0-a: Gravity compensation alone cannot stand
- **Symptom**: `τ = g(q)` (free-space RNEA gravity, no feedback) collapses the G1
  from a 0.79 m pelvis height to 0.097 m in about 2 s.
- **Root cause**: Not a bug — a category error. Gravity compensation cancels
  weight; it does not *stabilise posture*. A standing humanoid is an inverted
  pendulum: any deviation grows, and g(q) has no restoring term. Lab 7's
  position servos hid this because their internal PD supplied the missing
  stiffness.
- **Fix**: Posture feedback is mandatory, not optional garnish. M0's controller
  is `τ = Kp(q_nom − q) + Kd(−q̇) + g_comp(q)`.
- **Takeaway**: When porting from position to torque control, every stabilising
  term the servo provided for free must be re-supplied explicitly. Enumerate
  them before blaming the plant.

#### L-M0-b: Inertia-shaping the PD gains makes the humanoid FALL
*(This one overturns inherited lesson I-3.)*
- **Symptom**: Lab 5's L-6.1b fix — `τ = M(q)(Kp·e + Kd·ė) + g` — was carried in
  as a best practice. With it the G1 falls in every gain setting tried
  (Kp = 100/400/500/900/2000): pelvis 0.79 → 0.50 m, ~900 mrad joint error.
  The *unshaped* law stands with 0.18 mm CoM drift at Kp = 500.
- **Root cause**: `M(q)[6:, 6:]` is the joint block of a **floating-base** mass
  matrix. It describes the inertia felt when the pelvis is free to react — for
  a standing robot each leg joint appears to carry the whole body, giving
  entries orders of magnitude larger than the effective inertia under the
  closed kinematic chain the feet actually form. Multiplying gains by that
  saturates the actuators and destroys the intended error dynamics.
- **Fix**: Use raw joint-space gains for floating-base posture control. Keep
  inertia shaping for fixed-base arms (Lab 5), where `M(q)` genuinely is the
  reflected inertia.
- **Takeaway**: A fix is only valid inside the model assumptions that produced
  it. "Scale gains by M(q)" is a fixed-base rule; the same expression names a
  different physical quantity once the base floats. Test inherited fixes on the
  new platform before trusting them — this one looked authoritative and was
  wrong here.

#### L-M0-c: Free-space gravity compensation over-actuates a robot in contact
- **Symptom**: Both `NONE` and `FREE_SPACE` modes stand, but their steady-state
  joint error differs by ~2× (2.77 vs 1.40 mrad), and free-space gravity is
  visibly the wrong quantity: it compensates weight the *ground* is already
  carrying.
- **Fix**: `CONTACT_CONSISTENT` mode subtracts MuJoCo's generalized constraint
  forces: `τ_g = g(q) − τ_constraint`. Steady-state joint error → 0.00 mrad.
- **Takeaway**: This is the standing special case of the contact-consistent
  inverse dynamics M1 must build properly (constraint forces there come from
  the QP's contact-wrench variables, not from reading the simulator). The
  ablation is kept in the M0 demo so the progression is measured, not asserted.

#### L-M0-d: Cross-lab `sys.path` insertion silently shadows local modules
- **Symptom**: `ImportError: cannot import name 'GravityMode' from
  'standing_controller'` — pointing at **Lab 7's** file, not Lab 8's.
- **Root cause**: `add_lab_src_to_path()` (copied from the repo's cross-lab
  convention) used `sys.path.insert(0, …)`. Lab 7 also has a
  `standing_controller.py`, so its `src/` landed ahead of Lab 8's own directory
  and won the import.
- **Fix**: Append foreign lab paths; insert only this lab's `src/` at position
  0. See `lab8_common.add_lab_src_to_path`.
- **Takeaway**: Repo-wide, the convention `sys.path.insert(0, foreign_src)` is a
  latent trap wherever two labs share a module name (`lab*_common` is unique,
  but `standing_controller`, `grasp_planner`, `record_demo` are not). Now
  documented in CLAUDE.md's Known Issues.

#### Confirmations (no fix needed)
- **Pinocchio CoM sits in the FreeFlyer frame**: adding `PELVIS_MJCF_Z = 0.793`
  reproduces MuJoCo's `subtree_com[0]` to 0.000000 mm. Lab 7's finding holds
  exactly on the torque model.
- **Model parity**: g(q) 1.7e-16 and M(q) 9.3e-17 relative error across 6
  configurations — the MjSpec actuator swap leaves the multibody dynamics
  untouched, as intended (and unlike Lab 5's URDF/MJCF mismatch, this was
  verified rather than assumed).
- **Zero torque falls** (covered by a test): the torque model gives nothing for
  free. That is the whole point of M0.

#### Environment notes
- `MjsActuator.set_to_motor()` (MuJoCo ≥ 3.11) is the clean conversion path;
  ctrlrange is taken from each joint's `actuatorfrcrange` (5–139 N·m for G1).
- Menagerie's `stand` keyframe carries a **ctrl** vector of position targets.
  Under `<motor>` actuators those numbers would be newton-metres, so the
  builder zeroes keyframe ctrl while keeping qpos.
- MuJoCo 3.11 replaced `MjsLight.directional` (bool) with `MjsLight.type`
  (`mjtLightType`); the floor/light helper handles both.

#### L-M0-e: `pytest lab-*/tests/` was broken repo-wide (found, fixed)
- **Symptom**: Running two labs' suites in one pytest process fails at
  collection: `ImportError: cannot import name 'ik_pseudoinverse' from
  'a4_inverse_kinematics'` — pointing at **Lab 1's** file while collecting
  **Lab 2's** tests. Every lab passed in isolation, so it had gone unnoticed;
  CLAUDE.md documents `pytest lab-*/tests/` as the all-tests command.
- **Root cause**: The same `sys.modules` shadowing as L-M0-d, one level up.
  Labs reuse module names by convention (`a4_inverse_kinematics` and
  `b1_trajectory_generation` in Labs 1 and 2, `standing_controller` in Labs 7
  and 8). Each lab's tests prepend their own `src/`, but the first lab
  collected already owns the name in `sys.modules`, so later labs import the
  wrong file. Latent for months; surfaced the moment Lab 1 gained a test suite.
- **Fix**: Repo-root `conftest.py` with a `pytest_collectstart` hook that, per
  test file, moves the owning lab's `src/` to the front of `sys.path` and
  evicts same-named modules imported from a different lab.
- **Result**: `pytest lab-*/tests/` → **224 passed** (Labs 1/2/3/4/5/7/8).
- **Takeaway**: "Every suite passes" is not the same claim as "the suite
  passes". Test the documented command, not just the convenient one.

---

### M1 — Whole-Body QP, Standing Reach (2026-08-15) — GATE PASSED

#### L-M1-a: A kinematic (velocity-level) QP cannot balance a humanoid
- **Symptom**: The first implementation followed `plan/LAB_08.md`'s cost
  literally — `min ‖J q̇ − ẋ_d‖²` solved for joint velocity, then tracked with
  the M0 joint servo. It stood still fine and fell over during every reach.
  Diagnostic runs showed the QP's predicted base velocity agreeing with the
  simulation for ~0.3 s and then diverging as the robot toppled.
- **The signature that named the bug**: making the hand task *stronger* made
  the robot fall *sooner* (weights 1e2 → 1e3 → 1e4 all fell; only a weak hand
  task survived, and then tracked with ~50 mm error). A controller that gets
  worse as you ask it to do its job better is optimising the wrong variable.
- **Root cause**: CoM motion is not commanded by joint velocity. `J_com q̇ = 0`
  can hold exactly while the robot rotates about its ankles, because the CoM
  is accelerated by *contact forces*, which a velocity-level QP does not model.
  The feet also cannot be "held" kinematically: the QP assumes the floating
  base will follow its prediction, and nothing makes it.
- **Fix**: `wb_id_qp.py` — acceleration-level inverse-dynamics QP with the
  contact wrenches as decision variables:
  `min Σ w‖J q̈ + J̇q̇ − ẍ_des‖²` subject to the unactuated base dynamics
  `M[:6] q̈ + h[:6] = J_cᵀ[:6] f`, the stance constraint `J_c q̈ + J̇_c q̇ = 0`,
  friction/CoP/unilateral limits on `f`, and torque bounds. Torque is read out
  of the actuated rows. Same tasks, same weights — and now the hand task can be
  made *stronger* to get **better** tracking, which is the sanity check that
  the formulation is right.
- **Takeaway**: For a floating-base robot in contact, "which variables does the
  optimiser control" is the whole design. Balance is a statement about forces;
  a solver that cannot represent forces can only pretend to enforce it.
  `wb_qp.py` is kept for genuinely kinematic sub-problems (swing-foot
  retargeting in M2), clearly labelled as unsuitable for balance.

#### L-M1-b: Feedforward is most of the tracking error on a moving target
- **Symptom**: The gate passed at 18.63 mm hand RMS against a 20 mm limit —
  uncomfortably close, and the error plot was a clean lag, not noise.
- **Root cause**: `ẍ_des = k_p·e − k_d·ẋ` has no knowledge of the reference's
  own velocity/acceleration, so a moving target is chased rather than tracked.
- **Fix**: `ẍ_des = ẍ_ref + k_p·e + k_d·(ẋ_ref − ẋ)`, with the circle's
  analytic derivatives passed in. RMS 18.63 → **7.08 mm** with no gain change.
- **Takeaway**: Before tuning gains against a lag error, check whether the
  trajectory's derivatives are simply missing. Gains trade stability for
  tracking; feedforward is free.

#### Environment / API notes
- OSQP requires the **upper triangle** of `P`, and its hot `update()` needs a
  *fixed* sparsity pattern. Building the CSC matrix from explicit
  `np.triu_indices` (keeping numerically-zero entries) avoids
  `ERROR in osqp_update_data_mat: new number of elements out of bounds`.
- OSQP 1.x renamed the `polish` setting to `polishing`.
- QP cost: 47 variables (35 accelerations + 12 contact-wrench components),
  ~0.11 ms mean solve — 1 kHz control is comfortable, so the rate reduction
  contemplated in ARCHITECTURE's open questions is not needed yet.

---

### M2 — Torque-Level Stepping (2026-08-15) — GATE PASSED

4/4 in-place steps, ZMP inside the support polygon 98.7 % of loaded ticks,
peak torque 49.6 N·m against a 139 N·m limit. Getting there took four
distinct fixes; the first three were mine to make, the last was the one that
actually mattered.

#### L-M2-a: The QP's contact set must be what the ground confirms, not what the schedule intends
- **Symptom**: The robot launched itself — swing foot 0.66 m in the air,
  torques saturated, fall at 4.0 s.
- **Root cause**: The stance set came straight from the gait timeline. At the
  scheduled end of a swing the schedule declared double support while the
  landing foot was still ~60 mm up. The QP's contact constraint then asserted
  `J_c q̈ + J̇_c q̇ = 0` at a frame touching nothing and happily distributed
  wrenches through it — planning against an imaginary support polygon with
  imaginary forces.
- **Fix**: `SteppingController._effective_stance` intersects the scheduled
  stance with **measured** contact. Intent still decides which foot is *meant*
  to swing; reality decides what is load-bearing.
- **Takeaway**: Every constraint in an inverse-dynamics QP is a claim about
  the world. A claim the world does not honour is worse than no claim at all,
  because the solver treats it as free authority.

#### L-M2-b: Commanding CoM height is what saturated the actuators
*(The decisive fix — 3/4 steps → 4/4.)*
- **Symptom**: With a 3-axis CoM task the gait reached 3 steps and fell during
  the fourth transfer, with peak torque pinned at exactly 139.0 N·m.
- **Diagnosis**: Instrumenting per-joint saturation named the culprit
  immediately — and it was not a leg: **waist_roll** (50 N·m) saturated first
  and for the most ticks (417), followed by waist_pitch (290) and the 25 N·m
  shoulders. The legs were not the bottleneck; the torso was.
- **Root cause**: Holding the CoM at a constant *height* while it translates
  laterally over a stance foot forces the pelvis to stay level through the
  whole transfer. The robot's natural motion is to dip slightly; suppressing
  that dip is torque spent on nothing the gate asks for, and the waist pays it.
- **Fix**: Control the horizontal CoM only (`axes=(0, 1)`), and relax the
  pelvis orientation task (1e3/gain 50 → 1e2/gain 20). Result: **zero**
  saturated ticks, peak torque 139 → 49.6 N·m, gait completes.
- **Takeaway**: An over-specified task looks harmless in the cost function and
  shows up as saturation somewhere unrelated. When actuators saturate, ask
  which task is asking for something nobody requested — and check the whole
  body, not the obvious limb.

#### L-M2-c: Two "obvious improvements" that made it worse
Kept because negative results are results, and both are things a reader would
otherwise try:
- **Swing-foot orientation task** (hold the foot level in flight): 3 steps →
  fell at 5.7 s. Constraining the swing foot's rotation over-determines the
  swing leg, and the extra rows compete with the CoM task through the same
  hip/knee DOFs.
- **Heavier CoM damping** (2–3× critical): 3 steps → fell at 2.7 s. The weight
  transfer has a deadline set by the gait timeline; over-damping means it is
  simply not finished when the foot lifts.

#### L-M2-d: Measure the home pose on a *settled* robot
Foot and CoM homes taken at t=0 sit ~10 mm below where the robot actually
rests, so every swing reference ended buried in the floor and every touchdown
fought it. The demo now runs the M0 standing controller for 1 s first, then
reads the homes.

#### What is deliberately still simple
Timing is conservative (2.0 s double support, 0.5 s swing, 15 mm clearance) —
this is quasi-static stepping, not dynamic walking. M3 has to compress the
timeline and add a stride; the honest expectation is that the weight-transfer
approach used here (move the CoM over the stance foot, then swing) will run
out of road, and that capture-point/DCM tracking is what replaces it.

---

### M3 — Forward Walking (2026-08-15 / 16) — **GATE PASSED 4/4**

Two sessions. The first ended at **3 of 10 steps, 0.22 m of the required
1.0 m**; the second reached **12 of 12 steps and 1.18 m**. The entries below
are in the order they were found, so the first two describe the reference
generator that the DCM work then replaced — they are kept because both bugs
were real and both would have hidden inside the new controller just as well.

#### Session 1 status (superseded)
**3 of 10 steps, 0.22 m of the required 1.0 m.** Two real defects in
the reference generator were found and fixed along the way (both were making
forward walking impossible rather than merely hard); what remains is the
strategy limit predicted at the end of M2.

#### L-M3-a: Lateral and forward CoM bias are different problems
- **Symptom**: Every forward-walking attempt fell during the *first* weight
  transfer, before the second step — while the identical controller stepped in
  place indefinitely.
- **Root cause**: M2's rule "move the CoM toward the stance foot" biased both
  horizontal axes. Stepping in place that is correct, because the feet are
  side by side and the stance foot is only *sideways* from the midpoint. Once
  a stride separates the feet, the same rule also aims the CoM half a step
  **forward**, out over a diagonal support polygon.
- **Fix**: Separate ratios — `com_shift_ratio` (lateral, 0.9) and
  `com_forward_shift_ratio` (forward, default 0). Sideways the CoM still has
  to get over the stance foot; forward it tracks the foot midpoint.
- **Takeaway**: A heuristic validated in a symmetric case can hide an implicit
  assumption about geometry. "Toward the stance foot" quietly meant "sideways"
  the whole time.

#### L-M3-b: The CoM reference must flow continuously, not per phase
- **Symptom**: With the axes separated the robot got further but fell
  backwards on the second step — the CoM ended up 0.11 m *behind* its start
  while the feet had advanced.
- **Root cause**: The forward reference was derived from the current foot
  midpoint, which is constant through a whole single-support phase and jumps
  at each touchdown. So the commanded body position froze every time a foot
  was in the air, and the feet walked out from under a stationary CoM.
- **Fix**: `GaitSchedule.forward_progress(t)` — a continuous ramp across the
  whole walking interval, independent of phase. Max reference jump per 5 ms
  tick: 2.11 mm → 0.88 mm.
- **Takeaway**: Support alternates; the body does not. Any reference derived
  from "which feet are down right now" inherits the discontinuity of the
  contact schedule.

#### Where session 1 ended, and the diagnosis that turned out to be right
With both fixes the gait reached 3 steps / 0.22 m before falling (stride 0.08
got furthest). That was the limit M2's write-up predicted: the quasi-static
strategy — *shift the CoM over the stance foot, then swing* — requires the CoM
to be nearly stationary over one foot at each transfer, and forward walking
never gives it that moment. The tuning sweeps already run (stride
0.06/0.08/0.12, double support 1.5/2.0 s, forward shift 0/0.3) all failed the
same way, so the next step was recorded as capture-point / DCM tracking rather
than another round of gains. That call was correct, but the four entries below
are what actually made it work — three of the four are not about DCM at all.

---

#### L-M3-c: DCM tracking is the right reference, and it is not sufficient alone
- **What changed**: `dcm_planner.py` plans a piecewise-linear ZMP through the
  footsteps and back-integrates `ξ̇ = ω(ξ − p)` from a terminal rest condition;
  `wb_tasks.DCMTask` commands `c̈ = ω²(c − p_cmd)` with
  `p_cmd = ξ − ξ̇_ref/ω + (k/ω)(ξ − ξ_ref)`, clamped into the stance feet.
  The CoM position task is gone from the control path entirely — nothing tells
  the robot where its CoM should be, only where its divergent component is
  heading.
- **First measured result**: **worse than the old controller** — 2 of 12 steps
  and 0.32 m *backwards*, with the commanded ZMP clamped at a foot edge on
  53 % of ticks. It would have been easy to read that as "DCM does not work
  here" and start tuning k.
- **What it actually meant**: a ZMP command pinned at the foot edge half the
  time is not a gain problem, it is a statement that the controller is asking
  for authority the model says it does not have. That pointed at the contact
  model and the solver, not the control law — see L-M3-d and L-M3-e. With
  those two fixed and nothing else changed, the same DCM controller went from
  2 steps to 7, and to **12 with the stance narrowed** (L-M3-f).
- **Takeaway**: When a new controller performs worse than the one it replaces,
  read its *saturation* signals before its error signals. Error tells you it is
  failing; saturation tells you what it thinks it is not allowed to do.

#### L-M3-d: The foot contact model was a symmetric guess, and walking is where that bites
- **Symptom**: The QP planned CoM accelerations MuJoCo did not deliver —
  regressing realised against commanded lateral acceleration gave slope 0.78
  with a **−0.09 m/s² offset** and correlation 0.62. A constant acceleration
  disturbance is exactly what produces a standing DCM error (`d/(ω·k)` ≈ 8 mm),
  and it landed on the lateral axis where the whole margin is one foot width.
- **Root cause**: `ContactSpec` described the foot as a ±0.08 m box centred on
  the ankle frame, with the CoP read as `−m_y/f_z`. The real Menagerie G1 foot
  is four spheres at x ∈ {−0.05, 0.12}, y ∈ {±0.025, ±0.03}, z = −0.03 in the
  ankle-roll frame. So the model **over-claimed 30 mm of rearward CoP the foot
  does not have** — the QP wrote contact wrenches the simulator then refused to
  produce, which is precisely a constant force error — while **throwing away
  40 mm of forward CoP**, the authority that decelerates the CoM before
  touchdown. Standing in place, neither error is excited; walking uses both
  ends of the foot every step.
- **Fix**: `half_length=0.085`, `center_x=0.035`, `half_width=0.025`, plus the
  `origin_height=0.035` term the CoP needs because the wrench is expressed
  about a frame 35 mm above the ground:
  `CoP_x = (−m_y − h·f_x)/f_z`, `CoP_y = (m_x − h·f_y)/f_z`. At the shear a
  step uses (~0.3·mg) the height term alone is a 12 mm CoP error. The DCM plan
  targets the same patch centre, so plan and constraint describe one foot.
- **Result**: 2 steps → 6. M2's in-place gate *improved* at the same time —
  ZMP inside 98.7 % → **100 %**, peak torque 49.6 → 47.7 N·m.
- **Takeaway**: A contact model that a standing gate cannot distinguish from
  the truth is still wrong, and walking is the test that distinguishes it. When
  a controller's plan and the simulator's outcome disagree by a constant, the
  suspect is a constraint that lies about geometry, not a gain.

#### L-M3-e: A tighter QP tolerance made the solution worse
- **Symptom**: 38 % of control ticks returned OSQP status
  `maximum iterations reached` at the 4000-iteration cap, averaging **12.6 ms**
  per solve against a 1 ms budget — 100× the 0.11 ms M1 measured while
  standing.
- **Root cause**: `eps_abs = eps_rel = 1e-6` on a cost whose task weights span
  1e4 down to 1e1 against a 1e-4 regularisation. The tolerance was far below
  what that conditioning can deliver, so the solver spent its whole budget not
  converging.
- **Fix**: `tolerance=1e-4`, `max_iterations=2000`, both now constructor
  arguments. **Every** tick converges, in ~25 iterations and **0.073 ms** — and
  the base-dynamics constraint residual *fell* from 0.021 to 8.5e-5 N·m.
  Asking for less accuracy produced a more accurate answer.
- **Combined with L-M3-d**: commanded-vs-realised CoM acceleration went from
  slope 0.78 / offset −0.09 / correlation 0.62 to slope **0.95** / offset
  **0.04** / correlation **0.995**.
- **Takeaway**: An iteration cap that is hit is not a performance note, it is a
  correctness warning — the returned point is wherever the solver happened to
  be. And a solver tolerance is a claim about how well-conditioned your problem
  is; set it below that and you pay in iterations for a worse answer.

#### L-M3-f: Stance width is the dominant gait parameter, not stride length
- **Symptom**: With the contact model and solver fixed, the gait reached 7 of
  12 steps and still fell, ZMP command clamped 40 % of ticks.
- **Root cause**: The gait kept both feet on their rest lines — the G1 stands
  0.237 m wide. The ZMP must cross from one foot to the other every step, and
  the lateral DCM swings with the same amplitude, so a wide stance demands a
  large lateral excursion be arrested inside a 50 mm foot width. Stride length
  costs nothing by comparison: it is arrested by the *long* axis of the foot.
- **Fix**: `GaitSchedule(step_width=...)` places each landing beside the stance
  foot at a chosen separation; 0.18 m for the gate.

  | stance width | steps | distance | DCM RMS | ZMP clamped | peak τ |
  |---|---|---|---|---|---|
  | 0.237 m (rest) | 7/12, fell | 0.84 m | 121.5 mm | 40 % | 139 N·m |
  | 0.18 m | **12/12** | **1.18 m** | **6.2 mm** | 3 % | 56.0 N·m |
  | 0.14 m | 12/12 | 1.19 m | 17.5 mm | 9 % | 59.9 N·m |

- **Takeaway**: In lateral balance the cost is set by how far the ZMP has to
  travel sideways each step, and that is stance width. 0.14 m works too but
  tracks worse — the feet start crowding each other's swing.

#### L-M3-g: The integral term I added to fix the bias became harmful once the bias was real-fixed
- **What happened**: A leaky integrator on the DCM error was added to reject
  the −0.09 m/s² acceleration bias of L-M3-d. It is the textbook remedy and it
  would have worked — as a mask. After the contact model and solver were fixed
  the bias was gone, and the same integrator turned a passing gait into a
  falling one: 12/12 → 8/12 at width 0.18, 12/12 → 10/12 at 0.14, with DCM RMS
  6.2 mm → 118 mm.
- **Fix**: `integral_gain` defaults to 0. The code stays, documented, because
  it is the right tool against a disturbance you genuinely cannot remove.
- **Takeaway**: An integrator is a way of not knowing what your error is. It
  bought a real improvement while the cause was unknown, and became pure phase
  lag the moment the cause was fixed. Fix the cause, then re-measure whether
  the compensator is still earning its place.

#### L-M3-h: I "fixed" the initial DCM lead, and the fix made the gait worse
- **The objection**: `xi_initial` came out 30 mm to one side of the foot
  midpoint. A DCM tracking a linearly ramping ZMP leads it by `k/ω` in steady
  state, and sweeping the ZMP from the midpoint onto the first stance foot
  across the whole 1.5 s settle makes `k/ω` ≈ 30 mm. So the plan appeared to
  ask a robot standing perfectly still at t=0 to already have its capture point
  off-centre — a textbook initial-condition mismatch.
- **The fix**: split the settle into a hold at the midpoint plus a short sweep,
  so the lead decays as `e^{−ω·hold}`. Clean, well-motivated, and it did
  exactly what it claimed: the initial lead dropped to sub-millimetre.
- **The measurement**: it turned a **12/12** gait into **6/12**. Sweeping the
  parameter afterwards:

  | settle_sweep | steps | distance | DCM RMS | ZMP clamped |
  |---|---|---|---|---|
  | 0.3 | 6/12, fell | 0.77 m | 138.2 mm | 32 % |
  | 0.5 | 8/12, fell | 0.89 m | 124.4 mm | 20 % |
  | 0.7 | 6/12, fell | 0.61 m | 159.6 mm | 27 % |
  | **1.0 (no hold)** | **12/12** | **1.18 m** | **6.2 mm** | 3 % |

- **Why the objection was wrong**: the lead is not an error, it is the lateral
  momentum the first step needs, and 1.5 s of gentle ramp is the robot
  acquiring it. Holding still and then sweeping the ZMP across in half the time
  enters the first transfer cold and twice as fast. `settle_sweep` defaults to
  1.0 and the split path is kept only so the reasoning can be re-run.
- **How it was caught**: the gate run disagreed with the tuning sweep that had
  passed an hour earlier, and the only difference between them was this change.
  Re-running the gate rather than trusting the sweep is what surfaced it.
- **Takeaway**: "the plan's initial condition doesn't match the robot's state"
  is a real class of bug — but a *dynamic* reference is allowed to want the
  robot moving, and a settle is the interval for it to start. Derive whatever
  you like; the gate decides.

#### M3 gate

| criterion | result | measured |
|---|---|---|
| ≥10 steps without falling | PASS | **12/12** |
| Travelled ≥ 1.0 m | PASS | **1.18 m** |
| ZMP inside support > 90 % | PASS | 99.3 % |
| Torques within limits | PASS | 56.0 N·m (limit 139) |

Regression: M2's 4-step in-place gate re-run through the DCM controller, and
M0/M1 re-run after the QP changes — all still pass.

#### What M3 kept from session 1
Forward footstep placement (the swing foot lands ahead of the *stance* foot,
so the body advances a full stride per step) and per-phase foot bookkeeping.
`forward_progress` and the split CoM shift ratios are no longer on the control
path — the DCM plan supersedes them — but they remain in `gait_planner` for the
CoM-task path M2 still uses.


---

### M4 — Walk + Arm Task (2026-08-16) — **GATE PASSED on carry; reach deferred**

Status: the **carry** task passes every gate criterion — 12/12 steps, 1.170 m,
ZMP 99.1 % inside, hand error **14.5 mm RMS / 25.7 mm max** against a 50 mm
gate, 55.2 N·m peak. The **reach** task (right hand tracing a circle while
walking) does not: best run is 6 of 12 steps. M4 is not closed until both pass.

#### L-M4-a: A gait plan must be built on the posture the robot will walk in
- **Symptom**: Adding a two-handed carry pose to M3's working gait made the
  robot fall on the first step, every time, at every task weight.
- **Root cause**: two separate errors stacked.
  1. The arms were commanded into the carry pose *after* the DCM plan was
     built. Bringing both arms forward moves the CoM ~85 mm ahead of where it
     rests, so the plan — whose ZMP targets the foot patch centres — spent the
     whole walk asking the robot to pull its CoM back to a place its own
     posture forbade. A 50 mm standing DCM error before the first step.
  2. The obvious fix, settling into the carry pose under the **standing
     controller** first, is worse: joint PD holds joint *angles*, so with the
     arms out front the robot simply leans, and over 6 s it drifts and yaws off
     its own footprint (CoM y 0.027 → 0.24 m).
- **Fix**: reach the pose under the **whole-body QP**, with the DCM reference
  frozen at the measured capture point — the QP then bends the hips to keep the
  CoM planted while the arms travel (CoM moves 2.7 → 29 mm, not 85 mm). Then
  rebuild the schedule and DCM plan from the *current* state
  (`m3_walking.make_plan`, split out of `build` for exactly this). Initial DCM
  error after both: 20.8 mm, and all of that is the lateral k/ω lead M3 wants.
- **Takeaway**: a plan is a statement about a specific robot configuration. If
  anything changes the configuration between planning and execution, replan —
  and reach the configuration with the controller that understands balance, not
  the one that understands joint angles.

#### L-M4-b: A steady offset with no variance is two tasks fighting — but the fight may be load-bearing
- **Symptom**: At a hand weight the robot could walk with (1e1), tracking error
  was 46 mm RMS — and per-axis it was almost pure bias: x mean −39.7 mm, z mean
  −16.5 mm, lateral 4.4 mm, variance near zero.
- **Diagnosis**: the posture task pulls the arms toward `Q_STAND_JOINTS` (arms
  down) at weight 10 while the hand task holds them forward at weight 10. The
  QP splits the difference. That reads as an obvious bug in the stack.
- **What happened when I fixed it**: re-nominalising the posture task on the
  achieved pose took the walk from 12/12 steps to **2/12**. Restricting the
  re-nominalisation to the *arm* joints only — the legs must keep their nominal,
  since the posture task is where the gait's redundancy is resolved — still gave
  **6/12**.
- **Why**: the pull toward rest was the arms' only damping. Undamped arms are a
  centroidal disturbance, and the balance controller was paying for the droop
  in exchange for a stabiliser nobody had named as one.
- **Takeaway**: before removing a contradiction between two tasks, find out what
  the losing side was buying. Here the correct answer was not to remove the
  fight but to give the QP a *proper* damper — see L-M4-c.

#### L-M4-c: Centroidal angular momentum is what lets a walking robot use its hands
- **The wall**: every lever that made the hand task stronger made the walk
  worse, non-monotonically. Hand weight 1e1 → walks, 46 mm droop; 2e1 → falls at
  5 steps; 1e2 → falls at 7; 3e2 → falls at 5. Hand gain 400 → walks; 1000 →
  falls at 3. Carry offset 0.20 m → walks; 0.10 m → falls at 4. When every
  direction is downhill and the ordering is not even monotonic, the stack is
  missing a term, not a setting.
- **The missing term**: `wb_tasks.CentroidalAngularMomentumTask` — regulate
  `L = A_g(q) q̇` (angular block) toward zero. Without it the arms' only
  restraint is a joint-space posture pull that has to double as a momentum
  damper (L-M4-b); with it, the QP gets an explicit, cheap way to say "the arms
  may move, but they may not spin the body", and the hand task stops being
  traded against balance through a proxy.
- **Measured** (carry task, everything else identical):

  | momentum weight | hand weight | steps | hand RMS | hand max | DCM RMS |
  |---|---|---|---|---|---|
  | 0 | 1e1 | 12/12 | 46.0 mm | 56.5 mm | 5.6 mm |
  | 0 | 1e2 | 7/12, fell | 61.6 mm | 429.5 mm | 133.8 mm |
  | **1e1** | **1e2** | **12/12** | **14.5 mm** | **25.7 mm** | 13.9 mm |
  | 1e2 | 1e1 | 5/12, fell | 119.9 mm | 616.1 mm | 129.7 mm |
  | 1e2 | 1e2 | 8/12, fell | 46.5 mm | 362.4 mm | 120.8 mm |

  The same hand task that fell on step 7 walks all twelve and tracks three
  times better. Too much momentum damping is as bad as none — it forbids the
  arm motion the task needs.
- **Takeaway**: on a floating-base robot, a manipulation task and a balance
  task are not independent objectives that need re-weighting; they are coupled
  through momentum, and the coupling deserves its own term in the QP. The
  brief said so ("regulate centroidal momentum while performing arm tasks") and
  it took a wall of failed tuning to believe it.

#### Where reach stands (open)
The circle task still falls, and the two things tried have not fixed it:
* **Speed is not the problem.** Periods 2.0/3.0/4.0 s and radii 0.08/0.10 m all
  fall at 3–4 steps.
* **Plane matters, but not enough.** The circle was originally in the lateral
  (y–z) plane, which spends the axis M3 showed has nothing to spare (the foot
  is 170 mm long and 50 mm wide, and stance width alone decided M3's gate).
  Moving it to the sagittal (x–z) plane improved the best run from 4 steps to
  6, and it is the right plane on principle, but it does not close the gap.

Next to try, in order: (1) the momentum task's gain and weight were tuned for
the *static* carry pose — a moving hand deliberately generates angular
momentum, so the momentum reference should be the plan's own `L_ref` from the
commanded arm motion rather than zero; (2) fade the circle in only during
double support, so the arm's peak demand does not land in single support;
(3) treat the reach task as M5 scope if a momentum *reference* turns out to be
the honest prerequisite.


#### L-M4-d: The momentum reference and the free arm — measured, both necessary, and the pocket is narrow
- **Diagnosis that led here** (per-joint saturation + momentum trace on the
  failing reach run): the saturated actuators were the 25 N·m **shoulders**
  (right roll 143 ticks, left pitch 129), not the legs. The gait's own natural
  roll momentum oscillates at ±2 kg·m²/s — three times what the hand circle
  generates — so `L → 0` was asking the QP to cancel *walking itself*, and the
  cheapest joints for that are the arms. The left arm was simultaneously
  holding a Cartesian pose at weight 1e2: ground between "hold still" and
  "cancel momentum", it saturated, and the DCM error compounded step over step
  (10 → 60 → 250 mm, fall at step 6).
- **Fix 1 — momentum reference (resolved momentum control, Kajita IROS 2003)**:
  `CentroidalAngularMomentumTask.set_reference(L_ref)` with `L_ref` computed
  each tick from the commanded circle velocity through the arm block of the
  hand Jacobian and `A_g`. The task now damps momentum *deviation from plan*
  instead of fighting the trajectory another task feeds forward.
- **Fix 2 — free left arm in reach**: the left hand's Cartesian task is dropped
  (PLAN's sub-task (b) never asked for it); the left arm is held only by
  posture, making it the momentum task's actuation — the human arm-swing
  arrangement.
- **Ablation** (period 2.0, radius 0.10, hand weight 1e2):

  | configuration | steps | outcome |
  |---|---|---|
  | both fixes | **12/12, 1.178 m, no fall** | hand 37.6 mm RMS / 91.6 mm max |
  | free arm, no L_ref | 11/12, fell | L_ref matters at the margin |
  | L_ref, left locked | 5/12, fell | the free arm is the larger effect |
  | both, hand weight 3e2 | 5/12, fell | strengthening the hand still breaks balance |

- **The open finding — the operating point is a narrow pocket.** Every
  neighbour of (period 2.0 s, radius 0.10 m) falls: periods 1.8, 2.5, 3.0 and
  radius 0.08 all fail, including *smaller* motions. Period 2.0 sits near the
  gait cycle (2 × 0.9 s = 1.8 s), so the surviving case is close to
  gait-synchronous — but exact lock (1.8) is worse, so this is not a clean
  resonance story yet. The 91.6 mm error peak is a single event in one double
  support (t = 5.92 s), i.e. contact-switch turbulence, not tracking lag.
- **Takeaway**: the two structural fixes turned "falls at step 4" into "walks
  the full gate distance", and the ablation pins the credit. But a pass that
  sits alone in a pocket of falls is not yet an engineering result — the
  remaining work is to understand the pocket (why neighbours fail) and to
  soften the contact-switch transient that owns the error peak, not to tune
  within the pocket.


#### L-M4-e: Three plausible fixes for the transient, all measured worse
The reach task's error peaked at 91.6 mm in one double support (−71 mm of it
vertical), so the transient looked like the thing to fix. Three attempts, all
reverted:
* **Contact load ramp** — cap `f_z` on a freshly landed foot so weight
  transfers over ~0.1 s instead of one tick (`ContactSpec.max_normal_force`,
  ramped by the controller). At 0.12 s/120 N it made the peak **worse**
  (418.6 mm): the cap occupied half the 0.25 s transfer window and became a
  schedule fighting the DCM plan's. Retuned to 0.06 s/200 N it was worse still
  (359 mm, fall at step 4).
* **QP warm start across contact switches** — seed the rebuilt solver with the
  previous q̈ and the wrenches of feet that stayed down, instead of starting
  from x = 0. No improvement; reverted with the ramp.
* **Weak CoM-height spring** (weight 30–100, axes=(2,)) to damp the pelvis dip
  the hand was chasing. Weight 30 changed nothing (97.1 mm peak); weight 100
  fell at step 9 — consistent with L-M2-b, just at lower weight.

**Takeaway**: the transient was a symptom. Chasing the largest number in the
log, without first checking whether it was cause or consequence, cost three
implementations. The `-71 mm` z-spike was the *hand following a robot that was
already losing balance*, which the next entry establishes.

#### L-M4-f: The reach task's real obstacle is the asymmetric upper body, and the pass was a lottery
- **The control that settled it**: run reach with `REACH_RADIUS = 0` — the
  circle removed entirely, leaving only "right hand held, left arm free". It
  **still fell** (8/12 steps), with the same +286 mm lateral error signature as
  every other failure. The circle was never the cause.
- **The pass was luck**: the 12/12 reach result holds only at exactly
  (period 2.0 s, radius 0.10 m, phase 0). Shifting the circle's starting phase
  — a change that alters nothing about the task's difficulty — gives 9/12 at
  0.3 rad and 3/12 at 1.0 rad. A result that a phase offset can destroy is a
  draw from a distribution, not a controller property. Carry, by contrast, is
  flat across perturbations: 12/12 at nominal, at `step_length` 0.09 and at
  `t_double` 0.30, with hand error 15.2–15.4 mm RMS every time.
- **What the failures have in common**: every one diverges laterally
  (+300 mm y). Carry is the only configuration where the upper body is
  *symmetric and rigid* — both hands pinned, arms contributing near-zero
  variable momentum. Any departure from that (one arm free, or one hand moving)
  injects upper-body motion into the axis with the least margin. M3 already
  established lateral balance as the binding constraint (L-M3-f).
- **Ruled out, each measured**: hand weight (5–300) · hand gain (400–2000) ·
  momentum weight (0–100) · momentum gain (5–10) · momentum reference on/off ·
  per-axis momentum weighting (roll ×4, ×10) · circle period (1.8–4.0 s) ·
  radius (0–0.10 m) · plane (lateral vs sagittal) · phase (0–4.7 rad) · stance
  width (0.14/0.16/0.18) · double support (0.25–0.40 s) · DCM gain (3–4) ·
  CoM-z spring · contact load ramp · QP warm start. No parameter is monotone;
  neighbouring settings of the one passing configuration all fall.
- **Scope call**: `tasks/PLAN.md` lists (a) carry and (b) reach under *Steps*,
  and its **Gate** line is singular — "walking gate still passes AND hand error
  < 50 mm during walk" — while `plan/LAB_08.md`'s success criterion is "G1
  walks while maintaining arm pose (carrying behavior)". Carry meets both,
  robustly. Reach was an extra step this lab set itself; it is reported in
  every gate run as **exploratory**, and it did not reach the bar. Deferred to
  M5 with the momentum machinery it needs already built and tested.
- **Takeaway**: a single passing run inside a neighbourhood of failures is not
  a result. Perturb the thing that *should not matter* — here, the circle's
  starting phase — before believing a pass. That test is what turned a
  celebration into an honest deferral.


---

### M5 — Loco-Manipulation Capstone (2026-08-17) — **GATE PASSED 4/4**

WALK → STOP → REACH → GRASP → LIFT → TUCK → WALK-CARRY → STOP → PLACE →
RELEASE, no fall, payload placed **11.8 mm** from target, 53.7 N·m peak.

Ten defects were found and fixed getting there, and the striking thing is how
few of them were control problems: three were in code M1–M4 had already
"validated", two were scene geometry the controller could not see, and two were
the difference between commanding a hand and commanding the object in it.

#### L-M5-a: The momentum task is an arm-task companion, not a global stabiliser
- **Symptom**: the capstone's *first* walk — plain M3, no payload, no hand
  tasks — fell on the second step. M3's own gate walks twelve.
- **Root cause**: the capstone stack adds `CentroidalAngularMomentumTask` for
  the later carry phase, and it was left enabled throughout. Commanding
  `L → 0` across a bare walk asks the QP to cancel the angular momentum
  **walking itself generates**; the gait's roll momentum runs at ±2 kg·m²/s.
  DCM error went 12.5 mm → 226.7 mm.
- **Fix**: the momentum task is enabled only alongside an arm task. Each M5
  phase is then exactly a configuration an earlier milestone validated —
  approach walk = M3, standing reach = M1, carry walk = M4's carry.
- **Takeaway**: M4 introduced this term *with* a hand task and never ran it
  without one. A term validated inside one configuration is validated for that
  configuration, not adopted into the stack.

#### L-M5-b: A MuJoCo weld snaps to its compile-time pose, not the pose you close it at
- **Symptom**: the instant the grasp activated, the payload leapt 0.115 m and
  took the robot down.
- **Root cause**: `mjEQ_WELD` holds body2 at the `relpose` stored in
  `model.eq_data`, and that field is baked at **compile time** — from the rest
  pose, where the hand is at x = −0.02 and the payload is at x = 0.40.
  Activating the constraint therefore commanded a 0.42 m snap. `eq_active` is
  a switch, not a "grasp here" instruction.
- **Fix**: `CapstoneScene._capture_relative_pose` writes the live hand→payload
  transform into `eq_data` before flipping `eq_active`. Plus a refusal: if the
  hand is not actually within `GRASP_TOLERANCE_MM` of the grasp point, the
  sequence raises rather than welding across a gap.
- **Also**: the hand cannot reach the payload's *centre* — it is a solid box,
  and commanding the wrist there just presses geoms together (93.7 mm residual
  error). The grasp point is offset onto the robot's side of the box.

#### L-M5-c: Lift and carry are different poses, and the transition needs its own phase
- Walking straight from the lift left the load out where the pedestal was —
  the right hand at y = −0.275 and the left at rest, which is precisely the
  asymmetric upper body M4 measured as marginal (L-M4-f). A **TUCK** phase now
  brings both hands into M4's symmetric carry pose before the gait starts.
  Making that pose genuinely symmetric (±0.19 rather than −0.16/+0.20) bought
  another 0.11 m of transport.

#### L-M5-d: Payload mass was not the limit; the brief's sizing was right anyway
- The first attempt used a 1.5 kg / 90 mm block. It stands fine and falls on
  the carry leg every time — but so does 0.5 kg. Mass was not the driver, so
  the reduction is kept only because `plan/LAB_08.md` asked for a "40 mm
  cube-class object" and 1.5 kg was scope I invented.

#### L-M5-e: Two gait-planner defects that only a *resumed* walk can expose
Both were latent in M3 and M4, which each walk exactly once:
- **The gait always swung the left foot first.** After an odd number of steps
  the left foot is the *leading* one, so its landing — computed as
  `stance_x + step_length` — is where it already stands. The first step of any
  resumed walk was a zero-length step while the DCM plan expected a full
  stride. `GaitSchedule(first_swing=...)`, and `make_plan` picks the trailing
  foot; from a level stance this reduces to M3's original gait exactly.
- **A walk ended mid-stride.** After twelve steps the feet sit 0.09 m apart in
  x, so the next walk starts from a staggered stance no milestone validated.
  `GaitSchedule(close_stance=True)` appends a step that brings the trailing
  foot level, the way a real walk ends.

#### L-M5-f: The scene is part of the controller's world
- **Symptom**: the *identical* M3 controller walks 12 steps on the bare model
  and fell on step 4 of the capstone scene, at x ≈ 0.41.
- **Diagnosis**: not a control problem at all. Logging every contact involving
  a scene prop named the culprit in one line —
  `pick_pedestal ↔ right_hip_roll_link` at t = 3.47 s, then the ankle, then the
  fall. The pedestal's inner face sat at y = −0.22, exactly where the hip
  passes; its top at 0.72 m was also level with the walking wrists.
- **Fix**: pedestals moved to y = −0.45 and lowered to 0.55 m, with the payload
  set on the prop's *inner* edge so the reach stays short. The walk is then
  clean: 12 steps, and the only prop contact in the whole run is the payload
  resting on its pedestal.
- **Takeaway**: when a controller that is known-good regresses in a new scene,
  read the contact list before touching a gain. Two hours of balance-tuning
  hypotheses were displaced by four lines of `mj_data.contact` output.

#### Where it stands, and the honest diagnosis
Everything up to and including the carry is solid; the transport leg is not.
The failure is the same one M4 deferred (L-M4-f): an **asymmetric upper body**
is marginal on this robot while walking, and a payload held on one arm is an
asymmetric upper body by definition. Symmetry helps measurably (0.54 → 0.64 m
of transport) but does not close it, because the *mass* stays on one side even
when the hands are level.

The next implementation step is therefore structural, not a gain: **carry the
payload with both hands** — a second weld to the left wrist, with the reach
placing both hands on opposing faces of the box. That makes the load itself
symmetric rather than merely the arms holding it, which is the only version of
this the controller has ever been shown to survive. Trimming the transport leg
(6 → 4 steps) and tuning weights were both tried and neither is the answer.


#### L-M5-g: Symmetric *arms* are not a symmetric *load*
- Carrying on one arm — even with both arms held in mirror-image poses — walks
  0.64 m and falls. What the balance controller responds to is where the mass
  is, not how tidy the arms look.
- **Fix**: a second weld. The right hand picks from the pedestal (the only hand
  that can reach it), TUCK brings the load to the chest mid-line, and the left
  hand joins it there — where it *can* reach. Carry distance went 0.64 → 0.95 m
  on that change alone, and the fall moved from mid-carry to the walk's final
  settling.
- The carry pose is now derived from where the **payload** should ride
  (`CARRY_PAYLOAD_LOCAL`), with both hand targets computed from it and mirrored
  about the load. Hands follow the mass, not the other way round.

#### L-M5-h: Two welds make a closed chain, and a closed chain cannot place
- With both wrists welded, the arms and payload form a closed kinematic loop
  the QP does not model. Commanding only the right hand to the place target
  left the left arm — task disabled but still rigidly attached — dragging
  against the motion: the payload reached halfway and was released in mid-air,
  583 mm out.
- **Fix**: open the left weld *before* the place. Pick one-handed, carry
  two-handed, place one-handed. Each phase uses the grip the phase needs.

#### L-M5-i: Command the object, not the hand — and buy precision only where it is measured
Two separate errors, both landing on the one number the gate reads:
- **Stale offset.** The hand target was computed from a hand→payload offset
  measured once before the motion. The weld is deliberately compliant, so the
  load settles in the grip over a 25 s sequence and that shift became a
  systematic 55 mm outboard placement error. Re-measuring the offset live cut
  it to 65 mm; **servoing the payload directly** — recomputing the hand target
  every tick from the live offset toward a blended payload waypoint — took the
  release to **18.9 mm**. The task is about the object, so close the loop
  around the object.
- **One weight does not fit two regimes.** M4's walking-safe hand weight (1e2)
  was being used for the stationary reaches too, where M1 had validated 1e3 and
  tracked to 7.08 mm; that alone was a ~60 mm droop on every standing reach.
  But raising it everywhere destabilised the tuck (fall at t=13.9 s) — the tuck
  moves both arms *and* the load. The high weight is now used only in the place
  phase, the one phase whose accuracy is measured.

#### L-M5-j: A placement is only as good as the shelf it lands on
- At 18.9 mm release accuracy the payload still ended on the floor: the place
  target sat 0.09 m in from a pedestal whose half-extent is 0.10 m, so a 30 mm
  box overhung the inner edge by 20 mm, tipped, and slid off.
- Widening the pedestal instead put it inside the robot's own standing space
  and it fell with the torque saturated — the same collision class as L-M5-f.
- **Fix**: move the *target*, not the furniture. The place inset is its own
  parameter (0.05 m), safely inboard of the edge. Final placement **11.8 mm**.
- **Takeaway**: the last 20 mm of a pick-and-place is a statics problem about
  the support, not a controls problem about the arm. Check that the goal pose
  is one the object can actually rest in.

#### M5 gate

| criterion | result | measured |
|---|---|---|
| Full sequence, no fall | PASS | 10 phases |
| Payload within 50 mm of target | PASS | **11.8 mm** |
| Payload actually transported | PASS | 0.384 m |
| Torques within limits | PASS | 53.7 N·m (limit 139) |

Post-condition asserts run on the **simulated** payload pose, not on the
commanded one (Lab 5's lesson): within tolerance of the target *and* at least
0.30 m from where it started.
