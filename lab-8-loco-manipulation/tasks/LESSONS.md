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

### M4 — Walk + Arm Task (2026-08-16) — **IN PROGRESS: carry PASSES 5/5, reach does not**

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
