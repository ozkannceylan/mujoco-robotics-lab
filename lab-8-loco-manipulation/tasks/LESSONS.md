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
