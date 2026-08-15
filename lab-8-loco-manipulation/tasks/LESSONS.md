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
