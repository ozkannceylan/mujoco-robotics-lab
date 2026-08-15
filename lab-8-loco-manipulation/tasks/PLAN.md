# Lab 8 — Whole-Body Loco-Manipulation: Milestone Plan

> Created: 2026-08-14 · Brief: `plan/LAB_08.md` · Platform: Unitree G1 (Menagerie) on MuJoCo + Pinocchio
> Capstone: G1 walks to a table, picks up an object, carries it while walking.

## The One Big Deviation from the Brief (read first)

`plan/LAB_08.md` Phase 2 says *"combine Lab 7's gait generator with the whole-body QP"*.
**Lab 7 has no working gait generator.** Its M4 ZMP walking is formally BLOCKED: the
Menagerie G1's position actuators cannot track dynamic references (M3e failed 6
attempts — IK converges, PD replay fails; see `lab-7-locomotion/README.md` Scope
Deferral). The unblock path Lab 7 itself identified — torque-level control through
inverse dynamics — is exactly what this lab's architecture prescribes (QP → RNEA →
torques). Therefore **Lab 8 owns gait generation as a first-class deliverable**
(milestones M2–M3 below), instead of importing one. Lab 7's LIPM planner and
floating-base Pinocchio stack ARE reusable and will be imported.

This also makes Lab 8 the critical path for Lab 9: the VLA's demonstration data
comes from Labs 3–8 controllers.

## Ground Rules (inherited)

- ONE milestone per session. Gate + evidence (video/plot/table in `media/mN_*`) before the next.
- All foot/CoM control through Pinocchio; `pin.LOCAL_WORLD_ALIGNED` Jacobians;
  `pin.integrate()` for configuration updates (nq ≠ nv); finite-difference-validate every
  new Jacobian usage.
- Don't implement QP from scratch — OSQP (verified available: osqp 1.1.3).
- Strict task priorities (balance is never traded for manipulation), implemented as
  hierarchy-by-constraint or lexicographic weights with a documented gap (≥1e3).

## Milestones

### M0 — Torque-Actuated G1 Bring-Up  ← the enabler everything else depends on
The Menagerie `g1.xml` ships position actuators (`<position kp=...>`). Lab 8 needs
torque command authority.
- Steps:
  - 0.1 Build `models/g1_torque.xml`: a thin overlay/patched MJCF that replaces the 29
        position actuators with `<motor>` torque actuators (same joint order), reusing
        Menagerie meshes from `third_party/mujoco_menagerie/unitree_g1/`. Document
        torque limits per joint (from Menagerie actuator forcerange / Unitree specs).
  - 0.2 `src/lab8_common.py`: paths, constants, model loaders (MuJoCo torque scene +
        floating-base Pinocchio from MJCF), joint-map table, quaternion helpers —
        reuse/import `lab7_common` where possible rather than duplicating.
  - 0.3 Gravity-compensated standing under pure torque control: τ = RNEA(q, 0, 0)
        with both feet on the ground. This is the "hello world" that position servos
        made trivial and torque control makes honest.
- Gate: G1 stands 10 s under torque-only gravity compensation; CoM drift < 30 mm;
  no falls. Cross-validation table g_pin vs qfrc_bias (< 1e-6 relative).
- Evidence: `media/m0_torque_standing.mp4` + printed gate table.

### M1 — Whole-Body QP, Standing
- Steps:
  - 1.1 `src/wb_tasks.py`: task residuals + Jacobians — CoM (2D/3D), foot pose (6D
        per foot), hand position (3D), posture (nv-dim, low priority).
  - 1.2 `src/wb_qp.py`: velocity-level QP in OSQP —
        min ‖J q̇ − ẋ_d‖² over prioritized stack; constraints: joint velocity/position
        limits, (later) contact.  Start with weighted lexicographic (weights 1e6 /
        1e4 / 1e2 / 1), document the choice; revisit strict HQP only if weights fail.
  - 1.3 `src/inverse_dynamics.py`: q̇_des → q_des (pin.integrate) → τ via RNEA with
        contact-consistent terms + joint-space PD on the tracking error
        (τ = RNEA(q, q̇_d, q̈_d) + Kp·e + Kd·ė).
        **Amended after M0**: gains stay *raw*. The original plan said to
        inertia-shape them like Lab 5; M0 measured that doing so makes the
        floating-base G1 fall at every gain setting (LESSONS L-M0-b).
  - 1.4 Standing reach demo: right hand tracks a 20 cm box trajectory while both feet
        stay planted and CoM stays inside the support polygon.
- Gate: hand tracking RMS < 20 mm over the trajectory; CoM stays ≥ 20 mm inside
  support polygon edge; feet do not move (< 5 mm); no fall.
- Evidence: `media/m1_standing_reach.mp4` + tracking/CoM plots.

### M2 — Torque-Level Stepping (own the gait, part 1)
- Steps:
  - 2.1 Import/adapt Lab 7's LIPM planner for CoM + footstep references (it is
        validated at the planning level — 18 tests).
  - 2.2 Contact schedule + swing-foot trajectory task in the QP (support-foot
        contact as a hard constraint / frozen-foot task at top priority).
  - 2.3 Weight shift → single step → step-in-place cycle, all torque-level.
- Gate: 4 consecutive in-place steps; no fall; ZMP (from MuJoCo contact forces)
  stays inside the support polygon > 95% of stance time.
- Evidence: `media/m2_stepping.mp4` + ZMP plot.

### M3 — Forward Walking (own the gait, part 2 — retires Lab 7's deferred capstone)
- Steps: forward LIPM references, 0.10–0.15 m strides; disturbance-free flat ground;
  tune QP weights/PD gains; instrument falls with the Lab 7 debugging checklist.
- Gate: ≥ 10 consecutive forward steps, ≥ 1.0 m traveled, no fall, arms in nominal pose.
- Evidence: `media/m3_walking.mp4` + stride/ZMP plots.
- Note: this gate deliberately equals Lab 7's abandoned "10+ steps" capstone — the
  point is to demonstrate the actuator-model diagnosis was correct.

### M4 — Walk + Arm Task
- Steps: (a) walk with both arms holding a fixed Cartesian pose (carry posture);
  (b) walk while the right hand tracks a moving target.
- Gate: walking gate (M3) still passes AND hand error < 50 mm (brief's 5 cm) during walk.
- Evidence: `media/m4_walk_reach.mp4` + hand-error plot.

### M5 — Loco-Manipulation Capstone
- Steps: scene with table + object (reuse Lab 5 sizing: 40 mm cube-class object or a
  handled payload attached via weld on contact — grasp stays SIMPLE per the brief);
  sequence state machine: WALK → STOP → REACH → GRASP (weld) → LIFT → WALK-CARRY →
  STOP → PLACE; payload mass folded into the QP's CoM/momentum model at grasp time.
- Gate: full sequence completes without a fall; object placed within 50 mm of target;
  post-condition assert on object pose (Lab 5's lesson: DONE must verify the object moved).
- Evidence: `media/m5_capstone.mp4` + sequence plots + gate table.

### M6 — Documentation & Blog
- `docs/` + `docs-turkish/` (ARCHITECTURE + CODE_WALKTHROUGH pattern of Labs 6–7),
  README with milestone-gated evidence tables, blog post (write it — Labs 3–4 taught
  us the criterion silently rots otherwise), update root README/MASTER_PLAN/status board.

## Risks

| Risk | Mitigation |
|---|---|
| ~~Torque-model G1 unstable at 1 kHz (small-inertia joints chatter, cf. Lab 5 L-6.1b)~~ | **Retired at M0**: no chatter observed at 1 kHz with raw gains (Kp 500 / Kd 50, \|τ\|max 3.9 N·m). The mitigation originally proposed here — inertia-shaped PD — turned out to be the *cause* of instability on a floating base (L-M0-b). |
| QP infeasible during contact switches | Slack on lower-priority tasks; contact schedule hysteresis; log infeasibilities, never silently clamp |
| Weighted hierarchy leaks balance error | Weight gap ≥ 1e3 between levels; monitor per-task residuals; escalate to strict HQP only on evidence |
| Grasp sophistication creep | Weld-on-contact grasp; Lab 5 owns real grasping — out of scope here |
| Sim speed (29 DOF QP at 1 kHz) | QP at 100–200 Hz with torque interpolation; profile before optimizing |

## Definition of Done

All six milestone gates passed with committed evidence; success criteria in
`plan/LAB_08.md` mapped: QP stability → M1, standing reach → M1, walk+carry → M4,
capstone → M5, LAB_08 docs + blog → M6. Update `plan/LAB_08.md` status header on completion.
