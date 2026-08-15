# Lab 8 — Architecture

> Written before code (per workflow). Updated as modules land; if this drifts from
> reality, fix it in the same commit that causes the drift.

## Module Map

```
lab-8-loco-manipulation/
├── models/                      # (empty at M0 — see "Model Files" below;
│                                #  the torque G1 is built programmatically)
├── src/
│   ├── lab8_common.py           # paths, constants, loaders, joint map, quat helpers
│   ├── g1_torque_model.py       # M0: MjSpec builder — position servos → <motor>
│   ├── standing_controller.py   # M0: joint PD + selectable gravity mode
│   ├── wb_tasks.py              # M1: task residuals, Jacobians, J̇q̇ drift, feedforward
│   ├── wb_id_qp.py              # M1: OSQP acceleration-level ID QP  ← the control path
│   ├── wb_qp.py                 # M1: velocity-level QP — kinematic sub-problems ONLY
│   ├── inverse_dynamics.py      # M0-style velocity tracker (off the M1 control path)
│   ├── gait_planner.py          # M2: LIPM refs + contact schedule (adapts Lab 7 lipm_planner)
│   ├── locomotion_controller.py # M2–M3: stepping/walking loop tying gait → QP → τ
│   ├── loco_manip_fsm.py        # M5: WALK→STOP→REACH→GRASP→CARRY→PLACE sequencer
│   └── mN_*.py                  # one runnable demo per milestone, writes media/mN_*
├── tests/                       # per-milestone pytest (task Jacobians vs finite diff,
│                                #   QP feasibility, RNEA cross-validation, gait refs)
└── media/                       # gate evidence only — no orphans (Lab 7 lesson)
```

## Data Flow (one control tick)

```
sensors (mj_data: qpos, qvel)
   │
   ▼
state: mj_state_to_pin  (pelvis z-offset, quaternion order, base twist world→body)
   │
   ▼
references: gait_planner (CoM/footsteps, contact schedule)   hand target (fixed or FSM)
   │                                          │
   ▼                                          ▼
wb_tasks — e_i, J_i, J̇_i q̇, and feedforward ẋ_ref/ẍ_ref  (LOCAL_WORLD_ALIGNED)
   │
   ▼
wb_id_qp (OSQP), variables [q̈ (35); contact wrenches f (6 per stance foot)]:
     min Σ w_i ‖J_i q̈ + J̇_i q̇ − ẍ_i‖² + λ_a‖q̈‖² + λ_f‖f‖²
     s.t.  M[:6] q̈ + h[:6] = J_cᵀ[:6] f          (unactuated base)
           J_c q̈ + J̇_c q̇ = 0                    (stance feet)
           friction pyramid, CoP in foot, f_z ≥ f_min, |τ| ≤ τ_max
   │  τ = M[6:] q̈ + h[6:] − J_cᵀ[6:] f   (read out of the actuated rows)
   ▼
mj_data.ctrl[:29] → mj_step   (1 kHz; measured 0.11 ms mean solve, no rate reduction needed)
```

**Why acceleration level (M1 finding).** The velocity-level formulation in
`plan/LAB_08.md` was implemented first and measured to be structurally unable to
balance: a kinematic QP can satisfy `J_com q̇ = 0` exactly while the robot topples,
because CoM motion is produced by contact forces it does not model. Strengthening
the hand task made the fall *faster*. Full write-up: LESSONS L-M1-a.

## Key Interfaces

- `wb_tasks.Task`: `error(model, data, q)`, `jacobian(model, data, q)`,
  `drift(model, data)` (`J̇q̇`), `desired_acceleration(...)` (PD + feedforward),
  plus `weight` / `gain` / `name`. Every Jacobian ships with a finite-difference test.
- `wb_tasks.TaskStack.update_dynamics(q, v)` evaluates FK, frame placements,
  Jacobians and CoM **once** per tick with zero acceleration, so every task's
  reported acceleration *is* its drift term.
- `wb_id_qp.WholeBodyIDQP.solve(stack, q, v) -> IDQPResult(tau, qddot, forces, …)`
  — raises `QPInfeasible` rather than clamping; a silently zeroed solution looks
  like a working controller while the robot falls.
- `wb_id_qp.ContactSpec(frame_name, friction, half_length, half_width, …)` — the
  stance set. Changing this list per phase is how M2 will schedule contacts.
- `gait_planner.references(t) -> GaitRefs(com, zmp, foot_targets, contact_state)`.
- `loco_manip_fsm` mirrors Lab 5's `GraspStateMachine` contract: milestone-gated
  transitions, convergence-gated handoffs, and a **post-condition on the object pose**
  in the capstone (a run cannot claim DONE without the payload arriving).

## Model Files

| File | Source | Notes |
|---|---|---|
| `third_party/mujoco_menagerie/unitree_g1/g1.xml` | upstream (gitignored, setup_env.sh) | the single source of truth for kinematics, inertias, meshes |
| torque-actuated G1 | **built at runtime** by `src/g1_torque_model.py` | `MjSpec.set_to_motor()` on all 29 actuators; ctrlrange from each joint's `actuatorfrcrange`; floor + light added; keyframe ctrl zeroed |
| capstone scene (M5) | this lab (tracked) | table + payload; sizes reuse Lab 5 conventions |

**M0 decision — the torque model is generated, not committed.** PLAN.md originally
called for a tracked `models/g1_torque.xml`. Building it from `MjSpec` instead
(a) keeps Menagerie authoritative so upstream fixes flow through rather than
diverging from a stale fork, (b) avoids the `meshdir`-resolution shim a relocated
copy needs (Lab 2 required a tracked `models/assets` symlink for exactly this),
and (c) matches the `build_mujoco_scene_spec` convention Labs 3–4 already use.
`g1_torque_model.export_xml()` writes an inspection snapshot on demand; nothing
at runtime depends on it.

## Cross-Lab Dependencies

| From | What | How |
|---|---|---|
| Lab 7 | `lab7_common` (G1 joint maps, pelvis frame math), `lipm_planner` (validated, 18 tests), whole-body IK patterns | `add-to-sys.path` import per repo convention |
| Lab 5 | inertia-shaped PD lesson (L-6.1b), post-condition-assert pattern, simple-grasp philosophy | pattern reuse, not code import |
| Lab 3 | RNEA/CRBA cross-validation discipline | pattern reuse |
| tools/ | `video_producer.py` for the final demo composition (M6) | direct import |

## Pinocchio Rules In Force (from CLAUDE.md — non-negotiable)

- Floating base: `nq = 36 ≠ nv = 35`-class mismatch (quaternion); ALL configuration
  updates via `pin.integrate`, never `q += dq`.
- Jacobians in `pin.LOCAL_WORLD_ALIGNED`, finite-difference validated on first use.
- Analytical model built from the SAME MJCF the sim runs (`g1_torque.xml`) —
  Lab 5's L-6.1c showed what happens when the brain models a different body.
- Cross-validate g(q)/M(q) against `qfrc_bias`/`mj_fullM` at M0 gate time.
  (MuJoCo ≥ 3.11: `mj_fullM(model, data, dst)` — qM attribute is gone.)

## Open Questions

Resolved during M0/M1:

1. ~~QP rate: 1 kHz or 100–200 Hz with interpolation?~~ **1 kHz.** Measured 0.11 ms
   mean solve for 47 variables; no interpolation needed.
2. ~~Weighted vs strict HQP?~~ **Weighted**, ladder CoM 1e4 → hand 1e3 → posture 1.
   Per-task errors are logged so leakage would show in the gate tables; escalate to
   hierarchical QP (Escande et al.) only on evidence. None so far.
3. ~~Frozen-foot task vs contact-wrench constraints?~~ **Contact wrenches**, and not
   as an optimisation — it is what makes balance representable at all (L-M1-a).

Still open:

4. Whether Lab 7's `whole_body_ik.py` stacked-Jacobian solver is worth reusing for
   QP warm starts (not needed yet at current solve times).
5. M2: contact-switch handling — instantaneous set change vs force ramping through
   double support. Expect the naive switch to spike torques; measure before choosing.
