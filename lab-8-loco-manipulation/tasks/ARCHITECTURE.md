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
│   ├── wb_tasks.py              # M1: task residuals + Jacobians (CoM, foot, hand, posture)
│   ├── wb_qp.py                 # M1: OSQP velocity-level solve over the task stack
│   ├── inverse_dynamics.py      # M1: q̇_des → τ (pin.rnea + inertia-shaped PD)
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
sensors (mj_data: qpos, qvel, contact forces)
   │
   ▼
state estimator (direct state readout in sim; pelvis pose from freejoint)
   │
   ▼
references: gait_planner (CoM/footsteps, contact schedule)   hand target (fixed or FSM)
   │                                          │
   ▼                                          ▼
wb_tasks — residuals e_i, Jacobians J_i (pin, LOCAL_WORLD_ALIGNED)
   │
   ▼
wb_qp (OSQP):  min Σ w_i ‖J_i q̇ − ẋ_i‖²  s.t. q̇/q limits, support-foot constraint
   │  q̇_des (nv)
   ▼
inverse_dynamics:  q_des = pin.integrate(q, q̇_des·dt);  τ = rnea(q, q̇_d, q̈_d) + M(q)(Kp e + Kd ė)
   │  τ (29)
   ▼
mj_data.ctrl[:29] → mj_step (1 kHz; QP possibly at 100–200 Hz with τ interpolation)
```

## Key Interfaces

- `wb_tasks.Task`: `residual(q, v) -> np.ndarray`, `jacobian(q) -> np.ndarray`,
  `weight: float`, `name: str`. Every Jacobian ships with a finite-difference test.
- `wb_qp.solve(tasks, q, v, limits) -> qdot_des | raises QPInfeasible` — infeasibility
  is an exception with the failing constraint set logged, never a silent clamp.
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

## Open Questions (resolve during M0/M1, log answers in LESSONS.md)

1. QP rate: 1 kHz直 or 100–200 Hz with interpolation? (profile first)
2. Weighted lexicographic vs strict HQP — start weighted, escalate on evidence.
3. Contact modeling in the QP: frozen-foot equality task vs full contact-wrench
   constraints — start frozen-foot (M1–M2), add wrench cone only if ZMP gate fails.
4. Whether Lab 7's `whole_body_ik.py` stacked-Jacobian solver is worth reusing for
   QP warm starts.
