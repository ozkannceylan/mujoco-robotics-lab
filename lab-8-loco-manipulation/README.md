# Lab 8 — Whole-Body Loco-Manipulation

> **Status:** 🚧 In progress — **M0 complete** (2026-08-15), M1 next.
> **Platform:** Unitree G1 (MuJoCo Menagerie, 29 DOF) under **torque** control + Pinocchio
> **Goal:** A humanoid that walks and uses its hands at the same time — the operating
> mode Lab 9's VLA policy will have to produce.

Lab 7 took the G1 as far as its position servos allow: standing balance and
quasi-static weight shifting worked, and dynamic ZMP walking provably did not
(6 attempts; IK converged, PD replay diverged). The diagnosis was the actuator
model. **Lab 8 is the test of that diagnosis** — it re-actuates the G1 with
torque motors and rebuilds the control stack as whole-body QP → inverse
dynamics → joint torques, owning gait generation rather than inheriting it.

---

## Milestones

| # | Milestone | Gate | Status |
|---|---|---|---|
| M0 | Torque-actuated G1 bring-up | 10 s stand, CoM drift < 30 mm, model parity < 1e-6 | ✅ **PASS** |
| M1 | Whole-body QP (standing reach) | hand RMS < 20 mm, CoM inside support polygon | ⏳ next |
| M2 | Torque-level stepping | 4 in-place steps, ZMP inside polygon > 95% stance | — |
| M3 | Forward walking | ≥ 10 steps, ≥ 1.0 m, no fall | — |
| M4 | Walk + arm task | M3 gate holds, hand error < 50 mm while walking | — |
| M5 | Loco-manipulation capstone | walk → grasp → carry → place, object within 50 mm | — |
| M6 | Documentation & blog | docs EN/TR + blog post | — |

---

## M0 — Torque-Actuated G1 Bring-Up ✅

Menagerie ships the G1 with 29 `<position kp="500">` servos. Those compute a PD
law *inside* MuJoCo, which means the only available command is a joint angle —
there is nowhere to inject a torque from an inverse-dynamics pipeline. M0
replaces them with `<motor>` actuators and re-establishes standing from outside
the simulator.

```bash
MUJOCO_GL=egl python3 lab-8-loco-manipulation/src/m0_torque_standing.py
pytest lab-8-loco-manipulation/tests/          # 18 tests
```

### Gate results

| Criterion | Result | Measured |
|---|---|---|
| Stands 10 s without falling | PASS | no fall |
| CoM horizontal drift < 30 mm | PASS | **0.71 mm** |
| Both feet in contact at end | PASS | yes |
| CoM inside support polygon | PASS | 52.7 mm margin |
| g(q) parity vs MuJoCo `qfrc_bias` | PASS | 1.74e-16 (relative) |
| M(q) parity vs MuJoCo `mj_fullM` | PASS | 9.32e-17 (relative) |
| Torque command authority | PASS | motor actuators, 5–139 N·m |

![M0 standing metrics](media/m0_standing_metrics.png)

Video: [`media/m0_torque_standing.mp4`](media/m0_torque_standing.mp4)

### What M0 actually measured

The milestone is not "the robot stands" — position servos did that for free in
Lab 7. It is *which terms are needed once they are gone*, so the ablation is
part of the deliverable:

| Gravity mode | Result | CoM drift | Steady joint error | \|τ\|max |
|---|---|---|---|---|
| none (pure joint PD) | STAND | 0.18 mm | 2.77 mrad | 1.4 N·m |
| free-space `g(q)` | STAND | 0.96 mm | 1.40 mrad | 1.6 N·m |
| contact-consistent `g(q) − τ_c` | STAND | 0.62 mm | **0.00 mrad** | 3.9 N·m |
| `g(q)` alone, no PD | **FELL** | — | collapses to 0.097 m in ~2 s | — |

Two findings worth carrying forward (full write-ups in
[`tasks/LESSONS.md`](tasks/LESSONS.md)):

- **Gravity compensation alone cannot stand.** It cancels weight without
  stabilising posture; a standing humanoid is an inverted pendulum. Every
  stabilising term the servo used to provide has to be re-supplied explicitly.
- **Inertia-shaping the PD gains makes the G1 fall** — and that fix
  (`τ = M(q)(Kp·e + Kd·ė) + g`) was inherited from Lab 5, where it was correct.
  `M(q)[6:,6:]` on a *floating* base is not the reflected inertia a standing
  robot feels through its closed leg chains; multiplying gains by it saturates
  the actuators. Raw joint-space gains stand with 0.18 mm drift. An inherited
  fix is only valid inside the assumptions that produced it.

---

## Architecture

```
gait refs (M2+)   hand target (M1+)
        │               │
        ▼               ▼
   task stack: CoM · feet · hand · posture      (Pinocchio, LOCAL_WORLD_ALIGNED)
        │
        ▼
   whole-body QP  (OSQP)  → q̇_desired
        │
        ▼
   inverse dynamics (Pinocchio RNEA) → τ
        │
        ▼
   MuJoCo, torque actuators, 1 kHz
```

Full module map, data flow and interface contracts:
[`tasks/ARCHITECTURE.md`](tasks/ARCHITECTURE.md).
Milestone plan and gates: [`tasks/PLAN.md`](tasks/PLAN.md).

### Modules (M0)

| File | Role |
|---|---|
| `src/g1_torque_model.py` | Builds the torque-actuated G1 from Menagerie via `MjSpec` (servos → motors, limits from `actuatorfrcrange`, floor, keyframe hygiene) |
| `src/lab8_common.py` | Paths, constants, model loaders, MuJoCo↔Pinocchio state conversion, CoM / contact / support-polygon helpers |
| `src/standing_controller.py` | Joint PD + selectable gravity mode (`none` / `free_space` / `contact_consistent`) |
| `src/m0_torque_standing.py` | M0 gate: cross-validation, ablation, recorded 10 s hold |
| `tests/test_torque_model.py` | 18 tests: actuator semantics, torque limits, model parity, frame conventions |

The torque model is **generated at runtime, not committed** — Menagerie stays
the single source of truth. Rationale in `tasks/ARCHITECTURE.md` § Model Files.

---

## Setup

```bash
./tools/setup_env.sh          # deps + Menagerie clone (incl. unitree_g1)
pip install osqp              # whole-body QP solver (M1+)
export MUJOCO_GL=egl          # headless rendering
```
