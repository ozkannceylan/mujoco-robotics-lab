# Lab 9 — VLA Integration

> **Status:** 🚧 In progress — **M0 complete** (2026-08-17)
> **Platform:** Unitree G1 under torque control (Lab 8) + ACT policy + frozen CLIP text tower
> **Goal:** one sentence in, autonomous loco-manipulation out — the destination
> the whole series was built toward.

Labs 1–8 built the manual pipeline: kinematics, dynamics, planning, grasping,
locomotion, whole-body control. Lab 9 replaces the hand-coded task logic with a
learned policy that takes camera images plus a language instruction and emits
actions. Understanding the manual stack is what makes the learned one debuggable
— and in this lab it is also what *runs underneath* it: the policy decides what
to do, and Lab 8's whole-body QP keeps the robot upright while it does it.

---

## Milestones

| # | Milestone | Gate | Status |
|---|---|---|---|
| M0 | Scene, cameras, obs/action contract | expert success ≥ 70 %, cameras render, codecs exact | ✅ **PASS** |
| M1 | Demonstration dataset | ≥ 50 demos/task, integrity, no seed leakage | 🚧 |
| M2 | Model | shapes, overfit-one-batch, language changes the action | 🚧 |
| M3 | Training | val error beats predict-the-mean | ⏳ |
| M4 | Closed-loop evaluation | > 70 % seen, > 40 % randomised, instruction swap works | ⏳ |
| M5 | Capstone + inference profiling | free-form language → autonomous episode, > 10 Hz | ⏳ |
| M6 | Documentation & blog | docs EN/TR + blog | ⏳ |

---

## Four deviations from the brief, and why

Each is forced by something measured. Full argument in
[`tasks/PLAN.md`](tasks/PLAN.md).

**1. The language head is not this lab's contribution.** `plan/LAB_09.md` says
to extend `ozkannceylan/humanoid_vla` by *adding* language conditioning. Reading
that repository first showed it already ships a frozen CLIP text tower, an
instruction bank baked into the checkpoint, and spatial vision tokens. What has
never existed is the brief's *other* bullet — the expert is Labs 3–8, so the
policy is trained on a **walking** humanoid, where balance is a live constraint
for every action it emits. That is the contribution.

**2. There is no GPU, and rendering is software.** Measured before planning
anything: no CUDA device, 4 cores; MuJoCo offscreen rendering at **97 ms/frame
and resolution-independent** (the cost is per-geometry setup in llvmpipe, not
fill — 64 px and 224 px cost the same, and 380 ms with shadows and reflections
on). So 128 px images, shadows off, and a training budget sized to four cores.
The brief's "INT8 on an RTX 4050" becomes CPU dynamic quantisation, reported as
that.

**3. The action space the brief specifies is the one Lab 7 predicts will fail.**
The brief asks for joint-position targets on all 29 DOF. On a floating base
that is precisely what Lab 7 measured cannot stabilise this robot. So the
primary head emits what Lab 8's QP consumes — hand targets, a gait command, a
grasp bit — and the brief's literal joint head is kept as the ablation that
tests the prediction.

**4. Two tasks, not the brief's three-to-five — because of the expert, not the
model.** See M0.

---

## M0 — Scene, Cameras, and the Observation/Action Contract ✅

A randomisable two-object loco-manipulation scene, egocentric cameras, a frozen
definition of what the policy sees and emits, and a programmatic expert built
from Lab 8's controller.

```bash
MUJOCO_GL=egl python3 lab-9-vla-integration/src/m0_scene_check.py --seeds 20
pytest lab-9-vla-integration/tests/      # 30 tests
```

### Gate results

| Criterion | Result | Measured |
|---|---|---|
| Expert success rate ≥ 70 % | PASS | **100 % (40/40)**, 20 seeds × 2 objects |
| Both cameras render at 128 px | PASS | head + wrist, non-degenerate |
| Action codec round-trips exactly | PASS | 5.9e-08 |
| State matches its declared dimension | PASS | 62 |
| Approach depends on the named object | PASS | 2 steps or 4, by target |
| Torques within limits on success | PASS | 92.6 N·m peak (limit 139) |

reach error **15.2 ± 7.3 mm** · lift height **90 ± 6 mm**

![M0 scene](media/m0_scene.png)

Video: [`media/m0_expert_rollout.mp4`](media/m0_expert_rollout.mp4)

### Two design decisions the rest of the lab rests on

**The state vector deliberately omits the pelvis's world x, y and yaw.** A
policy handed its own world coordinates can solve every task here by dead
reckoning — walk until x > 0.25, reach to a fixed offset — without ever looking
at an image or reading its instruction, and it would post an excellent success
rate having learned nothing about either. What remains (joint angles and
velocities, pelvis height, roll and pitch, grasp bit) is what a real IMU and
joint encoders observe. A test asserts that translating the robot in the world
leaves the state unchanged.

**The scene carries two objects, and the instruction chooses.** With one object
a four-task label set is inferable from the robot's own pose — walking and
reaching look nothing alike — so language would be free to ignore. With two, the
same image demands different actions under different instructions, which is the
only setup in which "the policy follows instructions" is falsifiable. The named
object even decides *how far to walk*: the near one is a two-step approach, the
far one four.

### The finding: an inherited expert is a hypothesis, not a given

The plan assumed Lab 8's capstone controller would demonstrate walk → pick →
carry → place. Lab 8's own gate is 4/4 on **one** configuration. Run over a
randomised two-object scene, the same sequence scored **1/8**.

That is not a bug in either lab — it is Lab 8's own M4 lesson applying to Lab 8:
*a result a no-op perturbation destroys is a draw from a distribution*, and
randomising object placement is such a perturbation. So each task was measured
separately and the set cut to what the expert can actually demonstrate:

| task | measured | why it fails |
|---|---|---|
| `walk` + `pick` | **40/40** | — |
| `carry` (walk holding the object) | 1/12 | `carry_targets` mirrors the grip about the payload, and at this grasp offset the two wrist targets come out 22–35 mm apart — both wrists are asked into nearly the same point |
| `place` (set it on a marker) | 5/10 | the hand tasks control position only, so the object is released at whatever tilt the wrist has (22° measured) 12 mm above the surface, lands on an edge and rolls 84 mm |

A demonstration set whose expert falls in half its episodes teaches a policy to
fall, and no model work recovers from that. Restoring `place` needs a
hand-**orientation** task, which is a change to Lab 8's controller and is
recorded as Lab 8 follow-up rather than smuggled in here.

### Three more measurements worth carrying

**A frozen balance reference has a shelf life, and it is about six seconds.**
Lab 8's `_freeze_balance` pins the DCM target at the value it had when the phase
began. Lab 9 has no carry-walk, so its whole manipulation happens in one
continuous stand — 11.5 s at Lab 8's timings, where Lab 8 never stood more than
~7 s before walking and replanning. The failure is unmistakable once traced: the
DCM error grows **exponentially at the LIPM rate**, doubling every ~0.15 s from
4.5 mm, while the hand still tracks to 5 mm and peak torque sits at 21 N·m.
Saturation only appears on the way down.

| continuous standing | episodes completed |
|---|---|
| 11.5 s (Lab 8's timings) | 0 / 4 |
| 6.9 s | 3 / 4 |
| **5.2 s** | **4 / 4** |

Splitting the motion into short re-anchored segments was tried and measured
*worse* — re-freezing repeatedly removes the feedback that was correcting the
drift. Lab 9's phase durations are sized to a 5.6 s budget and a test asserts it.

**Reach accuracy is the wrong quantity to tune a stopping distance on.** It is
flat at 7–11 mm for standoffs from −0.01 m to 0.37 m, so 0.22 m looked as good
as anything. At 0.22 m the arm is extended 0.43 m from the pelvis, and the
*lift* — not the reach — saturates the waist and puts the robot down. Lab 8's
own capstone stood 0.06 m from its payload, making the reach almost entirely
lateral with the arm folded.

**A borrowed constant hides the object it was measured on.** Lab 8's
`GRASP_OFFSET` is a fixed −0.060 m: its payload's 0.030 m half-extent plus a
0.030 m wrist clearance. Applied to a 0.040 m radius cup it puts the wrist
*inside* the object. Every one of M0's four failures at 90 % was a near cup,
reaching to 29–30 mm where the box reached to 7–11 mm from the identical
controller. Scaling the offset by the target's own half-extent: **36/40 →
40/40**.

---

## Architecture

```
      "pick up the red cup"
              │
     ┌────────▼─────────┐   frozen; embeddings baked into the checkpoint,
     │ CLIP text tower  │   so evaluation needs neither transformers nor
     └────────┬─────────┘   the network
              │ 512
head cam ─┐   │
wrist cam ┤   │        ┌──────────────────────────────┐
 (128²)   ├───┴───────►│  ACT policy                  │
proprio ──┘            │  ResNet18 (layer4 trainable) │
 (62-d)                │  + spatial tokens + state    │
                       │  + instruction token         │
                       │  → chunk of 20 actions       │
                       └───────────────┬──────────────┘
                                       │ 10 Hz
                       ┌───────────────▼──────────────┐
                       │  Lab 8 whole-body ID QP      │   1 kHz
                       │  balance is a hard constraint│
                       └───────────────┬──────────────┘
                                       │ τ (29)
                                  MuJoCo G1
```

The policy decides *what*; Lab 8's QP decides *how to stay upright*. Balance is
never a learned quantity.

Design record: [`tasks/ARCHITECTURE.md`](tasks/ARCHITECTURE.md) ·
Milestone plan: [`tasks/PLAN.md`](tasks/PLAN.md) ·
Every finding in long form: [`tasks/LESSONS.md`](tasks/LESSONS.md)

### Modules

| File | Role |
|---|---|
| `src/lab9_common.py` | Paths, obs/action constants, instruction vocabulary, Lab 8 re-exports |
| `src/vla_scene.py` | Two-object randomisable scene, egocentric cameras, four welds |
| `src/observations.py` | The observation and action contract — the only place the layout is defined |
| `src/expert.py` | Lab 8's capstone subclassed: seedable, target-selectable, observation-capturing |
| `src/m0_scene_check.py` | M0 gate |
| `src/collect_demos.py` | Multi-process demonstration generation, phase-sliced |
| `src/dataset.py` | Chunked dataset with pad masks and a seed-level split |
| `src/text_encoder.py` | Frozen CLIP text tower + instruction bank |
| `src/act_policy.py` | The ACT model: two cameras, derived token count, two action heads |
| `src/m2_model_check.py` | M2 gate |
| `src/train.py` | Training loop, masked L1, baseline-relative validation |
| `src/policy_runner.py` | Closed-loop execution: policy at 10 Hz over Lab 8's QP |
| `src/evaluate.py` | M4: per-task success, randomised, instruction swap, joint-head ablation |
| `src/capstone_demo.py` | M5: free-form language in, recorded episode out |
| `tests/` | Scene, contract, randomisation, instructions, model, checkpoint |

---

## Setup

```bash
./tools/setup_env.sh                     # deps + Menagerie clone
pip install osqp                         # Lab 8's QP solver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers                 # training only; evaluation uses the bank
export MUJOCO_GL=egl
```

---

## Credits

The ACT architecture is Zhao et al., *Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware* (RSS 2023). The implementation here is
adapted from [`ozkannceylan/humanoid_vla`](https://github.com/ozkannceylan/humanoid_vla),
which supplied the design of the frozen text tower, the instruction bank and the
in-checkpoint normalisation; the deltas (two cameras, derived token count, this
lab's action heads) are listed in `tasks/ARCHITECTURE.md`.
