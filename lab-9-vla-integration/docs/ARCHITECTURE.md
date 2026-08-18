# Lab 9 — Architecture

A language-conditioned policy that drives a walking humanoid: how the pieces fit
and why each is shaped the way it is. Milestone-by-milestone narrative in
[`../tasks/LESSONS.md`](../tasks/LESSONS.md); guided read of the source in
[`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md).

---

## The one-paragraph version

A frozen CLIP text tower turns an instruction into a 512-d vector. An ACT
policy takes that vector, two 128 px egocentric camera views and a
proprioception vector, and emits a chunk of twenty future actions at 10 Hz.
Those actions are not joint torques and not joint angles: they are the
*references* Lab 8's whole-body inverse-dynamics QP already consumes — hand
targets, a gait command, a grasp bit. The QP runs at 1 kHz underneath and keeps
the robot upright. The policy decides *what to do*; Lab 8 decides *how not to
fall over*.

```
      "pick up the red cup"
              │
     ┌────────▼─────────┐   frozen; every instruction's embedding is baked into
     │ CLIP text tower  │   the checkpoint, so evaluation needs neither
     └────────┬─────────┘   transformers nor the network
              │ 512
head cam ─┐   │
wrist cam ┤   │        ┌──────────────────────────────┐
 (128²)   ├───┴───────►│  ACT policy   ~15.8M params  │
proprio ──┘            │  ResNet18, layer4 trainable  │
 (62-d)                │  16 spatial tokens / camera  │
                       │  + state + instruction token │
                       │  → 20 actions per inference  │
                       └───────────────┬──────────────┘
                                       │ 10 Hz, 9-d action
                       ┌───────────────▼──────────────┐
                       │  Lab 8 whole-body ID QP      │   1 kHz
                       │  balance is a hard constraint│
                       └───────────────┬──────────────┘
                                       │ τ (29)
                                  MuJoCo G1
```

---

## Why the policy does not output joint targets

`plan/LAB_09.md` specifies the action space as *"joint position targets for all
actuated DOFs"*. On a fixed-base arm that is right, and it is what
`humanoid_vla` does. On a floating base it is the thing Lab 7 already measured:
a joint-position reference tracked by PD cannot stabilise this robot. That
finding is why Lab 7's ZMP walking failed after six attempts and why Lab 8 exists
at all.

So the primary head emits what Lab 8's QP consumes and the brief's literal head
is kept as an ablation. Both are trained; both are measured. The arc from Lab 7
through Lab 8 to here is only closed by running it rather than citing it.

The `task` head, 9 dimensions, in the pelvis's yaw-only frame:

| slice | meaning |
|---|---|
| `0:3` | right-hand target |
| `3:6` | left-hand target |
| `6` | gait command — take a walk unit, or stand |
| `7` | close the right grasp |
| `8` | close the left grasp |

**Yaw-only, not the full pelvis rotation.** The pelvis pitches and rolls
continuously while walking; folding that into a hand target would inject gait
oscillation into a quantity the policy is supposed to hold still. Expressing the
target relative to the pelvis at all is what makes the same reach the same
action wherever along the walk it happens.

---

## What the policy is not allowed to know

`state` is 62 numbers: 29 joint positions, 29 joint velocities, pelvis height,
pelvis roll, pelvis pitch, and the grasp bit. It excludes the pelvis's **world
x, y and yaw**, and that exclusion is the difference between an evaluation that
measures something and one that does not.

A policy handed its own world coordinates can solve every task here by dead
reckoning — walk until `x > 0.25`, reach to a fixed offset — without ever
looking at an image or reading its instruction. It would post an excellent
success rate having learned neither vision nor language. Everything the policy
knows about where it is has to come through the pixels.

What remains is exactly what a real robot observes without external
instrumentation: joint encoders, and an IMU for height, roll and pitch. The
restriction is physical rather than arbitrary, and
`tests/test_scene_and_contract.py` asserts that translating the robot in the
world leaves the state unchanged.

---

## Why the scene carries two objects

A policy conditioned on task labels alone can infer the task from the robot's
own pose — walking and reaching look nothing alike — and ignore the language
entirely. Any success rate measured that way is a statement about the scene.

So a **red cup** and a **blue box** stand on the pedestal, which one is nearer is
randomised per seed, and the instruction names the target. The same image demands
different actions under different instructions. That makes three things possible
that otherwise are not:

- the **instruction-swap test** — same initial state, the other object's
  sentence, does the behaviour follow;
- a *walk* task that is genuinely language-conditioned, because the named object
  decides how far to go (two steps or four);
- the brief's capstone sentence, *"pick up the red cup"*, taken literally rather
  than as a label for the only object present.

---

## The expert, and why the task set is two tasks

Demonstrations come from Lab 8's whole-body controller, unmodified.
`expert.VLAExpert` subclasses Lab 8's `Capstone` and inherits every phase method;
only the scene, the target selection and an observation-capture hook differ.

The task set is **`walk` and `pick`**, not the brief's three-to-five, and the
reason is the expert rather than the model. Lab 8's capstone gate is 4/4 on one
configuration; over a randomised two-object scene the same sequence scored 1/8.
Measured per task:

| task | measured | mechanism |
|---|---|---|
| `walk` + `pick` | **40/40** | — |
| `carry` | 1/12 | `carry_targets` mirrors the grip about the payload; at this grasp offset the two wrist targets come out 22–35 mm apart, so both wrists are asked into nearly the same point |
| `place` | 5/10 | the hand tasks control position only, so the object is released at whatever tilt the wrist has (22° measured) and rolls off the marker |

A demonstration set whose expert falls in half its episodes teaches a policy to
fall, and no model work recovers from that. Restoring `place` needs a
hand-**orientation** task in Lab 8's stack, which is Lab 8 work.

### The standing-stability budget

Lab 8's `_freeze_balance` pins the DCM target at the value it had when the phase
began. That is correct for a short motion and wrong for a long one: moving an arm
shifts the centre of mass, and the frozen target then commands the robot back
toward a snapshot that no longer describes a resting configuration.

Lab 8 never hit this because it *walked* between manipulation phases, which
replans the DCM from scratch. Lab 9 has no carry-walk, so its whole manipulation
is one continuous stand:

| continuous standing | episodes completed |
|---|---|
| 11.5 s (Lab 8's timings) | 0 / 4 |
| 6.9 s | 3 / 4 |
| **5.2 s** | **4 / 4** |

The signature is unmistakable: the DCM error grows exponentially at the LIPM
rate, doubling every ~0.15 s from 4.5 mm, while the hand still tracks to 5 mm and
peak torque sits at 21 N·m. An instability, not a saturation. Lab 9's phase
durations are sized to a 5.6 s budget and a test asserts it.

---

## Modules

| File | Role |
|---|---|
| `lab9_common.py` | Paths, image/observation constants, instruction vocabulary, Lab 8 re-exports |
| `vla_scene.py` | Two-object randomisable scene, head + wrist cameras, four welds, drop marker |
| `observations.py` | The observation and action contract — the only place the layout is defined |
| `expert.py` | Lab 8's capstone subclassed: seedable, target-selectable, observation-capturing |
| `collect_demos.py` | Multi-process rollouts, phase-sliced into labelled task segments |
| `dataset.py` | Chunked windows with pad masks, seed-level split, normalisation statistics |
| `text_encoder.py` | Frozen CLIP text tower and the instruction bank |
| `act_policy.py` | The ACT model: two cameras, derived token count, two action heads |
| `train.py` | Masked L1 training, baseline-relative validation |
| `policy_runner.py` | Closed-loop execution: policy at 10 Hz over Lab 8's QP |
| `evaluate.py` | Per-task success, randomised range, instruction swap, joint ablation |
| `capstone_demo.py` | Free-form language in, recorded episode out, inference profiling |
| `mN_*.py` | One gate script per milestone, each writing its evidence to `media/` |

Nothing about the scene is committed: it is built at runtime from Menagerie plus
Lab 8's spec builder, exactly as Lab 8 does.

---

## Data flow

### Collecting one demonstration

```
seed ──► Randomisation ──► scene (two objects, cameras, welds)
                             │
                     Lab 8 controller runs the episode at 1 kHz
                             │  every 100 ticks:
                             ▼
        two 128px renders + 62-d state + the expert's own command
                             │  sliced by phase
                             ▼
              walk segment          pick segment
        "walk to the red cup"   "pick up the red cup"
```

The stored action is the **expert's command**, not the achieved state. Behaviour
cloning imitates what the expert did; on a compliant, disturbed system the two
differ, and training on the outcome teaches the policy to chase its own past.

Only successful episodes are written. A failed one is a recording of a robot
falling over, and its frames look exactly like good ones until the moment it
goes down.

### One closed-loop tick

```
MuJoCo state ─► two renders + state ─► ACT ─► chunk (20, 9)
                    │ ~194 ms                    │ first action
                    │ (software rendering,       ▼
                    │  the real rate limiter)  decode into world frame
                                                 │
                                          gait? ─┴─ stand
                                            │         │
                                     Lab 8 walk unit  hand tasks + frozen DCM
                                            │         │
                                            └────┬────┘
                                          Lab 8 QP at 1 kHz → τ
```

A biped cannot be told to stop in the middle of a step, so the gait command is
acted on only at walk-unit boundaries, and a unit is one step plus its closing
step — the configuration Lab 8 validated (L-M5-e: a walk that ends mid-stride
hands the next one a stance it cannot survive).

---

## Model

Adapted from `ozkannceylan/humanoid_vla`'s `ACTPolicy`, itself Zhao et al. (RSS
2023). The deltas, and why:

| upstream | here | why |
|---|---|---|
| 49 spatial tokens, hardcoded | derived from the feature map (16 at 128 px) | 49 is ResNet18's 7×7 output for a 224 px input and silently wrong at any other size |
| one camera | head + wrist, each with its own camera embedding | at 128 px the objects are a handful of pixels in the head view by the time the hand is near them |
| action = joint targets | two heads, `task` primary | Lab 7's finding |
| optional temporal ensembling | absent | an extra forward pass per step, and this control loop is already render-bound |
| ImageNet ResNet18, layer4 fine-tuned | same | keep it |
| norm stats as module buffers | same | keep it — a checkpoint that cannot denormalise its own output loads cleanly and is wrong by a scale factor |

~15.8 M parameters, ~13.0 M trainable.

### The instruction bank

The text tower is used at *training* time to embed the vocabulary, and the
embeddings are stored inside the checkpoint. Evaluation, the closed-loop runner
and the capstone then need neither `transformers` nor the network — they look the
instruction up. Encoding a genuinely novel sentence at inference time still
works and still needs the tower.

Measured on this vocabulary: paraphrases of the same command sit at cosine 0.957,
commands that mean different things at 0.846 — a margin of 0.111. Paraphrase
robustness and instruction separability are the two properties the conditioning
needs, and they pull in opposite directions, so both are checked before training
rather than inferred from a bad success rate afterwards.

---

## Training

Masked L1 over the chunk. Two details that are easy to get wrong:

**The mask.** Near the end of a segment there are fewer than `chunk_size` real
actions left. Unmasked, the padded tail teaches the policy to stop moving two
seconds before the task ends.

**The split is by scene seed, never by frame.** Two frames 100 ms apart in the
same episode are near-duplicates; a frame-level split reports a validation loss
that measures memorisation, and it looks excellent.

Validation is reported in **raw units next to a predict-the-mean baseline**. A
normalised L1 of 0.31 says nothing. The same number in millimetres of hand
target, beside what predicting the training mean would score, says whether the
model learned anything at all.

`layer4` gets a tenth of the head's learning rate: it carries ImageNet features
that a few thousand samples can destroy faster than they can improve.

---

## Frame and unit conventions

- Hand targets are **pelvis-relative, yaw-only**, in metres.
- Images are uint8 RGB at 128×128, normalised with ImageNet statistics inside
  the model.
- Lab 8's conventions carry through unchanged: `pin.LOCAL_WORLD_ALIGNED`
  Jacobians, `pin.integrate` for configuration updates (`nq ≠ nv` on a floating
  base), Pinocchio's world sitting `PELVIS_MJCF_Z` below MuJoCo's.
- Foreign labs go on `sys.path` with **`append`, never `insert(0)`** — labs share
  module names and a foreign `src/` ahead of this one silently shadows local
  modules.

---

## Where the compute went

| quantity | measured |
|---|---|
| CPU / RAM | 4 cores, 15 GB, **no CUDA device** |
| MuJoCo offscreen render | **97 ms/frame**, resolution-independent (380 ms with shadows, reflection and skybox on) |
| ResNet18 fwd+bwd, batch 16 | 32 samp/s @224 px · 117 @128 px · 193 @96 px |
| One expert episode | ~35 s wall for ~11 s of simulation |

Every one of these was measured *before* the plan was written, and they set the
image size, the shadow settings, the dataset size and the epoch budget. A plan
written against assumed hardware is a plan for a machine you do not have.
