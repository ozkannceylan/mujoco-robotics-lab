# Lab 9 — VLA Integration: Milestone Plan

> Created: 2026-08-17 · Brief: `plan/LAB_09.md` · Platform: Unitree G1 (torque-actuated, Lab 8) on MuJoCo
> Capstone: a language command — *"pick up the red cup"* — drives an autonomous
> loco-manipulation episode end to end.

---

## Four deviations from the brief (read first)

Every one of these is forced by something measured, not preferred. They are
listed here so the gates below can be read against what the lab actually
promises.

### 1. `humanoid_vla` already has the thing the brief says to add

`plan/LAB_09.md` says: *"Extend: add language conditioning (humanoid_vla uses
fixed tasks)."* That is stale. Upstream `ozkannceylan/humanoid_vla` already
ships `models/text_encoder.py` (frozen CLIP text tower, 512-d, projected and
L2-normalised, with an instruction bank baked into the checkpoint) and
`ACTPolicy(conditioning="text", vision_tokens="spatial")`. Language
conditioning is not this lab's contribution.

What *is* this lab's contribution is the brief's other bullet, the one the
master plan calls the critical path: **the expert is Labs 3–8, so the policy is
trained on a walking humanoid.** `humanoid_vla` is fixed-base tabletop
manipulation. Lab 9 is loco-manipulation — the robot walks to the object, and
balance is a live constraint for every action the policy emits. That changes the
observation (egocentric view from a moving base), the action space (see 3), and
the failure modes.

So: **reuse the ACT design, re-implement it here** adapted to this lab's
constraints (small feature maps, this lab's action space, this lab's
normalisation), and credit it. Do not vendor a copy of a repo whose defaults
were chosen for a different problem.

### 2. There is no GPU, and rendering is software

Measured on this machine before planning anything:

| quantity | measured |
|---|---|
| CPU / RAM | 4 cores / 15 GB, **no CUDA device** |
| MuJoCo offscreen render (EGL → llvmpipe) | **97 ms/frame**, and *resolution-independent* (64 px and 224 px cost the same; 380 ms with shadows + reflection + skybox on) |
| ResNet18 fwd+bwd, layer4 trainable, batch 16 | 32 samp/s @224 px · 117 @128 px · **193 @96 px** |
| One Lab 8 capstone episode, no render | 53.6 s wall for 25.7 s of sim |

The brief's *"cloud GPU: Lambda Labs or RunPod"* and *"INT8 for local inference
on RTX 4050"* have no hardware here. Consequences, all of them planned around
rather than discovered late:

- **128×128 images**, not 224. Resolution is free on the render side and 3.7×
  cheaper on the training side.
- **Shadows, reflections and the skybox are off** during data collection —
  4× cheaper per frame, and they are the least informative pixels in the image.
- **One expert rollout per seed, sliced into several labelled task segments**,
  so 60 sim runs yield ~240 demonstrations instead of 240 sim runs.
- Training set sized to what fits: ~15–20 k samples, ~40 epochs, ≈2 h on 4 cores.
- INT8 becomes **torch dynamic quantisation on CPU**, reported as the CPU
  analogue with a measured Hz, not as the brief's RTX number.

### 3. The action space the brief specifies is the one Lab 7 predicts will fail

`plan/LAB_09.md` Phase 2: *"Define action space: joint position targets for all
actuated DOFs."* On a fixed-base arm that is correct and is what `humanoid_vla`
does. On a floating base it walks straight into Lab 7's finding: a joint-position
reference tracked by PD **cannot stabilise this robot** — that is precisely why
Lab 7's ZMP walking failed and why Lab 8 exists.

Lab 9 therefore has two heads and measures both:

- **`task` (primary)** — the policy emits the reference Lab 8's whole-body QP
  consumes: right/left hand targets, a gait command, and a grasp bit. The QP
  keeps the robot upright; the policy decides what to do. This is also what real
  VLAs emit (RT-1/RT-2 output end-effector deltas plus a gripper bit, not joint
  torques).
- **`joint` (ablation)** — the brief's literal 29-DOF joint-target head, executed
  by joint PD under torque control.

If the ablation falls, that is the lab's headline finding and it closes the arc
Lab 7 opened. If it does not, the primary head was unnecessary and the brief was
right. Either way the number goes in the README.

### 3b. The task set is two tasks, because that is what the expert can demonstrate
*(added at M0, from measurement — see `tasks/LESSONS.md` § L-M0-a, c, d, e)*

The plan above assumed Lab 8's capstone would demonstrate walk → pick → carry →
place. It does not. Lab 8's gate is 4/4 on **one** configuration; the same
sequence over a randomised two-object scene scored **1/8**, and the two failing
halves were characterised rather than tuned around:

| task | best measured | why it fails |
|---|---|---|
| `walk` + `pick` | **40/40** | — |
| `carry` (walk holding the object) | 1/12 | the tuck asks both wrists into nearly the same point (22–35 mm apart); one-armed carry is Lab 8's own known failure (L-M5-g) |
| `place` (set it on a marker) | 5/10 | the hand tasks control position only, so the object is released at whatever tilt the wrist has (22° measured) and rolls |

So Lab 9 ships **two** tasks — `walk to the {object}` and `pick up the
{object}` — with an expert that succeeds 100 % of the time. Language stays fully
load-bearing: the named object decides how far to walk (2 steps or 4) and where
to reach. Restoring `place` needs a hand-*orientation* task, which is a change
to Lab 8's controller and therefore Lab 8 work; that is recorded as the follow-up
rather than smuggled in here.

### 4. Language has to be load-bearing, or the evaluation is worthless

A policy conditioned on 4 task labels can ignore the text entirely and infer the
task from the image — the robot's pose alone says whether it is walking or
reaching. Any success rate measured that way says nothing about language.

So the scene carries **two distinct objects on the pick pedestal — a red cup and
a blue box — at randomised positions**, and the instruction names which one.
Identical image, different instruction, different correct action. A policy that
ignores language cannot beat chance on object selection, and the
**instruction-swap test** (feed the other object's instruction on the same
initial state, measure whether the behaviour follows) is a first-class gate, not
a nice-to-have. It is also exactly the brief's capstone sentence — *"pick up the
red cup"* — taken literally.

---

## Ground rules (inherited from Labs 6–8)

- ONE milestone at a time; each ends with a gate check + evidence in
  `media/mN_*` (video, plot, or printed table). If a gate fails, fix *that*
  milestone.
- Expert demonstrations come from Lab 8's controllers unmodified. If Lab 8 code
  needs changing, that is a Lab 8 regression and its gates get re-run.
- Cross-lab imports use `sys.path.append` for the foreign lab, never
  `insert(0)` (Labs share module names — Lab 8 lost an afternoon to this).
- Report what was measured. An exploratory result is labelled exploratory
  (Lab 8's M4 precedent), and a no-op perturbation check is run before a pass is
  believed.
- Every stored artefact that a gate asserts on is a **simulated** outcome, never
  a commanded one (Lab 5's precedent).

---

## Milestones

### M0 — Scene, cameras, and the observation/action contract

The lab's foundation: a randomisable two-object loco-manipulation scene with
egocentric cameras, and a frozen definition of what the policy sees and emits.

- 0.1 `lab9_common.py` — paths, constants, Lab 8 imports, image/observation
      constants, instruction vocabulary, the task enum.
- 0.2 `vla_scene.py` — Lab 8's capstone scene plus: a **red cup** (cylinder) and
      a **blue box** on the pick pedestal, a **head camera** on `torso_link` and
      a **wrist camera** on `right_wrist_yaw_link`, per-seed randomisation of
      object positions/order, lighting and object hue jitter.
- 0.3 `observations.py` — the observation builder (image(s) + proprioception)
      and the action encoder/decoder for both heads, with round-trip tests.
- 0.4 `expert.py` — the Lab 8 capstone sequence rewritten as a *scriptable*
      episode over the two-object scene, emitting a phase-labelled log.
- **Gate**: 20 randomised seeds run end to end; expert success rate printed and
  ≥ 70 %; both cameras render; obs/action round-trip exact; the scene's own
  Lab 8 regression (M3 walk unchanged) still passes.
- **Evidence**: `media/m0_scene.png` (both camera views + third-person),
  `media/m0_expert_rollout.mp4`, printed randomisation/success table.

### M1 — Demonstration dataset

- 1.1 `collect_demos.py` — multi-process generation, one expert rollout per
      seed, sliced by phase into labelled task segments, images at 10 Hz.
- 1.2 Storage: one `.npz` per episode + a manifest; normalisation statistics
      computed over the train split only.
- 1.3 `dataset.py` — chunked `torch.utils.data.Dataset` (action chunks with
      end-of-episode padding + a pad mask), deterministic train/val split by
      *seed*, never by frame.
- **Gate**: ≥ 50 demonstrations per task; every episode passes integrity checks
  (no NaNs, monotone time, image dtype/shape, action within stated bounds);
  train/val split leaks no seed; a rendered contact sheet shows the
  randomisation actually varies.
- **Evidence**: `media/m1_dataset_grid.png`, printed dataset table.

### M2 — Model

- 2.1 `text_encoder.py` — frozen CLIP text tower, cached, with an
      **instruction bank** baked into the checkpoint so evaluation needs neither
      `transformers` nor the network (upstream's design, kept).
- 2.2 `act_policy.py` — ACT adapted: ResNet18 with only `layer4` trainable,
      spatial vision tokens with the token count **derived from the feature map**
      (upstream hardcodes 49 for 224 px; at 128 px it is 16), text or task-id
      conditioning, both action heads, normalisation buffers inside the module.
- **Gate**: parameter count reported; forward/backward shape tests;
  overfit-one-batch drives loss below a stated floor; text conditioning changes
  the output for a fixed observation (a *necessary* condition for M4's
  instruction-swap test to be meaningful); checkpoint round-trips.
- **Evidence**: printed model table, `tests/test_act_policy.py` green.

### M3 — Training

- 3.1 `train.py` — L1 chunk loss with pad masking, AdamW, cosine schedule,
      val-on-seed-holdout, checkpointing, resumable.
- 3.2 Train the primary (`task`, text-conditioned) policy to convergence.
- 3.3 Train the two ablations: `task_id` conditioning, and the `joint` head.
- **Gate**: validation action error below a stated per-dimension bar and clearly
  better than a predict-the-mean baseline; loss curves recorded; the three runs'
  numbers tabulated together.
- **Evidence**: `media/m3_training_curves.png`, printed comparison table.

### M4 — Closed-loop evaluation

The milestone that decides whether the lab has a result.

- 4.1 `evaluate.py` — closed-loop rollouts: policy at 10 Hz → Lab 8 QP at 1 kHz,
      automatic per-task success detection on **simulated** state.
- 4.2 Seen configurations, then position-randomised (wider than training).
- 4.3 The **instruction-swap** test and the **joint-head** ablation.
- **Gate**: > 70 % on seen configurations, > 40 % on randomised ones (the
  brief's numbers); the instruction swap changes behaviour; every rate is
  reported with its episode count.
- **Evidence**: `media/m4_success_rates.png`, per-episode CSV, failure taxonomy.

### M5 — Capstone demo and inference profiling

- 5.1 `capstone_demo.py` — free-form language in, autonomous episode out,
      recorded.
- 5.2 Inference profiling: per-call latency and achievable control rate,
      float32 and dynamically quantised.
- **Gate**: the capstone runs from a typed instruction with no task index
  anywhere on the path; inference > 10 Hz measured; the recorded episode's
  success asserted on simulated state.
- **Evidence**: `media/m5_capstone.mp4`, printed latency table.

### M6 — Documentation and blog

- `docs/ARCHITECTURE.md` + `docs/CODE_WALKTHROUGH.md`,
  `docs-turkish/ARCHITECTURE_TR.md`, `blog/` post covering the whole nine-lab
  arc, README with per-milestone evidence, project status boards.
- **Gate**: all four documents exist, the README carries every milestone's gate
  table, and the deviations above appear in the README rather than only here.

---

## Risks, and what each one costs

| Risk | Signal it is happening | Response |
|---|---|---|
| Expert success collapses under randomisation | M0 gate < 70 % | Narrow the randomisation range and *report the range*; the expert is Lab 8's, tuned for one configuration |
| CPU training is too slow to converge | M3 epoch time ≫ plan | 96 px, fewer demos, smaller decoder — in that order; image size is the cheapest lever |
| BC policy cannot close the loop at all | M4 success ≈ 0 on both heads | Report it. A negative result with a measured cause is this project's normal output (Lab 7 M4, Lab 8 M4-reach) |
| Language ignored | Instruction swap changes nothing | Report it as the headline; it is the exact failure the two-object scene was built to detect |
| CLIP unreachable at some later point | `transformers` download fails | The instruction bank is baked into the checkpoint at M2 — evaluation never needs the network |

## Success criteria (from the brief), mapped

| Brief criterion | Milestone | Note |
|---|---|---|
| 50+ demos per task from a programmatic expert | M1 | Expert is Lab 8's whole-body controller |
| ACT trains with language conditioning | M2, M3 | Frozen CLIP text tower |
| > 70 % on training configurations | M4 | |
| > 40 % on position-randomised variants | M4 | |
| Real-time inference > 10 Hz | M5 | CPU + dynamic quantisation, not RTX INT8 |
| Capstone "pick up the red cup" end to end | M5 | Two-object scene makes the sentence load-bearing |
| `LAB_09.md` complete | M6 | |
| Blog post covering the series arc | M6 | Written in-milestone (Lab 8 L-M6-a) |
