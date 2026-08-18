# Lab 9 — Lessons

> Live journal. Symptom / Root cause / Fix / Takeaway, logged as it happens.

## Pre-M0 (2026-08-17)

#### L-P0-a: Probe the machine before writing the plan, not the milestone
- Every number in `tasks/PLAN.md` traces to a measurement taken before the plan
  existed: no CUDA device, 97 ms/frame software rendering, 117 ResNet samples/s
  at 128 px, 53.6 s per expert episode.
- The one that changed the design most was **rendering cost being
  resolution-independent** — 64 px and 224 px both cost ~380 ms with the default
  visual flags, because the cost is per-geometry setup in llvmpipe, not fill.
  The instinctive optimisation (shrink the image) buys nothing on the data side;
  turning off shadows, reflection and the skybox buys 4×.
- Had the plan been written first, it would have specified 224 px images and a
  cloud GPU, and M1 would have discovered a 10-hour data collection.
- **Takeaway**: a plan written against assumed hardware is a plan for a machine
  you do not have. Ten minutes of benchmarking bought the whole milestone
  structure.

#### L-P0-b: Check whether the thing the brief asks you to build already exists
- `plan/LAB_09.md` says to *extend* `humanoid_vla` by adding language
  conditioning, describing it as fixed-task. Reading the upstream source first
  showed `models/text_encoder.py` (frozen CLIP, instruction bank baked into the
  checkpoint) and `ACTPolicy(conditioning="text", vision_tokens="spatial")`
  already shipped.
- Building it again would have produced a lab whose headline contribution was a
  re-implementation of someone else's finished work — and would have crowded out
  the brief's *other* bullet, the one the master plan calls the critical path:
  the expert is Labs 3–8, so the policy is trained on a **walking** humanoid.
- **Takeaway**: a brief describes the world on the day it was written. Read the
  dependency's actual source before scoping against its description, and re-aim
  the lab at whatever is genuinely still missing.

#### L-P0-c: Design the evaluation so the shortcut is impossible, before collecting data
- The naive setup — four task labels, one object — can be solved without ever
  reading the instruction: the robot's own pose says whether it is walking or
  reaching. A success rate measured that way is a number about the *scene*, not
  about language.
- The same hazard sits in the observation vector. Handing the policy its base
  x/y/yaw lets it dead-reckon the whole episode and ignore the camera too.
- Both were designed out before any code: two objects the instruction must
  choose between, and proprioception restricted to what an IMU and joint
  encoders could actually observe.
- **Takeaway**: decide what the evaluation must be unable to pass *before* the
  dataset exists. Afterwards the fix costs a full re-collection.

---

## M0 — Scene, cameras, observation/action contract (2026-08-17)

#### L-M0-a: An expert inherited from another lab is a hypothesis, not a given
- The plan assumed Lab 8's capstone controller would generate demonstrations.
  Lab 8's own gate is 4/4 on **one** configuration; run over a randomised
  two-object scene the same sequence scored **1/8**.
- The gap is not a bug in either lab. Lab 8 tuned a single trajectory to the
  edge of what its controller can do, which its own M4 write-up says plainly
  (L-M4-f: a result a no-op perturbation destroys is a draw from a
  distribution). Randomising object placement is exactly such a perturbation.
- **Fix**: measure the expert *before* building anything on top of it, and
  scope the task set to what it can actually demonstrate.
- **Takeaway**: a demonstration set inherits its expert's failure rate. Half a
  dataset of falls teaches a policy to fall, and no amount of model work
  recovers from it. Screen the expert first.

#### L-M0-b: Reach accuracy is the wrong quantity to tune a stopping distance on
- An early sweep chose the standing distance by reach error and found it flat
  at 7-11 mm for standoffs from -0.01 m to 0.37 m, so 0.22 m looked as good as
  anything.
- At 0.22 m the arm is extended ~0.43 m from the pelvis. The *reach* is fine
  there; the **lift** is not — half a kilogram at that extension saturates
  `waist_roll` and the robot goes down at the end of the lift, not during the
  reach. Lab 8's own capstone stood 0.06 m from its payload, which makes the
  reach almost entirely lateral with the arm folded.
- **Fix**: standoff 0.07 m, and the stopping point aimed at the midpoint of the
  object and the drop marker rather than at the object.
- **Takeaway**: tune a parameter on the quantity that fails, not on the one
  that is easy to measure. The reach was never the binding constraint.

#### L-M0-c: The two-handed carry does not survive leaving its configuration
- Lab 8's carry — tuck the load to the chest mid-line, close a second weld,
  walk — is the lab's most delicate phase, and it does not transfer.
- Measured over seeds and objects: 1/12 with the tuck, 1/6 with the tuck
  removed and the load simply held where the lift left it. Several failures are
  in the tuck itself, before any walking.
- One mechanism is visible in the geometry: `carry_targets` mirrors the right
  hand's grip about the payload, and when the grasp offset is near the object's
  sagittal plane the two wrist targets come out **22-35 mm apart** — the
  controller is asked to put both wrists in nearly the same place.
- **Fix**: no `carry` task. Documented with its numbers rather than dropped
  quietly.
- **Takeaway**: when a downstream lab cannot reproduce an upstream result off
  its tuned point, the honest move is to report the transfer measurement, not
  to keep tuning until one seed works.

#### L-M0-d: A frozen balance reference has a shelf life, and it is about six seconds
- Lab 8's `stand` calls `_freeze_balance`, which pins the DCM target at the
  divergent component's value at that instant. Lab 9's sequence has no
  carry-walk, so its whole manipulation happens in one continuous stand — 11.5 s
  at Lab 8's timings, where Lab 8 never stood for more than ~7 s before walking
  and replanning.
- The failure is unmistakable once traced: the DCM error grows **exponentially
  at the LIPM rate** (doubling every ~0.15 s) from 4.5 mm, while the hand still
  tracks to 5 mm and peak torque sits at 21 N·m. Saturation only appears
  afterwards, on the way down. An instability, not a limit.
- Measured budget over four configurations: **11.5 s → 0/4 complete; 6.9 s →
  1/4 fall; 5.2 s → 4/4 complete**, with what remains being accuracy rather
  than balance.
- Splitting the motion into short re-anchored segments was tried and measured
  **worse** (falls 3 s earlier): re-freezing repeatedly removes the feedback
  that was correcting the drift.
- **Fix**: Lab 9 phase durations sized to a 5.6 s budget, asserted in a test.
- **Takeaway**: an inherited controller carries operating limits its own gates
  never had to state, because its own sequence never approached them.

#### L-M0-e: Position-only hand tasks cannot place an object
- The `place` task reached 5/10 at best. The release accuracy was good (6-16 mm)
  and the *final* position was not (58-127 mm).
- Traced: the object is held at whatever tilt the wrist has — 22° measured — and
  released 12 mm above the surface because the hand task has not converged.
  It lands on an edge and rolls 84 mm. Commanding *through* the surface so
  contact stops it made things worse: the stored compliance kicks the object
  when the weld opens.
- The task stack has frame *position* tasks and no orientation task on the
  hands, so the object's attitude is an uncontrolled output.
- **Fix**: no `place` task at this scope. It needs a hand-orientation task,
  which is a controller change and therefore Lab 8 work.
- **Takeaway**: "grasp stays simple" (a weld, no fingers) is fine for picking
  something up and not sufficient for putting it down. Placing is an
  orientation problem.

#### L-M0-f: A constant borrowed from another lab hides the object it was measured on
- Lab 8's `GRASP_OFFSET` is a fixed −0.060 m in x: its payload's 0.030 m
  half-extent plus a 0.030 m wrist clearance. Applied unchanged to a 0.040 m
  radius cup it puts the wrist *inside* the object's footprint.
- Every one of M0's four failures at 90 % was a near cup, reaching to 29-30 mm
  where the box reached to 7-11 mm from the identical controller.
- **Fix**: scale the offset by the target's own half-extent — same surface
  clearance for both. 36/40 → **40/40**.
- **Takeaway**: a borrowed constant carries its source's geometry inside it.
  Re-derive it from the quantity it was really about.

#### M0 gate

| criterion | result | measured |
|---|---|---|
| Expert success rate ≥ 70 % | PASS | **100 % (40/40)**, 20 seeds × 2 objects |
| Both cameras render at 128 px | PASS | head + wrist, non-degenerate |
| Action codec round-trip exact | PASS | 5.9e-08 |
| State matches declared dimension | PASS | 62 |
| Approach depends on the named object | PASS | 2 or 4 steps by target |
| Torques within limits on success | PASS | 92.6 N·m peak (limit 139) |

reach error 15.2 ± 7.3 mm (max 27.0) · lift height 90 ± 6 mm

Evidence: `media/m0_scene.png`, `media/m0_expert_rollout.mp4`, `media/m0_gate.json`.

---

## M1 — Demonstration dataset (2026-08-18)

#### L-M1-a: Slice the episode, do not re-simulate the task
- Rendering dominates collection on this machine — ~97 ms per frame, twice per
  captured tick for two cameras — so an episode is expensive and a task is not.
- Every expert rollout is captured once at 10 Hz and **sliced by phase** into its
  labelled task segments, so 120 simulations yielded 240 demonstrations
  (120 `walk`, 120 `pick`) and 12,180 frames in 38 minutes on 4 cores.
- **Takeaway**: when the cost is in the rollout rather than the task, the unit of
  work is the rollout. Label the segments afterwards.

#### L-M1-b: Split by scene seed, never by frame
- Two captured frames 100 ms apart in the same episode are near-duplicates: the
  robot has moved a few millimetres and the lighting, object colours and object
  placement are *identical*.
- A frame-level train/validation split therefore reports a validation loss that
  measures memorisation, and it looks excellent — which is exactly why it is
  dangerous. The split here is by scene seed (48 train / 12 validation), and
  `build_datasets` raises if the two sets ever intersect.
- **Takeaway**: the unit of independence in a demonstration set is the episode's
  *scene*, not its frames.

#### L-M1-c: Store the expert's command, not the achieved state
- Behaviour cloning imitates what the expert *did*. On this system the two are
  not the same: the grasp weld is compliant by design, the balance controller is
  continuously correcting, and the achieved hand position lags the commanded one.
- Training on the achieved state teaches the policy to reproduce its own past
  rather than to act. The recorded action is the effective target of each hand
  task at that tick — and for a hand whose task is *disabled*, its current
  position, because "leave it where it is" is the honest label for a limb the
  expert is not commanding.
- **Takeaway**: the label is the controller's input, not its output.

#### L-M1-d: Write only successful episodes
- A failed episode is a recording of a robot falling over, and its frames are
  indistinguishable from a good episode's until the moment it goes down. Half of
  such a set would teach the policy the failure.
- 120 of 120 attempts succeeded, which is what M0's expert gate bought.

## M2 — Model (2026-08-18)

#### L-M2-a: A failed overfit-one-batch check can be the check's fault
- The model plateaued at exactly the constant-predictor level on eight samples.
  That reads as an architecture bug — the network cannot distinguish its inputs
  — and it was tempting to go looking for one in the token assembly.
- Sweeping the learning rate first: ratio 0.19 at 1e-3, 0.17 at 3e-4, 0.15 at
  1e-4, and a plateau at 3e-3. **The optimiser was destabilising the
  transformer**, and the architecture was fine.
- Two changes came out of it. The check runs at 1e-3, and it scores against the
  best *constant* predictor rather than against the initial loss: with N(0,1)
  targets the constant predictor already scores 0.76, so a ratio to the starting
  loss cannot distinguish "memorised the batch" from "learned its mean".
- **Takeaway**: before suspecting the model, check that the thing measuring it is
  not the thing that is broken. And pick a baseline the metric can actually
  separate from failure.

#### L-M2-b: Check the conditioning separates *meanings*, not just strings
- The obvious diagnostic — mean pairwise cosine similarity of the instruction
  bank — reads 0.866 and says nothing useful, because every sentence in this
  vocabulary shares most of its structure.
- The quantity that matters is the *contrast*: paraphrases of the same command sit
  at 0.957, commands that mean different things at 0.846, a margin of 0.111.
  Paraphrase robustness and instruction separability pull in opposite directions,
  and a policy cannot follow instructions its conditioning cannot tell apart.
- Both are checked before training, not inferred from a bad success rate after.
- **Takeaway**: a single aggregate over a similarity matrix hides the structure
  the model actually needs. Compare within-group against across-group.

#### M1 gate

| criterion | result | measured |
|---|---|---|
| ≥ 50 demonstrations per task | PASS | **120 per task** (240 total) |
| Every attempted episode succeeded | PASS | 120 / 120 |
| Integrity checks | PASS | no NaNs, shapes and dtypes as declared |
| Train/val split leaks no seed | PASS | 48 / 12, intersection empty |
| Randomisation visibly varies | PASS | `media/m1_dataset_grid.png` |

12,180 frames · 7,860 labelled `walk`, 4,320 `pick` · 244 MB · 38.4 min on 4 cores

#### M2 gate

| criterion | result | measured |
|---|---|---|
| Parameter count reported | PASS | 15.75 M total, 12.96 M trainable |
| Token count derived from image size | PASS | 16 tokens/camera at 128 px |
| Instruction changes the action | PASS | max delta 0.0020 |
| Same instruction is deterministic | PASS | 0.0 |
| Meanings separate further than paraphrases | PASS | margin 0.111 |
| Overfits one batch | PASS | 0.250 × the constant-predictor baseline |
| Checkpoint round-trips predictions | PASS | 0.0 |
