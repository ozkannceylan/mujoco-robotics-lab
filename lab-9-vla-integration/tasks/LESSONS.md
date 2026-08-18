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

## M3 — Training (2026-08-18)

#### L-M3-a: A label that is a function of the instruction teaches nothing
- The first training run reached a validation error 0.099x the predict-the-mean
  baseline, with 5.0 mm hand-target error. Every number looked excellent.
- In closed loop the policy walked to the 6-unit cap on **every** episode and
  never stopped. It was not broken: the demonstrations' `walk` segment ended
  where the expert stopped, and every frame in it carried `gait = 1`. The stop
  itself was labelled `pick`. So the gait bit was a pure function of the
  instruction label, the policy learned exactly that, and *when to stop* was
  never a decision anywhere in the training signal.
- **Fix**: move the stop phase into the `walk` task, so a walk demonstration ends
  with the robot standing and the transition has to be predicted from vision.
- **Takeaway**: check that each output dimension varies *within* an instruction,
  not just across instructions. A validation loss cannot see this — the label was
  perfectly predictable and the model predicted it perfectly.

#### L-M3-b: Idle frames make a behaviour-cloning policy a fixed point
- The `pick` policy predicted the hand target as its own current hand position,
  for 25 consecutive polls, 188 mm from the object. It never started reaching.
- The `pick` segment began at the expert's *stop* phase, where the hand tasks are
  disabled and the recorded action is therefore "leave the hand where it is". And
  the observation of a stopped robot with a resting arm is identical whether it
  has been standing for 0.1 s or 5 s — nothing observable says the reach should
  begin. "Stay put" is a self-fulfilling prediction: acting on it reproduces the
  observation that produced it.
- **Fix**: start the `pick` segment at the reach. Every frame in it then has a
  hand that is moving somewhere.
- **Takeaway**: an action that reproduces its own observation is an absorbing
  state for behaviour cloning. Do not label one unless the policy is supposed to
  stay there.

#### L-M3-c: Store the raw labels, not just the derived ones
- Both fixes above are relabellings of the same 12,180 frames. Because each
  episode stores its **per-frame phase** alongside the derived task segments,
  fixing them cost a `--reslice` pass over the manifest instead of 40 minutes of
  re-simulation.
- **Takeaway**: persist the primitive a label was derived from. The derivation is
  the part most likely to be wrong.

#### L-M3-d: A chunk predictor hedges a rare, sharp transition at the chunk head
- After the relabelling, the policy still walked past the objects. Reading its
  *whole* predicted chunk rather than its first action explained why. Two frames
  before the expert stops, the true chunk is `[0, 0, …]` and the prediction is
  `[0.99, 0.99, 0.99, 0.00, 0.00, …]` — the stop is there, placed about nine
  steps late.
- The head of the chunk is where the transition is rarest: only the handful of
  frames straddling the stop have it at index 0, while every frame within two
  seconds of the stop has it *somewhere*. The model learns where it is well and
  when it is imminent poorly.
- **Fix**: decide from the chunk's mean — "the fraction of the next two seconds I
  expect to still be walking" — instead of from its first entry. Stopping
  accuracy on near-object episodes went from 0.21 m of error to **0.001 m**.
- **Takeaway**: an action-chunking policy predicts a *plan*. Reading only the
  first action throws away most of it, and specifically throws away the part that
  says when the current behaviour ends.

## M4 — Closed-loop evaluation (2026-08-18)

#### L-M4-a: The policy ignores the instruction, and the demonstrations are why
- Measured directly on stored validation frames, feeding the identical
  observation with each of the two instructions:

  | quantity | difference between "red cup" and "blue box" |
  |---|---|
  | right-hand target | **0.3 mm** |
  | gait command | **0.0018** |

  The language conditioning contributes essentially nothing. In closed loop the
  robot walks to the *nearer* object's stopping distance under either
  instruction, which is correct half the time by construction.
- The two-object scene was designed at M0 specifically to make language
  necessary, and it does — *in principle*. What makes it unnecessary in practice
  is the **expert's own behaviour**: the expert walks until the named object is
  the one in front of it, so by the time the `pick` segment begins, "reach for
  the nearest object" is the correct action in every training frame. And in the
  `walk` segment the instruction only discriminates during the handful of frames
  around the stop; everywhere else both instructions want `gait = 1`.
- So the shortcut is available and cheap, and behaviour cloning takes it. This is
  causal confusion of the ordinary kind: the state already encodes the decision
  that was made at the start of the episode, and predicting the next action from
  the state never requires recovering *why*.
- **Takeaway**: a scene in which two instructions demand different actions is not
  enough. The **demonstrations** have to contain states where the correct action
  differs under the two instructions *and the state does not reveal which one is
  in force*. Ours mostly do not, and no amount of model capacity fixes that.
- The fix is a data-collection change, not a training one: the expert would have
  to be positioned so that both objects are equally reachable at the moment of
  the reach, so the instruction is the only thing that can select the target.
  That is a re-collection, recorded here as the follow-up.

#### L-M4-b: The reach converges and then stops converging
- The `pick` policy is not inert — it tracks the reach. Over one episode the hand
  closes from 188 mm to 84 mm in about 3.5 s, following commanded targets that
  lead it by roughly 10 mm per poll.
- Then it **plateaus at 82.6 mm** and stays there for the remaining 35 polls,
  against a 70 mm grasp gate and an expert that reaches 15.2 ± 7.3 mm. The grasp
  bit never rises above 0.06.
- The mechanism is compounding error, and the plateau is the same absorbing state
  as L-M3-b in a new place: a hand hovering 83 mm from the object never occurs in
  a demonstration, because the expert's reach is smooth and fast and passes
  straight through that distance. Off the demonstration manifold the policy's
  commanded target collapses onto its own current hand position, and acting on
  that keeps it there.
- The expert reaches in 1.3 s; the policy is still short after 7 s. Under-
  committing per step is what opens the gap in the first place — each command
  moves the hand about two thirds as far as the expert's did.
- **Takeaway**: behaviour cloning fails at the *end* of a motion, not the start.
  Nothing in the demonstrations says what to do from a state the expert never
  visited, and the nearest thing the policy knows is "hold".

#### M4 gate — FAILED

| criterion | result | measured |
|---|---|---|
| > 70 % success on seen configurations | **FAIL** | 25 % (3/12) overall; walk 50 % (3/6), pick 0 % (0/6) |
| > 40 % on position-randomised | **FAIL** | 25 % (3/12); walk 50 %, pick 0 % |
| Held-out paraphrases | — | 25 % (3/12), identical to seen — the wording is not the problem |
| Instruction changes the behaviour | **FAIL** | commanded stopping separation 0.159 m, produced **0.000 m** (ratio 0.00, 3 pairs) |
| No falls | partial | 2 falls in 18 pick episodes; 0 in 18 walk episodes |

Walk at exactly 50 % across all three conditions is the signature, not a
coincidence: the policy stops at the *near* object's distance whichever object is
named, and which object is near is randomised 50/50.

#### M5 gate — inference PASSED, task FAILED

| criterion | result | measured |
|---|---|---|
| Free-form sentence in, no task index on the path | PASS | instruction embedded by the frozen tower; nothing else selects behaviour |
| Inference > 10 Hz | **PASS** | **37.0 Hz** float32 (27.1 ms), **38.4 Hz** dynamically quantised (26.0 ms) |
| Episode succeeds on simulated state | **FAIL** | walked to 0.253 m and stopped, never grasped, payload unmoved |
| No fall | PASS | 51 N·m peak of a 139 N·m limit |

Dynamic quantisation buys 4 % here rather than the large factor INT8 gives on a
GPU: the backbone is convolutional and stays float, so only the decoder's linear
layers are quantised. The brief's ">10 Hz on an RTX 4050 with INT8" is met on
four CPU cores without it.

Evidence: `media/m4_success_rates.png`, `media/m4_episodes.csv`,
`media/m5_capstone.mp4`, `media/m5_capstone.json`.

#### Not run: the joint-head ablation

`tasks/PLAN.md` deviation 3 promised to train the brief's literal 29-DOF joint
action space and measure it against Lab 7's prediction that a joint-position
reference cannot stabilise this robot. The code path exists
(`policy_runner.joint_tick`, with Lab 8's standing gains and gravity
compensation so the comparison is about the action space rather than a strawman
controller) and the head trains from the same dataset, but the run was not made:
it is ~50 minutes of training plus ~20 of evaluation, and the primary policy's
own result had already turned into the lab's headline. Recorded as unmeasured
rather than quietly dropped.
