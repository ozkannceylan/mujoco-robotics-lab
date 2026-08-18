# The Hardest Part of Building a VLA Was the Robot That Wasn't Learning

*Lab 9 of a robotics portfolio series using MuJoCo and Pinocchio — and the last one*

---

Nine labs ago I started with a two-link arm drawing a square. The plan, written
at the beginning, ended here: a humanoid that takes a sentence — *"pick up the
red cup"* — and does it. Vision in, language in, actions out, no hand-coded task
logic anywhere on the path.

This post is about getting there, and about the thing that turned out to be
hard. It was not the transformer. It was not the language conditioning. It was
that the *expert* — the hand-written controller whose demonstrations the policy
learns from — turned out to be far more fragile than its own passing test suite
suggested. Most of this lab was spent discovering that, measuring it precisely,
and deciding what to build on what was left.

## What a VLA actually needs from you

The architecture is not the interesting part, so let me get it out of the way.

A frozen CLIP text encoder turns the instruction into a 512-dimensional vector.
An [ACT policy](https://arxiv.org/abs/2304.13705) — action chunking with
transformers — takes that vector, two 128-pixel camera views (one on the torso,
one on the wrist) and a 62-number proprioception vector, and emits twenty future
actions at once, ten times a second. Roughly 15.8 million parameters, of which
13 million train. It is a small, well-understood model, and the design is
adapted from my own earlier
[`humanoid_vla`](https://github.com/ozkannceylan/humanoid_vla) project.

The interesting decisions are all about what the policy is *not* allowed to do.

**It does not output joint angles.** The obvious action space for a robot policy
is "the position every joint should be in", and that is what my own lab brief
specified. On a fixed-base arm it is right. On a walking humanoid it walks
straight into the finding that ended Lab 7 of this series: a joint-position
reference tracked by a PD controller cannot stabilise a floating-base robot.
That is why Lab 7's walking failed after six attempts and why Lab 8 exists.

So the policy emits what Lab 8's whole-body optimiser already consumes — where
the hands should be, whether to take a step, whether to close the grip — and Lab
8's quadratic program runs at 1 kHz underneath, keeping the robot upright.
**Balance is never a learned quantity.** The policy decides what to do; the
optimiser decides how not to fall over.

**It does not know where it is.** The proprioception vector carries joint angles,
joint velocities, pelvis height, roll, pitch and a grip bit. It deliberately
omits the pelvis's world x, y and yaw.

This one is worth dwelling on, because it is the difference between an
evaluation that measures something and one that does not. Hand a policy its own
world coordinates and it can solve every task in this lab by dead reckoning:
walk until x exceeds 0.25, reach to a fixed offset. It never has to look at an
image or read its instruction, and it will post a beautiful success rate having
learned neither. Everything it knows about where it is has to arrive through the
pixels. What remains — encoders and an IMU — is what a real robot actually
observes.

**The scene has two objects, and the sentence has to choose.** A red cup and a
blue box stand on the table; which one is nearer is randomised. With a single
object, a policy conditioned on task labels can infer the task from the robot's
own posture — walking and reaching look nothing alike — and ignore the language
entirely. With two, the same image demands different actions under different
instructions. It also means the walk itself is language-conditioned: how far to
walk depends on which object you were asked for.

Decide what the evaluation must be *unable* to pass before you collect a single
frame. Afterwards, fixing it costs a full re-collection.

## The expert was a hypothesis, and I had assumed it was a given

Lab 8 ended with a capstone I was pleased with: the humanoid walks to a
pedestal, picks up a payload, brings it to its chest, secures it with the second
hand, carries it, and sets it down 11.8 millimetres from the target. Gate passed
4/4. Video and all.

Lab 9's plan assumed that controller would produce demonstrations. Walk, pick,
carry, place — four tasks, fifty demonstrations each, straight out of Lab 8.

I ran it over a randomised two-object scene. **1 out of 8.**

Not a bug. Lab 8's own write-up contains the explanation, and I had written it
myself two days earlier: *a result that a no-op perturbation destroys is a draw
from a distribution, not a property of the controller.* Lab 8 said that about one
of its own sub-tasks. Randomising where an object sits is exactly such a
perturbation, and the capstone had been tuned — honestly, but tightly — to one
configuration.

The right move was not to keep tuning until a seed worked. It was to measure
each task separately and find out what the expert could actually do:

| task | measured | why |
|---|---|---|
| walk + pick | **40/40** | — |
| carry (walk holding the object) | 1/12 | see below |
| place (set it on a marker) | 5/10 | see below |

**The carry** asks both hands to hold the load at the chest mid-line. Lab 8
computes the second hand's target by mirroring the first about the payload, and
at this grasp offset the two wrist targets come out **22 to 35 millimetres
apart** — the controller is being asked to put both wrists in nearly the same
place. It worked in Lab 8 because Lab 8's specific grasp geometry happened to
produce a workable spread.

**The place** fails for a reason I find genuinely instructive. Release accuracy
was fine: 6 to 16 millimetres. Final position was not: 58 to 127. Tracing it, the
object is held at whatever tilt the wrist happens to have — 22 degrees, measured
— and released 12 millimetres above the surface, because the hand task has not
converged. It lands on an edge and rolls 84 mm. The hand tasks in Lab 8's stack
control *position only*. The object's orientation is an uncontrolled output.

I tried commanding the target *through* the surface so contact would stop it.
Worse: the stored compliance kicks the object when the weld opens.

So: two tasks, not four or five. Walk and pick, with an expert that succeeds
100% of the time across randomisation. Restoring "place" needs a hand-orientation
task, which is a change to Lab 8's controller, and I wrote it down as Lab 8 work
rather than smuggling a half-working task into a demonstration set.

A demonstration set inherits its expert's failure rate. Half a dataset of falls
teaches a policy to fall, and no amount of model work recovers from that.

## Three measurements that changed the design

**A frozen balance reference has a shelf life, and it is about six seconds.**

Lab 8's standing controller freezes its balance target — the "divergent component
of motion", the part of the centre-of-mass state that can run away — at the value
it had when the phase began. Fine for a short motion. Lab 9 has no carry-walk, so
its entire manipulation happens in one continuous stand, and at Lab 8's timings
that is 11.5 seconds.

Zero out of four episodes survived. The signature was unmistakable once I plotted
it: the balance error growing *exponentially*, doubling every 0.15 seconds from
4.5 millimetres, while the hand still tracked to 5 mm and peak torque sat at
21 N·m out of 139. Not a saturation. An instability. Moving an arm shifts the
centre of mass, and the frozen target commands the robot back toward a snapshot
that no longer describes a resting configuration.

| continuous standing | episodes completed |
|---|---|
| 11.5 s (Lab 8's timings) | 0 / 4 |
| 6.9 s | 3 / 4 |
| 5.2 s | 4 / 4 |

Lab 8 never hit this because it *walked* between manipulation phases, and walking
replans the whole balance trajectory. Its gates never had to state an operating
limit that its own sequence never approached.

My first fix was to split the motion into short segments, each re-anchoring the
reference. It made things **worse** — the robot fell three seconds earlier.
Re-freezing repeatedly does not restore the feedback; it replaces it with a fresh
snapshot of the drift. The actual fix was to make the whole sequence fit inside
the budget, and a unit test now asserts it.

**Reach accuracy was the wrong quantity to tune on.**

I needed to choose how far in front of the object the robot should stop. I swept
it and measured reach error: flat at 7–11 mm anywhere from −0.01 m to 0.37 m. So
0.22 m looked as good as anything.

At 0.22 m the arm is extended 0.43 m from the pelvis. The reach is fine there.
The **lift** is not — half a kilogram at that extension saturates the waist and
the robot goes down at the end of the lift, not during the reach. Lab 8's own
capstone stood 0.06 m from its payload, which makes the reach almost entirely
lateral with the arm folded. Tune a parameter on the quantity that fails, not on
the one that is easy to measure.

**A borrowed constant hides the object it was measured on.**

Lab 8's grasp offset is a fixed −0.060 m: its payload's 0.030 m half-extent plus
a 0.030 m wrist clearance. Applied unchanged to a 0.040 m radius cup, it puts the
wrist *inside* the object. Every single failure at 90% expert success was a near
cup, reaching to 29–30 mm where the box reached to 7–11 from the identical
controller. Scaling the offset by the target's own half-extent took the gate from
36/40 to **40/40**.

Constants carry their source's geometry inside them. Re-derive them from the
quantity they were really about.

## The bug that looked like a broken architecture

Before training anything, I run an overfit-one-batch check: can the model
memorise eight samples? If it cannot, something is wired wrong, and finding that
in a minute beats finding it after an epoch.

It failed. The loss plateaued at *exactly* the level a constant predictor
achieves — the classic signature of a model that cannot distinguish its inputs.
I started reading the token assembly code looking for a broadcasting mistake.

There wasn't one. Sweeping the learning rate first would have told me in two
minutes: ratio 0.19 at 1e-3, 0.17 at 3e-4, 0.15 at 1e-4, and a plateau at 3e-3.
**My check's optimiser was destabilising the transformer.** The model was fine.

Two things came out of it. The check runs at a sane learning rate now. And it
scores against the best *constant* predictor rather than against the initial
loss, because with standard-normal targets the constant predictor already scores
0.76 — a ratio to the starting loss cannot tell "memorised the batch" from
"learned its mean".

Before suspecting the model, check that the thing measuring it isn't the thing
that's broken.

## Training on four CPU cores and no GPU

The lab brief said "cloud GPU: Lambda Labs or RunPod" and "INT8 for local
inference on an RTX 4050". I had neither. Four cores, fifteen gigabytes, no CUDA
device.

The measurement that mattered most was one I nearly didn't take. MuJoCo's
offscreen rendering here goes through a software OpenGL stack, and it costs about
**97 milliseconds per frame regardless of resolution** — 64 pixels and 224 pixels
cost the same, because the expense is per-geometry setup rather than fill. The
instinctive optimisation, shrinking the image, buys nothing on the data side.
Turning off shadows, reflections and the skybox buys a factor of four.

Every number in the plan came from ten minutes of benchmarking before the plan
was written: 128-pixel images (free on the collection side, 3.7× cheaper on the
training side), one expert rollout per seed sliced into several labelled task
segments, a dataset sized to what four cores can chew through. Had I planned
against assumed hardware, milestone one would have discovered a ten-hour data
collection.

240 demonstrations, 12,180 frames, 244 MB, 38 minutes. Then roughly four minutes
per training epoch.

One detail in the dataset that I want to underline because it is so easy to get
wrong and so flattering when you do: **the train/validation split is by scene
seed, never by frame.** Two frames a tenth of a second apart in the same episode
are near-duplicates — same lighting, same object placement, the robot a few
millimetres along. Split by frame and your validation loss measures memorisation.
It will look excellent.

## What it learned, and what it didn't

The policy trains cleanly. Twenty-four epochs, 110 minutes, validation error at
**0.11×** what predicting the training mean would score, hand-target error
**4.1 mm**. Every curve is the shape you want.

In closed loop it walks to the object and stops in the right place — and then
does not pick anything up.

| condition | walk | pick |
|---|---|---|
| seen configurations | 3/6 (50%) | 0/6 |
| position-randomised (wider than training) | 3/6 (50%) | 0/6 |
| held-out paraphrases | 3/6 (50%) | 0/6 |

The lab's gate wanted >70% on seen and >40% on randomised. It failed. What makes
the failure worth the nine labs is that it is *legible* — I know precisely why,
in two separate mechanisms.

### Fifty percent is not a coincidence

Look at the walk column. Exactly 50%, in all three conditions.

The robot stops at the **nearer** object's distance regardless of which object
the instruction names, and which object is nearer is randomised 50/50. It is
scoring chance on a binary choice.

I checked it directly, feeding one stored observation through the policy twice
with the two different instructions:

| quantity | difference between "the red cup" and "the blue box" |
|---|---|
| right-hand target | **0.3 mm** |
| gait command | **0.0018** |

The language conditioning contributes essentially nothing. The paired closed-loop
test says the same thing in metres: the two instructions should put the robot
0.159 m apart, and they put it **0.000 m** apart.

Here is the part I did not see coming. I designed the two-object scene at
milestone zero *specifically* so language would be necessary, and wrote a note
to myself that this was the one thing to get right before collecting data. The
scene does make language necessary. **The demonstrations don't.**

The expert walks until the named object is the one in front of it. So by the
time the reach begins, "reach for the nearest object" is the correct action in
every single training frame — the instruction is redundant given the state. And
during the walk, the instruction only discriminates for the handful of frames
around the stop; everywhere else both instructions want the same thing.

Behaviour cloning takes the cheap route, as it should. The shortcut is available
because the *expert's own competence* removed the ambiguity that the language was
supposed to resolve. Two objects in the scene is a necessary condition and not a
sufficient one: what you need is demonstration *states* where the correct action
differs under the two instructions and the state does not reveal which one is in
force.

That is a data-collection fix, not a training one, and it is the first thing I
would change.

### The reach converges, and then stops converging

The pick fails differently, and the trace is worth reading:

```
poll  0:  hand 188 mm from the cup
poll 24:  hand 102 mm
poll 36:  hand  84 mm
poll 48:  hand  84 mm
poll 69:  hand  83 mm     grasp gate is 70 mm; the expert reaches 15 mm
```

It is not inert. It tracks the reach for three and a half seconds, closing a
hundred millimetres. Then it stops, twelve millimetres short of the gate, and
holds there for the remaining thirty-five polls.

A hand hovering 83 mm from an object never occurs in a demonstration — the
expert's reach is smooth and fast and goes straight through that distance in a
tenth of a second. Off the demonstration manifold, the policy's commanded target
collapses onto its own current hand position, and acting on that keeps it exactly
where it is. It is the same absorbing state that bit me during labelling, arrived
at from a different direction.

The reason it ends up off-manifold at all is that each command moves the hand
about two thirds as far as the expert's did. Under-commit slightly, every step,
and you drift somewhere the expert never was.

### What did work

The walking half of this is genuinely good, and worth separating from the rest.
The policy decides *when to stop* from vision, and when it is right it is very
right — stopping error **0.001 m** against a target it has to infer from a
128-pixel image. Getting there took two protocol corrections, and the second is
the more interesting.

An action-chunking policy predicts twenty future actions. I was reading the first
one. Two frames before the expert stops, the true chunk is `[0, 0, …]` and the
prediction is `[0.99, 0.99, 0.99, 0.00, 0.00, …]` — the stop is *in there*, placed
about nine steps late. The head of the chunk is where a rare transition is
rarest, so that is exactly where the model hedges it. Reading the chunk's mean
instead — "what fraction of the next two seconds do I expect to be walking" —
took stopping error from 0.21 m to 0.001 m without touching the weights.

An action-chunking policy predicts a *plan*. Reading only its first action throws
away the part that says when the current behaviour ends.

Inference runs at **37 Hz** on four CPU cores, 38 Hz dynamically quantised. The
brief asked for 10 Hz on a GPU with INT8. The control loop is limited by
software rendering at 97 ms a frame, not by the network.


## What nine labs taught me

Looking back across the series, the things that transferred were not the
algorithms.

**Measure the thing you are about to build on.** Lab 9's plan assumed Lab 8's
controller worked because Lab 8's gate said so. The gate was true and the
assumption was false, and the gap between them cost most of a lab. A passing test
is a statement about the conditions it ran under.

**Design the evaluation to be failable before you collect the data.** Two
objects instead of one; no world coordinates in the state vector. Both were
cheap to decide up front and would have been expensive to retrofit — and without
them, every success rate in this post would have been a number about the scene
rather than about the policy.

**Write down what you cut, and why, with the numbers.** This lab ships two tasks
where the brief asked for three to five. That is a smaller result, and the
honest version of it is more useful than a bigger result with a quiet failure
inside. The carry doesn't work because two wrist targets end up 22 mm apart. The
place doesn't work because position-only hand tasks leave orientation
uncontrolled. Those sentences are worth more than a success rate.

**The interesting failures are almost never in the part you were working on.**
The walking controller was fine; the foot model was a guess. The transformer was
fine; the test's learning rate was wrong. The policy architecture was fine; the
expert was fragile. Every time, the instinct was to debug the thing I had just
written.

---

*Code, gates and full measurement tables:
[`lab-9-vla-integration/`](../README.md). Every failure summarised here has a
long-form entry in [`tasks/LESSONS.md`](../tasks/LESSONS.md). The series starts
at [Lab 1](../../lab-1-2link-arm/).*
