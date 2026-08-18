# Lab 9 — Code Walkthrough

A guided read of the source in dependency order. Each section says what the file
is for, what in it is load-bearing, and the measurement that put it there.

Design rationale: [`ARCHITECTURE.md`](ARCHITECTURE.md).
Every finding in long form: [`../tasks/LESSONS.md`](../tasks/LESSONS.md).

---

## 1. `lab9_common.py` — the contract's constants

Start here. It sets the paths, puts Lab 8's `src/` on `sys.path`, and fixes the
numbers everything else is built on.

```python
_LAB8_SRC: Path = PROJECT_ROOT / "lab-8-loco-manipulation" / "src"
if str(_LAB8_SRC) not in sys.path:
    sys.path.append(str(_LAB8_SRC))
```

`append`, never `insert(0)`. Labs share module names (`standing_controller`,
`capstone_scene`, `record_demo`) and putting a foreign `src/` ahead of this one
silently shadows local modules — Lab 8 lost an afternoon to exactly that.

```python
IMAGE_SIZE: int = 128
STATE_DIM: int = 2 * NU + 4      # 29 q + 29 qd + pelvis z, roll, pitch + grasp
POLICY_HZ: float = 10.0
CHUNK_SIZE: int = 20             # 2 s of lookahead
```

128 px rather than the ACT paper's 224 because offscreen rendering here goes
through software EGL at ~97 ms/frame *regardless of resolution* — the cost is
per-geometry setup, not fill. So a smaller image is free on the collection side
and 3.7× cheaper on the training side.

The instruction vocabulary lives here too, and its shape is deliberate:

```python
TASKS = {
    "walk": "walk to the {object}",
    "pick": "pick up the {object}",
}
OBJECTS = {"cup": "red cup", "box": "blue box"}
```

The walk instruction names the **object**, not the table, because how far to walk
depends on which object was asked for. An instruction that did not name it would
make the task undecidable from the observation.

---

## 2. `vla_scene.py` — two objects, two cameras, four welds

Lab 8's capstone scene with three changes, each of which exists to make a
specific measurement possible.

**Two objects.** With one, a policy can infer the task from the robot's pose and
ignore the language. With two, the same image demands different actions under
different instructions — the only setup in which "follows instructions" is
falsifiable.

**Egocentric cameras.** `MjsCamera` exposes `quat`, not MJCF's `xyaxes`
shortcut, so the frames are assembled explicitly:

```python
def _camera_quat(right, up):
    """A MuJoCo camera looks along its own -z with +y up."""
    x = right / norm(right)
    y = up - x * (x @ up); y /= norm(y)     # re-orthogonalise
    z = np.cross(x, y)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).reshape(9))
```

The head camera is aimed with a wide field of view rather than centred: the line
of sight to the objects swings from (yaw −48°, pitch 53°) where the walk starts
to (−79°, 60°) at the stopping point, so no fixed aim centres both.

**Geometry that is not free to change.** The pedestal stands at y = −0.45 because
at −0.32 its inner face is where the right hip passes, and the identical walking
controller that manages twelve steps on bare ground falls on step four (Lab 8
L-M5-f). It is long in x and narrow in y for the same reason. Objects sit 0.09 m
in from its centre — at 0.06 m they are 30 mm further out laterally, and
*lifting* half a kilogram from there saturates `left_hip_roll` and takes the
robot down mid-lift.

Objects are separated by 0.16 m. At 0.11 m the forearm sweeps the distractor off
the pedestal on the way past — measured, it moved 0.63 m.

---

## 3. `observations.py` — the only place the layout is defined

Everything else goes through these functions, so a layout change is one edit
rather than six.

```python
def build_state(mj_data, grasped):
    joints = mj_data.qpos[7 : 7 + NU]
    velocities = mj_data.qvel[6 : 6 + NU]
    ...
    return concat([joints, velocities, [qpos[2], roll, pitch, grasped]])
```

Note what is absent: the pelvis's world **x, y and yaw**. A policy handed its own
coordinates dead-reckons every task here and ignores both camera and instruction.
`tests/test_scene_and_contract.py::test_state_excludes_base_position` translates
the robot 3 m in x and 2 m in y and asserts the state is unchanged.

```python
pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
```

The clip matters. `asin`'s argument leaves [−1, 1] only on a denormalised
quaternion, and a NaN there would poison a whole demonstration silently.

Actions are a pure encode/decode pair with a round-trip test:

```python
action[0:3] = world_to_pelvis(right_hand, pelvis_position, pelvis_yaw)
```

Yaw-only, not the full pelvis rotation: the pelvis pitches and rolls
continuously while walking, and folding that in would inject gait oscillation
into a quantity the policy is supposed to hold still.

---

## 4. `expert.py` — Lab 8's controller, seeded

`VLAExpert` subclasses Lab 8's `Capstone` and inherits every phase method —
`stand`, `walk`, `payload_goal_to_hand`, `_freeze_balance`, `_step`. Only three
things change: the scene, the target selection, and an observation hook.

```python
def _step(self, controller=None):
    super()._step(controller)
    self._ticks += 1
    if self._capture and self._ticks % POLICY_DECIMATION == 0:
        self._capture_frame()
```

Lab 8's own source is not edited. `Capstone.__init__` hardcodes its scene, so
this class replaces `__init__` and calls the inherited `_settle` and
`_build_stack` — twenty duplicated lines against a Lab 8 regression re-run, and
the ground rules pick the duplication.

Three constants in this file each carry a measurement:

```python
REACH_STANDOFF: float = 0.07
```

Reach *accuracy* is flat at 7–11 mm for standoffs from −0.01 m to 0.37 m, which
is why 0.22 m looked fine. At 0.22 m the arm is extended 0.43 m from the pelvis
and the **lift** — not the reach — saturates the waist and puts the robot down.
Tune a parameter on the quantity that fails.

```python
def approach_steps_for(object_x, marker_x=None, pelvis_x=0.0):
    reference = object_x if marker_x is None else 0.5 * (object_x + marker_x)
```

The robot stops once and does everything from there, so the stopping point is
chosen for the midpoint of the object and the marker, not for the object alone.

```python
def grasp_offset(self):
    clearance = GRASP_OFFSET[0] + PAYLOAD_HALF      # Lab 8's wrist clearance
    return [clearance - self.scene.object_half_x(self.scene.target), ...]
```

Lab 8's `GRASP_OFFSET` is a fixed −0.060 m: its payload's half-extent plus a
wrist clearance. Applied to a 0.040 m radius cup it puts the wrist *inside* the
object. Every one of M0's failures at 90 % was a near cup, reaching to 29–30 mm
where the box reached to 7–11 mm. Scaling by the target's own half-extent took
the gate to **40/40**.

And the phase durations:

```python
T_STOP_L9, T_REACH_L9, T_REACH_SETTLE = 0.55, 1.30, 0.45
T_GRASP_L9, T_LIFT_L9, T_HOLD_L9      = 0.30, 0.60, 0.40
STAND_BUDGET_S = sum(...)             # 5.6 s, asserted in a test
```

Lab 8's timings give 11.5 s of continuous standing and 0/4 episodes. The failure
is not saturation: the DCM error grows exponentially at the LIPM rate from
4.5 mm while the hand still tracks to 5 mm and torque sits at 21 N·m.
`_freeze_balance` pins the divergent-component target at the phase's start, and
an arm motion moves the centre of mass out from under it. Lab 8 never hit this
because it *walked* between phases, which replans.

---

## 5. `collect_demos.py` and `dataset.py` — the demonstration set

One expert rollout per (seed, object), sliced by phase into labelled segments.
Slicing rather than one episode per task matters because rendering is the
expensive part.

Only successful episodes are written:

```python
if not record.success or set(segments) != set(TASK_NAMES):
    summary["written"] = False
    return summary
```

A failed episode is a recording of a robot falling over, and its frames are
indistinguishable from good ones until the moment it goes down.

In `dataset.py`, two details:

```python
chunk[:available] = actions[frame : frame + available]
mask[:available] = 1.0
```

Near the end of a segment there are fewer than `chunk_size` actions left. Without
the mask, the padded tail teaches the policy to freeze two seconds early.

```python
val_seeds  = sorted(shuffled[:n_val])
train_seeds = sorted(shuffled[n_val:])
```

Split by **scene seed**, never by frame. Two frames 100 ms apart are
near-duplicates, so a frame-level split reports a validation loss that measures
memorisation — and it looks excellent. `build_datasets` raises if the two sets
ever intersect.

```python
scale = values.std(axis=0)
return np.where(scale < 1e-4, 1.0, scale)
```

A constant dimension — the grasp bit through a walk segment — has zero spread,
and dividing by it produces `inf` that propagates silently into the loss.

---

## 6. `text_encoder.py` and `act_policy.py` — the model

The text tower is frozen and used only at training time; its embeddings are
baked into the checkpoint, so evaluation needs neither `transformers` nor the
network.

```python
def _spatial_tokens(image_size):
    side = max(1, int(np.ceil(image_size / 32)))
    return side * side
```

Upstream hardcodes 49, which is ResNet18's 7×7 output for a 224 px input and
silently wrong at any other size. At 128 px it is 16.

```python
for index, camera in enumerate(self.cameras):
    features = self.image_proj(self.backbone(image)).flatten(2).transpose(1, 2)
    tokens.append(features + self.image_pos + self.camera_embed[index])
```

Two cameras, each contributing spatial tokens with a per-camera embedding, so the
decoder can tell a head-view token from a wrist-view token at the same spatial
position.

```python
self.register_buffer("action_mean", torch.zeros(self.action_dim))
self.register_buffer("action_scale", torch.ones(self.action_dim))
```

Normalisation statistics are buffers, so they travel inside the checkpoint. A
checkpoint that cannot denormalise its own output loads cleanly, runs, and is
wrong by a scale factor.

### The overfit check that looked like an architecture bug

`m2_model_check.py` runs an overfit-one-batch test. The first version used
learning rate 3e-3 and scored against the initial loss, and it *failed* — the
loss plateaued exactly at the constant-predictor level, which reads as "the model
cannot distinguish its samples".

It can. Sweeping the learning rate first: at 1e-3 the ratio is 0.19, at 3e-4 it
is 0.17, at 1e-4 it is 0.15. **3e-3 destabilises this transformer.** Two changes
came out of it — the check uses 1e-3, and it scores against the best *constant*
predictor rather than the initial loss, because with N(0,1) targets the constant
predictor already scores 0.76 and a ratio to the starting loss cannot tell
"memorised the batch" from "learned its mean".

---

## 7. `train.py` — masked loss, baseline-relative validation

```python
def masked_l1(prediction, target, mask):
    error = (prediction - target).abs().mean(dim=-1)
    return (error * mask).sum() / mask.sum().clamp(min=1.0)
```

Validation is reported in raw units next to a predict-the-mean baseline:

```python
baseline = (action_mean.view(1, 1, -1) - target).abs().mean(dim=-1)
```

A normalised L1 of 0.31 says nothing. The same number in millimetres of hand
target, beside what predicting the training mean would score, says whether the
model learned anything at all. A policy that cannot beat the mean has learned the
dataset's average pose — and it still produces a smooth, falling training curve.

```python
optimiser = torch.optim.AdamW([
    {"params": other_params, "lr": learning_rate},
    {"params": backbone_params, "lr": learning_rate * 0.1},
])
```

`layer4` carries ImageNet features that a few thousand samples can destroy faster
than they can improve.

---

## 8. `policy_runner.py` — closed loop

A two-state machine, and its shape is forced by the gait:

```python
if action.gait > 0.5:
    runner.walk_unit()      # one step plus its closing step
else:
    runner.stand_tick(action)
```

A biped cannot be told to stop in the middle of a step, so the gait command is
acted on only at unit boundaries. A unit is one step plus its closing step —
Lab 8 L-M5-e: a walk that ends mid-stride hands the next one a stance it cannot
survive.

```python
nearest = min(distances, key=distances.get)
if distances[nearest] * 1000.0 > GRASP_GATE_MM:
    return False
self.scene.set_target(nearest)
```

The grasp closes on whatever the hand actually reached, **not** on what the
instruction named. Picking up the wrong object is a result the evaluation has to
be able to see; resolving the grasp by instruction would hide exactly the failure
the two-object scene was built to detect.

`joint_tick` is the ablation: the brief's 29 joint targets tracked by PD, with
Lab 8's standing gains and gravity compensation supplied so the comparison is
about the *action space* rather than a strawman controller.

---

## 9. `evaluate.py` and `capstone_demo.py`

The `walk` task is scored on the **standoff achieved to the named object**, not
on a step count:

```python
named = "cup" if "cup" in text else "box"
result.expert_pelvis_x = 0.5 * (named_x + marker_x) - REACH_STANDOFF
```

The two objects' correct stopping points are ~0.30 m apart, so going to the wrong
one cannot pass, and the result records how far it was from the *other* object's
stopping point too — a policy that reliably goes to the wrong one is a different
finding from one that stops randomly.

The capstone takes a free-form sentence with no task index anywhere on the path,
and profiles inference in float32 and under dynamic quantisation — the CPU
analogue of the brief's "INT8 on an RTX 4050", labelled as that rather than
reported as the thing asked for.

---

## Running things

```bash
export MUJOCO_GL=egl

# M0 — scene, cameras, contract, expert success rate
python3 lab-9-vla-integration/src/m0_scene_check.py --seeds 20

# M1 — demonstrations (about 40 min on 4 cores)
python3 lab-9-vla-integration/src/collect_demos.py --seeds 60
python3 lab-9-vla-integration/src/dataset.py --grid

# M2 — model
python3 lab-9-vla-integration/src/m2_model_check.py

# M3 — training
python3 lab-9-vla-integration/src/train.py --epochs 30
python3 lab-9-vla-integration/src/train.py --epochs 12 --conditioning task_id
python3 lab-9-vla-integration/src/train.py --epochs 12 --action-head joint

# M4 — closed-loop evaluation
python3 lab-9-vla-integration/src/evaluate.py --episodes 8

# M5 — capstone
python3 lab-9-vla-integration/src/capstone_demo.py \
    --instruction "pick up the red cup"

pytest lab-9-vla-integration/tests/
```
