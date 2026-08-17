# Lab 9 — Architecture (pre-implementation design record)

> Written before any code, per the project workflow. Brief: `plan/LAB_09.md`.
> Milestone plan and the four scope deviations: `tasks/PLAN.md`.
> The reader-facing version lives in `docs/ARCHITECTURE.md` once M6 lands.

---

## The shape of the system

```
      "pick up the red cup"
              │
     ┌────────▼─────────┐   frozen, offline
     │ CLIP text tower  │   → 512-d, L2-normalised, cached in an
     └────────┬─────────┘     instruction bank inside the checkpoint
              │ 512
head cam ─┐   │
wrist cam ┤   │        ┌──────────────────────────────┐
 (128²)   ├───┴───────►│  ACT policy                  │
proprio ──┘            │  ResNet18 (layer4 trainable) │
 (62-d)                │  + spatial tokens + state    │
                       │  + instruction token         │
                       │  → transformer decoder       │
                       │  → chunk of K future actions │
                       └───────────────┬──────────────┘
                                       │ 10 Hz, K-step chunk
                       ┌───────────────▼──────────────┐
                       │  action decoder              │
                       │  task head  → hand targets,  │
                       │               gait cmd, grasp│
                       │  joint head → 29 q_des       │
                       └───────────────┬──────────────┘
                                       │
                       ┌───────────────▼──────────────┐
                       │  Lab 8 whole-body ID QP      │   1 kHz
                       │  balance is a hard constraint│
                       └───────────────┬──────────────┘
                                       │ τ (29)
                                  MuJoCo G1
```

The load-bearing structural claim: **the policy decides *what*, Lab 8's QP
decides *how to stay upright*.** Balance never becomes a learned quantity. The
brief's alternative — a 29-DOF joint-target policy tracked by PD — is built as
an ablation because Lab 7 already measured what happens to a position reference
on this robot, and the arc from Lab 7 → Lab 8 → Lab 9 is only closed by
measuring it here too.

---

## Modules

| File | Role | Depends on |
|---|---|---|
| `lab9_common.py` | Paths, image/obs constants, task enum, instruction vocabulary, Lab 8 re-exports, seeding | Lab 8 `lab8_common` |
| `vla_scene.py` | Two-object randomisable scene + egocentric cameras; wraps Lab 8's `build_capstone_scene` pattern | `g1_torque_model`, `capstone_scene` |
| `observations.py` | Observation builder + action encode/decode for both heads; the only place the obs/action layout is defined | `lab9_common` |
| `expert.py` | Lab 8's capstone sequence as a scriptable, seedable episode over the two-object scene; emits a phase-labelled log | Lab 8 `m5_capstone` internals, `locomotion_controller`, `wb_id_qp` |
| `collect_demos.py` | Multi-process rollout → per-episode `.npz` + manifest; phase slicing into labelled tasks | `expert`, `observations` |
| `dataset.py` | Chunked `Dataset`, pad masks, seed-level split, normalisation statistics | `collect_demos` output |
| `text_encoder.py` | Frozen CLIP text tower + instruction bank | `transformers` (train only) |
| `act_policy.py` | The ACT model, both heads, normalisation buffers | `torch`, `torchvision` |
| `train.py` | Training loop, checkpointing, resume | `dataset`, `act_policy` |
| `evaluate.py` | Closed-loop rollouts, success detection, ablations | `expert` scene, `act_policy` |
| `capstone_demo.py` | Free-form instruction → recorded autonomous episode | everything |

---

## Interfaces

### Observation

```python
obs = {
    "head":   uint8 (128, 128, 3),   # torso camera, egocentric
    "wrist":  uint8 (128, 128, 3),   # right wrist camera
    "state":  float32 (62,),          # proprioception, see below
}
```

`state` is deliberately *not* the full 71-d MuJoCo state:

| block | dims | why |
|---|---|---|
| joint positions `qpos[7:36]` | 29 | the robot's own configuration |
| joint velocities `qvel[6:35]` | 29 | ACT is a feedforward chunk predictor; velocity is what makes the state Markov-ish |
| pelvis height | 1 | the one base coordinate a policy may legitimately use — it is proprioceptive (IMU/leg kinematics), unlike base *x/y* which would leak global position |
| pelvis roll/pitch | 2 | likewise IMU-observable |
| grasp state | 1 | whether the weld is closed; unobservable from the wrist camera at this resolution |

Base *x*, *y* and yaw are **excluded on purpose**. A policy given its own world
coordinates can solve these tasks by dead reckoning and ignore both the camera
and the instruction — the evaluation would then measure nothing. Everything the
policy knows about where it is must come through the pixels.

### Action

Both heads are `chunk_size` steps deep. The chunk is predicted at 10 Hz and
executed open-loop until the next inference, i.e. temporal ensembling is *not*
used (upstream's optional feature; adds latency budget this machine does not
have — recorded as a deliberate simplification).

`task` head — 9 dims, all in **world coordinates relative to the pelvis** so the
representation is translation-invariant along the walk:

| slice | meaning |
|---|---|
| `0:3` | right hand target, pelvis-relative |
| `3:6` | left hand target, pelvis-relative |
| `6` | gait command: forward step rate, 0 = stand |
| `7` | grasp: right weld closed |
| `8` | grasp: left weld closed |

`joint` head — 29 dims, the joint configuration one control period ahead.

Encode/decode is a pure function pair with a round-trip test; nothing else in
the lab is allowed to know the layout.

### Instruction

```python
TASKS = {
    "walk":  "walk to the table",
    "pick":  "pick up the {object}",
    "carry": "carry the {object} to the shelf",
    "place": "put the {object} on the shelf",
}
OBJECTS = {"cup": "red cup", "box": "blue box"}
```

Each task/object pair carries several paraphrases; one set is held out from
training so M4 can measure paraphrase robustness rather than memorisation.

---

## Data flow, one closed-loop tick

```
MuJoCo state ─► observations.build(model, data, renderer, grasp_state)
                     │  two 128² renders (97 ms each — the rate limiter)
                     ▼
              ACT.predict(images, state, instruction_embedding)
                     │  chunk (K, 9)
                     ▼
              observations.decode_task_action(a, pelvis_pose)
                     │  hand targets (world), gait cmd, grasp bits
                     ▼
              Lab 8 task stack + whole-body ID QP  ──► τ  ──► mj_step ×100
```

The 10 Hz policy rate is set by rendering, not by the model: two software
renders cost ~194 ms, which caps the honest closed-loop rate at ~5 Hz wall
clock even though the *policy* runs far faster. Rendering cost is therefore
reported separately from inference cost, and the > 10 Hz criterion is a claim
about **inference**, stated as such.

---

## Model

Adapted from `ozkannceylan/humanoid_vla`'s `ACTPolicy`; the deltas are:

| upstream | here | why |
|---|---|---|
| 49 spatial tokens (hardcoded, 224 px) | token count derived from the feature map (16 at 128 px) | the constant is wrong for any other input size |
| one camera | two (head + wrist), concatenated as tokens with a per-camera embedding | the wrist view is what makes the last 5 cm of a grasp observable |
| action = joint targets | two heads (see above) | Lab 7's finding |
| temporal ensembling optional | absent | latency budget |
| ImageNet-pretrained ResNet18, layer4 fine-tuned | same | keep it |
| norm stats as buffers in the module | same | keep it — a checkpoint that cannot denormalise itself is a trap |

---

## Model files

Nothing is committed. `vla_scene.py` builds the scene at runtime from
Menagerie + Lab 8's spec builder, exactly as Lab 8 does, so Menagerie stays the
single source of truth. `export_xml()` emits a snapshot for inspection.

`data/` and `models/` (checkpoints) are gitignored: a demonstration set is
reproducible from its seed list, and a CPU-trained checkpoint is ~60 MB.
The **manifest** and the **metrics** are committed, because those are what a
reader needs to check the claims.

---

## Cross-lab dependencies

| From | What |
|---|---|
| Lab 8 | The entire control path: `wb_id_qp`, `wb_tasks`, `gait_planner`, `dcm_planner`, `locomotion_controller`, `capstone_scene`, `g1_torque_model`, and the M5 sequence logic |
| Lab 7 (via Lab 8) | G1 conventions: joint order, qpos/qvel slices, pelvis MJCF offset, quaternion order |
| `humanoid_vla` | ACT architecture and the instruction-bank idea (design reuse, credited; no code vendored) |

`sys.path.append` for foreign labs, never `insert(0)`.

---

## What would make this lab a failure

Stated up front so the gates cannot be quietly softened later:

1. A success rate reported without an episode count or without the
   randomisation range it was measured over.
2. An evaluation that a policy ignoring its instruction could pass.
3. Success detection that reads a commanded value rather than simulated state.
4. A train/val split that shares a seed between the two.
5. Any claim that the policy "walks" when Lab 8's QP is doing the walking.
