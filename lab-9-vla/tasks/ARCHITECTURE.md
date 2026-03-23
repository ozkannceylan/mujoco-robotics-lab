# Lab 9: VLA Integration — Architecture

## Module Map

```
lab-9-vla/
├── src/
│   ├── lab9_common.py                # Paths, constants, model loaders, camera helpers
│   ├── demo_collector.py             # IK-based expert demonstration collector
│   ├── domain_randomizer.py          # Scene randomization (objects, lighting, textures)
│   ├── collect_dataset.py            # Script: collect full demo dataset
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vision_encoder.py         # ResNet-18 dual-camera visual encoder
│   │   ├── language_encoder.py       # Frozen CLIP text encoder
│   │   └── act_policy.py             # ACT transformer policy (CVAE encoder-decoder)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py                # VLADataset: HDF5 → torch DataLoader
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py                 # TrainConfig dataclass
│   │   ├── trainer.py                # ACTTrainer: training loop, logging, checkpoints
│   │   └── evaluator.py              # PolicyEvaluator: rollout, success detection, metrics
│   │
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── quantize.py               # INT8 post-training quantization
│   │   ├── inference_loop.py         # Real-time VLA inference with temporal ensemble
│   │   └── multi_task_demo.py        # Multi-task language-conditioned demo
│   │
│   └── capstone_demo.py              # Capstone: "pick up the red cup" end-to-end
│
├── models/
│   ├── scene_vla.xml                 # G1 + table + objects + cameras
│   ├── g1_vla.xml                    # G1 MJCF (from mujoco_menagerie)
│   ├── g1_vla.urdf                   # G1 URDF for Pinocchio
│   └── scene_tabletop.xml            # Base tabletop scene
│
├── data/
│   ├── demos/                        # HDF5 demonstration files
│   │   ├── pick_red_cup/
│   │   ├── pick_blue_box/
│   │   ├── pick_green_bottle/
│   │   ├── move_cup_left/
│   │   └── move_box_right/
│   ├── dataset_manifest.json
│   └── checkpoints/                  # Trained model checkpoints
│
├── tests/
│   ├── test_scene.py
│   ├── test_demo_collector.py
│   ├── test_domain_randomizer.py
│   ├── test_vision_encoder.py
│   ├── test_language_encoder.py
│   ├── test_act_policy.py
│   ├── test_dataset.py
│   ├── test_trainer.py
│   ├── test_evaluator.py
│   ├── test_quantize.py
│   └── test_inference_loop.py
│
├── docs/
│   ├── 01_scene_and_data.md
│   ├── 02_model_architecture.md
│   ├── 03_training.md
│   └── 04_deployment.md
│
├── docs-turkish/
│   ├── 01_sahne_ve_veri.md
│   ├── 02_model_mimarisi.md
│   ├── 03_egitim.md
│   └── 04_dagitim.md
│
├── media/                            # Videos, plots, figures
├── tasks/                            # PLAN, ARCHITECTURE, TODO, LESSONS
└── README.md
```

## Data Flow

```
         "Pick up the red cup"
                  │
                  ▼
     ┌─────────────────────────┐
     │   Language Encoder       │
     │   (CLIP ViT-B/32 text)  │
     │   Frozen, FP16           │
     └───────────┬─────────────┘
                 │ lang_emb (512,)
                 ▼
Camera 30Hz ──→ ┌─────────────────────────────────────────┐
  wrist_cam     │             ACT Policy                    │
  head_cam      │                                           │
    │           │  ┌──────────┐  ┌────────┐  ┌──────────┐  │
    │           │  │ Vision   │  │ CVAE   │  │ Decoder  │  │
    └──────────▶│  │ Encoder  │─▶│ Encoder│─▶│ (Transf) │  │
                │  │ (ResNet  │  │ z~N(μ,σ│  │ cross-   │  │
                │  │  -18x2)  │  │  )     │  │ attend   │  │
                │  └──────────┘  └────────┘  └────┬─────┘  │
  proprio ─────▶│                                  │        │
  (qpos, qvel)  │  lang_emb ─────────────────────▶│        │
                │                                  │        │
                └──────────────────────────────────┼────────┘
                                                   │
                                          action chunk (10, action_dim)
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │  Temporal Ensemble     │
                                        │  Exponential weighted  │
                                        │  average of overlapping│
                                        │  action chunks         │
                                        └──────────┬─────────────┘
                                                   │ a_t (action_dim,)
                                                   ▼
                                        ┌──────────────────────┐
                                        │  MuJoCo G1            │
                                        │  data.ctrl = a_t      │
                                        │  mj_step()            │
                                        │  Render cameras        │
                                        └──────────────────────┘
                                                   │
                                                   ▼
                                          Next frame → loop
```

### Data flow summary:
1. **Language Encoder** (frozen CLIP) encodes the task instruction once per episode into a 512-dim embedding.
2. **Camera rendering** produces two 640x480 RGB images at 30 Hz from wrist and head cameras.
3. **Vision Encoder** (ResNet-18 backbone) encodes both images into a 1024-dim visual embedding.
4. **Proprioception** (joint positions + velocities) extracted from `mj_data`.
5. **ACT Policy** (CVAE Transformer) outputs an action chunk of 10 joint position targets.
6. **Temporal Ensemble** blends overlapping action chunks with exponential weighting.
7. **MuJoCo** applies the blended action as `ctrl`, steps physics, renders next frame.


## Key Interfaces

### lab9_common.py

```python
# Paths, constants, camera rendering, proprioception extraction
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
ACTION_CHUNK_SIZE: int = 10
POLICY_DT: float = 1.0 / 30.0

TASKS: dict[str, dict]  # {task_name: {lang: str, objects: list, target: ...}}

def load_mujoco_model() -> tuple: ...
def render_camera(mj_model, mj_data, camera_name) -> np.ndarray: ...
def get_proprioception(mj_data) -> np.ndarray: ...
```

### models/act_policy.py

```python
class ACTPolicy(nn.Module):
    """CVAE Transformer: vision + proprio + language → action chunk."""
    def forward(self, vision_emb, proprio, lang_emb, actions=None) -> dict: ...
    def compute_loss(self, pred, target_actions, is_pad) -> dict: ...
    def get_action(self, vision_emb, proprio, lang_emb) -> torch.Tensor: ...
```

### deployment/inference_loop.py

```python
class TemporalEnsemble:
    """Blends overlapping action chunks with exponential weighting."""
    def add_chunk(self, action_chunk: np.ndarray) -> None: ...
    def get_action(self) -> np.ndarray: ...

class VLAInferenceLoop:
    """Real-time VLA inference pipeline."""
    def run_episode(self, language_instruction: str, max_steps: int = 300) -> tuple[bool, dict]: ...
    def step(self) -> np.ndarray: ...
```


## Dependencies on Previous Labs

| Lab | What it contributes |
|-----|---------------------|
| Lab 3 | Impedance control for compliant manipulation in demos |
| Lab 5 | Grasp state machine for pick-and-place demos |
| Lab 7 | Locomotion gait for walking demos |
| Lab 8 | Whole-body QP controller — the expert demonstrator |

### New dependencies (not in previous labs)

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | Neural network training and inference |
| `torchvision >= 0.15` | ResNet-18, image transforms |
| `open_clip_torch >= 2.20` | CLIP text encoder |
| `h5py >= 3.8` | HDF5 dataset storage |
| `wandb >= 0.15` | Training logging |


## Key Design Decisions

1. **IK-based demo generation, not teleoperation.** Lab 8's controller generates expert trajectories programmatically. Faster, more repeatable.
2. **ACT over diffusion policy.** Simpler (~15M params), trains faster, validated in humanoid_vla.
3. **Dual-camera input.** Wrist = close-up manipulation, head = scene context. Depth cues without depth sensor.
4. **Frozen CLIP text encoder.** No fine-tuning needed — already captures semantic meaning.
5. **Temporal ensemble.** Prevents jerky transitions between action chunks.
6. **INT8 quantization.** Train on cloud A100, deploy locally at >10 Hz.
7. **HDF5 for demos.** Efficient binary format for large image+action datasets.
8. **Domain randomization at collection time.** Expert demonstrates under variation, teaching the policy invariance.
