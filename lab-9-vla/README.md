# Lab 9: VLA Integration

Language command in, autonomous robot execution out. This is the capstone of
the robotics lab series: a Vision-Language-Action (VLA) policy that takes
camera images and a natural language instruction ("pick up the red cup") and
outputs joint actions for the Unitree G1 humanoid in MuJoCo.

Every preceding lab built a piece of this puzzle: kinematics (Labs 1–2),
dynamics and force control (Lab 3), motion planning (Lab 4), grasping (Lab 5),
dual-arm coordination (Lab 6), locomotion (Lab 7), and whole-body
loco-manipulation (Lab 8). Now, a learned policy replaces the hand-coded
pipeline.

---

## Architecture

```
         "Pick up the red cup"
                  │
                  ▼
       ┌──────────────────┐
       │ CLIP Text Encoder │  (frozen, 512-dim embedding)
       └────────┬─────────┘
                │
Cameras ──→ ┌───────────────────────────┐
 wrist_cam   │       ACT Policy          │
 head_cam    │                           │
   │         │  ResNet-18 ──→ Vision Emb │
   └────────▶│  + Proprioception         │
             │  + Language Emb           │
             │  ──→ CVAE Transformer     │
             │  ──→ Action Chunk (10)    │
             └───────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ Temporal Ensemble │  (smooth blending)
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  MuJoCo G1       │
              │  Execute action   │
              │  Step physics     │
              │  Render cameras   │
              └──────────────────┘
                       │
                       ▼
              Next camera frame → loop
```

**Key principle:** Pinocchio generates expert demonstrations via IK-based
control (analytical brain). The ACT policy learns from these demonstrations
to produce actions directly from vision and language (learned brain). MuJoCo
is the physics simulator throughout.

---

## Key Concepts

### Action Chunking with Transformers (ACT)
The policy predicts a *chunk* of 10 future actions at once. This smooths
the output, reduces compounding errors from behavior cloning, and lets the
robot commit to coherent motion plans.

### Language Conditioning
A frozen CLIP text encoder converts instructions into 512-dim embeddings,
injected into the Transformer decoder via cross-attention.

### Temporal Ensemble
Overlapping action chunks are blended with exponential weighting for smoother
motion at each timestep.

### Domain Randomization
Object positions, colors, lighting, and camera angles are randomized during
demonstration collection. This teaches invariance to irrelevant visual features.

---

## Repository Structure

```
lab-9-vla/
├── src/
│   ├── lab9_common.py, demo_collector.py, domain_randomizer.py
│   ├── collect_dataset.py
│   ├── models/ (vision_encoder, language_encoder, act_policy)
│   ├── data/ (dataset.py)
│   ├── training/ (config, trainer, evaluator)
│   ├── deployment/ (quantize, inference_loop, multi_task_demo)
│   └── capstone_demo.py
├── models/                         # G1 + scene MJCFs
├── data/demos/                     # HDF5 demonstrations
├── data/checkpoints/               # Trained models
├── tests/                          # 11 test files
├── docs/ + docs-turkish/           # 4 articles each
├── tasks/                          # PLAN, ARCHITECTURE, TODO, LESSONS
└── media/
```

---

## Dependencies

```
Python          >= 3.10
MuJoCo          >= 3.0
pinocchio       >= 2.6
numpy           >= 1.24
matplotlib      >= 3.7
torch           >= 2.0
torchvision     >= 0.15
open_clip_torch >= 2.20
h5py            >= 3.8
wandb           >= 0.15
tqdm            >= 4.65
```

---

## Running

```bash
# Data collection
python src/collect_dataset.py --tasks all --demos-per-task 50

# Training (cloud GPU recommended)
python -m src.training.trainer --config configs/default.yaml

# Evaluation
python -m src.training.evaluator --checkpoint data/checkpoints/best.pt

# INT8 quantization
python src/deployment/quantize.py --checkpoint data/checkpoints/best.pt

# Multi-task demo
python src/deployment/multi_task_demo.py --instruction "pick up the red cup"

# Capstone
python src/capstone_demo.py --record --output media/capstone_demo.mp4

# Tests
pytest tests/ -v
```

---

## Results Summary

| Metric | Target | Actual |
|--------|--------|--------|
| Demos per task | 50+ | — |
| Training task success | >70% | — |
| Randomized variant success | >40% | — |
| Inference speed (INT8) | >10 Hz | — |
| Capstone demo success | >70% (10 trials) | — |
| Model parameters | 15–20M | — |

---

## Connection to Prior Labs

| Lab | What it contributes |
|-----|---------------------|
| Labs 1–2 | FK/IK foundations for expert demo generation |
| Lab 3 | Impedance control for compliant manipulation |
| Lab 4 | Collision-free planning for expert trajectories |
| Lab 5 | Grasp state machine for pick-and-place demos |
| Lab 6 | Dual-arm coordination patterns |
| Lab 7 | Locomotion gait for walking demos |
| Lab 8 | Whole-body QP controller — the expert demonstrator |
