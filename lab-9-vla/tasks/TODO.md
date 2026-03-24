# Lab 9: TODO

## Phase 1: Scene & Data Pipeline
- [ ] Step 1.1: Create G1 tabletop MuJoCo scene with objects
- [ ] Step 1.2: Create lab9_common.py (paths, constants, camera helpers, proprioception)
- [ ] Step 1.3: Build expert demonstration collector (IK-based, HDF5 output)
- [ ] Step 1.4: Implement domain randomization (object pos, colors, lighting, camera jitter)
- [ ] Step 1.5: Collect full demonstration dataset (50+ demos/task, 5 tasks)
- [ ] Step 1.6: Phase 1 tests (test_scene, test_demo_collector, test_domain_randomizer)

## Phase 2: Model Architecture
- [ ] Step 2.1: Implement vision encoder (ResNet-18 dual-camera, 1024-dim output)
- [ ] Step 2.2: Implement language encoder (frozen CLIP ViT-B/32, 512-dim output)
- [ ] Step 2.3: Implement ACT policy network (CVAE Transformer, ~15-20M params)
- [ ] Step 2.4: Implement dataset and dataloader (HDF5 → torch Dataset)
- [ ] Step 2.5: Phase 2 tests (test_vision_encoder, test_language_encoder, test_act_policy, test_dataset)

## Phase 3: Training
- [ ] Step 3.1: Implement training loop (L1 + KL loss, AdamW, cosine LR, wandb)
- [ ] Step 3.2: Implement evaluation pipeline (rollout, success detection, metrics)
- [ ] Step 3.3: Implement training configuration (TrainConfig dataclass)
- [ ] Step 3.4: Train baseline model (500 epochs on cloud GPU, target >70%/>40% success)
- [ ] Step 3.5: Iterate on training (if below target success rates)
- [ ] Step 3.6: Phase 3 tests (test_trainer, test_evaluator)

## Phase 4: Deployment & Demo
- [ ] Step 4.1: Implement INT8 quantization (PyTorch static quant, benchmark FP32 vs INT8)
- [ ] Step 4.2: Implement real-time inference loop (temporal ensemble, >10 Hz)
- [ ] Step 4.3: Multi-task demonstration (3+ language commands, video recording)
- [ ] Step 4.4: Capstone demo — "Pick up the red cup" (end-to-end, >70% over 10 trials)
- [ ] Step 4.5: Phase 4 tests (test_quantize, test_inference_loop)

## Phase 5: Documentation & Blog
- [ ] Step 5.1: English documentation (4 docs: scene/data, architecture, training, deployment)
- [ ] Step 5.2: Turkish documentation (4 translated docs)
- [ ] Step 5.3: Blog post — "From Manual Control to Learned Autonomy: The VLA Journey"
- [ ] Step 5.4: Final README (lab-9-vla/README.md)

## Current Focus
> Step 1.1: Create G1 tabletop MuJoCo scene with objects

## Blockers
> None
