# Lab 9: VLA Integration — Implementation Plan

## Phase 1: Scene & Data Pipeline

### Step 1.1: Create G1 tabletop MuJoCo scene with objects
- Build `models/scene_vla.xml` — G1 humanoid standing at a table with graspable objects (red cup, blue box, green bottle) on its surface
- Include egocentric cameras: wrist camera (attached to right hand link) and head camera (attached to head link)
- Objects are free bodies with freejoint, colored distinctly for vision-based identification
- Configure contact pairs: hand ↔ objects, objects ↔ table
- **Verify:** Scene loads, G1 stands stable, all objects visible from both cameras, render 640x480 images from each camera

### Step 1.2: Create lab9_common.py
- Define paths (LAB_DIR, MODELS_DIR, MEDIA_DIR, DATA_DIR), constants (NUM_JOINTS, DT, CAMERA_WIDTH=640, CAMERA_HEIGHT=480, CAMERA_FPS=30, ACTION_CHUNK_SIZE=10)
- Model loading helpers: `load_mujoco_model()`, `load_pinocchio_model()`
- Camera rendering helper: `render_camera(mj_model, mj_data, camera_name) -> np.ndarray`
- Proprioception extraction: `get_proprioception(mj_data) -> np.ndarray` (joint positions + velocities)
- Quaternion utilities (reused from earlier labs)
- Task definitions dict: `TASKS = {"pick_red_cup": {...}, "pick_blue_box": {...}, ...}`
- **Verify:** Models load, cameras render valid images, proprioception vector has correct shape

### Step 1.3: Build expert demonstration collector
- Build `src/demo_collector.py` with `DemoCollector` class
- Uses Lab 8's whole-body controller and Lab 5's grasp state machine to generate expert trajectories programmatically (IK-based, not teleoperation)
- Each demonstration records at 30 Hz: camera images (wrist + head), proprioception (joint pos/vel), actions (joint position targets), language instruction string, success flag
- Save demonstrations in HDF5 format: `data/demos/{task_name}/{demo_id}.hdf5`
- **Verify:** Collect 5 test demos for `pick_red_cup`, inspect HDF5 structure, validate image shapes and action dimensions

### Step 1.4: Implement domain randomization
- Build `src/domain_randomizer.py` with `DomainRandomizer` class
- Randomize: object positions (± 10 cm), object colors (hue shift ± 30°), lighting direction/intensity, camera pose jitter (± 2°)
- Apply randomization before each episode via MuJoCo model modifications
- **Verify:** Render 10 randomized scenes, visually confirm variation, save comparison grid

### Step 1.5: Collect full demonstration dataset
- Build `src/collect_dataset.py` script
- Collect 50+ demonstrations per task (5 tasks = 250+ total demos)
- Each demo: randomized scene, expert controller execution, success detection
- Save dataset manifest: `data/dataset_manifest.json`
- **Verify:** 50+ successful demos per task, spot-check 5 random demos by replaying

### Step 1.6: Phase 1 tests
- `tests/test_scene.py`: Scene loading, camera rendering, object positions
- `tests/test_demo_collector.py`: HDF5 format validation, image/action shape checks
- `tests/test_domain_randomizer.py`: Randomization ranges, reproducibility with fixed seed
- **Verify:** All tests pass


## Phase 2: Model Architecture

### Step 2.1: Implement vision encoder
- Build `src/models/vision_encoder.py` with `DualCameraVisionEncoder`
- ResNet-18 backbone (pretrained on ImageNet), one per camera
- Input: 2 camera images (3x224x224 each) → Output: visual embedding (1024-dim, 512 per camera)
- **Verify:** Forward pass with random images produces correct output shape (batch, 1024)

### Step 2.2: Implement language encoder
- Build `src/models/language_encoder.py` with `CLIPLanguageEncoder`
- Frozen CLIP text encoder (ViT-B/32 via `open_clip`)
- Input: language instruction string → Output: 512-dim embedding
- Cache embeddings for known task instructions
- **Verify:** Different instructions produce different embeddings (cosine distance > 0.1)

### Step 2.3: Implement ACT policy network
- Build `src/models/act_policy.py` with `ACTPolicy` class
- CVAE Transformer encoder-decoder architecture
- Encoder: observation (vision 1024 + proprio ~60 + language 512) → latent z (32-dim)
- Decoder: cross-attend to observation, decode action chunk (10 x action_dim)
- Transformer: 4 layers, 8 heads, hidden_dim=256. Target: ~15-20M parameters
- **Verify:** Forward pass produces correct action chunk shape, parameter count in range

### Step 2.4: Implement dataset and dataloader
- Build `src/data/dataset.py` with `VLADataset(torch.utils.data.Dataset)`
- Load HDF5 demos, return (images, proprioception, language, action_chunk, is_pad_mask)
- Temporal chunking: action_chunk = actions[t:t+CHUNK_SIZE]
- Image augmentation: random crop, color jitter (training only)
- **Verify:** Iterate one batch, check all tensor shapes, verify temporal alignment

### Step 2.5: Phase 2 tests
- `tests/test_vision_encoder.py`, `tests/test_language_encoder.py`, `tests/test_act_policy.py`, `tests/test_dataset.py`
- **Verify:** All tests pass


## Phase 3: Training

### Step 3.1: Implement training loop
- Build `src/training/trainer.py` with `ACTTrainer`
- Loss: L1 reconstruction on action chunks + KL divergence (beta=10)
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Cosine LR schedule with warmup (500 steps)
- Mixed precision (torch.cuda.amp), wandb logging, checkpointing
- **Verify:** Training runs for 5 epochs, loss decreases, checkpoint saved

### Step 3.2: Implement evaluation pipeline
- Build `src/training/evaluator.py` with `PolicyEvaluator`
- Load checkpoint, run policy in MuJoCo, detect success (object ± 3 cm of target)
- Metrics: success rate, trajectory smoothness (jerk), episode length, inference time
- **Verify:** Evaluation runs for 10 episodes, metrics logged

### Step 3.3: Implement training configuration
- Build `src/training/config.py` with `TrainConfig` dataclass
- All hyperparameters: epochs=500, batch_size=64, chunk_size=10, lr=1e-4, beta_kl=10
- **Verify:** Config loads with valid defaults

### Step 3.4: Train baseline model
- Train for 500 epochs on cloud GPU (~2-4 hours on A100)
- Evaluate at checkpoints: epoch 100, 200, 300, 400, 500
- **Verify:** >70% success on training configs, >40% on randomized variants

### Step 3.5: Iterate on training
- If below targets: increase demos, adjust randomization, tune chunk_size and beta_kl
- **Verify:** Improved success rates after iteration

### Step 3.6: Phase 3 tests
- `tests/test_trainer.py`, `tests/test_evaluator.py`


## Phase 4: Deployment & Demo

### Step 4.1: Implement INT8 quantization
- Build `src/deployment/quantize.py`
- Post-training static quantization (PyTorch)
- Benchmark FP32 vs INT8: inference time and accuracy
- **Verify:** INT8 maintains >90% of FP32 success rate, inference < 100ms per step

### Step 4.2: Implement real-time inference loop
- Build `src/deployment/inference_loop.py` with `VLAInferenceLoop`
- Temporal ensemble: overlap action chunks with exponential weighting
- Pipeline: render → encode → predict → ensemble → apply action
- **Verify:** >10 Hz inference, smooth robot motion

### Step 4.3: Multi-task demonstration
- Build `src/deployment/multi_task_demo.py`
- Test with: "pick up the red cup", "pick up the blue box", "move the green bottle to the left"
- Record video for each task with language instruction overlay
- **Verify:** At least 3 out of 5 tasks succeed

### Step 4.4: Capstone demo — "Pick up the red cup"
- Build `src/capstone_demo.py`
- Full end-to-end: language → camera → policy → action → success
- Record video with language command, camera PiP, third-person view, success indicator
- **Verify:** >70% success over 10 trials

### Step 4.5: Phase 4 tests
- `tests/test_quantize.py`, `tests/test_inference_loop.py`


## Phase 5: Documentation & Blog

### Step 5.1: English documentation
- `docs/01_scene_and_data.md`, `docs/02_model_architecture.md`, `docs/03_training.md`, `docs/04_deployment.md`

### Step 5.2: Turkish documentation
- `docs-turkish/01_sahne_ve_veri.md`, `docs-turkish/02_model_mimarisi.md`, `docs-turkish/03_egitim.md`, `docs-turkish/04_dagitim.md`

### Step 5.3: Blog post
- `blog/lab_09_vla_integration.md` — "From Manual Control to Learned Autonomy: The VLA Journey"

### Step 5.4: Final README
- `lab-9-vla/README.md`
