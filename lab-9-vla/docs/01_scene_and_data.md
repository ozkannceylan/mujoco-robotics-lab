# 01: Scene Setup and Data Pipeline

## Overview

Lab 9 brings together every preceding lab into a single learned pipeline. Before we can
train a Vision-Language-Action (VLA) model, we need three things: a rich simulation scene
that produces visual observations, an expert controller that generates successful
demonstrations, and a data pipeline that packages everything into a training-ready format.

This document covers all three.

---

## 1. Scene Setup: G1 at the Tabletop

### 1.1 Scene Composition

The MuJoCo scene (`models/scene_vla.xml`) places the Unitree G1 humanoid standing at a
table. On the table surface sit several graspable objects:

- **Red cup** — cylinder with radius ~3 cm, height ~10 cm
- **Blue box** — cube with side length ~5 cm
- **Green bottle** — capsule with radius ~2.5 cm, height ~15 cm

Each object is a MuJoCo free body with a `<freejoint>`, which means it has full 6-DOF
mobility. The objects are colored with distinct RGBA values so the vision encoder can
distinguish them easily.

```xml
<body name="red_cup" pos="0.4 -0.1 0.76">
  <freejoint name="red_cup_joint"/>
  <geom type="cylinder" size="0.03 0.05" rgba="0.9 0.1 0.1 1"/>
</body>
```

### 1.2 Contact Configuration

Contact pairs are configured to ensure physically meaningful interactions:

- **Hand-object contacts**: The G1's hand link geoms can grip and push the objects.
  `contype` and `conaffinity` bitmasks are set so the hand meshes collide with object
  geoms.
- **Object-table contacts**: Objects rest on the table under gravity. The table surface
  geom has matching contact affinity with all object geoms.
- **Self-collision filtering**: Adjacent body links on the G1 are filtered out using
  MuJoCo's default adjacency rules to prevent false contacts at joints.

### 1.3 Stability Verification

Before any data collection begins, the scene is validated:

1. G1 stands without falling for 5 seconds of simulation time with zero control input
   (gravity compensation active)
2. All objects remain on the table surface without penetration
3. Both cameras render valid images (non-black, correct resolution)

---

## 2. Camera Rendering

### 2.1 Camera Placement

Two egocentric cameras provide the visual observations that the VLA policy will consume:

**Wrist camera** — attached to the right hand link of the G1. This camera moves with
the hand and provides a close-up view of the manipulation workspace. It sees the object
during approach and grasp phases.

**Head camera** — attached to the head link. This provides a wider, more stable view of
the entire tabletop. It captures the spatial layout of objects and gives the policy
global context about the scene.

Both cameras render at 640x480 resolution. During training, images are resized to
224x224 to match the vision encoder's expected input.

### 2.2 Rendering Pipeline

```python
def render_camera(mj_model: mujoco.MjModel,
                  mj_data: mujoco.MjData,
                  camera_name: str,
                  width: int = 640,
                  height: int = 480) -> np.ndarray:
    """Render an RGB image from the named camera.

    Returns:
        np.ndarray: RGB image with shape (height, width, 3), dtype uint8.
    """
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    renderer.update_scene(mj_data, camera=camera_name)
    return renderer.render()
```

The renderer is created once and reused across timesteps for efficiency. At 30 Hz
collection frequency, each episode of ~5 seconds produces approximately 150 frames per
camera, so 300 images total per demonstration.

### 2.3 Image Conventions

- Color space: RGB (not BGR)
- Pixel format: uint8, range [0, 255]
- Coordinate system: top-left origin, row-major
- Camera intrinsics: MuJoCo uses a symmetric pinhole model; focal length is derived from
  the `<camera fovy="...">` attribute

---

## 3. Expert Demonstration Collection

### 3.1 Why IK-Based Demonstrations

The expert controller does not use teleoperation. Instead, it uses the inverse kinematics
and whole-body controller infrastructure built in Labs 2, 5, and 8 to generate
programmatic demonstrations. This approach has several advantages:

- **Reproducibility**: The same initial conditions produce the same trajectory
- **Scalability**: Hundreds of demonstrations can be collected without human effort
- **Quality control**: Every demonstration is verified for success automatically

### 3.2 The Expert Controller Pipeline

The demonstration collector (`src/demo_collector.py`) follows a state machine:

```
APPROACH → PRE_GRASP → GRASP → LIFT → PLACE → RELEASE → RETRACT
```

Each state computes target joint positions using Pinocchio's IK solver, and the MuJoCo
simulation executes the motion using the whole-body controller. The expert uses:

- **Pinocchio IK** for computing joint angle targets from Cartesian end-effector goals
- **Whole-body controller** (from Lab 8) for balancing and tracking
- **Grasp state machine** (from Lab 5) for approach, grip, and release

### 3.3 Recording Protocol

At each simulation timestep (30 Hz recording frequency), the collector records:

| Field               | Shape / Type         | Description                              |
|---------------------|----------------------|------------------------------------------|
| `wrist_image`       | (480, 640, 3) uint8  | Wrist camera RGB image                   |
| `head_image`        | (480, 640, 3) uint8  | Head camera RGB image                    |
| `qpos`              | (N,) float64         | Joint positions (proprioception)         |
| `qvel`              | (N,) float64         | Joint velocities (proprioception)        |
| `action`            | (N,) float64         | Joint position targets for next step     |
| `language`          | string               | Task instruction, e.g. "pick up red cup" |
| `success`           | bool                 | Whether the episode succeeded            |
| `timestamp`         | float64              | Simulation time in seconds               |

### 3.4 Success Detection

A demonstration is marked successful if the task's success criterion is met:

- **Pick tasks**: Object is lifted at least 10 cm above the table surface
- **Place tasks**: Object is within 3 cm of the target location
- **Move tasks**: Object has been displaced by the specified direction/distance

Failed demonstrations are discarded and not included in the training dataset.

---

## 4. Domain Randomization

### 4.1 Why Randomize

A policy trained on a single scene configuration will overfit to the exact pixel patterns
of that configuration. Domain randomization forces the policy to learn features that are
invariant to visual nuisance factors, improving generalization to novel object placements,
lighting conditions, and even real-world transfer.

### 4.2 Randomization Axes

The `DomainRandomizer` class (`src/domain_randomizer.py`) perturbs the scene along four
axes before each episode:

**Object positions** — Each object's initial (x, y) position on the table is jittered by
up to +/- 10 cm from its nominal location. The z-position is kept fixed at the table
surface height. This ensures the policy cannot memorize a fixed spatial layout.

```python
obj_pos = nominal_pos + np.random.uniform(-0.10, 0.10, size=2)
```

**Object colors** — The hue component of each object's RGBA color is shifted by up to
+/- 30 degrees (in HSV space). Saturation and value are kept constant. The red cup
remains "reddish" but the exact shade varies episode to episode.

**Lighting** — The directional light's azimuth and elevation are jittered by +/- 15
degrees. Intensity is scaled by a factor in [0.7, 1.3]. This creates varying shadow
patterns and brightness levels.

**Camera pose jitter** — Both cameras receive small rotational perturbations of +/- 2
degrees around each axis. This simulates mounting imprecision and prevents the policy
from relying on exact pixel coordinates.

### 4.3 Reproducibility

Every randomization is seeded. The episode index serves as the base seed, so any
demonstration can be reproduced exactly:

```python
rng = np.random.RandomState(seed=episode_id)
randomizer.apply(mj_model, rng)
```

---

## 5. Dataset Format

### 5.1 HDF5 Structure

Each demonstration is stored as a single HDF5 file under
`data/demos/{task_name}/{demo_id}.hdf5`. The file structure is:

```
demo_0042.hdf5
├── observations/
│   ├── wrist_images     (T, 480, 640, 3)  uint8
│   ├── head_images      (T, 480, 640, 3)  uint8
│   ├── qpos             (T, N)            float64
│   └── qvel             (T, N)            float64
├── actions              (T, N)            float64
├── language             scalar string
├── success              scalar bool
└── attrs:
    ├── dt               0.0333...
    ├── num_timesteps     T
    ├── episode_seed     int
    └── task_name        string
```

T is the number of timesteps (typically 100-200 for a 3-7 second episode), and N is the
action dimension (number of controlled joints).

### 5.2 Why HDF5

HDF5 was chosen over alternatives for several reasons:

- **Random access**: Individual timesteps can be read without loading the entire episode
- **Compression**: Images compress well with gzip, reducing storage by 3-5x
- **Metadata**: Attributes store episode-level information alongside the data
- **Ecosystem**: PyTorch dataloaders can read HDF5 efficiently via `h5py`

### 5.3 Dataset Manifest

A JSON manifest file (`data/dataset_manifest.json`) indexes all demonstrations:

```json
{
  "tasks": {
    "pick_red_cup": {
      "num_demos": 55,
      "demo_ids": ["demo_0000", "demo_0001", ...],
      "success_rate": 0.92
    },
    "pick_blue_box": { ... },
    ...
  },
  "total_demos": 275,
  "collection_date": "2026-03-23"
}
```

---

## 6. Data Pipeline Design

### 6.1 Collection Script

The `src/collect_dataset.py` script orchestrates the full collection pipeline:

```
for each task in TASKS:
    collected = 0
    while collected < target_demos:
        1. Reset scene
        2. Apply domain randomization (seeded)
        3. Run expert controller
        4. Check success
        5. If successful: save HDF5, increment count
        6. If failed: discard, log failure reason
```

The target is 50+ successful demonstrations per task, across 5 tasks, yielding 250+ total
demonstrations. With an expert success rate of approximately 85-92%, this requires roughly
300 total episodes.

### 6.2 Storage Estimates

Each demonstration with images at full resolution (640x480) consumes approximately:

- Wrist images: 150 frames x 640 x 480 x 3 = ~138 MB (uncompressed)
- Head images: ~138 MB (uncompressed)
- Proprioception + actions: ~0.5 MB
- Total per demo (uncompressed): ~277 MB
- With gzip compression: ~50-80 MB per demo

For 275 demonstrations: approximately 15-22 GB total. This fits comfortably on a single
GPU machine's local storage.

### 6.3 Data Validation

After collection, a validation script checks every HDF5 file:

- All expected datasets exist with correct shapes and dtypes
- Image values are in [0, 255] with non-zero variance
- Joint positions are within model limits
- Actions are within actuator range
- Temporal alignment: action[t] corresponds to the transition from observation[t] to
  observation[t+1]

### 6.4 Train/Validation Split

The dataset is split 90/10 by episode (not by timestep). The split is stratified by
task to ensure each task is represented in both sets. The split is deterministic given a
fixed random seed, documented in the manifest.

---

## 7. Connection to the Full Pipeline

The data pipeline is the foundation for everything that follows:

- **Phase 2 (Model Architecture)** consumes the HDF5 files through a PyTorch Dataset
  that loads images, proprioception, language, and action chunks
- **Phase 3 (Training)** iterates over batches from the dataloader
- **Phase 4 (Deployment)** uses the same camera rendering and proprioception extraction
  functions at inference time, ensuring consistency between training and deployment

The principle is: never preprocess differently at train time vs. inference time. The
`render_camera()` and `get_proprioception()` functions defined in `lab9_common.py` are
used identically in both the data collector and the inference loop.

---

## Summary

| Component              | File                          | Purpose                             |
|------------------------|-------------------------------|-------------------------------------|
| Scene definition       | `models/scene_vla.xml`        | G1 + table + objects + cameras      |
| Common utilities       | `src/lab9_common.py`          | Paths, rendering, proprioception    |
| Demo collector         | `src/demo_collector.py`       | Expert controller + recording       |
| Domain randomizer      | `src/domain_randomizer.py`    | Scene perturbation                  |
| Collection script      | `src/collect_dataset.py`      | Orchestrates full data collection   |
| Dataset storage        | `data/demos/{task}/{id}.hdf5` | Individual demonstrations           |
| Manifest               | `data/dataset_manifest.json`  | Dataset index and statistics        |
