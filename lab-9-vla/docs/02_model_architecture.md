# 02: Model Architecture

## Overview

The VLA (Vision-Language-Action) model takes three inputs — camera images, proprioceptive
state, and a language instruction — and outputs a sequence of future actions. This
document describes the full architecture: how each modality is encoded, how the encodings
are fused, and how the policy network produces action chunks.

---

## 1. The VLA Pipeline at a Glance

```
                    ┌────────────────┐
                    │  Language       │
                    │  Instruction    │
                    │  "pick red cup" │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  Language       │
                    │  Encoder        │
                    │  (CLIP frozen)  │
                    └───────┬────────┘
                            │ 512-dim
           ┌────────────────┼────────────────┐
           │                │                │
   ┌───────▼────────┐      │      ┌─────────▼──────┐
   │  Wrist Camera   │      │      │  Head Camera    │
   │  Image (224x224)│      │      │  Image (224x224)│
   └───────┬────────┘      │      └─────────┬──────┘
           │                │                │
   ┌───────▼────────┐      │      ┌─────────▼──────┐
   │  Vision Encoder │      │      │  Vision Encoder │
   │  (ResNet-18)    │      │      │  (ResNet-18)    │
   └───────┬────────┘      │      └─────────┬──────┘
           │ 512-dim        │                │ 512-dim
           └────────┬───────┼────────────────┘
                    │       │
                    ▼       ▼
              ┌─────────────────────┐
              │  Concatenation       │
              │  [vis_w | vis_h |    │
              │   lang | proprio]    │
              │  1024+512+proprio    │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  ACT Policy Network  │
              │  (Transformer)       │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Action Chunk        │
              │  (10 x action_dim)   │
              └─────────────────────┘
```

The pipeline processes three modalities in parallel, fuses them into a single observation
vector, and decodes an action chunk through a transformer-based policy.

---

## 2. Vision Encoder

### 2.1 Architecture: DualCameraVisionEncoder

Each camera image is processed by its own ResNet-18 backbone. The two backbones share
the same architecture but have independent weights, allowing them to specialize: the
wrist camera encoder learns to detect objects at close range, while the head camera
encoder learns spatial layout features.

```python
class DualCameraVisionEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.wrist_backbone = resnet18(pretrained=pretrained)
        self.head_backbone = resnet18(pretrained=pretrained)

        # Replace classification head with projection
        self.wrist_backbone.fc = nn.Linear(512, embed_dim)
        self.head_backbone.fc = nn.Linear(512, embed_dim)

    def forward(self, wrist_img: Tensor, head_img: Tensor) -> Tensor:
        """
        Args:
            wrist_img: (B, 3, 224, 224) normalized RGB
            head_img:  (B, 3, 224, 224) normalized RGB
        Returns:
            (B, 1024) concatenated visual embedding
        """
        wrist_feat = self.wrist_backbone(wrist_img)   # (B, 512)
        head_feat = self.head_backbone(head_img)       # (B, 512)
        return torch.cat([wrist_feat, head_feat], dim=-1)
```

### 2.2 Why ResNet-18

ResNet-18 is chosen as a balance between representation capacity and computational cost:

- **Parameter count**: ~11M parameters per backbone. Two backbones total ~22M, which is
  manageable for training on a single GPU.
- **Pretrained initialization**: ImageNet pretraining gives useful low-level features
  (edges, colors, textures) that transfer well to robotic manipulation scenes.
- **Inference speed**: ResNet-18 runs in under 5ms per forward pass on a modern GPU,
  leaving most of the latency budget for the policy network.

Larger backbones (ResNet-50, ViT) could be substituted for more complex visual scenes at
the cost of increased training time and inference latency.

### 2.3 Image Preprocessing

Raw 640x480 images from MuJoCo are preprocessed before entering the encoder:

1. **Resize** to 224x224 (bilinear interpolation)
2. **Normalize** to [0, 1] then apply ImageNet mean/std normalization
3. **Training augmentation**: random crop (from 240x240 to 224x224), color jitter
   (brightness/contrast/saturation +/- 0.1)

The same resize and normalization are applied at inference time. Augmentation is training
only.

---

## 3. Language Encoder

### 3.1 Architecture: CLIPLanguageEncoder

The language encoder converts a natural language instruction string into a fixed-size
embedding vector. We use the text encoder from CLIP (ViT-B/32), frozen during training.

```python
class CLIPLanguageEncoder(nn.Module):
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.text_encoder = clip_model.text
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Freeze all parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, instruction: str) -> Tensor:
        """
        Args:
            instruction: Natural language string, e.g. "pick up the red cup"
        Returns:
            (512,) language embedding
        """
        tokens = self.tokenizer([instruction])
        with torch.no_grad():
            embedding = self.text_encoder(tokens)
        return embedding.squeeze(0)
```

### 3.2 Why Frozen CLIP

Freezing the language encoder has three benefits:

1. **Reduced training cost**: The text encoder's ~63M parameters do not need gradients,
   saving memory and compute.
2. **Semantic grounding**: CLIP was trained on 400M image-text pairs. Its text embeddings
   already capture rich semantic relationships ("red" is closer to "crimson" than to
   "large"). Fine-tuning on our small dataset of 5 task strings would destroy this.
3. **Zero-shot potential**: New task descriptions not seen during training may still
   produce meaningful embeddings, enabling some degree of language generalization.

### 3.3 Embedding Cache

Since the number of distinct task instructions is small (5 tasks), embeddings are
precomputed and cached:

```python
TASK_EMBEDDINGS = {
    "pick up the red cup": encoder("pick up the red cup"),
    "pick up the blue box": encoder("pick up the blue box"),
    ...
}
```

At inference time, the language encoder is never called — the cached embedding is used
directly. This eliminates one network forward pass from the inference loop.

---

## 4. Policy Network: ACT (Action Chunking with Transformers)

### 4.1 What Is ACT

ACT (Action Chunking with Transformers) is a policy architecture designed for imitation
learning in robotics. It was introduced by Zhao et al. (2023) and addresses a
fundamental problem: single-step action prediction is fragile.

A standard behavior cloning policy predicts one action at a time:

```
observation_t → action_t
```

This leads to compounding errors. A small prediction error at time t shifts the
observation at time t+1 away from the training distribution, causing larger errors,
which cascade into failure.

ACT predicts an entire chunk of K future actions at once:

```
observation_t → [action_t, action_{t+1}, ..., action_{t+K-1}]
```

This is called **action chunking**. Each chunk is a coherent trajectory segment, so the
temporal consistency of the predicted actions is enforced by the model architecture itself.

### 4.2 CVAE Structure

ACT uses a Conditional Variational Autoencoder (CVAE) framework. During training, the
encoder sees the ground-truth action chunk and encodes it into a latent variable z. The
decoder then reconstructs the action chunk conditioned on the observation and z.

```
Training:
  Encoder: (observation, action_chunk_gt) → z ~ N(μ, σ²)
  Decoder: (observation, z) → action_chunk_pred

Inference:
  z ~ N(0, I)    ← sample from prior
  Decoder: (observation, z) → action_chunk_pred
```

The CVAE structure captures the multimodality of expert demonstrations. The same
observation might lead to multiple valid action sequences (e.g., approaching an object
from the left or the right). The latent variable z captures this variation.

### 4.3 Transformer Architecture

The policy network uses a standard Transformer encoder-decoder:

**Encoder**: Processes the observation tokens (vision + language + proprioception + z).
Each modality is projected to the transformer's hidden dimension (256) and treated as a
separate token in the sequence.

```
Encoder input tokens:
  [z_token, vision_wrist, vision_head, language, proprio]
  → 5 tokens, each 256-dim
```

**Decoder**: Cross-attends to the encoder output. The decoder has K query tokens (one per
timestep in the action chunk), each initialized as a learned embedding. Through
cross-attention, each query token gathers information from the observation to predict its
corresponding action.

```
Decoder query tokens:
  [action_query_0, action_query_1, ..., action_query_9]
  → 10 tokens, each 256-dim

Output:
  Each query → linear projection → action_dim
  Final shape: (10, action_dim)
```

### 4.4 Detailed Dimensions

| Component          | Dimension    | Notes                            |
|--------------------|-------------|----------------------------------|
| Visual embedding   | 1024        | 512 per camera                   |
| Language embedding | 512         | Frozen CLIP ViT-B/32             |
| Proprioception     | ~60         | Joint positions + velocities     |
| Latent z           | 32          | CVAE latent variable             |
| Transformer hidden | 256         | All projections map to this      |
| Transformer layers | 4           | Both encoder and decoder         |
| Attention heads    | 8           | Head dim = 256 / 8 = 32          |
| Action chunk size  | 10          | 10 future timesteps              |
| Action dimension   | N           | Number of controlled joints      |

### 4.5 Parameter Budget

Approximate parameter count breakdown:

| Module               | Parameters | Trainable? |
|----------------------|-----------|------------|
| Wrist ResNet-18      | ~11M      | Yes        |
| Head ResNet-18       | ~11M      | Yes        |
| CLIP text encoder    | ~63M      | No         |
| Input projections    | ~0.5M     | Yes        |
| Transformer encoder  | ~1.1M     | Yes        |
| Transformer decoder  | ~1.5M     | Yes        |
| Action output heads  | ~0.1M     | Yes        |
| **Total trainable**  | **~25M**  |            |

The ~25M trainable parameters are well within the capacity that 250 demonstrations can
supervise, especially with pretrained vision backbones providing a strong initialization.

---

## 5. Action Chunking Theory

### 5.1 Why Predict Multiple Steps

Single-step prediction suffers from three problems that chunking addresses:

**Compounding error**: In behavior cloning, the policy is trained on expert observations
but evaluated on its own observations. Small errors accumulate over hundreds of timesteps.
Action chunking reduces the number of decision points by a factor of K (the chunk size),
directly reducing the number of steps where errors can compound.

**Temporal consistency**: A single-step policy can produce jittery, inconsistent actions.
Each prediction is independent, with no guarantee of smoothness. A chunk is predicted as
a whole, so the transformer enforces temporal coherence across the K timesteps.

**Pausing and hesitation**: Without chunking, the policy must make a decision at every
timestep. Near decision boundaries (e.g., "should I grasp now or wait?"), this causes
oscillation. A chunk commits to a plan for K steps, eliminating frame-by-frame hesitation.

### 5.2 Chunk Size Selection

The chunk size K = 10 corresponds to 10 / 30 Hz = 0.33 seconds of future actions. This
is chosen to balance:

- **Short enough** that the chunk doesn't extend past the next significant state change
  (e.g., contact with the object)
- **Long enough** to capture meaningful trajectory segments and reduce compounding error
- **Matching the task granularity**: most manipulation subtasks (approach, grasp, lift)
  last several seconds, so a 0.33-second chunk subdivides each subtask into ~10 planning
  decisions

### 5.3 Temporal Ensemble at Inference

At inference time, chunks overlap. At timestep t, the policy predicts actions for t
through t+9. At timestep t+1, it predicts actions for t+1 through t+10. For timestep
t+1, there are now two predictions: one from the chunk generated at t, and one from the
chunk generated at t+1.

The temporal ensemble blends these overlapping predictions using exponential weighting
that favors more recent predictions. This is covered in detail in
`04_deployment.md`.

---

## 6. Observation and Action Spaces

### 6.1 Observation Space

The observation at each timestep consists of:

```python
observation = {
    "wrist_image": np.ndarray,   # (3, 224, 224) float32, normalized
    "head_image": np.ndarray,    # (3, 224, 224) float32, normalized
    "proprioception": np.ndarray, # (N_joints * 2,) float64 — [qpos, qvel]
    "language": str,              # Task instruction
}
```

Proprioception includes all controlled joint positions and velocities, concatenated into
a single vector. This gives the policy information about the robot's current
configuration that is not directly observable from images (e.g., exact joint angles).

### 6.2 Action Space

Actions are joint position targets:

```python
action = np.ndarray  # (N_joints,) float64 — target joint positions
```

The policy outputs absolute joint position targets, not deltas. This matches the MuJoCo
`general` actuator convention used throughout the lab series, where `ctrl[i]` is the
desired position for joint i.

### 6.3 Why Joint Positions (Not Cartesian or Torques)

- **Cartesian targets** require an IK solver in the loop, which adds complexity and can
  fail near singularities. Joint positions bypass IK entirely.
- **Torques** are too low-level for behavior cloning. The mapping from desired behavior
  to torque sequences is highly nonlinear and dynamics-dependent. Position targets let
  the MuJoCo actuator model handle the low-level tracking.
- **Joint positions** are what the expert controller records, so there is no
  representation mismatch between the demonstration data and the policy output.

---

## 7. Summary

The VLA architecture is a three-stage pipeline:

1. **Encode**: Convert images, language, and proprioception into fixed-size embeddings
2. **Fuse**: Concatenate embeddings and process through a transformer encoder
3. **Decode**: Produce a chunk of K future joint position targets

The key design decisions are:
- Dual ResNet-18 vision encoders (one per camera, ~22M parameters)
- Frozen CLIP language encoder (zero-shot semantic grounding)
- ACT policy with CVAE (captures demonstration multimodality)
- Action chunking with K=10 (reduces compounding error)
- Joint position action space (compatible with MuJoCo actuators)
