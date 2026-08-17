"""Lab 9 — M2: the action-chunking transformer policy.

Adapted from `ozkannceylan/humanoid_vla`'s `ACTPolicy`, which is itself Zhao et
al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
(RSS 2023). What is reused is the design: a frozen-except-`layer4` ResNet18
producing spatial vision tokens, a state token, an instruction token, and a
transformer decoder over learned queries that emits a whole chunk of future
actions at once. What is different here, and why:

* **Two cameras.** Head and wrist, each contributing its own spatial tokens with
  a per-camera embedding added. At 128 px the objects are a handful of pixels in
  the head view by the time the hand is near them; the wrist view is what makes
  the last few centimetres observable.
* **Token count derived from the feature map.** Upstream hardcodes 49 tokens,
  which is ResNet18's 7x7 output for a 224 px input and silently wrong for any
  other size. At 128 px it is 4x4 = 16.
* **Two action heads.** `task` emits what Lab 8's whole-body QP consumes;
  `joint` emits the brief's literal 29 joint targets. The second exists to be
  measured against Lab 7's prediction that a joint-position reference cannot
  stabilise this robot — see `tasks/PLAN.md` deviation 3.
* **Normalisation inside the module.** The dataset's statistics live as buffers,
  so they travel with the checkpoint. A checkpoint that cannot denormalise its
  own output is a trap: it loads, it runs, and it is wrong by a scale factor.
* **No temporal ensembling.** Upstream's optional smoothing costs an extra
  forward pass per step, and on this machine the control loop is already
  render-bound. Recorded as a deliberate simplification, not an oversight.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from lab9_common import CHUNK_SIZE, IMAGE_SIZE, STATE_DIM
from observations import ACTION_DIMS

__all__ = ["ACTPolicy", "IMAGENET_MEAN", "IMAGENET_STD", "save_checkpoint", "load_checkpoint"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _spatial_tokens(image_size: int) -> int:
    """How many tokens ResNet18's layer4 produces for a square input.

    Args:
        image_size: Input edge length in pixels.

    Returns:
        Token count (the feature map is ``ceil(size / 32)`` on a side).
    """
    side = max(1, int(np.ceil(image_size / 32)))
    return side * side


class ACTPolicy(nn.Module):
    """Language-conditioned action-chunking transformer.

    Args:
        action_head: ``"task"`` or ``"joint"``.
        state_dim: Proprioception dimension.
        chunk_size: Actions predicted per forward pass.
        hidden_dim: Transformer model dimension.
        nhead: Attention heads.
        num_layers: Decoder layers.
        conditioning: ``"text"`` (frozen sentence embedding) or ``"task_id"``
            (integer lookup, the ablation baseline).
        text_dim: Sentence-embedding dimension.
        num_tasks: Vocabulary size for ``task_id`` conditioning.
        cameras: Camera names, in a fixed order.
        image_size: Input edge length.
        pretrained_backbone: Load ImageNet weights (off in unit tests).
    """

    def __init__(
        self,
        action_head: str = "task",
        state_dim: int = STATE_DIM,
        chunk_size: int = CHUNK_SIZE,
        hidden_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        conditioning: str = "text",
        text_dim: int = 512,
        num_tasks: int = 8,
        cameras: tuple[str, ...] = ("head", "wrist"),
        image_size: int = IMAGE_SIZE,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        if action_head not in ACTION_DIMS:
            raise ValueError(f"action_head must be one of {sorted(ACTION_DIMS)}")
        if conditioning not in ("text", "task_id"):
            raise ValueError("conditioning must be 'text' or 'task_id'")

        self.action_head = action_head
        self.action_dim = ACTION_DIMS[action_head]
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.conditioning = conditioning
        self.cameras = tuple(cameras)
        self.image_size = image_size
        self.tokens_per_camera = _spatial_tokens(image_size)

        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        # conv1..layer4 — keep the feature map, do not average it away.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone[7].parameters():   # layer4
            parameter.requires_grad = True

        self.image_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.image_pos = nn.Parameter(
            torch.randn(1, self.tokens_per_camera, hidden_dim) * 0.02
        )
        # One embedding per camera, so the decoder can tell a head-view token
        # from a wrist-view token at the same spatial position.
        self.camera_embed = nn.Parameter(
            torch.randn(len(self.cameras), 1, hidden_dim) * 0.02
        )

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if conditioning == "text":
            self.task_proj = nn.Linear(text_dim, hidden_dim)
        else:
            self.task_embed = nn.Embedding(num_tasks, hidden_dim)

        self.queries = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, self.action_dim)

        # Normalisation travels with the weights. Identity by default so a
        # freshly built model is usable before statistics exist.
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_scale", torch.ones(self.action_dim))
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    # -- normalisation ---------------------------------------------------

    def set_norm_stats(self, stats) -> None:
        """Copy dataset statistics into the module's buffers.

        Args:
            stats: A `dataset.NormStats`.
        """
        self.state_mean.copy_(torch.as_tensor(stats.state_mean))
        self.state_scale.copy_(torch.as_tensor(stats.state_scale))
        self.action_mean.copy_(torch.as_tensor(stats.action_mean))
        self.action_scale.copy_(torch.as_tensor(stats.action_scale))

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Raw actions to the normalised space the network predicts in."""
        return (action - self.action_mean) / self.action_scale

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Network output back to raw action units."""
        return action * self.action_scale + self.action_mean

    # -- forward ---------------------------------------------------------

    def _vision_tokens(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """Spatial tokens from every camera, with positional and camera embeddings."""
        tokens = []
        for index, camera in enumerate(self.cameras):
            image = (images[camera] - self.image_mean) / self.image_std
            features = self.image_proj(self.backbone(image))       # (B, C, h, w)
            features = features.flatten(2).transpose(1, 2)          # (B, hw, C)
            tokens.append(features + self.image_pos + self.camera_embed[index])
        return torch.cat(tokens, dim=1)

    def forward(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        instruction: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a **normalised** action chunk.

        Args:
            images: ``{camera: (B, 3, H, W)}`` in [0, 1].
            state: ``(B, state_dim)``, raw units.
            instruction: ``(B, text_dim)`` sentence embeddings, or ``(B,)`` long
                task ids under ``task_id`` conditioning.

        Returns:
            ``(B, chunk_size, action_dim)``, normalised.
        """
        memory = self._vision_tokens(images)

        normalised_state = (state - self.state_mean) / self.state_scale
        state_token = self.state_proj(normalised_state).unsqueeze(1)

        if self.conditioning == "text":
            task_token = self.task_proj(instruction).unsqueeze(1)
        else:
            task_token = self.task_embed(instruction.long()).unsqueeze(1)

        memory = torch.cat([memory, state_token, task_token], dim=1)
        queries = self.queries.unsqueeze(0).expand(state.shape[0], -1, -1)
        return self.head(self.decoder(queries, memory))

    @torch.no_grad()
    def predict(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        instruction: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a chunk in raw action units.

        Args:
            images: ``{camera: (B, 3, H, W)}`` in [0, 1].
            state: ``(B, state_dim)``, raw units.
            instruction: Conditioning, as in :meth:`forward`.

        Returns:
            ``(B, chunk_size, action_dim)`` in raw units.
        """
        self.eval()
        return self.denormalize_action(self(images, state, instruction))

    # -- reporting -------------------------------------------------------

    def parameter_counts(self) -> dict:
        """Total and trainable parameter counts, by block."""
        def count(module) -> tuple[int, int]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        blocks = {
            "backbone": self.backbone,
            "decoder": self.decoder,
            "state_proj": self.state_proj,
            "head": self.head,
        }
        summary = {name: count(module) for name, module in blocks.items()}
        summary["total"] = (
            sum(p.numel() for p in self.parameters()),
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )
        return summary


def save_checkpoint(path, model: ACTPolicy, bank, stats, extra: dict | None = None) -> None:
    """Write a self-contained checkpoint.

    Self-contained means: weights, normalisation statistics, the instruction
    bank, and the configuration needed to rebuild the module. Evaluation then
    needs neither the dataset nor `transformers` nor the network.

    Args:
        path: Destination file.
        model: The trained policy.
        bank: An `text_encoder.InstructionBank`.
        stats: A `dataset.NormStats`.
        extra: Anything else worth carrying (training config, metrics).
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "action_head": model.action_head,
                "state_dim": model.state_dim,
                "chunk_size": model.chunk_size,
                "conditioning": model.conditioning,
                "cameras": list(model.cameras),
                "image_size": model.image_size,
                "hidden_dim": model.queries.shape[1],
            },
            "instruction_bank": bank.to_dict(),
            "norm_stats": stats.to_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, device: str = "cpu"):
    """Rebuild a policy, its instruction bank and its statistics from disk.

    Args:
        path: Checkpoint file.
        device: Torch device.

    Returns:
        ``(model, bank, stats, extra)``.
    """
    from dataset import NormStats
    from text_encoder import InstructionBank

    payload = torch.load(path, map_location=device, weights_only=False)
    config = dict(payload["config"])
    hidden_dim = config.pop("hidden_dim", 256)
    config["cameras"] = tuple(config["cameras"])
    model = ACTPolicy(hidden_dim=hidden_dim, pretrained_backbone=False, **config)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return (
        model,
        InstructionBank.from_dict(payload["instruction_bank"]),
        NormStats.from_dict(payload["norm_stats"]),
        payload.get("extra", {}),
    )
