"""Lab 9 — M3: train the language-conditioned ACT policy.

Behaviour cloning with an L1 chunk loss. Three things in here exist because of
how easy each is to get silently wrong:

**The loss is masked.** Near the end of a segment there are fewer than
``chunk_size`` real actions left. Unmasked, the padded tail teaches the policy
to stop moving two seconds before the task ends.

**Validation is reported in raw units, against a baseline.** A normalised L1 of
0.31 says nothing. The same number in millimetres of hand target, next to what
predicting the training mean would score, says whether the model learned
anything at all. A policy that cannot beat the mean has learned the dataset's
average pose and nothing else, and it will still produce a smooth-looking
training curve.

**The backbone learns slower than the head.** `layer4` carries ImageNet
features that a few thousand samples can destroy faster than they can improve;
it gets a tenth of the head's learning rate.

Run:
    python3 lab-9-vla-integration/src/train.py --epochs 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from act_policy import ACTPolicy, save_checkpoint
from dataset import build_datasets
from lab9_common import CHECKPOINT_DIR, CHUNK_SIZE, DATA_DIR, MEDIA_DIR, TASK_NAMES
from text_encoder import InstructionBank, build_instruction_bank

__all__ = ["train", "evaluate_split", "collate"]


def collate(batch: list[dict]) -> dict:
    """Stack a batch, keeping the instruction strings as strings.

    Args:
        batch: Samples from :class:`dataset.DemoDataset`.

    Returns:
        A dict of stacked tensors plus the per-sample metadata lists.
    """
    return {
        "head": torch.stack([b["head"] for b in batch]),
        "wrist": torch.stack([b["wrist"] for b in batch]),
        "state": torch.stack([b["state"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
        "task": [b["task"] for b in batch],
        "target": [b["target"] for b in batch],
    }


def _conditioning(
    batch: dict, bank: InstructionBank, conditioning: str, device: str
) -> torch.Tensor:
    """Build the conditioning tensor for one batch.

    Args:
        batch: A collated batch.
        bank: The instruction bank.
        conditioning: ``"text"`` or ``"task_id"``.
        device: Torch device.

    Returns:
        ``(B, 512)`` embeddings or ``(B,)`` long ids.
    """
    if conditioning == "text":
        return torch.from_numpy(bank.batch(batch["instruction"])).to(device)
    # The ablation: an integer per (task, object) pair, which is exactly the
    # information a text-free policy would be handed.
    ids = [
        TASK_NAMES.index(task) * 2 + (0 if target == "cup" else 1)
        for task, target in zip(batch["task"], batch["target"], strict=True)
    ]
    return torch.tensor(ids, dtype=torch.long, device=device)


def masked_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error over the real entries of a chunk.

    Args:
        prediction: ``(B, T, A)``.
        target: ``(B, T, A)``.
        mask: ``(B, T)``, 1 where the action is real.

    Returns:
        Scalar loss.
    """
    error = (prediction - target).abs().mean(dim=-1)
    return (error * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate_split(
    model: ACTPolicy,
    loader: DataLoader,
    bank: InstructionBank,
    device: str,
    action_mean: torch.Tensor,
) -> dict:
    """Validation error in raw units, next to a predict-the-mean baseline.

    Args:
        model: The policy.
        loader: Validation loader.
        bank: Instruction bank.
        device: Torch device.
        action_mean: Training-set mean action, the baseline's prediction.

    Returns:
        Per-dimension-group errors and the baseline's.
    """
    model.eval()
    totals = {"model": 0.0, "baseline": 0.0, "count": 0.0}
    per_task: dict[str, list[float]] = {task: [] for task in TASK_NAMES}
    hand_error, gait_error, grasp_error = [], [], []

    for batch in loader:
        images = {c: batch[c].to(device) for c in model.cameras}
        state = batch["state"].to(device)
        target = batch["action"].to(device)
        mask = batch["mask"].to(device)
        instruction = _conditioning(batch, bank, model.conditioning, device)

        prediction = model.denormalize_action(model(images, state, instruction))
        error = (prediction - target).abs().mean(dim=-1)
        weight = mask.sum()
        totals["model"] += float((error * mask).sum())
        baseline = (action_mean.view(1, 1, -1) - target).abs().mean(dim=-1)
        totals["baseline"] += float((baseline * mask).sum())
        totals["count"] += float(weight)

        per_sample = (error * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        for task, value in zip(batch["task"], per_sample.tolist(), strict=True):
            per_task[task].append(value)

        if model.action_head == "task":
            axis_error = (prediction - target).abs()
            expand = mask.unsqueeze(-1)
            hand_error.append(
                float((axis_error[..., :6] * expand).sum() / (weight * 6))
            )
            gait_error.append(float((axis_error[..., 6] * mask).sum() / weight))
            grasp_error.append(
                float((axis_error[..., 7:] * expand).sum() / (weight * 2))
            )

    result = {
        "val_l1": totals["model"] / max(totals["count"], 1.0),
        "baseline_l1": totals["baseline"] / max(totals["count"], 1.0),
        "per_task": {
            task: float(np.mean(values)) if values else float("nan")
            for task, values in per_task.items()
        },
    }
    if hand_error:
        result["hand_mm"] = float(np.mean(hand_error)) * 1000.0
        result["gait_err"] = float(np.mean(gait_error))
        result["grasp_err"] = float(np.mean(grasp_error))
    return result


def train(
    root: Path = DATA_DIR,
    action_head: str = "task",
    conditioning: str = "text",
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    backbone_lr_scale: float = 0.1,
    chunk_size: int = CHUNK_SIZE,
    workers: int = 2,
    tag: str | None = None,
    threads: int = 4,
) -> dict:
    """Train one policy and write its checkpoint.

    Args:
        root: Dataset directory.
        action_head: ``"task"`` or ``"joint"``.
        conditioning: ``"text"`` or ``"task_id"``.
        epochs: Training epochs.
        batch_size: Samples per step.
        learning_rate: Head/decoder learning rate.
        backbone_lr_scale: Multiplier for the fine-tuned `layer4`.
        chunk_size: Actions per sample.
        workers: DataLoader workers.
        tag: Checkpoint name; derived from the configuration if omitted.
        threads: Torch CPU threads.

    Returns:
        The training summary that is also written next to the checkpoint.
    """
    torch.set_num_threads(threads)
    device = "cpu"
    tag = tag or f"{action_head}_{conditioning}"

    train_set, val_set, stats = build_datasets(
        root, action_key=f"{action_head}_action", chunk_size=chunk_size
    )
    print(f"[{tag}] train {len(train_set)} windows / {len(train_set.episodes)} episodes"
          f"   val {len(val_set)} / {len(val_set.episodes)}")

    bank = build_instruction_bank()
    model = ACTPolicy(
        action_head=action_head,
        chunk_size=chunk_size,
        conditioning=conditioning,
    ).to(device)
    model.set_norm_stats(stats)
    counts = model.parameter_counts()
    print(f"[{tag}] {counts['total'][0] / 1e6:.2f}M params, "
          f"{counts['total'][1] / 1e6:.2f}M trainable, "
          f"{model.tokens_per_camera} tokens/camera x {len(model.cameras)} cameras")

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids
    ]
    optimiser = torch.optim.AdamW(
        [
            {"params": other_params, "lr": learning_rate},
            {"params": backbone_params, "lr": learning_rate * backbone_lr_scale},
        ],
        weight_decay=1e-4,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
        collate_fn=collate, drop_last=True, persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=collate, persistent_workers=workers > 0,
    )
    action_mean = torch.as_tensor(stats.action_mean, device=device)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"act_{tag}.pt"
    history: list[dict] = []
    best = float("inf")
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running, steps = 0.0, 0
        epoch_started = time.time()
        for batch in train_loader:
            images = {c: batch[c].to(device) for c in model.cameras}
            state = batch["state"].to(device)
            target = model.normalize_action(batch["action"].to(device))
            mask = batch["mask"].to(device)
            instruction = _conditioning(batch, bank, model.conditioning, device)

            prediction = model(images, state, instruction)
            loss = masked_l1(prediction, target, mask)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimiser.step()
            running += float(loss)
            steps += 1
        schedule.step()

        metrics = evaluate_split(model, val_loader, bank, device, action_mean)
        entry = {
            "epoch": epoch,
            "train_l1_norm": running / max(steps, 1),
            "seconds": round(time.time() - epoch_started, 1),
            **metrics,
        }
        history.append(entry)
        marker = ""
        if metrics["val_l1"] < best:
            best = metrics["val_l1"]
            marker = "  *"
            save_checkpoint(
                checkpoint_path, model, bank, stats,
                extra={
                    "tag": tag,
                    "epoch": epoch,
                    "val": metrics,
                    "action_head": action_head,
                    "conditioning": conditioning,
                    "train_episodes": len(train_set.episodes),
                    "val_episodes": len(val_set.episodes),
                },
            )
        extra = ""
        if "hand_mm" in metrics:
            extra = (f"  hand {metrics['hand_mm']:6.1f}mm  gait {metrics['gait_err']:.3f}"
                     f"  grasp {metrics['grasp_err']:.3f}")
        print(f"[{tag}] epoch {epoch:>3}/{epochs}  "
              f"train {entry['train_l1_norm']:.4f}  "
              f"val {metrics['val_l1']:.4f}  "
              f"(mean-baseline {metrics['baseline_l1']:.4f})"
              f"{extra}  {entry['seconds']:.0f}s{marker}")

    summary = {
        "tag": tag,
        "action_head": action_head,
        "conditioning": conditioning,
        "epochs": epochs,
        "best_val_l1": best,
        "final": history[-1],
        "history": history,
        "wall_minutes": round((time.time() - started) / 60.0, 1),
        "checkpoint": str(checkpoint_path),
    }
    (CHECKPOINT_DIR / f"train_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"[{tag}] done in {summary['wall_minutes']:.1f} min, "
          f"best val {best:.4f} -> {checkpoint_path.name}")
    return summary


def plot_histories(tags: list[str]) -> Path:
    """Plot the training curves of several runs together.

    Args:
        tags: Checkpoint tags to plot.

    Returns:
        The figure path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for tag in tags:
        path = CHECKPOINT_DIR / f"train_{tag}.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        history = summary["history"]
        epochs = [h["epoch"] for h in history]
        axes[0].plot(epochs, [h["train_l1_norm"] for h in history], label=f"{tag} train")
        axes[0].plot(epochs, [h["val_l1"] for h in history], "--", label=f"{tag} val")
        if "hand_mm" in history[0]:
            axes[1].plot(epochs, [h["hand_mm"] for h in history], label=tag)
        axes[0].axhline(
            history[-1]["baseline_l1"], color="0.6", ls=":", lw=1,
        )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("masked L1 (normalised)")
    axes[0].set_title("training / validation loss\n(dotted: predict-the-mean baseline)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("hand target error [mm]")
    axes[1].set_title("validation hand-target error, raw units")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    figure.suptitle("Lab 9 M3 — training on 4 CPU cores, no GPU", fontsize=11)
    figure.tight_layout()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    path = MEDIA_DIR / "m3_training_curves.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Lab 9 VLA policy.")
    parser.add_argument("--root", type=Path, default=DATA_DIR)
    parser.add_argument("--action-head", choices=("task", "joint"), default="task")
    parser.add_argument("--conditioning", choices=("text", "task_id"), default="text")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--plot", nargs="*", default=None,
                        help="tags to plot instead of training")
    args = parser.parse_args()

    if args.plot is not None:
        print(f"wrote {plot_histories(args.plot)}")
        return

    train(
        root=args.root,
        action_head=args.action_head,
        conditioning=args.conditioning,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        workers=args.workers,
        tag=args.tag,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
