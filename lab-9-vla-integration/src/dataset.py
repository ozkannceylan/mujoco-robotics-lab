"""Lab 9 — M1: the chunked dataset the ACT policy trains on.

Action chunking means every sample is a window: one observation, and the next
``chunk_size`` actions the expert took from it. Two details that are easy to get
wrong and expensive to discover late:

**Padding.** Near the end of a segment there are fewer than ``chunk_size``
actions left. The window is padded and a mask marks which entries are real, so
the loss never asks the policy to predict actions that do not exist. Without the
mask the last two seconds of every demonstration teach it to freeze.

**Splitting by seed.** Two frames 100 ms apart in the same episode are
near-duplicates. A frame-level train/val split therefore reports a validation
loss that measures memorisation, and it will look excellent. The split here is
by *scene seed*, so a validation episode shares no object placement, no colour
and no lighting with anything trained on.

Normalisation statistics are computed on the **train split only** and stored in
the checkpoint, so evaluation never needs the dataset to interpret a prediction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from collect_demos import episode_path, load_manifest
from lab9_common import (
    CHUNK_SIZE,
    DATA_DIR,
    HELD_OUT_INSTRUCTIONS,
    IMAGE_SIZE,
    STATE_DIM,
    TASK_NAMES,
    instruction_label,
)

__all__ = ["NormStats", "DemoDataset", "build_datasets", "compute_norm_stats"]


@dataclass
class NormStats:
    """Per-dimension mean and scale for the state and the action."""

    state_mean: np.ndarray
    state_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray

    @staticmethod
    def _scale(values: np.ndarray) -> np.ndarray:
        """Standard deviation with a floor.

        A constant dimension — the grasp bit through a walk segment, a joint at
        a limit — has zero spread, and dividing by it produces inf that
        propagates silently into the loss.
        """
        scale = values.std(axis=0)
        return np.where(scale < 1e-4, 1.0, scale).astype(np.float32)

    @classmethod
    def fit(cls, states: np.ndarray, actions: np.ndarray) -> "NormStats":
        """Fit statistics to the training split.

        Args:
            states: ``(N, state_dim)``.
            actions: ``(N, action_dim)``.

        Returns:
            The fitted statistics.
        """
        return cls(
            state_mean=states.mean(axis=0).astype(np.float32),
            state_scale=cls._scale(states),
            action_mean=actions.mean(axis=0).astype(np.float32),
            action_scale=cls._scale(actions),
        )

    def to_dict(self) -> dict:
        return {k: v.tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, payload: dict) -> "NormStats":
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in payload.items()})


def _episode_arrays(root: Path, seed: int, target: str) -> dict:
    with np.load(episode_path(root, seed, target), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class DemoDataset(Dataset):
    """Windows of (observation, action chunk, instruction) over the demos.

    Args:
        root: Dataset directory.
        seeds: Which scene seeds this split owns.
        action_key: ``"task_action"`` or ``"joint_action"``.
        chunk_size: Actions predicted per sample.
        stats: Normalisation statistics; fitted from this split if omitted.
        paraphrases: How many instruction variants to sample from. The default
            uses only the training paraphrases, leaving the rest for the
            evaluation's generalisation check.
        augment: Apply brightness/contrast jitter to the images.
    """

    def __init__(
        self,
        root: Path,
        seeds: list[int],
        action_key: str = "task_action",
        chunk_size: int = CHUNK_SIZE,
        stats: NormStats | None = None,
        paraphrases: int = HELD_OUT_INSTRUCTIONS,
        augment: bool = False,
    ):
        manifest = load_manifest(root)
        self.root = root
        self.action_key = action_key
        self.chunk_size = chunk_size
        self.paraphrases = paraphrases
        self.augment = augment

        wanted = set(seeds)
        self.episodes = [
            e for e in manifest["episodes"] if e["seed"] in wanted and e["written"]
        ]
        if not self.episodes:
            raise ValueError(f"no episodes for seeds {sorted(wanted)[:8]}...")

        # (episode index, task, frame index) for every window start.
        self.index: list[tuple[int, str, int]] = []
        self._cache: dict[int, dict] = {}
        for position, episode in enumerate(self.episodes):
            for task in TASK_NAMES:
                start, stop = episode["segments"][task]
                for frame in range(start, stop):
                    self.index.append((position, task, frame))

        self.stats = stats or compute_norm_stats(root, seeds, action_key)

    # -- data ------------------------------------------------------------

    def _arrays(self, position: int) -> dict:
        if position not in self._cache:
            episode = self.episodes[position]
            self._cache[position] = _episode_arrays(
                self.root, episode["seed"], episode["target"]
            )
        return self._cache[position]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict:
        position, task, frame = self.index[item]
        episode = self.episodes[position]
        arrays = self._arrays(position)
        _, stop = episode["segments"][task]

        images = {}
        for camera in ("head", "wrist"):
            image = arrays[camera][frame].astype(np.float32) / 255.0
            if self.augment:
                # Brightness and contrast only. Geometric augmentation would
                # break the relationship between what the camera sees and where
                # the hand has to go, which is the entire signal here.
                rng = np.random.default_rng((item * 7919) % (2**31))
                image = np.clip(
                    (image - 0.5) * rng.uniform(0.85, 1.15)
                    + 0.5 + rng.uniform(-0.08, 0.08),
                    0.0, 1.0,
                )
            images[camera] = torch.from_numpy(image.transpose(2, 0, 1).copy())

        actions = arrays[self.action_key]
        available = min(self.chunk_size, stop - frame)
        chunk = np.zeros((self.chunk_size, actions.shape[1]), dtype=np.float32)
        mask = np.zeros(self.chunk_size, dtype=np.float32)
        chunk[:available] = actions[frame : frame + available]
        mask[:available] = 1.0
        # Hold the last real action through the padding. The mask zeroes its
        # contribution to the loss either way; a constant tail keeps the padded
        # region from looking like a jump to the origin in any diagnostic plot.
        if available < self.chunk_size:
            chunk[available:] = actions[frame + available - 1]

        # One paraphrase per sample, drawn deterministically from the item index
        # so an epoch sees a mix without the sampling depending on worker order.
        variant = item % max(1, self.paraphrases)
        instruction = instruction_label(task, episode["target"], variant)

        return {
            "head": images["head"],
            "wrist": images["wrist"],
            "state": torch.from_numpy(arrays["state"][frame].astype(np.float32)),
            "action": torch.from_numpy(chunk),
            "mask": torch.from_numpy(mask),
            "instruction": instruction,
            "task": task,
            "target": episode["target"],
            "seed": episode["seed"],
        }


def compute_norm_stats(
    root: Path, seeds: list[int], action_key: str = "task_action"
) -> NormStats:
    """Fit normalisation statistics over a set of seeds.

    Args:
        root: Dataset directory.
        seeds: Seeds to fit on — the **train** split, never all of them.
        action_key: Which action head's statistics to fit.

    Returns:
        The fitted statistics.
    """
    manifest = load_manifest(root)
    wanted = set(seeds)
    states, actions = [], []
    for episode in manifest["episodes"]:
        if episode["seed"] not in wanted or not episode["written"]:
            continue
        arrays = _episode_arrays(root, episode["seed"], episode["target"])
        for task in TASK_NAMES:
            start, stop = episode["segments"][task]
            states.append(arrays["state"][start:stop])
            actions.append(arrays[action_key][start:stop])
    return NormStats.fit(np.concatenate(states), np.concatenate(actions))


def build_datasets(
    root: Path = DATA_DIR,
    action_key: str = "task_action",
    chunk_size: int = CHUNK_SIZE,
    augment: bool = True,
) -> tuple[DemoDataset, DemoDataset, NormStats]:
    """Build the train and validation datasets from the manifest's split.

    Args:
        root: Dataset directory.
        action_key: Which action head to train.
        chunk_size: Actions per sample.
        augment: Image jitter on the training split.

    Returns:
        ``(train, val, stats)``. The statistics come from the train split and
        are shared with validation, which is the only correct direction.
    """
    manifest = load_manifest(root)
    stats = compute_norm_stats(root, manifest["train_seeds"], action_key)
    train = DemoDataset(
        root, manifest["train_seeds"], action_key, chunk_size, stats, augment=augment
    )
    val = DemoDataset(
        root, manifest["val_seeds"], action_key, chunk_size, stats, augment=False
    )
    overlap = set(manifest["train_seeds"]) & set(manifest["val_seeds"])
    if overlap:
        raise RuntimeError(f"train/val seed leakage: {sorted(overlap)}")
    return train, val, stats


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the demonstration set.")
    parser.add_argument("--root", type=Path, default=DATA_DIR)
    parser.add_argument("--grid", action="store_true", help="write a contact sheet")
    args = parser.parse_args()

    train, val, stats = build_datasets(args.root)
    manifest = load_manifest(args.root)
    print(f"train windows {len(train):>6}   from {len(train.episodes)} episodes")
    print(f"val   windows {len(val):>6}   from {len(val.episodes)} episodes")
    print(f"image {IMAGE_SIZE}px  state {STATE_DIM}  chunk {train.chunk_size}")
    print(f"action '{train.action_key}' dim {stats.action_mean.shape[0]}")
    print("state  scale range "
          f"[{stats.state_scale.min():.3g}, {stats.state_scale.max():.3g}]")
    print("action scale range "
          f"[{stats.action_scale.min():.3g}, {stats.action_scale.max():.3g}]")

    sample = train[0]
    print(f"sample head {tuple(sample['head'].shape)} "
          f"action {tuple(sample['action'].shape)} "
          f"mask sum {sample['mask'].sum():.0f}")
    print(f"instruction: {sample['instruction']!r}")

    leaks = set(manifest["train_seeds"]) & set(manifest["val_seeds"])
    print(f"seed leakage: {'NONE' if not leaks else sorted(leaks)}")

    if args.grid:
        _write_grid(train, manifest)


def _write_grid(dataset: DemoDataset, manifest: dict) -> None:
    """Write the M1 evidence contact sheet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lab9_common import MEDIA_DIR

    picks = np.linspace(0, len(dataset) - 1, 8).astype(int)
    figure, axes = plt.subplots(2, 8, figsize=(17, 5))
    for column, item in enumerate(picks):
        sample = dataset[int(item)]
        for row, camera in enumerate(("head", "wrist")):
            axes[row, column].imshow(sample[camera].numpy().transpose(1, 2, 0))
            axes[row, column].axis("off")
        axes[0, column].set_title(
            f"{sample['task']}/{sample['target']}\nseed {sample['seed']}",
            fontsize=8,
        )
    axes[0, 0].set_ylabel("head")
    axes[1, 0].set_ylabel("wrist")
    figure.suptitle(
        f"Lab 9 M1 — {len(dataset.episodes)} training episodes, "
        f"{len(dataset)} windows. Object placement, order, colour and lighting "
        "vary per seed.",
        fontsize=11,
    )
    figure.tight_layout()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    path = MEDIA_DIR / "m1_dataset_grid.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    print(f"wrote {path}")
    del manifest


if __name__ == "__main__":
    _main()
