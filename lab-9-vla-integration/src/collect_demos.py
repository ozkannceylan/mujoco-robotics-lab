"""Lab 9 — M1: generate the demonstration set from Lab 8's controller.

One expert rollout per (seed, object) pair, captured at the policy rate and
**sliced by phase** into the lab's two labelled tasks. Slicing rather than
running a separate episode per task matters on this machine: rendering is
software and costs ~97 ms per frame, so a rollout is expensive and every task
segment it can yield is one that does not have to be simulated again.

What is stored, and what is not
-------------------------------
Stored per frame: both camera images, the proprioception vector, and the
**expert's own command** — the hand targets it was driving, whether it was
walking, and the weld states. Not the achieved state. Behaviour cloning imitates
what the expert *did*; on a compliant, disturbed system the command and the
outcome differ, and training on the outcome teaches the policy to chase its own
past instead of acting.

Only successful episodes are written. A failed episode is a recording of a robot
falling over, and its frames are indistinguishable from good ones until the
moment it goes down.

Run:
    MUJOCO_GL=egl python3 lab-9-vla-integration/src/collect_demos.py --seeds 50
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lab9_common import (
    DATA_DIR,
    HELD_OUT_INSTRUCTIONS,
    IMAGE_SIZE,
    OBJECT_NAMES,
    STATE_DIM,
    TASK_NAMES,
    instruction_label,
)
from observations import JOINT_ACTION_DIM, TASK_ACTION_DIM

__all__ = ["collect", "episode_path", "load_manifest", "VAL_FRACTION"]

#: Fraction of *seeds* held out for validation. Splitting by seed, never by
#: frame: two frames 100 ms apart in the same episode are near-duplicates, and a
#: frame-level split reports a validation loss that measures memorisation.
VAL_FRACTION: float = 0.2

MANIFEST = "manifest.json"


def episode_path(root: Path, seed: int, target: str) -> Path:
    """Where one episode's arrays live.

    Args:
        root: Dataset directory.
        seed: Scene seed.
        target: Object name.

    Returns:
        The `.npz` path.
    """
    return root / "episodes" / f"seed{seed:04d}_{target}.npz"


def _collect_one(job: tuple[int, str, str, bool]) -> dict:
    """Worker: run one episode and write it if it succeeded."""
    seed, target, root_str, wide = job
    root = Path(root_str)
    import expert as expert_module

    started = time.time()
    record = expert_module.run_episode(seed, target=target, wide=wide, capture=True)
    segments = expert_module.task_segments(record)

    summary = {
        "seed": seed,
        "target": target,
        "wide": wide,
        "near_object": record.near_object,
        "success": record.success,
        "reason": record.reason,
        "frames": len(record),
        "segments": {task: list(bounds) for task, bounds in segments.items()},
        "approach_steps": record.approach_steps,
        "reach_error_mm": record.metrics.get("reach_error_mm", float("nan")),
        "lift_m": record.metrics.get("lift_m", 0.0),
        "wall_s": round(time.time() - started, 1),
    }
    if not record.success or set(segments) != set(TASK_NAMES):
        summary["written"] = False
        if record.success:
            summary["reason"] = f"missing segments: {sorted(segments)}"
        return summary

    path = episode_path(root, seed, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        head=np.stack(record.head).astype(np.uint8),
        wrist=np.stack(record.wrist).astype(np.uint8),
        state=np.stack(record.state).astype(np.float32),
        task_action=np.stack(record.task_action).astype(np.float32),
        joint_action=np.stack(record.joint_action).astype(np.float32),
        time=np.asarray(record.time, dtype=np.float32),
        phase=np.array(record.phase),
        segment_starts=np.array(
            [segments[task][0] for task in TASK_NAMES], dtype=np.int32
        ),
        segment_stops=np.array(
            [segments[task][1] for task in TASK_NAMES], dtype=np.int32
        ),
    )
    summary["written"] = True
    summary["bytes"] = path.stat().st_size
    return summary


def collect(
    seeds: int,
    root: Path = DATA_DIR,
    workers: int = 3,
    wide: bool = False,
    start: int = 0,
) -> dict:
    """Generate demonstrations and write the manifest.

    Args:
        seeds: How many scene seeds; each yields one episode per object.
        root: Dataset directory.
        workers: Parallel worker processes.
        wide: Draw object placement from the wider evaluation range.
        start: First seed, so a run can be extended without re-simulating.

    Returns:
        The manifest dict.
    """
    jobs = [
        (seed, target, str(root), wide)
        for seed in range(start, start + seeds)
        for target in OBJECT_NAMES
    ]
    print(f"collecting {len(jobs)} episodes with {workers} workers ...")
    started = time.time()
    summaries: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, summary in enumerate(pool.map(_collect_one, jobs), start=1):
            summaries.append(summary)
            mark = "ok " if summary.get("written") else "SKIP"
            print(
                f"  [{index:>3}/{len(jobs)}] {mark} seed {summary['seed']:>4} "
                f"{summary['target']:<4} frames={summary['frames']:>4} "
                f"{summary['wall_s']:>5.1f}s {summary['reason'][:38]}"
            )

    written = [s for s in summaries if s.get("written")]
    seeds_written = sorted({s["seed"] for s in written})
    # Split by seed. Deterministic and independent of collection order, so
    # extending the dataset never reshuffles what was already validated on.
    rng = np.random.default_rng(20260817)
    shuffled = list(seeds_written)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(VAL_FRACTION * len(shuffled))))
    val_seeds = sorted(shuffled[:n_val])
    train_seeds = sorted(shuffled[n_val:])

    manifest = {
        "created": "2026-08-17",
        "image_size": IMAGE_SIZE,
        "state_dim": STATE_DIM,
        "task_action_dim": TASK_ACTION_DIM,
        "joint_action_dim": JOINT_ACTION_DIM,
        "tasks": list(TASK_NAMES),
        "objects": list(OBJECT_NAMES),
        "held_out_paraphrase_index": HELD_OUT_INSTRUCTIONS,
        "wide": wide,
        "episodes": written,
        "attempted": len(summaries),
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "wall_s": round(time.time() - started, 1),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(root: Path = DATA_DIR) -> dict:
    """Read the dataset manifest.

    Args:
        root: Dataset directory.

    Returns:
        The manifest dict.

    Raises:
        FileNotFoundError: If no dataset has been collected.
    """
    path = root / MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"no dataset at {root}. Run collect_demos.py first."
        )
    return json.loads(path.read_text())


def report(manifest: dict) -> None:
    """Print the dataset table the M1 gate is read from."""
    written = manifest["episodes"]
    print("\n" + "=" * 66)
    print(f"{'demonstrations':<34}{len(written)} of {manifest['attempted']} attempted")
    per_task = {task: len(written) for task in manifest["tasks"]}
    for task, count in per_task.items():
        print(f"{'  demos labelled ' + task:<34}{count}")
    for obj in manifest["objects"]:
        count = sum(1 for e in written if e["target"] == obj)
        print(f"{'  episodes targeting ' + obj:<34}{count}")

    frames = sum(e["frames"] for e in written)
    segment_frames = {
        task: sum(e["segments"][task][1] - e["segments"][task][0] for e in written)
        for task in manifest["tasks"]
    }
    print(f"{'frames captured':<34}{frames}")
    for task, count in segment_frames.items():
        print(f"{'  frames labelled ' + task:<34}{count}")
    size_mb = sum(e.get("bytes", 0) for e in written) / 1e6
    print(f"{'on disk':<34}{size_mb:.0f} MB")
    print(f"{'train / val seeds':<34}"
          f"{len(manifest['train_seeds'])} / {len(manifest['val_seeds'])}")
    overlap = set(manifest["train_seeds"]) & set(manifest["val_seeds"])
    print(f"{'seed leakage':<34}{'NONE' if not overlap else sorted(overlap)}")
    print(f"{'collection wall time':<34}{manifest['wall_s'] / 60:.1f} min")
    print("=" * 66)

    instructions = sorted({
        instruction_label(task, e["target"])
        for e in written for task in manifest["tasks"]
    })
    print(f"instructions in the set ({len(instructions)}): {instructions}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Lab 9 demonstrations.")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--root", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    manifest = collect(
        args.seeds, root=args.root, workers=args.workers,
        wide=args.wide, start=args.start,
    )
    report(manifest)


if __name__ == "__main__":
    main()
