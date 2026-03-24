"""Lab 9 — Dataset collection script.

Collects expert demonstrations for all VLA tasks with domain randomization.
Saves demonstrations as npz files organized by task name.

Usage:
    python collect_dataset.py [--demos-per-task 50] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_collector import DemoCollector
from domain_randomizer import DomainRandomizer
from lab9_common import DATA_DIR, DEMOS_DIR, TASKS


def collect_dataset(
    demos_per_task: int = 50,
    seed: int = 42,
    image_width: int = 160,
    image_height: int = 120,
    max_episode_steps: int = 200,
) -> dict:
    """Collect demonstrations for all tasks.

    Args:
        demos_per_task: Number of demonstrations per task.
        seed: Base random seed.
        image_width: Camera image width (lower for speed).
        image_height: Camera image height (lower for speed).
        max_episode_steps: Maximum policy steps per episode.

    Returns:
        Manifest dict with dataset statistics.
    """
    collector = DemoCollector(
        image_width=image_width,
        image_height=image_height,
        max_episode_steps=max_episode_steps,
    )
    randomizer = DomainRandomizer(seed=seed)

    manifest: dict = {
        "seed": seed,
        "demos_per_task": demos_per_task,
        "image_size": [image_height, image_width],
        "tasks": {},
    }

    total_demos = 0
    total_successes = 0
    t0 = time.time()

    for task_name, task_def in TASKS.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name} — '{task_def.language}'")
        print(f"{'='*60}")

        task_successes = 0
        task_files: list[str] = []

        for demo_id in range(demos_per_task):
            # Reset randomizer seed for reproducibility
            randomizer.reset_seed(seed + total_demos)

            success, filepath = collector.collect_and_save(
                task_def, demo_id, randomizer=randomizer
            )

            task_successes += int(success)
            total_demos += 1
            total_successes += int(success)
            task_files.append(str(filepath.relative_to(DATA_DIR)))

            status = "OK" if success else "FAIL"
            print(
                f"  Demo {demo_id + 1:3d}/{demos_per_task} [{status}] "
                f"({task_successes}/{demo_id + 1} success) "
                f"-> {filepath.name}"
            )

        manifest["tasks"][task_name] = {
            "language": task_def.language,
            "num_demos": demos_per_task,
            "successes": task_successes,
            "success_rate": task_successes / demos_per_task,
            "files": task_files,
        }

    elapsed = time.time() - t0
    manifest["total_demos"] = total_demos
    manifest["total_successes"] = total_successes
    manifest["total_success_rate"] = (
        total_successes / total_demos if total_demos > 0 else 0.0
    )
    manifest["collection_time_s"] = elapsed

    # Save manifest
    manifest_path = DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset collection complete!")
    print(f"Total demos: {total_demos}")
    print(f"Total successes: {total_successes} ({manifest['total_success_rate']:.1%})")
    print(f"Time: {elapsed:.1f}s")
    print(f"Manifest: {manifest_path}")
    print(f"{'='*60}")

    return manifest


def main() -> None:
    """Entry point for dataset collection."""
    parser = argparse.ArgumentParser(description="Collect VLA demonstration dataset")
    parser.add_argument(
        "--demos-per-task",
        type=int,
        default=50,
        help="Number of demos per task (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=160,
        help="Camera width for demos (default: 160)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=120,
        help="Camera height for demos (default: 120)",
    )
    args = parser.parse_args()

    collect_dataset(
        demos_per_task=args.demos_per_task,
        seed=args.seed,
        image_width=args.image_width,
        image_height=args.image_height,
    )


if __name__ == "__main__":
    main()
