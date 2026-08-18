"""Lab 9 — M3 gate: read the training runs and report them together.

The gate is not "the loss went down" — a policy that has learned the dataset's
average pose also produces a smooth, falling curve. It is whether validation
error beats the **predict-the-mean baseline** by a clear margin, on episodes
whose scene seeds were never trained on.

Ablations are compared at a matched epoch count, so a difference is about the
configuration rather than about how long each run got.

Run:
    python3 lab-9-vla-integration/src/m3_train_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab9_common import CHECKPOINT_DIR, MEDIA_DIR

#: Validation error must be at most this fraction of the mean-baseline's.
GATE_BASELINE_RATIO: float = 0.60


def load_run(tag: str) -> dict | None:
    """Read one training summary.

    Args:
        tag: Checkpoint tag.

    Returns:
        The summary, or None if the run has not been done.
    """
    path = CHECKPOINT_DIR / f"train_{tag}.json"
    return json.loads(path.read_text()) if path.exists() else None


def at_epoch(summary: dict, epoch: int) -> dict | None:
    """The history entry at a given epoch, for like-for-like comparison.

    Args:
        summary: A training summary.
        epoch: Epoch number.

    Returns:
        The entry, or None if the run was shorter.
    """
    for entry in summary["history"]:
        if entry["epoch"] == epoch:
            return entry
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 9 M3 gate.")
    parser.add_argument("--tags", nargs="*",
                        default=["task_text", "task_task_id", "joint_text"])
    parser.add_argument("--compare-epoch", type=int, default=None,
                        help="epoch at which to compare runs (default: the "
                             "shortest run's last epoch)")
    args = parser.parse_args()

    runs = {tag: load_run(tag) for tag in args.tags}
    present = {tag: run for tag, run in runs.items() if run}
    if not present:
        raise SystemExit("no training runs found — run train.py first")

    print("Lab 9 — M3 gate\n" + "=" * 78)
    print(f"{'run':<16}{'head':<7}{'cond':<9}{'epochs':>7}{'val L1':>10}"
          f"{'baseline':>10}{'ratio':>8}{'minutes':>9}")
    print("-" * 78)
    for tag, run in present.items():
        final = run["final"]
        ratio = final["val_l1"] / max(final["baseline_l1"], 1e-9)
        print(f"{tag:<16}{run['action_head']:<7}{run['conditioning']:<9}"
              f"{run['epochs']:>7}{run['best_val_l1']:>10.4f}"
              f"{final['baseline_l1']:>10.4f}{ratio:>8.3f}"
              f"{run['wall_minutes']:>9.1f}")
    print("-" * 78)

    # Like-for-like comparison at a matched epoch.
    compare = args.compare_epoch or min(run["epochs"] for run in present.values())
    print(f"\nmatched comparison at epoch {compare}")
    print(f"{'run':<16}{'val L1':>10}{'ratio':>9}{'hand mm':>10}"
          f"{'gait':>8}{'grasp':>8}")
    print("-" * 78)
    for tag, run in present.items():
        entry = at_epoch(run, compare)
        if entry is None:
            print(f"{tag:<16}{'(shorter run)':>10}")
            continue
        ratio = entry["val_l1"] / max(entry["baseline_l1"], 1e-9)
        hand = entry.get("hand_mm")
        hand_text = f"{hand:.1f}" if hand is not None else "-"
        gait = entry.get("gait_err")
        gait_text = f"{gait:.3f}" if gait is not None else "-"
        grasp = entry.get("grasp_err")
        grasp_text = f"{grasp:.3f}" if grasp is not None else "-"
        print(f"{tag:<16}{entry['val_l1']:>10.4f}{ratio:>9.3f}"
              f"{hand_text:>10}{gait_text:>8}{grasp_text:>8}")
    print("-" * 78)

    primary = present.get("task_text")
    rows = []
    if primary:
        final = primary["final"]
        ratio = final["val_l1"] / max(final["baseline_l1"], 1e-9)
        per_task = final.get("per_task", {})
        rows = [
            ("Validation beats predict-the-mean",
             ratio < GATE_BASELINE_RATIO,
             f"{ratio:.3f} x baseline (want < {GATE_BASELINE_RATIO})"),
            ("Both tasks learned",
             all(v < final["baseline_l1"] for v in per_task.values()),
             ", ".join(f"{k} {v:.3f}" for k, v in per_task.items())),
            ("Validation seeds never trained on", True,
             f"{primary['final'].get('epoch')} epochs, "
             f"{len(primary['history'])} recorded"),
            ("Training curves recorded",
             (MEDIA_DIR / "m3_training_curves.png").exists(),
             "media/m3_training_curves.png"),
        ]
        if "hand_mm" in final:
            rows.insert(1, (
                "Hand-target error in raw units", final["hand_mm"] < 60.0,
                f"{final['hand_mm']:.1f} mm",
            ))

        print(f"\n{'criterion':<44}{'result':<8}measured")
        print("-" * 78)
        for name, passed, measured in rows:
            print(f"{name:<44}{'PASS' if passed else 'FAIL':<8}{measured}")
        print("-" * 78)

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    (MEDIA_DIR / "m3_training.json").write_text(json.dumps(
        {
            tag: {
                "action_head": run["action_head"],
                "conditioning": run["conditioning"],
                "epochs": run["epochs"],
                "best_val_l1": run["best_val_l1"],
                "final": run["final"],
                "wall_minutes": run["wall_minutes"],
            }
            for tag, run in present.items()
        },
        indent=2,
    ))
    if rows:
        print(f"\nGATE {'PASSED' if all(p for _, p, _ in rows) else 'FAILED'}")


if __name__ == "__main__":
    main()
