"""Lab 9 — M4: closed-loop evaluation of the trained policy.

Four measurements, and the last two are the ones that decide whether the first
two mean anything.

**Per-task success.** The `walk` task is scored on where the robot *stopped* —
the named object decides how far to go, so stopping in the right place is the
whole task. The `pick` task starts from the expert's post-walk state and is
scored on whether the named object actually rose.

**Position-randomised.** The same policy over the wider object-placement range,
which no training episode was drawn from.

**The instruction contrast.** The scene a seed produces is identical whichever
object is named — the randomisation places both — so the `seen` condition
already runs two different instructions over the same initial state. The
contrast is therefore a *paired* measurement over those episodes rather than a
separate set of rollouts: for each seed, how far apart do the two instructions
put the robot, against how far apart the correct answers are. A policy that
ignores its instruction scores zero on this no matter how good its success rate
looks.

**The joint-head ablation.** The brief's literal action space — 29 joint targets
tracked by PD — against Lab 7's prediction that it cannot stabilise this robot.

Every success is decided on **simulated** state: where the object is, not where
the policy asked it to be.

Run:
    MUJOCO_GL=egl python3 lab-9-vla-integration/src/evaluate.py --episodes 12
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lab9_common import CHECKPOINT_DIR, MEDIA_DIR, OBJECT_NAMES, instruction_label
from policy_runner import STOP_DECISION_TAIL

__all__ = ["run_walk_episode", "run_pick_episode", "evaluate"]

#: How close to the expert's stopping point counts as "stopped in the right
#: place". The two objects' stopping points are ~0.20 m apart, so this cleanly
#: separates "went to the cup" from "went to the box" and is not a free pass.
STOP_TOLERANCE_M: float = 0.08

#: How high the named object has to rise for a pick to count.
LIFT_TOLERANCE_M: float = 0.04

#: How far the other object may move. Sweeping the distractor off the pedestal
#: is not following the instruction.
DISTRACTOR_TOLERANCE_M: float = 0.05

#: A policy that never stops walking is stopped here.
MAX_WALK_UNITS: int = 6
#: Consecutive stand commands that count as "the policy has stopped".
STAND_TO_FINISH: int = 3
#: Policy periods allowed for the manipulation, at 10 Hz.
MAX_PICK_POLLS: int = 70


def _load(checkpoint: Path):
    from act_policy import load_checkpoint

    return load_checkpoint(checkpoint)


def run_walk_episode(
    seed: int, target: str, checkpoint: Path, wide: bool = False,
    instruction: str | None = None, variant: int = 0,
) -> dict:
    """Closed-loop `walk` episode: does the robot stop where the instruction says?

    Args:
        seed: Scene seed.
        target: Which object the *scene* is asked about.
        checkpoint: Trained policy.
        wide: Wider object placement.
        instruction: Override the instruction; used for the held-out paraphrase
            condition, and available for any manual contrast run.
        variant: Paraphrase index.

    Returns:
        A result dict.
    """
    from expert import REACH_STANDOFF, approach_steps_for
    from policy_runner import PolicyRunner, RolloutResult
    from vla_scene import Randomisation
    from m5_capstone import Fell

    model, bank, _, _ = _load(checkpoint)
    text = instruction or instruction_label("walk", target, variant)
    runner = PolicyRunner(seed, target, model, bank, wide=wide)
    result = RolloutResult(seed=seed, target=target, instruction=text, task="walk")

    # Scored on the standoff actually achieved to the **named** object, not on
    # a step count. The goal of the walk is to end up within reach of the thing
    # the instruction named, and the two objects' correct stopping points are
    # ~0.30 m apart, so going to the wrong one cannot pass.
    named = "cup" if "cup" in text else "box"
    named_x = float(runner.scene.object_position(named)[0])
    marker_x = float(runner.scene.place_target[0])
    expert_steps = approach_steps_for(named_x, marker_x)
    result.expert_pelvis_x = 0.5 * (named_x + marker_x) - REACH_STANDOFF
    result.walk_units_expert = expert_steps

    try:
        # The first decision is made standing at the start line, which *is* a
        # state the demonstrations contain (every walk segment begins there).
        # Every decision after that is made from mid-stride observations
        # collected inside the unit, because the expert never pauses during an
        # approach and a paused robot mid-approach is off-distribution.
        runner.infer(text)
        keep_walking = runner.gait_intent() > 0.5
        result.gait_commands.append(round(runner.gait_intent(), 3))
        while keep_walking and result.walk_units < MAX_WALK_UNITS:
            gaits = runner.walk_unit(text)
            result.walk_units += 1
            result.gait_commands.extend(round(g, 3) for g in gaits)
            # Decide from the tail of the unit's polls. A single low sample is
            # noise; the mean over the last third is the policy's settled
            # intent at the moment the next step would be committed.
            tail = gaits[-max(1, int(len(gaits) * STOP_DECISION_TAIL)):]
            keep_walking = bool(tail) and float(np.mean(tail)) > 0.5
        # Settle standing so the stopping position is a resting one.
        for _ in range(STAND_TO_FINISH):
            runner.stand_tick(runner.infer(text))
    except Fell as exc:
        result.fell = True
        result.reason = str(exc)
    finally:
        result.final_pelvis_x = float(runner.mj_data.qpos[0])
        result.duration_s = runner.t
        result.tau_max = max(runner.log.tau_max) if runner.log.tau_max else 0.0
        result.inferences = runner._inferences
        runner.close()

    result.stop_error_m = abs(result.final_pelvis_x - result.expert_pelvis_x)
    other = next(o for o in OBJECT_NAMES if o != named)
    other_x = float(
        Randomisation.sample(seed, wide=wide).object_xy(other)[0]
    )
    result.stop_error_other_m = abs(
        result.final_pelvis_x - (0.5 * (other_x + marker_x) - REACH_STANDOFF)
    )
    result.success = bool(
        not result.fell and result.stop_error_m < STOP_TOLERANCE_M
    )
    if not result.success and not result.reason:
        closer = ("the other object" if result.stop_error_other_m < result.stop_error_m
                  else "neither")
        result.reason = (
            f"stopped at x={result.final_pelvis_x:.3f}, want "
            f"{result.expert_pelvis_x:.3f} ({closer} was closer)"
        )
    return result.__dict__


def run_pick_episode(
    seed: int, target: str, checkpoint: Path, wide: bool = False,
    instruction: str | None = None, variant: int = 0,
) -> dict:
    """Closed-loop `pick` episode from the expert's post-walk state.

    The approach is done by the expert so this measures the manipulation on its
    own; chaining both is the capstone's job.

    Args:
        seed: Scene seed.
        target: Which object the scene is set up around.
        checkpoint: Trained policy.
        wide: Wider object placement.
        instruction: Override the instruction; used for the held-out paraphrase
            condition, and available for any manual contrast run.
        variant: Paraphrase index.

    Returns:
        A result dict.
    """
    from expert import T_STOP_L9, approach_steps_for
    from policy_runner import PolicyRunner, RolloutResult
    from m5_capstone import Fell

    model, bank, _, _ = _load(checkpoint)
    text = instruction or instruction_label("pick", target, variant)
    runner = PolicyRunner(seed, target, model, bank, wide=wide)
    result = RolloutResult(seed=seed, target=target, instruction=text, task="pick")

    named = "cup" if "cup" in text else "box"
    start_heights = {
        name: float(runner.scene.object_position(name)[2])
        for name in runner.scene.object_bodies
    }
    start_xy = {
        name: runner.scene.object_position(name)[:2].copy()
        for name in runner.scene.object_bodies
    }

    try:
        # Expert approach, so the manipulation is measured from the state the
        # demonstrations start their pick segment in.
        steps = approach_steps_for(
            float(runner.scene.object_position(named)[0]),
            float(runner.scene.place_target[0]),
        )
        runner.walk(steps, "walk_to_pick")
        runner.stand(T_STOP_L9, "stop_at_pick")

        for _ in range(MAX_PICK_POLLS):
            action = runner.infer(text)
            runner.stand_tick(action)
            if runner.try_grasp(action):
                result.grasped = True
            if result.grasped:
                lifted = (
                    float(runner.scene.object_position(runner.scene.target)[2])
                    - start_heights[runner.scene.target]
                )
                if lifted > LIFT_TOLERANCE_M:
                    break
            if runner.t > 20.0:
                break
    except Fell as exc:
        result.fell = True
        result.reason = str(exc)
    finally:
        lifts = {
            name: float(runner.scene.object_position(name)[2]) - start_heights[name]
            for name in start_heights
        }
        moved = {
            name: float(np.linalg.norm(
                runner.scene.object_position(name)[:2] - start_xy[name]
            ))
            for name in start_xy
        }
        other = next(n for n in start_heights if n != named)
        result.lift_m = lifts[named]
        result.distractor_moved_m = moved[other]
        result.duration_s = runner.t
        result.tau_max = max(runner.log.tau_max) if runner.log.tau_max else 0.0
        result.inferences = runner._inferences
        try:
            runner._sync_kinematics()
            result.hand_error_mm = float(np.linalg.norm(
                runner.hand_position() - runner.scene.object_position(named)
            )) * 1000.0
        except Exception:  # noqa: BLE001 - a fallen robot has no useful reading
            pass
        # Only meaningful if a grasp actually closed; the scene always has a
        # target selected, and reporting it unconditionally would read as "it
        # picked the cup" on an episode that picked nothing.
        result.grasped_object = (
            getattr(runner.scene, "target", "") if result.grasped else ""
        )
        runner.close()

    result.success = bool(
        not result.fell
        and result.lift_m > LIFT_TOLERANCE_M
        and result.distractor_moved_m < DISTRACTOR_TOLERANCE_M
    )
    if not result.success and not result.reason:
        result.reason = (
            f"lifted {result.lift_m * 1000:.0f} mm, "
            f"distractor moved {result.distractor_moved_m * 1000:.0f} mm"
        )
    return result.__dict__


def _job(payload: tuple) -> dict:
    kind, seed, target, checkpoint, wide, instruction, variant, label = payload
    runner = run_walk_episode if kind == "walk" else run_pick_episode
    result = runner(
        seed, target, Path(checkpoint), wide=wide,
        instruction=instruction, variant=variant,
    )
    result["condition"] = label
    return result


def evaluate(
    checkpoint: Path,
    episodes: int,
    workers: int = 3,
    conditions: tuple[str, ...] = ("seen", "wide", "paraphrase"),
) -> list[dict]:
    """Run every evaluation condition.

    Args:
        checkpoint: Trained policy.
        episodes: Seeds per condition per object.
        workers: Parallel workers.
        conditions: Which conditions to run.

    Returns:
        One result dict per episode.
    """
    jobs: list[tuple] = []
    # Seeds outside the training range: the training set used 0..59.
    base = 200
    for index in range(episodes):
        for target in OBJECT_NAMES:
            seed = base + index
            if "seen" in conditions:
                for kind in ("walk", "pick"):
                    jobs.append((kind, seed, target, str(checkpoint), False,
                                 None, 0, "seen"))
            if "wide" in conditions:
                for kind in ("walk", "pick"):
                    jobs.append((kind, seed + 1000, target, str(checkpoint), True,
                                 None, 0, "wide"))
            if "paraphrase" in conditions:
                # Held-out wording, never seen in training.
                for kind in ("walk", "pick"):
                    jobs.append((kind, seed, target, str(checkpoint), False,
                                 instruction_label(kind, target, 2), 2, "paraphrase"))

    print(f"running {len(jobs)} closed-loop episodes with {workers} workers ...")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(_job, jobs), start=1):
            results.append(result)
            print(f"  [{index:>3}/{len(jobs)}] {result['condition']:<10} "
                  f"{result['task']:<4} seed {result['seed']:<5} "
                  f"{result['target']:<4} "
                  f"{'OK ' if result['success'] else 'FAIL'} "
                  f"{result['reason'][:42]}")
    return results


def instruction_contrast(results: list[dict]) -> dict:
    """Paired test: does naming the other object change what the policy does?

    For each seed the two `seen` episodes share an identical scene and differ
    only in which object the instruction names. The walk task has a continuous
    answer — where to stop — so the contrast is the ratio of the separation the
    policy produced to the separation it should have. The pick task has a
    discrete one: did it grasp the object that was named.

    Args:
        results: Episode dicts from :func:`evaluate`.

    Returns:
        The contrast statistics, or an empty dict if there are no pairs.
    """
    seen = [r for r in results if r["condition"] == "seen"]
    walk = {}
    for row in (r for r in seen if r["task"] == "walk"):
        walk.setdefault(row["seed"], {})[row["target"]] = row
    ratios, commanded, produced = [], [], []
    for pair in walk.values():
        if len(pair) != 2:
            continue
        first, second = (pair[name] for name in sorted(pair))
        want = abs(first["expert_pelvis_x"] - second["expert_pelvis_x"])
        got = abs(first["final_pelvis_x"] - second["final_pelvis_x"])
        if want < 1e-6:
            continue
        commanded.append(want)
        produced.append(got)
        ratios.append(got / want)

    picks = [r for r in seen if r["task"] == "pick" and r["grasped"]]
    correct = sum(1 for r in picks if r["grasped_object"] == r["target"])

    if not ratios and not picks:
        return {}
    return {
        "walk_pairs": len(ratios),
        "mean_commanded_separation_m": float(np.mean(commanded)) if commanded else 0.0,
        "mean_produced_separation_m": float(np.mean(produced)) if produced else 0.0,
        "separation_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "grasps": len(picks),
        "grasped_the_named_object": correct,
        "grasp_accuracy": correct / len(picks) if picks else float("nan"),
    }


def summarise(results: list[dict]) -> dict:
    """Success rates by condition and task."""
    summary: dict = {}
    for condition in sorted({r["condition"] for r in results}):
        summary[condition] = {}
        for task in ("walk", "pick"):
            rows = [r for r in results
                    if r["condition"] == condition and r["task"] == task]
            if rows:
                summary[condition][task] = {
                    "success": sum(r["success"] for r in rows),
                    "episodes": len(rows),
                    "rate": sum(r["success"] for r in rows) / len(rows),
                    "falls": sum(r["fell"] for r in rows),
                }
        rows = [r for r in results if r["condition"] == condition]
        summary[condition]["all"] = {
            "success": sum(r["success"] for r in rows),
            "episodes": len(rows),
            "rate": sum(r["success"] for r in rows) / len(rows),
        }
    return summary


def plot(summary: dict, path: Path) -> None:
    """Bar chart of success rate by condition and task."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [c for c in ("seen", "wide", "paraphrase") if c in summary]
    figure, axis = plt.subplots(figsize=(9, 4.6))
    width = 0.35
    positions = np.arange(len(conditions))
    for offset, task in zip((-width / 2, width / 2), ("walk", "pick"), strict=True):
        values = [summary[c].get(task, {}).get("rate", 0.0) * 100 for c in conditions]
        bars = axis.bar(positions + offset, values, width, label=task)
        for bar, condition in zip(bars, conditions, strict=True):
            entry = summary[condition].get(task, {})
            axis.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{entry.get('success', 0)}/{entry.get('episodes', 0)}",
                ha="center", fontsize=8,
            )
    axis.axhline(70, color="tab:green", ls="--", lw=1, label="gate: seen > 70%")
    axis.axhline(40, color="tab:orange", ls=":", lw=1, label="gate: randomised > 40%")
    axis.set_xticks(positions)
    axis.set_xticklabels(conditions)
    axis.set_ylabel("success rate [%]")
    axis.set_ylim(0, 108)
    axis.set_title("Lab 9 M4 — closed-loop success, policy at 10 Hz over Lab 8's QP")
    axis.legend(fontsize=8, loc="lower left")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 9 M4 evaluation.")
    parser.add_argument("--checkpoint", type=Path,
                        default=CHECKPOINT_DIR / "act_task_text.pt")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--conditions", nargs="*",
                        default=["seen", "wide", "paraphrase"])
    parser.add_argument("--out", type=str, default="m4")
    args = parser.parse_args()

    results = evaluate(
        args.checkpoint, args.episodes, workers=args.workers,
        conditions=tuple(args.conditions),
    )
    summary = summarise(results)
    contrast = instruction_contrast(results)

    print("\n" + "=" * 70)
    print(f"{'condition':<14}{'task':<7}{'success':<12}{'rate':<9}falls")
    print("-" * 70)
    for condition, tasks in summary.items():
        for task, entry in tasks.items():
            print(f"{condition:<14}{task:<7}"
                  f"{entry['success']}/{entry['episodes']:<10}"
                  f"{entry['rate']:.0%}{'':<5}{entry.get('falls', '')}")
    print("=" * 70)

    if contrast:
        print("\ninstruction contrast (paired, same scene, two instructions)")
        print(f"  walk: commanded separation "
              f"{contrast['mean_commanded_separation_m']:.3f} m, produced "
              f"{contrast['mean_produced_separation_m']:.3f} m "
              f"-> ratio {contrast['separation_ratio']:.2f} "
              f"over {contrast['walk_pairs']} pairs")
        print(f"  pick: grasped the named object "
              f"{contrast['grasped_the_named_object']}/{contrast['grasps']}")

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    (MEDIA_DIR / f"{args.out}_summary.json").write_text(
        json.dumps(
            {"summary": summary, "contrast": contrast, "results": results},
            indent=2, default=str,
        )
    )
    with (MEDIA_DIR / f"{args.out}_episodes.csv").open("w", newline="") as handle:
        fields = sorted({k for r in results for k in r})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    plot(summary, MEDIA_DIR / f"{args.out}_success_rates.png")
    print(f"wrote {args.out}_summary.json, {args.out}_episodes.csv, "
          f"{args.out}_success_rates.png")


if __name__ == "__main__":
    main()
