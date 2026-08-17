"""Lab 9 — M2 gate: the model and its conditioning, before any training.

Prints the table the milestone is read from and asserts the things that would
otherwise only show up as a mediocre success rate hours later:

* parameter counts, and how many of them actually train;
* the spatial token count is **derived** from the input size rather than the
  224 px constant upstream hardcodes;
* two different instructions on an identical observation produce different
  actions — the necessary condition for M4's instruction-swap test;
* the instruction bank separates instructions that mean different things by
  more than it separates paraphrases of the same thing, which is the property
  the language conditioning actually needs;
* a checkpoint round-trips its own predictions.

Run:
    python3 lab-9-vla-integration/src/m2_model_check.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from act_policy import ACTPolicy, load_checkpoint, save_checkpoint
from dataset import NormStats
from lab9_common import (
    CHUNK_SIZE,
    IMAGE_SIZE,
    MEDIA_DIR,
    OBJECT_NAMES,
    STATE_DIM,
    TASK_NAMES,
    instruction_label,
)
from text_encoder import TEXT_EMBED_DIM, build_instruction_bank


def conditioning_separation(bank) -> dict:
    """Compare within-meaning and across-meaning instruction similarity.

    Paraphrases of the same command *should* be close — that is what makes the
    policy paraphrase-robust. Commands that mean different things must be
    further apart than that, or no amount of training can separate the
    behaviours they are supposed to select.

    Args:
        bank: An `text_encoder.InstructionBank`.

    Returns:
        Mean cosine similarity within and across meanings, and the margin.
    """
    groups = {
        (task, obj): [instruction_label(task, obj, i) for i in range(3)]
        for task in TASK_NAMES for obj in OBJECT_NAMES
    }
    within, across = [], []
    keys = sorted(groups)
    for index, key in enumerate(keys):
        embeddings = bank.batch(groups[key])
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                within.append(float(embeddings[i] @ embeddings[j]))
        for other in keys[index + 1:]:
            other_embeddings = bank.batch(groups[other])
            across.append(float((embeddings @ other_embeddings.T).mean()))
    return {
        "within_meaning": float(np.mean(within)),
        "across_meaning": float(np.mean(across)),
        "margin": float(np.mean(within) - np.mean(across)),
        "groups": len(keys),
    }


def main() -> None:
    torch.manual_seed(0)
    print("Lab 9 — M2 gate\n" + "=" * 66)

    model = ACTPolicy(pretrained_backbone=True)
    counts = model.parameter_counts()
    print(f"{'block':<16}{'params':>12}{'trainable':>12}")
    print("-" * 66)
    for name, (total, trainable) in counts.items():
        print(f"{name:<16}{total:>12,}{trainable:>12,}")
    print("-" * 66)
    print(f"cameras                {list(model.cameras)}")
    print(f"tokens per camera      {model.tokens_per_camera} "
          f"(derived from {IMAGE_SIZE} px; upstream hardcodes 49 for 224 px)")
    print(f"memory tokens          "
          f"{model.tokens_per_camera * len(model.cameras) + 2} "
          f"(vision + state + instruction)")
    print(f"chunk size             {CHUNK_SIZE} actions "
          f"({CHUNK_SIZE / 10:.1f} s at 10 Hz)")
    print(f"state dim              {STATE_DIM}")

    # -- conditioning sensitivity ---------------------------------------
    model.eval()
    images = {c: torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE) for c in model.cameras}
    state = torch.randn(1, STATE_DIM)
    bank = build_instruction_bank()
    cup = torch.from_numpy(bank.get(instruction_label("pick", "cup"))).unsqueeze(0)
    box = torch.from_numpy(bank.get(instruction_label("pick", "box"))).unsqueeze(0)
    delta = float((model.predict(images, state, cup)
                   - model.predict(images, state, box)).abs().max())
    repeat = float((model.predict(images, state, cup)
                    - model.predict(images, state, cup)).abs().max())

    separation = conditioning_separation(bank)
    print(f"\ninstruction bank       {len(bank)} entries, dim {TEXT_EMBED_DIM}")
    print(f"  cosine within meaning  {separation['within_meaning']:.3f}")
    print(f"  cosine across meanings {separation['across_meaning']:.3f}")
    print(f"  margin                 {separation['margin']:.3f}")

    # -- overfit a batch -------------------------------------------------
    batch_images = {c: torch.rand(8, 3, IMAGE_SIZE, IMAGE_SIZE) for c in model.cameras}
    batch_state = torch.randn(8, STATE_DIM)
    batch_instruction = torch.randn(8, TEXT_EMBED_DIM)
    target = torch.randn(8, CHUNK_SIZE, model.action_dim)
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-3
    )
    model.train()
    first = None
    for _ in range(60):
        loss = (model(batch_images, batch_state, batch_instruction)
                - target).abs().mean()
        first = first if first is not None else float(loss)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    ratio = float(loss) / first

    # -- checkpoint round trip ------------------------------------------
    stats = NormStats(
        state_mean=np.zeros(STATE_DIM, np.float32),
        state_scale=np.ones(STATE_DIM, np.float32),
        action_mean=np.zeros(model.action_dim, np.float32),
        action_scale=np.ones(model.action_dim, np.float32),
    )
    model.eval()
    before = model.predict(images, state, cup)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "policy.pt"
        save_checkpoint(path, model, bank, stats)
        restored, _, _, _ = load_checkpoint(path)
        after = restored.predict(images, state, cup)
    round_trip = float((before - after).abs().max())

    rows = [
        ("Parameter count reported", True,
         f"{counts['total'][0] / 1e6:.2f}M total, "
         f"{counts['total'][1] / 1e6:.2f}M trainable"),
        ("Token count derived from image size",
         model.tokens_per_camera == 16 and IMAGE_SIZE == 128,
         f"{model.tokens_per_camera} tokens/camera at {IMAGE_SIZE} px"),
        ("Instruction changes the action", delta > 1e-4, f"max delta {delta:.4f}"),
        ("Same instruction is deterministic", repeat < 1e-6, f"{repeat:.2e}"),
        ("Meanings separate further than paraphrases",
         separation["margin"] > 0.02, f"margin {separation['margin']:.3f}"),
        ("Overfits one batch", ratio < 0.35, f"loss ratio {ratio:.3f}"),
        ("Checkpoint round-trips predictions", round_trip < 1e-6,
         f"{round_trip:.2e}"),
    ]
    print("\n" + "-" * 66)
    print(f"{'criterion':<44}{'result':<8}measured")
    print("-" * 66)
    for name, passed, measured in rows:
        print(f"{name:<44}{'PASS' if passed else 'FAIL':<8}{measured}")
    print("-" * 66)

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    (MEDIA_DIR / "m2_model.json").write_text(json.dumps({
        "params": {k: list(v) for k, v in counts.items()},
        "tokens_per_camera": model.tokens_per_camera,
        "cameras": list(model.cameras),
        "instruction_separation": separation,
        "conditioning_delta": delta,
        "overfit_ratio": ratio,
        "checkpoint_round_trip": round_trip,
        "gate_passed": all(p for _, p, _ in rows),
    }, indent=2))
    print(f"\nGATE {'PASSED' if all(p for _, p, _ in rows) else 'FAILED'}")


if __name__ == "__main__":
    main()
