"""Lab 9 — shared configuration for the VLA integration lab.

The central hub every other Lab 9 module imports: paths, the observation and
action contract's constants, the instruction vocabulary, and the Lab 8 imports
that make this lab's expert possible.

Cross-lab import rule
---------------------
Lab 8's ``src/`` is **appended** to ``sys.path``, never inserted at position 0.
Labs share module names (``standing_controller``, ``record_demo``,
``capstone_scene``), and putting a foreign lab ahead of this one silently
shadows local modules — Lab 8 lost an afternoon to exactly that.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

LAB_DIR: Path = Path(__file__).resolve().parent.parent
PROJECT_ROOT: Path = LAB_DIR.parent
SRC_DIR: Path = LAB_DIR / "src"
DATA_DIR: Path = LAB_DIR / "data"
MEDIA_DIR: Path = LAB_DIR / "media"
CHECKPOINT_DIR: Path = LAB_DIR / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_LAB8_SRC: Path = PROJECT_ROOT / "lab-8-loco-manipulation" / "src"
if str(_LAB8_SRC) not in sys.path:
    sys.path.append(str(_LAB8_SRC))

from lab8_common import (  # noqa: E402
    DT,
    NQ,
    NU,
    NV,
    load_g1_pinocchio,
    mj_state_to_pin,
    pin_point_to_world,
    robot_com,
)

__all__ = [
    "LAB_DIR",
    "PROJECT_ROOT",
    "SRC_DIR",
    "DATA_DIR",
    "MEDIA_DIR",
    "CHECKPOINT_DIR",
    "DT",
    "NQ",
    "NV",
    "NU",
    "IMAGE_SIZE",
    "CAMERAS",
    "STATE_DIM",
    "POLICY_HZ",
    "POLICY_PERIOD",
    "CHUNK_SIZE",
    "TASKS",
    "TASK_NAMES",
    "OBJECTS",
    "OBJECT_NAMES",
    "INSTRUCTIONS",
    "HELD_OUT_INSTRUCTIONS",
    "instruction_label",
    "all_instructions",
    "seed_everything",
    "load_g1_pinocchio",
    "mj_state_to_pin",
    "pin_point_to_world",
    "robot_com",
]

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

# 128 px, not the ACT paper's 224. Offscreen rendering on this machine goes
# through software EGL (llvmpipe) and costs ~97 ms/frame *regardless of
# resolution* — the cost is per-geometry setup, not fill — so a smaller image
# is free on the data-collection side and 3.7x cheaper on the training side
# (ResNet18 fwd+bwd: 32 samples/s at 224 px, 117 at 128 px). See
# tasks/LESSONS.md § L-P0-a.
IMAGE_SIZE: int = 128

#: Egocentric cameras, in the order the policy receives them.
CAMERAS: tuple[str, ...] = ("head", "wrist")

#: Proprioception layout — see below and tasks/ARCHITECTURE.md § Observation.
STATE_DIM: int = 2 * NU + 4  # 29 q + 29 qd + pelvis z + roll + pitch + grasp

POLICY_HZ: float = 10.0
POLICY_PERIOD: float = 1.0 / POLICY_HZ
#: Control ticks between two policy inferences (1 kHz / 10 Hz).
POLICY_DECIMATION: int = int(round(POLICY_PERIOD / DT))

#: Actions predicted per inference. 20 chunks at 10 Hz is 2 s of lookahead,
#: which spans a full reach phase.
CHUNK_SIZE: int = 20

# ---------------------------------------------------------------------------
# Tasks, objects, instructions
# ---------------------------------------------------------------------------

#: The sub-tasks the expert episode is sliced into. The keys are the phase
#: labels written into every demonstration.
#:
#: Two tasks, not the brief's three-to-five. Both of the ones that were cut were
#: cut because the *expert* cannot perform them reliably, and the numbers are in
#: tasks/LESSONS.md:
#:   * `carry` (walk holding the object) — 1/12 with Lab 8's two-handed tuck,
#:     1/6 without it, against 12/12 for the same controller on Lab 8's single
#:     tuned configuration (L-M0-c).
#:   * `place` (set the object on a marker) — 5/10 at its best, limited by an
#:     uncontrolled object orientation at release and by the standing-stability
#:     budget (L-M0-d, L-M0-e).
#: Walk + pick measures 12/12. A demonstration set whose expert falls in half its
#: episodes teaches a policy to fall.
#:
#: Note the walk instruction names the **object**, not the table. How far to
#: walk depends on which object was asked for — the near one is a one-step
#: approach, the far one two or three — so an instruction that did not name it
#: would make the task undecidable from the observation.
TASKS: dict[str, str] = {
    "walk": "walk to the {object}",
    "pick": "pick up the {object}",
}
TASK_NAMES: tuple[str, ...] = tuple(TASKS)

#: Two objects, so the instruction has to *choose*. A single-object scene lets
#: a policy infer the task from the robot's own pose and ignore the language
#: entirely, and every success rate measured that way says nothing about
#: language (tasks/LESSONS.md § L-P0-c).
OBJECTS: dict[str, str] = {"cup": "red cup", "box": "blue box"}
OBJECT_NAMES: tuple[str, ...] = tuple(OBJECTS)

#: Paraphrases per (task, object). The first entry is the canonical form used
#: for demonstrations; the rest exercise paraphrase robustness at evaluation.
INSTRUCTIONS: dict[tuple[str, str], tuple[str, ...]] = {
    ("walk", "cup"): (
        "walk to the red cup",
        "go to the red cup",
        "approach the red cup",
    ),
    ("walk", "box"): (
        "walk to the blue box",
        "go to the blue box",
        "approach the blue box",
    ),
    ("pick", "cup"): (
        "pick up the red cup",
        "grab the red cup",
        "lift the red cup",
    ),
    ("pick", "box"): (
        "pick up the blue box",
        "grab the blue box",
        "lift the blue box",
    ),
}

#: Paraphrases withheld from training so M4 measures generalisation rather
#: than memorisation. Index 0 is always used for training.
HELD_OUT_INSTRUCTIONS: int = 2  # index >= 2 is held out


def instruction_label(task: str, obj: str, variant: int = 0) -> str:
    """The instruction string for a (task, object) pair.

    Args:
        task: One of :data:`TASK_NAMES`.
        obj: One of :data:`OBJECT_NAMES`.
        variant: Paraphrase index; 0 is the canonical training form.

    Returns:
        The instruction sentence.
    """
    variants = INSTRUCTIONS[(task, obj)]
    return variants[variant % len(variants)]


def all_instructions(train_only: bool = False) -> list[str]:
    """Every distinct instruction string in the vocabulary.

    Args:
        train_only: Exclude the held-out paraphrases.

    Returns:
        Sorted, de-duplicated instruction strings.
    """
    out: set[str] = set()
    for variants in INSTRUCTIONS.values():
        chosen = variants[:HELD_OUT_INSTRUCTIONS] if train_only else variants
        out.update(chosen)
    return sorted(out)


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (if present) torch; return a fresh generator.

    Args:
        seed: The seed.

    Returns:
        A ``numpy.random.Generator`` for the caller's own sampling — module
        globals are seeded for the libraries that need it, but scene
        randomisation should draw from the returned generator so it is
        reproducible independently of library call order.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:  # torch is only present once the training extras are installed
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)
