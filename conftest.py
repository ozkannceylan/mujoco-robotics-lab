"""Repo-wide pytest bootstrap — keeps same-named lab modules from colliding.

Labs are independent projects that happen to live in one repository, and they
reuse module names by convention: Labs 1 and 2 both have
``a4_inverse_kinematics`` and ``b1_trajectory_generation``; Labs 7 and 8 both
have ``standing_controller``. Each lab's tests put their own ``src/`` on
``sys.path``, which is fine in isolation — but in a single pytest process the
first lab collected wins ``sys.modules``, and every later lab silently imports
the wrong file::

    pytest lab-1-2link-arm/tests/ lab-2-Ur5e-robotics-lab/tests/
    ImportError: cannot import name 'ik_pseudoinverse' from
                 'a4_inverse_kinematics'  (…/lab-1-2link-arm/src/…)

which is why ``pytest lab-*/tests/`` (the command in CLAUDE.md) used to fail
while every lab passed on its own.

Before collecting each test file, this hook puts the owning lab's ``src/`` at
the front of ``sys.path`` and evicts any module of the same name that was
imported from a *different* lab. Per-lab runs are unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _owning_lab_src(path: Path) -> Path | None:
    """Return the ``src/`` of the lab directory containing ``path``."""
    for parent in path.resolve().parents:
        if parent.parent == _ROOT and parent.name.startswith("lab-"):
            src = parent / "src"
            return src if src.is_dir() else None
    return None


def _activate(src: Path) -> None:
    """Make ``src`` win imports, dropping same-named modules from other labs."""
    src_str = str(src)
    if src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)

    local_names = {p.stem for p in src.glob("*.py")}
    for name in list(sys.modules):
        if name not in local_names:
            continue
        module_file = getattr(sys.modules[name], "__file__", None)
        if module_file and not Path(module_file).resolve().is_relative_to(src):
            del sys.modules[name]


def pytest_collectstart(collector) -> None:  # noqa: D103 - pytest hook
    path = getattr(collector, "path", None)
    if path is None:
        return
    src = _owning_lab_src(Path(path))
    if src is not None:
        _activate(src)
