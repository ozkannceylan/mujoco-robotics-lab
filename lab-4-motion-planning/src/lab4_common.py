"""Lab 4 — Common constants and canonical real-stack loading helpers."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

_LAB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _LAB_DIR.parent
MODELS_DIR = _LAB_DIR / "models"
MEDIA_DIR = _LAB_DIR / "media"
URDF_PATH = PROJECT_ROOT / "lab-3-dynamics-force-control" / "models" / "ur5e.urdf"

_LAB3_SRC_DIR = PROJECT_ROOT / "lab-3-dynamics-force-control" / "src"
if str(_LAB3_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB3_SRC_DIR))

from lab3_common import (  # noqa: E402
    DT,
    NUM_JOINTS,
    Q_HOME,
    TORQUE_LIMITS,
    VELOCITY_LIMITS,
    build_mujoco_scene_spec,
    get_mj_ee_site_id,
    load_pinocchio_model as load_lab3_pinocchio_model,
    apply_arm_torques as apply_lab3_arm_torques,
)


ACC_LIMITS = np.array([8.0, 8.0, 8.0, 16.0, 16.0, 16.0])
VEL_LIMITS = VELOCITY_LIMITS.copy()

JOINT_LOWER = np.full(NUM_JOINTS, -2 * math.pi)
JOINT_UPPER = np.full(NUM_JOINTS, 2 * math.pi)


@dataclass(frozen=True)
class ObstacleSpec:
    """Axis-aligned obstacle box specification."""

    name: str
    position: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.85, 0.22, 0.22, 1.0)


OBSTACLES = (
    ObstacleSpec("obs1", (0.40, -0.15, 0.415), (0.05, 0.05, 0.10), (0.85, 0.22, 0.22, 1.0)),   # Red
    ObstacleSpec("obs2", (0.50, 0.15, 0.415), (0.05, 0.05, 0.10), (0.85, 0.45, 0.10, 1.0)),  # Orange
    ObstacleSpec("obs3", (0.60, -0.15, 0.415), (0.05, 0.05, 0.10), (0.85, 0.22, 0.22, 1.0)),    # Red
    ObstacleSpec("obs4", (0.70, 0.15, 0.415), (0.05, 0.05, 0.10), (0.85, 0.45, 0.10, 1.0)),   # Orange
)
CAPSTONE_OBSTACLES = OBSTACLES + (
    ObstacleSpec("cap_block", (0.60, -0.18, 0.415), (0.05, 0.06, 0.10), (0.90, 0.45, 0.10, 1.0)), # Orange
)
TABLE_SPEC = ObstacleSpec(
    "table",
    (0.45, 0.0, 0.30),
    (0.30, 0.40, 0.015),
    (0.55, 0.38, 0.20, 1.0),
)


def _obstacle_key(obstacle_specs: tuple[ObstacleSpec, ...]) -> tuple[tuple, ...]:
    """Convert obstacle specs to a hashable cache key."""
    return tuple(
        (
            spec.name,
            tuple(spec.position),
            tuple(spec.half_extents),
            tuple(spec.rgba),
        )
        for spec in obstacle_specs
    )


def _add_obstacles(spec, obstacle_specs: tuple[ObstacleSpec, ...]) -> None:
    """Add the obstacle boxes to the shared UR5e + Robotiq scene."""
    for obs in obstacle_specs:
        body = spec.worldbody.add_body()
        body.name = obs.name
        body.pos = list(obs.position)

        geom = body.add_geom()
        geom.name = f"{obs.name}_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = list(obs.half_extents)
        geom.rgba = list(obs.rgba)


@lru_cache(maxsize=8)
def _compile_scene_model(
    obstacle_key: tuple[tuple, ...],
    include_table: bool,
):
    """Compile and cache the canonical Lab 4 scene."""
    obstacle_specs = tuple(
        ObstacleSpec(
            name=entry[0],
            position=tuple(entry[1]),
            half_extents=tuple(entry[2]),
            rgba=tuple(entry[3]),
        )
        for entry in obstacle_key
    )
    # If include_table is True, the base scene already has a "table" body.
    # Filter out any obstacle spec named "table" to avoid "repeated name" errors.
    if include_table:
        obstacle_specs = tuple(obs for obs in obstacle_specs if obs.name != "table")
    spec = build_mujoco_scene_spec(with_table=include_table)
    _add_obstacles(spec, obstacle_specs)
    return spec.compile()


def load_mujoco_model(
    scene_path: Path | None = None,
    obstacle_specs: tuple[ObstacleSpec, ...] | list[ObstacleSpec] | None = None,
    include_table: bool = True,
):
    """Load the canonical MuJoCo scene used by Lab 4.

    `scene_path` is kept for backward compatibility. The canonical scene is built
    programmatically from the Menagerie UR5e + Robotiq stack plus the configured
    obstacles.
    """
    del scene_path
    specs = tuple(OBSTACLES if obstacle_specs is None else obstacle_specs)
    mj_model = _compile_scene_model(_obstacle_key(specs), include_table)
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def load_pinocchio_model(urdf_path: Path | None = None):
    """Load the canonical Pinocchio model used for FK and gravity terms."""
    return load_lab3_pinocchio_model(urdf_path or URDF_PATH)


def get_ee_pos(pin_model, pin_data, ee_fid: int, q: np.ndarray) -> np.ndarray:
    """Compute the canonical end-effector position for configuration q."""
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    return pin_data.oMf[ee_fid].translation.copy()


def get_mj_ee_pos(mj_model, mj_data) -> np.ndarray:
    """Read the MuJoCo pinch-site world position."""
    site_id = get_mj_ee_site_id(mj_model)
    return mj_data.site_xpos[site_id].copy()


def apply_arm_torques(mj_model, mj_data, tau: np.ndarray) -> None:
    """Apply desired arm torques through the Menagerie actuator model."""
    apply_lab3_arm_torques(mj_model, mj_data, tau)


def clip_torques(tau: np.ndarray) -> np.ndarray:
    """Clip arm torques to the UR5e torque limits."""
    return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
