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
    # Staggered boxes on the table creating a slalom corridor.
    # Each box: 10 cm deep, 10 cm wide, 20 cm tall (z = 0.315 to 0.515).
    # Boxes alternate left/right so the arm must weave through gaps.
    ObstacleSpec("box_1", (0.40, -0.15, 0.415), (0.05, 0.05, 0.10), (0.85, 0.22, 0.22, 1.0)),
    ObstacleSpec("box_2", (0.50,  0.15, 0.415), (0.05, 0.05, 0.10), (0.85, 0.45, 0.10, 1.0)),
    ObstacleSpec("box_3", (0.60, -0.15, 0.415), (0.05, 0.05, 0.10), (0.85, 0.22, 0.22, 1.0)),
    ObstacleSpec("box_4", (0.70,  0.15, 0.415), (0.05, 0.05, 0.10), (0.85, 0.45, 0.10, 1.0)),
)

# Forward-only slalom waypoints with gap-midpoint via-points at z = 0.56.
# Gap midpoints (Y~0) between consecutive boxes split each crossing into
# two short segments, preventing RRT* from solving long narrow corridors.
# Verified reachable with collision-free IK and min clearance > 0.03 m.
SLALOM_WAYPOINTS = (
    np.array([0.32, -0.30, 0.56]),   # 0  Start
    np.array([0.38, -0.28, 0.56]),   # 1  Left of Box 1
    np.array([0.45,  0.00, 0.56]),   # 2  Gap 1-2
    np.array([0.50,  0.22, 0.56]),   # 3  Right of Box 2
    np.array([0.55,  0.00, 0.56]),   # 4  Gap 2-3
    np.array([0.60, -0.22, 0.56]),   # 5  Left of Box 3
    np.array([0.65,  0.00, 0.56]),   # 6  Gap 3-4
    np.array([0.70,  0.22, 0.56]),   # 7  Right of Box 4
    np.array([0.76,  0.00, 0.56]),   # 8  Exit
)
SLALOM_LABELS = (
    "Start", "Left of Box 1", "Gap 1-2", "Right of Box 2",
    "Gap 2-3", "Left of Box 3", "Gap 3-4", "Right of Box 4", "Exit",
)
TABLE_SPEC = ObstacleSpec(
    "table",
    (0.45, 0.0, 0.30),
    (0.30, 0.40, 0.015),
    (0.55, 0.38, 0.20, 1.0),
)

# ── Slalom planning parameters ──────────────────────────────────────────
PLANNER_STEP_SIZE = 0.16
PLANNER_GOAL_BIAS = 0.20
PLANNER_REWIRE_RADIUS = 0.90
PLANNER_GOAL_TOLERANCE = 0.10
PLANNER_MAX_ITER = 2000
PLANNER_SAMPLING_MARGIN = np.full(NUM_JOINTS, np.pi * 0.75)
PLANNER_SEED_BANK = (42, 17, 7, 100, 200, 333, 500)
SHORTCUT_ITER = 220
IK_ERROR_TOL = 2e-3
MIN_CLEARANCE_M = 0.03
VEL_SCALE = 0.18
ACC_SCALE = 0.14


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


def build_ik_seed_bank() -> list[np.ndarray]:
    """Build a fixed 23-seed bank for position-only IK.

    Q_HOME plus shoulder and elbow variants across 11 base rotation
    offsets.  Gives robust coverage of the UR5e configuration space
    for reachable table-top positions.
    """
    seeds = [Q_HOME.copy()]
    for base_delta in np.linspace(-3.0, 2.0, 11):
        shoulder_seed = Q_HOME.copy()
        shoulder_seed[0] += base_delta
        shoulder_seed[1] += 0.6
        shoulder_seed[2] -= 0.4
        seeds.append(shoulder_seed)

        elbow_seed = Q_HOME.copy()
        elbow_seed[0] += base_delta
        elbow_seed[1] += 0.2
        elbow_seed[2] -= 0.8
        seeds.append(elbow_seed)
    return seeds


def densify_path(
    path: list[np.ndarray],
    max_step: float = 0.05,
) -> list[np.ndarray]:
    """Insert intermediate configs so consecutive points are <= max_step apart.

    Prevents the TOPP-RA cubic spline from overshooting far beyond the
    collision-free straight-line edges verified during shortcutting.
    """
    dense: list[np.ndarray] = [path[0].copy()]
    for i in range(len(path) - 1):
        diff = path[i + 1] - path[i]
        dist = float(np.linalg.norm(diff))
        if dist <= max_step:
            dense.append(path[i + 1].copy())
            continue
        n_steps = max(2, int(np.ceil(dist / max_step)))
        for j in range(1, n_steps + 1):
            dense.append(path[i] + (j / n_steps) * diff)
    return dense
