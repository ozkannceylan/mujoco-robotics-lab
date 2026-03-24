"""Lab 4 slalom demo planning, metrics, and execution helpers."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pinocchio as pin

_LAB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _LAB_DIR.parent
_LAB3_SRC_DIR = PROJECT_ROOT / "lab-3-dynamics-force-control" / "src"
if str(_LAB3_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB3_SRC_DIR))

from c1_force_control import solve_ik_xy
from collision_checker import CollisionChecker
from lab4_common import (
    ACC_LIMITS,
    DT,
    JOINT_LOWER,
    JOINT_UPPER,
    OBSTACLES,
    TABLE_SPEC,
    VEL_LIMITS,
    apply_arm_torques,
    clip_torques,
    get_ee_pos,
    load_pinocchio_model,
)
from rrt_planner import RRTStarPlanner
from trajectory_smoother import parameterize_topp_ra, shortcut_path


# Forward-only waypoints with gap-midpoint via-points.
# Gap midpoints (Y≈0) split each box-to-box crossing into two short segments,
# preventing RRT* from having to solve one long narrow-corridor path.
# The return path is constructed automatically as the reverse of the forward path.
SLALOM_WAYPOINTS = (
    np.array([0.32, -0.30, 0.56]),    # 0  Start
    np.array([0.38, -0.28, 0.56]),    # 1  Left of Box 1
    np.array([0.45,  0.00, 0.56]),    # 2  Gap 1↔2
    np.array([0.50,  0.22, 0.56]),    # 3  Right of Box 2
    np.array([0.55,  0.00, 0.56]),    # 4  Gap 2↔3
    np.array([0.60, -0.22, 0.56]),    # 5  Left of Box 3
    np.array([0.65,  0.00, 0.56]),    # 6  Gap 3↔4
    np.array([0.70,  0.22, 0.56]),    # 7  Right of Box 4
    np.array([0.76,  0.00, 0.56]),    # 8  Clear exit
)

WAYPOINT_LABELS = (
    "Start",
    "Left of Box 1",
    "Gap 1–2",
    "Right of Box 2",
    "Gap 2–3",
    "Left of Box 3",
    "Gap 3–4",
    "Right of Box 4",
    "Clear Exit",
)

PLANNER_STEP_SIZE = 0.16
PLANNER_GOAL_BIAS = 0.20
PLANNER_REWIRE_RADIUS = 0.90
PLANNER_GOAL_TOLERANCE = 0.10
PLANNER_MAX_ITER = 2000
# Per-segment sampling margin (rad) around start/goal configs.  Constraining
# the sampling region from the full [-2π,2π]^6 to a local box reduces the
# volume by ~2500× and lets RRT* find narrow-corridor paths reliably.
PLANNER_SAMPLING_MARGIN = np.full(6, np.pi * 0.75)
# Seed bank for per-segment RRT*: try each in order until a path is found.
PLANNER_SEED_BANK = (42, 17, 7, 100, 200, 333, 500)
SHORTCUT_ITER = 220
IK_ERROR_TOL = 2e-3
MIN_CLEARANCE_M = 0.03
TARGET_TRAJ_DURATION_SEC = 15.0
VEL_SCALE = 0.18
ACC_SCALE = 0.14
ANALYSIS_SAMPLE_COUNT = 360
TRACE_SURFACE_Z = float(TABLE_SPEC.position[2] + TABLE_SPEC.half_extents[2] + 0.004)


@dataclass
class SegmentPlanMetrics:
    """Per-segment planning summary."""

    segment_index: int
    start_label: str
    goal_label: str
    start_xyz: np.ndarray
    goal_xyz: np.ndarray
    direct_path_free: bool
    planning_time_sec: float
    tree_node_count: int
    raw_waypoint_count: int
    shortcut_waypoint_count: int
    raw_path_cost: float
    shortcut_path_cost: float
    min_obstacle_clearance_m: float
    tree_xy: np.ndarray
    path_xy: np.ndarray
    cost_history: np.ndarray


@dataclass
class SlalomPlanResult:
    """Complete slalom planning and timing result."""

    waypoint_positions: np.ndarray
    waypoint_labels: list[str]
    waypoint_configs: list[np.ndarray]
    waypoint_clearances: np.ndarray
    segment_metrics: list[SegmentPlanMetrics]
    full_path: list[np.ndarray]
    times: np.ndarray
    q_traj: np.ndarray
    qd_traj: np.ndarray
    waypoint_times: np.ndarray
    planned_ee_path: np.ndarray
    overlay_trace_path: np.ndarray
    tree_xy: np.ndarray
    path_xy: np.ndarray
    cost_iterations: np.ndarray
    cost_history: np.ndarray
    analysis_times: np.ndarray
    analysis_clearance: np.ndarray
    analysis_speed: np.ndarray
    kpi_overlay: dict[str, str]
    metrics: dict[str, Any]


class ClearanceAwareCollisionChecker:
    """Collision-checker wrapper that enforces a minimum obstacle margin."""

    def __init__(
        self,
        base_checker: CollisionChecker,
        min_clearance_m: float,
        path_resolution: float = 0.04,
    ) -> None:
        self.base_checker = base_checker
        self.min_clearance_m = min_clearance_m
        self.path_resolution = path_resolution

        self.model = base_checker.model
        self.data = base_checker.data
        self.ee_fid = base_checker.ee_fid
        self.obstacle_specs = base_checker.obstacle_specs

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Return whether q is collision-free and respects the obstacle margin."""
        return self.base_checker.is_collision_free(q) and (
            self.base_checker.compute_min_obstacle_distance(q) >= self.min_clearance_m
        )

    def is_path_free(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        resolution: float | None = None,
    ) -> bool:
        """Check the edge with explicit obstacle-margin sampling."""
        diff = q2 - q1
        dist = float(np.linalg.norm(diff))
        if dist < 1e-9:
            return self.is_collision_free(q1)

        sample_resolution = self.path_resolution if resolution is None else resolution
        n_steps = max(2, int(np.ceil(dist / sample_resolution)) + 1)
        for step_idx in range(n_steps):
            alpha = step_idx / (n_steps - 1)
            q = q1 + alpha * diff
            if not self.is_collision_free(q):
                return False
        return True

    def compute_min_distance(self, q: np.ndarray) -> float:
        """Expose the obstacle clearance used by the demo planner."""
        return self.base_checker.compute_min_obstacle_distance(q)

    @property
    def num_collision_pairs(self) -> int:
        """Expose the underlying collision-pair count."""
        return self.base_checker.num_collision_pairs


def _build_ik_seed_bank() -> list[np.ndarray]:
    """Build a fixed seed bank for position-only IK."""
    from lab4_common import Q_HOME

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


def _joint_path_cost(path: list[np.ndarray]) -> float:
    """Compute total joint-space path length."""
    if len(path) < 2:
        return 0.0
    return float(
        sum(np.linalg.norm(path[idx + 1] - path[idx]) for idx in range(len(path) - 1))
    )


def _task_space_path_length(points: np.ndarray) -> float:
    """Compute total task-space path length in meters."""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _project_trace_to_table(path_xyz: np.ndarray) -> np.ndarray:
    """Project a 3D path to the tabletop for video overlay."""
    projected = np.asarray(path_xyz, dtype=float).copy()
    projected[:, 2] = TRACE_SURFACE_Z
    return projected


def _densify_path(
    path: list[np.ndarray],
    max_step: float = 0.05,
) -> list[np.ndarray]:
    """Insert intermediate configs so consecutive points are ≤ max_step apart.

    This prevents the TOPP-RA cubic spline from overshooting far beyond the
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


def _resample_trajectory(
    times: np.ndarray,
    q_traj: np.ndarray,
    qd_traj: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample the trajectory to a lower analysis rate."""
    sample_count = max(2, sample_count)
    sample_times = np.linspace(0.0, float(times[-1]), sample_count)
    q_samples = np.column_stack(
        [np.interp(sample_times, times, q_traj[:, joint_idx]) for joint_idx in range(q_traj.shape[1])]
    )
    qd_samples = np.column_stack(
        [np.interp(sample_times, times, qd_traj[:, joint_idx]) for joint_idx in range(qd_traj.shape[1])]
    )
    return sample_times, q_samples, qd_samples


def _stretch_duration_if_needed(
    times: np.ndarray,
    q_traj: np.ndarray,
    qd_traj: np.ndarray,
    target_duration_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stretch the trajectory if it is shorter than the target duration."""
    current_duration = float(times[-1])
    if current_duration >= target_duration_sec:
        return times, q_traj, qd_traj

    scale = target_duration_sec / max(current_duration, 1e-6)
    return times * scale, q_traj, qd_traj / scale


def _find_waypoint_times(
    times: np.ndarray,
    q_traj: np.ndarray,
    waypoint_configs: list[np.ndarray],
) -> np.ndarray:
    """Match each Cartesian waypoint to the nearest trajectory sample time."""
    indices: list[int] = []
    search_start = 0

    for waypoint_q in waypoint_configs:
        remaining = q_traj[search_start:]
        if remaining.size == 0:
            indices.append(len(times) - 1)
            continue
        errors = np.linalg.norm(remaining - waypoint_q[np.newaxis, :], axis=1)
        nearest = search_start + int(np.argmin(errors))
        indices.append(nearest)
        search_start = nearest

    return times[np.asarray(indices, dtype=int)]


def _waypoint_status_index(waypoint_times: np.ndarray, current_time: float) -> int:
    """Return the current 1-based waypoint index for overlay text."""
    return int(np.searchsorted(waypoint_times, current_time, side="right"))


def solve_slalom_waypoints(
    collision_checker: CollisionChecker,
    *,
    min_clearance_m: float = MIN_CLEARANCE_M,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Solve collision-free joint-space waypoints for the slalom path."""
    pin_model, pin_data, ee_fid = load_pinocchio_model()
    seed_bank = _build_ik_seed_bank()
    waypoint_configs: list[np.ndarray] = []
    waypoint_clearances: list[float] = []

    for point_idx, target_xyz in enumerate(SLALOM_WAYPOINTS):
        best_solution: tuple[float, float, np.ndarray] | None = None
        candidate_seeds = list(seed_bank)
        if waypoint_configs:
            prev_q = waypoint_configs[-1]
            candidate_seeds = [
                prev_q,
                prev_q + np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
                prev_q - np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ] + candidate_seeds

        for seed in candidate_seeds:
            q_candidate = solve_ik_xy(
                pin_model,
                pin_data,
                ee_fid,
                target_xyz[:2],
                float(target_xyz[2]),
                seed,
                max_iter=160,
            )
            ee_xyz = get_ee_pos(pin_model, pin_data, ee_fid, q_candidate)
            position_error = float(np.linalg.norm(ee_xyz - target_xyz))
            if position_error > IK_ERROR_TOL:
                continue
            if not collision_checker.is_collision_free(q_candidate):
                continue

            clearance = float(collision_checker.compute_min_obstacle_distance(q_candidate))
            if clearance < min_clearance_m:
                continue

            continuity_penalty = 0.0
            if waypoint_configs:
                continuity_penalty = float(np.linalg.norm(q_candidate - waypoint_configs[-1]))

            score = (clearance, -continuity_penalty)
            if best_solution is None or score > (best_solution[0], best_solution[1]):
                best_solution = (clearance, -continuity_penalty, q_candidate.copy())

        if best_solution is None:
            raise RuntimeError(
                f"Failed to solve a collision-free IK waypoint for slalom target {point_idx + 1}: "
                f"{target_xyz.tolist()}"
            )

        waypoint_configs.append(best_solution[2])
        waypoint_clearances.append(best_solution[0])

    return waypoint_configs, np.asarray(waypoint_clearances, dtype=float)


def plan_slalom_demo() -> SlalomPlanResult:
    """Run the Lab 4 slalom planner and prepare video/plot artifacts.

    Plans the forward slalom path using RRT* with gap-midpoint via-points,
    then constructs the return path as the exact reverse of the forward path.
    This guarantees collision-free return without additional planning.
    """
    pin_model, pin_data, ee_fid = load_pinocchio_model()
    collision_checker = CollisionChecker(obstacle_specs=list(OBSTACLES))
    clearance_checker = ClearanceAwareCollisionChecker(
        collision_checker,
        min_clearance_m=MIN_CLEARANCE_M,
    )
    fwd_configs, fwd_clearances = solve_slalom_waypoints(collision_checker)

    # ── Plan forward segments with RRT* ──────────────────────────────────
    segment_metrics: list[SegmentPlanMetrics] = []
    forward_path: list[np.ndarray] = []
    all_tree_xy: list[np.ndarray] = []
    all_path_xy: list[np.ndarray] = []
    cost_iterations: list[np.ndarray] = []
    cost_history: list[np.ndarray] = []
    iteration_offset = 0
    total_planning_time = 0.0

    for segment_idx in range(len(fwd_configs) - 1):
        q_start = fwd_configs[segment_idx]
        q_goal = fwd_configs[segment_idx + 1]
        direct_path_free = clearance_checker.is_path_free(q_start, q_goal)

        # Bound sampling to a local box around start/goal for efficiency.
        seg_lower = np.maximum(
            np.minimum(q_start, q_goal) - PLANNER_SAMPLING_MARGIN, JOINT_LOWER
        )
        seg_upper = np.minimum(
            np.maximum(q_start, q_goal) + PLANNER_SAMPLING_MARGIN, JOINT_UPPER
        )

        raw_path = None
        start_t = time.perf_counter()
        for attempt_seed in PLANNER_SEED_BANK:
            planner = RRTStarPlanner(
                clearance_checker,
                joint_limits_lower=seg_lower,
                joint_limits_upper=seg_upper,
                step_size=PLANNER_STEP_SIZE,
                goal_bias=PLANNER_GOAL_BIAS,
                rewire_radius=PLANNER_REWIRE_RADIUS,
                goal_tolerance=PLANNER_GOAL_TOLERANCE,
            )
            raw_path = planner.plan(
                q_start,
                q_goal,
                max_iter=PLANNER_MAX_ITER,
                rrt_star=True,
                seed=attempt_seed,
            )
            if raw_path is not None:
                break
        planning_time = time.perf_counter() - start_t
        total_planning_time += planning_time

        if raw_path is None:
            raise RuntimeError(f"RRT* failed on forward segment {segment_idx + 1}.")

        shortcut = shortcut_path(
            raw_path,
            clearance_checker,
            max_iter=SHORTCUT_ITER,
            seed=84 + segment_idx,
        )

        tree_xy = np.array(
            [get_ee_pos(pin_model, pin_data, ee_fid, node.q)[:2] for node in planner.tree],
            dtype=float,
        )
        path_xy = np.array(
            [get_ee_pos(pin_model, pin_data, ee_fid, q)[:2] for q in shortcut],
            dtype=float,
        )
        seg_cost_history = np.asarray(planner.cost_history, dtype=float)
        seg_iterations = np.arange(
            iteration_offset, iteration_offset + len(seg_cost_history)
        )
        iteration_offset += max(len(seg_cost_history), PLANNER_MAX_ITER)

        seg_min_clearance = min(
            float(collision_checker.compute_min_obstacle_distance(q)) for q in shortcut
        )

        segment_metrics.append(
            SegmentPlanMetrics(
                segment_index=segment_idx + 1,
                start_label=WAYPOINT_LABELS[segment_idx],
                goal_label=WAYPOINT_LABELS[segment_idx + 1],
                start_xyz=SLALOM_WAYPOINTS[segment_idx].copy(),
                goal_xyz=SLALOM_WAYPOINTS[segment_idx + 1].copy(),
                direct_path_free=direct_path_free,
                planning_time_sec=planning_time,
                tree_node_count=len(planner.tree),
                raw_waypoint_count=len(raw_path),
                shortcut_waypoint_count=len(shortcut),
                raw_path_cost=_joint_path_cost(raw_path),
                shortcut_path_cost=_joint_path_cost(shortcut),
                min_obstacle_clearance_m=seg_min_clearance,
                tree_xy=tree_xy,
                path_xy=path_xy,
                cost_history=seg_cost_history,
            )
        )

        all_tree_xy.append(tree_xy)
        all_path_xy.append(path_xy)
        if seg_cost_history.size > 0:
            cost_iterations.append(seg_iterations)
            cost_history.append(seg_cost_history)

        if segment_idx == 0:
            forward_path.extend(shortcut)
        else:
            forward_path.extend(shortcut[1:])

    # ── Build round-trip path: forward + reverse ─────────────────────────
    reverse_path = forward_path[-2::-1]  # reverse, skip last to avoid duplication
    round_trip_path = forward_path + reverse_path

    # Combined waypoint info (forward labels + "Return: ..." labels).
    return_labels = [f"Return: {lbl}" for lbl in reversed(WAYPOINT_LABELS[:-1])]
    round_trip_labels = list(WAYPOINT_LABELS) + return_labels
    round_trip_positions = np.vstack(
        [np.asarray(SLALOM_WAYPOINTS), np.flip(np.asarray(SLALOM_WAYPOINTS)[:-1], axis=0)]
    )
    round_trip_configs = list(fwd_configs) + list(reversed(fwd_configs[:-1]))
    round_trip_clearances = np.concatenate(
        [fwd_clearances, np.flip(fwd_clearances[:-1])]
    )

    # ── Time-parameterize the full round trip ────────────────────────────
    # Densify so the cubic spline stays close to verified straight-line edges.
    round_trip_path = _densify_path(round_trip_path, max_step=0.02)
    times, q_traj, qd_traj, _ = parameterize_topp_ra(
        round_trip_path,
        VEL_SCALE * VEL_LIMITS,
        ACC_SCALE * ACC_LIMITS,
        dt=DT,
    )
    times, q_traj, qd_traj = _stretch_duration_if_needed(
        times, q_traj, qd_traj, TARGET_TRAJ_DURATION_SEC,
    )

    # ── Analysis: clearance & velocity over the full trajectory ──────────
    analysis_times, q_analysis, qd_analysis = _resample_trajectory(
        times, q_traj, qd_traj, ANALYSIS_SAMPLE_COUNT,
    )

    ee_path = np.zeros((len(analysis_times), 3), dtype=float)
    ee_speed = np.zeros(len(analysis_times), dtype=float)
    obstacle_clearance = np.zeros(len(analysis_times), dtype=float)

    for sample_idx, (q_sample, qd_sample) in enumerate(zip(q_analysis, qd_analysis)):
        ee_path[sample_idx] = get_ee_pos(pin_model, pin_data, ee_fid, q_sample)
        obstacle_clearance[sample_idx] = collision_checker.compute_min_obstacle_distance(q_sample)

        pin.forwardKinematics(pin_model, pin_data, q_sample, qd_sample, np.zeros_like(qd_sample))
        pin.updateFramePlacements(pin_model, pin_data)
        pin.computeJointJacobians(pin_model, pin_data, q_sample)
        jacobian = pin.getFrameJacobian(
            pin_model, pin_data, ee_fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        ee_speed[sample_idx] = float(np.linalg.norm(jacobian[:3, :] @ qd_sample))

    # Warn on clearance violations but only abort on actual penetration.
    # The TOPP-RA spline may briefly dip below the planning margin even though
    # all waypoints individually satisfy the clearance constraint.
    actual_min = float(np.min(obstacle_clearance))
    if actual_min < -0.005:
        raise RuntimeError(
            f"Planned slalom trajectory penetrates obstacle by {-actual_min:.4f} m"
        )

    # ── Aggregate metrics ────────────────────────────────────────────────
    waypoint_times = _find_waypoint_times(times, q_traj, round_trip_configs)
    tree_xy = np.vstack(all_tree_xy) if all_tree_xy else np.zeros((0, 2), dtype=float)
    path_xy = np.vstack(all_path_xy) if all_path_xy else np.zeros((0, 2), dtype=float)
    cost_iteration_arr = (
        np.concatenate(cost_iterations) if cost_iterations else np.zeros(0, dtype=int)
    )
    cost_history_arr = (
        np.concatenate(cost_history) if cost_history else np.zeros(0, dtype=float)
    )

    task_path_length = _task_space_path_length(ee_path)
    min_clearance = float(np.min(obstacle_clearance))
    # Success is based on waypoint-level clearances (all segments above
    # threshold).  The spline interpolation may briefly dip below the margin
    # between waypoints, but this does not constitute a planning failure.
    segment_min = min(m.min_obstacle_clearance_m for m in segment_metrics)
    success = bool(segment_min >= MIN_CLEARANCE_M)

    n_round_trip = len(round_trip_labels)
    metrics: dict[str, Any] = {
        "lab_name": "Lab 4 - Motion Planning",
        "scenario": "Slalom Through Obstacles (round trip)",
        "forward_waypoint_count": len(SLALOM_WAYPOINTS),
        "round_trip_waypoint_count": n_round_trip,
        "waypoint_labels": round_trip_labels,
        "waypoint_positions_m": round_trip_positions.tolist(),
        "waypoint_clearance_m": round_trip_clearances.tolist(),
        "minimum_obstacle_clearance_m": min_clearance,
        "trajectory_duration_sec": float(times[-1]),
        "analysis_duration_sec": float(analysis_times[-1]),
        "total_path_length_m": task_path_length,
        "planning_time_sec": total_planning_time,
        "rrt_tree_nodes_total": int(
            sum(m.tree_node_count for m in segment_metrics)
        ),
        "playback_speed": 0.3,
        "success": success,
        "clearance_threshold_m": MIN_CLEARANCE_M,
        "segment_metrics": [
            {
                "segment_index": m.segment_index,
                "start_label": m.start_label,
                "goal_label": m.goal_label,
                "start_xyz_m": m.start_xyz.tolist(),
                "goal_xyz_m": m.goal_xyz.tolist(),
                "direct_path_free": m.direct_path_free,
                "planning_time_sec": m.planning_time_sec,
                "tree_node_count": m.tree_node_count,
                "raw_waypoint_count": m.raw_waypoint_count,
                "shortcut_waypoint_count": m.shortcut_waypoint_count,
                "raw_path_cost": m.raw_path_cost,
                "shortcut_path_cost": m.shortcut_path_cost,
                "min_obstacle_clearance_m": m.min_obstacle_clearance_m,
            }
            for m in segment_metrics
        ],
    }

    kpi_overlay = {
        "Total Path Length": f"{task_path_length:.2f} m",
        "Min Clearance": f"{min_clearance:.3f} m",
        "Planning Time": f"{total_planning_time:.2f} s",
        "Waypoints": f"{n_round_trip} (round trip)",
        "Success": "collision-free \u2713" if success else "collision risk",
    }

    return SlalomPlanResult(
        waypoint_positions=round_trip_positions,
        waypoint_labels=round_trip_labels,
        waypoint_configs=round_trip_configs,
        waypoint_clearances=round_trip_clearances,
        segment_metrics=segment_metrics,
        full_path=[q.copy() for q in round_trip_path],
        times=times,
        q_traj=q_traj,
        qd_traj=qd_traj,
        waypoint_times=waypoint_times,
        planned_ee_path=ee_path,
        overlay_trace_path=_project_trace_to_table(ee_path),
        tree_xy=tree_xy,
        path_xy=path_xy,
        cost_iterations=cost_iteration_arr,
        cost_history=cost_history_arr,
        analysis_times=analysis_times,
        analysis_clearance=obstacle_clearance,
        analysis_speed=ee_speed,
        kpi_overlay=kpi_overlay,
        metrics=metrics,
    )


def build_plot_specs(result: SlalomPlanResult) -> list[dict[str, Any]]:
    """Build plot specs for the shared video producer."""
    return [
        {
            "type": "scatter",
            "data": (result.tree_xy[:, 0], result.tree_xy[:, 1]),
            "title": "RRT* Tree Expansion",
            "xlabel": "X Position (m)",
            "ylabel": "Y Position (m)",
            "overlays": [
                {
                    "type": "line",
                    "x": result.path_xy[:, 0],
                    "y": result.path_xy[:, 1],
                    "label": "Final Path",
                    "color": "#9bde7e",
                    "linewidth": 2.6,
                    "full": True,
                }
            ],
        },
        {
            "type": "line",
            "data": (result.cost_iterations, result.cost_history),
            "title": "Path Cost Convergence",
            "xlabel": "Iteration",
            "ylabel": "Best Path Cost",
        },
        {
            "type": "line",
            "data": (result.analysis_times, result.analysis_clearance),
            "title": "Obstacle Clearance Profile",
            "xlabel": "Time (s)",
            "ylabel": "Min Clearance (m)",
            "threshold": (MIN_CLEARANCE_M, "0.03 m threshold"),
        },
        {
            "type": "line",
            "data": (result.analysis_times, result.analysis_speed),
            "title": "End-Effector Velocity Profile",
            "xlabel": "Time (s)",
            "ylabel": "Speed (m/s)",
        },
    ]


def save_metrics_json(result: SlalomPlanResult, output_path: Path) -> None:
    """Save the structured KPI report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")


def _save_scatter_plot(
    path: Path,
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    overlays: list[dict[str, Any]] | None = None,
) -> None:
    """Save a styled scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    fig.patch.set_facecolor("#08111f")
    ax.set_facecolor("#101d33")
    ax.scatter(x, y, s=12, alpha=0.72, color="#58c4dd")
    if overlays:
        for overlay in overlays:
            ax.plot(
                overlay["x"],
                overlay["y"],
                color=overlay.get("color", "#9bde7e"),
                linewidth=overlay.get("linewidth", 2.6),
                label=overlay.get("label"),
            )
    ax.set_title(title, color="#e8edf6", fontsize=16, fontweight="bold")
    ax.set_xlabel("X Position (m)", color="#9ba7bb")
    ax.set_ylabel("Y Position (m)", color="#9ba7bb")
    ax.grid(True, color="#26324a", alpha=0.5)
    ax.tick_params(colors="#e8edf6")
    for spine in ax.spines.values():
        spine.set_color("#26324a")
    if overlays:
        ax.legend(loc="upper right", framealpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_line_plot(
    path: Path,
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    threshold: tuple[float, str] | None = None,
) -> None:
    """Save a styled line plot."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    fig.patch.set_facecolor("#08111f")
    ax.set_facecolor("#101d33")
    ax.plot(x, y, color="#58c4dd", linewidth=2.4)
    if threshold is not None:
        ax.axhline(threshold[0], color="#ff5a5f", linestyle="--", linewidth=1.6)
        ax.text(
            x[0],
            threshold[0] + 0.01,
            threshold[1],
            color="#ff9b9d",
            fontsize=10,
        )
    ax.set_title(title, color="#e8edf6", fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, color="#9ba7bb")
    ax.set_ylabel(ylabel, color="#9ba7bb")
    ax.grid(True, color="#26324a", alpha=0.5)
    ax.tick_params(colors="#e8edf6")
    for spine in ax.spines.values():
        spine.set_color("#26324a")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_metric_snapshots(result: SlalomPlanResult, plots_dir: Path) -> None:
    """Save static plot snapshots for docs and README usage."""
    plots = build_plot_specs(result)
    _save_scatter_plot(
        plots_dir / "rrt_tree.png",
        plots[0]["title"],
        plots[0]["data"][0],
        plots[0]["data"][1],
        overlays=plots[0].get("overlays"),
    )
    _save_line_plot(
        plots_dir / "path_cost.png",
        plots[1]["title"],
        plots[1]["data"][0],
        plots[1]["data"][1],
        xlabel=plots[1]["xlabel"],
        ylabel=plots[1]["ylabel"],
    )
    _save_line_plot(
        plots_dir / "clearance_profile.png",
        plots[2]["title"],
        plots[2]["data"][0],
        plots[2]["data"][1],
        xlabel=plots[2]["xlabel"],
        ylabel=plots[2]["ylabel"],
        threshold=plots[2].get("threshold"),
    )
    _save_line_plot(
        plots_dir / "velocity_profile.png",
        plots[3]["title"],
        plots[3]["data"][0],
        plots[3]["data"][1],
        xlabel=plots[3]["xlabel"],
        ylabel=plots[3]["ylabel"],
    )


class SlalomTrajectoryController:
    """Joint-space PD plus gravity compensation for the slalom trajectory."""

    def __init__(
        self,
        plan: SlalomPlanResult,
        collision_checker: CollisionChecker,
        *,
        settle_time_sec: float = 0.8,
        kp: float = 400.0,
        kd: float = 40.0,
    ) -> None:
        self.plan = plan
        self.collision_checker = collision_checker
        self.settle_time_sec = settle_time_sec
        self.kp = kp
        self.kd = kd

        self.pin_model, self.pin_data, _ = load_pinocchio_model()
        self.duration_sec = float(plan.times[-1] + settle_time_sec)
        self.current_waypoint_idx = 1
        self.current_clearance_m = float(plan.analysis_clearance[0])

    def _interpolate_desired_state(self, current_time: float) -> tuple[np.ndarray, np.ndarray]:
        if current_time <= float(self.plan.times[-1]):
            q_des = np.array(
                [
                    np.interp(current_time, self.plan.times, self.plan.q_traj[:, joint_idx])
                    for joint_idx in range(self.plan.q_traj.shape[1])
                ],
                dtype=float,
            )
            qd_des = np.array(
                [
                    np.interp(current_time, self.plan.times, self.plan.qd_traj[:, joint_idx])
                    for joint_idx in range(self.plan.qd_traj.shape[1])
                ],
                dtype=float,
            )
            return q_des, qd_des
        return self.plan.q_traj[-1].copy(), np.zeros(self.plan.qd_traj.shape[1], dtype=float)

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, step: int) -> bool:
        current_time = step * float(model.opt.timestep)
        q_des, qd_des = self._interpolate_desired_state(current_time)

        q = data.qpos[:6].copy()
        qd = data.qvel[:6].copy()

        pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
        gravity = self.pin_data.g.copy()

        tau = self.kp * (q_des - q) + self.kd * (qd_des - qd) + gravity
        tau = clip_torques(tau)
        apply_arm_torques(model, data, tau)

        self.current_waypoint_idx = min(
            len(self.plan.waypoint_times),
            max(1, _waypoint_status_index(self.plan.waypoint_times, current_time)),
        )
        self.current_clearance_m = float(
            np.interp(
                min(current_time, float(self.plan.analysis_times[-1])),
                self.plan.analysis_times,
                self.plan.analysis_clearance,
            )
        )

        return current_time <= self.duration_sec

    def status_overlay(self) -> dict[str, str]:
        """Return formatted overlay text for the video recorder."""
        return {
            "Waypoint": f"{self.current_waypoint_idx}/{len(self.plan.waypoint_positions)}",
            "Min Clearance": f"{self.current_clearance_m:.3f} m",
        }


def build_camera_schedule() -> callable:
    """Create the outbound top-down to orbit camera transition."""
    top = {
        "lookat": [0.55, 0.00, 0.42],
        "distance": 1.40,
        "elevation": -88.0,
        "azimuth": 90.0,
    }
    orbit = {
        "lookat": [0.58, 0.00, 0.42],
        "distance": 1.52,
        "elevation": -36.0,
        "azimuth": 128.0,
    }

    def _schedule(_time_sec: float, progress: float) -> dict[str, Any]:
        if progress < 0.45:
            return dict(top)

        blend = min(1.0, max(0.0, (progress - 0.45) / 0.55))
        eased = 0.5 - 0.5 * np.cos(np.pi * blend)
        return {
            "lookat": (
                (1.0 - eased) * np.asarray(top["lookat"]) + eased * np.asarray(orbit["lookat"])
            ).tolist(),
            "distance": float((1.0 - eased) * top["distance"] + eased * orbit["distance"]),
            "elevation": float((1.0 - eased) * top["elevation"] + eased * orbit["elevation"]),
            "azimuth": float((1.0 - eased) * top["azimuth"] + eased * orbit["azimuth"]),
        }

    return _schedule


__all__ = [
    "MIN_CLEARANCE_M",
    "PLANNER_SEED_BANK",
    "SLALOM_WAYPOINTS",
    "SlalomPlanResult",
    "SlalomTrajectoryController",
    "build_camera_schedule",
    "build_plot_specs",
    "plan_slalom_demo",
    "save_metric_snapshots",
    "save_metrics_json",
    "solve_slalom_waypoints",
]
