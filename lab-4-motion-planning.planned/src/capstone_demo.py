"""Lab 4 Capstone — Slalom Through Obstacles

Solve collision-free IK for tabletop slalom waypoints, plan each segment
with RRT*, shortcut and time-parameterize the full path, then execute on
the torque-controlled UR5e with PD + gravity compensation.

Usage:
    python3 lab-4-motion-planning/src/capstone_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_LAB3_SRC = _SRC.parent.parent / "lab-3-dynamics-force-control" / "src"
if str(_LAB3_SRC) not in sys.path:
    sys.path.insert(0, str(_LAB3_SRC))

from c1_force_control import solve_ik_xy
from collision_checker import CollisionChecker
from lab4_common import (
    ACC_LIMITS,
    ACC_SCALE,
    DT,
    IK_ERROR_TOL,
    JOINT_LOWER,
    JOINT_UPPER,
    MEDIA_DIR,
    MIN_CLEARANCE_M,
    OBSTACLES,
    PLANNER_GOAL_BIAS,
    PLANNER_GOAL_TOLERANCE,
    PLANNER_MAX_ITER,
    PLANNER_REWIRE_RADIUS,
    PLANNER_SAMPLING_MARGIN,
    PLANNER_SEED_BANK,
    PLANNER_STEP_SIZE,
    SHORTCUT_ITER,
    SLALOM_LABELS,
    SLALOM_WAYPOINTS,
    VEL_LIMITS,
    VEL_SCALE,
    build_ik_seed_bank,
    densify_path,
    get_ee_pos,
    load_pinocchio_model,
)
from rrt_planner import RRTStarPlanner
from trajectory_executor import execute_trajectory
from trajectory_smoother import parameterize_topp_ra, shortcut_path


# ---------------------------------------------------------------------------
# Clearance-aware collision checking
# ---------------------------------------------------------------------------


class ClearanceAwareChecker:
    """Collision checker wrapper that enforces a minimum obstacle margin."""

    def __init__(
        self,
        base: CollisionChecker,
        min_clearance: float = MIN_CLEARANCE_M,
    ) -> None:
        self.base = base
        self.min_clearance = min_clearance

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Return True if q is collision-free and respects the obstacle margin."""
        return self.base.is_collision_free(q) and (
            self.base.compute_min_obstacle_distance(q) >= self.min_clearance
        )

    def is_path_free(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        resolution: float = 0.04,
    ) -> bool:
        """Check the edge with obstacle-margin sampling."""
        diff = q2 - q1
        dist = float(np.linalg.norm(diff))
        if dist < 1e-9:
            return self.is_collision_free(q1)
        n = max(2, int(np.ceil(dist / resolution)) + 1)
        for i in range(n):
            if not self.is_collision_free(q1 + (i / (n - 1)) * diff):
                return False
        return True


# ---------------------------------------------------------------------------
# IK solving
# ---------------------------------------------------------------------------


def solve_waypoints(
    cc: CollisionChecker,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Solve collision-free IK configs for each SLALOM_WAYPOINTS entry."""
    pin_model, pin_data, ee_fid = load_pinocchio_model()
    seeds = build_ik_seed_bank()
    configs: list[np.ndarray] = []
    clearances: list[float] = []

    for idx, target in enumerate(SLALOM_WAYPOINTS):
        best: tuple[float, float, np.ndarray] | None = None
        candidates = list(seeds)
        if configs:
            prev = configs[-1]
            candidates = [
                prev,
                prev + np.array([0.5, 0, 0, 0, 0, 0]),
                prev - np.array([0.5, 0, 0, 0, 0, 0]),
            ] + candidates

        for seed in candidates:
            q = solve_ik_xy(
                pin_model, pin_data, ee_fid,
                target[:2], float(target[2]), seed, max_iter=160,
            )
            ee = get_ee_pos(pin_model, pin_data, ee_fid, q)
            if np.linalg.norm(ee - target) > IK_ERROR_TOL:
                continue
            if not cc.is_collision_free(q):
                continue
            clr = float(cc.compute_min_obstacle_distance(q))
            if clr < MIN_CLEARANCE_M:
                continue
            cont = float(np.linalg.norm(q - configs[-1])) if configs else 0.0
            score = (clr, -cont)
            if best is None or score > (best[0], best[1]):
                best = (clr, -cont, q.copy())

        if best is None:
            raise RuntimeError(
                f"IK failed for waypoint {idx} ({SLALOM_LABELS[idx]}): {target}"
            )
        configs.append(best[2])
        clearances.append(best[0])

    return configs, np.array(clearances)


# ---------------------------------------------------------------------------
# Per-segment RRT* planning
# ---------------------------------------------------------------------------


def plan_segments(
    checker: ClearanceAwareChecker,
    configs: list[np.ndarray],
) -> tuple[list[np.ndarray], float, int]:
    """Plan each segment with RRT* and concatenate into a full path.

    Returns:
        full_path: Concatenated collision-free waypoint list.
        total_planning_time: Wall-clock seconds for all segments.
        total_tree_nodes: Sum of RRT* tree sizes across segments.
    """
    full_path: list[np.ndarray] = []
    total_time = 0.0
    total_nodes = 0

    for i in range(len(configs) - 1):
        q_s, q_g = configs[i], configs[i + 1]
        lower = np.maximum(
            np.minimum(q_s, q_g) - PLANNER_SAMPLING_MARGIN, JOINT_LOWER,
        )
        upper = np.minimum(
            np.maximum(q_s, q_g) + PLANNER_SAMPLING_MARGIN, JOINT_UPPER,
        )

        raw = None
        t0 = time.perf_counter()
        for seed in PLANNER_SEED_BANK:
            planner = RRTStarPlanner(
                checker,
                joint_limits_lower=lower,
                joint_limits_upper=upper,
                step_size=PLANNER_STEP_SIZE,
                goal_bias=PLANNER_GOAL_BIAS,
                rewire_radius=PLANNER_REWIRE_RADIUS,
                goal_tolerance=PLANNER_GOAL_TOLERANCE,
            )
            raw = planner.plan(
                q_s, q_g, max_iter=PLANNER_MAX_ITER, rrt_star=True, seed=seed,
            )
            if raw is not None:
                break
        dt = time.perf_counter() - t0
        total_time += dt

        if raw is None:
            raise RuntimeError(f"RRT* failed on segment {i + 1}")

        short = shortcut_path(raw, checker, max_iter=SHORTCUT_ITER, seed=84 + i)
        total_nodes += len(planner.tree)

        if i == 0:
            full_path.extend(short)
        else:
            full_path.extend(short[1:])

        print(
            f"    Seg {i + 1}/{len(configs) - 1}: "
            f"{len(raw)} -> {len(short)} wps, "
            f"tree={len(planner.tree)}, {dt:.2f}s"
        )

    return full_path, total_time, total_nodes


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _plot_ee_slalom(ee_pos: np.ndarray, save_path: Path) -> None:
    """Plot EE trajectory XY with obstacle footprints."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    fig.patch.set_facecolor("#08111f")
    ax.set_facecolor("#101d33")

    for obs in OBSTACLES:
        x, y, _ = obs.position
        hx, hy, _ = obs.half_extents
        rect = plt.Rectangle(
            (x - hx, y - hy), 2 * hx, 2 * hy,
            facecolor="red", alpha=0.35, edgecolor="red", linewidth=1.2,
        )
        ax.add_patch(rect)

    ax.plot(ee_pos[:, 0], ee_pos[:, 1], color="#58c4dd", linewidth=2.0, label="EE Path")
    ax.scatter(ee_pos[0, 0], ee_pos[0, 1], c="#9bde7e", s=80, zorder=5, label="Start")
    ax.scatter(ee_pos[-1, 0], ee_pos[-1, 1], c="#ff5a5f", s=80, zorder=5, label="Goal")

    ax.set_title("End-Effector Slalom Path", color="#e8edf6", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)", color="#9ba7bb")
    ax.set_ylabel("Y (m)", color="#9ba7bb")
    ax.legend(loc="upper right", framealpha=0.3)
    ax.grid(True, color="#26324a", alpha=0.5)
    ax.tick_params(colors="#e8edf6")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color("#26324a")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_clearance(
    times: np.ndarray, clearance: np.ndarray, save_path: Path,
) -> None:
    """Plot obstacle clearance over time."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    fig.patch.set_facecolor("#08111f")
    ax.set_facecolor("#101d33")
    ax.plot(times, clearance, color="#58c4dd", linewidth=2.0)
    ax.axhline(MIN_CLEARANCE_M, color="#ff5a5f", linestyle="--", linewidth=1.4)
    ax.text(
        times[0], MIN_CLEARANCE_M + 0.005, f"{MIN_CLEARANCE_M} m threshold",
        color="#ff9b9d", fontsize=9,
    )
    ax.set_title("Obstacle Clearance", color="#e8edf6", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)", color="#9ba7bb")
    ax.set_ylabel("Min Clearance (m)", color="#9ba7bb")
    ax.grid(True, color="#26324a", alpha=0.5)
    ax.tick_params(colors="#e8edf6")
    for spine in ax.spines.values():
        spine.set_color("#26324a")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_tracking(result: dict, save_path: Path) -> None:
    """Plot max joint tracking error over time."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    fig.patch.set_facecolor("#08111f")
    ax.set_facecolor("#101d33")
    err = np.max(np.abs(result["q_actual"] - result["q_desired"]), axis=1)
    ax.plot(result["time"], err, color="#c792ea", linewidth=2.0)
    ax.set_title(
        "Max Joint Tracking Error", color="#e8edf6", fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Time (s)", color="#9ba7bb")
    ax.set_ylabel("Error (rad)", color="#9ba7bb")
    ax.grid(True, color="#26324a", alpha=0.5)
    ax.tick_params(colors="#e8edf6")
    for spine in ax.spines.values():
        spine.set_color("#26324a")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_capstone() -> dict:
    """Run the full slalom capstone demonstration.

    Returns a dict of results for use by the demo video recorder.
    """
    print("=" * 60)
    print("Lab 4 Capstone: Slalom Through Obstacles")
    print("=" * 60)

    cc = CollisionChecker(obstacle_specs=list(OBSTACLES))
    cc_clr = ClearanceAwareChecker(cc)
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    # 1. IK waypoints
    print("\n[1/5] Solving IK for slalom waypoints...")
    configs, wp_clearances = solve_waypoints(cc)
    for i, (cfg, clr, lbl) in enumerate(
        zip(configs, wp_clearances, SLALOM_LABELS)
    ):
        ee = get_ee_pos(pin_model, pin_data, ee_fid, cfg)
        print(
            f"  WP{i} ({lbl:16s}): "
            f"EE=[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]  clr={clr:.3f}m"
        )

    # 2. Per-segment RRT*
    print("\n[2/5] Planning per-segment RRT*...")
    full_path, planning_time, tree_nodes = plan_segments(cc_clr, configs)
    print(
        f"  Total: {len(full_path)} waypoints, "
        f"{tree_nodes} tree nodes, {planning_time:.2f}s"
    )

    # 3. Densify + TOPP-RA
    print("\n[3/5] Time-parameterizing...")
    dense = densify_path(full_path, max_step=0.02)
    times, q_traj, qd_traj, _ = parameterize_topp_ra(
        dense, VEL_SCALE * VEL_LIMITS, ACC_SCALE * ACC_LIMITS, dt=DT,
    )
    print(f"  Duration: {times[-1]:.2f}s, samples: {len(times)}")

    # 4. Execute
    print("\n[4/5] Executing trajectory...")
    result = execute_trajectory(times, q_traj, qd_traj)
    rms = float(np.sqrt(np.mean((result["q_actual"] - result["q_desired"]) ** 2)))
    final_err = float(np.linalg.norm(result["q_actual"][-1] - configs[-1]))
    print(f"  RMS tracking: {rms:.4f} rad, final error: {final_err:.4f} rad")

    # 5. Clearance analysis
    print("\n[5/5] Analyzing obstacle clearance...")
    n_samples = min(360, len(result["time"]))
    sample_idx = np.linspace(0, len(result["time"]) - 1, n_samples, dtype=int)
    clr_profile = np.array([
        cc.compute_min_obstacle_distance(result["q_actual"][i])
        for i in sample_idx
    ])
    min_clr = float(np.min(clr_profile))
    print(f"  Min obstacle clearance: {min_clr:.4f}m")

    # 6. Plots
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    _plot_ee_slalom(result["ee_pos"], MEDIA_DIR / "capstone_ee_trajectory.png")
    _plot_clearance(
        result["time"][sample_idx], clr_profile,
        MEDIA_DIR / "capstone_clearance.png",
    )
    _plot_tracking(result, MEDIA_DIR / "capstone_tracking.png")

    # Summary table
    print("\n" + "=" * 60)
    print("Capstone Summary")
    print("=" * 60)
    print(f"  {'Slalom waypoints:':<24} {len(SLALOM_WAYPOINTS)}")
    print(f"  {'Path waypoints:':<24} {len(full_path)}")
    print(f"  {'Tree nodes:':<24} {tree_nodes}")
    print(f"  {'Planning time:':<24} {planning_time:.2f}s")
    print(f"  {'Trajectory duration:':<24} {times[-1]:.2f}s")
    print(f"  {'RMS tracking error:':<24} {rms:.4f} rad")
    print(f"  {'Final error:':<24} {final_err:.4f} rad")
    print(f"  {'Min clearance:':<24} {min_clr:.4f}m")
    print(f"  {'Plots saved to:':<24} {MEDIA_DIR}/")

    return {
        "configs": configs,
        "full_path": full_path,
        "dense_path": dense,
        "times": times,
        "q_traj": q_traj,
        "qd_traj": qd_traj,
        "result": result,
        "planning_time": planning_time,
        "tree_nodes": tree_nodes,
        "rms": rms,
        "final_err": final_err,
        "min_clearance": min_clr,
        "clr_profile": clr_profile,
        "sample_times": result["time"][sample_idx],
    }


if __name__ == "__main__":
    run_capstone()
