"""Lab 4 Capstone — Full motion planning pipeline.

Plan → Shortcut → TOPP-RA → Execute on torque-controlled UR5e.
Generates comparison plots and task-space visualization.
"""

from __future__ import annotations

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from collision_checker import CollisionChecker
from lab4_common import (
    ACC_LIMITS,
    MEDIA_DIR,
    Q_HOME,
    VEL_LIMITS,
    get_ee_pos,
    load_pinocchio_model,
)
from rrt_planner import RRTStarPlanner, visualize_plan
from trajectory_executor import execute_trajectory
from trajectory_smoother import parameterize_topp_ra, shortcut_path


def run_capstone() -> None:
    """Run the full capstone demonstration."""
    print("=" * 60)
    print("Lab 4 Capstone: Motion Planning & Collision Avoidance")
    print("=" * 60)

    # --- Setup ---
    cc = CollisionChecker()
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    # Start: Q_HOME (arm tucked)
    q_start = Q_HOME.copy()
    # Goal: arm rotated to other side, reaching toward table
    q_goal = np.array([0.3, -1.3, 1.0, -1.2, -np.pi / 2, 0.0])

    ee_start = get_ee_pos(pin_model, pin_data, ee_fid, q_start)
    ee_goal = get_ee_pos(pin_model, pin_data, ee_fid, q_goal)

    print(f"\nStart config: {q_start}")
    print(f"Goal config:  {q_goal}")
    print(f"Start EE pos: {ee_start}")
    print(f"Goal EE pos:  {ee_goal}")
    print(f"Start free:   {cc.is_collision_free(q_start)}")
    print(f"Goal free:    {cc.is_collision_free(q_goal)}")
    print(f"Direct path free: {cc.is_path_free(q_start, q_goal)}")

    # --- RRT Planning ---
    print("\n--- RRT Planning ---")
    planner_rrt = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.15)
    t0 = time.time()
    path_rrt = planner_rrt.plan(
        q_start, q_goal, max_iter=5000, rrt_star=False, seed=42
    )
    t_rrt = time.time() - t0
    assert path_rrt is not None, "RRT failed to find path"
    cost_rrt = _path_cost(path_rrt)
    print(f"RRT: {len(path_rrt)} waypoints, cost={cost_rrt:.3f}, "
          f"tree={len(planner_rrt.tree)}, time={t_rrt:.2f}s")

    # --- RRT* Planning ---
    print("\n--- RRT* Planning ---")
    planner_star = RRTStarPlanner(
        cc, step_size=0.3, goal_bias=0.1, rewire_radius=1.0
    )
    t0 = time.time()
    path_star = planner_star.plan(
        q_start, q_goal, max_iter=5000, rrt_star=True, seed=42
    )
    t_star = time.time() - t0
    assert path_star is not None, "RRT* failed to find path"
    cost_star = _path_cost(path_star)
    print(f"RRT*: {len(path_star)} waypoints, cost={cost_star:.3f}, "
          f"tree={len(planner_star.tree)}, time={t_star:.2f}s")

    # --- Shortcutting ---
    print("\n--- Path Shortcutting ---")
    short_rrt = shortcut_path(path_rrt, cc, max_iter=300, seed=42)
    short_star = shortcut_path(path_star, cc, max_iter=300, seed=42)
    print(f"RRT shortcut:  {len(path_rrt)} -> {len(short_rrt)} waypoints, "
          f"cost={_path_cost(short_rrt):.3f}")
    print(f"RRT* shortcut: {len(path_star)} -> {len(short_star)} waypoints, "
          f"cost={_path_cost(short_star):.3f}")

    # --- TOPP-RA ---
    print("\n--- TOPP-RA Parameterization ---")
    t_rrt_traj, q_rrt_traj, qd_rrt_traj, _ = parameterize_topp_ra(
        short_rrt, VEL_LIMITS, ACC_LIMITS
    )
    t_star_traj, q_star_traj, qd_star_traj, _ = parameterize_topp_ra(
        short_star, VEL_LIMITS, ACC_LIMITS
    )
    print(f"RRT trajectory:  duration={t_rrt_traj[-1]:.3f}s, "
          f"samples={len(t_rrt_traj)}")
    print(f"RRT* trajectory: duration={t_star_traj[-1]:.3f}s, "
          f"samples={len(t_star_traj)}")

    # --- Execution ---
    print("\n--- Trajectory Execution ---")
    res_rrt = execute_trajectory(t_rrt_traj, q_rrt_traj, qd_rrt_traj)
    res_star = execute_trajectory(t_star_traj, q_star_traj, qd_star_traj)

    rms_rrt = np.sqrt(np.mean((res_rrt["q_actual"] - res_rrt["q_desired"]) ** 2))
    rms_star = np.sqrt(np.mean((res_star["q_actual"] - res_star["q_desired"]) ** 2))
    final_rrt = np.linalg.norm(res_rrt["q_actual"][-1] - q_goal)
    final_star = np.linalg.norm(res_star["q_actual"][-1] - q_goal)

    print(f"RRT execution:  RMS={rms_rrt:.4f} rad, final_err={final_rrt:.4f} rad")
    print(f"RRT* execution: RMS={rms_star:.4f} rad, final_err={final_star:.4f} rad")

    # --- Visualizations ---
    print("\n--- Generating Visualizations ---")
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    # Tree + path visualizations
    visualize_plan(
        planner_rrt, path_rrt, ee_fid,
        title="RRT Plan (Capstone)", save_path=MEDIA_DIR / "capstone_rrt_tree.png",
    )
    visualize_plan(
        planner_star, path_star, ee_fid,
        title="RRT* Plan (Capstone)", save_path=MEDIA_DIR / "capstone_rrt_star_tree.png",
    )

    # Execution comparison
    _plot_execution_comparison(
        res_rrt, res_star, q_goal,
        save_path=MEDIA_DIR / "capstone_execution_comparison.png",
    )

    # EE trajectory in 3D
    _plot_ee_trajectory(
        res_rrt, res_star,
        save_path=MEDIA_DIR / "capstone_ee_trajectory.png",
    )

    # Summary
    print("\n" + "=" * 60)
    print("Capstone Summary")
    print("=" * 60)
    print(f"{'Metric':<30} {'RRT':>12} {'RRT*':>12}")
    print("-" * 54)
    print(f"{'Planning time (s)':<30} {t_rrt:>12.2f} {t_star:>12.2f}")
    print(f"{'Tree nodes':<30} {len(planner_rrt.tree):>12} {len(planner_star.tree):>12}")
    print(f"{'Raw path cost':<30} {cost_rrt:>12.3f} {cost_star:>12.3f}")
    print(f"{'Shortcut cost':<30} {_path_cost(short_rrt):>12.3f} {_path_cost(short_star):>12.3f}")
    print(f"{'Trajectory duration (s)':<30} {t_rrt_traj[-1]:>12.3f} {t_star_traj[-1]:>12.3f}")
    print(f"{'Tracking RMS (rad)':<30} {rms_rrt:>12.4f} {rms_star:>12.4f}")
    print(f"{'Final error (rad)':<30} {final_rrt:>12.4f} {final_star:>12.4f}")
    print(f"\nVisualizations saved to {MEDIA_DIR}/")


def _path_cost(path: list[np.ndarray]) -> float:
    """Compute total C-space path length."""
    return sum(
        np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)
    )


def _plot_execution_comparison(
    res_rrt: dict,
    res_star: dict,
    q_goal: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot tracking error and torques for RRT vs RRT*."""
    from pathlib import Path

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    labels = [f"J{i+1}" for i in range(6)]

    for j in range(6):
        err = res_rrt["q_actual"][:, j] - res_rrt["q_desired"][:, j]
        axes[0, 0].plot(res_rrt["time"], err, alpha=0.7, label=labels[j])
    axes[0, 0].set_title("RRT — Tracking Error")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Error (rad)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    for j in range(6):
        err = res_star["q_actual"][:, j] - res_star["q_desired"][:, j]
        axes[0, 1].plot(res_star["time"], err, alpha=0.7, label=labels[j])
    axes[0, 1].set_title("RRT* — Tracking Error")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Error (rad)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    for j in range(6):
        axes[1, 0].plot(res_rrt["time"], res_rrt["tau"][:, j], alpha=0.7, label=labels[j])
    axes[1, 0].set_title("RRT — Joint Torques")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Torque (Nm)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    for j in range(6):
        axes[1, 1].plot(res_star["time"], res_star["tau"][:, j], alpha=0.7, label=labels[j])
    axes[1, 1].set_title("RRT* — Joint Torques")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Torque (Nm)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    fig.suptitle("Capstone: RRT vs RRT* Execution Comparison", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_ee_trajectory(
    res_rrt: dict,
    res_star: dict,
    save_path: Path | None = None,
) -> None:
    """Plot EE trajectories in 3D."""
    from pathlib import Path

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ee_rrt = res_rrt["ee_pos"]
    ee_star = res_star["ee_pos"]

    ax.plot(ee_rrt[:, 0], ee_rrt[:, 1], ee_rrt[:, 2],
            "b-", alpha=0.7, linewidth=1.5, label="RRT")
    ax.plot(ee_star[:, 0], ee_star[:, 1], ee_star[:, 2],
            "g-", alpha=0.7, linewidth=1.5, label="RRT*")

    ax.scatter(*ee_rrt[0], c="blue", s=80, marker="^", label="Start")
    ax.scatter(*ee_rrt[-1], c="blue", s=80, marker="*")
    ax.scatter(*ee_star[-1], c="green", s=80, marker="*", label="Goal")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Capstone: End-Effector Trajectories")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_capstone()
