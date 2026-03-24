"""Record the Lab 4 demo video: theory metrics + MuJoCo simulation.

Uses the shared video pipeline (tools/video_producer.py) with:
  Phase 1 — Animated metrics showing RRT vs RRT* comparison, path costs, tracking.
  Phase 2 — Native MuJoCo rendering of RRT* trajectory execution through obstacles.
  Phase 3 — ffmpeg composition with title/end cards and crossfades.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import mujoco
import numpy as np

_SRC = Path(__file__).resolve().parent
_LAB_DIR = _SRC.parent
_PROJECT_ROOT = _LAB_DIR.parent

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "tools"))

from collision_checker import CollisionChecker
from lab4_common import (
    ACC_LIMITS,
    DT,
    MEDIA_DIR,
    NUM_JOINTS,
    Q_HOME,
    TORQUE_LIMITS,
    VEL_LIMITS,
    apply_arm_torques,
    clip_torques,
    get_ee_pos,
    get_mj_ee_pos,
    load_mujoco_model,
    load_pinocchio_model,
)
from rrt_planner import RRTStarPlanner
from trajectory_executor import execute_trajectory
from trajectory_smoother import parameterize_topp_ra, shortcut_path
from video_producer import LabVideoProducer, add_marker_to_scene, add_polyline_to_scene

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAB_NAME = "Lab 4: Motion Planning & Collision Avoidance"
OUTPUT_DIR = MEDIA_DIR


def _path_cost(path: list[np.ndarray]) -> float:
    return sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))


# ---------------------------------------------------------------------------
# Run the planning pipeline and collect metrics
# ---------------------------------------------------------------------------

def compute_all_metrics() -> dict:
    """Run RRT, RRT*, shortcutting, TOPP-RA, execution and collect all metrics."""
    cc = CollisionChecker()
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    q_start = Q_HOME.copy()
    q_goal = np.array([0.3, -1.3, 1.0, -1.2, -np.pi / 2, 0.0])

    ee_start = get_ee_pos(pin_model, pin_data, ee_fid, q_start)
    ee_goal = get_ee_pos(pin_model, pin_data, ee_fid, q_goal)
    direct_free = cc.is_path_free(q_start, q_goal)

    # RRT planning
    print("  Planning with RRT...")
    planner_rrt = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.15)
    t0 = time.time()
    path_rrt = planner_rrt.plan(q_start, q_goal, max_iter=5000, rrt_star=False, seed=42)
    t_rrt = time.time() - t0
    assert path_rrt is not None, "RRT failed"

    # RRT* planning
    print("  Planning with RRT*...")
    planner_star = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.1, rewire_radius=1.0)
    t0 = time.time()
    path_star = planner_star.plan(q_start, q_goal, max_iter=5000, rrt_star=True, seed=42)
    t_star = time.time() - t0
    assert path_star is not None, "RRT* failed"

    # Shortcutting
    print("  Shortcutting paths...")
    short_rrt = shortcut_path(path_rrt, cc, max_iter=300, seed=42)
    short_star = shortcut_path(path_star, cc, max_iter=300, seed=42)

    # TOPP-RA
    print("  Time-parameterizing...")
    t_rrt_traj, q_rrt_traj, qd_rrt_traj, _ = parameterize_topp_ra(short_rrt, VEL_LIMITS, ACC_LIMITS)
    t_star_traj, q_star_traj, qd_star_traj, _ = parameterize_topp_ra(short_star, VEL_LIMITS, ACC_LIMITS)

    # Execution
    print("  Executing RRT trajectory...")
    res_rrt = execute_trajectory(t_rrt_traj, q_rrt_traj, qd_rrt_traj)
    print("  Executing RRT* trajectory...")
    res_star = execute_trajectory(t_star_traj, q_star_traj, qd_star_traj)

    rms_rrt = float(np.sqrt(np.mean((res_rrt["q_actual"] - res_rrt["q_desired"]) ** 2)))
    rms_star = float(np.sqrt(np.mean((res_star["q_actual"] - res_star["q_desired"]) ** 2)))
    final_rrt = float(np.linalg.norm(res_rrt["q_actual"][-1] - q_goal))
    final_star = float(np.linalg.norm(res_star["q_actual"][-1] - q_goal))

    # EE trajectory for plotting
    ee_rrt_traj = res_rrt["ee_pos"]
    ee_star_traj = res_star["ee_pos"]

    # Cost history from RRT*
    cost_history = planner_star.cost_history if hasattr(planner_star, "cost_history") else []

    return {
        "q_start": q_start,
        "q_goal": q_goal,
        "ee_start": ee_start,
        "ee_goal": ee_goal,
        "direct_free": direct_free,
        # RRT
        "rrt_waypoints": len(path_rrt),
        "rrt_cost": _path_cost(path_rrt),
        "rrt_tree_size": len(planner_rrt.tree),
        "rrt_time": t_rrt,
        "rrt_shortcut_waypoints": len(short_rrt),
        "rrt_shortcut_cost": _path_cost(short_rrt),
        "rrt_duration": float(t_rrt_traj[-1]),
        "rrt_rms": rms_rrt,
        "rrt_final_err": final_rrt,
        # RRT*
        "star_waypoints": len(path_star),
        "star_cost": _path_cost(path_star),
        "star_tree_size": len(planner_star.tree),
        "star_time": t_star,
        "star_shortcut_waypoints": len(short_star),
        "star_shortcut_cost": _path_cost(short_star),
        "star_duration": float(t_star_traj[-1]),
        "star_rms": rms_star,
        "star_final_err": final_star,
        # Cost convergence
        "cost_history": cost_history,
        # Execution results for plots
        "res_rrt": res_rrt,
        "res_star": res_star,
        # Trajectories for simulation
        "t_star_traj": t_star_traj,
        "q_star_traj": q_star_traj,
        "qd_star_traj": qd_star_traj,
        "short_star": short_star,
        "ee_star_traj": ee_star_traj,
        "ee_rrt_traj": ee_rrt_traj,
    }


# ---------------------------------------------------------------------------
# Build metrics plots (4 panels)
# ---------------------------------------------------------------------------

def build_plots(m: dict) -> list[dict]:
    res_rrt = m["res_rrt"]
    res_star = m["res_star"]

    # Plot 1: RRT vs RRT* tracking error
    err_rrt = np.max(np.abs(res_rrt["q_actual"] - res_rrt["q_desired"]), axis=1)
    err_star = np.max(np.abs(res_star["q_actual"] - res_star["q_desired"]), axis=1)
    plot_tracking = {
        "title": "Tracking Error (max joint)",
        "xlabel": "Time (s)",
        "ylabel": "Error (rad)",
        "type": "line",
        "series": [
            {"type": "line", "x": res_rrt["time"], "y": err_rrt, "label": "RRT", "color": "#ff9e66"},
            {"type": "line", "x": res_star["time"], "y": err_star, "label": "RRT*", "color": "#58c4dd"},
        ],
    }

    # Plot 2: EE trajectory XY
    plot_ee = {
        "title": "End-Effector XY Path",
        "xlabel": "X (m)",
        "ylabel": "Y (m)",
        "type": "line",
        "series": [
            {"type": "line", "x": m["ee_rrt_traj"][:, 0], "y": m["ee_rrt_traj"][:, 1],
             "label": "RRT", "color": "#ff9e66"},
            {"type": "line", "x": m["ee_star_traj"][:, 0], "y": m["ee_star_traj"][:, 1],
             "label": "RRT*", "color": "#58c4dd"},
        ],
    }

    # Plot 3: RRT* cost convergence
    if m["cost_history"]:
        hist = np.array(m["cost_history"])
        plot_cost = {
            "title": "RRT* Path Cost Convergence",
            "xlabel": "Iteration",
            "ylabel": "Cost (rad)",
            "data": (np.arange(len(hist)), hist),
            "color": "#9bde7e",
        }
    else:
        plot_cost = {
            "title": "RRT* Path Cost Convergence",
            "xlabel": "Iteration",
            "ylabel": "Cost (rad)",
            "data": (np.array([0, 1]), np.array([m["star_cost"], m["star_cost"]])),
            "color": "#9bde7e",
        }

    # Plot 4: Joint torques during RRT* execution
    tau_max = np.max(np.abs(res_star["tau"]), axis=1)
    plot_torque = {
        "title": "RRT* Max Joint Torque",
        "xlabel": "Time (s)",
        "ylabel": "Torque (Nm)",
        "data": (res_star["time"], tau_max),
        "color": "#c792ea",
    }

    return [plot_tracking, plot_ee, plot_cost, plot_torque]


def build_kpi(m: dict) -> dict[str, str]:
    return {
        "RRT Tree": f"{m['rrt_tree_size']} nodes",
        "RRT* Tree": f"{m['star_tree_size']} nodes",
        "RRT* Cost": f"{m['star_shortcut_cost']:.2f} rad",
        "Track RMS": f"{m['star_rms']:.4f} rad",
        "Final Error": f"{m['star_final_err']:.4f} rad",
        "Duration": f"{m['star_duration']:.2f} s",
    }


# ---------------------------------------------------------------------------
# Controller for MuJoCo simulation recording
# ---------------------------------------------------------------------------

def make_trajectory_controller(
    t_traj: np.ndarray,
    q_traj: np.ndarray,
    qd_traj: np.ndarray,
    Kp: float = 400.0,
    Kd: float = 40.0,
) -> tuple:
    """Create a controller function for MuJoCo recording of trajectory execution."""
    import pinocchio as pin_lib

    pin_model, pin_data, _ = load_pinocchio_model()
    traj_duration = float(t_traj[-1])
    total_duration = traj_duration + 1.0

    def controller_fn(model: mujoco.MjModel, data: mujoco.MjData, step: int) -> bool | None:
        t = step * DT

        if t <= traj_duration:
            q_d = np.array([np.interp(t, t_traj, q_traj[:, j]) for j in range(NUM_JOINTS)])
            qd_d = np.array([np.interp(t, t_traj, qd_traj[:, j]) for j in range(NUM_JOINTS)])
        else:
            q_d = q_traj[-1]
            qd_d = np.zeros(NUM_JOINTS)

        q = data.qpos[:NUM_JOINTS].copy()
        qd = data.qvel[:NUM_JOINTS].copy()

        pin_lib.computeGeneralizedGravity(pin_model, pin_data, q)
        g = pin_data.g.copy()

        tau = Kp * (q_d - q) + Kd * (qd_d - qd) + g
        tau = clip_torques(tau)
        apply_arm_torques(model, data, tau)

        if t > total_duration:
            return False
        return None

    return controller_fn, total_duration


def make_status_text_fn(m: dict):
    """Create a status overlay for the simulation recording."""
    traj_duration = float(m["t_star_traj"][-1])

    def status_fn(model, data, step, current_time):
        ee_pos = get_mj_ee_pos(model, data)
        phase = "Tracking" if current_time <= traj_duration else "Holding"
        return {
            "RRT* Trajectory Execution": "",
            f"Time: {current_time:.1f}s / {traj_duration:.1f}s": "",
            f"Phase: {phase}": "",
            f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]": "",
        }

    return status_fn


def camera_schedule(current_time: float, progress: float) -> dict:
    """Orbit camera during simulation."""
    azimuth = 140.0 + 40.0 * progress
    elevation = -30.0 - 15.0 * progress
    return {
        "lookat": [0.45, 0.0, 0.42],
        "distance": 1.50,
        "elevation": elevation,
        "azimuth": azimuth,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Path:
    print("=" * 60)
    print("Lab 4 Demo Video — Theory + Simulation")
    print("=" * 60)

    # Step 1: Compute all metrics
    print("\n[1/3] Running planning and execution pipeline...")
    metrics = compute_all_metrics()

    # Step 2: Build video producer
    producer = LabVideoProducer(lab_name=LAB_NAME, output_dir=OUTPUT_DIR)

    # Step 3: Create metrics clip
    print("\n[2/3] Creating animated metrics clip...")
    plots = build_plots(metrics)
    kpi = build_kpi(metrics)
    metrics_clip = producer.create_metrics_clip(
        plots=plots,
        kpi_overlay=kpi,
        title_text="RRT vs RRT* Planning Comparison",
        duration_sec=12.0,
    )
    print(f"  Metrics clip: {metrics_clip}")

    # Step 4: Record MuJoCo simulation (RRT* execution)
    print("\n[3/3] Recording MuJoCo simulation...")
    mj_model, mj_data = load_mujoco_model()

    mj_data.qpos[:NUM_JOINTS] = metrics["q_start"].copy()
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    controller_fn, total_duration = make_trajectory_controller(
        metrics["t_star_traj"],
        metrics["q_star_traj"],
        metrics["qd_star_traj"],
    )

    # Planned EE path for visual trace
    pin_model, pin_data, ee_fid = load_pinocchio_model()
    planned_ee = np.array([
        get_ee_pos(pin_model, pin_data, ee_fid, q)
        for q in metrics["short_star"]
    ])

    simulation_clip = producer.record_simulation(
        model=mj_model,
        data=mj_data,
        controller_fn=controller_fn,
        camera_name="orbit_45",
        playback_speed=0.4,
        trace_ee=True,
        trace_site_name="2f85_pinch",
        duration_sec=total_duration + 1.0,
        camera_schedule=camera_schedule,
        planned_trace_points=planned_ee,
        planned_trace_color=(0.22, 0.92, 0.35, 0.80),
        actual_trace_color=(0.22, 0.52, 0.98, 0.90),
        status_text_fn=make_status_text_fn(metrics),
        start_hold_sec=1.5,
        end_hold_sec=2.5,
    )
    print(f"  Simulation clip: {simulation_clip}")

    # Step 5: Compose final video
    print("\nComposing final video...")
    final_path = producer.compose_final_video(
        metrics_clip=metrics_clip,
        simulation_clip=simulation_clip,
        title_card_text="Lab 4: Motion Planning & Collision Avoidance",
        crossfade_sec=0.6,
    )
    print(f"\nDemo video saved: {final_path}")
    return final_path


if __name__ == "__main__":
    main()
