"""Generate the Lab 4 slalom demo video and supporting artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
_LAB_DIR = _SRC_DIR.parent
PROJECT_ROOT = _LAB_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab4_common import load_mujoco_model
from collision_checker import CollisionChecker
from slalom_demo import (
    SlalomTrajectoryController,
    build_camera_schedule,
    build_plot_specs,
    plan_slalom_demo,
    save_metric_snapshots,
    save_metrics_json,
)
from tools.video_producer import add_marker_to_scene, generate_demo_video


LAB_NAME = "Lab 4 — Motion Planning"
METRICS_TITLE = "RRT* Slalom Through Tabletop Obstacles"
PLAYBACK_SPEED = 0.5


def main() -> None:
    """Run the full slalom planning and demo-video pipeline."""
    media_dir = _LAB_DIR / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Lab 4 Demo Generation")
    print("=" * 72)

    print("\n[1/4] Planning slalom scenario...")
    plan = plan_slalom_demo()
    plan.metrics["playback_speed"] = PLAYBACK_SPEED
    metrics_path = media_dir / "slalom_metrics.json"
    save_metrics_json(plan, metrics_path)
    save_metric_snapshots(plan, media_dir)
    print(f"  Metrics saved to: {metrics_path}")
    print(f"  Plots saved to:   {media_dir}")

    print("\n[2/4] Preparing MuJoCo simulation...")
    mj_model, mj_data = load_mujoco_model()
    mj_data.qpos[:6] = plan.q_traj[0]
    mj_data.qvel[:6] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    controller = SlalomTrajectoryController(plan, collision_checker=CollisionChecker())

    waypoint_markers = plan.waypoint_positions.copy()
    waypoint_markers[:, 2] = float(plan.overlay_trace_path[0, 2])

    def overlay_fn(
        scene: mujoco.MjvScene,
        _model: mujoco.MjModel,
        _data: mujoco.MjData,
        _step: int,
    ) -> None:
        for point in waypoint_markers:
            add_marker_to_scene(
                scene,
                np.asarray(point, dtype=float),
                size=0.008,
                rgba=(0.95, 0.95, 0.95, 0.85),
            )
        current_idx = min(controller.current_waypoint_idx - 1, len(waypoint_markers) - 1)
        add_marker_to_scene(
            scene,
            waypoint_markers[current_idx],
            size=0.012,
            rgba=(1.0, 0.82, 0.24, 1.0),
        )

    print("\n[3/4] Rendering video clips...")
    final_video = generate_demo_video(
        lab_name=LAB_NAME,
        output_dir=media_dir,
        plots=build_plot_specs(plan),
        kpi_overlay=plan.kpi_overlay,
        metrics_title=METRICS_TITLE,
        model=mj_model,
        data=mj_data,
        controller_fn=controller,
        title_card_text=LAB_NAME,
        metrics_duration_sec=10.0,
        crossfade_sec=0.5,
        playback_speed=PLAYBACK_SPEED,
        trace_ee=True,
        duration_sec=controller.duration_sec,
        camera_schedule=build_camera_schedule(),
        trace_site_name="2f85_pinch",
        trace_project_z=float(plan.overlay_trace_path[0, 2]),
        planned_trace_points=plan.overlay_trace_path,
        overlay_fn=overlay_fn,
        status_text_fn=lambda _m, _d, _step, _t: controller.status_overlay(),
    )

    print("\n[4/4] Final artifact summary")
    print(f"  Metrics clip:    {media_dir / 'lab4_metrics.mp4'}")
    print(f"  Simulation clip: {media_dir / 'lab4_simulation.mp4'}")
    print(f"  Final demo:      {final_video}")
    print(f"  Min clearance:   {plan.metrics['minimum_obstacle_clearance_m']:.3f} m")
    print(f"  Traj duration:   {plan.metrics['trajectory_duration_sec']:.2f} s")

    print("\n" + "=" * 72)
    print("Lab 4 Demo Generation — COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
