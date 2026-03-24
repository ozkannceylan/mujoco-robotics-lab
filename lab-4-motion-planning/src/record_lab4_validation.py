"""Record a Lab 4 validation video using native MuJoCo offscreen rendering.

Produces a real MuJoCo simulation video of tabletop obstacle-avoidance,
comparable to Lab 2 native recordings.

The scenario: UR5e moves from Q_HOME to a table-reaching configuration.
The direct path is blocked by colored obstacle boxes on the table, so the
RRT planner finds a collision-free path that lifts the arm over/around them.

Usage:
    python3 lab-4-motion-planning/src/record_lab4_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from collision_checker import CollisionChecker
from lab4_common import (
    ACC_LIMITS,
    CAPSTONE_OBSTACLES,
    MEDIA_DIR,
    Q_HOME,
    VEL_LIMITS,
    get_ee_pos,
    load_mujoco_model,
    load_pinocchio_model,
)
from rrt_planner import RRTStarPlanner
from trajectory_executor import execute_trajectory
from trajectory_smoother import parameterize_topp_ra, shortcut_path

# ── Video settings ────────────────────────────────────────────────────────
VIDEO_PATH = MEDIA_DIR / "lab4_validation_real_stack.mp4"
WIDTH, HEIGHT = 1280, 720
FPS = 30

# Goal: arm reaching right-far side of the table (EE ≈ [0.50, -0.20, 0.50]).
# Found via position-only IK with seed j0≈-3.6. The direct path from Q_HOME
# to this config is blocked by the table obstacles (collision at α≈0.63-0.79).
Q_GOAL = np.array([-3.592, -1.558, 1.257, -1.526, -1.955, -0.103])

# Slower limits for smooth, readable motion in the video
VIDEO_VEL_LIMITS = 0.55 * VEL_LIMITS
VIDEO_ACC_LIMITS = 0.45 * ACC_LIMITS


# ── Scene overlay helpers ─────────────────────────────────────────────────

def _add_sphere_to_scene(
    scn: mujoco.MjvScene,
    pos: np.ndarray,
    size: float,
    rgba: tuple[float, float, float, float],
) -> None:
    """Append a visual-only sphere geom to the MuJoCo scene."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, 0.0, 0.0]),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _add_trail_to_scene(
    scn: mujoco.MjvScene,
    trail: np.ndarray,
    size: float = 0.005,
    rgba: tuple[float, float, float, float] = (0.15, 0.50, 0.95, 0.90),
) -> None:
    """Draw the EE trail as blue spheres in the rendered scene."""
    for pos in trail:
        _add_sphere_to_scene(scn, pos, size, rgba)


def _add_markers(
    scn: mujoco.MjvScene,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
) -> None:
    """Add green (start) and red (goal) marker spheres."""
    _add_sphere_to_scene(scn, start_pos, 0.018, (0.1, 0.75, 0.2, 1.0))
    _add_sphere_to_scene(scn, goal_pos, 0.018, (0.9, 0.1, 0.1, 1.0))


def _path_cost(path: list[np.ndarray]) -> float:
    """Compute total L2 cost of a joint-space path."""
    return sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))


# ── Main recording function ──────────────────────────────────────────────

def record_video(output_path: Path = VIDEO_PATH) -> Path:
    """Run the full Lab 4 validation pipeline and record to MP4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Lab 4 Validation — Native MuJoCo Recording")
    print("=" * 72)

    # ── 1. Verify scenario ────────────────────────────────────────────────
    print("\n  [1/5] Building collision checker and verifying scenario...")
    cc = CollisionChecker(obstacle_specs=list(CAPSTONE_OBSTACLES))

    assert cc.is_collision_free(Q_HOME), "Q_HOME in collision!"
    assert cc.is_collision_free(Q_GOAL), "Q_GOAL in collision!"
    assert not cc.is_path_free(Q_HOME, Q_GOAL), "Direct path must be blocked!"

    pin_model, pin_data, ee_fid = load_pinocchio_model()
    ee_start = get_ee_pos(pin_model, pin_data, ee_fid, Q_HOME)
    ee_goal = get_ee_pos(pin_model, pin_data, ee_fid, Q_GOAL)
    print(f"  Start EE: [{ee_start[0]:.3f}, {ee_start[1]:.3f}, {ee_start[2]:.3f}]")
    print(f"  Goal EE:  [{ee_goal[0]:.3f}, {ee_goal[1]:.3f}, {ee_goal[2]:.3f}]")
    print(f"  Direct path blocked: True")

    # ── 2. Plan collision-free path ───────────────────────────────────────
    print("\n  [2/5] Running RRT planner...")
    planner = RRTStarPlanner(cc, step_size=0.3, goal_bias=0.15)
    raw_path = planner.plan(Q_HOME, Q_GOAL, max_iter=8000, rrt_star=False, seed=42)
    if raw_path is None:
        raise RuntimeError("RRT failed to find a path.")

    short_path = shortcut_path(raw_path, cc, max_iter=300, seed=42)
    print(f"  Raw: {len(raw_path)} wps (cost {_path_cost(raw_path):.3f})")
    print(f"  Short: {len(short_path)} wps (cost {_path_cost(short_path):.3f})")

    for i, q in enumerate(short_path):
        ee = get_ee_pos(pin_model, pin_data, ee_fid, q)
        print(f"    wp {i}: EE=[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")

    # ── 3. Time-parameterize and execute ──────────────────────────────────
    print("\n  [3/5] TOPP-RA + MuJoCo execution...")
    times, pos, vel, _ = parameterize_topp_ra(short_path, VIDEO_VEL_LIMITS, VIDEO_ACC_LIMITS)
    result = execute_trajectory(times, pos, vel, obstacle_specs=CAPSTONE_OBSTACLES)

    rms = float(np.sqrt(np.mean((result["q_actual"] - result["q_desired"]) ** 2)))
    final_err = float(np.linalg.norm(result["q_actual"][-1] - Q_GOAL))
    print(f"  Trajectory duration: {times[-1]:.3f} s")
    print(f"  RMS tracking error:  {rms:.4f} rad")
    print(f"  Final position error: {final_err:.4f} rad")

    # ── 4. Render with mujoco.Renderer ────────────────────────────────────
    print(f"\n  [4/5] Rendering {WIDTH}x{HEIGHT} @ {FPS} fps...")
    mj_model, mj_data = load_mujoco_model(obstacle_specs=CAPSTONE_OBSTACLES)
    renderer = mujoco.Renderer(mj_model, HEIGHT, WIDTH)

    cam = mujoco.MjvCamera()
    cam.lookat = np.array([0.30, 0.0, 0.40])
    cam.distance = 1.65
    cam.elevation = -25.0

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    # Camera orbit: sweep 120° so obstacles are visible from all sides
    AZ_START = 200.0
    AZ_END = 320.0

    # Sample sim data at video frame rate
    sim_dt = 0.001
    stride = max(1, int(round(1.0 / (FPS * sim_dt))))
    num_steps = len(result["time"])
    frame_indices = np.arange(0, num_steps, stride, dtype=int)
    if frame_indices[-1] != num_steps - 1:
        frame_indices = np.append(frame_indices, num_steps - 1)

    HOLD_START = FPS * 2   # 2s hold at start
    HOLD_END = FPS * 3     # 3s hold at end
    total_motion = len(frame_indices)
    frames: list[np.ndarray] = []

    # -- Start hold --
    print("    Start hold (2s)...")
    mj_data.qpos[:6] = result["q_actual"][0]
    mj_data.qvel[:6] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    cam.azimuth = AZ_START
    renderer.update_scene(mj_data, camera=cam, scene_option=opt)
    _add_markers(renderer.scene, ee_start, ee_goal)
    start_frame = renderer.render().copy()
    frames.extend([start_frame] * HOLD_START)

    # -- Motion --
    print("    Motion sequence...")
    for frame_i, idx in enumerate(frame_indices):
        mj_data.qpos[:6] = result["q_actual"][idx]
        mj_data.qvel[:6] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        t_frac = frame_i / max(total_motion - 1, 1)
        cam.azimuth = AZ_START + (AZ_END - AZ_START) * t_frac

        renderer.update_scene(mj_data, camera=cam, scene_option=opt)

        # EE trail (subsample to stay within maxgeom)
        trail_pts = result["ee_pos"][: idx + 1 : max(1, stride * 2)]
        _add_trail_to_scene(renderer.scene, trail_pts)

        # Current EE marker
        _add_sphere_to_scene(
            renderer.scene, result["ee_pos"][idx], 0.012, (0.9, 0.1, 0.1, 1.0)
        )
        _add_markers(renderer.scene, ee_start, ee_goal)

        frames.append(renderer.render().copy())

        if frame_i % 50 == 0:
            pct = 100.0 * frame_i / total_motion
            print(f"\r      {pct:5.1f}%  t={result['time'][idx]:.2f}s", end="", flush=True)

    print(f"\r      100.0%  t={result['time'][-1]:.2f}s  ({total_motion} frames)")

    # -- End hold --
    print("    End hold (3s)...")
    mj_data.qpos[:6] = result["q_actual"][-1]
    mj_data.qvel[:6] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    cam.azimuth = AZ_END
    renderer.update_scene(mj_data, camera=cam, scene_option=opt)
    trail_pts = result["ee_pos"][:: max(1, stride * 2)]
    _add_trail_to_scene(renderer.scene, trail_pts)
    _add_sphere_to_scene(renderer.scene, result["ee_pos"][-1], 0.012, (0.9, 0.1, 0.1, 1.0))
    _add_markers(renderer.scene, ee_start, ee_goal)
    end_frame = renderer.render().copy()
    frames.extend([end_frame] * HOLD_END)

    renderer.close()

    # ── 5. Write video ────────────────────────────────────────────────────
    print(f"\n  [5/5] Writing {len(frames)} frames ({len(frames)/FPS:.1f}s)...")
    iio.imwrite(str(output_path), np.stack(frames), fps=FPS, codec="libx264")

    print(f"\n  Video saved: {output_path}")
    print(f"  Duration: {len(frames)/FPS:.1f} s")
    print(f"  Metrics: RMS={rms:.4f} rad, final_err={final_err:.4f} rad")
    print(f"  Path: {len(raw_path)} raw -> {len(short_path)} short waypoints")
    print("\n" + "=" * 72)
    print("Lab 4 Validation — COMPLETE")
    print("=" * 72)

    return output_path


def main() -> None:
    """Entry point."""
    record_video()


if __name__ == "__main__":
    main()
