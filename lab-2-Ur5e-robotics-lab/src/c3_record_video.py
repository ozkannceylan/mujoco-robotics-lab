"""Record a video of the UR5e cube-drawing demo using MuJoCo offscreen rendering.

Produces an MP4 file without needing a display (headless). Reuses all
trajectory generation and control logic from c3_draw_cube.py.

Usage:
    python3 lab-2-Ur5e-robotics-lab/src/c3_record_video.py
    # → lab-2-Ur5e-robotics-lab/media/c3_draw_cube.mp4
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import imageio.v3 as iio
import mujoco
import numpy as np
import pinocchio as pin

from ur5e_common import MJCF_SCENE_PATH, MEDIA_DIR, NUM_JOINTS, Q_HOME, load_pinocchio_model
from c3_draw_cube import (
    CUBE_CENTER,
    CUBE_SIDE,
    EE_MARKER_COLOR,
    KD,
    KP,
    SEGMENT_DURATION,
    TRAIL_COLOR,
    TRAIL_DECIMATE,
    _add_sphere,
    _draw_cube_wireframe,
    _draw_trail,
    build_cube_trajectory,
    cube_drawing_path,
    cube_vertices,
    solve_cube_ik,
)

# ── Video settings ────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
FPS = 30
OUT_PATH = MEDIA_DIR / "c3_draw_cube.mp4"


def main() -> None:
    print("=" * 72)
    print("C3: Record Cube-Drawing Video (Offscreen)")
    print("=" * 72)

    # Load models
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    # Cube geometry
    verts = cube_vertices(CUBE_CENTER, CUBE_SIDE)
    path = cube_drawing_path()
    n_edges = len(path) - 1

    # Solve IK
    print("\n  Solving IK for cube path...")
    joint_wps = solve_cube_ik(verts, path, pin_model, pin_data, ee_fid)
    print(f"  Solved {len(joint_wps)} waypoints")

    # Build trajectory
    dt = 0.002
    traj = build_cube_trajectory(joint_wps, dt)
    total_time = traj[-1].t
    steps_per_frame = max(1, round(1.0 / (FPS * dt)))

    print(f"\n  Recording {total_time:.1f}s → {OUT_PATH}")
    print(f"  Resolution: {WIDTH}x{HEIGHT} @ {FPS} fps")
    print(f"  Sim steps/frame: {steps_per_frame} (dt={dt})")

    # MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(MJCF_SCENE_PATH))
    data = mujoco.MjData(model)

    # Initialise at first trajectory waypoint
    data.qpos[:NUM_JOINTS] = traj[0].q
    data.qvel[:NUM_JOINTS] = 0.0
    data.ctrl[:NUM_JOINTS] = traj[0].q
    mujoco.mj_forward(model, data)

    # Settle at start position (3 s) so PD servo reaches steady state
    print("  Settling at start position (3 s)...")
    for _ in range(1500):
        g = data.qfrc_bias[:NUM_JOINTS].copy()
        data.ctrl[:NUM_JOINTS] = traj[0].q + g / KP
        mujoco.mj_step(model, data)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    # Renderer
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

    # Camera — angled view to see the 3D cube
    cam = mujoco.MjvCamera()
    cam.lookat[:] = CUBE_CENTER
    cam.distance = 1.2
    cam.elevation = -30.0
    cam.azimuth = 135.0

    # Simulation + rendering loop
    trail: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    cart_errors: list[float] = []
    max_torque = 0.0

    for step in range(len(traj)):
        tp = traj[step]

        # Position control with gravity compensation + velocity feedforward
        g = data.qfrc_bias[:NUM_JOINTS].copy()
        data.ctrl[:NUM_JOINTS] = tp.q + g / KP + KD * tp.qd / KP
        mujoco.mj_step(model, data)

        # Metrics
        ee = data.site_xpos[ee_site_id].copy()
        pin.forwardKinematics(pin_model, pin_data, tp.q)
        pin.updateFramePlacements(pin_model, pin_data)
        ee_des = pin_data.oMf[ee_fid].translation.copy()
        err = np.linalg.norm(ee_des - ee)
        cart_errors.append(err)
        act_force = data.qfrc_actuator[:NUM_JOINTS]
        max_torque = max(max_torque, float(np.max(np.abs(act_force))))

        if step % TRAIL_DECIMATE == 0:
            trail.append(ee.copy())

        # Render frame
        if step % steps_per_frame == 0:
            renderer.update_scene(data, cam)

            # Add overlay geoms
            _draw_cube_wireframe(renderer.scene, verts)
            _draw_trail(renderer.scene, trail)
            _add_sphere(renderer.scene, ee, 0.006, EE_MARKER_COLOR)

            frame = renderer.render()
            frames.append(frame.copy())

            pct = 100.0 * step / len(traj)
            print(f"\r  {pct:5.1f}%  t={tp.t:.2f}s  frames={len(frames)}", end="", flush=True)

    renderer.close()

    # Write video
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(OUT_PATH), np.stack(frames), fps=FPS, codec="libx264")

    rms = math.sqrt(sum(e * e for e in cart_errors) / len(cart_errors))
    print(f"\n\n  Video saved: {OUT_PATH}  ({len(frames)} frames)")
    print(f"  RMS error : {rms * 1000:.3f} mm")
    print(f"  Max error : {max(cart_errors) * 1000:.3f} mm")
    print(f"  Max torque: {max_torque:.2f} N·m")

    print("\n" + "=" * 72)
    print("C3: Video Recording — COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
