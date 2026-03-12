#!/usr/bin/env python3
"""Record a video of the square-drawing demo using MuJoCo offscreen rendering.

Produces an MP4 file without needing a display (headless).  Reuses all
trajectory generation and control logic from c1_draw_square.py.

Usage:
    python3 src/c1_record_video.py          # → media/c1_draw_square.mp4
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

import imageio.v3 as iio
import mujoco
import numpy as np

from c1_draw_square import (
    EE_MARKER_COLOR,
    KD,
    KP,
    MODEL_PATH,
    PATH_COLOR,
    SEGMENT_DURATION,
    SQUARE_CENTER,
    SQUARE_SIDE,
    TORQUE_LIMIT,
    TRAIL_COLOR,
    TRAIL_DECIMATE,
    TrajPoint,
    _add_sphere,
    _draw_desired_square,
    _draw_trail,
    _ee_pos,
    build_trajectory,
)

# ── Video settings ────────────────────────────────────────────────────────
WIDTH, HEIGHT = 640, 480
FPS = 30                          # output video framerate
SIM_STEPS_PER_FRAME = None        # computed from dt and FPS
OUT_DIR = Path(__file__).resolve().parent.parent / "media"
OUT_PATH = OUT_DIR / "c1_draw_square.mp4"


def main() -> None:
    # ── Load model ────────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0

    # ── Build trajectory ──────────────────────────────────────────────────
    traj = build_trajectory(dt)
    total_time = traj[-1].t
    steps_per_frame = max(1, round(1.0 / (FPS * dt)))

    print(f"Recording {total_time:.1f}s → {OUT_PATH}")
    print(f"  Resolution : {WIDTH}×{HEIGHT} @ {FPS} fps")
    print(f"  Sim steps/frame: {steps_per_frame}  (dt={dt})")

    # ── Initialise ────────────────────────────────────────────────────────
    data.qpos[0] = traj[0].q1
    data.qpos[1] = traj[0].q2
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # ── Renderer ───────────────────────────────────────────────────────────
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    scene_option = mujoco.MjvOption()

    # Camera
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [SQUARE_CENTER[0], SQUARE_CENTER[1], 0.0]
    cam.distance = 0.85
    cam.elevation = -55.0
    cam.azimuth = 90.0

    # ── Simulation + rendering loop ───────────────────────────────────────
    trail: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    cart_errors: list[float] = []
    max_torque = 0.0
    step = 0

    while step < len(traj):
        tp = traj[step]

        # ── Computed torque control ───────────────────────────────────────
        q = data.qpos.copy()
        qd = data.qvel.copy()

        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)
        bias = data.qfrc_bias.copy()
        passive = data.qfrc_passive.copy()

        e_vec = np.array([tp.q1 - q[0], tp.q2 - q[1]])
        ed = np.array([tp.q1d - qd[0], tp.q2d - qd[1]])
        qdd_ff = np.array([tp.q1dd, tp.q2dd])

        u = qdd_ff + KP * e_vec + KD * ed
        tau = M @ u + bias - passive
        tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        # ── Metrics ───────────────────────────────────────────────────────
        ee = _ee_pos(model, data)
        err = math.hypot(tp.x - ee[0], tp.y - ee[1])
        cart_errors.append(err)
        max_torque = max(max_torque, float(np.max(np.abs(tau))))

        if step % TRAIL_DECIMATE == 0:
            trail.append(ee)

        # ── Render frame ──────────────────────────────────────────────────
        if step % steps_per_frame == 0:
            renderer.update_scene(data, cam, scene_option)

            # Add overlay geoms directly to renderer's scene
            _draw_desired_square(renderer.scene)
            _draw_trail(renderer.scene, trail)
            _add_sphere(renderer.scene, ee, 0.008, EE_MARKER_COLOR)

            frame = renderer.render()
            frames.append(frame.copy())

            # Progress
            pct = 100.0 * step / len(traj)
            print(f"\r  {pct:5.1f}%  t={tp.t:.2f}s  frames={len(frames)}", end="", flush=True)

        step += 1

    renderer.close()

    # ── Write video ───────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(OUT_PATH), np.stack(frames), fps=FPS, codec="libx264")

    rms = math.sqrt(sum(e * e for e in cart_errors) / len(cart_errors))
    print(f"\n\nVideo saved: {OUT_PATH}  ({len(frames)} frames)")
    print(f"  RMS error : {rms * 1000:.3f} mm")
    print(f"  Max error : {max(cart_errors) * 1000:.3f} mm")
    print(f"  Max torque: {max_torque:.3f} N·m")


if __name__ == "__main__":
    main()
