#!/usr/bin/env python3
"""M1 — Standing with Joint PD + Gravity Compensation.

The Menagerie G1 uses position servos: tau = Kp*(ctrl - qpos) - Kd*qvel.
To add gravity compensation, we feedforward the gravity/Coriolis bias:

    ctrl = q_ref + qfrc_bias / Kp

This yields: tau = Kp*(q_ref - qpos) + qfrc_bias - Kd*qvel
Which is PD control + gravity compensation.

Protocol:
  - Reset to "stand" keyframe
  - Run 10s with gravity-compensated PD
  - Apply 5N lateral push at t=3s for 0.2s
  - Record pelvis height, video

Gate:
  - Robot stands 10s without falling (pelvis height within 5cm of initial)
  - Recovers from 5N push
  - Video saved to media/m1_standing.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

from lab7_common import (
    CTRL_LEFT_ARM,
    CTRL_RIGHT_ARM,
    CTRL_STAND,
    ARM_NEUTRAL_LEFT,
    ARM_NEUTRAL_RIGHT,
    MEDIA_DIR,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    load_g1_mujoco,
)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

SIM_DURATION: float = 10.0      # seconds
PUSH_TIME: float = 3.0          # when to push [s]
PUSH_DURATION: float = 0.2      # push duration [s]
PUSH_FORCE_Y: float = 5.0       # lateral push force [N]
KP: float = 500.0               # position actuator gain (from Menagerie)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_mp4(frames: list[np.ndarray], path: Path, fps: int = RENDER_FPS) -> None:
    """Save frames as H.264 MP4."""
    import imageio
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  Saved: {path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    m, d = load_g1_mujoco()
    dt = m.opt.timestep
    n_steps = int(SIM_DURATION / dt)
    render_every = max(1, int(1.0 / (RENDER_FPS * dt)))

    # Get initial pelvis height after settling
    pelvis_id = m.body("pelvis").id
    torso_id = m.body("torso_link").id
    pelvis_z_init = d.qpos[2]

    # Reference joint positions from the stand keyframe (all actuated joints)
    q_ref = CTRL_STAND.copy()

    print(f"  Sim: {SIM_DURATION}s, dt={dt}, steps={n_steps}")
    print(f"  Push: {PUSH_FORCE_Y}N lateral at t={PUSH_TIME}s for {PUSH_DURATION}s")
    print(f"  Pelvis Z init: {pelvis_z_init:.4f}m")

    # Logging arrays
    times: list[float] = []
    pelvis_z_log: list[float] = []
    com_z_log: list[float] = []
    frames: list[np.ndarray] = []

    # Renderer
    renderer = mujoco.Renderer(m, RENDER_HEIGHT, RENDER_WIDTH)
    cam = mujoco.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -15.0
    cam.azimuth = 140.0
    cam.lookat[:] = [0.0, 0.0, 0.8]

    for step in range(n_steps):
        t = step * dt

        # --- Gravity-compensated PD control ---
        # For each actuator i (dof = 6+i):
        #   ctrl[i] = q_ref[i] + qfrc_bias[6+i] / Kp
        bias = d.qfrc_bias[6:35]  # 29 actuated DOFs
        d.ctrl[:] = q_ref + bias / KP

        # Lock arms at neutral (overwrite arm ctrl)
        d.ctrl[CTRL_LEFT_ARM] = ARM_NEUTRAL_LEFT + bias[15:22] / KP
        d.ctrl[CTRL_RIGHT_ARM] = ARM_NEUTRAL_RIGHT + bias[22:29] / KP

        # --- Apply lateral push ---
        if PUSH_TIME <= t < PUSH_TIME + PUSH_DURATION:
            d.xfrc_applied[torso_id, 1] = PUSH_FORCE_Y  # lateral Y force
        else:
            d.xfrc_applied[torso_id, :] = 0.0

        mujoco.mj_step(m, d)

        # Log
        times.append(t)
        pelvis_z_log.append(d.qpos[2])
        com_z_log.append(d.subtree_com[0, 2])

        # Render
        if step % render_every == 0:
            renderer.update_scene(d, cam)
            frames.append(renderer.render().copy())

    renderer.close()

    # Save video
    save_mp4(frames, MEDIA_DIR / "m1_standing.mp4")

    # --- Analysis ---
    pelvis_z = np.array(pelvis_z_log)
    times_arr = np.array(times)

    # Pelvis height stats
    z_min = pelvis_z.min()
    z_max = pelvis_z.max()
    z_final = pelvis_z[-1]
    z_range = z_max - z_min
    z_deviation = abs(z_final - pelvis_z_init)

    # Find max deviation during push
    push_mask = (times_arr >= PUSH_TIME) & (times_arr < PUSH_TIME + 2.0)
    z_during_push = pelvis_z[push_mask]
    z_push_deviation = abs(z_during_push - pelvis_z_init).max() if len(z_during_push) > 0 else 0.0

    # Recovery: check pelvis height 2s after push
    post_push_mask = times_arr >= PUSH_TIME + 2.0
    if post_push_mask.any():
        z_post_push = pelvis_z[post_push_mask]
        z_recovered = abs(z_post_push.mean() - pelvis_z_init) < 0.05
    else:
        z_recovered = False

    # Standing check: pelvis stays within 5cm
    within_5cm = (abs(pelvis_z - pelvis_z_init) < 0.05).all()

    print(f"\n{'=' * 80}")
    print("  M1 GATE RESULTS")
    print(f"{'=' * 80}")
    print(f"  {'Criterion':<45} {'Value':<20} {'Status'}")
    print(f"{'-' * 80}")
    print(f"  {'Pelvis Z initial':<45} {pelvis_z_init:<20.4f} {'—'}")
    print(f"  {'Pelvis Z final':<45} {z_final:<20.4f} {'—'}")
    print(f"  {'Pelvis Z range (max-min)':<45} {z_range:<20.4f} {'—'}")
    print(f"  {'Pelvis Z max deviation from initial':<45} {abs(pelvis_z - pelvis_z_init).max():<20.4f} {'—'}")

    stands_pass = "PASS" if within_5cm else "FAIL"
    print(f"  {'Stands 10s (height within 5cm)':<45} {str(within_5cm):<20} {stands_pass}")

    push_pass = "PASS" if z_push_deviation < 0.05 else f"dev={z_push_deviation:.3f}m"
    print(f"  {'Push deviation':<45} {z_push_deviation:<20.4f} {push_pass}")

    recovery_pass = "PASS" if z_recovered else "FAIL"
    print(f"  {'Recovers from 5N push':<45} {str(z_recovered):<20} {recovery_pass}")
    print(f"  {'Video saved':<45} {'media/m1_standing.mp4':<20} {'PASS'}")
    print(f"{'=' * 80}")

    # Save a plot of pelvis height over time
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(times_arr, pelvis_z, "b-", linewidth=1.0, label="Pelvis Z")
        ax.axhline(pelvis_z_init, color="gray", linestyle="--", alpha=0.5, label=f"Initial ({pelvis_z_init:.3f}m)")
        ax.axhline(pelvis_z_init - 0.05, color="r", linestyle=":", alpha=0.5, label="±5cm threshold")
        ax.axhline(pelvis_z_init + 0.05, color="r", linestyle=":", alpha=0.5)
        ax.axvspan(PUSH_TIME, PUSH_TIME + PUSH_DURATION, alpha=0.2, color="orange", label=f"{PUSH_FORCE_Y}N push")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pelvis Z [m]")
        ax.set_title("M1: Standing Balance — Pelvis Height")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, SIM_DURATION)
        fig.tight_layout()
        plot_path = MEDIA_DIR / "m1_pelvis_height.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {plot_path}")
    except Exception as e:
        print(f"  Plot failed: {e}")


if __name__ == "__main__":
    main()
