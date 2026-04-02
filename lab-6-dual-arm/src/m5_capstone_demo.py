"""M5 — Capstone demo: full M0-M4 pipeline with state-overlay video and trajectory plot.

Loads the dual-arm scene, validates joint counts, runs the 6-state bimanual
state machine (APPROACH → CLOSE → GRASP → LIFT → CARRY → PLACE), records
high-quality video with text overlay showing the current state name, and
generates a summary trajectory plot.

Outputs:
  - media/m5_capstone.mp4
  - media/m5_trajectory.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from bimanual_state_machine import BimanualStateMachine, State
from dual_arm_model import DualArmModel
from joint_pd_controller import DualArmJointPD
from lab6_common import (
    DT,
    MEDIA_DIR,
    NUM_JOINTS_PER_ARM,
    NUM_JOINTS_TOTAL,
    SCENE_DUAL_PATH,
)

# Video settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
FRAME_SKIP = int(1.0 / (DT * VIDEO_FPS))

# Controller gains
KP = 300.0
KD = 40.0

# State display colors (BGR-ish for readability on dark background)
STATE_COLORS = {
    State.APPROACH: (100, 200, 255),
    State.CLOSE:    (255, 200, 100),
    State.GRASP:    (100, 255, 100),
    State.LIFT:     (200, 150, 255),
    State.CARRY:    (255, 150, 200),
    State.PLACE:    (200, 255, 150),
    State.DONE:     (200, 200, 200),
}


def burn_text(frame: np.ndarray, text: str, color: tuple[int, ...]) -> np.ndarray:
    """Burn text overlay onto a video frame using simple pixel drawing.

    Uses a minimal 5x7 bitmap font rendered at 4x scale for visibility.
    """
    out = frame.copy()
    scale = 4
    x0, y0 = 40, 40

    # Minimal bitmap font — uppercase + digits + space/colon
    glyphs = _get_glyphs()

    cx = x0
    for ch in text.upper():
        glyph = glyphs.get(ch)
        if glyph is None:
            cx += 4 * scale
            continue
        for row_idx, row in enumerate(glyph):
            for col_idx, bit in enumerate(row):
                if bit:
                    py = y0 + row_idx * scale
                    px = cx + col_idx * scale
                    out[py:py + scale, px:px + scale] = color
        cx += (len(glyph[0]) + 1) * scale

    return out


def _get_glyphs() -> dict:
    """Return minimal 5x7 bitmap glyphs for uppercase letters and digits."""
    # Each glyph is a list of rows (top to bottom), each row is a list of 0/1
    g = {}
    g["A"] = [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,0,0,1]]
    g["B"] = [[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0]]
    g["C"] = [[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,1,1]]
    g["D"] = [[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,0]]
    g["E"] = [[1,1,1,1],[1,0,0,0],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]]
    g["F"] = [[1,1,1,1],[1,0,0,0],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]
    g["G"] = [[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,1,1],[1,0,0,1],[1,0,0,1],[0,1,1,1]]
    g["H"] = [[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,0,0,1]]
    g["I"] = [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,1,1]]
    g["J"] = [[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]]
    g["K"] = [[1,0,0,1],[1,0,1,0],[1,1,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1],[1,0,0,1]]
    g["L"] = [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]]
    g["M"] = [[1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1]]
    g["N"] = [[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1]]
    g["O"] = [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]]
    g["P"] = [[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]
    g["Q"] = [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,1,1],[1,0,0,1],[0,1,1,1]]
    g["R"] = [[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0],[1,0,1,0],[1,0,0,1],[1,0,0,1]]
    g["S"] = [[0,1,1,1],[1,0,0,0],[1,0,0,0],[0,1,1,0],[0,0,0,1],[0,0,0,1],[1,1,1,0]]
    g["T"] = [[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]]
    g["U"] = [[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]]
    g["V"] = [[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0]]
    g["W"] = [[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,1,0,1],[1,1,0,1,1],[1,0,0,0,1]]
    g["X"] = [[1,0,0,1],[1,0,0,1],[0,1,1,0],[0,1,1,0],[0,1,1,0],[1,0,0,1],[1,0,0,1]]
    g["Y"] = [[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]]
    g["Z"] = [[1,1,1,1],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]]
    g[" "] = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    g[":"] = [[0],[0],[1],[0],[0],[1],[0]]
    g["."] = [[0],[0],[0],[0],[0],[0],[1]]
    g["0"] = [[0,1,1,0],[1,0,0,1],[1,0,1,1],[1,1,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]]
    g["1"] = [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,1,1]]
    g["2"] = [[0,1,1,0],[1,0,0,1],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],[1,1,1,1]]
    g["3"] = [[0,1,1,0],[1,0,0,1],[0,0,0,1],[0,1,1,0],[0,0,0,1],[1,0,0,1],[0,1,1,0]]
    g["4"] = [[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]]
    g["5"] = [[1,1,1,1],[1,0,0,0],[1,1,1,0],[0,0,0,1],[0,0,0,1],[1,0,0,1],[0,1,1,0]]
    g["6"] = [[0,1,1,0],[1,0,0,0],[1,0,0,0],[1,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]]
    g["7"] = [[1,1,1,1],[0,0,0,1],[0,0,1,0],[0,0,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]
    g["8"] = [[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]]
    g["9"] = [[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,1],[0,0,0,1],[0,0,0,1],[0,1,1,0]]
    return g


def plot_trajectory(sm: BimanualStateMachine, save_path: Path) -> None:
    """Plot box xyz trajectory with state boundaries."""
    times = np.array(sm.time_log)
    traj = np.array(sm.box_trajectory)
    states = sm.state_log

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#08111f")

    labels = ["x", "y", "z"]
    colors = ["#4fc3f7", "#81c784", "#ffb74d"]

    # State transitions
    transitions: list[tuple[float, str]] = []
    prev = states[0]
    for i, s in enumerate(states):
        if s != prev:
            transitions.append((times[i], s.name))
            prev = s

    for ax, label, color, dim in zip(axes, labels, colors, range(3)):
        ax.set_facecolor("#0d1b2a")
        ax.plot(times, traj[:, dim], color=color, linewidth=1.5)
        ax.set_ylabel(f"{label} (m)", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, color="white")
        for spine in ax.spines.values():
            spine.set_color("#1b2838")

        for t_tr, s_name in transitions:
            ax.axvline(t_tr, color="#e57373", alpha=0.5, linestyle="--", linewidth=0.8)
            if dim == 0:
                ax.text(t_tr, ax.get_ylim()[1], s_name,
                        color="#e57373", fontsize=7, rotation=45,
                        ha="left", va="bottom")

    # Initial position reference
    if sm.box_init_pos is not None:
        for ax, dim in zip(axes, range(3)):
            ax.axhline(sm.box_init_pos[dim], color="white", alpha=0.3,
                       linestyle=":", linewidth=0.8)

    axes[-1].set_xlabel("Time (s)", color="white", fontsize=11)
    axes[0].set_title("Lab 6 Capstone: Dual-Arm Cooperative Manipulation",
                      color="white", fontsize=14, pad=10)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run M5 capstone demo."""
    print("=" * 70)
    print("M5: Capstone Demo — Full Dual-Arm Cooperative Manipulation")
    print("=" * 70)

    # ---- Step 1: Load and validate scene ----
    print("\n  Loading scene...")
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_DUAL_PATH))
    mj_data = mujoco.MjData(mj_model)

    n_hinge = sum(1 for i in range(mj_model.njnt)
                  if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE)
    n_act = mj_model.nu
    print(f"  Joints: {n_hinge} hinge (expected {NUM_JOINTS_TOTAL})")
    print(f"  Actuators: {n_act} (expected {NUM_JOINTS_TOTAL})")
    assert n_hinge == NUM_JOINTS_TOTAL, f"Expected {NUM_JOINTS_TOTAL} hinge joints"
    assert n_act == NUM_JOINTS_TOTAL, f"Expected {NUM_JOINTS_TOTAL} actuators"

    # ---- Step 2: Create models and controller ----
    dual = DualArmModel()
    controller = DualArmJointPD(kp=KP, kd=KD)
    renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

    # ---- Step 3: Run state machine with rendering ----
    sm = BimanualStateMachine(
        mj_model, mj_data, dual, controller,
        renderer=renderer, frame_skip=FRAME_SKIP,
    )

    wall_start = time.time()
    success = sm.run()
    wall_elapsed = time.time() - wall_start
    renderer.close()

    print(f"\n  Pipeline {'COMPLETED' if success else 'FAILED'}")
    print(f"  Sim time: {sm.sim_time:.2f}s | Wall time: {wall_elapsed:.1f}s | Frames: {len(sm.frames)}")

    if not success:
        print("\n  *** Pipeline failed ***")
        return

    # ---- Step 4: Burn state overlay onto frames ----
    print("\n  Burning state overlay onto frames...")
    # Build frame index → state mapping
    # Each frame is captured every FRAME_SKIP steps.
    # state_log has one entry per sim step; we sample at frame times.
    overlay_frames = []
    steps_per_frame = FRAME_SKIP
    for fi, raw_frame in enumerate(sm.frames):
        step_idx = min(fi * steps_per_frame, len(sm.state_log) - 1)
        state = sm.state_log[step_idx]
        t = sm.time_log[min(step_idx, len(sm.time_log) - 1)]
        label = f"STATE: {state.name}  T:{t:.1f}S"
        color = STATE_COLORS.get(state, (200, 200, 200))
        overlay_frames.append(burn_text(raw_frame, label, color))

    # ---- Step 5: Save video ----
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    video_path = MEDIA_DIR / "m5_capstone.mp4"
    print(f"  Writing video to {video_path} ...")
    writer = imageio.get_writer(
        str(video_path), fps=VIDEO_FPS, codec="libx264",
        quality=8, macro_block_size=1,
    )
    for frame in overlay_frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Video saved: {video_path}")

    # ---- Step 6: Trajectory plot ----
    plot_path = MEDIA_DIR / "m5_trajectory.png"
    plot_trajectory(sm, plot_path)
    print(f"  Trajectory plot saved: {plot_path}")

    # ---- Gate summary ----
    print("\n" + "=" * 70)
    print("M5 CAPSTONE GATE")
    print("=" * 70)
    print(f"  Video:     {'EXISTS' if video_path.exists() else 'MISSING'}")
    print(f"  Plot:      {'EXISTS' if plot_path.exists() else 'MISSING'}")
    print(f"  Pipeline:  {'PASS' if success else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
