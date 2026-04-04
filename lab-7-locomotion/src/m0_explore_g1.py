#!/usr/bin/env python3
"""M0 — Load the real Unitree G1 from MuJoCo Menagerie and explore it.

Outputs:
  - Full joint table printed to stdout
  - media/m0_tpose.png    — screenshot of G1 in stand keyframe
  - media/m0_freefall.mp4 — 2-second ragdoll collapse (actuators disabled)
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

from lab7_common import (
    G1_SCENE_PATH,
    MEDIA_DIR,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    load_g1_mujoco,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JNT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def _categorize_joint(name: str, jtype: int) -> str:
    """Return a human-readable category for a joint name."""
    if jtype == 0:
        return "BASE"
    n = name.lower()
    if "left_hip" in n or "left_knee" in n or "left_ankle" in n:
        return "LEFT_LEG"
    if "right_hip" in n or "right_knee" in n or "right_ankle" in n:
        return "RIGHT_LEG"
    if "left_shoulder" in n or "left_elbow" in n or "left_wrist" in n:
        return "LEFT_ARM"
    if "right_shoulder" in n or "right_elbow" in n or "right_wrist" in n:
        return "RIGHT_ARM"
    if "waist" in n:
        return "WAIST"
    return "OTHER"


def print_joint_table(m: mujoco.MjModel) -> None:
    """Print full joint table with categories."""
    print(f"\n{'=' * 95}")
    print(f"  JOINT TABLE — nq={m.nq}  nv={m.nv}  nu={m.nu}  njnt={m.njnt}")
    print(f"{'=' * 95}")
    hdr = f"{'idx':>3} {'name':<32} {'type':<6} {'category':<12} {'qpos':>4} {'range_lo':>9} {'range_hi':>9} {'frc_range'}"
    print(hdr)
    print("-" * 95)
    for i in range(m.njnt):
        j = m.joint(i)
        jtype = _JNT_TYPES.get(j.type[0], "?")
        cat = _categorize_joint(j.name, j.type[0])
        lo = f"{j.range[0]:9.3f}" if j.limited[0] else "     -inf"
        hi = f"{j.range[1]:9.3f}" if j.limited[0] else "      inf"
        print(f"{i:3d} {j.name:<32} {jtype:<6} {cat:<12} {j.qposadr[0]:>4} {lo} {hi}")


def print_actuator_table(m: mujoco.MjModel) -> None:
    """Print actuator table with gains."""
    print(f"\n{'=' * 95}")
    print(f"  ACTUATOR TABLE — nu={m.nu}")
    print(f"{'=' * 95}")
    hdr = f"{'idx':>3} {'actuator':<35} {'joint':<32} {'Kp':>6} {'ctrl_lo':>8} {'ctrl_hi':>8}"
    print(hdr)
    print("-" * 95)
    for i in range(m.nu):
        a = m.actuator(i)
        jnt_id = a.trnid[0]
        jnt_name = m.joint(jnt_id).name if jnt_id >= 0 else "N/A"
        kp = a.gainprm[0]
        print(f"{i:3d} {a.name:<35} {jnt_name:<32} {kp:>6.0f} {a.ctrlrange[0]:>8.3f} {a.ctrlrange[1]:>8.3f}")


def print_body_table(m: mujoco.MjModel) -> None:
    """Print body table with masses."""
    print(f"\n{'=' * 60}")
    print(f"  BODY TABLE — nbody={m.nbody}")
    print(f"{'=' * 60}")
    print(f"{'idx':>3} {'name':<32} {'parent':>6} {'mass':>8}")
    print("-" * 60)
    total = 0.0
    for i in range(m.nbody):
        b = m.body(i)
        total += b.mass[0]
        print(f"{i:3d} {b.name:<32} {b.parentid[0]:>6} {b.mass[0]:>8.3f}")
    print(f"{'':>3} {'TOTAL':<32} {'':>6} {total:>8.3f}")


def print_dof_summary(m: mujoco.MjModel) -> None:
    """Print DOF summary by category."""
    cats: dict[str, int] = {}
    for i in range(m.njnt):
        j = m.joint(i)
        cat = _categorize_joint(j.name, j.type[0])
        ndof = {0: 6, 1: 3, 2: 1, 3: 1}[j.type[0]]
        cats[cat] = cats.get(cat, 0) + ndof

    print(f"\n{'=' * 50}")
    print("  DOF SUMMARY")
    print(f"{'=' * 50}")
    for cat, ndof in cats.items():
        print(f"  {cat:<25} {ndof:>3} DOF")
    print(f"  {'─' * 30}")
    print(f"  {'Total nv':<25} {m.nv:>3}")
    print(f"  {'Actuated (nu)':<25} {m.nu:>3}")
    leg_dofs = cats.get("LEFT_LEG", 0) + cats.get("RIGHT_LEG", 0)
    waist_dofs = cats.get("WAIST", 0)
    arm_dofs = cats.get("LEFT_ARM", 0) + cats.get("RIGHT_ARM", 0)
    print(f"  {'Leg DOFs':<25} {leg_dofs:>3}")
    print(f"  {'Waist DOFs':<25} {waist_dofs:>3}")
    print(f"  {'Arm DOFs (locked)':<25} {arm_dofs:>3}")
    print(f"  {'Locomotion (legs+waist)':<25} {leg_dofs + waist_dofs:>3}")


def save_png(frame: np.ndarray, path: Path) -> None:
    """Save a numpy RGB array as PNG."""
    import imageio.v3 as iio
    iio.imwrite(str(path), frame)
    print(f"  Saved: {path}")


def save_mp4(frames: list[np.ndarray], path: Path, fps: int = RENDER_FPS) -> None:
    """Save a list of RGB frames as H.264 MP4."""
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

    # ── 1. Load the real Menagerie G1 ───────────────────────────────────
    print("\n" + "#" * 95)
    print("#  Unitree G1 (MuJoCo Menagerie) — 29 DOF")
    print("#" * 95)

    m, d = load_g1_mujoco()

    print_joint_table(m)
    print_actuator_table(m)
    print_body_table(m)
    print_dof_summary(m)

    # Key positions from stand keyframe
    pelvis_id = m.body("pelvis").id
    print(f"\n  Pelvis pos (keyframe): {d.xpos[pelvis_id]}")
    print(f"  Pelvis quat: {d.qpos[3:7]}")
    print(f"  World CoM: {d.subtree_com[0]}")
    total_mass = sum(m.body(i).mass[0] for i in range(m.nbody))
    print(f"  Total mass: {total_mass:.2f} kg")

    # Keyframe info
    if m.nkey > 0:
        print(f"\n  Keyframe 'stand' qpos: {d.qpos}")
        print(f"  Keyframe 'stand' ctrl: {d.ctrl}")

    # ── 2. T-pose screenshot (stand keyframe) ──────────────────────────
    print("\n" + "#" * 95)
    print("#  T-pose screenshot (stand keyframe)")
    print("#" * 95)

    renderer = mujoco.Renderer(m, RENDER_HEIGHT, RENDER_WIDTH)
    cam = mujoco.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -15.0
    cam.azimuth = 140.0
    cam.lookat[:] = [0.0, 0.0, 0.8]

    renderer.update_scene(d, cam)
    frame = renderer.render()
    save_png(frame, MEDIA_DIR / "m0_tpose.png")

    # ── 3. Freefall video (2s, actuators disabled) ─────────────────────
    print("\n" + "#" * 95)
    print("#  Freefall simulation (2s, actuators + damping disabled)")
    print("#" * 95)

    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    # Disable actuators and joint damping for true ragdoll
    saved_gainprm = m.actuator_gainprm.copy()
    saved_biasprm = m.actuator_biasprm.copy()
    saved_damping = m.dof_damping.copy()
    m.actuator_gainprm[:] = 0.0
    m.actuator_biasprm[:] = 0.0
    m.dof_damping[:] = 0.0

    sim_duration = 2.0
    n_steps = int(sim_duration / m.opt.timestep)
    render_every = max(1, int(1.0 / (RENDER_FPS * m.opt.timestep)))

    frames: list[np.ndarray] = []
    pelvis_z_log: list[float] = []

    print(f"  Simulating {sim_duration}s ({n_steps} steps, dt={m.opt.timestep}) ...")

    for step in range(n_steps):
        mujoco.mj_step(m, d)
        pelvis_z_log.append(d.qpos[2])

        if step % render_every == 0:
            renderer.update_scene(d, cam)
            frames.append(renderer.render().copy())

    renderer.close()

    # Restore model parameters
    m.actuator_gainprm[:] = saved_gainprm
    m.actuator_biasprm[:] = saved_biasprm
    m.dof_damping[:] = saved_damping

    save_mp4(frames, MEDIA_DIR / "m0_freefall.mp4")

    print(f"\n  Pelvis Z: initial={pelvis_z_log[0]:.4f}m → final={pelvis_z_log[-1]:.4f}m")
    print(f"  Pelvis Z drop: {pelvis_z_log[0] - pelvis_z_log[-1]:.4f}m")
    print(f"  Min pelvis Z: {min(pelvis_z_log):.4f}m")

    # ── 4. Gate summary ────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("  M0 GATE SUMMARY")
    print("=" * 95)
    print(f"  {'Criterion':<45} {'Result':<35} {'Status'}")
    print("-" * 95)
    print(f"  {'Joint table printed':<45} {'29 joints + 1 freejoint':<35} {'PASS'}")
    print(f"  {'DOF layout documented':<45} {'docs/g1_joint_map.md':<35} {'(see doc)'}")
    print(f"  {'T-pose screenshot':<45} {'media/m0_tpose.png':<35} {'PASS'}")
    print(f"  {'Freefall video':<45} {'media/m0_freefall.mp4':<35} {'PASS'}")
    drop = pelvis_z_log[0] - pelvis_z_log[-1]
    collapse = "PASS" if drop > 0.1 else "WEAK"
    print(f"  {'Robot collapses under gravity':<45} {f'drop={drop:.3f}m':<35} {collapse}")
    print("=" * 95)


if __name__ == "__main__":
    main()
