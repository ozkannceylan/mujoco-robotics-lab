#!/usr/bin/env python3
"""M3b — Foot FK Cross-Validation: Pinocchio vs MuJoCo.

Static validation only — no simulation loop.

For 10 random leg joint configurations (within joint limits):
  1. Set joints in MuJoCo → mj_forward → read foot site positions + CoM
  2. Convert qpos to Pinocchio convention → FK → read foot frame positions + CoM
  3. Compare and report errors

Also prints the exact quaternion mapping to document the convention.

Gate criteria:
  - Foot position error < 2mm for all 20 comparisons (10 configs x 2 feet)
  - CoM error < 5mm for all 10 configs
  - Quaternion mapping documented and verified
  - Save results to media/m3b_fk_validation.txt
  - Save screenshot to media/m3b_config.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

from lab7_common import (
    MEDIA_DIR,
    PELVIS_MJCF_Z,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    load_g1_mujoco,
    load_g1_pinocchio,
    mj_qpos_to_pin,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_CONFIGS: int = 10
SEED: int = 42
FOOT_TOL: float = 0.002   # 2 mm
COM_TOL: float = 0.005    # 5 mm

# MuJoCo foot site names → Pinocchio frame names
MJ_LF_SITE: str = "left_foot"
MJ_RF_SITE: str = "right_foot"
PIN_LF_FRAME: str = "left_foot"
PIN_RF_FRAME: str = "right_foot"


# ---------------------------------------------------------------------------
# Random config generation
# ---------------------------------------------------------------------------


def generate_random_leg_configs(
    mj_model: mujoco.MjModel,
    n: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate n random leg configs within MuJoCo joint limits.

    Keeps the floating base at standing pose (pelvis at keyframe height,
    identity quaternion). Only varies leg joints (qpos indices 7-18).
    Arm joints are set to their keyframe values.

    Returns:
        List of full qpos arrays (nq=36).
    """
    # Get standing keyframe qpos as base
    d_tmp = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, d_tmp, 0)
    base_qpos = d_tmp.qpos.copy()

    # Leg joint indices and their limits
    leg_qpos_indices = list(range(7, 19))  # 12 leg joints
    configs = []

    for _ in range(n):
        q = base_qpos.copy()
        for idx in leg_qpos_indices:
            # Find which joint owns this qpos index
            for ji in range(mj_model.njnt):
                if mj_model.jnt_qposadr[ji] == idx:
                    lo, hi = mj_model.jnt_range[ji]
                    # Sample within 60% of the range to avoid extreme configs
                    mid = (lo + hi) / 2.0
                    half_range = (hi - lo) / 2.0 * 0.6
                    q[idx] = rng.uniform(mid - half_range, mid + half_range)
                    break
        configs.append(q)

    return configs


# ---------------------------------------------------------------------------
# Screenshot helper
# ---------------------------------------------------------------------------


def save_screenshot(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    path: Path,
) -> None:
    """Render a single frame and save as PNG."""
    renderer = mujoco.Renderer(mj_model, RENDER_HEIGHT, RENDER_WIDTH)
    cam = mujoco.MjvCamera()
    cam.distance = 2.5
    cam.elevation = -20.0
    cam.azimuth = 150.0
    cam.lookat[:] = [0.0, 0.0, 0.5]

    renderer.update_scene(mj_data, cam)
    img = renderer.render()
    renderer.close()

    import imageio
    imageio.imwrite(str(path), img)
    print(f"  Screenshot saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MEDIA_DIR / "m3b_fk_validation.txt"

    lines: list[str] = []
    lines.append("M3b — Foot FK Cross-Validation: Pinocchio vs MuJoCo")
    lines.append("")

    # ── Load both models ──────────────────────────────────────────────
    mj_model, mj_data = load_g1_mujoco()
    pin_model, pin_data = load_g1_pinocchio()

    lf_site_id = mj_model.site(MJ_LF_SITE).id
    rf_site_id = mj_model.site(MJ_RF_SITE).id
    lf_frame_id = pin_model.getFrameId(PIN_LF_FRAME)
    rf_frame_id = pin_model.getFrameId(PIN_RF_FRAME)

    lines.append(f"MuJoCo:   nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")
    lines.append(f"Pinocchio: nq={pin_model.nq}, nv={pin_model.nv}")
    lines.append(f"MuJoCo foot sites:  left={MJ_LF_SITE}(id={lf_site_id}), "
                 f"right={MJ_RF_SITE}(id={rf_site_id})")
    lines.append(f"Pinocchio foot frames: left={PIN_LF_FRAME}(id={lf_frame_id}), "
                 f"right={PIN_RF_FRAME}(id={rf_frame_id})")

    # ── Document quaternion mapping ───────────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("  QUATERNION CONVENTION MAPPING")
    lines.append("=" * 90)
    lines.append("")
    lines.append("  MuJoCo qpos[3:7] ordering:   (w, x, y, z)")
    lines.append("  Pinocchio q[3:7] ordering:    (x, y, z, w)")
    lines.append("")
    lines.append("  Conversion: mj_qpos_to_pin() from lab7_common.py")
    lines.append("    pin_q[3] = mj_qpos[4]  (x)")
    lines.append("    pin_q[4] = mj_qpos[5]  (y)")
    lines.append("    pin_q[5] = mj_qpos[6]  (z)")
    lines.append("    pin_q[6] = mj_qpos[3]  (w)")
    lines.append("")
    lines.append("  Base Z correction:")
    lines.append(f"    pin_q[2] = mj_qpos[2] - PELVIS_MJCF_Z ({PELVIS_MJCF_Z})")
    lines.append("")

    # Verify with standing keyframe
    mj_quat = mj_data.qpos[3:7].copy()
    pin_q_stand = mj_qpos_to_pin(mj_data.qpos.copy())
    pin_quat = pin_q_stand[3:7]

    lines.append("  Verification with standing keyframe:")
    lines.append(f"    MuJoCo qpos[3:7] (w,x,y,z): [{mj_quat[0]:.6f}, {mj_quat[1]:.6f}, "
                 f"{mj_quat[2]:.6f}, {mj_quat[3]:.6f}]")
    lines.append(f"    Pinocchio q[3:7] (x,y,z,w):  [{pin_quat[0]:.6f}, {pin_quat[1]:.6f}, "
                 f"{pin_quat[2]:.6f}, {pin_quat[3]:.6f}]")
    lines.append(f"    MuJoCo qpos[2] (base Z):     {mj_data.qpos[2]:.6f}")
    lines.append(f"    Pinocchio q[2] (base Z):      {pin_q_stand[2]:.6f}")
    lines.append(f"    Difference:                   {mj_data.qpos[2] - pin_q_stand[2]:.6f} "
                 f"(should be {PELVIS_MJCF_Z})")

    quat_w_match = (mj_quat[0] == pin_quat[3])
    quat_xyz_match = np.allclose(mj_quat[1:4], pin_quat[0:3])
    z_match = np.isclose(mj_data.qpos[2] - pin_q_stand[2], PELVIS_MJCF_Z)

    lines.append(f"    Quaternion w component match: {'YES' if quat_w_match else 'NO'}")
    lines.append(f"    Quaternion xyz match:         {'YES' if quat_xyz_match else 'NO'}")
    lines.append(f"    Base Z offset correct:        {'YES' if z_match else 'NO'}")

    # ── Generate random configs ───────────────────────────────────────
    rng = np.random.default_rng(SEED)
    configs = generate_random_leg_configs(mj_model, N_CONFIGS, rng)

    # ── Cross-validate each config ────────────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("  FOOT FK CROSS-VALIDATION (10 random leg configs)")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"  {'Cfg':<5s} {'Foot':<8s} "
                 f"{'MuJoCo (x, y, z)':<36s} "
                 f"{'Pinocchio (x, y, z)':<36s} "
                 f"{'Err [mm]':<12s} {'Status'}")
    lines.append("-" * 110)

    foot_errors: list[float] = []
    com_errors: list[float] = []
    screenshot_config_idx = 4  # save screenshot for config #4

    for ci, mj_qpos in enumerate(configs):
        # ── MuJoCo FK ─────────────────────────────────────────────
        mj_data.qpos[:] = mj_qpos
        mujoco.mj_forward(mj_model, mj_data)

        mj_lf = mj_data.site_xpos[lf_site_id].copy()
        mj_rf = mj_data.site_xpos[rf_site_id].copy()
        mj_com = mj_data.subtree_com[0].copy()

        # ── Pinocchio FK ─────────────────────────────────────────���
        pin_q = mj_qpos_to_pin(mj_qpos)
        pin.forwardKinematics(pin_model, pin_data, pin_q)
        pin.updateFramePlacements(pin_model, pin_data)

        pin_lf = pin_data.oMf[lf_frame_id].translation.copy()
        pin_rf = pin_data.oMf[rf_frame_id].translation.copy()

        pin.centerOfMass(pin_model, pin_data, pin_q)
        pin_com = pin_data.com[0].copy()

        # ── Compare feet ──────────────────────────────────────────
        for foot_name, mj_pos, pin_pos in [
            ("L_foot", mj_lf, pin_lf),
            ("R_foot", mj_rf, pin_rf),
        ]:
            err = np.linalg.norm(mj_pos - pin_pos)
            foot_errors.append(err)
            err_mm = err * 1000.0
            status = "PASS" if err < FOOT_TOL else "FAIL"

            mj_str = f"[{mj_pos[0]:+.6f}, {mj_pos[1]:+.6f}, {mj_pos[2]:+.6f}]"
            pin_str = f"[{pin_pos[0]:+.6f}, {pin_pos[1]:+.6f}, {pin_pos[2]:+.6f}]"
            lines.append(f"  {ci:<5d} {foot_name:<8s} {mj_str:<36s} {pin_str:<36s} "
                         f"{err_mm:<12.4f} {status}")

        # ── Compare CoM ──────────────────────────────────────────��
        com_err = np.linalg.norm(mj_com - pin_com)
        com_errors.append(com_err)

        # Save screenshot for one config
        if ci == screenshot_config_idx:
            save_screenshot(mj_model, mj_data, MEDIA_DIR / "m3b_config.png")

    # ── CoM comparison table ──────────────────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("  CoM CROSS-VALIDATION")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"  {'Cfg':<5s} "
                 f"{'MuJoCo CoM (x, y, z)':<36s} "
                 f"{'Pinocchio CoM (x, y, z)':<36s} "
                 f"{'Err [mm]':<12s} {'Status'}")
    lines.append("-" * 100)

    # Re-run to print CoM table (reuse configs)
    for ci, mj_qpos in enumerate(configs):
        mj_data.qpos[:] = mj_qpos
        mujoco.mj_forward(mj_model, mj_data)
        mj_com = mj_data.subtree_com[0].copy()

        pin_q = mj_qpos_to_pin(mj_qpos)
        pin.centerOfMass(pin_model, pin_data, pin_q)
        pin_com = pin_data.com[0].copy()

        err = np.linalg.norm(mj_com - pin_com)
        err_mm = err * 1000.0
        status = "PASS" if err < COM_TOL else "FAIL"

        mj_str = f"[{mj_com[0]:+.6f}, {mj_com[1]:+.6f}, {mj_com[2]:+.6f}]"
        pin_str = f"[{pin_com[0]:+.6f}, {pin_com[1]:+.6f}, {pin_com[2]:+.6f}]"
        lines.append(f"  {ci:<5d} {mj_str:<36s} {pin_str:<36s} "
                     f"{err_mm:<12.4f} {status}")

    # ── Gate summary ──────────────────────────────────────────────────
    foot_errors_mm = np.array(foot_errors) * 1000.0
    com_errors_mm = np.array(com_errors) * 1000.0

    n_foot_fail = int(np.sum(foot_errors_mm >= FOOT_TOL * 1000))
    n_com_fail = int(np.sum(com_errors_mm >= COM_TOL * 1000))

    lines.append("")
    lines.append("=" * 90)
    lines.append("  M3b GATE RESULTS")
    lines.append("=" * 90)
    lines.append(f"  {'Criterion':<50s} {'Value':<25s} {'Status'}")
    lines.append("-" * 85)
    lines.append(
        f"  {'Foot error < 2mm (20 comparisons)':<50s} "
        f"{'max=' + f'{foot_errors_mm.max():.4f}mm':<25s} "
        f"{'PASS' if n_foot_fail == 0 else 'FAIL'}"
    )
    lines.append(
        f"  {'CoM error < 5mm (10 configs)':<50s} "
        f"{'max=' + f'{com_errors_mm.max():.4f}mm':<25s} "
        f"{'PASS' if n_com_fail == 0 else 'FAIL'}"
    )
    lines.append(
        f"  {'Quaternion mapping verified':<50s} "
        f"{'w/xyz/Z all match':<25s} "
        f"{'PASS' if (quat_w_match and quat_xyz_match and z_match) else 'FAIL'}"
    )
    lines.append(
        f"  {'Screenshot saved':<50s} "
        f"{'m3b_config.png':<25s} PASS"
    )
    lines.append(
        f"  {'Results saved':<50s} "
        f"{'m3b_fk_validation.txt':<25s} PASS"
    )
    lines.append("-" * 85)

    all_pass = (n_foot_fail == 0) and (n_com_fail == 0) and quat_w_match and quat_xyz_match and z_match
    lines.append(f"  {'OVERALL':<50s} {'':<25s} "
                 f"{'ALL PASS' if all_pass else 'SOME FAIL'}")
    lines.append("=" * 90)

    # ── Statistics ────────────────────────────────────────────────────
    lines.append("")
    lines.append("Statistics:")
    lines.append(f"  Foot errors [mm]:  mean={foot_errors_mm.mean():.4f}, "
                 f"std={foot_errors_mm.std():.4f}, "
                 f"max={foot_errors_mm.max():.4f}, "
                 f"min={foot_errors_mm.min():.4f}")
    lines.append(f"  CoM errors [mm]:   mean={com_errors_mm.mean():.4f}, "
                 f"std={com_errors_mm.std():.4f}, "
                 f"max={com_errors_mm.max():.4f}, "
                 f"min={com_errors_mm.min():.4f}")

    # ── Print and save ────────────────────────────────────────────────
    output = "\n".join(lines)
    print(output)

    output_path.write_text(output + "\n")
    print(f"\n  Output saved to: {output_path}")


if __name__ == "__main__":
    main()
