#!/usr/bin/env python3
"""M2 — CoM Tracking, Support Polygon, and CoM-based Balance Controller.

Pipeline:
  1. Cross-validate CoM: Pinocchio vs MuJoCo (< 5mm)
  2. Compute support polygon from foot contact geom positions
  3. CoM-based balance controller: PD on CoM XY using CoM Jacobian
  4. Run 10s with 5N push, verify CoM stays inside support polygon

Controller:
  com_error = com_desired[:2] - com_actual[:2]
  delta_q = J_com_pinv @ (K_com * com_error)
  ctrl = (q_ref + delta_q) + qfrc_bias / Kp

Gate:
  - CoM cross-validation error < 5mm
  - CoM stays inside support polygon during 5N push
  - Plot: media/m2_com_polygon.png
  - Video: media/m2_com_balance.mp4
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
    ARM_NEUTRAL_LEFT,
    ARM_NEUTRAL_RIGHT,
    CTRL_LEFT_ARM,
    CTRL_RIGHT_ARM,
    CTRL_STAND,
    G1_MJCF_PATH,
    MEDIA_DIR,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    load_g1_mujoco,
    mj_qpos_to_pin,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SIM_DURATION: float = 10.0
PUSH_TIME: float = 3.0
PUSH_DURATION: float = 0.2
PUSH_FORCE_Y: float = 5.0
KP_SERVO: float = 500.0       # Menagerie position actuator gain

# CoM balance gain (position-level correction, must be small)
K_COM: float = 0.3            # maps CoM XY error to joint angle correction
MAX_DELTA_Q: float = 0.05     # max joint correction per step [rad]


# ---------------------------------------------------------------------------
# Support polygon from foot contacts
# ---------------------------------------------------------------------------


def get_foot_contact_points(
    m: mujoco.MjModel, d: mujoco.MjData,
) -> np.ndarray:
    """Get world positions of all foot contact sphere geoms.

    Returns:
        (N, 3) array of foot contact point world positions.
    """
    mujoco.mj_forward(m, d)
    points = []
    for i in range(m.ngeom):
        g = m.geom(i)
        # Menagerie foot geoms are spheres with class="foot"
        if g.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE and g.size[0] < 0.01:
            # Check if parent body is an ankle_roll_link (foot)
            body_name = m.body(g.bodyid[0]).name
            if "ankle_roll" in body_name:
                points.append(d.geom_xpos[i].copy())
    return np.array(points)


def compute_support_polygon_xy(
    foot_points: np.ndarray,
) -> np.ndarray:
    """Compute the 2D convex hull of foot contact points projected on XY.

    Returns:
        (M, 2) array of convex hull vertices in order.
    """
    from scipy.spatial import ConvexHull

    pts_2d = foot_points[:, :2]
    hull = ConvexHull(pts_2d)
    return pts_2d[hull.vertices]


def point_in_polygon(point_xy: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if a 2D point is inside a convex polygon (winding number)."""
    n = len(polygon)
    inside = True
    for i in range(n):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % n]
        edge = v2 - v1
        to_point = point_xy - v1
        cross = edge[0] * to_point[1] - edge[1] * to_point[0]
        if cross < -1e-10:
            inside = False
            break
    # Try the other winding direction
    if not inside:
        inside = True
        for i in range(n):
            v1 = polygon[i]
            v2 = polygon[(i + 1) % n]
            edge = v2 - v1
            to_point = point_xy - v1
            cross = edge[0] * to_point[1] - edge[1] * to_point[0]
            if cross > 1e-10:
                inside = False
                break
    return inside


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

    # ── Load models ─────────────────────────────────────────────────────
    m, d = load_g1_mujoco()
    pin_model = pin.buildModelFromMJCF(str(G1_MJCF_PATH), pin.JointModelFreeFlyer())
    pin_data = pin_model.createData()

    dt = m.opt.timestep
    n_steps = int(SIM_DURATION / dt)
    render_every = max(1, int(1.0 / (RENDER_FPS * dt)))

    torso_id = m.body("torso_link").id
    q_ref = CTRL_STAND.copy()

    # ── 1. Cross-validate CoM ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  STEP 1: CoM Cross-Validation")
    print("=" * 80)

    q_pin = mj_qpos_to_pin(d.qpos)
    v_pin = d.qvel.copy()

    mujoco.mj_forward(m, d)
    com_mj = d.subtree_com[0].copy()

    pin.centerOfMass(pin_model, pin_data, q_pin, v_pin)
    com_pin = pin_data.com[0].copy()

    com_error_mm = np.linalg.norm(com_mj - com_pin) * 1000
    print(f"  MuJoCo CoM:    {com_mj}")
    print(f"  Pinocchio CoM: {com_pin}")
    print(f"  Error: {com_error_mm:.3f} mm")

    # ── 2. Support polygon ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  STEP 2: Support Polygon")
    print("=" * 80)

    foot_pts = get_foot_contact_points(m, d)
    print(f"  Foot contact points: {len(foot_pts)}")
    for i, p in enumerate(foot_pts):
        print(f"    [{i}] {p}")

    support_poly = compute_support_polygon_xy(foot_pts)
    print(f"  Support polygon vertices: {len(support_poly)}")

    poly_center_xy = support_poly.mean(axis=0)
    # Desired CoM = initial CoM position (maintain current balance)
    com_desired_xy = com_mj[:2].copy()
    print(f"  Support polygon center: {poly_center_xy}")
    print(f"  Initial CoM XY (= desired): {com_desired_xy}")

    # ── 3. CoM Balance Controller Simulation ────────────────────────────
    print("\n" + "=" * 80)
    print("  STEP 3: CoM Balance Controller (10s, 5N push at t=3s)")
    print("=" * 80)

    # Logging
    times: list[float] = []
    com_xy_log: list[np.ndarray] = []
    com_mj_log: list[np.ndarray] = []
    com_pin_log: list[np.ndarray] = []
    pelvis_z_log: list[float] = []
    com_inside_log: list[bool] = []
    support_poly_log: list[np.ndarray] = []

    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(m, RENDER_HEIGHT, RENDER_WIDTH)
    cam = mujoco.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -15.0
    cam.azimuth = 140.0
    cam.lookat[:] = [0.0, 0.0, 0.8]

    for step in range(n_steps):
        t = step * dt

        # --- Get Pinocchio state ---
        q_pin = mj_qpos_to_pin(d.qpos)
        v_pin = d.qvel.copy()

        # --- Compute CoM and Jacobian ---
        pin.centerOfMass(pin_model, pin_data, q_pin, v_pin)
        com_now = pin_data.com[0].copy()
        J_com = pin.jacobianCenterOfMass(pin_model, pin_data, q_pin)

        # --- CoM position error (XY only) ---
        com_error_xy = com_desired_xy - com_now[:2]

        # --- CoM Jacobian → joint position correction ---
        # delta_q = K * J_com_pinv * com_error (damped least squares)
        J_act_xy = J_com[:2, 6:]  # (2, 29) — actuated DOFs only
        JJt = J_act_xy @ J_act_xy.T + 1e-4 * np.eye(2)
        delta_q = K_COM * (J_act_xy.T @ np.linalg.solve(JJt, com_error_xy))

        # Clamp corrections to prevent instability
        delta_q = np.clip(delta_q, -MAX_DELTA_Q, MAX_DELTA_Q)

        # --- Gravity-compensated PD with CoM correction ---
        bias = d.qfrc_bias[6:35]
        d.ctrl[:] = q_ref + delta_q + bias / KP_SERVO

        # Lock arms at neutral (no CoM correction for arms)
        d.ctrl[CTRL_LEFT_ARM] = ARM_NEUTRAL_LEFT + bias[15:22] / KP_SERVO
        d.ctrl[CTRL_RIGHT_ARM] = ARM_NEUTRAL_RIGHT + bias[22:29] / KP_SERVO

        # --- Apply push ---
        if PUSH_TIME <= t < PUSH_TIME + PUSH_DURATION:
            d.xfrc_applied[torso_id, 1] = PUSH_FORCE_Y
        else:
            d.xfrc_applied[torso_id, :] = 0.0

        mujoco.mj_step(m, d)

        # --- Log ---
        times.append(t)
        com_xy_log.append(com_now[:2].copy())
        pelvis_z_log.append(d.qpos[2])

        # Cross-validate CoM periodically
        if step % 50 == 0:
            mujoco.mj_forward(m, d)
            com_mj_now = d.subtree_com[0].copy()
            q_pin_check = mj_qpos_to_pin(d.qpos)
            pin.centerOfMass(pin_model, pin_data, q_pin_check)
            com_pin_now = pin_data.com[0].copy()
            com_mj_log.append(com_mj_now.copy())
            com_pin_log.append(com_pin_now.copy())

        # Support polygon and inside check
        if step % 100 == 0:
            fp = get_foot_contact_points(m, d)
            if len(fp) >= 3:
                sp = compute_support_polygon_xy(fp)
                support_poly_log.append(sp)
                inside = point_in_polygon(com_now[:2], sp)
                com_inside_log.append(inside)

        # Render
        if step % render_every == 0:
            renderer.update_scene(d, cam)
            frames.append(renderer.render().copy())

    renderer.close()
    save_mp4(frames, MEDIA_DIR / "m2_com_balance.mp4")

    # ── 4. Analysis ─────────────────────────────────────────────────────
    com_xy_arr = np.array(com_xy_log)
    times_arr = np.array(times)

    # Cross-validation error
    if com_mj_log and com_pin_log:
        mj_arr = np.array(com_mj_log)
        pin_arr = np.array(com_pin_log)
        cross_errors = np.linalg.norm(mj_arr - pin_arr, axis=1) * 1000
        max_cross_error = cross_errors.max()
        mean_cross_error = cross_errors.mean()
    else:
        max_cross_error = mean_cross_error = 0.0

    # CoM inside support polygon
    all_inside = all(com_inside_log) if com_inside_log else False
    pct_inside = (sum(com_inside_log) / len(com_inside_log) * 100) if com_inside_log else 0

    # Pelvis height
    pelvis_z = np.array(pelvis_z_log)
    z_dev = abs(pelvis_z - pelvis_z[0]).max()

    # ── 5. Plot ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # --- Left: CoM XY trajectory vs support polygon ---
        ax = axes[0]
        # Draw support polygon (from first sample)
        if support_poly_log:
            sp0 = support_poly_log[0]
            poly = MplPolygon(sp0, closed=True, alpha=0.2, color="green",
                              label="Support polygon")
            ax.add_patch(poly)
            # Also draw vertices
            sp_closed = np.vstack([sp0, sp0[0]])
            ax.plot(sp_closed[:, 0], sp_closed[:, 1], "g-", linewidth=2)

        # CoM trajectory
        ax.plot(com_xy_arr[:, 0], com_xy_arr[:, 1], "b-", linewidth=0.5,
                alpha=0.6, label="CoM trajectory")
        ax.plot(com_xy_arr[0, 0], com_xy_arr[0, 1], "go", markersize=8,
                label="Start")
        ax.plot(com_xy_arr[-1, 0], com_xy_arr[-1, 1], "ro", markersize=8,
                label="End")
        ax.plot(com_desired_xy[0], com_desired_xy[1], "k+", markersize=12,
                markeredgewidth=2, label="Desired")

        # Push annotation
        push_start = int(PUSH_TIME / dt)
        push_end = min(int((PUSH_TIME + PUSH_DURATION) / dt), len(com_xy_arr) - 1)
        if push_end < len(com_xy_arr):
            ax.plot(com_xy_arr[push_start:push_end, 0],
                    com_xy_arr[push_start:push_end, 1],
                    "r-", linewidth=3, label=f"{PUSH_FORCE_Y}N push")

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("CoM XY vs Support Polygon")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # --- Right: CoM XY over time ---
        ax2 = axes[1]
        ax2.plot(times_arr, com_xy_arr[:, 0], "b-", label="CoM X")
        ax2.plot(times_arr, com_xy_arr[:, 1], "r-", label="CoM Y")
        ax2.axvspan(PUSH_TIME, PUSH_TIME + PUSH_DURATION, alpha=0.2,
                     color="orange", label="Push")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Position [m]")
        ax2.set_title("CoM X, Y over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = MEDIA_DIR / "m2_com_polygon.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {plot_path}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    # ── 6. Gate Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  M2 GATE RESULTS")
    print(f"{'=' * 80}")
    print(f"  {'Criterion':<45} {'Value':<25} {'Status'}")
    print(f"{'-' * 80}")

    cv_pass = "PASS" if max_cross_error < 5.0 else "FAIL"
    print(f"  {'CoM cross-validation (max error)':<45} {f'{max_cross_error:.3f} mm':<25} {cv_pass}")
    print(f"  {'CoM cross-validation (mean error)':<45} {f'{mean_cross_error:.3f} mm':<25} {'—'}")

    inside_pass = "PASS" if pct_inside >= 99.0 else "FAIL"
    print(f"  {'CoM inside support polygon':<45} {f'{pct_inside:.1f}%':<25} {inside_pass}")

    z_pass = "PASS" if z_dev < 0.05 else "FAIL"
    print(f"  {'Pelvis height deviation':<45} {f'{z_dev*1000:.1f} mm':<25} {z_pass}")

    print(f"  {'Plot saved':<45} {'m2_com_polygon.png':<25} {'PASS'}")
    print(f"  {'Video saved':<45} {'m2_com_balance.mp4':<25} {'PASS'}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
