#!/usr/bin/env python3
"""M3e — Validate LIPM + ZMP Preview Control Walking Plan (pure math, no simulation).

Pipeline:
  1. Generate footstep plan (12 steps)
  2. Generate ZMP reference
  3. Run LIPM preview control to get CoM trajectory
  4. Generate swing foot trajectories for each step
  5. Plot: CoM XY, ZMP reference, foot positions, support polygon boundaries
  6. Verify: ZMP stays inside support polygon at all times
  7. Save plot to media/m3e_zmp_plan.png

Gate criteria:
  - ZMP reference alternates between feet correctly
  - CoM trajectory is smooth (no discontinuities)
  - CoM trajectory leads ZMP transitions (preview effect visible)
  - Plot clearly shows the walking pattern
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

from lab7_common import MEDIA_DIR, Z_C
from lipm_preview_control import generate_com_trajectory
from zmp_reference import (
    Footstep,
    WalkingTiming,
    generate_footstep_plan,
    generate_zmp_reference,
    get_phase_at_time,
)
from swing_trajectory import generate_swing_trajectory


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_STEPS = 12
STRIDE_X = 0.08       # 8 cm forward per step
FOOT_Y = 0.075        # 7.5 cm lateral half-spacing (15 cm between feet)
DT = 0.02             # 50 Hz control rate
STEP_HEIGHT = 0.03    # 3 cm swing foot clearance
FOOT_LENGTH = 0.17    # G1 foot length (toe-to-heel)
FOOT_WIDTH = 0.06     # G1 foot width

# Foot ground height (from M3a: ankle_roll_link at z≈0.033)
FOOT_Z = 0.033

TIMING = WalkingTiming(t_ss=0.8, t_ds=0.2, t_init=1.0, t_final=1.0)


# ---------------------------------------------------------------------------
# Support polygon helper
# ---------------------------------------------------------------------------


def foot_polygon_xy(fx: float, fy: float) -> np.ndarray:
    """Return 4 corners of foot rectangle centered at (fx, fy)."""
    hl = FOOT_LENGTH / 2.0
    hw = FOOT_WIDTH / 2.0
    return np.array([
        [fx - hl, fy - hw],
        [fx + hl, fy - hw],
        [fx + hl, fy + hw],
        [fx - hl, fy + hw],
    ])


def point_in_polygon(px: float, py: float, poly: np.ndarray) -> bool:
    """Check if point (px, py) is inside a convex polygon (vertices CCW)."""
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -1e-6:
            return False
    return True


def get_support_polygon(
    t_query: float,
    footsteps: list[Footstep],
    timing: WalkingTiming,
) -> np.ndarray:
    """Get the support polygon at a given time as vertex array.

    Returns an Nx2 array of polygon vertices.
    """
    phase, step_idx, _ = get_phase_at_time(t_query, footsteps, timing)

    if phase == "init_ds":
        # Double support: both initial feet
        p0 = foot_polygon_xy(footsteps[0].x, footsteps[0].y)
        p1 = foot_polygon_xy(footsteps[1].x, footsteps[1].y)
        all_pts = np.vstack([p0, p1])
    elif phase == "ss":
        # Single support: stance foot only (footsteps[step_idx])
        all_pts = foot_polygon_xy(footsteps[step_idx].x, footsteps[step_idx].y)
    elif phase == "ds":
        # Double support: current + next stance
        p0 = foot_polygon_xy(footsteps[step_idx].x, footsteps[step_idx].y)
        p1 = foot_polygon_xy(footsteps[step_idx + 1].x, footsteps[step_idx + 1].y)
        all_pts = np.vstack([p0, p1])
    elif phase == "final_ss":
        # Single support on last stepping foot
        all_pts = foot_polygon_xy(footsteps[-2].x, footsteps[-2].y)
    elif phase == "final_ds":
        # Double support: last two feet
        p0 = foot_polygon_xy(footsteps[-2].x, footsteps[-2].y)
        p1 = foot_polygon_xy(footsteps[-1].x, footsteps[-1].y)
        all_pts = np.vstack([p0, p1])
    else:
        all_pts = np.zeros((4, 2))

    # Compute convex hull (simple: sort by angle from centroid)
    cx, cy = all_pts.mean(axis=0)
    angles = np.arctan2(all_pts[:, 1] - cy, all_pts[:, 0] - cx)
    order = np.argsort(angles)
    return all_pts[order]


# ---------------------------------------------------------------------------
# Swing foot trajectories for all steps
# ---------------------------------------------------------------------------


def compute_all_swing_trajectories(
    footsteps: list[Footstep],
    timing: WalkingTiming,
    dt: float,
) -> dict[int, tuple[np.ndarray, float, float]]:
    """Compute swing foot trajectories for all walking steps.

    Returns:
        Dict mapping step_index -> (positions_array, t_start, t_end).
        positions_array is (N_swing, 3).
    """
    n_steps = len(footsteps) - 2
    trajectories = {}

    # Track where each foot is currently
    left_pos = np.array([footsteps[0].x if footsteps[0].foot == "L" else footsteps[1].x,
                         FOOT_Y, FOOT_Z])
    right_pos = np.array([footsteps[0].x if footsteps[0].foot == "R" else footsteps[1].x,
                          -FOOT_Y, FOOT_Z])

    # Initial stance foot
    # First step (i=0) is the initial stance. Steps 1..n_steps are forward steps.
    # We need to determine the initial positions of both feet.
    # footsteps[0] is the first stance foot.
    if footsteps[0].foot == "R":
        right_pos = np.array([footsteps[0].x, footsteps[0].y, FOOT_Z])
        left_pos = np.array([0.0, FOOT_Y, FOOT_Z])  # left starts at origin
    else:
        left_pos = np.array([footsteps[0].x, footsteps[0].y, FOOT_Z])
        right_pos = np.array([0.0, -FOOT_Y, FOOT_Z])

    for i in range(n_steps):
        # During SS phase i, the swing foot moves from its current pos to footsteps[i+1]
        t_start = timing.t_init + i * (timing.t_ss + timing.t_ds)

        swing_foot = footsteps[i + 1].foot
        target_pos = np.array([footsteps[i + 1].x, footsteps[i + 1].y, FOOT_Z])

        if swing_foot == "L":
            start_pos = left_pos.copy()
            swing_traj = generate_swing_trajectory(
                start_pos, target_pos, timing.t_ss, dt, STEP_HEIGHT,
            )
            left_pos = target_pos.copy()
        else:
            start_pos = right_pos.copy()
            swing_traj = generate_swing_trajectory(
                start_pos, target_pos, timing.t_ss, dt, STEP_HEIGHT,
            )
            right_pos = target_pos.copy()

        trajectories[i] = (swing_traj, t_start, t_start + timing.t_ss)

    # Final alignment step
    t_start = timing.t_init + n_steps * (timing.t_ss + timing.t_ds)
    swing_foot = footsteps[-1].foot
    target_pos = np.array([footsteps[-1].x, footsteps[-1].y, FOOT_Z])

    if swing_foot == "L":
        start_pos = left_pos.copy()
    else:
        start_pos = right_pos.copy()

    swing_traj = generate_swing_trajectory(
        start_pos, target_pos, timing.t_ss, dt, STEP_HEIGHT,
    )
    trajectories[n_steps] = (swing_traj, t_start, t_start + timing.t_ss)

    return trajectories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  M3e — LIPM + ZMP Preview Control Walking Plan Validation")
    print("=" * 70)

    # ── Step 1: Generate footstep plan ────────────────────────────
    footsteps = generate_footstep_plan(
        n_steps=N_STEPS, stride_x=STRIDE_X, foot_y=FOOT_Y, start_foot="R",
    )
    print(f"\nFootstep plan ({len(footsteps)} steps):")
    for i, fs in enumerate(footsteps):
        print(f"  [{i:2d}] foot={fs.foot}  x={fs.x:.3f}  y={fs.y:+.3f}")

    # ── Step 2: Generate ZMP reference ────────────────────────────
    zmp_x, zmp_y, t, total_time = generate_zmp_reference(footsteps, TIMING, DT)
    print(f"\nZMP reference: N={len(t)}, total_time={total_time:.2f}s")

    # ── Step 3: Run LIPM preview control ──────────────────────────
    # Initial CoM position = midpoint between first two feet
    com_x0 = (footsteps[0].x + footsteps[1].x) / 2.0
    com_y0 = (footsteps[0].y + footsteps[1].y) / 2.0

    com_x, com_y, com_dx, com_dy = generate_com_trajectory(
        zmp_x, zmp_y, z_c=Z_C, dt=DT,
        com_x0=com_x0, com_y0=com_y0,
        Q_e=1.0, R=1e-6, n_preview=80,
    )
    print(f"CoM trajectory generated: N={len(com_x)}")

    # ── Step 4: Generate swing foot trajectories ──────────────────
    swing_trajs = compute_all_swing_trajectories(footsteps, TIMING, DT)
    print(f"Swing trajectories: {len(swing_trajs)} steps")

    # ── Step 5: Verify ZMP inside support polygon ─────────────────
    n_violations = 0
    n_checked = 0
    for k in range(0, len(t), 5):  # check every 5th sample for speed
        poly = get_support_polygon(t[k], footsteps, TIMING)
        if not point_in_polygon(zmp_x[k], zmp_y[k], poly):
            n_violations += 1
        n_checked += 1

    zmp_inside_pct = 100.0 * (1.0 - n_violations / n_checked)
    print(f"ZMP inside support polygon: {zmp_inside_pct:.1f}% ({n_violations} violations out of {n_checked})")

    # ── Smoothness check: max CoM acceleration jump ───────────────
    com_ddx = np.diff(com_dx) / DT
    com_ddy = np.diff(com_dy) / DT
    max_jerk_x = np.max(np.abs(np.diff(com_ddx)))
    max_jerk_y = np.max(np.abs(np.diff(com_ddy)))
    print(f"Max CoM jerk: X={max_jerk_x:.4f}, Y={max_jerk_y:.4f} m/s^3")

    # ── Preview effect: CoM should lead ZMP transitions ───────────
    # Look at the first SS→DS transition (after init phase).
    # At t = t_init + t_ss, the ZMP starts moving from stance foot 0
    # to stance foot 1. The CoM should start moving BEFORE this.
    first_ds_time = TIMING.t_init + TIMING.t_ss  # when first DS starts
    first_ds_idx = int(first_ds_time / DT)

    # The ZMP Y at the first stance foot
    zmp_y_stance = footsteps[0].y  # first stance foot Y

    # Find when CoM Y starts moving away from the stance foot Y direction
    # The CoM should be pulled toward the next foot BEFORE the ZMP transitions
    # Measure: when does CoM Y deviate by > 2mm from its value at t_init?
    com_y_at_init = com_y[int(TIMING.t_init / DT)]
    next_foot_y = footsteps[1].y
    # Direction of expected CoM movement
    direction = np.sign(next_foot_y - com_y_at_init)

    # Find when CoM starts moving in the correct direction by > 2mm
    ss_start_idx = int(TIMING.t_init / DT)
    com_move_idx = first_ds_idx  # default: no lead
    for k in range(ss_start_idx, first_ds_idx):
        if direction * (com_y[k] - com_y_at_init) > 0.002:
            com_move_idx = k
            break

    preview_lead = first_ds_time - t[com_move_idx]
    print(f"Preview effect: CoM leads ZMP Y transition by {preview_lead:.3f}s")

    # ── Step 6: Plot ──────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#08111f")

    # Color scheme
    bg_color = "#08111f"
    text_color = "#e0e0e0"
    grid_color = "#1a2a3f"
    zmp_color = "#ff6b6b"
    com_color = "#4ecdc4"
    lf_color = "#45b7d1"
    rf_color = "#f7dc6f"
    poly_color = "#2ecc71"

    # Layout: 2x2
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ── Plot 1: XY trajectory (top-left, large) ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg_color)

    # Draw foot rectangles
    for fs in footsteps:
        corners = foot_polygon_xy(fs.x, fs.y)
        color = lf_color if fs.foot == "L" else rf_color
        poly_patch = MplPolygon(corners, closed=True,
                                facecolor=color, alpha=0.2, edgecolor=color, linewidth=1.5)
        ax1.add_patch(poly_patch)
        ax1.text(fs.x, fs.y, fs.foot, ha="center", va="center",
                 color=color, fontsize=8, fontweight="bold")

    # ZMP reference
    ax1.plot(zmp_x, zmp_y, "-", color=zmp_color, linewidth=1.5, alpha=0.7, label="ZMP ref")

    # CoM trajectory
    ax1.plot(com_x, com_y, "-", color=com_color, linewidth=2.5, label="CoM")

    # Footstep centers
    for fs in footsteps:
        marker = "^" if fs.foot == "L" else "v"
        color = lf_color if fs.foot == "L" else rf_color
        ax1.plot(fs.x, fs.y, marker, color=color, markersize=8, zorder=5)

    ax1.set_xlabel("X [m]", color=text_color)
    ax1.set_ylabel("Y [m]", color=text_color)
    ax1.set_title("Walking Plan — XY View", color=text_color, fontsize=14, fontweight="bold")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper left", facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    ax1.grid(True, color=grid_color, alpha=0.5)
    ax1.tick_params(colors=text_color)
    for spine in ax1.spines.values():
        spine.set_color(grid_color)

    # ── Plot 2: ZMP X vs CoM X over time (top-right) ─────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(bg_color)
    ax2.plot(t, zmp_x, "-", color=zmp_color, linewidth=1.5, alpha=0.8, label="ZMP ref X")
    ax2.plot(t, com_x, "-", color=com_color, linewidth=2, label="CoM X")
    ax2.set_xlabel("Time [s]", color=text_color)
    ax2.set_ylabel("X position [m]", color=text_color)
    ax2.set_title("Sagittal (X): ZMP Reference vs CoM", color=text_color, fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    ax2.grid(True, color=grid_color, alpha=0.5)
    ax2.tick_params(colors=text_color)
    for spine in ax2.spines.values():
        spine.set_color(grid_color)

    # ── Plot 3: ZMP Y vs CoM Y over time (bottom-left) ───────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(bg_color)
    ax3.plot(t, zmp_y, "-", color=zmp_color, linewidth=1.5, alpha=0.8, label="ZMP ref Y")
    ax3.plot(t, com_y, "-", color=com_color, linewidth=2, label="CoM Y")

    # Add foot Y positions for reference
    for fs in footsteps:
        color = lf_color if fs.foot == "L" else rf_color
        ax3.axhline(y=fs.y, color=color, linestyle=":", alpha=0.3)

    ax3.set_xlabel("Time [s]", color=text_color)
    ax3.set_ylabel("Y position [m]", color=text_color)
    ax3.set_title("Lateral (Y): ZMP Reference vs CoM", color=text_color, fontsize=14, fontweight="bold")
    ax3.legend(loc="upper left", facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    ax3.grid(True, color=grid_color, alpha=0.5)
    ax3.tick_params(colors=text_color)
    for spine in ax3.spines.values():
        spine.set_color(grid_color)

    # ── Plot 4: Swing foot Z trajectories (bottom-right) ─────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(bg_color)

    for step_i, (traj, t_start, t_end) in swing_trajs.items():
        if step_i >= len(footsteps) - 1:
            continue
        swing_foot = footsteps[step_i + 1].foot if step_i < len(footsteps) - 2 else footsteps[-1].foot
        color = lf_color if swing_foot == "L" else rf_color
        t_swing = np.linspace(t_start, t_end, len(traj))
        ax4.plot(t_swing, traj[:, 2] * 1000, "-", color=color, linewidth=1.5, alpha=0.8)

    ax4.axhline(y=FOOT_Z * 1000, color=text_color, linestyle="--", alpha=0.3, label=f"Ground ({FOOT_Z*1000:.0f}mm)")
    ax4.axhline(y=(FOOT_Z + STEP_HEIGHT) * 1000, color=poly_color, linestyle="--", alpha=0.3, label=f"Peak ({(FOOT_Z+STEP_HEIGHT)*1000:.0f}mm)")
    ax4.set_xlabel("Time [s]", color=text_color)
    ax4.set_ylabel("Foot Z [mm]", color=text_color)
    ax4.set_title("Swing Foot Vertical Trajectories", color=text_color, fontsize=14, fontweight="bold")
    ax4.legend(loc="upper right", facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    ax4.grid(True, color=grid_color, alpha=0.5)
    ax4.tick_params(colors=text_color)
    for spine in ax4.spines.values():
        spine.set_color(grid_color)

    # ── Save ──────────────────────────────────────────────────────
    plot_path = MEDIA_DIR / "m3e_zmp_plan.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig)
    print(f"\nPlot saved: {plot_path}")

    # ── Gate Results ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  M3e Step 4 — GATE RESULTS")
    print("=" * 70)
    print(f"  {'Criterion':<50s} {'Value':<20s} {'Status'}")
    print("-" * 75)

    # Gate 1: ZMP reference alternates between feet
    zmp_y_unique = np.unique(np.round(zmp_y[t > TIMING.t_init], 4))
    zmp_alternates = len(zmp_y_unique) >= 2
    print(f"  {'ZMP alternates between feet':<50s} {f'{len(zmp_y_unique)} Y levels':<20s} {'PASS' if zmp_alternates else 'FAIL'}")

    # Gate 2: CoM trajectory smooth
    com_smooth = max_jerk_x < 100 and max_jerk_y < 100
    print(f"  {'CoM smooth (jerk < 100 m/s^3)':<50s} {f'X={max_jerk_x:.2f} Y={max_jerk_y:.2f}':<20s} {'PASS' if com_smooth else 'FAIL'}")

    # Gate 3: Preview effect visible
    preview_ok = preview_lead > 0.05
    print(f"  {'CoM leads ZMP (> 50ms preview)':<50s} {f'{preview_lead*1000:.0f}ms':<20s} {'PASS' if preview_ok else 'FAIL'}")

    # Gate 4: ZMP inside support polygon
    zmp_ok = zmp_inside_pct > 95.0
    print(f"  {'ZMP inside support polygon (> 95%)':<50s} {f'{zmp_inside_pct:.1f}%':<20s} {'PASS' if zmp_ok else 'FAIL'}")

    # Gate 5: Plot saved
    plot_saved = plot_path.exists()
    print(f"  {'Plot saved':<50s} {'m3e_zmp_plan.png':<20s} {'PASS' if plot_saved else 'FAIL'}")

    print("-" * 75)
    all_pass = zmp_alternates and com_smooth and preview_ok and zmp_ok and plot_saved
    print(f"  {'OVERALL':<50s} {'':<20s} {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
