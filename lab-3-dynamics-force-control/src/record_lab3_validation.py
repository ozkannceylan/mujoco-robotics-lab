"""Record a headless Lab 3 validation video from the real simulation data.

This recorder avoids MuJoCo OpenGL rendering so it can run in restricted
environments while still visualizing the actual UR5e + Robotiq simulation
results, force traces, and pass/fail metrics.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import imageio.v2 as imageio
import matplotlib
import numpy as np
import mujoco

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from a1_dynamics_fundamentals import cross_validate_gravity, cross_validate_mass_matrix
from a2_gravity_compensation import run_gravity_comp_sim
from c1_force_control import DEFAULT_APPROACH_HEIGHT, HybridGains, run_hybrid_force_sim
from c2_line_trace import run_line_trace
from lab3_common import (
    DT,
    MEDIA_DIR,
    NUM_JOINTS,
    Q_HOME,
    Q_ZEROS,
    SCENE_TABLE_PATH,
    TABLE_SURFACE_Z,
    load_mujoco_model,
    load_pinocchio_model,
)


VIDEO_PATH = MEDIA_DIR / "lab3_validation_real_stack.mp4"
FPS = 25
FIGSIZE = (12.8, 7.2)
DPI = 100


@dataclass(frozen=True)
class SegmentSpec:
    """Animation settings for a simulation segment."""

    title: str
    speedup: float


def _representative_configs() -> list[np.ndarray]:
    """Return deterministic joint configurations for dynamics cross-checks."""
    return [
        Q_ZEROS.copy(),
        Q_HOME.copy(),
        np.array([-1.2, -1.7, 1.8, -1.5, -1.2, 0.2]),
        np.array([-0.8, -1.1, 1.1, -1.7, -1.4, 0.8]),
    ]


def compute_dynamics_summary() -> dict[str, float]:
    """Compute representative Pinocchio vs MuJoCo dynamics parity metrics."""
    pin_model, pin_data, _ = load_pinocchio_model()
    mj_model, mj_data = load_mujoco_model()

    grav_errs = []
    mass_errs = []
    for q in _representative_configs():
        grav_errs.append(cross_validate_gravity(pin_model, pin_data, mj_model, mj_data, q))
        mass_errs.append(cross_validate_mass_matrix(pin_model, pin_data, mj_model, mj_data, q))

    return {
        "max_gravity_err": float(np.max(grav_errs)),
        "max_mass_err": float(np.max(mass_errs)),
        "num_configs": float(len(grav_errs)),
    }


def compute_gravity_metrics(
    hold_results: dict[str, np.ndarray],
    perturb_results: dict[str, np.ndarray],
) -> dict[str, float]:
    """Summarize gravity-compensation performance using the lab's two checks."""
    hold_q_error = hold_results["q_error"]
    perturb_qd = perturb_results["qd"]
    max_err_rad = float(np.max(np.abs(hold_q_error)))
    max_err_deg = float(np.degrees(max_err_rad))
    final_vel_rad = float(np.max(np.abs(perturb_qd[-100:])))
    final_vel_deg = float(np.degrees(final_vel_rad))
    return {
        "max_err_rad": max_err_rad,
        "max_err_deg": max_err_deg,
        "final_vel_rad": final_vel_rad,
        "final_vel_deg": final_vel_deg,
        "pass_hold": max_err_rad < 0.01,
        "pass_settle": final_vel_rad < 0.1,
    }


def compute_hybrid_metrics(
    results: dict[str, np.ndarray],
    xy_des: np.ndarray,
    f_desired: float,
) -> dict[str, float]:
    """Summarize hybrid-force performance against the lab criteria."""
    phase = results["phase"]
    hybrid_mask = phase == 1
    if not np.any(hybrid_mask):
        return {
            "contact_reached": False,
            "mean_force": float("nan"),
            "pct_in_band": 0.0,
            "max_xy_err_mm": float("inf"),
            "contact_time": float("nan"),
            "settle_time": float("nan"),
        }

    hybrid_start = int(np.argmax(hybrid_mask))
    settle_idx = min(hybrid_start + 500, len(results["force_z"]) - 1)
    force_settled = results["force_z"][settle_idx:]
    xy_err_mm = (
        np.linalg.norm(results["ee_pos"][settle_idx:, :2] - xy_des[np.newaxis, :], axis=1)
        * 1000.0
    )
    return {
        "contact_reached": True,
        "mean_force": float(np.mean(force_settled)),
        "pct_in_band": float(np.mean(np.abs(force_settled - f_desired) < 1.0) * 100.0),
        "max_xy_err_mm": float(np.max(xy_err_mm)),
        "contact_time": float(results["time"][hybrid_start]),
        "settle_time": float(results["time"][settle_idx]),
    }


def compute_line_metrics(
    results: dict[str, np.ndarray],
    f_desired: float,
) -> dict[str, float]:
    """Summarize line-tracing performance against the lab criteria."""
    phase = results["phase"]
    trace_mask = phase == 2
    if not np.any(trace_mask):
        return {
            "trace_started": False,
            "pct_in_band": 0.0,
            "max_xy_err_mm": float("inf"),
            "trace_time": float("nan"),
        }

    xy_actual = results["ee_pos"][trace_mask, :2]
    xy_desired = results["xy_des"][trace_mask]
    xy_err_mm = np.linalg.norm(xy_actual - xy_desired, axis=1) * 1000.0
    return {
        "trace_started": True,
        "pct_in_band": float(np.mean(np.abs(results["force_z"][trace_mask] - f_desired) < 1.0) * 100.0),
        "max_xy_err_mm": float(np.max(xy_err_mm)),
        "trace_time": float(results["time"][np.argmax(trace_mask)]),
    }


def _frame_rgb(fig: plt.Figure) -> np.ndarray:
    """Convert the current Agg canvas to an RGB frame."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def _sample_indices(num_steps: int, speedup: float) -> np.ndarray:
    """Select animation frames for a simulation segment."""
    stride = max(1, int(round(speedup / (FPS * DT))))
    indices = np.arange(0, num_steps, stride, dtype=int)
    if len(indices) == 0 or indices[-1] != num_steps - 1:
        indices = np.append(indices, num_steps - 1)
    return indices


def _phase_label_force(phase: int) -> str:
    return {0: "Approach", 1: "Hybrid contact"}.get(int(phase), "Unknown")


def _phase_label_line(phase: int) -> str:
    return {0: "Approach", 1: "Settle", 2: "Trace", 3: "Hold"}.get(int(phase), "Unknown")


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _project_points(points: np.ndarray, view: str) -> np.ndarray:
    """Project 3D world points into an orthographic 2D view."""
    if view == "side":
        return points[:, [0, 2]]
    if view == "top":
        return points[:, [0, 1]]
    raise ValueError(f"Unsupported view: {view}")


def _view_depth(pos: np.ndarray, view: str) -> float:
    """Depth key used for painter-style sorting."""
    if view == "side":
        return float(pos[1])
    if view == "top":
        return float(pos[2])
    raise ValueError(f"Unsupported view: {view}")


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the 2D convex hull of a point cloud using the monotonic chain."""
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float)


def _box_corners(pos: np.ndarray, rot: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """Return the 8 world-frame corners of an oriented bounding box."""
    local = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    ) * half_extents[np.newaxis, :]
    return pos[np.newaxis, :] + local @ rot.T


def _geom_half_extents(geom_type: int, size: np.ndarray) -> np.ndarray | None:
    """Approximate a MuJoCo geom by a projected bounding volume."""
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return np.array([size[0], size[0], size[0]], dtype=float)
    if geom_type in (
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    ):
        return np.array([size[0], size[0], size[1] + size[0]], dtype=float)
    return np.maximum(np.asarray(size, dtype=float), 1e-4)


def _draw_mujoco_projection(
    ax: plt.Axes,
    mj_model,
    mj_data,
    view: str,
    title: str,
) -> None:
    """Draw a simple orthographic MuJoCo scene view from actual geom poses."""
    geom_items = []
    for geom_id in range(mj_model.ngeom):
        rgba = np.asarray(mj_model.geom_rgba[geom_id], dtype=float)
        if rgba[3] <= 0.01:
            continue

        geom_type = int(mj_model.geom_type[geom_id])
        pos = np.asarray(mj_data.geom_xpos[geom_id], dtype=float)
        rot = np.asarray(mj_data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        size = np.asarray(mj_model.geom_size[geom_id], dtype=float)
        geom_items.append((geom_id, geom_type, pos, rot, size, rgba))

    geom_items.sort(key=lambda item: _view_depth(item[2], view))

    if view == "side":
        ax.axhline(0.0, color="#9aa0a6", linewidth=1.0, alpha=0.6, zorder=0)

    for _, geom_type, pos, rot, size, rgba in geom_items:
        face = np.clip(rgba[:3], 0.0, 1.0)
        edge = tuple(np.clip(face * 0.55, 0.0, 1.0))

        if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue

        half_extents = _geom_half_extents(geom_type, size)
        if half_extents is None:
            continue

        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            center = _project_points(pos[np.newaxis, :], view)[0]
            radius = float(size[0])
            ax.add_patch(
                Circle(
                    center,
                    radius=radius,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.8,
                    alpha=rgba[3],
                )
            )
            continue

        corners = _box_corners(pos, rot, half_extents)
        projected = _project_points(corners, view)
        hull = _convex_hull(projected)
        if len(hull) >= 3:
            ax.add_patch(
                Polygon(
                    hull,
                    closed=True,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.8,
                    alpha=rgba[3],
                )
            )

    ax.set_title(title)
    _style_axis(ax)


def _intro_frames(summary: dict[str, float]) -> list[np.ndarray]:
    """Create an intro title card with dynamics parity metrics."""
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="#f7f4ed")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.05, 0.87, "Lab 3 Validation", fontsize=30, fontweight="bold", color="#1d2433")
    ax.text(
        0.05,
        0.79,
        "Real stack: MuJoCo Menagerie UR5e + mounted Robotiq 2F-85",
        fontsize=16,
        color="#344054",
    )
    ax.text(0.05, 0.66, "Dynamics parity", fontsize=20, fontweight="bold", color="#7a2e0b")
    ax.text(
        0.05,
        0.58,
        f"Representative configurations checked: {int(summary['num_configs'])}",
        fontsize=15,
        color="#1d2433",
    )
    ax.text(
        0.05,
        0.51,
        f"Max gravity-vector mismatch |g_pin - g_mj|: {summary['max_gravity_err']:.3e}",
        fontsize=15,
        color="#1d2433",
    )
    ax.text(
        0.05,
        0.44,
        f"Max mass-matrix mismatch |M_pin - M_mj|: {summary['max_mass_err']:.3e}",
        fontsize=15,
        color="#1d2433",
    )
    ax.text(0.05, 0.31, "Validation sequence", fontsize=20, fontweight="bold", color="#7a2e0b")
    ax.text(0.05, 0.23, "1. Gravity compensation with perturbation", fontsize=15, color="#1d2433")
    ax.text(0.05, 0.17, "2. Hybrid force control on the table", fontsize=15, color="#1d2433")
    ax.text(0.05, 0.11, "3. Constant-force line tracing", fontsize=15, color="#1d2433")

    frame = _frame_rgb(fig)
    plt.close(fig)
    return [frame] * (FPS * 2)


def render_gravity_frames(
    results: dict[str, np.ndarray],
    metrics: dict[str, float],
    mj_model,
    mj_data,
    segment: SegmentSpec,
) -> list[np.ndarray]:
    """Render the gravity-compensation dashboard frames."""
    indices = _sample_indices(len(results["time"]), segment.speedup)
    max_err_deg_series = np.max(np.abs(np.degrees(results["q_error"])), axis=1)

    frames = []
    for idx in indices:
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.1], hspace=0.28, wspace=0.20)
        ax_robot = fig.add_subplot(gs[:, 0])
        ax_err = fig.add_subplot(gs[0, 1])
        ax_tau = fig.add_subplot(gs[1, 1])

        mj_data.qpos[:NUM_JOINTS] = results["q"][idx]
        mj_data.qvel[:NUM_JOINTS] = results["qd"][idx]
        mujoco.mj_forward(mj_model, mj_data)
        _draw_mujoco_projection(ax_robot, mj_model, mj_data, view="side", title="MuJoCo scene")
        ax_robot.set_xlim(-0.35, 0.85)
        ax_robot.set_ylim(-0.05, 0.95)
        ax_robot.set_xlabel("X (m)")
        ax_robot.set_ylabel("Z (m)")
        fig.text(0.07, 0.93, segment.title, fontsize=18, fontweight="bold", color="#111827")
        status = "PASS" if metrics["pass_hold"] and metrics["pass_settle"] else "FAIL"
        ax_robot.text(
            0.03,
            0.97,
            "\n".join(
                [
                    f"t = {results['time'][idx]:.2f} s",
                    f"Max |q-q0| = {metrics['max_err_deg']:.3f} deg",
                    f"Final |qd| = {metrics['final_vel_deg']:.3f} deg/s",
                    f"Status: {status}",
                ]
            ),
            transform=ax_robot.transAxes,
            va="top",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
        )

        for joint_id in range(NUM_JOINTS):
            ax_err.plot(
                results["time"][: idx + 1],
                np.degrees(results["q_error"][: idx + 1, joint_id]),
                linewidth=1.2,
                label=f"J{joint_id + 1}",
            )
        ax_err.axhline(np.degrees(0.01), color="#dc2626", linestyle="--", linewidth=1.0)
        ax_err.axhline(-np.degrees(0.01), color="#dc2626", linestyle="--", linewidth=1.0)
        ax_err.set_title("Joint position error")
        ax_err.set_ylabel("deg")
        ax_err.set_xlim(results["time"][0], results["time"][-1])
        _style_axis(ax_err)
        ax_err.legend(loc="upper right", ncol=3, fontsize=8)

        for joint_id in range(NUM_JOINTS):
            ax_tau.plot(
                results["time"][: idx + 1],
                results["tau"][: idx + 1, joint_id],
                linewidth=1.2,
                label=f"J{joint_id + 1}",
            )
        ax_tau.set_title("Applied arm torques")
        ax_tau.set_xlabel("Time (s)")
        ax_tau.set_ylabel("Nm")
        ax_tau.set_xlim(results["time"][0], results["time"][-1])
        _style_axis(ax_tau)

        fig.text(0.63, 0.93, "Recovery-run joint error history", fontsize=11, color="#334155")

        frames.append(_frame_rgb(fig))
        plt.close(fig)

    return frames


def render_hybrid_frames(
    results: dict[str, np.ndarray],
    metrics: dict[str, float],
    mj_model,
    mj_data,
    xy_des: np.ndarray,
    f_desired: float,
    segment: SegmentSpec,
) -> list[np.ndarray]:
    """Render the hybrid-force dashboard frames."""
    indices = _sample_indices(len(results["time"]), segment.speedup)
    frames = []

    for idx in indices:
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.1], hspace=0.28, wspace=0.20)
        ax_robot = fig.add_subplot(gs[:, 0])
        ax_force = fig.add_subplot(gs[0, 1])
        ax_xy = fig.add_subplot(gs[1, 1])

        mj_data.qpos[:NUM_JOINTS] = results["q"][idx]
        mj_data.qvel[:NUM_JOINTS] = results["qd"][idx]
        mujoco.mj_forward(mj_model, mj_data)
        _draw_mujoco_projection(ax_robot, mj_model, mj_data, view="side", title="MuJoCo scene")
        ax_robot.scatter([xy_des[0]], [TABLE_SURFACE_Z], s=90, marker="x", color="#111827", zorder=6)
        ax_robot.set_xlim(-0.35, 0.85)
        ax_robot.set_ylim(-0.05, 0.95)
        ax_robot.set_xlabel("X (m)")
        ax_robot.set_ylabel("Z (m)")
        fig.text(0.07, 0.93, segment.title, fontsize=18, fontweight="bold", color="#111827")

        phase_name = _phase_label_force(int(results["phase"][idx]))
        xy_err_mm = float(np.linalg.norm(results["ee_pos"][idx, :2] - xy_des) * 1000.0)
        status = (
            "PASS"
            if metrics["contact_reached"]
            and abs(metrics["mean_force"] - f_desired) < 1.0
            and metrics["pct_in_band"] > 90.0
            and metrics["max_xy_err_mm"] < 5.0
            else "FAIL"
        )
        ax_robot.text(
            0.03,
            0.97,
            "\n".join(
                [
                    f"t = {results['time'][idx]:.2f} s",
                    f"Phase = {phase_name}",
                    f"Force Z = {results['force_z'][idx]:.2f} N",
                    f"XY error = {xy_err_mm:.2f} mm",
                    f"Status: {status}",
                ]
            ),
            transform=ax_robot.transAxes,
            va="top",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
        )

        ax_force.plot(results["time"][: idx + 1], results["force_z"][: idx + 1], color="#7c3aed", linewidth=2)
        ax_force.axhline(f_desired, color="#dc2626", linestyle="--", linewidth=1.0)
        ax_force.axhline(f_desired - 1.0, color="#f97316", linestyle=":", linewidth=1.0)
        ax_force.axhline(f_desired + 1.0, color="#f97316", linestyle=":", linewidth=1.0)
        ax_force.axvline(results["time"][idx], color="#111827", linestyle="--", linewidth=1.0)
        ax_force.set_title("Normal contact force")
        ax_force.set_ylabel("N")
        ax_force.set_xlim(results["time"][0], results["time"][-1])
        _style_axis(ax_force)

        ax_xy.plot(results["ee_pos"][: idx + 1, 0], results["ee_pos"][: idx + 1, 1], color="#0f766e", linewidth=2)
        ax_xy.scatter([xy_des[0]], [xy_des[1]], color="#111827", marker="x", s=80)
        ax_xy.scatter(results["ee_pos"][idx, 0], results["ee_pos"][idx, 1], color="#dc2626", s=70)
        ax_xy.set_title("XY contact tracking")
        ax_xy.set_xlabel("X (m)")
        ax_xy.set_ylabel("Y (m)")
        ax_xy.set_xlim(0.25, 0.55)
        ax_xy.set_ylim(-0.15, 0.15)
        _style_axis(ax_xy)
        ax_xy.text(
            0.02,
            0.03,
            "\n".join(
                [
                    f"Settled mean force = {metrics['mean_force']:.2f} N",
                    f"In-band samples = {metrics['pct_in_band']:.1f} %",
                    f"Max settled XY err = {metrics['max_xy_err_mm']:.2f} mm",
                ]
            ),
            transform=ax_xy.transAxes,
            fontsize=11,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
        )

        frames.append(_frame_rgb(fig))
        plt.close(fig)

    return frames


def render_line_frames(
    results: dict[str, np.ndarray],
    metrics: dict[str, float],
    mj_model,
    mj_data,
    xy_start: np.ndarray,
    xy_end: np.ndarray,
    f_desired: float,
    segment: SegmentSpec,
) -> list[np.ndarray]:
    """Render the constant-force line-trace dashboard frames."""
    indices = _sample_indices(len(results["time"]), segment.speedup)
    frames = []

    for idx in indices:
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.15], hspace=0.28, wspace=0.22)
        ax_xy = fig.add_subplot(gs[:, 0])
        ax_force = fig.add_subplot(gs[0, 1])
        ax_robot = fig.add_subplot(gs[1, 1])

        phase_name = _phase_label_line(int(results["phase"][idx]))
        actual_xy = results["ee_pos"][: idx + 1, :2]
        desired_xy = results["xy_des"][: idx + 1]
        ax_xy.plot(desired_xy[:, 0], desired_xy[:, 1], color="#cbd5e1", linewidth=4, label="Desired")
        ax_xy.plot(actual_xy[:, 0], actual_xy[:, 1], color="#2563eb", linewidth=2.5, label="Actual")
        ax_xy.scatter([xy_start[0], xy_end[0]], [xy_start[1], xy_end[1]], marker="x", s=90, color="#111827")
        ax_xy.scatter(actual_xy[-1, 0], actual_xy[-1, 1], s=80, color="#dc2626")
        ax_xy.set_title(segment.title, fontsize=18, fontweight="bold")
        ax_xy.set_xlabel("X (m)")
        ax_xy.set_ylabel("Y (m)")
        ax_xy.set_xlim(0.30, 0.52)
        ax_xy.set_ylim(-0.08, 0.08)
        _style_axis(ax_xy)
        ax_xy.legend(loc="upper left")
        ax_xy.text(
            0.03,
            0.03,
            "\n".join(
                [
                    f"t = {results['time'][idx]:.2f} s",
                    f"Phase = {phase_name}",
                    f"Force Z = {results['force_z'][idx]:.2f} N",
                    f"Trace in-band = {metrics['pct_in_band']:.1f} %",
                    f"Max trace XY err = {metrics['max_xy_err_mm']:.2f} mm",
                ]
            ),
            transform=ax_xy.transAxes,
            fontsize=11,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
        )

        ax_force.plot(results["time"][: idx + 1], results["force_z"][: idx + 1], color="#7c3aed", linewidth=2)
        ax_force.axhline(f_desired, color="#dc2626", linestyle="--", linewidth=1.0)
        ax_force.axhline(f_desired - 1.0, color="#f97316", linestyle=":", linewidth=1.0)
        ax_force.axhline(f_desired + 1.0, color="#f97316", linestyle=":", linewidth=1.0)
        ax_force.axvline(results["time"][idx], color="#111827", linestyle="--", linewidth=1.0)
        ax_force.set_title("Force regulation during trace")
        ax_force.set_ylabel("N")
        ax_force.set_xlim(results["time"][0], results["time"][-1])
        _style_axis(ax_force)

        mj_data.qpos[:NUM_JOINTS] = results["q"][idx]
        mj_data.qvel[:NUM_JOINTS] = results["qd"][idx]
        mujoco.mj_forward(mj_model, mj_data)
        _draw_mujoco_projection(ax_robot, mj_model, mj_data, view="side", title="MuJoCo scene")
        ax_robot.set_xlim(-0.35, 0.85)
        ax_robot.set_ylim(-0.05, 0.95)
        ax_robot.set_xlabel("X (m)")
        ax_robot.set_ylabel("Z (m)")

        frames.append(_frame_rgb(fig))
        plt.close(fig)

    return frames


def _summary_frames(
    dynamics_summary: dict[str, float],
    gravity_metrics: dict[str, float],
    hybrid_metrics: dict[str, float],
    line_metrics: dict[str, float],
) -> list[np.ndarray]:
    """Create an outro summary card with the final pass/fail metrics."""
    status = (
        gravity_metrics["pass_hold"]
        and gravity_metrics["pass_settle"]
        and hybrid_metrics["contact_reached"]
        and abs(hybrid_metrics["mean_force"] - 5.0) < 1.0
        and hybrid_metrics["pct_in_band"] > 90.0
        and hybrid_metrics["max_xy_err_mm"] < 5.0
        and line_metrics["trace_started"]
        and line_metrics["pct_in_band"] > 90.0
        and line_metrics["max_xy_err_mm"] < 5.0
    )

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="#0f172a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.05, 0.88, "Lab 3 sign-off summary", fontsize=30, fontweight="bold", color="white")
    ax.text(
        0.05,
        0.76,
        f"Dynamics parity: gravity {dynamics_summary['max_gravity_err']:.3e}, mass {dynamics_summary['max_mass_err']:.3e}",
        fontsize=16,
        color="#d1d5db",
    )
    ax.text(
        0.05,
        0.63,
        f"Gravity compensation: max error {gravity_metrics['max_err_deg']:.3f} deg, final speed {gravity_metrics['final_vel_deg']:.3f} deg/s",
        fontsize=16,
        color="#d1d5db",
    )
    ax.text(
        0.05,
        0.52,
        f"Hybrid force control: mean force {hybrid_metrics['mean_force']:.2f} N, in-band {hybrid_metrics['pct_in_band']:.1f} %, max XY err {hybrid_metrics['max_xy_err_mm']:.2f} mm",
        fontsize=16,
        color="#d1d5db",
    )
    ax.text(
        0.05,
        0.41,
        f"Line trace: in-band {line_metrics['pct_in_band']:.1f} %, max XY err {line_metrics['max_xy_err_mm']:.2f} mm",
        fontsize=16,
        color="#d1d5db",
    )
    ax.text(
        0.05,
        0.24,
        f"Overall status: {'PASS' if status else 'FAIL'}",
        fontsize=24,
        fontweight="bold",
        color="#22c55e" if status else "#ef4444",
    )
    ax.text(
        0.05,
        0.14,
        f"Saved artifact: {VIDEO_PATH.name}",
        fontsize=15,
        color="#cbd5e1",
    )

    frame = _frame_rgb(fig)
    plt.close(fig)
    return [frame] * (FPS * 3)


def record_video(output_path: Path = VIDEO_PATH) -> Path:
    """Run the validation sims and write the headless dashboard video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Computing dynamics parity summary...")
    dynamics_summary = compute_dynamics_summary()

    print("Running gravity-compensation validation...")
    gravity_hold_results = run_gravity_comp_sim(duration=5.0)
    gravity_perturb_results = run_gravity_comp_sim(
        duration=5.0,
        perturb_at=1.0,
        perturb_torque=np.array([0.0, 20.0, 10.0, 5.0, 0.0, 0.0]),
        perturb_duration=0.2,
    )
    gravity_metrics = compute_gravity_metrics(gravity_hold_results, gravity_perturb_results)

    print("Running hybrid force-control validation...")
    force_target_xy = np.array([0.40, 0.0])
    force_gains = HybridGains(F_desired=5.0)
    hybrid_results = run_hybrid_force_sim(
        xy_des=force_target_xy,
        gains=force_gains,
        approach_height=DEFAULT_APPROACH_HEIGHT,
        duration=8.0,
    )
    hybrid_metrics = compute_hybrid_metrics(hybrid_results, force_target_xy, force_gains.F_desired)

    print("Running constant-force line trace validation...")
    line_start = np.array([0.40, 0.0])
    line_end = np.array([0.45, 0.0])
    line_results = run_line_trace(
        xy_start=line_start,
        xy_end=line_end,
        gains=HybridGains(K_p=2000.0, K_d=100.0, F_desired=5.0),
        approach_height=DEFAULT_APPROACH_HEIGHT,
        settle_time=1.5,
        trace_duration=6.0,
        post_time=0.5,
    )
    line_metrics = compute_line_metrics(line_results, 5.0)

    gravity_mj_model, gravity_mj_data = load_mujoco_model()
    table_mj_model, table_mj_data = load_mujoco_model(SCENE_TABLE_PATH)

    frames: list[np.ndarray] = []
    frames.extend(_intro_frames(dynamics_summary))
    frames.extend(
        render_gravity_frames(
            gravity_perturb_results,
            gravity_metrics,
            gravity_mj_model,
            gravity_mj_data,
            SegmentSpec(title="Gravity Compensation", speedup=1.5),
        )
    )
    frames.extend(
        render_hybrid_frames(
            hybrid_results,
            hybrid_metrics,
            table_mj_model,
            table_mj_data,
            force_target_xy,
            force_gains.F_desired,
            SegmentSpec(title="Hybrid Force Control", speedup=1.5),
        )
    )
    frames.extend(
        render_line_frames(
            line_results,
            line_metrics,
            table_mj_model,
            table_mj_data,
            line_start,
            line_end,
            5.0,
            SegmentSpec(title="Constant-Force Line Trace", speedup=2.0),
        )
    )
    frames.extend(_summary_frames(dynamics_summary, gravity_metrics, hybrid_metrics, line_metrics))

    print(f"Writing video to {output_path} ...")
    with imageio.get_writer(output_path, fps=FPS, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)

    print("Video saved.")
    return output_path


def main() -> None:
    record_video()


if __name__ == "__main__":
    main()
