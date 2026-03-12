#!/usr/bin/env python3
"""C1: Draw a Cartesian square — Computed Torque Control + MuJoCo viewer.

Integrates all previous modules:
  A2  Forward Kinematics   fk_endeffector
  A3  Jacobian             analytic_jacobian, J^-1 for velocity mapping
  A4  Inverse Kinematics   analytic IK with continuous branch selection
  A5  Dynamics             M(q), qfrc_bias from MuJoCo
  B1  Trajectory Gen       quintic_profile for smooth Cartesian paths
  B2  PD Control           Kp/Kd error terms inside computed torque law

Architecture
============
1. Path Planning     – 4 corners of a square, quintic Cartesian interpolation
                       per side gives smooth x(t), xdot(t), xddot(t).
2. IK + Jacobian     – Analytic IK -> q_des;  J^-1 * xdot -> qd_des;
                       central differences on qd_des -> qdd_des.
3. Computed Torque   – tau = M(q) * [qdd_des + Kp*e + Kd*edot] + bias - passive
                       Perfectly linearises the plant so error dynamics are
                       critically damped: e'' + Kd*e' + Kp*e = 0.
4. MuJoCo Viewer     – launch_passive with real-time trail visualisation and
                       desired-square overlay drawn via user_scn geoms.

Usage
-----
    cd src && python c1_draw_square.py
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ modules are importable regardless of cwd
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

import mujoco
import mujoco.viewer
import numpy as np

from a3_jacobian import analytic_jacobian, fk_endeffector
from a4_inverse_kinematics import analytic_ik
from b1_trajectory_generation import quintic_profile

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
MODEL_PATH = str(_SRC_DIR.parent / "models" / "two_link_torque.xml")

# Square geometry — centred well inside the reachable workspace
# Max reach ≈ 0.615 m; corners range from ~0.354 to ~0.495 m from origin.
SQUARE_CENTER = (0.30, 0.30)
SQUARE_SIDE = 0.10          # metres
SEGMENT_DURATION = 2.0      # seconds per side → 8 s total

# Computed-torque PD gains (critically damped, wn = 20 rad/s)
#   Kp = wn^2 = 400,  Kd = 2*wn = 40
KP = np.array([400.0, 400.0])
KD = np.array([40.0, 40.0])

TORQUE_LIMIT = 5.0          # N·m  (must match actuator ctrlrange in XML)

# Visualisation
TRAIL_DECIMATE = 5           # record trail point every N sim steps
TRAIL_COLOR = np.array([1.0, 0.15, 0.15, 0.85], dtype=np.float32)
PATH_COLOR = np.array([0.2, 0.80, 0.20, 0.50], dtype=np.float32)
EE_MARKER_COLOR = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class TrajPoint:
    """Single timestep of the desired trajectory."""

    t: float
    # Cartesian
    x: float
    y: float
    xd: float
    yd: float
    xdd: float
    ydd: float
    # Joint-space
    q1: float
    q2: float
    q1d: float
    q2d: float
    q1dd: float = 0.0
    q2dd: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Utility — IK branch selection & Jacobian inverse
# ═══════════════════════════════════════════════════════════════════════════
def _choose_ik(target: tuple[float, float],
               prev: tuple[float, float] | None) -> tuple[float, float]:
    """Pick the IK branch closest to *prev*, preferring elbow-down (q2 < 0).

    Elbow-down keeps the arm away from the base platform geoms and avoids
    the self-collision that occurs with large positive q2.
    """
    solutions = analytic_ik(target)
    if prev is None:
        # First point — prefer elbow-down (negative theta2)
        best = min(solutions, key=lambda s: s.theta2)
        return (best.theta1, best.theta2)
    best = min(
        solutions,
        key=lambda s: math.hypot(s.theta1 - prev[0], s.theta2 - prev[1]),
    )
    return (best.theta1, best.theta2)


def _jinv(q1: float, q2: float):
    """Invert the 2×2 analytic Jacobian.  Returns None near singularity."""
    j = analytic_jacobian(q1, q2)
    det = j[0][0] * j[1][1] - j[0][1] * j[1][0]
    if abs(det) < 1e-8:
        return None
    d = 1.0 / det
    return ((j[1][1] * d, -j[0][1] * d), (-j[1][0] * d, j[0][0] * d))


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Trajectory generation — quintic Cartesian square
# ═══════════════════════════════════════════════════════════════════════════
def build_trajectory(dt: float) -> list[TrajPoint]:
    """Build the full square trajectory.

    Each side is interpolated in Cartesian space with quintic profiles so
    that position, velocity **and** acceleration are zero at the corners
    (smooth start/stop on every edge).

    Joint-space quantities:
        q_des   ← analytic IK with continuous-branch selection
        qd_des  ← J^{-1} · [xd, yd]
        qdd_des ← central finite differences on qd_des
    """
    half = SQUARE_SIDE / 2.0
    cx, cy = SQUARE_CENTER
    corners = [
        (cx - half, cy - half),   # bottom-left
        (cx + half, cy - half),   # bottom-right
        (cx + half, cy + half),   # top-right
        (cx - half, cy + half),   # top-left
    ]

    traj: list[TrajPoint] = []
    prev_q: tuple[float, float] | None = None
    n_per_seg = int(round(SEGMENT_DURATION / dt))

    for seg_i in range(4):
        p0 = corners[seg_i]
        p1 = corners[(seg_i + 1) % 4]
        first = 1 if seg_i > 0 else 0           # avoid duplicate corner point

        for k in range(first, n_per_seg + 1):
            t_local = min(k * dt, SEGMENT_DURATION)  # clamp for fp safety

            # Quintic per Cartesian axis — guarantees zero vel/accel at ends
            x, xd, xdd = quintic_profile(p0[0], p1[0], SEGMENT_DURATION, t_local)
            y, yd, ydd = quintic_profile(p0[1], p1[1], SEGMENT_DURATION, t_local)

            # IK — pick the branch closest to the previous solution
            q1, q2 = _choose_ik((x, y), prev_q)
            prev_q = (q1, q2)

            # Joint velocity via Jacobian inverse:  qdot = J^{-1} · xdot
            ji = _jinv(q1, q2)
            if ji is not None:
                q1d = ji[0][0] * xd + ji[0][1] * yd
                q2d = ji[1][0] * xd + ji[1][1] * yd
            else:
                # Near singularity — quintic ensures vel≈0 at corners anyway
                q1d = q2d = 0.0

            traj.append(TrajPoint(
                t=seg_i * SEGMENT_DURATION + t_local,
                x=x, y=y, xd=xd, yd=yd, xdd=xdd, ydd=ydd,
                q1=q1, q2=q2, q1d=q1d, q2d=q2d,
            ))

    # --- Joint accelerations via central finite differences ---------------
    n = len(traj)
    for i in range(n):
        if i == 0:
            traj[i].q1dd = (traj[1].q1d - traj[0].q1d) / dt
            traj[i].q2dd = (traj[1].q2d - traj[0].q2d) / dt
        elif i == n - 1:
            traj[i].q1dd = (traj[-1].q1d - traj[-2].q1d) / dt
            traj[i].q2dd = (traj[-1].q2d - traj[-2].q2d) / dt
        else:
            traj[i].q1dd = (traj[i + 1].q1d - traj[i - 1].q1d) / (2.0 * dt)
            traj[i].q2dd = (traj[i + 1].q2d - traj[i - 1].q2d) / (2.0 * dt)

    return traj


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Visualisation helpers — user_scn geoms
# ═══════════════════════════════════════════════════════════════════════════
def _add_sphere(scn, pos, size: float, rgba):
    """Append a visual-only sphere to the scene."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, 0.0, 0.0]),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    scn.ngeom += 1


def _add_line(scn, a, b, width: float, rgba):
    """Draw a thin capsule line between two 3-D points."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_connector(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
    )
    scn.geoms[scn.ngeom].rgba = rgba
    scn.ngeom += 1


def _draw_desired_square(scn):
    """Overlay the target square outline (green)."""
    half = SQUARE_SIDE / 2.0
    cx, cy = SQUARE_CENTER
    c = [
        np.array([cx - half, cy - half, 0.0]),
        np.array([cx + half, cy - half, 0.0]),
        np.array([cx + half, cy + half, 0.0]),
        np.array([cx - half, cy + half, 0.0]),
    ]
    for i in range(4):
        _add_line(scn, c[i], c[(i + 1) % 4], 0.002, PATH_COLOR)


def _draw_trail(scn, trail: list[np.ndarray]):
    """Render the accumulated end-effector trail (red spheres)."""
    for pos in trail:
        _add_sphere(scn, pos, 0.004, TRAIL_COLOR)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  End-effector position reader
# ═══════════════════════════════════════════════════════════════════════════
_SITE_ID: int | None = None


def _ee_pos(model, data) -> np.ndarray:
    """Read the end_effector site position from MuJoCo."""
    global _SITE_ID
    if _SITE_ID is None:
        _SITE_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    return data.site_xpos[_SITE_ID].copy()


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Main simulation loop
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Load model ────────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep                          # 0.002 s

    # Disable geom collisions — the decorative base/joint geoms physically
    # overlap with link1 at certain configurations, generating unwanted
    # constraint forces.  Self-collision is irrelevant for this 2-link demo.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0

    print(f"Model       : {MODEL_PATH}")
    print(f"Timestep    : {dt} s | Torque limit: {TORQUE_LIMIT} N·m")
    print(f"Square      : center={SQUARE_CENTER}, side={SQUARE_SIDE} m")
    print(f"Segment time: {SEGMENT_DURATION} s/side  ({4 * SEGMENT_DURATION} s total)")
    print(f"Gains       : Kp={KP.tolist()}, Kd={KD.tolist()}")

    # ── Build trajectory ──────────────────────────────────────────────────
    traj = build_trajectory(dt)
    print(f"Trajectory  : {len(traj)} points, {traj[-1].t:.1f} s\n")

    # ── Initialise robot at first waypoint ────────────────────────────────
    data.qpos[0] = traj[0].q1
    data.qpos[1] = traj[0].q2
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # ── Accumulators ──────────────────────────────────────────────────────
    trail: list[np.ndarray] = []
    cart_errors: list[float] = []
    max_torque = 0.0
    seg_printed = 0

    # ── Launch viewer ─────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Camera: slightly angled top-down view of the workspace
        viewer.cam.lookat[:] = [SQUARE_CENTER[0], SQUARE_CENTER[1], 0.0]
        viewer.cam.distance = 1.0
        viewer.cam.elevation = -60.0
        viewer.cam.azimuth = 90.0

        print("Simulation running — close the viewer window to stop.\n")
        print(f"{'Step':>7}  {'Time':>6}  {'|e_cart|':>10}  {'|tau|_max':>10}  Segment")
        print("-" * 60)

        step = 0
        while viewer.is_running() and step < len(traj):
            wall_t0 = time.time()
            tp = traj[step]

            # ── Read state ────────────────────────────────────────────────
            q = data.qpos.copy()
            qd = data.qvel.copy()

            # ── Computed torque control ───────────────────────────────────
            #
            #   MuJoCo equation of motion:
            #       M · qacc + qfrc_bias = qfrc_actuator + qfrc_passive
            #
            #   We want qacc = qdd_des + Kp·e + Kd·edot  (linear error dynamics)
            #   So:
            #       ctrl = M · u + qfrc_bias − qfrc_passive
            #
            #   where u = qdd_des + Kp·(q_des − q) + Kd·(qd_des − qd)

            # Mass matrix (2×2, symmetric)
            M = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M, data.qM)

            bias = data.qfrc_bias.copy()          # Coriolis + gravity
            passive = data.qfrc_passive.copy()    # joint damping (−d·qvel)

            e = np.array([tp.q1 - q[0], tp.q2 - q[1]])
            ed = np.array([tp.q1d - qd[0], tp.q2d - qd[1]])
            qdd_ff = np.array([tp.q1dd, tp.q2dd])

            u = qdd_ff + KP * e + KD * ed
            tau = M @ u + bias - passive
            tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

            # ── Apply & step ──────────────────────────────────────────────
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            # ── Metrics ───────────────────────────────────────────────────
            ee = _ee_pos(model, data)
            err = math.hypot(tp.x - ee[0], tp.y - ee[1])
            cart_errors.append(err)
            max_torque = max(max_torque, float(np.max(np.abs(tau))))

            # ── Trail ─────────────────────────────────────────────────────
            if step % TRAIL_DECIMATE == 0:
                trail.append(ee)

            # ── Render scene overlays ─────────────────────────────────────
            viewer.user_scn.ngeom = 0
            _draw_desired_square(viewer.user_scn)
            _draw_trail(viewer.user_scn, trail)
            # Bright yellow marker at current EE position
            _add_sphere(viewer.user_scn, ee, 0.008, EE_MARKER_COLOR)

            viewer.sync()

            # ── Progress log (once per segment) ───────────────────────────
            current_seg = int(tp.t // SEGMENT_DURATION)
            if current_seg > seg_printed:
                seg_labels = ["bottom", "right", "top", "left"]
                label = seg_labels[min(seg_printed, 3)]
                print(f"{step:7d}  {tp.t:6.2f}  {err*1000:8.3f} mm  {max_torque:8.3f} Nm  {label} done")
                seg_printed = current_seg

            step += 1

            # ── Real-time pacing ──────────────────────────────────────────
            wall_elapsed = time.time() - wall_t0
            if wall_elapsed < dt:
                time.sleep(dt - wall_elapsed)

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    if cart_errors:
        rms = math.sqrt(sum(e * e for e in cart_errors) / len(cart_errors))
        print(f"\n{'=' * 60}")
        print("C1: Draw Square — Final Results")
        print(f"{'=' * 60}")
        print(f"  RMS  Cartesian error  : {rms * 1000:.3f} mm")
        print(f"  Max  Cartesian error  : {max(cart_errors) * 1000:.3f} mm")
        print(f"  Final Cartesian error : {cart_errors[-1] * 1000:.3f} mm")
        print(f"  Max torque applied    : {max_torque:.3f} N·m")
        print(f"  Trail points rendered : {len(trail)}")
        print(f"  Trajectory steps      : {len(cart_errors)}")
        print(f"{'=' * 60}")
    else:
        print("Viewer closed before simulation started.")


if __name__ == "__main__":
    main()
