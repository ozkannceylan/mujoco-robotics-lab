"""Record the Lab 3 demo video: theory metrics + MuJoCo simulation.

Uses the shared video pipeline (tools/video_producer.py) with:
  Phase 1 — Animated metrics showing dynamics parity, gravity comp, force control KPIs.
  Phase 2 — Native MuJoCo rendering of the constant-force line-tracing capstone.
  Phase 3 — ffmpeg composition with title/end cards and crossfades.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import mujoco
import numpy as np

_SRC = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC.parent.parent

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "tools"))

from a1_dynamics_fundamentals import cross_validate_gravity, cross_validate_mass_matrix
from a2_gravity_compensation import gravity_compensation_torque, run_gravity_comp_sim
from c1_force_control import DEFAULT_APPROACH_HEIGHT, HybridGains, run_hybrid_force_sim
from c2_line_trace import run_line_trace
from lab3_common import (
    DT,
    EE_SITE_NAME,
    MEDIA_DIR,
    NUM_JOINTS,
    Q_HOME,
    Q_ZEROS,
    SCENE_TABLE_PATH,
    TABLE_SURFACE_Z,
    apply_arm_torques,
    clip_torques,
    get_mj_ee_site_id,
    load_mujoco_model,
    load_pinocchio_model,
)
from video_producer import LabVideoProducer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAB_NAME = "Lab 3: Dynamics & Force Control"
OUTPUT_DIR = MEDIA_DIR
VIDEO_NAME = "lab3_demo.mp4"


# ---------------------------------------------------------------------------
# Metric computation (reuse validation logic)
# ---------------------------------------------------------------------------

def _representative_configs() -> list[np.ndarray]:
    return [
        Q_ZEROS.copy(),
        Q_HOME.copy(),
        np.array([-1.2, -1.7, 1.8, -1.5, -1.2, 0.2]),
        np.array([-0.8, -1.1, 1.1, -1.7, -1.4, 0.8]),
    ]


def compute_all_metrics() -> dict:
    """Run validation sims and collect all metrics for the video."""
    print("  Computing dynamics parity...")
    pin_model, pin_data, _ = load_pinocchio_model()
    mj_model, mj_data = load_mujoco_model()
    grav_errs, mass_errs = [], []
    for q in _representative_configs():
        grav_errs.append(cross_validate_gravity(pin_model, pin_data, mj_model, mj_data, q))
        mass_errs.append(cross_validate_mass_matrix(pin_model, pin_data, mj_model, mj_data, q))

    print("  Running gravity compensation...")
    grav_hold = run_gravity_comp_sim(duration=5.0)
    grav_perturb = run_gravity_comp_sim(
        duration=5.0,
        perturb_at=1.0,
        perturb_torque=np.array([0.0, 20.0, 10.0, 5.0, 0.0, 0.0]),
        perturb_duration=0.2,
    )
    hold_err = float(np.max(np.abs(grav_hold["q_error"])))
    final_vel = float(np.max(np.abs(grav_perturb["qd"][-100:])))

    print("  Running hybrid force control...")
    force_target_xy = np.array([0.40, 0.0])
    force_gains = HybridGains(F_desired=5.0)
    hybrid = run_hybrid_force_sim(
        xy_des=force_target_xy,
        gains=force_gains,
        approach_height=DEFAULT_APPROACH_HEIGHT,
        duration=8.0,
    )
    phase = hybrid["phase"]
    hybrid_mask = phase == 1
    if np.any(hybrid_mask):
        h_start = int(np.argmax(hybrid_mask))
        settle_idx = min(h_start + 500, len(hybrid["force_z"]) - 1)
        force_settled = hybrid["force_z"][settle_idx:]
        hybrid_mean_f = float(np.mean(force_settled))
        hybrid_in_band = float(np.mean(np.abs(force_settled - 5.0) < 1.0) * 100.0)
    else:
        hybrid_mean_f, hybrid_in_band = 0.0, 0.0

    print("  Running line trace...")
    line_start = np.array([0.40, 0.0])
    line_end = np.array([0.45, 0.0])
    line_gains = HybridGains(K_p=2000.0, K_d=100.0, F_desired=5.0)
    line = run_line_trace(
        xy_start=line_start,
        xy_end=line_end,
        gains=line_gains,
        approach_height=DEFAULT_APPROACH_HEIGHT,
        settle_time=1.5,
        trace_duration=6.0,
        post_time=0.5,
    )
    trace_mask = line["phase"] == 2
    if np.any(trace_mask):
        line_in_band = float(np.mean(np.abs(line["force_z"][trace_mask] - 5.0) < 1.0) * 100.0)
        xy_err_mm = float(
            np.max(np.linalg.norm(line["ee_pos"][trace_mask, :2] - line["xy_des"][trace_mask], axis=1)) * 1000.0
        )
    else:
        line_in_band, xy_err_mm = 0.0, float("inf")

    return {
        "max_grav_err": float(np.max(grav_errs)),
        "max_mass_err": float(np.max(mass_errs)),
        "hold_err_deg": float(np.degrees(hold_err)),
        "final_vel_deg": float(np.degrees(final_vel)),
        "hybrid_mean_f": hybrid_mean_f,
        "hybrid_in_band": hybrid_in_band,
        "line_in_band": line_in_band,
        "line_max_xy_err_mm": xy_err_mm,
        # raw data for plots
        "grav_perturb": grav_perturb,
        "hybrid": hybrid,
        "line": line,
        "line_start": line_start,
        "line_end": line_end,
    }


# ---------------------------------------------------------------------------
# Build metrics plots (4 panels for the animated metrics clip)
# ---------------------------------------------------------------------------

def build_plots(m: dict) -> list[dict]:
    """Build 4 plot specs for the animated metrics phase."""
    gp = m["grav_perturb"]
    hy = m["hybrid"]
    ln = m["line"]

    # Plot 1: Joint error during gravity comp recovery
    max_err_per_step = np.max(np.abs(np.degrees(gp["q_error"])), axis=1)
    plot_grav = {
        "title": "Gravity Comp Recovery",
        "xlabel": "Time (s)",
        "ylabel": "Max |error| (deg)",
        "data": (gp["time"], max_err_per_step),
        "color": "#58c4dd",
        "threshold": (np.degrees(0.01), "± 0.01 rad limit"),
    }

    # Plot 2: Force tracking during hybrid control
    phase_mask = hy["phase"] == 1
    if np.any(phase_mask):
        h_start = int(np.argmax(phase_mask))
        t_hybrid = hy["time"][h_start:]
        f_hybrid = hy["force_z"][h_start:]
    else:
        t_hybrid = hy["time"]
        f_hybrid = hy["force_z"]
    plot_force = {
        "title": "Hybrid Force Tracking",
        "xlabel": "Time (s)",
        "ylabel": "Force Z (N)",
        "data": (t_hybrid, f_hybrid),
        "color": "#c792ea",
        "threshold": (5.0, "F_desired = 5 N"),
    }

    # Plot 3: Line trace XY path
    trace_mask = ln["phase"] == 2
    if np.any(trace_mask):
        xy_actual = ln["ee_pos"][trace_mask, :2]
        xy_desired = ln["xy_des"][trace_mask]
    else:
        xy_actual = ln["ee_pos"][:, :2]
        xy_desired = ln["xy_des"]

    plot_xy = {
        "title": "Line Trace XY Path",
        "xlabel": "X (m)",
        "ylabel": "Y (m)",
        "type": "line",
        "data": (xy_actual[:, 0], xy_actual[:, 1]),
        "color": "#82aaff",
        "label": "Actual",
        "overlays": [
            {
                "type": "line",
                "x": xy_desired[:, 0],
                "y": xy_desired[:, 1],
                "color": "#9bde7e",
                "linewidth": 3.0,
                "linestyle": "--",
                "alpha": 0.7,
                "label": "Desired",
                "full": True,
            }
        ],
    }

    # Plot 4: Force during line trace
    if np.any(trace_mask):
        t_trace = ln["time"][trace_mask]
        f_trace = ln["force_z"][trace_mask]
    else:
        t_trace = ln["time"]
        f_trace = ln["force_z"]
    plot_trace_force = {
        "title": "Force During Line Trace",
        "xlabel": "Time (s)",
        "ylabel": "Force Z (N)",
        "data": (t_trace, f_trace),
        "color": "#ff9e66",
        "threshold": (5.0, "F_desired = 5 N"),
    }

    return [plot_grav, plot_force, plot_xy, plot_trace_force]


def build_kpi(m: dict) -> dict[str, str]:
    """Build the KPI overlay for the metrics clip."""
    return {
        "Gravity Parity": f"{m['max_grav_err']:.2e}",
        "Mass Parity": f"{m['max_mass_err']:.2e}",
        "Hold Error": f"{m['hold_err_deg']:.3f} deg",
        "Force In-Band": f"{m['hybrid_in_band']:.1f} %",
        "Trace In-Band": f"{m['line_in_band']:.1f} %",
        "Trace XY Err": f"{m['line_max_xy_err_mm']:.2f} mm",
    }


# ---------------------------------------------------------------------------
# Line-trace controller for MuJoCo simulation recording
# ---------------------------------------------------------------------------

def make_line_trace_controller(
    line_start: np.ndarray,
    line_end: np.ndarray,
    settle_time: float = 1.5,
    trace_duration: float = 6.0,
    post_time: float = 0.5,
) -> tuple:
    """Create a controller function for MuJoCo recording of line trace.

    Returns (controller_fn, total_duration).
    The controller_fn signature matches LabVideoProducer.record_simulation.
    """
    import pinocchio as pin

    pin_model, pin_data, ee_fid = load_pinocchio_model()
    mj_model_ctrl, mj_data_ctrl = load_mujoco_model(SCENE_TABLE_PATH)

    approach_height = DEFAULT_APPROACH_HEIGHT
    gains = HybridGains(K_p=2000.0, K_d=100.0, F_desired=5.0)
    target_z = TABLE_SURFACE_Z
    total_duration = settle_time + trace_duration + post_time

    # Controller state
    state = {"phase": 0, "approach_done": False}

    def controller_fn(model: mujoco.MjModel, data: mujoco.MjData, step: int) -> bool | None:
        t = step * DT
        q = data.qpos[:NUM_JOINTS].copy()
        qd = data.qvel[:NUM_JOINTS].copy()

        # FK
        pin.forwardKinematics(pin_model, pin_data, q, qd)
        pin.updateFramePlacements(pin_model, pin_data)
        ee_pos = pin_data.oMf[ee_fid].translation.copy()
        ee_vel = pin.getFrameVelocity(
            pin_model, pin_data, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).linear.copy()

        # Gravity comp
        pin.computeGeneralizedGravity(pin_model, pin_data, q)
        tau_g = pin_data.g.copy()

        # Jacobian
        J_full = pin.computeFrameJacobian(
            pin_model, pin_data, q, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = J_full[:3, :NUM_JOINTS]

        # Contact force
        ee_site_id = get_mj_ee_site_id(model)
        force_z = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == 0 or contact.geom2 == 0:
                continue
            c_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_force)
            force_z += abs(c_force[0])

        # Phase logic: 0=approach, 1=settle, 2=trace, 3=hold
        if state["phase"] == 0:
            # Descend toward table
            xy_target = line_start.copy()
            z_target = approach_height - (approach_height - target_z) * min(t / 1.0, 1.0)
            x_des = np.array([xy_target[0], xy_target[1], z_target])
            dx = x_des - ee_pos
            dxd = -ee_vel
            F = gains.K_p * dx + gains.K_d * dxd
            tau = J.T @ F + tau_g

            if ee_pos[2] < target_z + 0.002 or force_z > 0.5:
                state["phase"] = 1
                state["settle_start"] = t

        elif state["phase"] == 1:
            # Settle at start with force control
            x_des = np.array([line_start[0], line_start[1], ee_pos[2]])
            dx = x_des - ee_pos
            dx[2] = 0.0
            dxd = -ee_vel
            F_xy = gains.K_p * dx[:2] + gains.K_d * dxd[:2]
            f_err = gains.F_desired - force_z
            F_z = 15.0 * f_err - 5.0 * ee_vel[2]
            F = np.array([F_xy[0], F_xy[1], F_z])
            tau = J.T @ F + tau_g

            if t - state.get("settle_start", t) > settle_time:
                state["phase"] = 2
                state["trace_start"] = t

        elif state["phase"] == 2:
            # Trace line
            alpha = min((t - state.get("trace_start", t)) / trace_duration, 1.0)
            xy_des = line_start + alpha * (line_end - line_start)
            x_des = np.array([xy_des[0], xy_des[1], ee_pos[2]])
            dx = x_des - ee_pos
            dx[2] = 0.0
            dxd = -ee_vel
            F_xy = gains.K_p * dx[:2] + gains.K_d * dxd[:2]
            f_err = gains.F_desired - force_z
            F_z = 15.0 * f_err - 5.0 * ee_vel[2]
            F = np.array([F_xy[0], F_xy[1], F_z])
            tau = J.T @ F + tau_g

            if alpha >= 1.0:
                state["phase"] = 3
                state["hold_start"] = t

        else:
            # Hold at end
            x_des = np.array([line_end[0], line_end[1], ee_pos[2]])
            dx = x_des - ee_pos
            dx[2] = 0.0
            dxd = -ee_vel
            F_xy = gains.K_p * dx[:2] + gains.K_d * dxd[:2]
            f_err = gains.F_desired - force_z
            F_z = 15.0 * f_err - 5.0 * ee_vel[2]
            F = np.array([F_xy[0], F_xy[1], F_z])
            tau = J.T @ F + tau_g

            if t - state.get("hold_start", t) > post_time:
                return False

        tau = clip_torques(tau)
        apply_arm_torques(model, data, tau)
        return None

    return controller_fn, total_duration


def make_status_text_fn(line_start: np.ndarray, line_end: np.ndarray):
    """Create a status text overlay function for the simulation recording."""
    gains = HybridGains(K_p=2000.0, K_d=100.0, F_desired=5.0)

    def status_fn(model, data, step, current_time):
        ee_site_id = get_mj_ee_site_id(model)
        ee_pos = data.site_xpos[ee_site_id].copy()

        force_z = 0.0
        for i in range(data.ncon):
            c_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_force)
            force_z += abs(c_force[0])

        return {
            "Constant-Force Line Trace": "",
            f"Time: {current_time:.1f}s": "",
            f"Force Z: {force_z:.1f} N (target: {gains.F_desired:.0f} N)": "",
            f"EE pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]": "",
        }

    return status_fn


# ---------------------------------------------------------------------------
# Camera schedule — slow orbit during simulation
# ---------------------------------------------------------------------------

def camera_schedule(current_time: float, progress: float) -> dict:
    """Orbit the camera slowly around the workspace."""
    azimuth = 135.0 + 30.0 * progress
    elevation = -30.0 - 10.0 * progress
    return {
        "lookat": [0.42, 0.0, 0.38],
        "distance": 1.45,
        "elevation": elevation,
        "azimuth": azimuth,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Path:
    print("=" * 60)
    print("Lab 3 Demo Video — Theory + Simulation")
    print("=" * 60)

    # Step 1: Compute all metrics
    print("\n[1/3] Running validation simulations...")
    metrics = compute_all_metrics()

    # Step 2: Build video producer
    producer = LabVideoProducer(lab_name=LAB_NAME, output_dir=OUTPUT_DIR)

    # Step 3: Create metrics clip (theory presentation)
    print("\n[2/3] Creating animated metrics clip...")
    plots = build_plots(metrics)
    kpi = build_kpi(metrics)
    metrics_clip = producer.create_metrics_clip(
        plots=plots,
        kpi_overlay=kpi,
        title_text="Dynamics & Force Control Validation",
        duration_sec=12.0,
    )
    print(f"  Metrics clip saved: {metrics_clip}")

    # Step 4: Record MuJoCo simulation (line trace capstone)
    print("\n[3/3] Recording MuJoCo simulation...")
    mj_model, mj_data = load_mujoco_model(SCENE_TABLE_PATH)

    # Reset to home
    mj_data.qpos[:NUM_JOINTS] = Q_HOME.copy()
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    line_start = metrics["line_start"]
    line_end = metrics["line_end"]

    controller_fn, total_duration = make_line_trace_controller(
        line_start=line_start,
        line_end=line_end,
        settle_time=1.5,
        trace_duration=6.0,
        post_time=0.5,
    )

    # Planned trace line for visual reference
    planned_pts = np.column_stack([
        np.linspace(line_start[0], line_end[0], 50),
        np.linspace(line_start[1], line_end[1], 50),
        np.full(50, TABLE_SURFACE_Z),
    ])

    simulation_clip = producer.record_simulation(
        model=mj_model,
        data=mj_data,
        controller_fn=controller_fn,
        camera_name="orbit_45",
        playback_speed=0.3,
        trace_ee=True,
        trace_site_name=EE_SITE_NAME,
        duration_sec=total_duration + 2.0,
        camera_schedule=camera_schedule,
        planned_trace_points=planned_pts,
        planned_trace_color=(0.22, 0.92, 0.35, 0.80),
        actual_trace_color=(0.22, 0.52, 0.98, 0.90),
        status_text_fn=make_status_text_fn(line_start, line_end),
        start_hold_sec=1.5,
        end_hold_sec=2.5,
    )
    print(f"  Simulation clip saved: {simulation_clip}")

    # Step 5: Compose final video
    print("\nComposing final video...")
    final_path = producer.compose_final_video(
        metrics_clip=metrics_clip,
        simulation_clip=simulation_clip,
        title_card_text="Lab 3: Dynamics & Force Control",
        crossfade_sec=0.6,
    )
    print(f"\nDemo video saved: {final_path}")
    return final_path


if __name__ == "__main__":
    main()
