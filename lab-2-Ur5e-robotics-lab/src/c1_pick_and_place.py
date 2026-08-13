"""Phase 4.1 — Integrated Pick-and-Place Pipeline.

Combines all previous modules into a complete pick-and-place demo:
  1. IK to solve for target EE poses
  2. Quintic trajectory generation between waypoints
  3. Computed torque control for tracking
  4. Constraint checking throughout execution
  5. Performance logging and metrics

Run:
    python3 lab-2-Ur5e-robotics-lab/src/c1_pick_and_place.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ur5e_common import (
    DOCS_DIR,
    NUM_JOINTS,
    Q_HOME,
    load_pinocchio_model,
)
from mujoco_sim import UR5eSimulator
from a4_inverse_kinematics import ik_damped_least_squares
from b1_trajectory_generation import quintic_trajectory, JointTrajectoryPoint
from b2_control_hierarchy import computed_torque_control
from b3_constraints import (
    check_joint_limits,
    saturate_torques,
    self_collision_score,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PipelineLog:
    """Log of the full pipeline execution."""
    times: list[float] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    ee_errors: list[float] = field(default_factory=list)
    torque_norms: list[float] = field(default_factory=list)
    joint_margins: list[float] = field(default_factory=list)
    collision_scores: list[float] = field(default_factory=list)


@dataclass
class WaypointPose:
    """A target pose for the EE."""
    name: str
    position: np.ndarray  # (3,)
    rotation: np.ndarray  # (3,3)


# ---------------------------------------------------------------------------
# IK-based waypoint solving
# ---------------------------------------------------------------------------


def solve_waypoints(
    waypoints: list[WaypointPose],
    pin_model, pin_data, ee_fid: int,
    q_init: np.ndarray,
) -> list[np.ndarray]:
    """Solve IK for a sequence of waypoints.

    Args:
        waypoints: List of target EE poses.
        pin_model, pin_data: Pinocchio model.
        ee_fid: End-effector frame ID.
        q_init: Initial joint configuration.

    Returns:
        List of joint configurations (one per waypoint, plus initial).
    """
    solutions = [q_init.copy()]
    q_current = q_init.copy()

    for wp in waypoints:
        result = ik_damped_least_squares(
            pin_model, pin_data, ee_fid,
            wp.position, wp.rotation, q_current,
            max_iter=500, tol=1e-4,
        )
        if not result.success:
            print(f"    WARNING: IK failed for '{wp.name}' "
                  f"(err={result.final_error:.6f}), using best attempt")
        solutions.append(result.q.copy())
        q_current = result.q.copy()

    return solutions


# ---------------------------------------------------------------------------
# Trajectory generation through waypoints
# ---------------------------------------------------------------------------


def generate_pipeline_trajectory(
    joint_waypoints: list[np.ndarray],
    segment_duration: float = 2.0,
    dt: float = 0.002,
) -> list[JointTrajectoryPoint]:
    """Generate a multi-segment quintic trajectory through waypoints.

    Args:
        joint_waypoints: List of joint configurations.
        segment_duration: Duration per segment (seconds).
        dt: Timestep (seconds).

    Returns:
        List of trajectory points.
    """
    all_points = []
    t_offset = 0.0

    for i in range(len(joint_waypoints) - 1):
        seg = quintic_trajectory(
            joint_waypoints[i], joint_waypoints[i + 1],
            segment_duration, dt,
        )
        for pt in seg:
            all_points.append(JointTrajectoryPoint(
                t=pt.t + t_offset, q=pt.q, qd=pt.qd, qdd=pt.qdd,
            ))
        t_offset += segment_duration

    return all_points


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def run_pipeline(
    name: str,
    waypoints: list[WaypointPose],
    pin_model, pin_data, ee_fid: int,
    segment_duration: float = 2.0,
) -> PipelineLog:
    """Run the full pick-and-place pipeline.

    Steps:
      1. Solve IK for all waypoints
      2. Generate trajectory
      3. Execute with computed torque control in MuJoCo
      4. Log metrics

    Args:
        name: Pipeline name for display.
        waypoints: Target EE poses.
        pin_model, pin_data, ee_fid: Pinocchio model.
        segment_duration: Duration per segment.

    Returns:
        PipelineLog with execution metrics.
    """
    print(f"\n  --- {name} ---")

    # Step 1: IK
    print("  Step 1: Solving IK for waypoints...")
    q_init = Q_HOME.copy()
    joint_wps = solve_waypoints(waypoints, pin_model, pin_data, ee_fid, q_init)
    print(f"    Solved {len(waypoints)} waypoints")

    # Step 2: Trajectory
    print("  Step 2: Generating trajectory...")
    traj = generate_pipeline_trajectory(joint_wps, segment_duration)
    print(f"    {len(traj)} points, {traj[-1].t:.1f}s duration")

    # Step 3: Execute in MuJoCo
    print("  Step 3: Executing in MuJoCo...")
    sim = UR5eSimulator()
    sim.set_qpos(traj[0].q)
    sim.data.qvel[:NUM_JOINTS] = traj[0].qd
    mujoco.mj_forward(sim.model, sim.data)

    Kp = np.array([2000, 2000, 2000, 500, 500, 500], dtype=float)
    Kd = np.array([400, 400, 400, 100, 100, 100], dtype=float)

    log = PipelineLog()
    traj_idx = 0

    while sim.time < traj[-1].t + 0.01:
        # Find current trajectory point
        while traj_idx < len(traj) - 1 and traj[traj_idx + 1].t <= sim.time:
            traj_idx += 1
        pt = traj[traj_idx]

        # Determine phase
        current_phase_idx = min(
            int(sim.time / segment_duration),
            len(waypoints) - 1,
        )
        if current_phase_idx < len(waypoints):
            phase_name = waypoints[current_phase_idx].name
        else:
            phase_name = "done"

        q = sim.data.qpos[:NUM_JOINTS].copy()
        qd = sim.data.qvel[:NUM_JOINTS].copy()

        # Computed torque control
        tau = computed_torque_control(
            q, qd, pt.q, pt.qd, pt.qdd,
            pin_model, pin_data, Kp, Kd,
        )
        tau = saturate_torques(tau, np.full(NUM_JOINTS, 150.0))

        sim.set_ctrl(tau)
        sim.step()

        # Log metrics
        state = sim.get_state()

        # EE error
        pin.forwardKinematics(pin_model, pin_data, pt.q)
        pin.updateFramePlacements(pin_model, pin_data)
        ee_des = pin_data.oMf[ee_fid].translation.copy()
        ee_err = np.linalg.norm(ee_des - state.ee_pos)

        # Constraints
        _, margin = check_joint_limits(q)

        log.times.append(sim.time)
        log.phases.append(phase_name)
        log.ee_errors.append(ee_err)
        log.torque_norms.append(np.linalg.norm(tau))
        log.joint_margins.append(float(np.min(margin)))
        # Only compute collision score periodically (expensive)
        if len(log.times) % 50 == 0:
            log.collision_scores.append(self_collision_score(q))
        else:
            log.collision_scores.append(
                log.collision_scores[-1] if log.collision_scores else 0.5
            )

    return log


# ---------------------------------------------------------------------------
# Pre-defined demos
# ---------------------------------------------------------------------------


def make_pick_place_waypoints(pin_model, pin_data, ee_fid: int) -> list[WaypointPose]:
    """Create pick-and-place waypoints in the workspace.

    Sequence: approach -> pick -> lift -> move -> place -> retreat.
    """
    pin.forwardKinematics(pin_model, pin_data, Q_HOME)
    pin.updateFramePlacements(pin_model, pin_data)
    R_home = pin_data.oMf[ee_fid].rotation.copy()

    return [
        WaypointPose("approach", np.array([0.4, 0.15, 0.55]), R_home),
        WaypointPose("pick", np.array([0.4, 0.15, 0.44]), R_home),
        WaypointPose("lift", np.array([0.4, 0.15, 0.60]), R_home),
        WaypointPose("move", np.array([0.5, -0.15, 0.60]), R_home),
        WaypointPose("place", np.array([0.5, -0.15, 0.44]), R_home),
        WaypointPose("retreat", np.array([0.5, -0.15, 0.60]), R_home),
    ]


def make_circle_waypoints(pin_model, pin_data, ee_fid: int,
                          center: np.ndarray | None = None,
                          radius: float = 0.08,
                          n_points: int = 8) -> list[WaypointPose]:
    """Create circular trajectory waypoints."""
    if center is None:
        center = np.array([0.4, 0.0, 0.50])

    pin.forwardKinematics(pin_model, pin_data, Q_HOME)
    pin.updateFramePlacements(pin_model, pin_data)
    R_home = pin_data.oMf[ee_fid].rotation.copy()

    waypoints = []
    for i in range(n_points + 1):
        angle = 2.0 * np.pi * i / n_points
        pos = center.copy()
        pos[0] += radius * np.cos(angle)
        pos[1] += radius * np.sin(angle)
        waypoints.append(WaypointPose(f"circle_{i}", pos, R_home))

    return waypoints


# ---------------------------------------------------------------------------
# Metrics and CSV output
# ---------------------------------------------------------------------------


def print_metrics(name: str, log: PipelineLog) -> None:
    """Print summary metrics for a pipeline run."""
    ee_arr = np.array(log.ee_errors)
    tau_arr = np.array(log.torque_norms)
    margin_arr = np.array(log.joint_margins)
    col_arr = np.array(log.collision_scores)

    print(f"\n  Metrics for '{name}':")
    print(f"    Duration:           {log.times[-1]:.2f} s")
    print(f"    RMS EE error:       {np.sqrt(np.mean(ee_arr**2)):.6f} m")
    print(f"    Max EE error:       {np.max(ee_arr):.6f} m")
    print(f"    Mean torque norm:   {np.mean(tau_arr):.2f} Nm")
    print(f"    Max torque norm:    {np.max(tau_arr):.2f} Nm")
    print(f"    Min joint margin:   {np.min(margin_arr):.4f} rad")
    print(f"    Min collision dist: {np.min(col_arr):.4f} m")


def save_log_csv(name: str, log: PipelineLog, path: Path) -> Path:
    """Save pipeline log to CSV.

    Args:
        name: Pipeline name.
        log: PipelineLog data.
        path: Output CSV path.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "phase", "ee_error", "torque_norm",
                          "joint_margin", "collision_score"])
        for i in range(len(log.times)):
            writer.writerow([
                f"{log.times[i]:.6f}",
                log.phases[i],
                f"{log.ee_errors[i]:.8f}",
                f"{log.torque_norms[i]:.4f}",
                f"{log.joint_margins[i]:.4f}",
                f"{log.collision_scores[i]:.4f}",
            ])
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the integrated pick-and-place pipeline demos."""
    print("=" * 72)
    print("Phase 4.1 — Integrated Pick-and-Place Pipeline")
    print("=" * 72)

    pin_model, pin_data, ee_fid = load_pinocchio_model()

    # Demo 1: Pick and place
    print("\n" + "=" * 72)
    print("Demo 1: Pick and Place")
    print("=" * 72)
    pp_waypoints = make_pick_place_waypoints(pin_model, pin_data, ee_fid)
    pp_log = run_pipeline("pick_place", pp_waypoints,
                          pin_model, pin_data, ee_fid,
                          segment_duration=1.5)
    print_metrics("pick_place", pp_log)

    csv_path = DOCS_DIR / "c1_pick_place_log.csv"
    save_log_csv("pick_place", pp_log, csv_path)
    print(f"\n  Log saved: {csv_path}")

    # Demo 2: Circle tracking
    print("\n" + "=" * 72)
    print("Demo 2: Circle Tracking")
    print("=" * 72)
    circle_wps = make_circle_waypoints(pin_model, pin_data, ee_fid)
    circle_log = run_pipeline("circle", circle_wps,
                              pin_model, pin_data, ee_fid,
                              segment_duration=1.0)
    print_metrics("circle", circle_log)

    csv_path = DOCS_DIR / "c1_circle_log.csv"
    save_log_csv("circle", circle_log, csv_path)
    print(f"\n  Log saved: {csv_path}")

    # Summary
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"\n  {'Demo':<20} {'RMS EE err':>12} {'Max |τ|':>10} "
          f"{'Min margin':>12} {'Min col dist':>14}")
    print("  " + "-" * 70)
    for lbl, log in [("pick_place", pp_log), ("circle", circle_log)]:
        ee_rms = np.sqrt(np.mean(np.array(log.ee_errors) ** 2))
        max_tau = np.max(log.torque_norms)
        min_margin = np.min(log.joint_margins)
        min_col = np.min(log.collision_scores)
        print(f"  {lbl:<20} {ee_rms:>10.6f} m {max_tau:>8.2f} Nm "
              f"{min_margin:>10.4f} rad {min_col:>12.4f} m")

    # Save summary CSV
    summary_path = DOCS_DIR / "c1_metrics.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["demo", "rms_ee_error", "max_torque", "min_margin",
                          "min_collision_dist"])
        for lbl, log in [("pick_place", pp_log), ("circle", circle_log)]:
            writer.writerow([
                lbl,
                f"{np.sqrt(np.mean(np.array(log.ee_errors)**2)):.8f}",
                f"{np.max(log.torque_norms):.4f}",
                f"{np.min(log.joint_margins):.4f}",
                f"{np.min(log.collision_scores):.4f}",
            ])
    print(f"\n  Summary saved: {summary_path}")

    print("\n" + "=" * 72)
    print("Phase 4.1 — Pick-and-Place Pipeline: COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
