"""B3: Full pipeline demos: IK → trajectory → PD → plant."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from a3_jacobian import fk_endeffector
from a4_inverse_kinematics import analytic_ik, numeric_ik
from b1_trajectory_generation import JointSample, cartesian_trajectory, choose_continuous_ik, joint_trajectory
from b2_pd_controller import SimpleTwoLinkPlant, pd_control

PICK_PLACE_CSV = Path("docs/b3_pick_place_log.csv")
CIRCLE_CSV = Path("docs/b3_circle_log.csv")
SINGULARITY_CSV = Path("docs/b3_singularity_log.csv")
METRICS_CSV = Path("docs/b3_metrics.csv")
PICK_PLACE_GIF = Path("docs/b3_pick_place.gif")

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class PipelineSample:
    time: float
    q1: float
    q2: float
    q1_des: float
    q2_des: float
    x: float
    y: float
    x_des: float
    y_des: float
    tau1: float
    tau2: float
    err_norm: float


def save_pipeline_csv(path: Path, samples: list[PipelineSample]) -> None:
    """Write demo log to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "q1_deg", "q2_deg", "q1_des_deg", "q2_des_deg", "x", "y", "x_des", "y_des", "tau1", "tau2", "ee_error"])
        for sample in samples:
            writer.writerow(
                [
                    f"{sample.time:.6f}",
                    f"{math.degrees(sample.q1):.6f}",
                    f"{math.degrees(sample.q2):.6f}",
                    f"{math.degrees(sample.q1_des):.6f}",
                    f"{math.degrees(sample.q2_des):.6f}",
                    f"{sample.x:.10f}",
                    f"{sample.y:.10f}",
                    f"{sample.x_des:.10f}",
                    f"{sample.y_des:.10f}",
                    f"{sample.tau1:.10f}",
                    f"{sample.tau2:.10f}",
                    f"{sample.err_norm:.10f}",
                ]
            )


def trajectory_from_joint_samples(samples: list[JointSample]) -> list[tuple[float, Vector2, Vector2, Vector2]]:
    """Convert a JointSample list to the format expected by the pipeline."""
    return [
        (
            sample.time,
            (sample.theta1, sample.theta2),
            (sample.theta1_dot, sample.theta2_dot),
            (sample.x, sample.y),
        )
        for sample in samples
    ]


def follow_desired_sequence(
    desired_sequence: list[tuple[float, Vector2, Vector2, Vector2]],
    kp: Vector2 = (24.0, 18.0),
    kd: Vector2 = (6.0, 4.5),
    dt: float = 0.01,
) -> tuple[list[PipelineSample], dict[str, float]]:
    """Follow the desired joint sequence with PD control."""
    plant = SimpleTwoLinkPlant()
    if desired_sequence:
        plant.q[0], plant.q[1] = desired_sequence[0][1]

    samples: list[PipelineSample] = []
    sum_sq = 0.0
    max_err = 0.0
    for idx, (time_s, q_des, qd_des, xy_des) in enumerate(desired_sequence):
        gravity_term = plant.gravity_torque()
        torque = pd_control(q_des, qd_des, tuple(plant.q), tuple(plant.qd), kp, kd, gravity_term)
        x, y = fk_endeffector(plant.q[0], plant.q[1])
        err = math.hypot(xy_des[0] - x, xy_des[1] - y)
        sum_sq += err * err
        max_err = max(max_err, err)
        samples.append(
            PipelineSample(time_s, plant.q[0], plant.q[1], q_des[0], q_des[1], x, y, xy_des[0], xy_des[1], torque[0], torque[1], err)
        )
        if idx < len(desired_sequence) - 1:
            plant.step(torque, dt)

    rms_err = math.sqrt(sum_sq / max(1, len(samples)))
    final_err = samples[-1].err_norm if samples else 0.0
    return samples, {"rms_err": rms_err, "max_err": max_err, "final_err": final_err}


def pick_place_demo() -> tuple[list[PipelineSample], dict[str, float]]:
    """Pick-and-place-like waypoint sequence."""
    waypoints = [(0.20, 0.30), (0.40, 0.10), (0.20, 0.30)]
    time_offset = 0.0
    full_sequence: list[tuple[float, Vector2, Vector2, Vector2]] = []
    prev_angles: Vector2 | None = None

    for start_xy, end_xy in zip(waypoints[:-1], waypoints[1:]):
        start_angles = choose_continuous_ik(start_xy, prev_angles)
        end_angles = choose_continuous_ik(end_xy, start_angles)
        segment = joint_trajectory(start_angles, end_angles, duration=2.0, samples=201, mode="quintic")
        for sample in segment:
            full_sequence.append(
                (
                    time_offset + sample.time,
                    (sample.theta1, sample.theta2),
                    (sample.theta1_dot, sample.theta2_dot),
                    (sample.x, sample.y),
                )
            )
        time_offset = full_sequence[-1][0]
        prev_angles = end_angles

    samples, metrics = follow_desired_sequence(full_sequence)
    metrics["segments"] = 2.0
    return samples, metrics


def circle_tracking_demo() -> tuple[list[PipelineSample], dict[str, float]]:
    """Circular end-effector target."""
    center = (0.28, 0.20)
    radius = 0.08
    duration = 4.0
    samples_n = 401
    desired_sequence: list[tuple[float, Vector2, Vector2, Vector2]] = []
    previous_angles: Vector2 | None = None
    previous_q: Vector2 | None = None
    previous_time: float | None = None

    for idx in range(samples_n):
        t = duration * idx / (samples_n - 1)
        angle = 2.0 * math.pi * idx / (samples_n - 1)
        xy = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
        q = choose_continuous_ik(xy, previous_angles)
        if previous_q is None or previous_time is None:
            qd = (0.0, 0.0)
        else:
            dt = t - previous_time
            qd = ((q[0] - previous_q[0]) / dt, (q[1] - previous_q[1]) / dt)
        desired_sequence.append((t, q, qd, xy))
        previous_angles = q
        previous_q = q
        previous_time = t

    return follow_desired_sequence(desired_sequence, kp=(26.0, 19.0), kd=(6.0, 4.5))


def singularity_edge_demo() -> tuple[list[tuple[str, int, int, float]], dict[str, float]]:
    """Compare pinv and DLS solvers at the singularity boundary."""
    targets = [(0.46 + 0.008 * i, 0.12 - 0.005 * i) for i in range(20)]
    previous_guess = (0.0, 0.0)
    rows: list[tuple[str, int, int, float]] = []
    stats = {
        "pinv_success": 0.0,
        "dls_success": 0.0,
        "pinv_fail": 0.0,
        "dls_fail": 0.0,
    }

    for idx, target in enumerate(targets):
        pinv = numeric_ik(target, method="pinv", initial_guess=previous_guess)
        dls = numeric_ik(target, method="dls", initial_guess=previous_guess, damping=0.08, max_iters=250)
        rows.append(("pinv", idx, pinv.iterations, pinv.error_norm))
        rows.append(("dls", idx, dls.iterations, dls.error_norm))
        stats["pinv_success"] += 1.0 if pinv.success else 0.0
        stats["dls_success"] += 1.0 if dls.success else 0.0
        stats["pinv_fail"] += 0.0 if pinv.success else 1.0
        stats["dls_fail"] += 0.0 if dls.success else 1.0
        if dls.success:
            previous_guess = (dls.theta1, dls.theta2)

    return rows, stats


def save_singularity_csv(rows: list[tuple[str, int, int, float]]) -> None:
    """Save singularity solver comparison to CSV."""
    SINGULARITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SINGULARITY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "target_idx", "iterations", "error_norm"])
        for row in rows:
            writer.writerow([row[0], row[1], row[2], f"{row[3]:.10f}"])


def save_metrics_csv(metrics_rows: list[tuple[str, float, float, float]]) -> None:
    """Save demo summary metrics to CSV."""
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["demo", "rms_err", "max_err", "final_err"])
        for row in metrics_rows:
            writer.writerow([row[0], f"{row[1]:.10f}", f"{row[2]:.10f}", f"{row[3]:.10f}"])


def maybe_save_artifacts(pick_place: list[PipelineSample], circle: list[PipelineSample]) -> None:
    """Save plot and GIF artifacts if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ModuleNotFoundError:
        print("Matplotlib not found; B3 plot/video output skipped.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([s.x_des for s in pick_place], [s.y_des for s in pick_place], label="desired")
    axes[0].plot([s.x for s in pick_place], [s.y for s in pick_place], label="actual")
    axes[0].set_title("Pick-and-place path")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot([s.x_des for s in circle], [s.y_des for s in circle], label="desired")
    axes[1].plot([s.x for s in circle], [s.y for s in circle], label="actual")
    axes[1].set_title("Circle tracking path")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("docs/b3_pipeline_paths.png", dpi=120)
    plt.close()
    print("Plot saved: docs/b3_pipeline_paths.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0.1, 0.45)
    ax.set_ylim(0.05, 0.35)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    point_des, = ax.plot([], [], "ro", label="desired")
    point_act, = ax.plot([], [], "bo", label="actual")
    ax.legend()

    def update(frame_idx):
        sample = pick_place[frame_idx]
        point_des.set_data([sample.x_des], [sample.y_des])
        point_act.set_data([sample.x], [sample.y])
        return point_des, point_act

    try:
        ani = animation.FuncAnimation(fig, update, frames=len(pick_place), interval=30, blit=True)
        ani.save(PICK_PLACE_GIF, writer=animation.PillowWriter(fps=30))
        print(f"GIF saved: {PICK_PLACE_GIF}")
    except Exception:
        print("GIF save skipped: Pillow or a compatible writer not found.")
    finally:
        plt.close(fig)


def main() -> None:
    pick_place_samples, pick_place_metrics = pick_place_demo()
    circle_samples, circle_metrics = circle_tracking_demo()
    singular_rows, singular_stats = singularity_edge_demo()

    save_pipeline_csv(PICK_PLACE_CSV, pick_place_samples)
    save_pipeline_csv(CIRCLE_CSV, circle_samples)
    save_singularity_csv(singular_rows)
    save_metrics_csv(
        [
            ("pick_place", pick_place_metrics["rms_err"], pick_place_metrics["max_err"], pick_place_metrics["final_err"]),
            ("circle", circle_metrics["rms_err"], circle_metrics["max_err"], circle_metrics["final_err"]),
        ]
    )

    print("=" * 90)
    print("B3: Full Pipeline")
    print("=" * 90)
    print(
        f"Pick-place  : rms={pick_place_metrics['rms_err']:.6f} m, "
        f"max={pick_place_metrics['max_err']:.6f} m, final={pick_place_metrics['final_err']:.6f} m"
    )
    print(
        f"Circle track: rms={circle_metrics['rms_err']:.6f} m, "
        f"max={circle_metrics['max_err']:.6f} m, final={circle_metrics['final_err']:.6f} m"
    )
    print(
        f"Singularity : pinv_success={int(singular_stats['pinv_success'])}/20, "
        f"dls_success={int(singular_stats['dls_success'])}/20"
    )
    print(f"CSV saved: {PICK_PLACE_CSV}")
    print(f"CSV saved: {CIRCLE_CSV}")
    print(f"CSV saved: {SINGULARITY_CSV}")
    print(f"CSV saved: {METRICS_CSV}")
    maybe_save_artifacts(pick_place_samples, circle_samples)


if __name__ == "__main__":
    main()
