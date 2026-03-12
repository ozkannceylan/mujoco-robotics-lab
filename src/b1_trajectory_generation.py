"""B1: Trajectory generation — comparison of joint-space and Cartesian-space."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from a3_jacobian import fk_endeffector
from a4_inverse_kinematics import analytic_ik

CUBIC_CSV = Path("docs/b1_cubic_joint_traj.csv")
QUINTIC_CSV = Path("docs/b1_quintic_joint_traj.csv")
CARTESIAN_CSV = Path("docs/b1_cartesian_traj.csv")
PATH_PLOT = Path("docs/b1_trajectory_paths.png")
PROFILE_PLOT = Path("docs/b1_trajectory_profiles.png")

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class JointSample:
    time: float
    theta1: float
    theta2: float
    theta1_dot: float
    theta2_dot: float
    theta1_ddot: float
    theta2_ddot: float
    x: float
    y: float


def cubic_profile(q0: float, qf: float, duration: float, time_s: float) -> tuple[float, float, float]:
    """Cubic trajectory with zero start/end velocity."""
    delta = qf - q0
    a0 = q0
    a1 = 0.0
    a2 = 3.0 * delta / (duration * duration)
    a3 = -2.0 * delta / (duration * duration * duration)
    q = a0 + a1 * time_s + a2 * time_s * time_s + a3 * time_s * time_s * time_s
    qd = a1 + 2.0 * a2 * time_s + 3.0 * a3 * time_s * time_s
    qdd = 2.0 * a2 + 6.0 * a3 * time_s
    return q, qd, qdd


def quintic_profile(q0: float, qf: float, duration: float, time_s: float) -> tuple[float, float, float]:
    """Quintic trajectory with zero start/end velocity and acceleration."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    s = time_s / duration
    delta = qf - q0
    q = q0 + delta * (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)
    qd = delta * (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    qdd = delta * (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / (duration * duration)
    return q, qd, qdd


def joint_trajectory(
    start_angles: Vector2,
    end_angles: Vector2,
    duration: float,
    samples: int,
    mode: str,
) -> list[JointSample]:
    """Generate cubic or quintic joint-space trajectory."""
    trajectory: list[JointSample] = []
    for index in range(samples):
        time_s = duration * index / (samples - 1)
        if mode == "cubic":
            theta1, theta1_dot, theta1_ddot = cubic_profile(start_angles[0], end_angles[0], duration, time_s)
            theta2, theta2_dot, theta2_ddot = cubic_profile(start_angles[1], end_angles[1], duration, time_s)
        elif mode == "quintic":
            theta1, theta1_dot, theta1_ddot = quintic_profile(start_angles[0], end_angles[0], duration, time_s)
            theta2, theta2_dot, theta2_ddot = quintic_profile(start_angles[1], end_angles[1], duration, time_s)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        x, y = fk_endeffector(theta1, theta2)
        trajectory.append(JointSample(time_s, theta1, theta2, theta1_dot, theta2_dot, theta1_ddot, theta2_ddot, x, y))
    return trajectory


def choose_continuous_ik(target: Vector2, previous_angles: Vector2 | None) -> Vector2:
    """Pick the IK branch closest to the previous solution."""
    solutions = analytic_ik(target)
    if previous_angles is None:
        best = min(solutions, key=lambda solution: abs(solution.theta2))
        return (best.theta1, best.theta2)

    best = min(
        solutions,
        key=lambda solution: math.hypot(solution.theta1 - previous_angles[0], solution.theta2 - previous_angles[1]),
    )
    return (best.theta1, best.theta2)


def cartesian_trajectory(start_xy: Vector2, end_xy: Vector2, duration: float, samples: int) -> list[JointSample]:
    """Straight-line end-effector trajectory, solving IK at every step."""
    trajectory: list[JointSample] = []
    previous_angles: Vector2 | None = None
    previous_joint_sample: JointSample | None = None

    for index in range(samples):
        time_s = duration * index / (samples - 1)
        alpha = index / (samples - 1)
        x = start_xy[0] + (end_xy[0] - start_xy[0]) * alpha
        y = start_xy[1] + (end_xy[1] - start_xy[1]) * alpha
        theta1, theta2 = choose_continuous_ik((x, y), previous_angles)

        if previous_joint_sample is None:
            theta1_dot = 0.0
            theta2_dot = 0.0
            theta1_ddot = 0.0
            theta2_ddot = 0.0
        else:
            dt = time_s - previous_joint_sample.time
            theta1_dot = (theta1 - previous_joint_sample.theta1) / dt
            theta2_dot = (theta2 - previous_joint_sample.theta2) / dt
            theta1_ddot = (theta1_dot - previous_joint_sample.theta1_dot) / dt
            theta2_ddot = (theta2_dot - previous_joint_sample.theta2_dot) / dt

        sample = JointSample(time_s, theta1, theta2, theta1_dot, theta2_dot, theta1_ddot, theta2_ddot, x, y)
        trajectory.append(sample)
        previous_angles = (theta1, theta2)
        previous_joint_sample = sample

    return trajectory


def save_trajectory_csv(path: Path, trajectory: list[JointSample]) -> None:
    """Write trajectory samples to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "theta1_deg", "theta2_deg", "theta1_dot", "theta2_dot", "theta1_ddot", "theta2_ddot", "x", "y"])
        for sample in trajectory:
            writer.writerow(
                [
                    f"{sample.time:.6f}",
                    f"{math.degrees(sample.theta1):.6f}",
                    f"{math.degrees(sample.theta2):.6f}",
                    f"{sample.theta1_dot:.10f}",
                    f"{sample.theta2_dot:.10f}",
                    f"{sample.theta1_ddot:.10f}",
                    f"{sample.theta2_ddot:.10f}",
                    f"{sample.x:.10f}",
                    f"{sample.y:.10f}",
                ]
            )


def line_distance(point: Vector2, start: Vector2, end: Vector2) -> float:
    """Distance from a point to the line defined by start–end."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denom = math.hypot(dx, dy)
    if denom == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    return abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / denom


def max_line_deviation(points: list[Vector2], start: Vector2, end: Vector2) -> float:
    """Maximum line deviation for the full path."""
    return max(line_distance(point, start, end) for point in points)


def maybe_save_plots(cubic_traj, quintic_traj, cartesian_traj, start_xy, end_xy):
    """Save path and profile plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Matplotlib not found; plot output skipped.")
        return

    plt.figure(figsize=(8, 6))
    plt.plot([sample.x for sample in cubic_traj], [sample.y for sample in cubic_traj], label="joint cubic")
    plt.plot([sample.x for sample in quintic_traj], [sample.y for sample in quintic_traj], label="joint quintic")
    plt.plot([sample.x for sample in cartesian_traj], [sample.y for sample in cartesian_traj], label="cartesian + IK")
    plt.plot([start_xy[0], end_xy[0]], [start_xy[1], end_xy[1]], "k--", label="ideal line")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("End-effector paths")
    plt.tight_layout()
    plt.savefig(PATH_PLOT, dpi=120)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot([sample.time for sample in cubic_traj], [sample.theta1_dot for sample in cubic_traj], label="cubic theta1_dot")
    plt.plot([sample.time for sample in quintic_traj], [sample.theta1_dot for sample in quintic_traj], label="quintic theta1_dot")
    plt.plot([sample.time for sample in cubic_traj], [sample.theta1_ddot for sample in cubic_traj], label="cubic theta1_ddot")
    plt.plot([sample.time for sample in quintic_traj], [sample.theta1_ddot for sample in quintic_traj], label="quintic theta1_ddot")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Velocity and acceleration profiles")
    plt.tight_layout()
    plt.savefig(PROFILE_PLOT, dpi=120)
    plt.close()

    print(f"Plot saved: {PATH_PLOT}")
    print(f"Plot saved: {PROFILE_PLOT}")


def print_summary(cubic_traj, quintic_traj, cartesian_traj, start_xy, end_xy):
    """Print B1 validation summary."""
    cubic_points = [(sample.x, sample.y) for sample in cubic_traj]
    quintic_points = [(sample.x, sample.y) for sample in quintic_traj]
    cartesian_points = [(sample.x, sample.y) for sample in cartesian_traj]

    cubic_deviation = max_line_deviation(cubic_points, start_xy, end_xy)
    quintic_deviation = max_line_deviation(quintic_points, start_xy, end_xy)
    cartesian_deviation = max_line_deviation(cartesian_points, start_xy, end_xy)

    print("=" * 90)
    print("B1: Trajectory Generation")
    print("=" * 90)
    print(
        f"Cubic boundary velocities  : start=[{cubic_traj[0].theta1_dot:+.6f}, {cubic_traj[0].theta2_dot:+.6f}] "
        f"end=[{cubic_traj[-1].theta1_dot:+.6f}, {cubic_traj[-1].theta2_dot:+.6f}]"
    )
    print(
        f"Quintic boundary velocities: start=[{quintic_traj[0].theta1_dot:+.6f}, {quintic_traj[0].theta2_dot:+.6f}] "
        f"end=[{quintic_traj[-1].theta1_dot:+.6f}, {quintic_traj[-1].theta2_dot:+.6f}]"
    )
    print(
        f"Quintic boundary accel     : start=[{quintic_traj[0].theta1_ddot:+.6f}, {quintic_traj[0].theta2_ddot:+.6f}] "
        f"end=[{quintic_traj[-1].theta1_ddot:+.6f}, {quintic_traj[-1].theta2_ddot:+.6f}]"
    )
    print(
        f"Path deviation (max)       : cubic={cubic_deviation:.8f} m, "
        f"quintic={quintic_deviation:.8f} m, cartesian={cartesian_deviation:.8f} m"
    )
    print("Expected: cartesian deviation very close to the line, joint-space path deviation will be significant.")


def main() -> None:
    start_xy = (0.20, 0.30)
    end_xy = (0.40, 0.10)
    duration = 2.0
    samples = 101

    start_solution = analytic_ik(start_xy)[0]
    end_solution = analytic_ik(end_xy)[0]
    start_angles = (start_solution.theta1, start_solution.theta2)
    end_angles = (end_solution.theta1, end_solution.theta2)

    cubic_traj = joint_trajectory(start_angles, end_angles, duration, samples, mode="cubic")
    quintic_traj = joint_trajectory(start_angles, end_angles, duration, samples, mode="quintic")
    cartesian_traj = cartesian_trajectory(start_xy, end_xy, duration, samples)

    save_trajectory_csv(CUBIC_CSV, cubic_traj)
    save_trajectory_csv(QUINTIC_CSV, quintic_traj)
    save_trajectory_csv(CARTESIAN_CSV, cartesian_traj)

    print_summary(cubic_traj, quintic_traj, cartesian_traj, start_xy, end_xy)
    print(f"CSV saved: {CUBIC_CSV}")
    print(f"CSV saved: {QUINTIC_CSV}")
    print(f"CSV saved: {CARTESIAN_CSV}")

    maybe_save_plots(cubic_traj, quintic_traj, cartesian_traj, start_xy, end_xy)


if __name__ == "__main__":
    main()
