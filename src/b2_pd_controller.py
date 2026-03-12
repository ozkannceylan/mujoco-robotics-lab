"""B2: PD controller and gravity compensation experiments."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from b1_trajectory_generation import joint_trajectory

FIXED_NO_GC_CSV = Path("docs/b2_fixed_target_no_gc.csv")
FIXED_GC_CSV = Path("docs/b2_fixed_target_gc.csv")
TRACKING_NO_GC_CSV = Path("docs/b2_tracking_no_gc.csv")
TRACKING_GC_CSV = Path("docs/b2_tracking_gc.csv")

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class ControlSample:
    time: float
    q1: float
    q2: float
    q1_dot: float
    q2_dot: float
    q1_des: float
    q2_des: float
    q1_dot_des: float
    q2_dot_des: float
    tau1: float
    tau2: float
    err1: float
    err2: float


class SimpleTwoLinkPlant:
    """Lightweight, deterministic 2-joint plant for PD tuning.

    Not real MuJoCo dynamics; used to show control logic and gravity compensation effect.
    """

    def __init__(self):
        self.q = [0.0, 0.0]
        self.qd = [0.0, 0.0]
        self.inertia = (0.22, 0.12)
        self.damping = (0.55, 0.30)
        self.gravity_scale = (1.6, 0.75)

    def gravity_torque(self) -> Vector2:
        """Simple configuration-dependent gravity model."""
        q1, q2 = self.q
        g1 = self.gravity_scale[0] * math.sin(q1) + 0.18 * math.sin(q1 + q2)
        g2 = self.gravity_scale[1] * math.sin(q1 + q2)
        return (g1, g2)

    def step(self, torque: Vector2, dt: float) -> None:
        """Advance the plant with Euler integration."""
        gravity = self.gravity_torque()
        q1_ddot = (torque[0] - self.damping[0] * self.qd[0] - gravity[0]) / self.inertia[0]
        q2_ddot = (torque[1] - self.damping[1] * self.qd[1] - gravity[1]) / self.inertia[1]

        self.qd[0] += q1_ddot * dt
        self.qd[1] += q2_ddot * dt
        self.q[0] += self.qd[0] * dt
        self.q[1] += self.qd[1] * dt


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a scalar to a range."""
    return max(min_value, min(max_value, value))


def pd_control(
    q_des: Vector2,
    qd_des: Vector2,
    q: Vector2,
    qd: Vector2,
    kp: Vector2,
    kd: Vector2,
    gravity_term: Vector2 = (0.0, 0.0),
    torque_limit: float = 5.0,
) -> Vector2:
    """PD + optional gravity compensation."""
    tau1 = kp[0] * (q_des[0] - q[0]) + kd[0] * (qd_des[0] - qd[0]) + gravity_term[0]
    tau2 = kp[1] * (q_des[1] - q[1]) + kd[1] * (qd_des[1] - qd[1]) + gravity_term[1]
    return (clamp(tau1, -torque_limit, torque_limit), clamp(tau2, -torque_limit, torque_limit))


def save_control_csv(path: Path, samples: list[ControlSample]) -> None:
    """Write control log to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time",
                "q1_deg",
                "q2_deg",
                "q1_dot",
                "q2_dot",
                "q1_des_deg",
                "q2_des_deg",
                "q1_dot_des",
                "q2_dot_des",
                "tau1",
                "tau2",
                "err1_deg",
                "err2_deg",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    f"{sample.time:.6f}",
                    f"{math.degrees(sample.q1):.6f}",
                    f"{math.degrees(sample.q2):.6f}",
                    f"{sample.q1_dot:.10f}",
                    f"{sample.q2_dot:.10f}",
                    f"{math.degrees(sample.q1_des):.6f}",
                    f"{math.degrees(sample.q2_des):.6f}",
                    f"{sample.q1_dot_des:.10f}",
                    f"{sample.q2_dot_des:.10f}",
                    f"{sample.tau1:.10f}",
                    f"{sample.tau2:.10f}",
                    f"{math.degrees(sample.err1):.6f}",
                    f"{math.degrees(sample.err2):.6f}",
                ]
            )


def fixed_target_simulation(use_gravity_comp: bool, duration: float = 4.0, dt: float = 0.01) -> tuple[list[ControlSample], dict[str, float]]:
    """Step response for a single target angle."""
    plant = SimpleTwoLinkPlant()
    q_des = (math.radians(90.0), math.radians(-45.0))
    qd_des = (0.0, 0.0)
    kp = (18.0, 13.0)
    kd = (4.8, 3.3)

    samples: list[ControlSample] = []
    max_q1 = plant.q[0]
    max_q2 = plant.q[1]
    steps = int(duration / dt)
    for step in range(steps + 1):
        time_s = step * dt
        gravity_term = plant.gravity_torque() if use_gravity_comp else (0.0, 0.0)
        torque = pd_control(q_des, qd_des, tuple(plant.q), tuple(plant.qd), kp, kd, gravity_term)
        err1 = q_des[0] - plant.q[0]
        err2 = q_des[1] - plant.q[1]
        samples.append(
            ControlSample(time_s, plant.q[0], plant.q[1], plant.qd[0], plant.qd[1], q_des[0], q_des[1], 0.0, 0.0, torque[0], torque[1], err1, err2)
        )
        max_q1 = max(max_q1, plant.q[0])
        max_q2 = max(max_q2, plant.q[1])
        plant.step(torque, dt)

    overshoot_q1 = max(0.0, (max_q1 - q_des[0]) / abs(q_des[0]))
    overshoot_q2 = max(0.0, (abs(min(sample.q2 for sample in samples)) - abs(q_des[1])) / abs(q_des[1]))
    final_err = math.hypot(samples[-1].err1, samples[-1].err2)
    metrics = {
        "overshoot_q1": overshoot_q1,
        "overshoot_q2": overshoot_q2,
        "final_error_norm": final_err,
    }
    return samples, metrics


def trajectory_tracking_simulation(use_gravity_comp: bool, duration: float = 2.0, dt: float = 0.01) -> tuple[list[ControlSample], dict[str, float]]:
    """Cubic joint trajectory tracking experiment."""
    start_angles = (math.radians(0.247046), math.radians(108.261824))
    end_angles = (math.radians(-52.430505), math.radians(104.605701))
    desired = joint_trajectory(start_angles, end_angles, duration, int(duration / dt) + 1, mode="cubic")

    plant = SimpleTwoLinkPlant()
    plant.q[0] = start_angles[0]
    plant.q[1] = start_angles[1]
    kp = (24.0, 18.0)
    kd = (6.0, 4.5)
    samples: list[ControlSample] = []
    squared_error_sum = 0.0

    for desired_sample in desired:
        q_des = (desired_sample.theta1, desired_sample.theta2)
        qd_des = (desired_sample.theta1_dot, desired_sample.theta2_dot)
        gravity_term = plant.gravity_torque() if use_gravity_comp else (0.0, 0.0)
        torque = pd_control(q_des, qd_des, tuple(plant.q), tuple(plant.qd), kp, kd, gravity_term)
        err1 = q_des[0] - plant.q[0]
        err2 = q_des[1] - plant.q[1]
        squared_error_sum += err1 * err1 + err2 * err2
        samples.append(
            ControlSample(
                desired_sample.time,
                plant.q[0],
                plant.q[1],
                plant.qd[0],
                plant.qd[1],
                q_des[0],
                q_des[1],
                qd_des[0],
                qd_des[1],
                torque[0],
                torque[1],
                err1,
                err2,
            )
        )
        if desired_sample.time < duration:
            plant.step(torque, dt)

    rms_error = math.sqrt(squared_error_sum / (2.0 * len(samples)))
    final_error = math.hypot(samples[-1].err1, samples[-1].err2)
    metrics = {
        "rms_error": rms_error,
        "final_error_norm": final_error,
    }
    return samples, metrics


def maybe_save_plots(fixed_no_gc, fixed_gc, tracking_no_gc, tracking_gc) -> None:
    """Save step response and tracking plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Matplotlib not found; B2 plot output skipped.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot([s.time for s in fixed_no_gc], [math.degrees(s.q1) for s in fixed_no_gc], label="q1 no gc")
    axes[0].plot([s.time for s in fixed_gc], [math.degrees(s.q1) for s in fixed_gc], label="q1 gc")
    axes[0].plot([s.time for s in fixed_gc], [math.degrees(s.q1_des) for s in fixed_gc], "k--", label="q1 desired")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title("Fixed target step response")

    axes[1].plot([s.time for s in tracking_no_gc], [math.degrees(s.err1) for s in tracking_no_gc], label="err1 no gc")
    axes[1].plot([s.time for s in tracking_gc], [math.degrees(s.err1) for s in tracking_gc], label="err1 gc")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title("Trajectory tracking error")
    plt.tight_layout()
    plt.savefig("docs/b2_pd_controller.png", dpi=120)
    plt.close()
    print("Plot saved: docs/b2_pd_controller.png")


def run_optional_mujoco_stub() -> None:
    """Hook for connecting to a real sim with MuJoCo env in future."""
    try:
        import mujoco  # noqa: F401
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        print("MuJoCo validation skipped: `mujoco`/`numpy` not available in this sandbox.")
        return

    print("MuJoCo dependencies found. Real torque-model tracking can be run outside this environment.")


def main() -> None:
    fixed_no_gc, fixed_no_gc_metrics = fixed_target_simulation(use_gravity_comp=False)
    fixed_gc, fixed_gc_metrics = fixed_target_simulation(use_gravity_comp=True)
    tracking_no_gc, tracking_no_gc_metrics = trajectory_tracking_simulation(use_gravity_comp=False)
    tracking_gc, tracking_gc_metrics = trajectory_tracking_simulation(use_gravity_comp=True)

    save_control_csv(FIXED_NO_GC_CSV, fixed_no_gc)
    save_control_csv(FIXED_GC_CSV, fixed_gc)
    save_control_csv(TRACKING_NO_GC_CSV, tracking_no_gc)
    save_control_csv(TRACKING_GC_CSV, tracking_gc)

    print("=" * 90)
    print("B2: PD Controller")
    print("=" * 90)
    print(
        f"Fixed target overshoot q1: no_gc={fixed_no_gc_metrics['overshoot_q1'] * 100:.2f}%, "
        f"gc={fixed_gc_metrics['overshoot_q1'] * 100:.2f}%"
    )
    print(
        f"Fixed target final error : no_gc={fixed_no_gc_metrics['final_error_norm']:.6f} rad, "
        f"gc={fixed_gc_metrics['final_error_norm']:.6f} rad"
    )
    print(
        f"Tracking RMS error      : no_gc={tracking_no_gc_metrics['rms_error']:.6f} rad, "
        f"gc={tracking_gc_metrics['rms_error']:.6f} rad"
    )
    print(
        f"Tracking final error    : no_gc={tracking_no_gc_metrics['final_error_norm']:.6f} rad, "
        f"gc={tracking_gc_metrics['final_error_norm']:.6f} rad"
    )
    print(f"CSV saved: {FIXED_NO_GC_CSV}")
    print(f"CSV saved: {FIXED_GC_CSV}")
    print(f"CSV saved: {TRACKING_NO_GC_CSV}")
    print(f"CSV saved: {TRACKING_GC_CSV}")
    maybe_save_plots(fixed_no_gc, fixed_gc, tracking_no_gc, tracking_gc)
    run_optional_mujoco_stub()


if __name__ == "__main__":
    main()
