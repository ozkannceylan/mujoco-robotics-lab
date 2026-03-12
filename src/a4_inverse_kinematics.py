"""A4: Inverse Kinematics — compute joint angles from a target position."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from a3_jacobian import EE_OFFSET, L1, L2, MODEL_PATH, analytic_jacobian, fk_endeffector

OUTPUT_CSV = Path("docs/a4_ik_benchmark.csv")

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class IKSolution:
    branch: str
    theta1: float
    theta2: float
    error_norm: float


@dataclass(frozen=True)
class NumericIKResult:
    method: str
    success: bool
    iterations: int
    theta1: float
    theta2: float
    error_norm: float
    reason: str


def normalize_angle(angle: float) -> float:
    """Bring angle into [-pi, pi] range."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def target_radius_limits() -> tuple[float, float]:
    """Reachable radius limits including site offset."""
    l2_eff = L2 + EE_OFFSET
    return (abs(L1 - l2_eff), L1 + l2_eff)


def is_reachable(target: Vector2, tol: float = 1e-9) -> bool:
    """Is the target point within the reachable area of the 2-link arm?"""
    radius = math.hypot(target[0], target[1])
    inner, outer = target_radius_limits()
    return inner - tol <= radius <= outer + tol


def analytic_ik(target: Vector2) -> list[IKSolution]:
    """Analytic IK with two branches for a 2-link planar robot."""
    x, y = target
    l2_eff = L2 + EE_OFFSET
    radius_sq = x * x + y * y
    radius = math.sqrt(radius_sq)
    inner, outer = target_radius_limits()
    if radius < inner - 1e-9 or radius > outer + 1e-9:
        raise ValueError(f"Unreachable target: radius={radius:.6f}, limits=[{inner:.6f}, {outer:.6f}]")

    cos_theta2 = (radius_sq - L1 * L1 - l2_eff * l2_eff) / (2.0 * L1 * l2_eff)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2_abs = math.acos(cos_theta2)

    solutions: list[IKSolution] = []
    for branch, theta2 in (("elbow_down", theta2_abs), ("elbow_up", -theta2_abs)):
        theta1 = math.atan2(y, x) - math.atan2(l2_eff * math.sin(theta2), L1 + l2_eff * math.cos(theta2))
        theta1 = normalize_angle(theta1)
        theta2 = normalize_angle(theta2)
        x_fk, y_fk = fk_endeffector(theta1, theta2)
        error = math.hypot(target[0] - x_fk, target[1] - y_fk)
        solutions.append(IKSolution(branch=branch, theta1=theta1, theta2=theta2, error_norm=error))
    return solutions


def inverse_2x2(matrix: tuple[tuple[float, float], tuple[float, float]], eps: float = 1e-12):
    """Invert 2x2 matrix; returns None if singular."""
    a, b = matrix[0]
    c, d = matrix[1]
    det = a * d - b * c
    if abs(det) < eps:
        return None
    inv_det = 1.0 / det
    return ((d * inv_det, -b * inv_det), (-c * inv_det, a * inv_det))


def matvec2(matrix: tuple[tuple[float, float], tuple[float, float]], vector: Vector2) -> Vector2:
    """2x2 matrix × 2x1 vector."""
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    )


def pinv_step(theta1: float, theta2: float, error: Vector2, alpha: float) -> tuple[float, float] | None:
    """Pseudo-inverse step for square, full-rank case."""
    jac = analytic_jacobian(theta1, theta2)
    jac_inv = inverse_2x2(jac)
    if jac_inv is None:
        return None
    delta = matvec2(jac_inv, error)
    return (alpha * delta[0], alpha * delta[1])


def dls_step(theta1: float, theta2: float, error: Vector2, alpha: float, damping: float) -> Vector2 | None:
    """Damped Least Squares step: J^T (J J^T + lambda^2 I)^-1 e."""
    jac = analytic_jacobian(theta1, theta2)
    a, b = jac[0]
    c, d = jac[1]

    jjt = (
        (a * a + b * b + damping * damping, a * c + b * d),
        (a * c + b * d, c * c + d * d + damping * damping),
    )
    inv = inverse_2x2(jjt)
    if inv is None:
        return None

    temp = matvec2(inv, error)
    delta = (
        a * temp[0] + c * temp[1],
        b * temp[0] + d * temp[1],
    )
    return (alpha * delta[0], alpha * delta[1])


def numeric_ik(
    target: Vector2,
    method: str,
    initial_guess: Vector2 = (0.25, -0.35),
    alpha: float = 0.7,
    damping: float = 0.05,
    max_iters: int = 100,
    tol: float = 1e-5,
) -> NumericIKResult:
    """Iterative IK solution."""
    theta1, theta2 = initial_guess

    for iteration in range(1, max_iters + 1):
        current = fk_endeffector(theta1, theta2)
        error = (target[0] - current[0], target[1] - current[1])
        error_norm = math.hypot(error[0], error[1])
        if error_norm < tol:
            return NumericIKResult(method, True, iteration - 1, theta1, theta2, error_norm, "converged")

        if method == "pinv":
            delta = pinv_step(theta1, theta2, error, alpha=alpha)
            if delta is None:
                return NumericIKResult(method, False, iteration - 1, theta1, theta2, error_norm, "singular_jacobian")
        elif method == "dls":
            delta = dls_step(theta1, theta2, error, alpha=alpha, damping=damping)
            if delta is None:
                return NumericIKResult(method, False, iteration - 1, theta1, theta2, error_norm, "dls_inversion_failed")
        else:
            raise ValueError(f"Unknown method: {method}")

        theta1 = normalize_angle(theta1 + delta[0])
        theta2 = normalize_angle(theta2 + delta[1])

    final = fk_endeffector(theta1, theta2)
    final_error = math.hypot(target[0] - final[0], target[1] - final[1])
    return NumericIKResult(method, final_error < tol, max_iters, theta1, theta2, final_error, "max_iters")


def benchmark_targets(count: int = 20, seed: int = 7) -> list[Vector2]:
    """Generate target points from the reachable workspace."""
    rng = random.Random(seed)
    inner, outer = target_radius_limits()
    targets: list[Vector2] = []
    while len(targets) < count:
        radius = rng.uniform(inner + 0.02, outer - 0.03)
        angle = rng.uniform(-math.pi, math.pi)
        targets.append((radius * math.cos(angle), radius * math.sin(angle)))
    return targets


def benchmark_methods(targets: list[Vector2]) -> dict[str, dict[str, float]]:
    """Run pinv vs dls comparison over 20 targets."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "pinv": {"success": 0.0, "iterations": 0.0, "avg_error": 0.0},
        "dls": {"success": 0.0, "iterations": 0.0, "avg_error": 0.0},
    }

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_x", "target_y", "method", "success", "iterations", "theta1_deg", "theta2_deg", "error_norm"])
        for target in targets:
            for method in ("pinv", "dls"):
                result = numeric_ik(target, method=method)
                summary[method]["success"] += 1.0 if result.success else 0.0
                summary[method]["iterations"] += result.iterations
                summary[method]["avg_error"] += result.error_norm
                writer.writerow(
                    [
                        f"{target[0]:.6f}",
                        f"{target[1]:.6f}",
                        method,
                        str(result.success),
                        result.iterations,
                        f"{math.degrees(result.theta1):.6f}",
                        f"{math.degrees(result.theta2):.6f}",
                        f"{result.error_norm:.10f}",
                    ]
                )

    count = float(len(targets))
    for method in summary:
        summary[method]["success_rate"] = summary[method]["success"] / count
        summary[method]["avg_iterations"] = summary[method]["iterations"] / count
        summary[method]["avg_error"] = summary[method]["avg_error"] / count
    return summary


def singularity_stress_case() -> tuple[NumericIKResult, NumericIKResult]:
    """Compare pinv and dls behaviour near singularity."""
    target = (0.612, 0.020)
    initial = (0.0, 0.0)
    pinv_result = numeric_ik(target, method="pinv", initial_guess=initial)
    dls_result = numeric_ik(target, method="dls", initial_guess=initial, damping=0.08, max_iters=300)
    return pinv_result, dls_result


def mujoco_validate_solution(theta1: float, theta2: float, target: Vector2) -> float | None:
    """Return site position error from MuJoCo if available."""
    try:
        import mujoco
    except ModuleNotFoundError:
        return None

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    data.qpos[0] = theta1
    data.qpos[1] = theta2
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    actual = data.site_xpos[site_id]
    return math.hypot(target[0] - float(actual[0]), target[1] - float(actual[1]))


def print_analytic_examples() -> None:
    """Show two-branch analytic IK examples."""
    target = (0.34, 0.28)
    solutions = analytic_ik(target)
    print("=" * 90)
    print("A4: Inverse Kinematics")
    print("=" * 90)
    print(f"Target point: x={target[0]:.4f}, y={target[1]:.4f}")
    for solution in solutions:
        print(
            f"- {solution.branch:10s} -> "
            f"theta1={math.degrees(solution.theta1):+8.3f} deg, "
            f"theta2={math.degrees(solution.theta2):+8.3f} deg, "
            f"FK error={solution.error_norm:.10f}"
        )


def main() -> None:
    print_analytic_examples()

    targets = benchmark_targets()
    summary = benchmark_methods(targets)
    print(f"\nBenchmark saved: {OUTPUT_CSV}")

    print("\n20-target benchmark")
    for method in ("pinv", "dls"):
        print(
            f"- {method:4s}: "
            f"success={summary[method]['success_rate'] * 100:.1f}%, "
            f"avg_iter={summary[method]['avg_iterations']:.2f}, "
            f"avg_error={summary[method]['avg_error']:.10f}"
        )

    pinv_stress, dls_stress = singularity_stress_case()
    print("\nSingularity stress test (target=(0.612, 0.020), initial=(0, 0))")
    print(
        f"- pinv: success={pinv_stress.success}, "
        f"iters={pinv_stress.iterations}, "
        f"error={pinv_stress.error_norm:.10f}, "
        f"reason={pinv_stress.reason}"
    )
    print(
        f"- dls : success={dls_stress.success}, "
        f"iters={dls_stress.iterations}, "
        f"error={dls_stress.error_norm:.10f}, "
        f"reason={dls_stress.reason}"
    )

    target = (0.34, 0.28)
    mujoco_error = mujoco_validate_solution(analytic_ik(target)[0].theta1, analytic_ik(target)[0].theta2, target)
    if mujoco_error is None:
        print("\nMuJoCo validation skipped: `mujoco` cannot be imported in this environment.")
    else:
        print(f"\nMuJoCo site_xpos error: {mujoco_error:.10f}")


if __name__ == "__main__":
    main()
