"""Phase 2.1 — Jacobian Analysis: Geometric, Analytical, and Numerical.

This script implements:
  1. Geometric Jacobian construction from FK frames (using Pinocchio frame data)
  2. Pinocchio built-in Jacobian computation for comparison
  3. Numerical Jacobian via finite differences as a third cross-check
  4. Manipulability measure: w = sqrt(det(J @ J.T))
  5. Singularity detection (det(J) -> 0)
  6. Manipulability heatmap across workspace (sweep q2, q3)

UR5e Singularity Cases:
  - Wrist singularity:    q5 ~ 0 (wrist axes align)
  - Shoulder singularity: EE lies on J1 axis
  - Elbow singularity:    arm fully extended (links 2+3 collinear)

Run:
    python3 lab-2-Ur5e-robotics-lab/src/a3_jacobian.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ur5e_common import (
    DOCS_DIR,
    NUM_JOINTS,
    Q_HOME,
    Q_ZEROS,
    load_pinocchio_model,
    load_mujoco_model,
    get_mj_ee_site_id,
)


# ---------------------------------------------------------------------------
# Model loading (module-level singletons, lazy)
# ---------------------------------------------------------------------------

_pin_cache: tuple | None = None
_mj_cache: tuple | None = None


def _get_pin() -> tuple:
    """Return cached (model, data, ee_frame_id)."""
    global _pin_cache
    if _pin_cache is None:
        _pin_cache = load_pinocchio_model()
    return _pin_cache


def _get_mj() -> tuple:
    """Return cached (mj_model, mj_data)."""
    global _mj_cache
    if _mj_cache is None:
        _mj_cache = load_mujoco_model()
    return _mj_cache


# ---------------------------------------------------------------------------
# 1. Geometric Jacobian from FK frames
# ---------------------------------------------------------------------------


def geometric_jacobian(q: np.ndarray) -> np.ndarray:
    """Construct the 6x6 geometric Jacobian from Pinocchio FK frame data.

    For revolute joint i:
        J_linear[:, i]  = z_i x (p_ee - p_i)
        J_angular[:, i] = z_i

    where z_i is the joint rotation axis in world frame and p_i is the
    joint origin position in world frame.

    Args:
        q: Joint angles (6,).

    Returns:
        6x6 Jacobian matrix (top 3 rows = linear, bottom 3 = angular).
    """
    model, data, ee_frame_id = _get_pin()
    q = np.asarray(q, dtype=np.float64)

    # Run FK to populate joint and frame placements
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # EE position in world frame
    p_ee = data.oMf[ee_frame_id].translation.copy()

    J = np.zeros((6, NUM_JOINTS))

    for i in range(NUM_JOINTS):
        # Pinocchio joint indices are 1-based (0 is the universe joint)
        joint_id = i + 1

        # Joint placement in world frame
        oMi = data.oMi[joint_id]
        p_i = oMi.translation

        # Joint rotation axis in world frame
        # Must use the actual joint axis from the model (not always z!)
        # Pinocchio joint types: RZ rotates about z, RY about y
        joint = model.joints[joint_id]
        shortname = joint.shortname()
        if "RZ" in shortname:
            local_axis = np.array([0.0, 0.0, 1.0])
        elif "RY" in shortname:
            local_axis = np.array([0.0, 1.0, 0.0])
        elif "RX" in shortname:
            local_axis = np.array([1.0, 0.0, 0.0])
        else:
            local_axis = np.array([0.0, 0.0, 1.0])
        z_i = oMi.rotation @ local_axis

        # Linear part: z_i x (p_ee - p_i)
        J[:3, i] = np.cross(z_i, p_ee - p_i)

        # Angular part: z_i
        J[3:, i] = z_i

    return J


# ---------------------------------------------------------------------------
# 2. Pinocchio built-in Jacobian
# ---------------------------------------------------------------------------


def pinocchio_jacobian(q: np.ndarray) -> np.ndarray:
    """Compute the 6x6 frame Jacobian using Pinocchio's built-in method.

    Uses LOCAL_WORLD_ALIGNED reference frame for consistency with the
    geometric Jacobian (expressed in world-aligned frame at EE origin).

    Args:
        q: Joint angles (6,).

    Returns:
        6x6 Jacobian matrix.
    """
    model, data, ee_frame_id = _get_pin()
    q = np.asarray(q, dtype=np.float64)

    J = pin.computeFrameJacobian(
        model, data, q, ee_frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    return J.copy()


# ---------------------------------------------------------------------------
# 3. Numerical Jacobian (finite differences)
# ---------------------------------------------------------------------------


def numerical_jacobian(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute the 6x6 Jacobian by central finite differences.

    For each joint i, perturb q[i] by +/- eps, compute FK, and
    approximate the derivative of (position, orientation) w.r.t. q[i].

    Orientation derivative uses the log map of the relative rotation:
        delta_rot = log(R_plus @ R_minus.T) / (2*eps)

    Args:
        q: Joint angles (6,).
        eps: Perturbation size.

    Returns:
        6x6 Jacobian matrix.
    """
    model, data, ee_frame_id = _get_pin()
    q = np.asarray(q, dtype=np.float64)

    J = np.zeros((6, NUM_JOINTS))

    for i in range(NUM_JOINTS):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += eps
        q_minus[i] -= eps

        # Forward pass (+)
        pin.forwardKinematics(model, data, q_plus)
        pin.updateFramePlacements(model, data)
        p_plus = data.oMf[ee_frame_id].translation.copy()
        R_plus = data.oMf[ee_frame_id].rotation.copy()

        # Forward pass (-)
        pin.forwardKinematics(model, data, q_minus)
        pin.updateFramePlacements(model, data)
        p_minus = data.oMf[ee_frame_id].translation.copy()
        R_minus = data.oMf[ee_frame_id].rotation.copy()

        # Linear velocity approximation
        J[:3, i] = (p_plus - p_minus) / (2.0 * eps)

        # Angular velocity approximation via log map
        dR = R_plus @ R_minus.T
        # Convert to SE3 and use Pinocchio log
        omega = pin.log3(dR) / (2.0 * eps)
        J[3:, i] = omega

    return J


# ---------------------------------------------------------------------------
# 4. Manipulability measure
# ---------------------------------------------------------------------------


def manipulability(q: np.ndarray) -> float:
    """Yoshikawa manipulability index: w = sqrt(det(J @ J.T)).

    This scalar measures how far the robot is from a singular configuration.
    Higher values indicate better dexterity.

    Args:
        q: Joint angles (6,).

    Returns:
        Manipulability index (non-negative scalar).
    """
    J = pinocchio_jacobian(q)
    JJT = J @ J.T
    det_val = np.linalg.det(JJT)
    # Clamp to zero to handle numerical noise at singularities
    return float(np.sqrt(max(det_val, 0.0)))


def jacobian_determinant(q: np.ndarray) -> float:
    """Compute det(J) for the 6x6 Jacobian.

    For a square Jacobian, det(J) = 0 indicates a singularity.

    Args:
        q: Joint angles (6,).

    Returns:
        Determinant of the Jacobian.
    """
    J = pinocchio_jacobian(q)
    return float(np.linalg.det(J))


# ---------------------------------------------------------------------------
# 5. Singularity detection
# ---------------------------------------------------------------------------


def detect_singularity(q: np.ndarray, threshold: float = 1e-3) -> list[str]:
    """Detect UR5e singularity conditions at a given configuration.

    Checks three classical singularity types:
      1. Wrist singularity: q5 ~ 0 (wrist axes 4 and 6 align)
      2. Shoulder singularity: EE lies on the J1 (base z) axis
      3. Elbow singularity: links 2 and 3 are nearly collinear

    Also checks the Jacobian determinant as a general indicator.

    Args:
        q: Joint angles (6,).
        threshold: Threshold for singularity detection.

    Returns:
        List of detected singularity types (may be empty).
    """
    q = np.asarray(q, dtype=np.float64)
    singularities = []

    # General: check manipulability
    w = manipulability(q)
    if w < threshold:
        singularities.append(f"low_manipulability (w={w:.6f})")

    # Wrist singularity: q5 ~ 0
    if abs(q[4]) < 0.05:  # ~2.9 degrees
        singularities.append(f"wrist (q5={np.degrees(q[4]):.2f} deg)")

    # Shoulder singularity: EE xy distance from base z-axis ~ 0
    model, data, ee_frame_id = _get_pin()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p_ee = data.oMf[ee_frame_id].translation
    xy_dist = np.sqrt(p_ee[0] ** 2 + p_ee[1] ** 2)
    if xy_dist < 0.02:  # 2 cm from base axis
        singularities.append(f"shoulder (xy_dist={xy_dist:.4f} m)")

    # Elbow singularity: q3 ~ 0 or q3 ~ pi (arm extended)
    if abs(q[2]) < 0.05 or abs(abs(q[2]) - np.pi) < 0.05:
        singularities.append(f"elbow (q3={np.degrees(q[2]):.2f} deg)")

    return singularities


# ---------------------------------------------------------------------------
# 6. Manipulability heatmap (q2-q3 sweep)
# ---------------------------------------------------------------------------


def compute_manipulability_heatmap(
    n_samples: int = 25,
    q_base: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute manipulability over a grid of (q2, q3) values.

    Fixes joints 1, 4, 5, 6 at q_base values and sweeps q2 and q3.

    Args:
        n_samples: Number of samples per axis.
        q_base: Base configuration (6,). Defaults to Q_HOME.

    Returns:
        Tuple of (q2_values, q3_values, manipulability_grid, det_grid).
    """
    if q_base is None:
        q_base = Q_HOME.copy()

    q2_vals = np.linspace(-np.pi, np.pi / 6, n_samples)
    q3_vals = np.linspace(-np.pi, np.pi, n_samples)

    w_grid = np.zeros((n_samples, n_samples))
    det_grid = np.zeros((n_samples, n_samples))

    for i, q2 in enumerate(q2_vals):
        for j, q3 in enumerate(q3_vals):
            q = q_base.copy()
            q[1] = q2
            q[2] = q3
            w_grid[i, j] = manipulability(q)
            det_grid[i, j] = jacobian_determinant(q)

    return q2_vals, q3_vals, w_grid, det_grid


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def save_heatmap_csv(
    q2_vals: np.ndarray,
    q3_vals: np.ndarray,
    w_grid: np.ndarray,
    det_grid: np.ndarray,
    path: Path,
) -> Path:
    """Save the manipulability heatmap data to CSV.

    Args:
        q2_vals: Array of q2 values.
        q3_vals: Array of q3 values.
        w_grid: 2D array of manipulability values.
        det_grid: 2D array of determinant values.
        path: Output CSV path.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q2_rad", "q3_rad", "manipulability", "det_j"])
        for i, q2 in enumerate(q2_vals):
            for j, q3 in enumerate(q3_vals):
                writer.writerow([
                    f"{q2:.6f}",
                    f"{q3:.6f}",
                    f"{w_grid[i, j]:.8f}",
                    f"{det_grid[i, j]:.8f}",
                ])
    return path


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def demo_jacobian_comparison() -> None:
    """Compare geometric, Pinocchio, and numerical Jacobians."""
    print("=" * 72)
    print("Part 1: Jacobian Cross-Validation (Geometric vs Pinocchio vs Numerical)")
    print("=" * 72)

    configs = {
        "zeros":       Q_ZEROS.copy(),
        "home":        Q_HOME.copy(),
        "elbow_open":  np.array([0.1, -1.1, 1.8, -1.4, 0.8, 0.0]),
        "wrist_flip":  np.array([0.0, -1.4, 1.2, -1.1, 0.02, np.pi]),
        "random":      np.array([0.3, -1.2, 0.8, -0.6, 1.1, -0.4]),
    }

    print(f"\n  {'Config':<14} {'Geo vs Pin':>14} {'Geo vs Num':>14} {'Pin vs Num':>14}")
    print("  " + "-" * 58)

    all_ok = True
    for name, q in configs.items():
        J_geo = geometric_jacobian(q)
        J_pin = pinocchio_jacobian(q)
        J_num = numerical_jacobian(q)

        err_geo_pin = np.linalg.norm(J_geo - J_pin, "fro")
        err_geo_num = np.linalg.norm(J_geo - J_num, "fro")
        err_pin_num = np.linalg.norm(J_pin - J_num, "fro")

        status = "OK" if max(err_geo_pin, err_geo_num, err_pin_num) < 1e-4 else "FAIL"
        if status == "FAIL":
            all_ok = False

        print(f"  {name:<14} {err_geo_pin:>14.2e} {err_geo_num:>14.2e} "
              f"{err_pin_num:>14.2e}  [{status}]")

    print()
    if all_ok:
        print("  PASS: All three Jacobian methods agree (Frobenius < 1e-4)")
    else:
        print("  WARNING: Some configurations show Jacobian disagreement")

    # Print full Jacobian at home for inspection
    print(f"\n  Full Jacobian at HOME config (Pinocchio):")
    J = pinocchio_jacobian(Q_HOME)
    for row_idx in range(6):
        vals = "  ".join(f"{J[row_idx, c]:8.4f}" for c in range(6))
        label = ["vx", "vy", "vz", "wx", "wy", "wz"][row_idx]
        print(f"    {label}: [{vals}]")


def demo_manipulability() -> None:
    """Show manipulability at various configurations."""
    print("\n" + "=" * 72)
    print("Part 2: Manipulability Analysis")
    print("=" * 72)

    configs = {
        "zeros":       Q_ZEROS.copy(),
        "home":        Q_HOME.copy(),
        "elbow_open":  np.array([0.1, -1.1, 1.8, -1.4, 0.8, 0.0]),
        "random":      np.array([0.3, -1.2, 0.8, -0.6, 1.1, -0.4]),
    }

    print(f"\n  {'Config':<14} {'Manipulability':>16} {'det(J)':>16}")
    print("  " + "-" * 48)
    for name, q in configs.items():
        w = manipulability(q)
        d = jacobian_determinant(q)
        print(f"  {name:<14} {w:>16.6f} {d:>16.6f}")

    # Singular value decomposition at home
    J = pinocchio_jacobian(Q_HOME)
    U, s, Vt = np.linalg.svd(J)
    print(f"\n  Singular values at HOME config:")
    for i, sv in enumerate(s):
        bar = "#" * int(sv * 20)
        print(f"    sigma_{i+1} = {sv:8.4f}  {bar}")
    print(f"    Condition number: {s[0] / s[-1]:.2f}")


def demo_singularity_detection() -> None:
    """Probe known UR5e singularity configurations."""
    print("\n" + "=" * 72)
    print("Part 3: Singularity Detection")
    print("=" * 72)

    probes = {
        "wrist_singularity":    np.array([0.0, -1.3, 1.3, -1.4, 0.0, 0.0]),
        "near_wrist":           np.array([0.0, -1.3, 1.3, -1.4, 0.01, 0.0]),
        "elbow_extended":       np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.7, 0.0]),
        "shoulder_singularity": np.array([0.0, 0.0, np.pi, 0.0, np.pi / 2, 0.0]),
        "home (non-singular)":  Q_HOME.copy(),
    }

    print(f"\n  {'Config':<26} {'w':>12} {'det(J)':>14} {'Singularities'}")
    print("  " + "-" * 80)

    for name, q in probes.items():
        w = manipulability(q)
        d = jacobian_determinant(q)
        sings = detect_singularity(q)
        sing_str = ", ".join(sings) if sings else "none"
        print(f"  {name:<26} {w:>12.6f} {d:>14.6f}   {sing_str}")


def demo_heatmap() -> None:
    """Compute and save the manipulability heatmap."""
    print("\n" + "=" * 72)
    print("Part 4: Manipulability Heatmap (q2 vs q3)")
    print("=" * 72)

    q2_vals, q3_vals, w_grid, det_grid = compute_manipulability_heatmap(n_samples=25)

    csv_path = DOCS_DIR / "a3_manipulability_heatmap.csv"
    save_heatmap_csv(q2_vals, q3_vals, w_grid, det_grid, csv_path)
    print(f"\n  Heatmap saved to: {csv_path}")
    print(f"  Grid size: {len(q2_vals)} x {len(q3_vals)} = {len(q2_vals) * len(q3_vals)} points")

    # Summary stats
    print(f"\n  Manipulability statistics:")
    print(f"    min  = {w_grid.min():.6f}")
    print(f"    max  = {w_grid.max():.6f}")
    print(f"    mean = {w_grid.mean():.6f}")
    print(f"    std  = {w_grid.std():.6f}")

    # Find best and worst configurations
    best_idx = np.unravel_index(np.argmax(w_grid), w_grid.shape)
    worst_idx = np.unravel_index(np.argmin(w_grid), w_grid.shape)
    print(f"\n    Best  dexterity: q2={q2_vals[best_idx[0]]:.3f}, "
          f"q3={q3_vals[best_idx[1]]:.3f} (w={w_grid[best_idx]:.6f})")
    print(f"    Worst dexterity: q2={q2_vals[worst_idx[0]]:.3f}, "
          f"q3={q3_vals[worst_idx[1]]:.3f} (w={w_grid[worst_idx]:.6f})")

    # Count near-singular points
    n_singular = np.sum(w_grid < 1e-3)
    pct = 100.0 * n_singular / w_grid.size
    print(f"\n    Near-singular points (w < 1e-3): {n_singular}/{w_grid.size} ({pct:.1f}%)")


def main() -> None:
    """Run all Jacobian analysis demos."""
    demo_jacobian_comparison()
    demo_manipulability()
    demo_singularity_detection()
    demo_heatmap()

    print("\n" + "=" * 72)
    print("Phase 2.1 -- Jacobian Analysis: COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
