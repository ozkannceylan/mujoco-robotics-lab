"""Lab 3 — Phase 1: Dynamics Fundamentals.

Computes and visualizes the manipulator dynamics quantities:
  - M(q): joint-space mass/inertia matrix
  - C(q, q̇): Coriolis matrix
  - g(q): gravity vector

Cross-validates Pinocchio vs MuJoCo for both g(q) and M(q).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure lab src is on path
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab3_common import (
    Q_HOME,
    Q_ZEROS,
    NUM_JOINTS,
    MEDIA_DIR,
    load_mujoco_model,
    load_pinocchio_model,
)


# ---------------------------------------------------------------------------
# Dynamics computation functions
# ---------------------------------------------------------------------------

def compute_mass_matrix(pin_model, pin_data, q: np.ndarray) -> np.ndarray:
    """Compute joint-space mass matrix M(q) via Pinocchio CRBA.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        q: Joint angles (6,).

    Returns:
        Symmetric positive-definite mass matrix M(q) [6x6].
    """
    import pinocchio as pin

    pin.crba(pin_model, pin_data, q)
    M = pin_data.M.copy()
    # CRBA only fills upper triangle — symmetrize
    M = np.triu(M) + np.triu(M, 1).T
    return M


def compute_coriolis_matrix(
    pin_model, pin_data, q: np.ndarray, qd: np.ndarray
) -> np.ndarray:
    """Compute Coriolis matrix C(q, q̇) via Pinocchio.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        q: Joint angles (6,).
        qd: Joint velocities (6,).

    Returns:
        Coriolis matrix C(q, q̇) [6x6].
    """
    import pinocchio as pin

    pin.computeCoriolisMatrix(pin_model, pin_data, q, qd)
    return pin_data.C.copy()


def compute_gravity_vector(pin_model, pin_data, q: np.ndarray) -> np.ndarray:
    """Compute generalized gravity vector g(q) via Pinocchio.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        q: Joint angles (6,).

    Returns:
        Gravity torque vector g(q) [6].
    """
    import pinocchio as pin

    pin.computeGeneralizedGravity(pin_model, pin_data, q)
    return pin_data.g.copy()


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_gravity(
    pin_model, pin_data, mj_model, mj_data, q: np.ndarray
) -> float:
    """Compare Pinocchio g(q) vs MuJoCo qfrc_bias (at zero velocity).

    At qvel=0, MuJoCo's qfrc_bias equals the gravity vector g(q).

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        q: Joint angles (6,).

    Returns:
        Maximum absolute element-wise error.
    """
    import mujoco

    g_pin = compute_gravity_vector(pin_model, pin_data, q)

    mj_data.qpos[:NUM_JOINTS] = q
    mj_data.qvel[:NUM_JOINTS] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    g_mj = mj_data.qfrc_bias[:NUM_JOINTS].copy()

    err = np.abs(g_pin - g_mj)
    return float(np.max(err))


def cross_validate_mass_matrix(
    pin_model, pin_data, mj_model, mj_data, q: np.ndarray
) -> float:
    """Compare Pinocchio CRBA mass matrix vs MuJoCo mj_fullM().

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        q: Joint angles (6,).

    Returns:
        Maximum absolute element-wise error.
    """
    import mujoco

    M_pin = compute_mass_matrix(pin_model, pin_data, q)

    mj_data.qpos[:NUM_JOINTS] = q
    mj_data.qvel[:NUM_JOINTS] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    M_mj_full = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, M_mj_full, mj_data.qM)
    M_mj = M_mj_full[:NUM_JOINTS, :NUM_JOINTS]

    return float(np.max(np.abs(M_pin - M_mj)))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_dynamics(pin_model, pin_data, save: bool = True) -> None:
    """Generate diagnostic plots for M(q), g(q), and C variation.

    Creates three plots:
      1. M(q) heatmap at Q_HOME
      2. g(q) bar chart per joint at Q_HOME
      3. g(q) variation as shoulder_lift sweeps from -π to 0

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        save: If True, save plots to media/ directory.
    """
    import matplotlib.pyplot as plt

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Mass matrix heatmap at Q_HOME ---
    M = compute_mass_matrix(pin_model, pin_data, Q_HOME)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis", aspect="equal")
    ax.set_title("Mass Matrix M(q) at Q_HOME")
    ax.set_xlabel("Joint index")
    ax.set_ylabel("Joint index")
    labels = [f"J{i}" for i in range(NUM_JOINTS)]
    ax.set_xticks(range(NUM_JOINTS), labels)
    ax.set_yticks(range(NUM_JOINTS), labels)
    for i in range(NUM_JOINTS):
        for j in range(NUM_JOINTS):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] < M.max() * 0.5 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="kg·m²")
    plt.tight_layout()
    if save:
        fig.savefig(MEDIA_DIR / "mass_matrix_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Gravity vector bar chart ---
    g = compute_gravity_vector(pin_model, pin_data, Q_HOME)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if v >= 0 else "#F44336" for v in g]
    bars = ax.bar(range(NUM_JOINTS), g, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Gravity Torque g(q) at Q_HOME")
    ax.set_xlabel("Joint")
    ax.set_ylabel("Torque (Nm)")
    ax.set_xticks(range(NUM_JOINTS), labels)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, g):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val:.2f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=9)
    plt.tight_layout()
    if save:
        fig.savefig(MEDIA_DIR / "gravity_vector_bar.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: g(q) variation as shoulder_lift sweeps ---
    angles = np.linspace(-np.pi, 0, 50)
    g_sweep = np.zeros((len(angles), NUM_JOINTS))
    for i, angle in enumerate(angles):
        q_test = Q_HOME.copy()
        q_test[1] = angle  # shoulder_lift_joint
        g_sweep[i] = compute_gravity_vector(pin_model, pin_data, q_test)

    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(NUM_JOINTS):
        ax.plot(np.degrees(angles), g_sweep[:, j], label=f"J{j}", linewidth=1.5)
    ax.set_title("Gravity Torque vs Shoulder Lift Angle")
    ax.set_xlabel("Shoulder Lift Angle (deg)")
    ax.set_ylabel("Torque (Nm)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(MEDIA_DIR / "gravity_sweep.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {MEDIA_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run dynamics fundamentals: compute, cross-validate, and visualize."""
    pin_model, pin_data, ee_fid = load_pinocchio_model()
    mj_model, mj_data = load_mujoco_model()

    print("=" * 60)
    print("Lab 3 — Dynamics Fundamentals")
    print("=" * 60)

    # Compute dynamics at Q_HOME
    M = compute_mass_matrix(pin_model, pin_data, Q_HOME)
    g = compute_gravity_vector(pin_model, pin_data, Q_HOME)
    C = compute_coriolis_matrix(pin_model, pin_data, Q_HOME, np.zeros(NUM_JOINTS))

    print(f"\nMass matrix M(q) at Q_HOME:\n{M}")
    print(f"\nM(q) eigenvalues: {np.linalg.eigvalsh(M)}")
    print(f"M(q) is symmetric: {np.allclose(M, M.T, atol=1e-10)}")
    print(f"M(q) is positive-definite: {np.all(np.linalg.eigvalsh(M) > 0)}")

    print(f"\nGravity vector g(q) at Q_HOME:\n{g}")
    print(f"\nCoriolis matrix C(q, 0) at Q_HOME (should be ~zero):\n{C}")

    # Cross-validation at multiple configs
    print("\n" + "-" * 60)
    print("Cross-validation: Pinocchio vs MuJoCo")
    print("-" * 60)

    configs = {
        "Q_HOME": Q_HOME,
        "Q_ZEROS": Q_ZEROS,
        "Random 1": np.array([0.3, -1.2, 0.8, -0.6, 1.1, -0.4]),
        "Random 2": np.array([-1.0, -0.5, 1.5, 0.3, -0.8, 2.0]),
        "Random 3": np.array([1.5, -2.0, -0.5, 1.0, 0.5, -1.5]),
    }

    for name, q in configs.items():
        g_err = cross_validate_gravity(pin_model, pin_data, mj_model, mj_data, q)
        M_err = cross_validate_mass_matrix(pin_model, pin_data, mj_model, mj_data, q)
        g_status = "PASS" if g_err < 0.1 else "FAIL"
        M_status = "PASS" if M_err < 0.01 else "FAIL"
        print(f"  {name:12s}: g(q) max_err={g_err:.6f} Nm [{g_status}]  "
              f"M(q) max_err={M_err:.6f} [{M_status}]")

    # Generate plots
    print("\nGenerating plots...")
    plot_dynamics(pin_model, pin_data)

    print("\nDone.")


if __name__ == "__main__":
    main()
