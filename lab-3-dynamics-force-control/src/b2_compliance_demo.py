"""Lab 3 — Phase 2: Tunable Compliance Demo.

Compares soft, medium, and stiff impedance controllers on the same task
with an external perturbation, demonstrating the compliance trade-off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab3_common import (
    NUM_JOINTS,
    Q_HOME,
    MEDIA_DIR,
    get_ee_pose,
    load_pinocchio_model,
)
from b1_impedance_controller import ImpedanceGains, run_impedance_sim


def run_compliance_comparison(
    x_des: np.ndarray,
    perturb_at: float = 1.5,
    perturb_force: np.ndarray | None = None,
    duration: float = 4.0,
) -> dict:
    """Run impedance tracking with soft/medium/stiff gains and perturbation.

    Args:
        x_des: Target EE position (3,).
        perturb_at: Time to apply perturbation (s).
        perturb_force: Cartesian force perturbation at EE (3,).
        duration: Total sim time (s).

    Returns:
        Dictionary keyed by stiffness label with simulation results.
    """
    if perturb_force is None:
        perturb_force = np.array([40.0, 0.0, 0.0])

    configs = {
        "Soft (K=100)": ImpedanceGains(
            K_p=np.diag([100.0, 100.0, 100.0]),
            K_d=np.diag([20.0, 20.0, 20.0]),
        ),
        "Medium (K=500)": ImpedanceGains(
            K_p=np.diag([500.0, 500.0, 500.0]),
            K_d=np.diag([50.0, 50.0, 50.0]),
        ),
        "Stiff (K=2000)": ImpedanceGains(
            K_p=np.diag([2000.0, 2000.0, 2000.0]),
            K_d=np.diag([100.0, 100.0, 100.0]),
        ),
    }

    all_results = {}
    for label, gains in configs.items():
        print(f"  Running {label}...")
        results = run_impedance_sim(
            x_des=x_des,
            gains=gains,
            duration=duration,
            perturb_at=perturb_at,
            perturb_force=perturb_force,
            perturb_duration=0.1,
        )
        all_results[label] = results

    return all_results


def plot_compliance_comparison(
    all_results: dict,
    save: bool = True,
) -> None:
    """Generate comparison plots for soft/medium/stiff impedance.

    Args:
        all_results: Dict from run_compliance_comparison().
        save: If True, save to media/.
    """
    import matplotlib.pyplot as plt

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"Soft (K=100)": "#4CAF50", "Medium (K=500)": "#2196F3",
              "Stiff (K=2000)": "#F44336"}

    # --- Plot 1: Position tracking error norm ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    for label, results in all_results.items():
        t = results["time"]
        err_norm = np.linalg.norm(results["ee_pos_err"], axis=1) * 1000
        ax.plot(t, err_norm, label=label, color=colors[label], linewidth=1.5)
    ax.set_ylabel("Position Error (mm)")
    ax.set_title("Impedance Compliance Comparison — Position Error")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axvline(1.5, color="gray", linestyle=":", alpha=0.5, label="Perturbation")

    # --- Plot 2: Total torque norm ---
    ax = axes[1]
    for label, results in all_results.items():
        t = results["time"]
        tau_norm = np.linalg.norm(results["tau"], axis=1)
        ax.plot(t, tau_norm, label=label, color=colors[label], linewidth=1.5)
    ax.set_ylabel("Torque Norm (Nm)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Impedance Compliance Comparison — Joint Torques")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axvline(1.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save:
        fig.savefig(MEDIA_DIR / "compliance_comparison.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Per-axis displacement during perturbation ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axis_labels = ["X", "Y", "Z"]

    for j in range(3):
        ax = axes[j]
        for label, results in all_results.items():
            t = results["time"]
            err_j = results["ee_pos_err"][:, j] * 1000
            ax.plot(t, err_j, label=label, color=colors[label], linewidth=1.5)
        ax.set_ylabel(f"{axis_labels[j]} Error (mm)")
        ax.grid(alpha=0.3)
        ax.axvline(1.5, color="gray", linestyle=":", alpha=0.5)
        if j == 0:
            ax.set_title("Per-Axis Position Error During Perturbation")
            ax.legend(fontsize=9)
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    if save:
        fig.savefig(MEDIA_DIR / "compliance_per_axis.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run tunable compliance demo."""
    print("=" * 60)
    print("Lab 3 — Tunable Compliance Demo")
    print("=" * 60)

    pin_model, pin_data, ee_fid = load_pinocchio_model()
    x_home, _ = get_ee_pose(pin_model, pin_data, ee_fid, Q_HOME)

    # Target: offset from home
    x_target = x_home + np.array([0.0, 0.0, 0.10])
    print(f"Target: {x_target}")
    print(f"Perturbation: 40N in X at t=1.5s for 0.1s\n")

    all_results = run_compliance_comparison(
        x_des=x_target,
        perturb_at=1.5,
        perturb_force=np.array([40.0, 0.0, 0.0]),
        duration=4.0,
    )

    # Print summary
    print("\n--- Summary ---")
    for label, results in all_results.items():
        # Steady-state error (last 100 steps before perturbation)
        ss_idx = int(1.4 / 0.001)
        ss_err = np.linalg.norm(results["ee_pos_err"][ss_idx]) * 1000

        # Max deflection during perturbation
        perturb_start = int(1.5 / 0.001)
        perturb_end = int(2.5 / 0.001)
        max_deflection = np.max(
            np.linalg.norm(results["ee_pos_err"][perturb_start:perturb_end], axis=1)
        ) * 1000

        # Max torque during perturbation
        max_tau = np.max(np.linalg.norm(results["tau"][perturb_start:perturb_end], axis=1))

        # Final error
        final_err = np.linalg.norm(results["ee_pos_err"][-1]) * 1000

        print(f"  {label}:")
        print(f"    Pre-perturb SS error: {ss_err:.2f} mm")
        print(f"    Max deflection:       {max_deflection:.2f} mm")
        print(f"    Max torque norm:      {max_tau:.1f} Nm")
        print(f"    Final error:          {final_err:.2f} mm")

    print("\nGenerating plots...")
    plot_compliance_comparison(all_results)
    print(f"Plots saved to {MEDIA_DIR}/")


if __name__ == "__main__":
    main()
