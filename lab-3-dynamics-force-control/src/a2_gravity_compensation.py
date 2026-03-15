"""Lab 3 — Phase 1: Gravity Compensation Controller.

Implements τ = g(q): applying gravity torques so the robot "floats" in place.
Includes simulation with optional perturbation to verify the controller holds position.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab3_common import (
    DT,
    NUM_JOINTS,
    Q_HOME,
    MEDIA_DIR,
    clip_torques,
    load_mujoco_model,
    load_pinocchio_model,
)


def gravity_compensation_torque(
    pin_model, pin_data, q: np.ndarray
) -> np.ndarray:
    """Compute gravity compensation torque τ = g(q).

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        q: Joint angles (6,).

    Returns:
        Joint torques that exactly cancel gravity (6,).
    """
    import pinocchio as pin

    pin.computeGeneralizedGravity(pin_model, pin_data, q)
    return pin_data.g.copy()


def run_gravity_comp_sim(
    duration: float = 5.0,
    q_init: np.ndarray | None = None,
    perturb_at: float | None = None,
    perturb_torque: np.ndarray | None = None,
    perturb_duration: float = 0.2,
) -> Dict[str, np.ndarray]:
    """Run gravity compensation simulation.

    Args:
        duration: Total simulation time (seconds).
        q_init: Initial joint configuration. Defaults to Q_HOME.
        perturb_at: Time (s) at which to apply an external perturbation.
        perturb_torque: External torque to apply during perturbation (6,).
        perturb_duration: Duration of the perturbation pulse (seconds).

    Returns:
        Dictionary with keys:
          - time: timestamps (N,)
          - q: joint positions (N, 6)
          - qd: joint velocities (N, 6)
          - tau: applied torques (N, 6)
          - q_error: position error from initial config (N, 6)
    """
    import mujoco

    if q_init is None:
        q_init = Q_HOME.copy()
    if perturb_torque is None:
        perturb_torque = np.array([0.0, 20.0, 10.0, 0.0, 0.0, 0.0])

    pin_model, pin_data, _ = load_pinocchio_model()
    mj_model, mj_data = load_mujoco_model()

    # Set initial configuration
    mj_data.qpos[:NUM_JOINTS] = q_init
    mj_data.qvel[:NUM_JOINTS] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    n_steps = int(duration / DT)
    time_arr = np.zeros(n_steps)
    q_arr = np.zeros((n_steps, NUM_JOINTS))
    qd_arr = np.zeros((n_steps, NUM_JOINTS))
    tau_arr = np.zeros((n_steps, NUM_JOINTS))

    for i in range(n_steps):
        t = i * DT
        time_arr[i] = t

        # Read state
        q = mj_data.qpos[:NUM_JOINTS].copy()
        qd = mj_data.qvel[:NUM_JOINTS].copy()
        q_arr[i] = q
        qd_arr[i] = qd

        # Gravity compensation
        tau = gravity_compensation_torque(pin_model, pin_data, q)

        # Apply perturbation if requested
        if perturb_at is not None and perturb_at <= t < perturb_at + perturb_duration:
            tau += perturb_torque

        tau = clip_torques(tau)
        tau_arr[i] = tau

        # Apply and step
        mj_data.ctrl[:NUM_JOINTS] = tau
        mujoco.mj_step(mj_model, mj_data)

    q_error = q_arr - q_init[np.newaxis, :]

    return {
        "time": time_arr,
        "q": q_arr,
        "qd": qd_arr,
        "tau": tau_arr,
        "q_error": q_error,
    }


def plot_gravity_comp_results(
    results: Dict[str, np.ndarray],
    title_suffix: str = "",
    save: bool = True,
    filename: str = "gravity_comp",
) -> None:
    """Plot gravity compensation simulation results.

    Args:
        results: Output from run_gravity_comp_sim().
        title_suffix: Extra text for plot title.
        save: If True, save to media/.
        filename: Base filename for saved plots.
    """
    import matplotlib.pyplot as plt

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    t = results["time"]
    q_err = results["q_error"]
    tau = results["tau"]
    qd = results["qd"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Position error
    ax = axes[0]
    for j in range(NUM_JOINTS):
        ax.plot(t, np.degrees(q_err[:, j]), label=f"J{j}", linewidth=1)
    ax.set_ylabel("Position Error (deg)")
    ax.set_title(f"Gravity Compensation{title_suffix}")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    # Velocity
    ax = axes[1]
    for j in range(NUM_JOINTS):
        ax.plot(t, np.degrees(qd[:, j]), label=f"J{j}", linewidth=1)
    ax.set_ylabel("Joint Velocity (deg/s)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    # Torques
    ax = axes[2]
    for j in range(NUM_JOINTS):
        ax.plot(t, tau[:, j], label=f"J{j}", linewidth=1)
    ax.set_ylabel("Torque (Nm)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(MEDIA_DIR / f"{filename}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run gravity compensation demo: hold position + perturbation recovery."""
    print("=" * 60)
    print("Lab 3 — Gravity Compensation Controller")
    print("=" * 60)

    # Test 1: Hold position at Q_HOME for 5 seconds
    print("\nTest 1: Holding Q_HOME under gravity compensation (5s)...")
    results = run_gravity_comp_sim(duration=5.0)
    max_err_deg = np.max(np.abs(np.degrees(results["q_error"])))
    max_err_rad = np.max(np.abs(results["q_error"]))
    print(f"  Max position error: {max_err_deg:.4f} deg ({max_err_rad:.6f} rad)")
    status = "PASS" if max_err_rad < 0.01 else "FAIL"
    print(f"  Criterion: < 0.01 rad → [{status}]")
    plot_gravity_comp_results(results, title_suffix=" — Holding Q_HOME",
                              filename="gravity_comp_hold")

    # Test 2: Perturbation recovery
    print("\nTest 2: Perturbation at t=1.0s, then hold (5s)...")
    results_perturb = run_gravity_comp_sim(
        duration=5.0,
        perturb_at=1.0,
        perturb_torque=np.array([0.0, 20.0, 10.0, 5.0, 0.0, 0.0]),
        perturb_duration=0.2,
    )
    # After perturbation, arm should drift but settle at new position
    # Check that it's not still moving at the end
    final_vel = np.max(np.abs(results_perturb["qd"][-100:]))
    print(f"  Final velocity (last 100 steps): {np.degrees(final_vel):.4f} deg/s")
    # With damping=1.0, the arm should have very low velocity at end
    vel_status = "PASS" if final_vel < 0.1 else "FAIL"
    print(f"  Criterion: < 0.1 rad/s → [{vel_status}]")
    plot_gravity_comp_results(results_perturb, title_suffix=" — With Perturbation",
                              filename="gravity_comp_perturb")

    print(f"\nPlots saved to {MEDIA_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
