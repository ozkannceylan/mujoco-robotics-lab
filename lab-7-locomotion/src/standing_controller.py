"""Lab 7 — Phase 1: Standing balance controller with perturbation test.

Verifies that the G1's position servo actuators (kp=500) maintain stable
upright standing and recover after a horizontal push force.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab7_common import (
    CTRL_LEFT_ARM,
    CTRL_LEFT_LEG,
    CTRL_RIGHT_ARM,
    CTRL_RIGHT_LEG,
    CTRL_WAIST,
    MEDIA_DIR,
    Q_STAND_JOINTS,
    Z_C,
    load_g1_mujoco,
)


def run_standing_demo(
    duration_s: float = 6.0,
    perturbation_force_n: float = 5.0,
    perturbation_time_s: float = 3.0,
    perturbation_duration_s: float = 0.2,
    headless: bool = True,
) -> dict:
    """Run the standing balance demo and return CoM trajectory.

    The robot starts from the "stand" keyframe, holds position via
    kp=500 position actuators, and receives a 5 N horizontal push
    at t=3 s. The function records the CoM (x, y, z) and checks recovery.

    Args:
        duration_s: Total simulation duration.
        perturbation_force_n: Magnitude of the horizontal push force [N].
        perturbation_time_s: Time at which the push is applied [s].
        perturbation_duration_s: Duration of the push [s].
        headless: If False, opens a MuJoCo viewer window.

    Returns:
        dict with keys: 'times', 'com_x', 'com_y', 'com_z', 'success'
    """
    m, d = load_g1_mujoco()

    # Apply standing joint targets (arms + legs)
    d.ctrl[CTRL_LEFT_LEG] = Q_STAND_JOINTS[0:6]
    d.ctrl[CTRL_RIGHT_LEG] = Q_STAND_JOINTS[6:12]
    d.ctrl[CTRL_WAIST] = Q_STAND_JOINTS[12:15]
    d.ctrl[CTRL_LEFT_ARM] = Q_STAND_JOINTS[15:22]
    d.ctrl[CTRL_RIGHT_ARM] = Q_STAND_JOINTS[22:29]

    pelvis_id = m.body("pelvis").id

    dt = m.opt.timestep  # 0.002 s
    n_steps = int(duration_s / dt)

    times = np.zeros(n_steps)
    com_x = np.zeros(n_steps)
    com_y = np.zeros(n_steps)
    com_z = np.zeros(n_steps)

    t_pert_end = perturbation_time_s + perturbation_duration_s

    if not headless:
        viewer = mujoco.viewer.launch_passive(m, d)

    for i in range(n_steps):
        t = i * dt

        # Apply perturbation force on pelvis (world-frame x direction)
        if perturbation_time_s <= t < t_pert_end:
            d.xfrc_applied[pelvis_id, 0] = perturbation_force_n  # Fx [N]
        else:
            d.xfrc_applied[pelvis_id, :] = 0.0

        mujoco.mj_step(m, d)

        # Record CoM (using pelvis subtree which includes all child bodies)
        com = d.subtree_com[pelvis_id]
        times[i] = t
        com_x[i] = com[0]
        com_y[i] = com[1]
        com_z[i] = com[2]

        if not headless:
            viewer.sync()

    if not headless:
        viewer.close()

    # Check recovery: CoM should return close to initial position
    com_init = np.array([com_x[0], com_y[0], com_z[0]])
    com_final = np.array([com_x[-1], com_y[-1], com_z[-1]])
    recovery_error = float(np.linalg.norm(com_final - com_init))
    success = recovery_error < 0.015  # within 15 mm

    print(f"Standing demo: duration={duration_s}s, perturbation={perturbation_force_n}N at t={perturbation_time_s}s")
    print(f"  Initial CoM: {com_init}")
    print(f"  Final CoM:   {com_final}")
    print(f"  Recovery error: {recovery_error*1000:.1f} mm → {'PASS' if success else 'FAIL'}")
    print(f"  Min z: {com_z.min():.4f} m  (LIPM z_c reference: {Z_C:.3f} m)")

    result = {
        "times": times,
        "com_x": com_x,
        "com_y": com_y,
        "com_z": com_z,
        "success": success,
        "recovery_error_mm": recovery_error * 1000,
    }
    return result


def plot_standing_results(result: dict, save: bool = True) -> None:
    """Plot CoM trajectory from standing demo.

    Args:
        result: Output from run_standing_demo().
        save: If True, saves plot to media/.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    times = result["times"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ["CoM x [m]", "CoM y [m]", "CoM z [m]"]
    data = [result["com_x"], result["com_y"], result["com_z"]]
    colors = ["steelblue", "darkorange", "forestgreen"]

    for ax, label, vals, color in zip(axes, labels, data, colors):
        ax.plot(times, vals, color=color, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.4)
        ax.axvline(x=3.0, color="red", linestyle="--", alpha=0.7, label="perturbation")
        ax.axvline(x=3.2, color="red", linestyle=":", alpha=0.7, label="end push")
        if label.startswith("CoM z"):
            ax.axhline(y=Z_C, color="purple", linestyle="--", alpha=0.5, label=f"z_c={Z_C}")

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("G1 Standing Balance — CoM Trajectory\n(Red dashed: 5 N push at t=3 s)")
    axes[0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if save:
        MEDIA_DIR.mkdir(exist_ok=True)
        out = MEDIA_DIR / "standing_com_trajectory.png"
        plt.savefig(out, dpi=150)
        print(f"Plot saved: {out}")
    plt.show()


if __name__ == "__main__":
    result = run_standing_demo(duration_s=6.0)
    plot_standing_results(result)
