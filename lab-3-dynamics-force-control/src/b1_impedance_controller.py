"""Lab 3 — Phase 2: Cartesian Impedance Controller.

Implements task-space impedance control:
  F = K_p · Δx + K_d · Δẋ   (position + orientation)
  τ = J^T · F + g(q)

Supports translational-only (3D) and full 6D (position + orientation) modes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pinocchio as pin

from lab3_common import (
    DT,
    NUM_JOINTS,
    Q_HOME,
    MEDIA_DIR,
    clip_torques,
    get_ee_pose,
    load_mujoco_model,
    load_pinocchio_model,
)


# ---------------------------------------------------------------------------
# Gains dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImpedanceGains:
    """Impedance controller gains.

    For translational-only mode, use 3x3 matrices.
    For full 6D mode, use 6x6 matrices (upper-left 3x3 = translation,
    lower-right 3x3 = orientation).
    """
    K_p: np.ndarray = field(default_factory=lambda: np.diag([500.0, 500.0, 500.0]))
    K_d: np.ndarray = field(default_factory=lambda: np.diag([50.0, 50.0, 50.0]))

    @staticmethod
    def make_6d(
        K_pos: float = 500.0,
        K_ori: float = 50.0,
        D_pos: float = 50.0,
        D_ori: float = 5.0,
    ) -> "ImpedanceGains":
        """Create 6D gains with scalar stiffness/damping for position and orientation.

        Args:
            K_pos: Translational stiffness (N/m).
            K_ori: Rotational stiffness (Nm/rad).
            D_pos: Translational damping (N·s/m).
            D_ori: Rotational damping (Nm·s/rad).

        Returns:
            ImpedanceGains with 6x6 K_p and K_d.
        """
        K_p = np.diag([K_pos, K_pos, K_pos, K_ori, K_ori, K_ori])
        K_d = np.diag([D_pos, D_pos, D_pos, D_ori, D_ori, D_ori])
        return ImpedanceGains(K_p=K_p, K_d=K_d)


# ---------------------------------------------------------------------------
# Orientation error
# ---------------------------------------------------------------------------

def orientation_error(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Compute orientation error between desired and current rotation matrices.

    Uses the skew-symmetric extraction method:
      e_R = 0.5 * vee(R_d^T R - R^T R_d)

    This gives a 3D vector whose magnitude is the angle error and direction
    is the rotation axis, valid for small errors. For large errors it provides
    a smooth gradient toward the target.

    Args:
        R_des: Desired rotation matrix [3x3].
        R_cur: Current rotation matrix [3x3].

    Returns:
        Orientation error vector in world frame (3,).
    """
    R_err = R_des.T @ R_cur - R_cur.T @ R_des
    # Extract vector from skew-symmetric matrix (vee map)
    e = 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])
    return e


# ---------------------------------------------------------------------------
# Impedance controller
# ---------------------------------------------------------------------------

def compute_impedance_torque(
    pin_model,
    pin_data,
    ee_fid: int,
    q: np.ndarray,
    qd: np.ndarray,
    x_des: np.ndarray,
    R_des: np.ndarray | None,
    xd_des: np.ndarray | None,
    gains: ImpedanceGains,
) -> np.ndarray:
    """Compute joint torques from Cartesian impedance control.

    Supports two modes based on gains dimensions:
      - 3x3 gains: translational-only (position control in 3D)
      - 6x6 gains: full 6D (position + orientation)

    Control law:
      F = K_p · error + K_d · velocity_error
      τ = J^T · F + g(q)

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: End-effector frame ID.
        q: Joint angles (6,).
        qd: Joint velocities (6,).
        x_des: Desired EE position (3,).
        R_des: Desired EE rotation [3x3]. Required for 6D mode, ignored for 3D.
        xd_des: Desired EE velocity (6,) or (3,) or None (zero).
        gains: ImpedanceGains with K_p, K_d.

    Returns:
        Joint torques (6,).
    """
    dim = gains.K_p.shape[0]
    is_6d = dim == 6

    # FK + Jacobian
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    pin.computeJointJacobians(pin_model, pin_data, q)

    oMf = pin_data.oMf[ee_fid]
    x_cur = oMf.translation.copy()
    R_cur = oMf.rotation.copy()

    J_full = pin.getFrameJacobian(
        pin_model, pin_data, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    # Compute current EE velocity
    v_cur = J_full @ qd  # [linear(3), angular(3)]

    # Gravity compensation
    pin.computeGeneralizedGravity(pin_model, pin_data, q)
    g = pin_data.g.copy()

    if is_6d:
        # 6D error
        pos_err = x_des - x_cur
        ori_err = orientation_error(R_des, R_cur)
        error = np.concatenate([pos_err, ori_err])

        # Desired velocity (default zero)
        if xd_des is None:
            xd_des_full = np.zeros(6)
        elif xd_des.shape[0] == 3:
            xd_des_full = np.concatenate([xd_des, np.zeros(3)])
        else:
            xd_des_full = xd_des

        vel_err = xd_des_full - v_cur

        # Impedance force (6D)
        F = gains.K_p @ error + gains.K_d @ vel_err

        # Joint torques
        tau = J_full.T @ F + g
    else:
        # 3D translational only
        pos_err = x_des - x_cur
        J_trans = J_full[:3, :]

        if xd_des is None:
            xd_des_3 = np.zeros(3)
        else:
            xd_des_3 = xd_des[:3]

        vel_err = xd_des_3 - v_cur[:3]

        F = gains.K_p @ pos_err + gains.K_d @ vel_err

        tau = J_trans.T @ F + g

    return tau


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_impedance_sim(
    x_des: np.ndarray,
    R_des: np.ndarray | None = None,
    gains: ImpedanceGains | None = None,
    q_init: np.ndarray | None = None,
    duration: float = 3.0,
    perturb_at: float | None = None,
    perturb_force: np.ndarray | None = None,
    perturb_duration: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Run an impedance control simulation to a Cartesian target.

    Args:
        x_des: Target EE position (3,).
        R_des: Target EE rotation [3x3]. If None, uses initial orientation.
        gains: Impedance gains. Defaults to K_p=500, K_d=50 (3D).
        q_init: Initial joint config. Defaults to Q_HOME.
        duration: Simulation duration (s).
        perturb_at: Time to apply perturbation (s).
        perturb_force: Cartesian force perturbation (3,) applied at EE.
        perturb_duration: Duration of perturbation pulse (s).

    Returns:
        Dictionary with keys: time, q, qd, tau, ee_pos, ee_pos_err,
        ee_ori_err (if 6D).
    """
    import mujoco

    if gains is None:
        gains = ImpedanceGains()
    if q_init is None:
        q_init = Q_HOME.copy()
    if perturb_force is None:
        perturb_force = np.array([30.0, 0.0, 0.0])

    is_6d = gains.K_p.shape[0] == 6

    pin_model, pin_data, ee_fid = load_pinocchio_model()
    mj_model, mj_data = load_mujoco_model()
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")

    # Set initial config
    mj_data.qpos[:NUM_JOINTS] = q_init
    mj_data.qvel[:NUM_JOINTS] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # If no target orientation, use initial
    if R_des is None and is_6d:
        _, R_des = get_ee_pose(pin_model, pin_data, ee_fid, q_init)

    n_steps = int(duration / DT)
    time_arr = np.zeros(n_steps)
    q_arr = np.zeros((n_steps, NUM_JOINTS))
    qd_arr = np.zeros((n_steps, NUM_JOINTS))
    tau_arr = np.zeros((n_steps, NUM_JOINTS))
    ee_pos_arr = np.zeros((n_steps, 3))
    ee_pos_err_arr = np.zeros((n_steps, 3))
    ee_ori_err_arr = np.zeros((n_steps, 3)) if is_6d else None

    for i in range(n_steps):
        t = i * DT
        time_arr[i] = t

        q = mj_data.qpos[:NUM_JOINTS].copy()
        qd = mj_data.qvel[:NUM_JOINTS].copy()
        q_arr[i] = q
        qd_arr[i] = qd

        # Impedance torque
        tau = compute_impedance_torque(
            pin_model, pin_data, ee_fid, q, qd,
            x_des, R_des, None, gains,
        )

        # External perturbation via Jacobian transpose
        if perturb_at is not None and perturb_at <= t < perturb_at + perturb_duration:
            pin.computeJointJacobians(pin_model, pin_data, q)
            J_trans = pin.getFrameJacobian(
                pin_model, pin_data, ee_fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]
            tau += J_trans.T @ perturb_force

        tau = clip_torques(tau)
        tau_arr[i] = tau

        mj_data.ctrl[:NUM_JOINTS] = tau
        mujoco.mj_step(mj_model, mj_data)

        # Record EE position
        ee_pos = mj_data.site_xpos[site_id].copy()
        ee_pos_arr[i] = ee_pos
        ee_pos_err_arr[i] = x_des - ee_pos

        if is_6d and ee_ori_err_arr is not None:
            _, R_cur = get_ee_pose(pin_model, pin_data, ee_fid, q)
            ee_ori_err_arr[i] = orientation_error(R_des, R_cur)

    result: Dict[str, np.ndarray] = {
        "time": time_arr,
        "q": q_arr,
        "qd": qd_arr,
        "tau": tau_arr,
        "ee_pos": ee_pos_arr,
        "ee_pos_err": ee_pos_err_arr,
    }
    if ee_ori_err_arr is not None:
        result["ee_ori_err"] = ee_ori_err_arr
    return result


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Run impedance controller demos: 3D position-only and 6D with orientation."""
    import matplotlib.pyplot as plt

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    # Get EE position at Q_HOME as reference
    x_home, R_home = get_ee_pose(pin_model, pin_data, ee_fid, Q_HOME)
    print(f"EE at Q_HOME: pos={x_home}, R=\n{R_home}")

    # Target: 10cm forward from home
    x_target = x_home + np.array([0.0, 0.0, 0.10])

    # --- Demo 1: 3D translational impedance ---
    print("\n" + "=" * 60)
    print("Demo 1: 3D Translational Impedance (K=500, D=50)")
    print("=" * 60)
    gains_3d = ImpedanceGains(
        K_p=np.diag([500.0, 500.0, 500.0]),
        K_d=np.diag([50.0, 50.0, 50.0]),
    )
    results_3d = run_impedance_sim(x_des=x_target, gains=gains_3d, duration=3.0)

    final_err = results_3d["ee_pos_err"][-1]
    final_err_mm = np.linalg.norm(final_err) * 1000
    print(f"  Final position error: {final_err_mm:.2f} mm")
    print(f"  Final error components: {final_err * 1000} mm")

    # --- Demo 2: 6D impedance with orientation ---
    print("\n" + "=" * 60)
    print("Demo 2: 6D Impedance (K_pos=500, K_ori=50)")
    print("=" * 60)
    gains_6d = ImpedanceGains.make_6d(K_pos=500.0, K_ori=50.0, D_pos=50.0, D_ori=5.0)
    results_6d = run_impedance_sim(
        x_des=x_target, R_des=R_home, gains=gains_6d, duration=3.0,
    )

    final_err_6d = results_6d["ee_pos_err"][-1]
    final_err_6d_mm = np.linalg.norm(final_err_6d) * 1000
    final_ori_err = np.linalg.norm(results_6d["ee_ori_err"][-1])
    print(f"  Final position error: {final_err_6d_mm:.2f} mm")
    print(f"  Final orientation error: {np.degrees(final_ori_err):.3f} deg")

    # --- Plot: 3D tracking ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    t = results_3d["time"]
    labels = ["X", "Y", "Z"]
    for j in range(3):
        axes[0].plot(t, results_3d["ee_pos"][:, j] * 1000, label=f"{labels[j]} actual")
        axes[0].axhline(x_target[j] * 1000, color=f"C{j}", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Position (mm)")
    axes[0].set_title("3D Impedance: EE Position Tracking")
    axes[0].legend(loc="right", fontsize=8)
    axes[0].grid(alpha=0.3)

    for j in range(3):
        axes[1].plot(t, results_3d["ee_pos_err"][:, j] * 1000, label=labels[j])
    axes[1].set_ylabel("Error (mm)")
    axes[1].set_title("Position Error")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for j in range(NUM_JOINTS):
        axes[2].plot(t, results_3d["tau"][:, j], label=f"J{j}", linewidth=0.8)
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Joint Torques")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(MEDIA_DIR / "impedance_3d_tracking.png", dpi=150)
    plt.close(fig)

    # --- Plot: 6D tracking with orientation ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    t6 = results_6d["time"]
    for j in range(3):
        axes[0].plot(t6, results_6d["ee_pos_err"][:, j] * 1000, label=labels[j])
    axes[0].set_ylabel("Position Error (mm)")
    axes[0].set_title("6D Impedance: Position Error")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for j in range(3):
        axes[1].plot(t6, np.degrees(results_6d["ee_ori_err"][:, j]),
                     label=["Rx", "Ry", "Rz"][j])
    axes[1].set_ylabel("Orientation Error (deg)")
    axes[1].set_title("6D Impedance: Orientation Error")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for j in range(NUM_JOINTS):
        axes[2].plot(t6, results_6d["tau"][:, j], label=f"J{j}", linewidth=0.8)
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(MEDIA_DIR / "impedance_6d_tracking.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {MEDIA_DIR}/")


if __name__ == "__main__":
    main()
