"""Lab 3 — Phase 3: Hybrid Position-Force Controller.

Implements hybrid control: position control in XY, force control in Z.
Uses contact force measurement from MuJoCo for closed-loop force regulation.

Control law:
  Position (XY): F_p = K_p · S_p · (x_d - x) + K_d · S_p · (ẋ_d - ẋ)
  Force (Z):     F_f = S_f · (F_d + K_fp · e_f + K_fi · ∫e_f dt)
  Combined:      τ = J^T · (F_p + F_f) + g(q)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mujoco
import pinocchio as pin

from lab3_common import (
    DT,
    NUM_JOINTS,
    Q_HOME,
    MEDIA_DIR,
    SCENE_TABLE_PATH,
    clip_torques,
    get_ee_pose,
    load_mujoco_model,
    load_pinocchio_model,
)

# Starting config with EE at (0.4, 0.0, 0.35) — above table center
Q_ABOVE_TABLE = np.array([1.991929, -2.39295, 2.502256, -0.798855, -6.07376, 0.0])


# ---------------------------------------------------------------------------
# Gains
# ---------------------------------------------------------------------------

@dataclass
class HybridGains:
    """Gains for hybrid position-force controller."""
    K_p: float = 500.0     # Position stiffness (N/m) for XY
    K_d: float = 50.0      # Position damping (N·s/m) for XY
    K_fp: float = 5.0      # Force proportional gain
    K_fi: float = 10.0     # Force integral gain
    F_desired: float = 5.0  # Desired contact force (N), positive = into surface


# ---------------------------------------------------------------------------
# Contact force measurement
# ---------------------------------------------------------------------------

def read_contact_force_z(
    mj_model, mj_data, ee_body_ids: list[int], table_geom_id: int
) -> float:
    """Read the Z-component of contact force between EE bodies and table.

    Sums normal contact forces from all contacts involving any EE body geom
    and the table geom. Returns positive force when EE pushes into table
    (contact normal points upward, force pushes down).

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        ee_body_ids: List of body IDs whose geoms are considered "EE".
        table_geom_id: Geom ID of the table surface.

    Returns:
        Contact force in Z (positive = pressing into surface).
    """
    total_fz = 0.0
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        g1, g2 = c.geom1, c.geom2

        # Check if this contact involves EE and table
        is_ee_table = False
        ee_geom = -1
        if g1 == table_geom_id and mj_model.geom_bodyid[g2] in ee_body_ids:
            is_ee_table = True
            ee_geom = g2
        elif g2 == table_geom_id and mj_model.geom_bodyid[g1] in ee_body_ids:
            is_ee_table = True
            ee_geom = g1

        if not is_ee_table:
            continue

        # Get contact force (6D wrench in contact frame)
        wrench = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, wrench)

        # Normal force is wrench[0], contact normal is frame[:3]
        normal = c.frame[:3].copy()
        normal_force = wrench[0]

        # Project onto world Z: positive when pushing into table (downward)
        # Contact normal points from geom1 to geom2
        # Force is positive along normal direction
        fz = normal_force * normal[2]

        # If table is geom1, normal points away from table (upward),
        # and positive force = EE pushing into table → fz > 0 means pressing
        # If table is geom2, normal points toward table, flip sign
        if g2 == table_geom_id:
            fz = -fz

        total_fz += fz

    return abs(total_fz)


def get_ee_and_table_ids(mj_model):
    """Get body/geom IDs for EE body and table geom.

    Only counts tool0 body contacts (the EE tip), not wrist or forearm links
    that may also contact the table.

    Returns:
        Tuple of (ee_body_ids, table_geom_id).
    """
    tool0_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tool0")
    table_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    return [tool0_bid], table_gid


# ---------------------------------------------------------------------------
# Hybrid position-force controller
# ---------------------------------------------------------------------------

def compute_hybrid_torque(
    pin_model,
    pin_data,
    ee_fid: int,
    q: np.ndarray,
    qd: np.ndarray,
    xy_des: np.ndarray,
    xyd_des: np.ndarray | None,
    F_measured: float,
    force_integral: float,
    gains: HybridGains,
) -> tuple[np.ndarray, float]:
    """Compute joint torques for hybrid position-force control.

    Position control in XY, PI force control in Z.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
        q: Joint angles (6,).
        qd: Joint velocities (6,).
        xy_des: Desired XY position (2,).
        xyd_des: Desired XY velocity (2,) or None.
        F_measured: Measured contact force in Z (positive = pressing).
        force_integral: Accumulated force error integral.
        gains: HybridGains.

    Returns:
        Tuple of (joint_torques (6,), updated_force_integral).
    """
    # FK + Jacobian
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    pin.computeJointJacobians(pin_model, pin_data, q)

    oMf = pin_data.oMf[ee_fid]
    x_cur = oMf.translation.copy()

    J = pin.getFrameJacobian(
        pin_model, pin_data, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )[:3, :]  # translational Jacobian only

    v_cur = J @ qd

    # Gravity compensation
    pin.computeGeneralizedGravity(pin_model, pin_data, q)
    g = pin_data.g.copy()

    # --- Position control in XY ---
    pos_err_xy = np.zeros(3)
    pos_err_xy[0] = xy_des[0] - x_cur[0]
    pos_err_xy[1] = xy_des[1] - x_cur[1]
    # Z component is zero (force controlled)

    vel_des = np.zeros(3)
    if xyd_des is not None:
        vel_des[0] = xyd_des[0]
        vel_des[1] = xyd_des[1]
    vel_err = vel_des - v_cur
    vel_err[2] = 0.0  # don't damp Z velocity from position side

    F_pos = gains.K_p * pos_err_xy + gains.K_d * vel_err

    # --- Force control in Z (PI + velocity damping) ---
    force_err = gains.F_desired - F_measured
    force_integral += force_err * DT

    # Anti-windup: clamp integral
    max_integral = 20.0 / max(gains.K_fi, 1e-6)
    force_integral = np.clip(force_integral, -max_integral, max_integral)

    F_force = np.zeros(3)
    # Negative Z force = push downward into table
    F_force[2] = -(gains.K_fp * force_err + gains.K_fi * force_integral)

    # Z-velocity damping to reduce contact chattering
    K_dz = 30.0
    F_force[2] -= K_dz * v_cur[2]

    # Combined
    F_total = F_pos + F_force
    tau = J.T @ F_total + g

    return tau, float(force_integral)


def solve_ik_xy(
    pin_model, pin_data, ee_fid: int,
    xy: np.ndarray, z: float, q_init: np.ndarray,
    max_iter: int = 50,
) -> np.ndarray:
    """Solve IK for EE at (xy[0], xy[1], z) starting from q_init.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
        xy: Target XY (2,).
        z: Target Z height.
        q_init: Starting joint config.
        max_iter: Maximum iterations.

    Returns:
        Joint configuration (6,).
    """
    x_target = np.array([xy[0], xy[1], z])
    q = q_init.copy()

    for _ in range(max_iter):
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        pin.computeJointJacobians(pin_model, pin_data, q)

        x_cur = pin_data.oMf[ee_fid].translation.copy()
        err = x_target - x_cur
        if np.linalg.norm(err) < 1e-4:
            break

        J = pin.getFrameJacobian(
            pin_model, pin_data, ee_fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3, :]
        dq = np.linalg.lstsq(J, err, rcond=None)[0]
        q = q + dq * 0.5
        q = np.clip(q, pin_model.lowerPositionLimit, pin_model.upperPositionLimit)

    return q


def compute_hybrid_torque_joint(
    pin_model,
    pin_data,
    ee_fid: int,
    q: np.ndarray,
    qd: np.ndarray,
    q_des: np.ndarray,
    F_measured: float,
    force_integral: float,
    gains: HybridGains,
) -> tuple[np.ndarray, float]:
    """Hybrid torque using joint-space PD for position + Cartesian PI for force.

    Position control uses joint-space PD tracking of IK-solved targets.
    Force control uses Cartesian PI in Z mapped through J^T.

    Args:
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        ee_fid: EE frame ID.
        q: Current joint angles (6,).
        qd: Current joint velocities (6,).
        q_des: Desired joint angles from IK (6,).
        F_measured: Measured contact force in Z (positive = pressing).
        force_integral: Accumulated force error integral.
        gains: HybridGains (K_p used as joint stiffness, K_d as joint damping).

    Returns:
        Tuple of (joint_torques (6,), updated_force_integral).
    """
    # FK + Jacobian
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    pin.computeJointJacobians(pin_model, pin_data, q)

    J = pin.getFrameJacobian(
        pin_model, pin_data, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )[:3, :]
    v_cur = J @ qd

    # Gravity compensation
    pin.computeGeneralizedGravity(pin_model, pin_data, q)
    g = pin_data.g.copy()

    # --- Joint-space PD for position ---
    tau_pos = gains.K_p * (q_des - q) + gains.K_d * (0.0 - qd)

    # --- Force control in Z (PI + velocity damping) ---
    force_err = gains.F_desired - F_measured
    force_integral += force_err * DT

    max_integral = 20.0 / max(gains.K_fi, 1e-6)
    force_integral = np.clip(force_integral, -max_integral, max_integral)

    F_force = np.zeros(3)
    F_force[2] = -(gains.K_fp * force_err + gains.K_fi * force_integral)

    K_dz = 30.0
    F_force[2] -= K_dz * v_cur[2]

    # Combined: joint PD + J^T force + gravity
    tau = tau_pos + J.T @ F_force + g

    return tau, float(force_integral)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_hybrid_force_sim(
    xy_des: np.ndarray,
    q_start: np.ndarray | None = None,
    gains: HybridGains | None = None,
    approach_height: float = 0.25,
    duration: float = 5.0,
) -> Dict[str, np.ndarray]:
    """Run hybrid position-force control simulation.

    Phase 1 (approach): impedance control descends EE to approach_height,
    then slowly lowers toward table surface.
    Phase 2 (contact): once contact detected, switches to hybrid mode.

    Args:
        xy_des: Desired XY position on table (2,).
        q_start: Starting joint config. If None, uses Q_HOME.
        gains: HybridGains. Defaults to standard gains.
        approach_height: Z height to start approach from (m).
        duration: Total sim duration (s).

    Returns:
        Dict with: time, q, qd, tau, ee_pos, force_z, phase
        (phase: 0=approach, 1=contact/hybrid)
    """
    if gains is None:
        gains = HybridGains()
    if q_start is None:
        q_start = Q_ABOVE_TABLE.copy()

    pin_model, pin_data, ee_fid = load_pinocchio_model()
    mj_model, mj_data = load_mujoco_model(SCENE_TABLE_PATH)
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    ee_body_ids, table_geom_id = get_ee_and_table_ids(mj_model)

    # Table surface Z
    table_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_z = mj_model.body_pos[table_body][2] + 0.02  # box half-height

    # Initialize
    mj_data.qpos[:NUM_JOINTS] = q_start
    mj_data.qvel[:NUM_JOINTS] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    n_steps = int(duration / DT)
    time_arr = np.zeros(n_steps)
    q_arr = np.zeros((n_steps, NUM_JOINTS))
    qd_arr = np.zeros((n_steps, NUM_JOINTS))
    tau_arr = np.zeros((n_steps, NUM_JOINTS))
    ee_pos_arr = np.zeros((n_steps, 3))
    force_z_arr = np.zeros(n_steps)
    phase_arr = np.zeros(n_steps, dtype=int)

    force_integral = 0.0
    contact_established = False
    contact_time = None
    fz_filtered = 0.0  # Low-pass filtered force
    alpha_filter = 0.2  # EMA filter coefficient (lower = smoother)

    # Approach gains (impedance for descent)
    approach_Kp = np.diag([800.0, 800.0, 800.0])
    approach_Kd = np.diag([80.0, 80.0, 80.0])

    # Descent speed
    descent_speed = 0.05  # m/s

    for i in range(n_steps):
        t = i * DT
        time_arr[i] = t

        q = mj_data.qpos[:NUM_JOINTS].copy()
        qd = mj_data.qvel[:NUM_JOINTS].copy()
        q_arr[i] = q
        qd_arr[i] = qd

        # Read raw contact force and apply low-pass filter
        fz_raw = read_contact_force_z(mj_model, mj_data, ee_body_ids, table_geom_id)
        fz_filtered = alpha_filter * fz_raw + (1 - alpha_filter) * fz_filtered
        fz = fz_filtered
        force_z_arr[i] = fz

        # Current EE position
        ee_pos = mj_data.site_xpos[site_id].copy()
        ee_pos_arr[i] = ee_pos

        # Check contact
        if not contact_established and fz > 0.5:
            contact_established = True
            contact_time = t

        if not contact_established:
            # --- APPROACH PHASE: impedance descent ---
            phase_arr[i] = 0

            # Target: XY desired, Z descending slowly
            z_target = max(approach_height - descent_speed * t, table_z - 0.01)
            x_target = np.array([xy_des[0], xy_des[1], z_target])

            # Impedance control
            pin.forwardKinematics(pin_model, pin_data, q)
            pin.updateFramePlacements(pin_model, pin_data)
            pin.computeJointJacobians(pin_model, pin_data, q)

            oMf = pin_data.oMf[ee_fid]
            x_cur = oMf.translation.copy()
            J = pin.getFrameJacobian(
                pin_model, pin_data, ee_fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]
            v_cur = J @ qd

            pos_err = x_target - x_cur
            vel_err = -v_cur  # desired velocity ~ 0 for impedance
            F = approach_Kp @ pos_err + approach_Kd @ vel_err

            pin.computeGeneralizedGravity(pin_model, pin_data, q)
            tau = J.T @ F + pin_data.g.copy()
        else:
            # --- HYBRID PHASE: position XY + force Z ---
            phase_arr[i] = 1

            tau, force_integral = compute_hybrid_torque(
                pin_model, pin_data, ee_fid,
                q, qd, xy_des, None,
                fz, force_integral, gains,
            )

        tau = clip_torques(tau)
        tau_arr[i] = tau
        mj_data.ctrl[:NUM_JOINTS] = tau
        mujoco.mj_step(mj_model, mj_data)

    if contact_time is not None:
        print(f"  Contact established at t={contact_time:.3f}s")
    else:
        print("  WARNING: No contact established!")

    return {
        "time": time_arr,
        "q": q_arr,
        "qd": qd_arr,
        "tau": tau_arr,
        "ee_pos": ee_pos_arr,
        "force_z": force_z_arr,
        "phase": phase_arr,
    }


def plot_hybrid_results(
    results: Dict[str, np.ndarray],
    xy_des: np.ndarray,
    F_desired: float = 5.0,
    save: bool = True,
    filename: str = "hybrid_force_control",
) -> None:
    """Plot hybrid position-force control results.

    Args:
        results: Output from run_hybrid_force_sim().
        xy_des: Desired XY position (2,).
        F_desired: Desired force for reference line.
        save: If True, save to media/.
        filename: Base filename.
    """
    import matplotlib.pyplot as plt

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    t = results["time"]
    phase = results["phase"]
    contact_start = t[np.argmax(phase > 0)] if np.any(phase > 0) else None

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Force
    ax = axes[0]
    ax.plot(t, results["force_z"], "b-", linewidth=1.5, label="Measured")
    ax.axhline(F_desired, color="r", linestyle="--", label=f"Desired ({F_desired}N)")
    ax.axhline(F_desired - 1, color="r", linestyle=":", alpha=0.3)
    ax.axhline(F_desired + 1, color="r", linestyle=":", alpha=0.3)
    if contact_start:
        ax.axvline(contact_start, color="gray", linestyle=":", alpha=0.5, label="Contact")
    ax.set_ylabel("Force Z (N)")
    ax.set_title("Hybrid Position-Force Control")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # EE XY position
    ax = axes[1]
    ax.plot(t, results["ee_pos"][:, 0] * 1000, label="X")
    ax.plot(t, results["ee_pos"][:, 1] * 1000, label="Y")
    ax.axhline(xy_des[0] * 1000, color="C0", linestyle="--", alpha=0.5)
    ax.axhline(xy_des[1] * 1000, color="C1", linestyle="--", alpha=0.5)
    if contact_start:
        ax.axvline(contact_start, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("XY Position (mm)")
    ax.set_title("XY Position Tracking")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # EE Z position
    ax = axes[2]
    ax.plot(t, results["ee_pos"][:, 2] * 1000, "g-", linewidth=1.5)
    ax.axhline(170, color="brown", linestyle="--", alpha=0.5, label="Table surface")
    if contact_start:
        ax.axvline(contact_start, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Z Position (mm)")
    ax.set_title("EE Height (Z)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Torques
    ax = axes[3]
    for j in range(NUM_JOINTS):
        ax.plot(t, results["tau"][:, j], label=f"J{j}", linewidth=0.8)
    if contact_start:
        ax.axvline(contact_start, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Torque (Nm)")
    ax.set_xlabel("Time (s)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(MEDIA_DIR / f"{filename}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run hybrid force control demo."""
    print("=" * 60)
    print("Lab 3 — Hybrid Position-Force Control")
    print("=" * 60)

    # Target: above table center
    xy_des = np.array([0.4, 0.0])
    gains = HybridGains(K_p=500.0, K_d=50.0, K_fp=5.0, K_fi=10.0, F_desired=5.0)

    print(f"\nTarget XY: {xy_des}, F_desired: {gains.F_desired} N")
    print("Running approach + hybrid control (8s)...")

    results = run_hybrid_force_sim(
        xy_des=xy_des,
        gains=gains,
        approach_height=0.30,
        duration=8.0,
    )

    # Analyze hybrid phase
    phase = results["phase"]
    hybrid_mask = phase == 1
    if np.any(hybrid_mask):
        fz_hybrid = results["force_z"][hybrid_mask]
        # Skip first 0.5s of hybrid for settling
        hybrid_start = np.argmax(hybrid_mask)
        settle_idx = min(hybrid_start + 500, len(results["force_z"]) - 1)
        fz_settled = results["force_z"][settle_idx:]

        print(f"\n--- Hybrid Phase Analysis ---")
        print(f"  Mean force (after settling): {np.mean(fz_settled):.2f} N")
        print(f"  Std force: {np.std(fz_settled):.2f} N")
        print(f"  Force within 5±1N: {np.mean(np.abs(fz_settled - 5.0) < 1.0)*100:.1f}%")

        xy_err = np.sqrt(
            (results["ee_pos"][settle_idx:, 0] - xy_des[0])**2 +
            (results["ee_pos"][settle_idx:, 1] - xy_des[1])**2
        ) * 1000
        print(f"  Mean XY error: {np.mean(xy_err):.2f} mm")
        print(f"  Max XY error: {np.max(xy_err):.2f} mm")
    else:
        print("\nWARNING: Hybrid phase never reached!")

    print("\nGenerating plots...")
    plot_hybrid_results(results, xy_des, gains.F_desired)
    print(f"Plots saved to {MEDIA_DIR}/")


if __name__ == "__main__":
    main()
