"""Lab 7 — Phase 3 Capstone: Walking demo for the Unitree G1.

Pre-computes a LIPM-based walking trajectory (CoM + foot trajectories),
then executes it in MuJoCo using whole-body IK to convert desired CoM
and foot positions into joint angle targets for the position servo actuators.

Control loop (runs at MuJoCo timestep dt_sim):
    1. Look up desired CoM (x,y) and foot positions from the pre-computed trajectory.
    2. Compute desired pelvis position: pelvis = [com_x, com_y, pelvis_z_stand].
    3. Solve whole-body IK in Pinocchio: leg joints → place feet at desired positions.
    4. Send leg + waist joint targets to MuJoCo ctrl (arms stay at home).
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
    PELVIS_Z_STAND,
    Q_STAND_JOINTS,
    V_LEFT_LEG,
    V_RIGHT_LEG,
    Z_C,
    load_g1_mujoco,
    load_g1_pinocchio,
    mj_state_to_pin,
    pin_q_to_mj,
    pelvis_world_to_pin_base,
)
from lipm_planner import plan_walking_trajectory, plot_lipm_trajectory
from whole_body_ik import WholeBodyIK


# ---------------------------------------------------------------------------
# Walking controller
# ---------------------------------------------------------------------------


class WalkingController:
    """Executes a pre-planned LIPM walking trajectory on the G1 in MuJoCo.

    Args:
        traj: Output dict from plan_walking_trajectory().
        mj_model: MuJoCo model.
        mj_data:  MuJoCo data.
        ik: WholeBodyIK solver.
        dt_ctrl: Control update period [s] (must be multiple of mj timestep).
    """

    def __init__(
        self,
        traj: dict,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        ik: WholeBodyIK,
        dt_ctrl: float = 0.01,
    ) -> None:
        self.traj = traj
        self.m = mj_model
        self.d = mj_data
        self.ik = ik
        self.dt_ctrl = dt_ctrl
        self.dt_sim = mj_model.opt.timestep

        self.n_traj = len(traj["times"])
        # Pelvis height stays constant (LIPM assumption)
        self.pelvis_z = PELVIS_Z_STAND

        # Arm home position (fixed throughout)
        self._arm_ctrl = np.concatenate([
            Q_STAND_JOINTS[15:22],  # left arm
            Q_STAND_JOINTS[22:29],  # right arm
        ])

        # Latest q in Pinocchio convention (updated by IK)
        from lab7_common import build_pin_q_standing
        self._pin_q = build_pin_q_standing(
            pelvis_world_pos=np.array([0.0, 0.0, self.pelvis_z])
        )

        # Logging
        self.log_times: list[float] = []
        self.log_com_actual: list[np.ndarray] = []
        self.log_pelvis_actual: list[np.ndarray] = []
        self.log_ik_converged: list[bool] = []

    def _set_ctrl(self, q_pin: np.ndarray) -> None:
        """Write leg+waist joint angles from a Pinocchio q to MuJoCo ctrl.

        Args:
            q_pin: Full Pinocchio q (nq=36).  Indices 7:22 → leg+waist joints.
        """
        # qpos[7:13] = left leg, [13:19] = right leg, [19:22] = waist
        self.d.ctrl[CTRL_LEFT_LEG] = q_pin[7:13]
        self.d.ctrl[CTRL_RIGHT_LEG] = q_pin[13:19]
        self.d.ctrl[CTRL_WAIST] = q_pin[19:22]
        # Arms stay at home
        self.d.ctrl[CTRL_LEFT_ARM] = self._arm_ctrl[:7]
        self.d.ctrl[CTRL_RIGHT_ARM] = self._arm_ctrl[7:]

    def step(self, traj_idx: int) -> bool:
        """Execute one IK + control update and advance the simulation.

        Args:
            traj_idx: Index into the pre-computed trajectory.

        Returns:
            True if the IK converged, False otherwise.
        """
        traj = self.traj
        com_x_des = traj["com_x"][traj_idx]
        com_y_des = traj["com_y"][traj_idx]
        lfoot_des = traj["foot_pos_l"][traj_idx]
        rfoot_des = traj["foot_pos_r"][traj_idx]

        # Desired pelvis position (CoM moves in x,y; z stays constant)
        pelvis_des = np.array([com_x_des, com_y_des, self.pelvis_z])

        # Solve IK using Pinocchio
        q_sol, converged = self.ik.solve(
            self._pin_q,
            pelvis_des,
            lfoot_des,
            rfoot_des,
            max_iter=100,
            tol=3e-4,
        )
        self._pin_q = q_sol

        # Send to MuJoCo
        self._set_ctrl(q_sol)

        # Advance simulation by dt_ctrl / dt_sim steps
        n_sim_steps = max(1, int(round(self.dt_ctrl / self.dt_sim)))
        for _ in range(n_sim_steps):
            mujoco.mj_step(self.m, self.d)

        # Log
        pelvis_id = self.m.body("pelvis").id
        com_actual = self.d.subtree_com[pelvis_id].copy()
        self.log_times.append(traj["times"][traj_idx])
        self.log_com_actual.append(com_actual)
        self.log_pelvis_actual.append(self.d.xpos[pelvis_id].copy())
        self.log_ik_converged.append(converged)

        return converged

    def run(self, headless: bool = True) -> dict:
        """Run the full walking trajectory.

        Args:
            headless: If False, open a MuJoCo viewer.

        Returns:
            dict with logging data and success metrics.
        """
        viewer = None
        if not headless:
            viewer = mujoco.viewer.launch_passive(self.m, self.d)

        # Phase 1: let the robot settle into standing for 0.5 s
        settle_steps = int(0.5 / self.dt_sim)
        self._set_ctrl(self._pin_q)
        for _ in range(settle_steps):
            mujoco.mj_step(self.m, self.d)
            if viewer:
                viewer.sync()

        n_failed_ik = 0
        for k in range(self.n_traj):
            conv = self.step(k)
            if not conv:
                n_failed_ik += 1
            if viewer:
                viewer.sync()

        if viewer:
            viewer.close()

        # Check success: did the robot travel forward without falling?
        pelvis_id = self.m.body("pelvis").id
        final_pelvis_z = self.d.xpos[pelvis_id][2]
        fell = final_pelvis_z < 0.3  # if pelvis < 30 cm, the robot fell

        log = {
            "times": np.array(self.log_times),
            "com_actual": np.array(self.log_com_actual),
            "pelvis_actual": np.array(self.log_pelvis_actual),
            "ik_converged": np.array(self.log_ik_converged),
            "n_failed_ik": n_failed_ik,
            "fell": fell,
            "final_pelvis_z": float(final_pelvis_z),
            "success": not fell,
        }
        return log


# ---------------------------------------------------------------------------
# Results plotting
# ---------------------------------------------------------------------------


def plot_walking_results(traj: dict, log: dict, save: bool = True) -> None:
    """Plot walking results: planned vs actual CoM trajectory.

    Args:
        traj: Pre-computed walking trajectory dict.
        log:  Output from WalkingController.run().
        save: If True, saves plot to media/.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    times_plan = traj["times"]
    times_act = log["times"]
    com_act = log["com_actual"]
    pelvis_act = log["pelvis_actual"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # X
    ax = axes[0]
    ax.plot(times_plan, traj["com_x"], "b--", lw=1.5, label="Planned CoM x")
    if len(times_act) > 0:
        ax.plot(times_act, com_act[:, 0], "b-", lw=1.5, label="Actual CoM x")
        ax.plot(times_act, pelvis_act[:, 0], "c-", lw=1.0, alpha=0.6, label="Pelvis x")
    ax.set_ylabel("x [m]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_title("G1 Walking Demo — CoM Tracking\n(Blue dashed = planned, solid = actual)")

    # Y
    ax = axes[1]
    ax.plot(times_plan, traj["com_y"], "r--", lw=1.5, label="Planned CoM y")
    if len(times_act) > 0:
        ax.plot(times_act, com_act[:, 1], "r-", lw=1.5, label="Actual CoM y")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # Z (pelvis height — should stay near PELVIS_Z_STAND)
    ax = axes[2]
    if len(times_act) > 0:
        ax.plot(times_act, pelvis_act[:, 2], "g-", lw=1.5, label="Pelvis z (actual)")
    ax.axhline(y=PELVIS_Z_STAND, color="g", linestyle="--", alpha=0.6,
               label=f"Target z = {PELVIS_Z_STAND:.3f} m")
    ax.axhline(y=0.3, color="red", linestyle=":", alpha=0.6, label="Fall threshold")
    ax.set_ylabel("z [m]")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if save:
        MEDIA_DIR.mkdir(exist_ok=True)
        out = MEDIA_DIR / "walking_results.png"
        plt.savefig(out, dpi=150)
        print(f"Plot saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(headless: bool = True) -> None:
    """Run the complete walking capstone demo."""
    print("=" * 60)
    print("Lab 7: G1 Walking Demo")
    print("=" * 60)

    # 1. Plan walking trajectory
    print("\n[1/4] Planning LIPM walking trajectory …")
    traj = plan_walking_trajectory(
        n_steps=12,
        step_length=0.10,
        step_width=0.10,
        step_height=0.05,
        T_ss=0.8,
        T_ds=0.2,
        T_init_ds=0.5,
        T_final_ds=1.0,
        dt=0.01,
        z_c=Z_C,
        Q_e=1.0,
        R=1e-6,
        N_preview=200,
    )
    T_total = traj["times"][-1]
    print(f"  Steps: {len(traj['footsteps'])}, Duration: {T_total:.2f} s")

    # 2. Save LIPM plan plot
    MEDIA_DIR.mkdir(exist_ok=True)
    plot_lipm_trajectory(traj, save_path=MEDIA_DIR / "lipm_trajectory.png")

    # 3. Load MuJoCo + Pinocchio models
    print("\n[2/4] Loading models …")
    mj_model, mj_data = load_g1_mujoco()
    pin_model, pin_data = load_g1_pinocchio()
    ik = WholeBodyIK(pin_model, pin_data, lam=1e-2, step_size=0.5)
    print(f"  MuJoCo: nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")
    print(f"  Pinocchio: nq={pin_model.nq}, nv={pin_model.nv}")

    # 4. Run walking controller
    print("\n[3/4] Executing walking trajectory in MuJoCo …")
    controller = WalkingController(traj, mj_model, mj_data, ik, dt_ctrl=0.01)
    log = controller.run(headless=headless)

    # 5. Results
    print("\n[4/4] Results:")
    print(f"  Robot fell: {log['fell']}")
    print(f"  Final pelvis z: {log['final_pelvis_z']:.3f} m")
    print(f"  IK convergence failures: {log['n_failed_ik']} / {len(traj['times'])}")
    if len(log["times"]) > 0:
        forward_dist = log["pelvis_actual"][-1, 0] - log["pelvis_actual"][0, 0]
        print(f"  Forward distance walked: {forward_dist:.3f} m")

    status = "SUCCESS" if log["success"] else "FAILED (robot fell)"
    print(f"\n  Overall: {status}")

    plot_walking_results(traj, log, save=True)

    return log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lab 7: G1 Walking Demo")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer")
    args = parser.parse_args()

    main(headless=not args.viewer)
