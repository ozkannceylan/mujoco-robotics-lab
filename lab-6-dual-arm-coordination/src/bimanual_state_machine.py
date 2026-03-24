"""Lab 6 — Bimanual pick-carry-place state machine.

Both arms execute each phase SIMULTANEOUSLY.  Trajectory timing is
synchronised so both arms start and finish every segment together.

States:
  IDLE            → arms at Q_HOME, grippers open
  PLAN_APPROACH   → RRT* for both arms (offline, no sim steps)
  EXEC_APPROACH   → joint impedance tracking of synced approach trajectory
  DESCEND         → Cartesian impedance: both EEs descend to grasp height
  CLOSE           → both grippers close; wait for contact + settle
  LIFT            → Cartesian impedance: both EEs lift together
  PLAN_TRANSPORT  → RRT* from current q to q_preplace (both arms)
  EXEC_TRANSPORT  → joint impedance tracking of synced transport trajectory
  DESCEND_PLACE   → Cartesian impedance: both EEs descend to place height
  RELEASE         → both grippers open
  RETRACT         → RRT* back to Q_HOME (both arms)
  DONE            → cycle complete

Cross-lab imports:
  Lab 3: compute_impedance_torque
  Lab 4: CollisionChecker, RRTStarPlanner, shortcut_path, parameterize_topp_ra
  Lab 4: ObstacleSpec
"""

from __future__ import annotations

from enum import Enum, auto

import mujoco
import numpy as np
import pinocchio as pin

from lab6_common import (
    ACC_LIMITS,
    BAR_HALF_Z,
    BAR_PICK_POS,
    DT,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    LEFT_ARM_BASE_POS,
    LEFT_CTRL,
    LEFT_GRIP_CTRL,
    LEFT_QPOS,
    NUM_JOINTS,
    PREGRASP_CLEARANCE,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    RIGHT_ARM_BASE_POS,
    RIGHT_CTRL,
    RIGHT_GRIP_CTRL,
    RIGHT_QPOS,
    URDF_PATH,
    VEL_LIMITS,
    add_lab_src_to_path,
    clip_torques,
    get_ee_pose,
    get_mj_site_id,
)
from dual_arm_ik import BimanualGraspConfigs
from coordination_layer import sync_trajectories
from bimanual_controller import BimanualController

# Cross-lab imports
add_lab_src_to_path("lab-3-dynamics-force-control")
add_lab_src_to_path("lab-4-motion-planning")

from b1_impedance_controller import ImpedanceGains, compute_impedance_torque  # noqa: E402
from collision_checker import CollisionChecker  # noqa: E402
from rrt_planner import RRTStarPlanner  # noqa: E402
from trajectory_smoother import parameterize_topp_ra, shortcut_path  # noqa: E402
from lab4_common import ObstacleSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Table obstacle spec (must match dual_arm_scene.xml)
# ---------------------------------------------------------------------------

_TABLE = ObstacleSpec(
    "table",
    np.array([0.45, 0.0, 0.300]),
    np.array([0.35, 0.50, 0.015]),
)


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE           = auto()
    PLAN_APPROACH  = auto()
    EXEC_APPROACH  = auto()
    DESCEND        = auto()
    CLOSE          = auto()
    LIFT           = auto()
    PLAN_TRANSPORT = auto()
    EXEC_TRANSPORT = auto()
    DESCEND_PLACE  = auto()
    RELEASE        = auto()
    RETRACT        = auto()
    DONE           = auto()


# ---------------------------------------------------------------------------
# Bimanual state machine
# ---------------------------------------------------------------------------

class BimanualStateMachine:
    """Pick-carry-place state machine for two UR5e arms.

    Both arms execute each state simultaneously.  Trajectories are
    synchronised so they share the same time axis.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data (mutated throughout).
        pin_L: Pinocchio model for left arm.
        data_L: Pinocchio data for left arm.
        ee_fid_L: Left arm EE frame ID.
        pin_R: Pinocchio model for right arm.
        data_R: Pinocchio data for right arm.
        ee_fid_R: Right arm EE frame ID.
        cfgs: Pre-computed bimanual grasp configurations.
        Kp_joint: Joint impedance proportional gain.
        Kd_joint: Joint impedance derivative gain.
        Kp_cart: Cartesian impedance translational stiffness.
        Kd_cart: Cartesian impedance translational damping.
    """

    # Timing constants
    DESCEND_SPEED = 0.02   # m/s
    LIFT_SPEED    = 0.05   # m/s
    SETTLE_STEPS  = 500    # max steps for gripper settlement
    GRIPPER_HOLD_DURATION = 1.5  # s — extra hold after close/open

    def __init__(
        self,
        mj_model,
        mj_data,
        pin_L,
        data_L,
        ee_fid_L: int,
        pin_R,
        data_R,
        ee_fid_R: int,
        cfgs: BimanualGraspConfigs,
        Kp_joint: float = 400.0,
        Kd_joint: float = 40.0,
        Kp_cart: float = 600.0,
        Kd_cart: float = 60.0,
    ) -> None:
        self.mj_model  = mj_model
        self.mj_data   = mj_data
        self.pin_L     = pin_L
        self.data_L    = data_L
        self.ee_fid_L  = ee_fid_L
        self.pin_R     = pin_R
        self.data_R    = data_R
        self.ee_fid_R  = ee_fid_R
        self.cfgs      = cfgs

        self.ctrl = BimanualController(
            pin_L, data_L, ee_fid_L,
            pin_R, data_R, ee_fid_R,
            Kp=Kp_joint, Kd=Kd_joint,
            Kp_cart=Kp_cart, Kd_cart=Kd_cart,
        )

        # One collision checker per arm (share the table obstacle)
        self._cc_L = CollisionChecker(
            urdf_path=URDF_PATH,
            obstacle_specs=[_TABLE],
            self_collision=True,
        )
        self._cc_R = CollisionChecker(
            urdf_path=URDF_PATH,
            obstacle_specs=[_TABLE],
            self_collision=True,
        )
        self._planner_L = RRTStarPlanner(
            self._cc_L, step_size=0.3, goal_bias=0.15,
            rewire_radius=1.0, goal_tolerance=0.15,
        )
        self._planner_R = RRTStarPlanner(
            self._cc_R, step_size=0.3, goal_bias=0.15,
            rewire_radius=1.0, goal_tolerance=0.15,
        )

        self.state = State.IDLE
        self._site_id_L = get_mj_site_id(mj_model, "left_gripper_site")
        self._site_id_R = get_mj_site_id(mj_model, "right_gripper_site")

        # Logging
        self._log_time:    list[float]      = []
        self._log_q_L:     list[np.ndarray] = []
        self._log_q_R:     list[np.ndarray] = []
        self._log_ee_L:    list[np.ndarray] = []
        self._log_ee_R:    list[np.ndarray] = []
        self._log_bar_pos: list[np.ndarray] = []
        self._log_state:   list[str]        = []
        self._t = 0.0

        # Bar body ID for tracking
        self._bar_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "carry_bar")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full pick-carry-place cycle.

        Returns:
            Log dict: time, q_L, q_R, ee_L, ee_R, bar_pos, state.
        """
        # Initialise
        self.mj_data.qpos[LEFT_QPOS]   = Q_HOME_LEFT.copy()
        self.mj_data.qpos[RIGHT_QPOS]  = Q_HOME_RIGHT.copy()
        self.mj_data.qvel[LEFT_QPOS]   = 0.0
        self.mj_data.qvel[RIGHT_QPOS]  = 0.0
        self.mj_data.ctrl[LEFT_GRIP_CTRL]  = GRIPPER_OPEN
        self.mj_data.ctrl[RIGHT_GRIP_CTRL] = GRIPPER_OPEN
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.state = State.PLAN_APPROACH

        while self.state != State.DONE:

            if self.state == State.PLAN_APPROACH:
                self._log_transition()
                q_L = self.mj_data.qpos[LEFT_QPOS].copy()
                q_R = self.mj_data.qpos[RIGHT_QPOS].copy()
                self._approach_traj = self._plan_bimanual(
                    q_L, self.cfgs.left.q_pregrasp,
                    q_R, self.cfgs.right.q_pregrasp,
                )
                self.state = State.EXEC_APPROACH

            elif self.state == State.EXEC_APPROACH:
                self._log_transition()
                self._run_joint_impedance(*self._approach_traj)
                self.state = State.DESCEND

            elif self.state == State.DESCEND:
                self._log_transition()
                self._run_joint_descent(mode="descend")
                self.state = State.CLOSE

            elif self.state == State.CLOSE:
                self._log_transition()
                self._run_close_grippers()
                self.state = State.LIFT

            elif self.state == State.LIFT:
                self._log_transition()
                self._run_joint_descent(mode="lift")
                self.state = State.PLAN_TRANSPORT

            elif self.state == State.PLAN_TRANSPORT:
                self._log_transition()
                q_L = self.mj_data.qpos[LEFT_QPOS].copy()
                q_R = self.mj_data.qpos[RIGHT_QPOS].copy()
                self._transport_traj = self._plan_bimanual(
                    q_L, self.cfgs.left.q_preplace,
                    q_R, self.cfgs.right.q_preplace,
                )
                self.state = State.EXEC_TRANSPORT

            elif self.state == State.EXEC_TRANSPORT:
                self._log_transition()
                self._run_joint_impedance(*self._transport_traj)
                self.state = State.DESCEND_PLACE

            elif self.state == State.DESCEND_PLACE:
                self._log_transition()
                self._run_joint_descent(mode="descend_place")
                self.state = State.RELEASE

            elif self.state == State.RELEASE:
                self._log_transition()
                self._run_open_grippers()
                self.state = State.RETRACT

            elif self.state == State.RETRACT:
                self._log_transition()
                q_L = self.mj_data.qpos[LEFT_QPOS].copy()
                q_R = self.mj_data.qpos[RIGHT_QPOS].copy()
                retract_traj = self._plan_bimanual(
                    q_L, Q_HOME_LEFT,
                    q_R, Q_HOME_RIGHT,
                )
                self._run_joint_impedance(*retract_traj)
                self.state = State.DONE

        self._log_transition()
        print("  ✓ Bimanual pick-carry-place complete.")

        return {
            "time":    np.array(self._log_time),
            "q_L":     np.array(self._log_q_L),
            "q_R":     np.array(self._log_q_R),
            "ee_L":    np.array(self._log_ee_L),
            "ee_R":    np.array(self._log_ee_R),
            "bar_pos": np.array(self._log_bar_pos),
            "state":   self._log_state,
        }

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_one(
        self,
        planner: "RRTStarPlanner",
        q_start: np.ndarray,
        q_goal: np.ndarray,
        max_iter: int = 6000,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RRT* + shortcut + TOPP-RA for one arm.

        Args:
            planner: The arm's RRTStarPlanner.
            q_start: Start config (6,).
            q_goal: Goal config (6,).
            max_iter: RRT* iteration budget.
            seed: Random seed.

        Returns:
            (times, q_traj, qd_traj)
        """
        path = planner.plan(q_start, q_goal, max_iter=max_iter,
                            rrt_star=True, seed=seed)
        if path is None:
            raise RuntimeError(
                f"RRT* failed from {q_start.round(2)} to {q_goal.round(2)}")
        path = shortcut_path(path, planner.cc, max_iter=200, seed=seed)
        times, q, qd, _ = parameterize_topp_ra(path, VEL_LIMITS, ACC_LIMITS)
        return times, q, qd

    def _plan_bimanual(
        self,
        q_start_L: np.ndarray,
        q_goal_L: np.ndarray,
        q_start_R: np.ndarray,
        q_goal_R: np.ndarray,
    ) -> tuple:
        """Plan and synchronise trajectories for both arms.

        Returns:
            (t_sync, q_L, qd_L, q_R, qd_R)
        """
        print("    Planning left arm ...")
        traj_L = self._plan_one(self._planner_L, q_start_L, q_goal_L)
        print("    Planning right arm ...")
        traj_R = self._plan_one(self._planner_R, q_start_R, q_goal_R)
        return sync_trajectories(traj_L, traj_R)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _run_joint_impedance(
        self,
        times: np.ndarray,
        q_L:   np.ndarray,
        qd_L:  np.ndarray,
        q_R:   np.ndarray,
        qd_R:  np.ndarray,
    ) -> None:
        """Track a synced bimanual trajectory with joint impedance.

        Args:
            times: Common time axis (N,).
            q_L, qd_L: Left arm desired pos/vel (N×6).
            q_R, qd_R: Right arm desired pos/vel (N×6).
        """
        dur = times[-1]
        total_steps = int((dur + 0.3) / DT) + 1

        for step in range(total_steps):
            t = step * DT
            t_c = min(t, dur)
            q_d_L  = np.array([np.interp(t_c, times, q_L[:,  j]) for j in range(NUM_JOINTS)])
            qd_d_L = np.array([np.interp(t_c, times, qd_L[:, j]) for j in range(NUM_JOINTS)])
            q_d_R  = np.array([np.interp(t_c, times, q_R[:,  j]) for j in range(NUM_JOINTS)])
            qd_d_R = np.array([np.interp(t_c, times, qd_R[:, j]) for j in range(NUM_JOINTS)])
            if t > dur:
                qd_d_L = np.zeros(NUM_JOINTS)
                qd_d_R = np.zeros(NUM_JOINTS)

            self.ctrl.apply_joint_impedance(
                self.mj_data, q_d_L, qd_d_L, q_d_R, qd_d_R)
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._t += DT
            self._record()

    def _run_joint_descent(self, mode: str) -> None:
        """Move both arms through a descent or lift in joint space.

        Uses joint-space impedance to track a smooth linear interpolation
        between two IK configurations.  This is more reliable than Cartesian
        impedance (avoids Jacobian singularity issues and guarantees reaching
        the exact target config).

        Args:
            mode: One of:
                'descend'       — pregrasp → grasp (pick approach)
                'lift'          — grasp → pregrasp (after picking)
                'descend_place' — preplace → place (place approach)
        """
        if mode == "descend":
            q_start_L = self.cfgs.left.q_pregrasp
            q_end_L   = self.cfgs.left.q_grasp
            q_start_R = self.cfgs.right.q_pregrasp
            q_end_R   = self.cfgs.right.q_grasp
            duration  = PREGRASP_CLEARANCE / self.DESCEND_SPEED
        elif mode == "lift":
            q_start_L = self.cfgs.left.q_grasp
            q_end_L   = self.cfgs.left.q_pregrasp
            q_start_R = self.cfgs.right.q_grasp
            q_end_R   = self.cfgs.right.q_pregrasp
            duration  = PREGRASP_CLEARANCE / self.LIFT_SPEED
        else:  # descend_place
            q_start_L = self.cfgs.left.q_preplace
            q_end_L   = self.cfgs.left.q_place
            q_start_R = self.cfgs.right.q_preplace
            q_end_R   = self.cfgs.right.q_place
            duration  = PREGRASP_CLEARANCE / self.DESCEND_SPEED

        n_steps = int(duration / DT) + 1

        for step in range(n_steps):
            alpha = min(step * DT / duration, 1.0)
            q_d_L  = (1 - alpha) * q_start_L + alpha * q_end_L
            q_d_R  = (1 - alpha) * q_start_R + alpha * q_end_R

            self.ctrl.apply_joint_impedance(
                self.mj_data,
                q_d_L, np.zeros(NUM_JOINTS),
                q_d_R, np.zeros(NUM_JOINTS),
            )
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._t += DT
            self._record()

        # Settle: hold final config for 0.5 s
        self._hold(0.5, q_end_L, q_end_R)

    def _run_close_grippers(self) -> None:
        """Close both grippers and wait for contact + settle."""
        self.mj_data.ctrl[LEFT_GRIP_CTRL]  = GRIPPER_CLOSED
        self.mj_data.ctrl[RIGHT_GRIP_CTRL] = GRIPPER_CLOSED

        for _ in range(self.SETTLE_STEPS):
            q_L = self.mj_data.qpos[LEFT_QPOS].copy()
            q_R = self.mj_data.qpos[RIGHT_QPOS].copy()
            self.ctrl.apply_gravity_hold(self.mj_data, q_L, q_R)
            # Preserve gripper ctrl (gravity_hold only writes arm ctrl)
            self.mj_data.ctrl[LEFT_GRIP_CTRL]  = GRIPPER_CLOSED
            self.mj_data.ctrl[RIGHT_GRIP_CTRL] = GRIPPER_CLOSED
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._t += DT
            self._record()
            if self._grippers_settled():
                break

        q_hold_L = self.mj_data.qpos[LEFT_QPOS].copy()
        q_hold_R = self.mj_data.qpos[RIGHT_QPOS].copy()
        self._hold_grippers(self.GRIPPER_HOLD_DURATION, q_hold_L, q_hold_R,
                            GRIPPER_CLOSED, GRIPPER_CLOSED)

        in_contact_L = self._arm_in_contact("left")
        in_contact_R = self._arm_in_contact("right")
        print(f"    Grippers closed — left contact: {in_contact_L}, right: {in_contact_R}")

    def _run_open_grippers(self) -> None:
        """Open both grippers and wait for settlement."""
        self.mj_data.ctrl[LEFT_GRIP_CTRL]  = GRIPPER_OPEN
        self.mj_data.ctrl[RIGHT_GRIP_CTRL] = GRIPPER_OPEN

        for _ in range(self.SETTLE_STEPS):
            q_L = self.mj_data.qpos[LEFT_QPOS].copy()
            q_R = self.mj_data.qpos[RIGHT_QPOS].copy()
            self.ctrl.apply_gravity_hold(self.mj_data, q_L, q_R)
            self.mj_data.ctrl[LEFT_GRIP_CTRL]  = GRIPPER_OPEN
            self.mj_data.ctrl[RIGHT_GRIP_CTRL] = GRIPPER_OPEN
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._t += DT
            self._record()
            if self._grippers_settled():
                break

    def _hold(self, duration: float, q_L: np.ndarray, q_R: np.ndarray) -> None:
        """Hold both arms at fixed configs for a given duration.

        Args:
            duration: Hold duration (s).
            q_L, q_R: Target configs for left/right arm.
        """
        n = int(duration / DT)
        for _ in range(n):
            self.ctrl.apply_gravity_hold(self.mj_data, q_L, q_R)
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._t += DT
            self._record()

    def _hold_grippers(
        self,
        duration: float,
        q_L: np.ndarray,
        q_R: np.ndarray,
        grip_L: float,
        grip_R: float,
    ) -> None:
        """Hold both arms and override gripper setpoints.

        Args:
            duration: Hold duration (s).
            q_L, q_R: Arm configs.
            grip_L, grip_R: Gripper setpoints (m).
        """
        n = int(duration / DT)
        for _ in range(n):
            self.ctrl.apply_gravity_hold(self.mj_data, q_L, q_R)
            self.mj_data.ctrl[LEFT_GRIP_CTRL]  = grip_L
            self.mj_data.ctrl[RIGHT_GRIP_CTRL] = grip_R
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._t += DT
            self._record()

    # ------------------------------------------------------------------
    # Contact / settlement helpers
    # ------------------------------------------------------------------

    def _grippers_settled(self, vel_thresh: float = 5e-4) -> bool:
        """Return True when both gripper finger joints are nearly still.

        Args:
            vel_thresh: Velocity threshold (m/s).

        Returns:
            True if both left and right finger joints are settled.
        """
        for jname in ("left_left_finger_joint", "right_left_finger_joint"):
            jid = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            vel = float(
                self.mj_data.qvel[self.mj_model.jnt_dofadr[jid]])
            if abs(vel) >= vel_thresh:
                return False
        return True

    def _arm_in_contact(self, arm: str) -> bool:
        """Check if any finger geom of the given arm is in contact.

        Args:
            arm: 'left' or 'right'.

        Returns:
            True if at least one finger geom is in contact.
        """
        prefix = arm + "_"
        finger_geom_names = (
            f"{prefix}left_finger_geom", f"{prefix}right_finger_geom",
            f"{prefix}left_pad",          f"{prefix}right_pad",
        )
        finger_ids = {
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in finger_geom_names
        }
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            if c.geom1 in finger_ids or c.geom2 in finger_ids:
                return True
        return False

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _record(self) -> None:
        """Record one timestep of telemetry."""
        self._log_time.append(self._t)
        self._log_q_L.append(self.mj_data.qpos[LEFT_QPOS].copy())
        self._log_q_R.append(self.mj_data.qpos[RIGHT_QPOS].copy())
        self._log_ee_L.append(self.mj_data.site_xpos[self._site_id_L].copy())
        self._log_ee_R.append(self.mj_data.site_xpos[self._site_id_R].copy())
        self._log_bar_pos.append(
            self.mj_data.xpos[self._bar_body_id].copy())
        self._log_state.append(self.state.name)

    def _log_transition(self) -> None:
        """Print state transition to console."""
        print(f"  → {self.state.name}  (t={self._t:.2f}s)")
