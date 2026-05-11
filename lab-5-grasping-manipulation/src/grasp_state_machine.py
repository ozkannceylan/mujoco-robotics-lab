"""Lab 5 — Grasp state machine for pick-and-place.

Orchestrates the full pick-and-place cycle by driving the MuJoCo simulation
step-by-step through distinct states.  Each state corresponds to a specific
sub-controller:

  IDLE            → arm at Q_HOME, gripper open
  PLAN_APPROACH   → RRT* from Q_HOME to q_pregrasp (offline, no steps taken)
  EXEC_APPROACH   → joint-space impedance tracking of approach trajectory
  DESCEND         → Cartesian impedance: EE moves down from pregrasp to grasp height
  CLOSE           → position-actuated gripper closes until settled / contact
  LIFT            → Cartesian impedance: EE moves up (lift height = 0.15 m)
  PLAN_TRANSPORT  → RRT* from current q to q_preplace
  EXEC_TRANSPORT  → joint-space impedance tracking of transport trajectory
  DESCEND_PLACE   → Cartesian impedance: EE moves down to place height
  RELEASE         → gripper opens until settled
  RETRACT         → RRT* from current q back to Q_HOME + joint impedance
  DONE            → full cycle complete

Imports:
  - Lab 3: compute_impedance_torque, ImpedanceGains (Cartesian impedance)
  - Lab 4: CollisionChecker, RRTStarPlanner, shortcut_path, parameterize_topp_ra
"""

from __future__ import annotations

import sys
from enum import Enum, auto
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

# Lab 5 local imports
from lab5_common import (
    ACC_LIMITS,
    DT,
    GRIPPER_IDX,
    GRIPPER_OPEN,
    MODELS_DIR,
    NUM_JOINTS,
    PREGRASP_CLEARANCE,
    Q_HOME,
    TABLE_TOP_Z,
    URDF_PATH,
    VEL_LIMITS,
    add_lab_src_to_path,
    clip_torques,
    get_ee_pose,
    get_mj_site_id,
)
from gripper_controller import (
    close_gripper,
    is_gripper_in_contact,
    is_gripper_settled,
    open_gripper,
)
from grasp_planner import GraspConfigs

# Cross-lab imports
add_lab_src_to_path("lab-3-dynamics-force-control")
add_lab_src_to_path("lab-4-motion-planning")

from b1_impedance_controller import ImpedanceGains, compute_impedance_torque  # noqa: E402
from collision_checker import CollisionChecker  # noqa: E402
from rrt_planner import RRTStarPlanner  # noqa: E402
from trajectory_smoother import parameterize_topp_ra, shortcut_path  # noqa: E402

# Obstacle spec for Lab 5 collision checking (table only, no cluttered obstacles)
from lab4_common import ObstacleSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Table spec for Lab 5 (matches scene_grasp.xml)
# ---------------------------------------------------------------------------

_TABLE5 = ObstacleSpec(
    "table",
    np.array([0.45, 0.0, 0.300]),
    np.array([0.35, 0.45, 0.015]),
)


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE = auto()
    PLAN_APPROACH = auto()
    EXEC_APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    PLAN_TRANSPORT = auto()
    EXEC_TRANSPORT = auto()
    DESCEND_PLACE = auto()
    RELEASE = auto()
    RETRACT = auto()
    DONE = auto()


# ---------------------------------------------------------------------------
# Grasp state machine
# ---------------------------------------------------------------------------

class GraspStateMachine:
    """Full pick-and-place state machine for Lab 5.

    Manages the simulation loop through all pick-and-place states.
    Planning (RRT*) is done offline before simulation steps; execution
    states run the simulation step-by-step with appropriate controllers.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data (will be mutated throughout).
        pin_model: Pinocchio model (arm only, no gripper).
        pin_data: Pinocchio data.
        ee_fid: Pinocchio EE frame ID.
        grasp_cfgs: Pre-computed IK configs for this pick-and-place task.
        Kp_joint: Joint-space impedance proportional gain.
        Kd_joint: Joint-space impedance derivative gain.
        Kp_cart: Cartesian impedance translational stiffness (N/m).
        Kd_cart: Cartesian impedance translational damping.
    """

    # Tunable timing constants
    DESCEND_DURATION = 3.0    # s — impedance descent to grasp height
    LIFT_DURATION = 2.0       # s — impedance lift after grasp
    DESCEND_PLACE_DURATION = 3.0  # s — impedance descent to place height
    SETTLE_STEPS = 500        # max steps to wait for gripper settlement
    GRIPPER_WAIT_DURATION = 1.5  # s — extra stabilisation after close/open

    # Cartesian impedance: descend / lift speeds (m/s step target increment)
    DESCEND_SPEED = 0.02      # m/s (slow approach)
    LIFT_SPEED = 0.05         # m/s (faster lift)

    def __init__(
        self,
        mj_model,
        mj_data,
        pin_model,
        pin_data,
        ee_fid: int,
        grasp_cfgs: GraspConfigs,
        Kp_joint: float = 400.0,
        Kd_joint: float = 40.0,
        Kp_cart: float = 600.0,
        Kd_cart: float = 60.0,
    ) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.ee_fid = ee_fid
        self.cfgs = grasp_cfgs

        self.Kp_joint = Kp_joint
        self.Kd_joint = Kd_joint

        # Cartesian impedance gains (3D translational)
        self.cart_gains = ImpedanceGains(
            K_p=np.diag([Kp_cart] * 3),
            K_d=np.diag([Kd_cart] * 3),
        )

        self.state = State.IDLE
        self._site_id = get_mj_site_id(mj_model, "gripper_site")

        # Build collision checker (table only)
        self._cc = CollisionChecker(
            urdf_path=URDF_PATH,
            obstacle_specs=[_TABLE5],
            self_collision=True,
        )
        self._planner = RRTStarPlanner(
            self._cc,
            step_size=0.3,
            goal_bias=0.15,
            rewire_radius=1.0,
            goal_tolerance=0.15,
        )

        # Logging
        self._log_time: list[float] = []
        self._log_q: list[np.ndarray] = []
        self._log_ee_pos: list[np.ndarray] = []
        self._log_gripper: list[float] = []
        self._log_state: list[str] = []
        self._t = 0.0

    # ---------------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full pick-and-place cycle.

        Steps the MuJoCo simulation through all states in order.

        Returns:
            Log dict with keys: time, q, ee_pos, gripper_pos, state (string per step).
        """
        # Initialise arm at Q_HOME, gripper open
        self.mj_data.qpos[:NUM_JOINTS] = Q_HOME.copy()
        self.mj_data.qvel[:NUM_JOINTS] = 0.0
        self.mj_data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.state = State.PLAN_APPROACH

        while self.state != State.DONE:
            if self.state == State.PLAN_APPROACH:
                self._log_state_transition()
                q_current = self.mj_data.qpos[:NUM_JOINTS].copy()
                self._approach_traj = self._plan_and_smooth(q_current, self.cfgs.q_pregrasp)
                self.state = State.EXEC_APPROACH

            elif self.state == State.EXEC_APPROACH:
                self._log_state_transition()
                self._run_joint_impedance(*self._approach_traj)
                self.state = State.DESCEND

            elif self.state == State.DESCEND:
                self._log_state_transition()
                self._run_cartesian_descend(
                    target_z=self.cfgs.q_grasp,  # will use EE Z, computed inside
                    mode="descend",
                )
                self.state = State.CLOSE

            elif self.state == State.CLOSE:
                self._log_state_transition()
                self._run_close_gripper()
                self.state = State.LIFT

            elif self.state == State.LIFT:
                self._log_state_transition()
                self._run_cartesian_descend(
                    target_z=None,
                    mode="lift",
                )
                self.state = State.PLAN_TRANSPORT

            elif self.state == State.PLAN_TRANSPORT:
                self._log_state_transition()
                q_current = self.mj_data.qpos[:NUM_JOINTS].copy()
                self._transport_traj = self._plan_and_smooth(q_current, self.cfgs.q_preplace)
                self.state = State.EXEC_TRANSPORT

            elif self.state == State.EXEC_TRANSPORT:
                self._log_state_transition()
                self._run_joint_impedance(*self._transport_traj)
                self.state = State.DESCEND_PLACE

            elif self.state == State.DESCEND_PLACE:
                self._log_state_transition()
                self._run_cartesian_descend(
                    target_z=self.cfgs.q_place,
                    mode="descend",
                )
                self.state = State.RELEASE

            elif self.state == State.RELEASE:
                self._log_state_transition()
                self._run_open_gripper()
                self.state = State.RETRACT

            elif self.state == State.RETRACT:
                self._log_state_transition()
                q_current = self.mj_data.qpos[:NUM_JOINTS].copy()
                retract_traj = self._plan_and_smooth(q_current, Q_HOME)
                self._run_joint_impedance(*retract_traj)
                self.state = State.DONE

        self._log_state_transition()
        print("  ✓ Pick-and-place cycle complete.")

        return {
            "time": np.array(self._log_time),
            "q": np.array(self._log_q),
            "ee_pos": np.array(self._log_ee_pos),
            "gripper_pos": np.array(self._log_gripper),
            "state": self._log_state,
        }

    # ---------------------------------------------------------------------------
    # State implementations
    # ---------------------------------------------------------------------------

    def _plan_and_smooth(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        max_iter: int = 6000,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run RRT*, shortcut, and TOPP-RA; return (times, q_traj, qd_traj).

        Args:
            q_start: Start configuration (6,).
            q_goal: Goal configuration (6,).
            max_iter: RRT* iteration budget.
            seed: Random seed.

        Returns:
            Tuple of (times [N], q_traj [N×6], qd_traj [N×6]).

        Raises:
            RuntimeError: If RRT* fails to find a path.
        """
        path = self._planner.plan(q_start, q_goal, max_iter=max_iter,
                                  rrt_star=True, seed=seed)
        if path is None:
            raise RuntimeError(
                f"RRT* failed to find path from {q_start} to {q_goal}."
            )
        path = shortcut_path(path, self._cc, max_iter=200, seed=seed)
        times, q_traj, qd_traj, _ = parameterize_topp_ra(path, VEL_LIMITS, ACC_LIMITS)
        return times, q_traj, qd_traj

    def _run_joint_impedance(
        self,
        times: np.ndarray,
        q_traj: np.ndarray,
        qd_traj: np.ndarray,
    ) -> None:
        """Execute trajectory with joint-space impedance + gravity compensation.

        Tracks (q_d(t), qd_d(t)) via:
            τ = Kp*(q_d - q) + Kd*(qd_d - qd) + g(q)

        Args:
            times: Time array (N,) in seconds.
            q_traj: Desired joint positions (N×6).
            qd_traj: Desired joint velocities (N×6).
        """
        traj_duration = times[-1]
        total_steps = int((traj_duration + 0.3) / DT) + 1

        for step in range(total_steps):
            t_local = step * DT
            t_abs = t_local

            if t_local <= traj_duration:
                q_d = np.array(
                    [np.interp(t_local, times, q_traj[:, j]) for j in range(NUM_JOINTS)]
                )
                qd_d = np.array(
                    [np.interp(t_local, times, qd_traj[:, j]) for j in range(NUM_JOINTS)]
                )
            else:
                q_d = q_traj[-1]
                qd_d = np.zeros(NUM_JOINTS)

            q = self.mj_data.qpos[:NUM_JOINTS].copy()
            qd = self.mj_data.qvel[:NUM_JOINTS].copy()

            pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
            g = self.pin_data.g.copy()

            tau = (self.Kp_joint * (q_d - q)
                   + self.Kd_joint * (qd_d - qd)
                   + g)
            tau = clip_torques(tau)

            self.mj_data.ctrl[:NUM_JOINTS] = tau
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._t += DT
            self._record()

    def _run_cartesian_descend(
        self,
        target_z,
        mode: str,
    ) -> None:
        """Run Cartesian impedance to move EE along world Z.

        mode='descend': moves EE downward by PREGRASP_CLEARANCE metres.
        mode='lift': moves EE upward by PREGRASP_CLEARANCE metres.

        Args:
            target_z: Unused (kept for signature clarity). Target is derived
                from current EE position + vertical offset.
            mode: 'descend' or 'lift'.
        """
        q_cur = self.mj_data.qpos[:NUM_JOINTS].copy()
        x_start, R_des = get_ee_pose(self.pin_model, self.pin_data, self.ee_fid, q_cur)

        direction = -1.0 if mode == "descend" else 1.0
        total_z = PREGRASP_CLEARANCE
        speed = self.DESCEND_SPEED if mode == "descend" else self.LIFT_SPEED
        duration = total_z / speed
        n_steps = int(duration / DT) + 1

        for step in range(n_steps):
            progress = min(step * DT * speed, total_z)
            x_des = x_start + np.array([0.0, 0.0, direction * progress])

            q = self.mj_data.qpos[:NUM_JOINTS].copy()
            qd = self.mj_data.qvel[:NUM_JOINTS].copy()

            tau = compute_impedance_torque(
                self.pin_model, self.pin_data, self.ee_fid,
                q, qd,
                x_des, R_des, None,
                self.cart_gains,
            )
            tau = clip_torques(tau)
            self.mj_data.ctrl[:NUM_JOINTS] = tau
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._t += DT
            self._record()

        # Settle: hold final position for 0.5 s
        self._hold_position(duration=0.5, q_hold=self.mj_data.qpos[:NUM_JOINTS].copy())

    def _run_close_gripper(self) -> None:
        """Close gripper and wait for settlement, then hold.

        The arm is held at its current joint configuration with joint impedance
        while the gripper closes, preventing reaction forces from drifting the
        arm away from the object before contact is confirmed.
        """
        close_gripper(self.mj_data)
        q_hold_close = self.mj_data.qpos[:NUM_JOINTS].copy()
        settle_steps = self.SETTLE_STEPS
        contact_during_settle = False
        for _ in range(settle_steps):
            q = self.mj_data.qpos[:NUM_JOINTS].copy()
            qd = self.mj_data.qvel[:NUM_JOINTS].copy()
            pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
            g = self.pin_data.g.copy()
            tau = clip_torques(
                self.Kp_joint * (q_hold_close - q)
                + self.Kd_joint * (0.0 - qd)
                + g
            )
            self.mj_data.ctrl[:NUM_JOINTS] = tau
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._t += DT
            self._record()
            if is_gripper_in_contact(self.mj_model, self.mj_data):
                contact_during_settle = True
            if is_gripper_settled(self.mj_model, self.mj_data):
                break

        # Extra hold to let contact forces stabilise
        q_hold = self.mj_data.qpos[:NUM_JOINTS].copy()
        self._hold_position(duration=self.GRIPPER_WAIT_DURATION, q_hold=q_hold)

        in_contact = is_gripper_in_contact(self.mj_model, self.mj_data) or contact_during_settle
        print(f"    Gripper closed — contact: {in_contact}")

    def _run_open_gripper(self) -> None:
        """Open gripper and wait for settlement."""
        open_gripper(self.mj_data)
        for _ in range(self.SETTLE_STEPS):
            q = self.mj_data.qpos[:NUM_JOINTS].copy()
            pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
            g = self.pin_data.g.copy()
            tau = clip_torques(g)
            self.mj_data.ctrl[:NUM_JOINTS] = tau
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._t += DT
            self._record()
            if is_gripper_settled(self.mj_model, self.mj_data):
                break

    def _hold_position(self, duration: float, q_hold: np.ndarray) -> None:
        """Hold arm at q_hold with joint-space impedance for a given duration.

        Args:
            duration: Hold duration (s).
            q_hold: Joint configuration to hold (6,).
        """
        n_steps = int(duration / DT)
        for _ in range(n_steps):
            q = self.mj_data.qpos[:NUM_JOINTS].copy()
            qd = self.mj_data.qvel[:NUM_JOINTS].copy()
            pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
            g = self.pin_data.g.copy()
            tau = clip_torques(
                self.Kp_joint * (q_hold - q)
                + self.Kd_joint * (0.0 - qd)
                + g
            )
            self.mj_data.ctrl[:NUM_JOINTS] = tau
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._t += DT
            self._record()

    # ---------------------------------------------------------------------------
    # Logging helpers
    # ---------------------------------------------------------------------------

    def _record(self) -> None:
        """Record one timestep of data."""
        self._log_time.append(self._t)
        self._log_q.append(self.mj_data.qpos[:NUM_JOINTS].copy())
        ee = self.mj_data.site_xpos[self._site_id].copy()
        self._log_ee_pos.append(ee)

        jid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_joint"
        )
        self._log_gripper.append(
            float(self.mj_data.qpos[self.mj_model.jnt_qposadr[jid]])
        )
        self._log_state.append(self.state.name)

    def _log_state_transition(self) -> None:
        """Print state transition to console."""
        print(f"  → State: {self.state.name}  (t={self._t:.2f}s)")
