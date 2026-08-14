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
    BOX_B_POS,
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
    get_mj_body_id,
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
from rrt_planner import RRTStarPlanner  # noqa: E402
from trajectory_smoother import parameterize_topp_ra, shortcut_path  # noqa: E402

# ---------------------------------------------------------------------------
# Scene-derived collision checker
# ---------------------------------------------------------------------------

class SceneCollisionChecker:
    """Collision checker built from the Lab 5 scene itself (scene_grasp.xml).

    Duck-types the Lab 4 ``CollisionChecker`` planner interface
    (``is_collision_free`` / ``is_path_free``) so RRT* and shortcutting work
    unchanged, but evaluates collisions against the *simulated* robot's own
    geometry instead of the Lab 4 Menagerie UR5e + Robotiq stack.

    Why this deviation from Lab 4's "real-geometry truth" (Step 6.1): the
    Menagerie UR5e's upper arm is thicker than this lab's simplified box-geom
    arm, so the Lab 4 checker rejects the natural B-side solution family
    (upper arm 9 mm inside the table *for the real robot*, clear for this
    scene's robot) and forces the planner toward a family 5.4 rad away that
    RRT* cannot reliably bridge while carrying the box. A planner must plan
    for the robot it drives; for the capstone that robot is scene_grasp.xml.

    The grasp box is excluded from every check (it legitimately rests on the
    table and is inside the gripper during transport). Finger joints are held
    half-open so closed-finger pad contact does not read as self-collision.
    """

    _FINGER_CHECK_POS = 0.015  # m — half-open fingers during checks

    def __init__(self, mj_model) -> None:
        self.mj_model = mj_model
        self._data = mujoco.MjData(mj_model)

        def bid(name: str) -> int:
            return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

        self._box_bid = bid("grasp_box")
        # Robot bodies = subtree rooted at the arm base
        base_bid = bid("base")
        self._robot_bids = set()
        for b in range(mj_model.nbody):
            p = b
            while p != 0:
                if p == base_bid:
                    self._robot_bids.add(b)
                    break
                p = mj_model.body_parentid[p]

    def _tree_distance(self, b1: int, b2: int) -> int:
        """Number of parent hops between two bodies in the kinematic tree."""
        chain1 = []
        p = b1
        while p != 0:
            chain1.append(p)
            p = self.mj_model.body_parentid[p]
        chain1.append(0)
        p, hops = b2, 0
        while p not in chain1:
            p = self.mj_model.body_parentid[p]
            hops += 1
        return hops + chain1.index(p)

    def is_collision_free(self, q: np.ndarray) -> bool:
        """True if the arm at q touches neither the environment nor itself."""
        d = self._data
        d.qpos[:] = 0.0
        d.qpos[:NUM_JOINTS] = q
        d.qpos[NUM_JOINTS:NUM_JOINTS + 2] = self._FINGER_CHECK_POS
        # Park the box far away so it cannot shadow arm-table contacts
        d.qpos[NUM_JOINTS + 2:NUM_JOINTS + 5] = (2.0, 2.0, 0.5)
        d.qpos[NUM_JOINTS + 5] = 1.0  # unit quaternion w
        d.qpos[NUM_JOINTS + 6:NUM_JOINTS + 9] = 0.0
        mujoco.mj_forward(self.mj_model, d)

        for cid in range(d.ncon):
            c = d.contact[cid]
            if c.dist >= 0:
                continue
            b1 = self.mj_model.geom_bodyid[c.geom1]
            b2 = self.mj_model.geom_bodyid[c.geom2]
            if self._box_bid in (b1, b2):
                continue
            r1, r2 = b1 in self._robot_bids, b2 in self._robot_bids
            if r1 and r2:
                if self._tree_distance(b1, b2) <= 1:
                    continue  # adjacent links (Lab 4 adjacency_gap rule)
                return False
            if r1 or r2:
                return False  # robot vs table / floor / pad
        return True

    def is_path_free(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        resolution: float = 0.05,
    ) -> bool:
        """Discretised straight-line check, same contract as Lab 4."""
        diff = q2 - q1
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return self.is_collision_free(q1)
        n_steps = max(2, int(np.ceil(dist / resolution)) + 1)
        for i in range(n_steps):
            q = q1 + (i / (n_steps - 1)) * diff
            if not self.is_collision_free(q):
                return False
        return True


def make_collision_checker(mj_model=None) -> SceneCollisionChecker:
    """Build the capstone's collision checker from the Lab 5 scene.

    Exposed so that IK config validation (grasp_planner) and RRT* planning
    (GraspStateMachine) share the *same* collision truth — an IK solution the
    planner considers colliding must be rejected at IK time, not discovered
    as an unreachable RRT* goal mid-cycle (Step 6.1).
    """
    if mj_model is None:
        from lab5_common import load_mujoco_model
        mj_model, _ = load_mujoco_model()
    return SceneCollisionChecker(mj_model)


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

    # Convergence gating (Step 6.1 fix): a state may not hand off until its
    # controller has actually converged on the target, or a timeout elapses.
    JOINT_SETTLE_TOL = 0.010      # rad — max per-joint error to accept handoff
    JOINT_SETTLE_VEL = 0.050      # rad/s — max per-joint velocity to accept handoff
    JOINT_SETTLE_TIMEOUT = 3.0    # s — give up waiting and report residual
    CART_SETTLE_TOL = 0.003       # m — EE position error to accept handoff
    CART_SETTLE_TIMEOUT = 3.0     # s

    # Post-condition: max lateral distance between box centre and place target
    PLACE_TOL_MM = 30.0

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
        collision_checker: SceneCollisionChecker | None = None,
    ) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.ee_fid = ee_fid
        self.cfgs = grasp_cfgs

        # Joint-space gains are inertia-scaled (Step 6.1 chatter fix):
        #   τ = M(q)·(Kp·e + Kd·ė) + g(q)
        # With raw diagonal gains, the wrist joints chatter: their reflected
        # inertia (~0.015 kg·m²) makes the discrete 1 kHz damping term
        # unstable (Kd·dt/I > 2), producing a ±60 mrad torque-saturated limit
        # cycle that reads as a constant ~61 mrad "steady-state error".
        # Scaling by M(q) normalises every joint to the same critically-damped
        # error dynamics: ë = −Kd·ė − Kp·e  (ω = √Kp = 20 rad/s, ζ = 1).
        self.Kp_joint = Kp_joint
        self.Kd_joint = Kd_joint

        # Cartesian impedance gains — full 6D. The old 3×3 translational-only
        # gains left the wrist orientation uncontrolled during DESCEND, so the
        # top-down grasp orientation was maintained by nothing but joint
        # damping. Rotational stiffness holds R_des while translation servos.
        ROT_STIFFNESS = 20.0   # Nm/rad
        ROT_DAMPING = 2.0      # Nm·s/rad
        self.cart_gains = ImpedanceGains(
            K_p=np.diag([Kp_cart] * 3 + [ROT_STIFFNESS] * 3),
            K_d=np.diag([Kd_cart] * 3 + [ROT_DAMPING] * 3),
        )

        self.state = State.IDLE
        self._site_id = get_mj_site_id(mj_model, "gripper_site")
        self._box_bid = get_mj_body_id(mj_model, "grasp_box")
        # Bodies the box may legitimately land on during DESCEND_PLACE
        self._landing_bids = {
            get_mj_body_id(mj_model, "table"),
            get_mj_body_id(mj_model, "target_pad"),
            0,  # world (floor geom)
        }

        # Build collision checker (table only) unless the caller shares one
        self._cc = collision_checker or make_collision_checker()
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
        self._log_box_pos: list[np.ndarray] = []
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
                # Gentle timing while carrying the box: the friction pinch
                # grasp cannot survive a full-speed swing (Step 6.1).
                self._transport_traj = self._plan_and_smooth(
                    q_current, self.cfgs.q_preplace,
                    vel_scale=0.22, acc_scale=0.15,
                )
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
                    stop_on_box_touchdown=True,
                )
                self.state = State.RELEASE

            elif self.state == State.RELEASE:
                self._log_state_transition()
                self._run_open_gripper()
                self.state = State.RETRACT

            elif self.state == State.RETRACT:
                self._log_state_transition()
                # Ascend vertically first (mirror of the approach descend):
                # RRT plans in joint space, so its first segment can sweep the
                # open fingers sideways through the just-placed box — observed
                # as the box being dragged 47 mm off target after a perfect
                # 6 mm placement (Step 6.1). Clear the box, then plan home.
                self._run_cartesian_descend(target_z=None, mode="lift")
                q_current = self.mj_data.qpos[:NUM_JOINTS].copy()
                retract_traj = self._plan_and_smooth(q_current, Q_HOME)
                self._run_joint_impedance(*retract_traj)
                self.state = State.DONE

        self._log_state_transition()

        # Post-condition (Step 6.1): DONE is only a success if the box actually
        # moved to the place target. Previously the machine could reach DONE
        # having grasped air, and nothing checked.
        box_bid = get_mj_body_id(self.mj_model, "grasp_box")
        box_final = self.mj_data.xpos[box_bid].copy()
        lateral_err_mm = float(
            np.linalg.norm(box_final[:2] - BOX_B_POS[:2]) * 1000
        )
        transport_ok = lateral_err_mm < self.PLACE_TOL_MM

        verdict = "✓ SUCCESS" if transport_ok else "✗ FAILURE"
        print(f"  {verdict} — box lateral error to place target: "
              f"{lateral_err_mm:.1f} mm (tolerance {self.PLACE_TOL_MM:.0f} mm)")
        print("  ✓ Pick-and-place cycle complete.")

        return {
            "time": np.array(self._log_time),
            "q": np.array(self._log_q),
            "ee_pos": np.array(self._log_ee_pos),
            "gripper_pos": np.array(self._log_gripper),
            "box_pos": np.array(self._log_box_pos),
            "state": self._log_state,
            "box_final_pos": box_final,
            "box_lateral_error_mm": lateral_err_mm,
            "transport_ok": transport_ok,
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
        vel_scale: float = 0.5,
        acc_scale: float = 0.4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run RRT*, shortcut, and TOPP-RA; return (times, q_traj, qd_traj).

        Args:
            q_start: Start configuration (6,).
            q_goal: Goal configuration (6,).
            max_iter: RRT* iteration budget.
            seed: Random seed.
            vel_scale: Fraction of VEL_LIMITS given to TOPP-RA. Full limits
                (3.14 rad/s shoulder) are far too aggressive for this task —
                the friction pinch grasp loses the box during a full-speed
                transport swing (Step 6.1: box fell mid-EXEC_TRANSPORT).
            acc_scale: Fraction of ACC_LIMITS given to TOPP-RA.

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
        times, q_traj, qd_traj, _ = parameterize_topp_ra(
            path, VEL_LIMITS * vel_scale, ACC_LIMITS * acc_scale
        )
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
        total_steps = int(traj_duration / DT) + 1

        for step in range(total_steps):
            t_local = step * DT

            q_d = np.array(
                [np.interp(t_local, times, q_traj[:, j]) for j in range(NUM_JOINTS)]
            )
            qd_d = np.array(
                [np.interp(t_local, times, qd_traj[:, j]) for j in range(NUM_JOINTS)]
            )
            self._joint_impedance_step(q_d, qd_d)

        # Convergence gate (Step 6.1): keep servoing on the trajectory endpoint
        # until the arm has actually arrived, instead of handing off after a
        # fixed 0.3 s tail with whatever tracking error remains.
        q_goal = q_traj[-1]
        settle_steps = int(self.JOINT_SETTLE_TIMEOUT / DT)
        for _ in range(settle_steps):
            q = self.mj_data.qpos[:NUM_JOINTS]
            qd = self.mj_data.qvel[:NUM_JOINTS]
            if (np.max(np.abs(q_goal - q)) < self.JOINT_SETTLE_TOL
                    and np.max(np.abs(qd)) < self.JOINT_SETTLE_VEL):
                break
            self._joint_impedance_step(q_goal, np.zeros(NUM_JOINTS))

        err = np.max(np.abs(q_goal - self.mj_data.qpos[:NUM_JOINTS]))
        print(f"    joint settle: max|Δq| = {err*1000:.1f} mrad")

    def _joint_impedance_step(self, q_d: np.ndarray, qd_d: np.ndarray) -> None:
        """One inertia-scaled joint-impedance + gravity-comp simulation step.

        τ = M(q)·(Kp·(q_d−q) + Kd·(qd_d−qd)) + g(q) — see __init__ for why
        the gains go through the mass matrix.
        """
        q = self.mj_data.qpos[:NUM_JOINTS].copy()
        qd = self.mj_data.qvel[:NUM_JOINTS].copy()

        M = pin.crba(self.pin_model, self.pin_data, q)
        M = np.triu(M) + np.triu(M, 1).T  # CRBA fills the upper triangle
        pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
        g = self.pin_data.g.copy()

        tau = M @ (self.Kp_joint * (q_d - q)
                   + self.Kd_joint * (qd_d - qd)) + g
        self.mj_data.ctrl[:NUM_JOINTS] = clip_torques(tau)
        mujoco.mj_step(self.mj_model, self.mj_data)
        self._t += DT
        self._record()

    def _box_touched_down(self) -> bool:
        """True if the grasp box is in contact with the table / pad / floor."""
        for cid in range(self.mj_data.ncon):
            c = self.mj_data.contact[cid]
            b1 = self.mj_model.geom_bodyid[c.geom1]
            b2 = self.mj_model.geom_bodyid[c.geom2]
            if self._box_bid in (b1, b2):
                other = b2 if b1 == self._box_bid else b1
                if other in self._landing_bids:
                    return True
        return False

    def _run_cartesian_descend(
        self,
        target_z,
        mode: str,
        stop_on_box_touchdown: bool = False,
    ) -> None:
        """Run Cartesian impedance to move the EE to an absolute target pose.

        mode='descend': moves the EE in a straight line to the pose given by
            FK of `target_z` (a joint configuration, e.g. cfgs.q_grasp).
            Using the *absolute* IK-validated target — rather than "down by
            PREGRASP_CLEARANCE from wherever we are" — means any tracking
            error accumulated in earlier states is corrected here instead of
            being carried into the grasp (Step 6.1 root cause #1).
        mode='lift': moves the EE upward by PREGRASP_CLEARANCE metres
            relative to the current pose (a relative move is correct here —
            the lift only needs clearance, not precision).

        Args:
            target_z: Joint configuration whose FK pose is the descend target.
                Ignored for mode='lift'.
            mode: 'descend' or 'lift'.
            stop_on_box_touchdown: If True (DESCEND_PLACE), the descent ends as
                soon as the carried box has rested on the table/pad for a few
                consecutive steps. Without this the convergence gate keeps
                pushing toward an unreachable in-table target after touchdown,
                dragging the box across the pad (seen as ~40 mm placement
                error with a 27 mm blocked settle residual).
        """
        q_cur = self.mj_data.qpos[:NUM_JOINTS].copy()
        x_start, R_start = get_ee_pose(self.pin_model, self.pin_data, self.ee_fid, q_cur)

        if mode == "descend":
            x_target, R_des = get_ee_pose(
                self.pin_model, self.pin_data, self.ee_fid, target_z
            )
            speed = self.DESCEND_SPEED
        else:  # lift
            x_target = x_start + np.array([0.0, 0.0, PREGRASP_CLEARANCE])
            R_des = R_start
            speed = self.LIFT_SPEED

        delta = x_target - x_start
        distance = float(np.linalg.norm(delta))
        n_steps = max(int(distance / speed / DT), 1)

        TOUCHDOWN_CONFIRM_STEPS = 30  # ~30 ms of sustained contact
        touchdown_count = 0
        touched_down = False

        for step in range(n_steps):
            frac = min((step + 1) / n_steps, 1.0)
            x_des = x_start + frac * delta
            self._cartesian_impedance_step(x_des, R_des)
            if stop_on_box_touchdown:
                touchdown_count = touchdown_count + 1 if self._box_touched_down() else 0
                if touchdown_count >= TOUCHDOWN_CONFIRM_STEPS:
                    touched_down = True
                    break

        # Convergence gate (Step 6.1): keep servoing on the final target until
        # the EE has actually arrived. The old code froze the *current* joint
        # configuration here, locking in whatever tracking error remained
        # (root cause #2).
        if not touched_down:
            settle_steps = int(self.CART_SETTLE_TIMEOUT / DT)
            for _ in range(settle_steps):
                q = self.mj_data.qpos[:NUM_JOINTS].copy()
                x_now, _ = get_ee_pose(self.pin_model, self.pin_data, self.ee_fid, q)
                if np.linalg.norm(x_target - x_now) < self.CART_SETTLE_TOL:
                    break
                self._cartesian_impedance_step(x_target, R_des)
                if stop_on_box_touchdown:
                    touchdown_count = (
                        touchdown_count + 1 if self._box_touched_down() else 0
                    )
                    if touchdown_count >= TOUCHDOWN_CONFIRM_STEPS:
                        touched_down = True
                        break

        q = self.mj_data.qpos[:NUM_JOINTS].copy()
        x_now, _ = get_ee_pose(self.pin_model, self.pin_data, self.ee_fid, q)
        err_mm = np.linalg.norm(x_target - x_now) * 1000
        note = " [box touchdown]" if touched_down else ""
        print(f"    cartesian settle ({mode}): |Δx| = {err_mm:.1f} mm{note}")

        # Short hold at the converged pose to damp residual motion
        self._hold_position(duration=0.5, q_hold=self.mj_data.qpos[:NUM_JOINTS].copy())

    def _cartesian_impedance_step(self, x_des: np.ndarray, R_des: np.ndarray) -> None:
        """One Cartesian-impedance simulation step toward (x_des, R_des)."""
        q = self.mj_data.qpos[:NUM_JOINTS].copy()
        qd = self.mj_data.qvel[:NUM_JOINTS].copy()
        tau = compute_impedance_torque(
            self.pin_model, self.pin_data, self.ee_fid,
            q, qd,
            x_des, R_des, None,
            self.cart_gains,
        )
        self.mj_data.ctrl[:NUM_JOINTS] = clip_torques(tau)
        mujoco.mj_step(self.mj_model, self.mj_data)
        self._t += DT
        self._record()

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
        zero_qd = np.zeros(NUM_JOINTS)
        for _ in range(settle_steps):
            self._joint_impedance_step(q_hold_close, zero_qd)
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
        zero_qd = np.zeros(NUM_JOINTS)
        for _ in range(n_steps):
            self._joint_impedance_step(q_hold, zero_qd)

    # ---------------------------------------------------------------------------
    # Logging helpers
    # ---------------------------------------------------------------------------

    def _record(self) -> None:
        """Record one timestep of data."""
        self._log_time.append(self._t)
        self._log_q.append(self.mj_data.qpos[:NUM_JOINTS].copy())
        ee = self.mj_data.site_xpos[self._site_id].copy()
        self._log_ee_pos.append(ee)
        self._log_box_pos.append(self.mj_data.xpos[self._box_bid].copy())

        jid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_joint"
        )
        self._log_gripper.append(
            float(self.mj_data.qpos[self.mj_model.jnt_qposadr[jid]])
        )
        self._log_state.append(self.state.name)

    def _log_state_transition(self) -> None:
        """Print state transition (with live box position) to console."""
        box = self.mj_data.xpos[self._box_bid]
        print(f"  → State: {self.state.name}  (t={self._t:.2f}s)  "
              f"box=[{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}]")
