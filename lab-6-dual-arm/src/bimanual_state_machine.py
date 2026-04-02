"""M4 — Bimanual state machine for cooperative grasp, lift, carry, and place.

Six-state pipeline:
  1. APPROACH  — both arms to grasp standoff (5 cm from box)
  2. CLOSE     — push EEs 2 cm inside box surface for contact
  3. GRASP     — activate weld constraints, hold steady
  4. LIFT      — raise box 15 cm
  5. CARRY     — translate box 20 cm in +x
  6. PLACE     — lower box 15 cm, deactivate welds

Every motion is joint PD to an IK-solved target.
"""

from __future__ import annotations

import enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np
import pinocchio as pin

from dual_arm_model import DualArmModel
from grasp_pose_calculator import (
    APPROACH_STANDOFF,
    GRASP_STANDOFF,
    _rotation_facing,
    compute_grasp_poses,
)
from joint_pd_controller import DualArmJointPD
from lab6_common import (
    BOX_HALF_EXTENTS,
    DT,
    LEFT_JOINT_SLICE,
    NUM_JOINTS_PER_ARM,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    RIGHT_JOINT_SLICE,
    TABLE_SURFACE_Z,
    get_mj_site_id,
)


class State(enum.Enum):
    APPROACH = 1
    CLOSE = 2
    GRASP = 3
    LIFT = 4
    CARRY = 5
    PLACE = 6
    DONE = 7


# Contact standoff: 2 cm inside box surface
CONTACT_PENETRATION = 0.02

# Motion deltas
LIFT_DZ = 0.15
CARRY_DY = 0.20   # +y direction — symmetric for both arms (unlike +x which hits workspace limits)
PLACE_DZ = -0.15

# Timing
PHASE_MAX_TIME = 5.0
SETTLE_THRESHOLD = 0.005    # rad
SETTLE_DURATION = 0.15      # seconds
GRASP_HOLD_TIME = 0.15      # seconds to hold after activating welds
PLACE_HOLD_TIME = 0.15      # seconds to hold after deactivating welds


def _wrap_joints(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Wrap each joint to the equivalent angle closest to q_ref."""
    q_out = q.copy()
    for i in range(len(q)):
        diff = q[i] - q_ref[i]
        q_out[i] = q_ref[i] + ((diff + np.pi) % (2 * np.pi) - np.pi)
    return q_out


def _find_collision_free_ik(
    dual: DualArmModel,
    arm: str,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q_ref: np.ndarray,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    q_init: np.ndarray | None = None,
    n_trials: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray | None, bool]:
    """Search for a collision-free IK solution closest to q_init."""
    rng = np.random.default_rng(seed)
    model = dual.model_left if arm == "left" else dual.model_right
    data = dual.data_left if arm == "left" else dual.data_right
    ee_fid = dual.ee_frame_id_left if arm == "left" else dual.ee_frame_id_right
    base = dual.base_left if arm == "left" else dual.base_right
    jslice = LEFT_JOINT_SLICE if arm == "left" else RIGHT_JOINT_SLICE
    center = q_init if q_init is not None else q_ref

    best_q: np.ndarray | None = None
    best_dist = float("inf")

    for trial in range(n_trials):
        qi = center.copy() if trial == 0 else center + rng.uniform(-2.5, 2.5, size=NUM_JOINTS_PER_ARM)

        q_sol, ok, _ = dual._ik_single(
            model, data, ee_fid,
            target_pos - base, target_rot, qi,
            0.01, 200, 1e-4, 1e-3, 0.5,
        )
        if not ok:
            continue

        q_w = _wrap_joints(q_sol, q_ref)
        fk = dual.fk(arm, q_w)
        if np.linalg.norm(fk.translation - target_pos) > 0.001:
            continue

        # Collision check
        saved = mj_data.qpos[jslice].copy()
        mj_data.qpos[jslice] = q_w
        mujoco.mj_forward(mj_model, mj_data)

        bad = 0
        for c in range(mj_data.ncon):
            ct = mj_data.contact[c]
            if np.isnan(ct.dist) or ct.dist > -0.001:
                continue
            b1 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                    mj_model.geom_bodyid[ct.geom1]) or ""
            b2 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                    mj_model.geom_bodyid[ct.geom2]) or ""
            if ("table" in b1 and "box" in b2) or ("box" in b1 and "table" in b2):
                continue
            if arm in b1 or arm in b2:
                bad += 1

        mj_data.qpos[jslice] = saved
        mujoco.mj_forward(mj_model, mj_data)

        if bad > 0:
            continue

        dist = np.linalg.norm(q_w - center)
        if dist < best_dist:
            best_dist = dist
            best_q = q_w.copy()

    return best_q, best_q is not None


def _compute_ee_targets_from_box(
    box_pos: np.ndarray,
    box_rot: np.ndarray,
    standoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute left/right EE positions and rotations from box state.

    Args:
        box_pos: Box center position (3,).
        box_rot: Box rotation matrix (3, 3).
        standoff: Distance from box surface to EE along x-axis.

    Returns:
        (left_pos, left_rot, right_pos, right_rot)
    """
    box_x_axis = box_rot[:, 0]
    box_half_x = BOX_HALF_EXTENTS[0]

    left_pos = box_pos - box_x_axis * (box_half_x + standoff)
    right_pos = box_pos + box_x_axis * (box_half_x + standoff)

    left_rot = _rotation_facing(box_x_axis)
    right_rot = _rotation_facing(-box_x_axis)

    return left_pos, left_rot, right_pos, right_rot


def _solve_ik_pair(
    dual: DualArmModel,
    left_pos: np.ndarray,
    left_rot: np.ndarray,
    right_pos: np.ndarray,
    right_rot: np.ndarray,
    q_prev_left: np.ndarray,
    q_prev_right: np.ndarray,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    collision_free: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Solve IK for both arms, seeded from previous configs.

    Returns:
        (q_left, q_right, success)
    """
    if collision_free:
        q_left, ok_l = _find_collision_free_ik(
            dual, "left", left_pos, left_rot,
            Q_HOME_LEFT, mj_model, mj_data,
            q_init=q_prev_left, n_trials=300, seed=seed,
        )
        q_right, ok_r = _find_collision_free_ik(
            dual, "right", right_pos, right_rot,
            Q_HOME_RIGHT, mj_model, mj_data,
            q_init=q_prev_right, n_trials=300, seed=seed + 1,
        )
    else:
        # Simple IK without collision check (for CLOSE state where contact is expected)
        q_left_raw, ok_l, _ = dual.ik(
            "left", left_pos, left_rot,
            q_init=q_prev_left, damping=0.01, max_iter=200, n_restarts=10,
        )
        q_left = _wrap_joints(q_left_raw, Q_HOME_LEFT) if ok_l else None

        q_right_raw, ok_r, _ = dual.ik(
            "right", right_pos, right_rot,
            q_init=q_prev_right, damping=0.01, max_iter=200, n_restarts=10,
        )
        q_right = _wrap_joints(q_right_raw, Q_HOME_RIGHT) if ok_r else None

    if not ok_l or not ok_r:
        return q_left, q_right, False
    return q_left, q_right, True


class BimanualStateMachine:
    """State machine for cooperative bimanual manipulation.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        dual: DualArmModel instance.
        controller: DualArmJointPD instance.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        dual: DualArmModel,
        controller: DualArmJointPD,
        renderer: mujoco.Renderer | None = None,
        frame_skip: int = 33,
    ) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.dual = dual
        self.controller = controller
        self.renderer = renderer
        self.frame_skip = frame_skip

        self.box_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.left_ee_sid = get_mj_site_id(mj_model, "left_ee_site")
        self.right_ee_sid = get_mj_site_id(mj_model, "right_ee_site")
        self.left_weld_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, "left_grasp")
        self.right_weld_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, "right_grasp")

        self.state = State.APPROACH
        self.q_target_left = Q_HOME_LEFT.copy()
        self.q_target_right = Q_HOME_RIGHT.copy()

        # Logged data
        self.box_trajectory: list[np.ndarray] = []
        self.time_log: list[float] = []
        self.state_log: list[State] = []
        self.frames: list[np.ndarray] = []
        self.sim_time = 0.0
        self._step_count = 0

        # Box initial position (set after first forward)
        self.box_init_pos: np.ndarray | None = None

    def _set_weld_relpose(self, weld_id: int, body1_name: str, body2_name: str) -> None:
        """Update weld constraint eq_data to lock the current relative pose.

        MuJoCo weld eq_data layout (11 floats per constraint):
          [0:3]   anchor on body1 (body1 local frame)
          [3:6]   relative position of body2 in body1 frame
          [6:10]  relative quaternion (w,x,y,z) of body2 in body1 frame
          [10]    torquescale
        """
        b1 = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        b2 = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body2_name)

        pos1 = self.mj_data.xpos[b1]
        mat1 = self.mj_data.xmat[b1].reshape(3, 3)
        pos2 = self.mj_data.xpos[b2]
        mat2 = self.mj_data.xmat[b2].reshape(3, 3)

        # Relative transform: body2 in body1 frame
        rel_pos = mat1.T @ (pos2 - pos1)
        rel_mat = mat1.T @ mat2

        # Convert rotation matrix to MuJoCo quaternion (w,x,y,z)
        rel_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rel_quat, rel_mat.flatten())

        # Write to model eq_data
        d = self.mj_model.eq_data[weld_id]
        d[0:3] = 0.0           # anchor at body1 origin
        d[3:6] = rel_pos
        d[6:10] = rel_quat
        d[10] = 1.0            # torquescale

    def _box_pos(self) -> np.ndarray:
        return self.mj_data.xpos[self.box_id].copy()

    def _box_rot(self) -> np.ndarray:
        return self.mj_data.xmat[self.box_id].reshape(3, 3).copy()

    def _log(self) -> None:
        self.box_trajectory.append(self._box_pos())
        self.time_log.append(self.sim_time)
        self.state_log.append(self.state)

    def _step(self) -> None:
        self.controller.compute(self.mj_data, self.q_target_left, self.q_target_right)
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.sim_time += DT
        self._step_count += 1
        self._log()
        if self.renderer is not None and self._step_count % self.frame_skip == 0:
            self.renderer.update_scene(self.mj_data)
            self.frames.append(self.renderer.render().copy())

    def _run_pd_until_settle(
        self,
        max_time: float = PHASE_MAX_TIME,
        ramp_time: float = 2.0,
    ) -> dict:
        """Run PD until both arms settle or timeout.

        Settles on position convergence (<SETTLE_THRESHOLD) or, after the ramp
        completes, on velocity convergence (<0.02 rad/s). The velocity check
        handles weld-loaded phases where residual position error remains.

        Args:
            max_time: Maximum phase duration in seconds.
            ramp_time: Duration over which to smoothstep-interpolate targets
                from current joint positions to final targets.
        """
        VEL_SETTLE = 0.02  # rad/s — velocity-based early exit
        n_max = int(max_time / DT)
        n_settle_req = int(SETTLE_DURATION / DT)
        n_ramp = int(ramp_time / DT)
        l_count, r_count = 0, 0
        vel_count = 0
        l_time, r_time = None, None

        q_start_left = self.mj_data.qpos[LEFT_JOINT_SLICE].copy()
        q_start_right = self.mj_data.qpos[RIGHT_JOINT_SLICE].copy()
        q_final_left = self.q_target_left.copy()
        q_final_right = self.q_target_right.copy()

        for step in range(n_max):
            alpha = min(1.0, (step + 1) / n_ramp) if n_ramp > 0 else 1.0
            alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            self.q_target_left = q_start_left + alpha_smooth * (q_final_left - q_start_left)
            self.q_target_right = q_start_right + alpha_smooth * (q_final_right - q_start_right)

            self._step()
            t = (step + 1) * DT

            # Position-based settle
            el = np.max(np.abs(q_final_left - self.mj_data.qpos[LEFT_JOINT_SLICE]))
            er = np.max(np.abs(q_final_right - self.mj_data.qpos[RIGHT_JOINT_SLICE]))
            l_count = l_count + 1 if el < SETTLE_THRESHOLD else 0
            r_count = r_count + 1 if er < SETTLE_THRESHOLD else 0
            if l_time is None and l_count >= n_settle_req:
                l_time = t - SETTLE_DURATION
            if r_time is None and r_count >= n_settle_req:
                r_time = t - SETTLE_DURATION

            # Velocity-based settle (only after ramp completes)
            if alpha >= 1.0:
                vl = np.max(np.abs(self.mj_data.qvel[LEFT_JOINT_SLICE]))
                vr = np.max(np.abs(self.mj_data.qvel[RIGHT_JOINT_SLICE]))
                vel_count = vel_count + 1 if (vl < VEL_SETTLE and vr < VEL_SETTLE) else 0
            else:
                vel_count = 0

            pos_done = l_time is not None and r_time is not None
            vel_done = vel_count >= n_settle_req
            if pos_done or vel_done:
                self.q_target_left = q_final_left.copy()
                self.q_target_right = q_final_right.copy()
                for _ in range(int(0.1 / DT)):
                    self._step()
                break

        self.q_target_left = q_final_left.copy()
        self.q_target_right = q_final_right.copy()
        return {"left_settle_time": l_time, "right_settle_time": r_time}

    def _hold(self, duration: float) -> None:
        """Hold current targets for given duration."""
        n = int(duration / DT)
        for _ in range(n):
            self._step()

    def _print_state(self, label: str) -> None:
        box_pos = self._box_pos()
        left_pos = self.mj_data.site_xpos[self.left_ee_sid]
        right_pos = self.mj_data.site_xpos[self.right_ee_sid]
        print(f"  [{label}] t={self.sim_time:.2f}s  box={box_pos.round(4)}  "
              f"ncon={self.mj_data.ncon}")

    def run(self) -> bool:
        """Execute the full state machine. Returns True if all gates pass."""
        mj_model = self.mj_model
        mj_data = self.mj_data
        dual = self.dual

        # Initialize at home
        mj_data.qpos[LEFT_JOINT_SLICE] = Q_HOME_LEFT
        mj_data.qpos[RIGHT_JOINT_SLICE] = Q_HOME_RIGHT
        mujoco.mj_forward(mj_model, mj_data)
        self.box_init_pos = self._box_pos()

        # Settle at home
        self._hold(0.25)

        # ============================================================
        # STATE 1: APPROACH
        # ============================================================
        self.state = State.APPROACH
        print("\n  === State 1: APPROACH ===")

        box_pos = self._box_pos()
        box_rot = self._box_rot()

        # Approach poses (10 cm standoff)
        la_pos, la_rot, ra_pos, ra_rot = _compute_ee_targets_from_box(
            box_pos, box_rot, APPROACH_STANDOFF)

        q_la, ok_la = _find_collision_free_ik(
            dual, "left", la_pos, la_rot, Q_HOME_LEFT,
            mj_model, mj_data, seed=12345)
        q_ra, ok_ra = _find_collision_free_ik(
            dual, "right", ra_pos, ra_rot, Q_HOME_RIGHT,
            mj_model, mj_data, seed=12345)
        if not ok_la or not ok_ra:
            print(f"    IK FAILED: left={ok_la}, right={ok_ra}")
            return False

        # Grasp standoff poses (5 cm) seeded from approach
        lg_pos, lg_rot, rg_pos, rg_rot = _compute_ee_targets_from_box(
            box_pos, box_rot, GRASP_STANDOFF)

        q_lg, ok_lg = _find_collision_free_ik(
            dual, "left", lg_pos, lg_rot, Q_HOME_LEFT,
            mj_model, mj_data, q_init=q_la, seed=12345)
        q_rg, ok_rg = _find_collision_free_ik(
            dual, "right", rg_pos, rg_rot, Q_HOME_RIGHT,
            mj_model, mj_data, q_init=q_ra, seed=12345)
        if not ok_lg or not ok_rg:
            print(f"    Grasp IK FAILED: left={ok_lg}, right={ok_rg}")
            return False

        # Phase 1: HOME → approach
        self.q_target_left = q_la
        self.q_target_right = q_ra
        p1 = self._run_pd_until_settle()
        self._print_state("APPROACH phase1")

        # Phase 2: approach → grasp standoff
        self.q_target_left = q_lg
        self.q_target_right = q_rg
        p2 = self._run_pd_until_settle()
        self._print_state("APPROACH phase2")

        # ============================================================
        # STATE 2: CLOSE
        # ============================================================
        self.state = State.CLOSE
        print("\n  === State 2: CLOSE ===")

        # Contact targets: 2 cm inside box surface
        lc_pos, lc_rot, rc_pos, rc_rot = _compute_ee_targets_from_box(
            box_pos, box_rot, -CONTACT_PENETRATION)

        q_lc, q_rc, ok = _solve_ik_pair(
            dual, lc_pos, lc_rot, rc_pos, rc_rot,
            q_lg, q_rg, mj_model, mj_data,
            collision_free=False, seed=12345)
        if not ok:
            print(f"    CLOSE IK FAILED")
            return False

        self.q_target_left = q_lc
        self.q_target_right = q_rc
        self._run_pd_until_settle(max_time=4.0)
        # Hold for stable contact
        self._hold(0.15)
        self._print_state("CLOSE")

        # Gate: EEs within 3cm of box surface
        left_ee = self.mj_data.site_xpos[self.left_ee_sid]
        right_ee = self.mj_data.site_xpos[self.right_ee_sid]
        box_pos_now = self._box_pos()
        left_box_dist = abs(left_ee[0] - (box_pos_now[0] - BOX_HALF_EXTENTS[0]))
        right_box_dist = abs(right_ee[0] - (box_pos_now[0] + BOX_HALF_EXTENTS[0]))
        print(f"    EE-to-box-surface: L={left_box_dist*100:.1f}cm  R={right_box_dist*100:.1f}cm")

        # ============================================================
        # STATE 3: GRASP
        # ============================================================
        self.state = State.GRASP
        print("\n  === State 3: GRASP ===")

        # Set weld constraint data to current relative pose before activating.
        # Without this, the weld enforces the initial model pose and launches the box.
        self._set_weld_relpose(self.left_weld_id, "left_wrist_3_link", "box")
        self._set_weld_relpose(self.right_weld_id, "right_wrist_3_link", "box")

        mj_data.eq_active[self.left_weld_id] = 1
        mj_data.eq_active[self.right_weld_id] = 1
        print("    Weld constraints activated (relative pose locked)")

        self._hold(GRASP_HOLD_TIME)
        self._print_state("GRASP")

        box_drift = np.linalg.norm(self._box_pos() - self.box_init_pos)
        print(f"    Box drift from initial: {box_drift*100:.1f} cm (limit: 1 cm)")

        # ============================================================
        # STATE 4: LIFT
        # ============================================================
        self.state = State.LIFT
        print("\n  === State 4: LIFT ===")

        lifted_box_pos = self._box_pos() + np.array([0, 0, LIFT_DZ])
        ll_pos, ll_rot, rl_pos, rl_rot = _compute_ee_targets_from_box(
            lifted_box_pos, self._box_rot(), -CONTACT_PENETRATION)

        q_ll, q_rl, ok = _solve_ik_pair(
            dual, ll_pos, ll_rot, rl_pos, rl_rot,
            self.q_target_left, self.q_target_right,
            mj_model, mj_data, collision_free=False, seed=42)
        if not ok:
            print(f"    LIFT IK FAILED")
            return False

        self.q_target_left = q_ll
        self.q_target_right = q_rl
        self._run_pd_until_settle()
        self._print_state("LIFT")

        lift_dz = self._box_pos()[2] - self.box_init_pos[2]
        print(f"    Box z delta: {lift_dz:.4f} m (target: {LIFT_DZ})")

        # ============================================================
        # STATE 5: CARRY
        # ============================================================
        self.state = State.CARRY
        print("\n  === State 5: CARRY ===")

        carried_box_pos = self._box_pos() + np.array([0, CARRY_DY, 0])
        lcar_pos, lcar_rot, rcar_pos, rcar_rot = _compute_ee_targets_from_box(
            carried_box_pos, self._box_rot(), -CONTACT_PENETRATION)

        q_lcar, q_rcar, ok = _solve_ik_pair(
            dual, lcar_pos, lcar_rot, rcar_pos, rcar_rot,
            self.q_target_left, self.q_target_right,
            mj_model, mj_data, collision_free=False, seed=42)
        if not ok:
            print(f"    CARRY IK FAILED")
            return False

        self.q_target_left = q_lcar
        self.q_target_right = q_rcar
        self._run_pd_until_settle()
        self._print_state("CARRY")

        carry_dy = self._box_pos()[1] - self.box_init_pos[1]
        print(f"    Box y delta: {carry_dy:.4f} m (target: {CARRY_DY})")

        # ============================================================
        # STATE 6: PLACE
        # ============================================================
        self.state = State.PLACE
        print("\n  === State 6: PLACE ===")

        # Absolute z target: box resting on table surface
        place_z = TABLE_SURFACE_Z + BOX_HALF_EXTENTS[2]  # 0.245
        cur_box = self._box_pos()
        placed_box_pos = np.array([cur_box[0], cur_box[1], place_z])
        print(f"    Target box pos: {placed_box_pos.round(4)}")

        lpl_pos, lpl_rot, rpl_pos, rpl_rot = _compute_ee_targets_from_box(
            placed_box_pos, self._box_rot(), -CONTACT_PENETRATION)

        q_lpl, q_rpl, ok = _solve_ik_pair(
            dual, lpl_pos, lpl_rot, rpl_pos, rpl_rot,
            self.q_target_left, self.q_target_right,
            mj_model, mj_data, collision_free=False, seed=42)
        if not ok:
            print(f"    PLACE IK FAILED")
            return False

        self.q_target_left = q_lpl
        self.q_target_right = q_rpl
        self._run_pd_until_settle()
        self._print_state("PLACE (lowered)")

        # Deactivate welds
        mj_data.eq_active[self.left_weld_id] = 0
        mj_data.eq_active[self.right_weld_id] = 0
        print("    Weld constraints deactivated")

        # Withdraw: retract arms to approach standoff so they don't push the box
        withdraw_box_pos = self._box_pos()
        withdraw_rot = self._box_rot()
        lw_pos, lw_rot, rw_pos, rw_rot = _compute_ee_targets_from_box(
            withdraw_box_pos, withdraw_rot, APPROACH_STANDOFF)

        q_lw, q_rw, ok_w = _solve_ik_pair(
            dual, lw_pos, lw_rot, rw_pos, rw_rot,
            self.q_target_left, self.q_target_right,
            mj_model, mj_data, collision_free=False, seed=42)
        if ok_w:
            self.q_target_left = q_lw
            self.q_target_right = q_rw
            print("    Retracting arms...")
        else:
            print("    Withdraw IK failed — holding position")

        self._run_pd_until_settle(max_time=4.0)
        self._hold(PLACE_HOLD_TIME)
        self._print_state("PLACE (final)")

        self.state = State.DONE
        return True
