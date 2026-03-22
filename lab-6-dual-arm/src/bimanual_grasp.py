"""Lab 6 — Bimanual grasp state machine for cooperative pick-and-place.

Orchestrates a full bimanual manipulation pipeline:
  IDLE → APPROACH → PRE_GRASP → GRASP → LIFT → CARRY → LOWER → RELEASE → RETREAT → DONE

Each state uses impedance control (from cooperative_controller) with
appropriate targets derived from the object-centric frame.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import pinocchio as pin

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab6_common import (
    NUM_JOINTS_PER_ARM,
    LEFT_JOINT_SLICE,
    RIGHT_JOINT_SLICE,
    LEFT_CTRL_SLICE,
    RIGHT_CTRL_SLICE,
    TORQUE_LIMITS,
    clip_torques,
    BOX_HALF_EXTENTS,
    TABLE_SURFACE_Z,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
)
from dual_arm_model import DualArmModel, ObjectFrame
from cooperative_controller import DualImpedanceController, DualImpedanceGains


class BimanualState(Enum):
    IDLE = auto()
    APPROACH = auto()       # Move both arms to pre-grasp poses (above grasp points)
    PRE_GRASP = auto()      # Move to grasp poses (at object sides)
    GRASP = auto()          # Activate weld constraints, wait for settling
    LIFT = auto()           # Lift object
    CARRY = auto()          # Transport laterally
    LOWER = auto()          # Lower to table
    RELEASE = auto()        # Deactivate welds, open slightly
    RETREAT = auto()        # Move arms back to home
    DONE = auto()


@dataclass
class BimanualTaskConfig:
    """Configuration for a bimanual pick-carry-place task.

    Attributes:
        object_pos: Initial object center position (3,).
        target_pos: Target object center position (3,).
        lift_height: How high to lift above table (m).
        approach_clearance: Offset in X for pre-grasp approach (m, away from grasp pose).
        grasp_offset_x: Distance from object center to grasp point along X (m).
        position_tolerance: Position tolerance for state transition (m).
        settle_time: Time to wait in GRASP state for weld to settle (s).
    """

    object_pos: np.ndarray
    target_pos: np.ndarray
    lift_height: float = 0.15
    approach_clearance: float = 0.08
    grasp_offset_x: float = 0.18   # slightly outside box half-extent X (0.15)
    position_tolerance: float = 0.01
    settle_time: float = 0.5


def _rotation_facing_pos_x() -> np.ndarray:
    """Rotation matrix whose Z-axis points in world +X (left arm side-grasp).

    The UR5e EE Z-axis is the approach direction.  For a side-grasp from the
    left (negative X side) the EE must approach in the +X direction, so we
    rotate the identity frame such that:
        new_Z = world +X   →  column 2 = [1, 0, 0]
        new_Y = world +Z   →  column 1 = [0, 0, 1]   (tool up)
        new_X = new_Y × new_Z = [0, 1, 0] × [1, 0, 0] = [0*0-0*0, 0*1-1*0, 1*0-0*1]
                                 wait — let's be explicit

    Using a right-hand frame:
        Z_new = [1, 0, 0]   (approach +X)
        Y_new = [0, 0, 1]   (tool-up)
        X_new = Y_new × Z_new = [0,0,1] × [1,0,0] = [0*0-1*0, 1*1-0*0, 0*0-0*1]
                                                    = [0, 1, 0]

    Returns:
        3×3 rotation matrix (columns are X_new, Y_new, Z_new).
    """
    X_new = np.array([0.0, 1.0, 0.0])
    Y_new = np.array([0.0, 0.0, 1.0])
    Z_new = np.array([1.0, 0.0, 0.0])
    R = np.column_stack([X_new, Y_new, Z_new])
    return R


def _rotation_facing_neg_x() -> np.ndarray:
    """Rotation matrix whose Z-axis points in world -X (right arm side-grasp).

    The right arm approaches from the +X side and the EE must face -X:
        Z_new = [-1, 0, 0]
        Y_new = [ 0, 0, 1]   (tool-up)
        X_new = Y_new × Z_new = [0,0,1] × [-1,0,0] = [0*0-1*0, 1*(-1)-0*0, 0*0-0*(-1)]
                                                     = [0, -1, 0]

    Returns:
        3×3 rotation matrix (columns are X_new, Y_new, Z_new).
    """
    X_new = np.array([0.0, -1.0, 0.0])
    Y_new = np.array([0.0,  0.0, 1.0])
    Z_new = np.array([-1.0, 0.0, 0.0])
    R = np.column_stack([X_new, Y_new, Z_new])
    return R


# Cached orientation matrices for the two grasp directions.
_R_LEFT_GRASP = _rotation_facing_pos_x()   # left EE faces +X
_R_RIGHT_GRASP = _rotation_facing_neg_x()  # right EE faces -X


class BimanualGraspStateMachine:
    """Full bimanual pick-carry-place state machine.

    Drives both arms through ten states using impedance control computed by
    ``DualImpedanceController``.  The caller is responsible for stepping the
    MuJoCo simulation; this class provides the torque commands.

    Grasp geometry (side approach along world X):
        - Left arm grasps from the **-X** side of the object.
          Its EE is placed at  object_pos - [grasp_offset_x, 0, 0],
          with EE Z-axis pointing +X (toward the object centre).
        - Right arm grasps from the **+X** side.
          Its EE is placed at  object_pos + [grasp_offset_x, 0, 0],
          with EE Z-axis pointing -X (toward the object centre).
        - Pre-grasp poses add ``approach_clearance`` further away along X
          before the arms close in.

    State transitions are triggered by reaching position tolerances or
    elapsed settle time.

    Args:
        dual_model: DualArmModel providing FK, Jacobians, gravity.
        config: Task configuration.
        gains: Impedance gains; defaults to ``DualImpedanceGains()``.
    """

    def __init__(
        self,
        dual_model: DualArmModel,
        config: BimanualTaskConfig,
        gains: Optional[DualImpedanceGains] = None,
    ) -> None:
        self.model = dual_model
        self.config = config
        self.controller = DualImpedanceController(dual_model, gains or DualImpedanceGains())

        # State machine
        self.state: BimanualState = BimanualState.IDLE
        self._grasp_activate_time: float = -1.0
        self._release_time: float = -1.0

        # Log of (simulation_time, state_name) for debugging
        self.state_log: list[tuple[float, str]] = []

        # Precompute all fixed targets up front
        self._build_targets()

    # ------------------------------------------------------------------
    # Target precomputation
    # ------------------------------------------------------------------

    def _build_targets(self) -> None:
        """Precompute SE3 targets for each state from the task config."""
        cfg = self.config
        obj = cfg.object_pos
        tgt = cfg.target_pos
        dx = cfg.grasp_offset_x
        clr = cfg.approach_clearance

        # ---- Grasp and approach positions --------------------------------
        # Left arm: comes from -X side, so grasp point is at obj - [dx,0,0]
        self._grasp_pos_left = np.array([obj[0] - dx, obj[1], obj[2]])
        self._grasp_pos_right = np.array([obj[0] + dx, obj[1], obj[2]])

        # Pre-grasp: step further out along X before closing
        self._approach_pos_left = np.array([obj[0] - dx - clr, obj[1], obj[2]])
        self._approach_pos_right = np.array([obj[0] + dx + clr, obj[1], obj[2]])

        # ---- Lift targets (same X/Y as grasp, higher Z) ------------------
        lift_z = obj[2] + cfg.lift_height
        self._lift_pos_left = np.array([self._grasp_pos_left[0], obj[1], lift_z])
        self._lift_pos_right = np.array([self._grasp_pos_right[0], obj[1], lift_z])

        # ---- Carry targets (object at target_pos, same lift_height) ------
        carry_z = tgt[2] + cfg.lift_height
        self._carry_pos_left = np.array([tgt[0] - dx, tgt[1], carry_z])
        self._carry_pos_right = np.array([tgt[0] + dx, tgt[1], carry_z])

        # ---- Lower targets (place height = table surface + box half-z) ---
        place_z = TABLE_SURFACE_Z + BOX_HALF_EXTENTS[2]
        self._lower_pos_left = np.array([tgt[0] - dx, tgt[1], place_z])
        self._lower_pos_right = np.array([tgt[0] + dx, tgt[1], place_z])

        # ---- Retreat (same as approach but mirrored to target side) ------
        self._retreat_pos_left = np.array([tgt[0] - dx - clr, tgt[1], place_z])
        self._retreat_pos_right = np.array([tgt[0] + dx + clr, tgt[1], place_z])

        # Orientations (fixed for all states)
        self._R_left = _R_LEFT_GRASP
        self._R_right = _R_RIGHT_GRASP

    # ------------------------------------------------------------------
    # SE3 helpers
    # ------------------------------------------------------------------

    def _make_left_target(self, pos: np.ndarray) -> pin.SE3:
        """Build a left-arm SE3 target from a world position."""
        return pin.SE3(self._R_left.copy(), pos.copy())

    def _make_right_target(self, pos: np.ndarray) -> pin.SE3:
        """Build a right-arm SE3 target from a world position."""
        return pin.SE3(self._R_right.copy(), pos.copy())

    def _home_left(self) -> pin.SE3:
        """Left arm home pose in world frame."""
        return self.model.fk_left(Q_HOME_LEFT)

    def _home_right(self) -> pin.SE3:
        """Right arm home pose in world frame."""
        return self.model.fk_right(Q_HOME_RIGHT)

    # ------------------------------------------------------------------
    # State transition helpers
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: BimanualState, t: float) -> None:
        """Record a state transition and update the state machine.

        Args:
            new_state: Target state.
            t: Current simulation time (s).
        """
        self.state_log.append((t, new_state.name))
        self.state = new_state

    def _both_near(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        pos_left: np.ndarray,
        pos_right: np.ndarray,
    ) -> bool:
        """Return True if both EEs are within position_tolerance of targets.

        Args:
            q_left: Left joint angles (6,).
            q_right: Right joint angles (6,).
            pos_left: Desired left EE position (3,).
            pos_right: Desired right EE position (3,).

        Returns:
            True when both arms are within tolerance.
        """
        tol = self.config.position_tolerance
        cur_left = self.model.fk_left_pos(q_left)
        cur_right = self.model.fk_right_pos(q_right)
        err_left = np.linalg.norm(cur_left - pos_left)
        err_right = np.linalg.norm(cur_right - pos_right)
        return (err_left < tol) and (err_right < tol)

    def _left_near(self, q_left: np.ndarray, pos: np.ndarray) -> bool:
        """Check if left EE is within tolerance of a target position."""
        cur = self.model.fk_left_pos(q_left)
        return np.linalg.norm(cur - pos) < self.config.position_tolerance

    def _right_near(self, q_right: np.ndarray, pos: np.ndarray) -> bool:
        """Check if right EE is within tolerance of a target position."""
        cur = self.model.fk_right_pos(q_right)
        return np.linalg.norm(cur - pos) < self.config.position_tolerance

    # ------------------------------------------------------------------
    # Weld constraint management
    # ------------------------------------------------------------------

    def _activate_welds(self, mj_model, mj_data) -> None:
        """Enable left_grasp and right_grasp equality (weld) constraints.

        Args:
            mj_model: MuJoCo model.
            mj_data: MuJoCo data (unused but kept for API symmetry).
        """
        for name in ("left_grasp", "right_grasp"):
            eq_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
            if eq_id >= 0:
                mj_model.eq_active[eq_id] = 1

    def _deactivate_welds(self, mj_model, mj_data) -> None:
        """Disable left_grasp and right_grasp equality (weld) constraints.

        Args:
            mj_model: MuJoCo model.
            mj_data: MuJoCo data (unused but kept for API symmetry).
        """
        for name in ("left_grasp", "right_grasp"):
            eq_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
            if eq_id >= 0:
                mj_model.eq_active[eq_id] = 0

    # ------------------------------------------------------------------
    # Per-state target computation
    # ------------------------------------------------------------------

    def _targets_for_state(
        self,
        state: BimanualState,
        q_left: np.ndarray,
        q_right: np.ndarray,
    ) -> tuple[pin.SE3, pin.SE3, bool]:
        """Return (target_left, target_right, grasping) for the given state.

        The ``grasping`` flag is forwarded to the impedance controller to
        enable the internal squeeze force.

        Args:
            state: Current state.
            q_left: Left joint angles for fallback (home FK).
            q_right: Right joint angles for fallback (home FK).

        Returns:
            Tuple of (target_left SE3, target_right SE3, grasping bool).
        """
        if state == BimanualState.IDLE:
            # Hold current pose
            tgt_l = self.model.fk_left(q_left)
            tgt_r = self.model.fk_right(q_right)
            return tgt_l, tgt_r, False

        elif state == BimanualState.APPROACH:
            tgt_l = self._make_left_target(self._approach_pos_left)
            tgt_r = self._make_right_target(self._approach_pos_right)
            return tgt_l, tgt_r, False

        elif state == BimanualState.PRE_GRASP:
            tgt_l = self._make_left_target(self._grasp_pos_left)
            tgt_r = self._make_right_target(self._grasp_pos_right)
            return tgt_l, tgt_r, False

        elif state == BimanualState.GRASP:
            # Hold at grasp poses while weld settles
            tgt_l = self._make_left_target(self._grasp_pos_left)
            tgt_r = self._make_right_target(self._grasp_pos_right)
            return tgt_l, tgt_r, True

        elif state == BimanualState.LIFT:
            tgt_l = self._make_left_target(self._lift_pos_left)
            tgt_r = self._make_right_target(self._lift_pos_right)
            return tgt_l, tgt_r, True

        elif state == BimanualState.CARRY:
            tgt_l = self._make_left_target(self._carry_pos_left)
            tgt_r = self._make_right_target(self._carry_pos_right)
            return tgt_l, tgt_r, True

        elif state == BimanualState.LOWER:
            tgt_l = self._make_left_target(self._lower_pos_left)
            tgt_r = self._make_right_target(self._lower_pos_right)
            return tgt_l, tgt_r, True

        elif state == BimanualState.RELEASE:
            # Hold lower poses while welds are deactivated; no squeeze
            tgt_l = self._make_left_target(self._lower_pos_left)
            tgt_r = self._make_right_target(self._lower_pos_right)
            return tgt_l, tgt_r, False

        elif state == BimanualState.RETREAT:
            tgt_l = self._make_left_target(self._retreat_pos_left)
            tgt_r = self._make_right_target(self._retreat_pos_right)
            return tgt_l, tgt_r, False

        else:  # DONE
            # Hold retreat poses once done
            tgt_l = self._make_left_target(self._retreat_pos_left)
            tgt_r = self._make_right_target(self._retreat_pos_right)
            return tgt_l, tgt_r, False

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        q_left: np.ndarray,
        qd_left: np.ndarray,
        q_right: np.ndarray,
        qd_right: np.ndarray,
        t: float,
        mj_model=None,
        mj_data=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute joint torques for the current state and handle transitions.

        Evaluates state transition conditions first, then computes impedance
        torques toward the targets for the (possibly updated) state.

        Args:
            q_left: Left joint angles (6,).
            qd_left: Left joint velocities (6,).
            q_right: Right joint angles (6,).
            qd_right: Right joint velocities (6,).
            t: Current simulation time (s).
            mj_model: MuJoCo model (required for weld constraint activation).
            mj_data: MuJoCo data.

        Returns:
            Tuple of (tau_left, tau_right), each (6,), clipped to torque limits.
        """
        # ----------------------------------------------------------------
        # 1. Evaluate state transitions
        # ----------------------------------------------------------------
        self._maybe_transition(q_left, q_right, t, mj_model, mj_data)

        # ----------------------------------------------------------------
        # 2. Get targets for current state
        # ----------------------------------------------------------------
        tgt_l, tgt_r, grasping = self._targets_for_state(
            self.state, q_left, q_right
        )

        # ----------------------------------------------------------------
        # 3. Compute impedance torques
        # ----------------------------------------------------------------
        tau_left, tau_right = self.controller.compute_dual_torques(
            q_left=q_left,
            qd_left=qd_left,
            q_right=q_right,
            qd_right=qd_right,
            target_left=tgt_l,
            target_right=tgt_r,
            grasping=grasping,
        )

        return tau_left, tau_right

    # ------------------------------------------------------------------
    # State transition logic
    # ------------------------------------------------------------------

    def _maybe_transition(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        t: float,
        mj_model,
        mj_data,
    ) -> None:
        """Evaluate and execute any state transition that is ready.

        Args:
            q_left: Left joint angles (6,).
            q_right: Right joint angles (6,).
            t: Current simulation time (s).
            mj_model: MuJoCo model (needed for weld control).
            mj_data: MuJoCo data.
        """
        s = self.state

        if s == BimanualState.IDLE:
            # Transition to APPROACH immediately on first step
            self._transition_to(BimanualState.APPROACH, t)

        elif s == BimanualState.APPROACH:
            # Wait until both EEs reach approach (pre-grasp clearance) positions
            if self._both_near(
                q_left, q_right,
                self._approach_pos_left,
                self._approach_pos_right,
            ):
                self._transition_to(BimanualState.PRE_GRASP, t)

        elif s == BimanualState.PRE_GRASP:
            # Wait until both EEs are at the actual grasp contact positions
            if self._both_near(
                q_left, q_right,
                self._grasp_pos_left,
                self._grasp_pos_right,
            ):
                # Activate weld constraints to attach the object
                if mj_model is not None:
                    self._activate_welds(mj_model, mj_data)
                self._grasp_activate_time = t
                self._transition_to(BimanualState.GRASP, t)

        elif s == BimanualState.GRASP:
            # Wait for settle_time after weld activation
            elapsed = t - self._grasp_activate_time
            if elapsed >= self.config.settle_time:
                self._transition_to(BimanualState.LIFT, t)

        elif s == BimanualState.LIFT:
            # Wait until object is lifted to target height
            if self._both_near(
                q_left, q_right,
                self._lift_pos_left,
                self._lift_pos_right,
            ):
                self._transition_to(BimanualState.CARRY, t)

        elif s == BimanualState.CARRY:
            # Wait until both arms have reached the carry (transport) positions
            if self._both_near(
                q_left, q_right,
                self._carry_pos_left,
                self._carry_pos_right,
            ):
                self._transition_to(BimanualState.LOWER, t)

        elif s == BimanualState.LOWER:
            # Wait until object is lowered to table height
            if self._both_near(
                q_left, q_right,
                self._lower_pos_left,
                self._lower_pos_right,
            ):
                # Deactivate weld constraints to release the object
                if mj_model is not None:
                    self._deactivate_welds(mj_model, mj_data)
                self._release_time = t
                self._transition_to(BimanualState.RELEASE, t)

        elif s == BimanualState.RELEASE:
            # Brief pause (one settle_time) before retreating
            elapsed = t - self._release_time
            if elapsed >= self.config.settle_time:
                self._transition_to(BimanualState.RETREAT, t)

        elif s == BimanualState.RETREAT:
            # Wait until both arms have retreated to clearance positions
            if self._both_near(
                q_left, q_right,
                self._retreat_pos_left,
                self._retreat_pos_right,
            ):
                self._transition_to(BimanualState.DONE, t)

        # BimanualState.DONE: no further transitions

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        """True when the state machine has reached DONE."""
        return self.state == BimanualState.DONE

    def reset(self, config: Optional[BimanualTaskConfig] = None) -> None:
        """Reset the state machine to IDLE for a new task.

        Args:
            config: Optional new task configuration.  If omitted, the
                    existing configuration is reused with targets rebuilt
                    (useful if the config was mutated in-place).
        """
        if config is not None:
            self.config = config
        self.state = BimanualState.IDLE
        self._grasp_activate_time = -1.0
        self._release_time = -1.0
        self.state_log.clear()
        self._build_targets()

    def get_state_log(self) -> list[tuple[float, str]]:
        """Return a copy of the state transition log.

        Returns:
            List of (time, state_name) tuples in transition order.
        """
        return list(self.state_log)
