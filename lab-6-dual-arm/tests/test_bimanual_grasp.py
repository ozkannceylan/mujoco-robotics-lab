"""Tests for BimanualGraspStateMachine and BimanualState.

Coverage:
- BimanualState enum: all 10 states exist with correct names
- State machine initialization: starts in IDLE state
- State transitions: IDLE→APPROACH (immediate), APPROACH→PRE_GRASP,
  PRE_GRASP→GRASP, GRASP→LIFT, LIFT→CARRY, CARRY→LOWER,
  LOWER→RELEASE, RELEASE→RETREAT, RETREAT→DONE
- Grasping property: True only in GRASP, LIFT, CARRY, LOWER states
- Weld activation logic: activated on PRE_GRASP→GRASP, deactivated on LOWER→RELEASE
- Target pose computation: correct positions and orientations per state
- State log recording: transitions are logged with timestamps
- Reset: returns to IDLE, clears log, rebuilds targets
- Edge cases: DONE state has no further transitions, is_done accessor
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pinocchio as pin

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from lab6_common import (
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    TABLE_SURFACE_Z,
    BOX_HALF_EXTENTS,
)
from dual_arm_model import DualArmModel
from bimanual_grasp import (
    BimanualGraspStateMachine,
    BimanualState,
    BimanualTaskConfig,
    _rotation_facing_pos_x,
    _rotation_facing_neg_x,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dual_model() -> DualArmModel:
    """Shared DualArmModel instance."""
    return DualArmModel()


@pytest.fixture
def default_config() -> BimanualTaskConfig:
    """Default task config with object at center, target offset in Y."""
    return BimanualTaskConfig(
        object_pos=np.array([0.5, 0.0, 0.245]),
        target_pos=np.array([0.5, 0.3, 0.245]),
    )


@pytest.fixture
def state_machine(dual_model, default_config) -> BimanualGraspStateMachine:
    """Fresh BimanualGraspStateMachine instance."""
    return BimanualGraspStateMachine(dual_model, default_config)


# ---------------------------------------------------------------------------
# BimanualState enum
# ---------------------------------------------------------------------------


class TestBimanualStateEnum:
    """Tests for the BimanualState enum."""

    def test_all_ten_states_exist(self):
        """BimanualState has exactly 10 members."""
        assert len(BimanualState) == 10

    @pytest.mark.parametrize("state_name", [
        "IDLE", "APPROACH", "PRE_GRASP", "GRASP", "LIFT",
        "CARRY", "LOWER", "RELEASE", "RETREAT", "DONE",
    ])
    def test_state_exists_by_name(self, state_name):
        """Each expected state name is a valid member."""
        assert hasattr(BimanualState, state_name)
        assert BimanualState[state_name].name == state_name

    def test_states_are_unique(self):
        """All state values are distinct."""
        values = [s.value for s in BimanualState]
        assert len(values) == len(set(values))

    def test_state_is_enum(self):
        """BimanualState members are Enum instances."""
        assert isinstance(BimanualState.IDLE, BimanualState)
        assert isinstance(BimanualState.DONE, BimanualState)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Tests for BimanualGraspStateMachine construction."""

    def test_initial_state_is_idle(self, state_machine):
        """State machine starts in IDLE."""
        assert state_machine.state == BimanualState.IDLE

    def test_is_done_false_initially(self, state_machine):
        """is_done is False when just created."""
        assert state_machine.is_done is False

    def test_state_log_empty_initially(self, state_machine):
        """State log starts empty."""
        assert state_machine.state_log == []

    def test_grasp_activate_time_initialized(self, state_machine):
        """Internal _grasp_activate_time is -1.0 on init."""
        assert state_machine._grasp_activate_time == -1.0

    def test_release_time_initialized(self, state_machine):
        """Internal _release_time is -1.0 on init."""
        assert state_machine._release_time == -1.0

    def test_config_stored(self, state_machine, default_config):
        """Config is stored on the state machine."""
        assert state_machine.config is default_config

    def test_default_gains_used_when_none(self, dual_model, default_config):
        """When gains=None, default DualImpedanceGains are used."""
        from cooperative_controller import DualImpedanceGains
        sm = BimanualGraspStateMachine(dual_model, default_config, gains=None)
        assert isinstance(sm.controller.gains, DualImpedanceGains)


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


class TestRotationHelpers:
    """Tests for _rotation_facing_pos_x and _rotation_facing_neg_x."""

    def test_rotation_pos_x_is_valid_rotation(self):
        """_rotation_facing_pos_x returns a valid rotation matrix."""
        R = _rotation_facing_pos_x()
        assert R.shape == (3, 3)
        # Orthogonal: R^T R = I
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        # det = +1
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_rotation_neg_x_is_valid_rotation(self):
        """_rotation_facing_neg_x returns a valid rotation matrix."""
        R = _rotation_facing_neg_x()
        assert R.shape == (3, 3)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_rotation_pos_x_z_axis(self):
        """Left grasp rotation has Z-axis = [1, 0, 0] (approach +X)."""
        R = _rotation_facing_pos_x()
        z_axis = R[:, 2]
        assert np.allclose(z_axis, [1.0, 0.0, 0.0], atol=1e-12)

    def test_rotation_neg_x_z_axis(self):
        """Right grasp rotation has Z-axis = [-1, 0, 0] (approach -X)."""
        R = _rotation_facing_neg_x()
        z_axis = R[:, 2]
        assert np.allclose(z_axis, [-1.0, 0.0, 0.0], atol=1e-12)

    def test_rotations_face_each_other(self):
        """Left and right grasp Z-axes point in opposite directions."""
        R_left = _rotation_facing_pos_x()
        R_right = _rotation_facing_neg_x()
        z_left = R_left[:, 2]
        z_right = R_right[:, 2]
        assert np.isclose(np.dot(z_left, z_right), -1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# State transitions — IDLE → APPROACH (immediate)
# ---------------------------------------------------------------------------


class TestIdleToApproach:
    """IDLE transitions to APPROACH immediately on first step."""

    def test_idle_transitions_to_approach_on_step(self, state_machine):
        """Calling step in IDLE transitions to APPROACH."""
        assert state_machine.state == BimanualState.IDLE
        state_machine._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 0.0, None, None)
        assert state_machine.state == BimanualState.APPROACH

    def test_transition_logged(self, dual_model, default_config):
        """IDLE→APPROACH transition is logged with correct time."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 1.5, None, None)
        assert len(sm.state_log) == 1
        assert sm.state_log[0] == (1.5, "APPROACH")


# ---------------------------------------------------------------------------
# State transitions — position-based
# ---------------------------------------------------------------------------


class TestPositionBasedTransitions:
    """Tests for transitions that require both EEs to be near targets."""

    def test_approach_stays_when_far(self, dual_model, default_config):
        """APPROACH does not transition if EEs are far from approach targets."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.APPROACH
        # Use home config which is likely far from approach positions
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 1.0, None, None)
        assert sm.state == BimanualState.APPROACH

    def test_approach_to_pregrasp_when_near(self, dual_model, default_config):
        """APPROACH transitions to PRE_GRASP when EEs are at approach positions."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.APPROACH

        # Mock _both_near to return True
        sm._both_near = MagicMock(return_value=True)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 2.0, None, None)
        assert sm.state == BimanualState.PRE_GRASP

    def test_pregrasp_to_grasp_when_near(self, dual_model, default_config):
        """PRE_GRASP transitions to GRASP when EEs are at grasp positions."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.PRE_GRASP

        sm._both_near = MagicMock(return_value=True)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 3.0, None, None)
        assert sm.state == BimanualState.GRASP

    def test_pregrasp_to_grasp_records_activate_time(self, dual_model, default_config):
        """PRE_GRASP→GRASP records grasp activation time."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.PRE_GRASP
        sm._both_near = MagicMock(return_value=True)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 4.5, None, None)
        assert sm._grasp_activate_time == 4.5

    def test_lift_to_carry_when_near(self, dual_model, default_config):
        """LIFT transitions to CARRY when EEs are at lift positions."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.LIFT
        sm._both_near = MagicMock(return_value=True)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 6.0, None, None)
        assert sm.state == BimanualState.CARRY

    def test_carry_to_lower_when_near(self, dual_model, default_config):
        """CARRY transitions to LOWER when EEs are at carry positions."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.CARRY
        sm._both_near = MagicMock(return_value=True)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 7.0, None, None)
        assert sm.state == BimanualState.LOWER

    def test_lower_to_release_when_near(self, dual_model, default_config):
        """LOWER transitions to RELEASE when EEs are at lower positions."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.LOWER
        sm._both_near = MagicMock(return_value=True)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 8.0, None, None)
        assert sm.state == BimanualState.RELEASE

    def test_lower_to_release_records_release_time(self, dual_model, default_config):
        """LOWER→RELEASE records release time."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.LOWER
        sm._both_near = MagicMock(return_value=True)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 8.5, None, None)
        assert sm._release_time == 8.5

    def test_retreat_to_done_when_near(self, dual_model, default_config):
        """RETREAT transitions to DONE when EEs are at retreat positions."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.RETREAT
        sm._both_near = MagicMock(return_value=True)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 10.0, None, None)
        assert sm.state == BimanualState.DONE
        assert sm.is_done is True


# ---------------------------------------------------------------------------
# State transitions — time-based
# ---------------------------------------------------------------------------


class TestTimeBasedTransitions:
    """Tests for transitions gated by elapsed time (settle_time)."""

    def test_grasp_stays_before_settle_time(self, dual_model, default_config):
        """GRASP does not transition before settle_time elapses."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.GRASP
        sm._grasp_activate_time = 5.0

        # settle_time is 0.5 by default; try at 5.3 (only 0.3s elapsed)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 5.3, None, None)
        assert sm.state == BimanualState.GRASP

    def test_grasp_to_lift_after_settle_time(self, dual_model, default_config):
        """GRASP transitions to LIFT after settle_time elapses."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.GRASP
        sm._grasp_activate_time = 5.0

        # settle_time=0.5; at t=5.5 it should transition
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 5.5, None, None)
        assert sm.state == BimanualState.LIFT

    def test_grasp_to_lift_well_after_settle_time(self, dual_model, default_config):
        """GRASP transitions to LIFT when well past settle_time."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.GRASP
        sm._grasp_activate_time = 5.0
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 10.0, None, None)
        assert sm.state == BimanualState.LIFT

    def test_release_stays_before_settle_time(self, dual_model, default_config):
        """RELEASE does not transition before settle_time elapses."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.RELEASE
        sm._release_time = 8.0

        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 8.2, None, None)
        assert sm.state == BimanualState.RELEASE

    def test_release_to_retreat_after_settle_time(self, dual_model, default_config):
        """RELEASE transitions to RETREAT after settle_time elapses."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.RELEASE
        sm._release_time = 8.0

        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 8.5, None, None)
        assert sm.state == BimanualState.RETREAT


# ---------------------------------------------------------------------------
# Grasping property (via _targets_for_state)
# ---------------------------------------------------------------------------


class TestGraspingFlag:
    """Tests that _targets_for_state returns correct grasping flag per state."""

    @pytest.mark.parametrize("state,expected_grasping", [
        (BimanualState.IDLE, False),
        (BimanualState.APPROACH, False),
        (BimanualState.PRE_GRASP, False),
        (BimanualState.GRASP, True),
        (BimanualState.LIFT, True),
        (BimanualState.CARRY, True),
        (BimanualState.LOWER, True),
        (BimanualState.RELEASE, False),
        (BimanualState.RETREAT, False),
        (BimanualState.DONE, False),
    ])
    def test_grasping_flag_per_state(self, state_machine, state, expected_grasping):
        """Grasping flag matches expected value for each state."""
        _, _, grasping = state_machine._targets_for_state(
            state, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        assert grasping is expected_grasping, (
            f"State {state.name}: expected grasping={expected_grasping}, got {grasping}"
        )


# ---------------------------------------------------------------------------
# Weld activation logic
# ---------------------------------------------------------------------------


class TestWeldActivation:
    """Tests for weld constraint activation/deactivation."""

    def test_welds_activated_on_pregrasp_to_grasp(self, dual_model, default_config):
        """Weld constraints are activated when transitioning PRE_GRASP → GRASP."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.PRE_GRASP
        sm._both_near = MagicMock(return_value=True)

        mock_activate = MagicMock()
        sm._activate_welds = mock_activate

        mj_model = MagicMock()
        mj_data = MagicMock()
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 3.0, mj_model, mj_data)

        mock_activate.assert_called_once_with(mj_model, mj_data)

    def test_welds_not_activated_without_mj_model(self, dual_model, default_config):
        """When mj_model is None, _activate_welds is not called."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.PRE_GRASP
        sm._both_near = MagicMock(return_value=True)

        mock_activate = MagicMock()
        sm._activate_welds = mock_activate

        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 3.0, None, None)
        mock_activate.assert_not_called()
        # But should still transition
        assert sm.state == BimanualState.GRASP

    def test_welds_deactivated_on_lower_to_release(self, dual_model, default_config):
        """Weld constraints are deactivated when transitioning LOWER → RELEASE."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.LOWER
        sm._both_near = MagicMock(return_value=True)

        mock_deactivate = MagicMock()
        sm._deactivate_welds = mock_deactivate

        mj_model = MagicMock()
        mj_data = MagicMock()
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 8.0, mj_model, mj_data)

        mock_deactivate.assert_called_once_with(mj_model, mj_data)

    def test_welds_not_deactivated_without_mj_model(self, dual_model, default_config):
        """When mj_model is None, _deactivate_welds is not called."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.LOWER
        sm._both_near = MagicMock(return_value=True)

        mock_deactivate = MagicMock()
        sm._deactivate_welds = mock_deactivate

        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 8.0, None, None)
        mock_deactivate.assert_not_called()
        assert sm.state == BimanualState.RELEASE


# ---------------------------------------------------------------------------
# Target pose computation
# ---------------------------------------------------------------------------


class TestTargetPoseComputation:
    """Tests for precomputed target positions and _targets_for_state output."""

    def test_grasp_positions_symmetric_about_object(self, state_machine, default_config):
        """Left and right grasp positions are symmetric about object center along X."""
        obj_x = default_config.object_pos[0]
        dx = default_config.grasp_offset_x

        assert np.isclose(state_machine._grasp_pos_left[0], obj_x - dx)
        assert np.isclose(state_machine._grasp_pos_right[0], obj_x + dx)
        # Y and Z match object position
        assert np.isclose(state_machine._grasp_pos_left[1], default_config.object_pos[1])
        assert np.isclose(state_machine._grasp_pos_right[1], default_config.object_pos[1])
        assert np.isclose(state_machine._grasp_pos_left[2], default_config.object_pos[2])
        assert np.isclose(state_machine._grasp_pos_right[2], default_config.object_pos[2])

    def test_approach_positions_further_out(self, state_machine, default_config):
        """Approach positions are further from object center than grasp positions."""
        clr = default_config.approach_clearance
        assert np.isclose(
            state_machine._approach_pos_left[0],
            state_machine._grasp_pos_left[0] - clr,
        )
        assert np.isclose(
            state_machine._approach_pos_right[0],
            state_machine._grasp_pos_right[0] + clr,
        )

    def test_lift_positions_higher_z(self, state_machine, default_config):
        """Lift positions have Z = object_z + lift_height."""
        expected_z = default_config.object_pos[2] + default_config.lift_height
        assert np.isclose(state_machine._lift_pos_left[2], expected_z)
        assert np.isclose(state_machine._lift_pos_right[2], expected_z)

    def test_carry_positions_at_target_xy(self, state_machine, default_config):
        """Carry positions are at target XY with lift height."""
        dx = default_config.grasp_offset_x
        expected_z = default_config.target_pos[2] + default_config.lift_height

        assert np.isclose(state_machine._carry_pos_left[0], default_config.target_pos[0] - dx)
        assert np.isclose(state_machine._carry_pos_right[0], default_config.target_pos[0] + dx)
        assert np.isclose(state_machine._carry_pos_left[1], default_config.target_pos[1])
        assert np.isclose(state_machine._carry_pos_right[1], default_config.target_pos[1])
        assert np.isclose(state_machine._carry_pos_left[2], expected_z)
        assert np.isclose(state_machine._carry_pos_right[2], expected_z)

    def test_lower_positions_at_place_height(self, state_machine):
        """Lower positions have Z = TABLE_SURFACE_Z + BOX_HALF_EXTENTS[2]."""
        expected_z = TABLE_SURFACE_Z + BOX_HALF_EXTENTS[2]
        assert np.isclose(state_machine._lower_pos_left[2], expected_z)
        assert np.isclose(state_machine._lower_pos_right[2], expected_z)

    def test_retreat_positions_have_clearance(self, state_machine, default_config):
        """Retreat positions add approach_clearance beyond lower positions."""
        dx = default_config.grasp_offset_x
        clr = default_config.approach_clearance
        tgt = default_config.target_pos

        assert np.isclose(
            state_machine._retreat_pos_left[0],
            tgt[0] - dx - clr,
        )
        assert np.isclose(
            state_machine._retreat_pos_right[0],
            tgt[0] + dx + clr,
        )

    def test_targets_for_state_returns_se3(self, state_machine):
        """_targets_for_state returns pin.SE3 objects."""
        tgt_l, tgt_r, _ = state_machine._targets_for_state(
            BimanualState.APPROACH, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        assert isinstance(tgt_l, pin.SE3)
        assert isinstance(tgt_r, pin.SE3)

    def test_approach_target_positions_match(self, state_machine):
        """APPROACH state targets have the precomputed approach positions."""
        tgt_l, tgt_r, _ = state_machine._targets_for_state(
            BimanualState.APPROACH, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        assert np.allclose(tgt_l.translation, state_machine._approach_pos_left, atol=1e-12)
        assert np.allclose(tgt_r.translation, state_machine._approach_pos_right, atol=1e-12)

    def test_idle_targets_are_current_fk(self, state_machine, dual_model):
        """IDLE state targets equal current FK poses (hold in place)."""
        tgt_l, tgt_r, _ = state_machine._targets_for_state(
            BimanualState.IDLE, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        fk_l = dual_model.fk_left(Q_HOME_LEFT)
        fk_r = dual_model.fk_right(Q_HOME_RIGHT)
        assert np.allclose(tgt_l.translation, fk_l.translation, atol=1e-10)
        assert np.allclose(tgt_r.translation, fk_r.translation, atol=1e-10)

    def test_grasp_and_pregrasp_same_position(self, state_machine):
        """GRASP and PRE_GRASP states target the same grasp positions."""
        tgt_l_pre, tgt_r_pre, _ = state_machine._targets_for_state(
            BimanualState.PRE_GRASP, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        tgt_l_grasp, tgt_r_grasp, _ = state_machine._targets_for_state(
            BimanualState.GRASP, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        assert np.allclose(tgt_l_pre.translation, tgt_l_grasp.translation, atol=1e-12)
        assert np.allclose(tgt_r_pre.translation, tgt_r_grasp.translation, atol=1e-12)

    def test_release_and_lower_same_position(self, state_machine):
        """RELEASE state targets the same positions as LOWER."""
        tgt_l_lower, tgt_r_lower, _ = state_machine._targets_for_state(
            BimanualState.LOWER, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        tgt_l_release, tgt_r_release, _ = state_machine._targets_for_state(
            BimanualState.RELEASE, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        assert np.allclose(tgt_l_lower.translation, tgt_l_release.translation, atol=1e-12)
        assert np.allclose(tgt_r_lower.translation, tgt_r_release.translation, atol=1e-12)

    def test_done_and_retreat_same_position(self, state_machine):
        """DONE state targets the same positions as RETREAT."""
        tgt_l_ret, tgt_r_ret, _ = state_machine._targets_for_state(
            BimanualState.RETREAT, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        tgt_l_done, tgt_r_done, _ = state_machine._targets_for_state(
            BimanualState.DONE, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        assert np.allclose(tgt_l_ret.translation, tgt_l_done.translation, atol=1e-12)
        assert np.allclose(tgt_r_ret.translation, tgt_r_done.translation, atol=1e-12)

    def test_left_orientation_is_pos_x_grasp(self, state_machine):
        """Left arm targets use the +X facing rotation."""
        R_expected = _rotation_facing_pos_x()
        tgt_l, _, _ = state_machine._targets_for_state(
            BimanualState.APPROACH, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        assert np.allclose(tgt_l.rotation, R_expected, atol=1e-12)

    def test_right_orientation_is_neg_x_grasp(self, state_machine):
        """Right arm targets use the -X facing rotation."""
        R_expected = _rotation_facing_neg_x()
        _, tgt_r, _ = state_machine._targets_for_state(
            BimanualState.APPROACH, Q_HOME_LEFT, Q_HOME_RIGHT,
        )
        assert np.allclose(tgt_r.rotation, R_expected, atol=1e-12)


# ---------------------------------------------------------------------------
# State log
# ---------------------------------------------------------------------------


class TestStateLog:
    """Tests for state transition logging."""

    def test_log_records_transitions(self, dual_model, default_config):
        """State log accumulates transitions with timestamps."""
        sm = BimanualGraspStateMachine(dual_model, default_config)

        # IDLE → APPROACH
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 0.0, None, None)
        assert len(sm.state_log) == 1
        assert sm.state_log[0] == (0.0, "APPROACH")

    def test_log_records_multiple_transitions(self, dual_model, default_config):
        """Multiple transitions accumulate in the log."""
        sm = BimanualGraspStateMachine(dual_model, default_config)

        # IDLE → APPROACH
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 0.0, None, None)
        # Force APPROACH → PRE_GRASP
        sm._both_near = MagicMock(return_value=True)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 1.0, None, None)

        assert len(sm.state_log) == 2
        assert sm.state_log[0] == (0.0, "APPROACH")
        assert sm.state_log[1] == (1.0, "PRE_GRASP")

    def test_get_state_log_returns_copy(self, dual_model, default_config):
        """get_state_log returns a copy, not the internal list."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 0.0, None, None)
        log = sm.get_state_log()
        assert log == sm.state_log
        assert log is not sm.state_log  # Different object


# ---------------------------------------------------------------------------
# DONE state — no further transitions
# ---------------------------------------------------------------------------


class TestDoneState:
    """Tests for DONE state behavior."""

    def test_done_has_no_further_transitions(self, dual_model, default_config):
        """DONE state does not transition to anything else."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.DONE
        log_len_before = len(sm.state_log)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 100.0, None, None)
        assert sm.state == BimanualState.DONE
        assert len(sm.state_log) == log_len_before

    def test_is_done_true_in_done_state(self, dual_model, default_config):
        """is_done returns True when in DONE state."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.DONE
        assert sm.is_done is True

    def test_is_done_false_for_all_other_states(self, dual_model, default_config):
        """is_done returns False for all states except DONE."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        for state in BimanualState:
            if state == BimanualState.DONE:
                continue
            sm.state = state
            assert sm.is_done is False, f"is_done should be False in {state.name}"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for state machine reset."""

    def test_reset_returns_to_idle(self, dual_model, default_config):
        """reset() returns state to IDLE."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm.state = BimanualState.CARRY
        sm.reset()
        assert sm.state == BimanualState.IDLE

    def test_reset_clears_log(self, dual_model, default_config):
        """reset() clears the state log."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, 0.0, None, None)
        assert len(sm.state_log) > 0
        sm.reset()
        assert sm.state_log == []

    def test_reset_clears_timing(self, dual_model, default_config):
        """reset() resets _grasp_activate_time and _release_time."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        sm._grasp_activate_time = 5.0
        sm._release_time = 8.0
        sm.reset()
        assert sm._grasp_activate_time == -1.0
        assert sm._release_time == -1.0

    def test_reset_with_new_config(self, dual_model, default_config):
        """reset() with a new config updates targets."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        new_config = BimanualTaskConfig(
            object_pos=np.array([0.4, 0.1, 0.3]),
            target_pos=np.array([0.6, -0.1, 0.3]),
        )
        sm.reset(config=new_config)
        assert sm.config is new_config
        assert sm.state == BimanualState.IDLE
        # Check that targets were rebuilt with new config
        assert np.isclose(
            sm._grasp_pos_left[0],
            new_config.object_pos[0] - new_config.grasp_offset_x,
        )

    def test_reset_without_config_rebuilds_targets(self, dual_model, default_config):
        """reset() without config still rebuilds targets."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        original_grasp_left = sm._grasp_pos_left.copy()
        sm.reset()
        assert np.allclose(sm._grasp_pos_left, original_grasp_left, atol=1e-12)


# ---------------------------------------------------------------------------
# BimanualTaskConfig
# ---------------------------------------------------------------------------


class TestBimanualTaskConfig:
    """Tests for task configuration defaults."""

    def test_default_lift_height(self):
        """Default lift_height is 0.15."""
        cfg = BimanualTaskConfig(
            object_pos=np.zeros(3), target_pos=np.zeros(3),
        )
        assert cfg.lift_height == 0.15

    def test_default_approach_clearance(self):
        """Default approach_clearance is 0.08."""
        cfg = BimanualTaskConfig(
            object_pos=np.zeros(3), target_pos=np.zeros(3),
        )
        assert cfg.approach_clearance == 0.08

    def test_default_grasp_offset_x(self):
        """Default grasp_offset_x is 0.18."""
        cfg = BimanualTaskConfig(
            object_pos=np.zeros(3), target_pos=np.zeros(3),
        )
        assert cfg.grasp_offset_x == 0.18

    def test_default_position_tolerance(self):
        """Default position_tolerance is 0.01."""
        cfg = BimanualTaskConfig(
            object_pos=np.zeros(3), target_pos=np.zeros(3),
        )
        assert cfg.position_tolerance == 0.01

    def test_default_settle_time(self):
        """Default settle_time is 0.5."""
        cfg = BimanualTaskConfig(
            object_pos=np.zeros(3), target_pos=np.zeros(3),
        )
        assert cfg.settle_time == 0.5

    def test_custom_config_values(self):
        """Custom config values override defaults."""
        cfg = BimanualTaskConfig(
            object_pos=np.array([1.0, 2.0, 3.0]),
            target_pos=np.array([4.0, 5.0, 6.0]),
            lift_height=0.3,
            settle_time=1.0,
        )
        assert cfg.lift_height == 0.3
        assert cfg.settle_time == 1.0
        assert np.allclose(cfg.object_pos, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Full transition sequence
# ---------------------------------------------------------------------------


class TestFullSequence:
    """Test a complete state machine run-through using mocked _both_near."""

    def test_full_state_sequence(self, dual_model, default_config):
        """Walk through all 10 states in the correct order."""
        sm = BimanualGraspStateMachine(dual_model, default_config)
        t = 0.0

        # IDLE → APPROACH (immediate)
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, t, None, None)
        assert sm.state == BimanualState.APPROACH

        # APPROACH → PRE_GRASP
        sm._both_near = MagicMock(return_value=True)
        t = 1.0
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, t, None, None)
        assert sm.state == BimanualState.PRE_GRASP

        # PRE_GRASP → GRASP
        t = 2.0
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, t, None, None)
        assert sm.state == BimanualState.GRASP

        # GRASP → LIFT (need to wait settle_time=0.5)
        t = 2.3  # Only 0.3s elapsed
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, t, None, None)
        assert sm.state == BimanualState.GRASP  # Not yet

        t = 2.5  # 0.5s elapsed
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, t, None, None)
        assert sm.state == BimanualState.LIFT

        # LIFT → CARRY
        t = 3.0
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, t, None, None)
        assert sm.state == BimanualState.CARRY

        # CARRY → LOWER
        t = 4.0
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, t, None, None)
        assert sm.state == BimanualState.LOWER

        # LOWER → RELEASE
        t = 5.0
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, t, None, None)
        assert sm.state == BimanualState.RELEASE

        # RELEASE → RETREAT (need settle_time)
        t = 5.3
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, t, None, None)
        assert sm.state == BimanualState.RELEASE  # Not yet

        t = 5.5
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, t, None, None)
        assert sm.state == BimanualState.RETREAT

        # RETREAT → DONE
        t = 6.0
        sm._maybe_transition(Q_HOME_LEFT, Q_HOME_RIGHT, t, None, None)
        assert sm.state == BimanualState.DONE
        assert sm.is_done is True

        # Verify full log
        expected_states = [
            "APPROACH", "PRE_GRASP", "GRASP", "LIFT",
            "CARRY", "LOWER", "RELEASE", "RETREAT", "DONE",
        ]
        logged_states = [name for _, name in sm.state_log]
        assert logged_states == expected_states
