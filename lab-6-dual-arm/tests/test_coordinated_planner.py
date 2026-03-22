"""Tests for CoordinatedPlanner — joint-linear, synchronized-linear,
master-slave, symmetric planning, SynchronizedTrajectory dataclass, and edge cases."""
import sys
from pathlib import Path
import math

import numpy as np
import pytest
import pinocchio as pin

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from lab6_common import Q_HOME_LEFT, Q_HOME_RIGHT, DT, NUM_JOINTS_PER_ARM, VEL_LIMITS
from dual_arm_model import DualArmModel, ObjectFrame
from coordinated_planner import CoordinatedPlanner, SynchronizedTrajectory, _VEL_SAFETY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dual_model() -> DualArmModel:
    """Shared DualArmModel instance for the test module."""
    return DualArmModel()


@pytest.fixture(scope="module")
def planner(dual_model) -> CoordinatedPlanner:
    """Shared CoordinatedPlanner instance for the test module."""
    return CoordinatedPlanner(dual_model)


@pytest.fixture(scope="module")
def object_frame(dual_model) -> ObjectFrame:
    """ObjectFrame at the midpoint between the two home EE poses."""
    left_ee = dual_model.fk_left(Q_HOME_LEFT)
    right_ee = dual_model.fk_right(Q_HOME_RIGHT)
    mid_pos = 0.5 * (left_ee.translation + right_ee.translation)
    obj_pose = pin.SE3(np.eye(3), mid_pos)
    return ObjectFrame.from_ee_poses(obj_pose, left_ee, right_ee)


# ---------------------------------------------------------------------------
# SynchronizedTrajectory dataclass
# ---------------------------------------------------------------------------

class TestSynchronizedTrajectory:
    """Tests for the SynchronizedTrajectory dataclass."""

    def test_valid_construction(self):
        """Valid arrays produce a SynchronizedTrajectory without error."""
        T = 50
        traj = SynchronizedTrajectory(
            timestamps=np.linspace(0.0, 1.0, T),
            q_left=np.zeros((T, NUM_JOINTS_PER_ARM)),
            qd_left=np.zeros((T, NUM_JOINTS_PER_ARM)),
            q_right=np.zeros((T, NUM_JOINTS_PER_ARM)),
            qd_right=np.zeros((T, NUM_JOINTS_PER_ARM)),
        )
        assert traj.n_steps == T
        assert np.isclose(traj.duration, 1.0)

    def test_duration_property(self):
        """duration returns the last timestamp value."""
        T = 10
        ts = np.linspace(0.0, 2.5, T)
        traj = SynchronizedTrajectory(
            timestamps=ts,
            q_left=np.zeros((T, 6)),
            qd_left=np.zeros((T, 6)),
            q_right=np.zeros((T, 6)),
            qd_right=np.zeros((T, 6)),
        )
        assert np.isclose(traj.duration, 2.5)

    def test_n_steps_property(self):
        """n_steps matches the length of timestamps."""
        T = 37
        traj = SynchronizedTrajectory(
            timestamps=np.linspace(0.0, 1.0, T),
            q_left=np.zeros((T, 6)),
            qd_left=np.zeros((T, 6)),
            q_right=np.zeros((T, 6)),
            qd_right=np.zeros((T, 6)),
        )
        assert traj.n_steps == T

    def test_post_init_raises_on_wrong_q_left_rows(self):
        """ValueError if q_left has wrong number of rows."""
        T = 10
        with pytest.raises(ValueError, match="q_left"):
            SynchronizedTrajectory(
                timestamps=np.linspace(0.0, 1.0, T),
                q_left=np.zeros((T + 5, 6)),
                qd_left=np.zeros((T, 6)),
                q_right=np.zeros((T, 6)),
                qd_right=np.zeros((T, 6)),
            )

    def test_post_init_raises_on_wrong_q_right_cols(self):
        """ValueError if q_right has wrong number of columns."""
        T = 10
        with pytest.raises(ValueError, match="q_right"):
            SynchronizedTrajectory(
                timestamps=np.linspace(0.0, 1.0, T),
                q_left=np.zeros((T, 6)),
                qd_left=np.zeros((T, 6)),
                q_right=np.zeros((T, 4)),  # wrong columns
                qd_right=np.zeros((T, 6)),
            )

    def test_post_init_raises_on_wrong_qd_left_shape(self):
        """ValueError if qd_left has mismatched shape."""
        T = 10
        with pytest.raises(ValueError, match="qd_left"):
            SynchronizedTrajectory(
                timestamps=np.linspace(0.0, 1.0, T),
                q_left=np.zeros((T, 6)),
                qd_left=np.zeros((T - 2, 6)),
                q_right=np.zeros((T, 6)),
                qd_right=np.zeros((T, 6)),
            )

    def test_post_init_raises_on_wrong_qd_right_shape(self):
        """ValueError if qd_right has mismatched shape."""
        T = 10
        with pytest.raises(ValueError, match="qd_right"):
            SynchronizedTrajectory(
                timestamps=np.linspace(0.0, 1.0, T),
                q_left=np.zeros((T, 6)),
                qd_left=np.zeros((T, 6)),
                q_right=np.zeros((T, 6)),
                qd_right=np.zeros((T, 3)),
            )


# ---------------------------------------------------------------------------
# plan_joint_linear
# ---------------------------------------------------------------------------

class TestPlanJointLinear:
    """Tests for joint-space linear interpolation."""

    def test_start_position_matches(self, planner):
        """First row of trajectory matches the start configuration."""
        q_ls = Q_HOME_LEFT.copy()
        q_le = Q_HOME_LEFT + 0.1
        q_rs = Q_HOME_RIGHT.copy()
        q_re = Q_HOME_RIGHT + 0.1
        traj = planner.plan_joint_linear(q_ls, q_le, q_rs, q_re, duration=1.0)

        assert np.allclose(traj.q_left[0], q_ls, atol=1e-10)
        assert np.allclose(traj.q_right[0], q_rs, atol=1e-10)

    def test_end_position_matches(self, planner):
        """Last row of trajectory matches the end configuration."""
        q_ls = Q_HOME_LEFT.copy()
        q_le = Q_HOME_LEFT + 0.2
        q_rs = Q_HOME_RIGHT.copy()
        q_re = Q_HOME_RIGHT - 0.15
        traj = planner.plan_joint_linear(q_ls, q_le, q_rs, q_re, duration=1.0)

        assert np.allclose(traj.q_left[-1], q_le, atol=1e-10)
        assert np.allclose(traj.q_right[-1], q_re, atol=1e-10)

    def test_timestamp_spacing_is_dt(self, planner):
        """Consecutive timestamp differences are approximately DT."""
        traj = planner.plan_joint_linear(
            Q_HOME_LEFT, Q_HOME_LEFT + 0.1,
            Q_HOME_RIGHT, Q_HOME_RIGHT + 0.1,
            duration=0.5,
        )
        diffs = np.diff(traj.timestamps)
        assert np.allclose(diffs, DT, atol=1e-6), (
            f"Timestamp spacing not uniform: min={diffs.min():.6f}, max={diffs.max():.6f}, DT={DT}"
        )

    def test_timestamps_start_at_zero(self, planner):
        """Timestamps begin at 0.0."""
        traj = planner.plan_joint_linear(
            Q_HOME_LEFT, Q_HOME_LEFT + 0.05,
            Q_HOME_RIGHT, Q_HOME_RIGHT + 0.05,
            duration=0.2,
        )
        assert np.isclose(traj.timestamps[0], 0.0)

    def test_velocity_shape_matches_position(self, planner):
        """Velocity arrays have the same shape as position arrays."""
        traj = planner.plan_joint_linear(
            Q_HOME_LEFT, Q_HOME_LEFT + 0.1,
            Q_HOME_RIGHT, Q_HOME_RIGHT + 0.1,
            duration=1.0,
        )
        assert traj.qd_left.shape == traj.q_left.shape
        assert traj.qd_right.shape == traj.q_right.shape

    def test_velocity_consistent_with_position(self, planner):
        """For a linear trajectory, velocity should be roughly constant in the interior."""
        delta = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        traj = planner.plan_joint_linear(
            Q_HOME_LEFT, Q_HOME_LEFT + delta,
            Q_HOME_RIGHT, Q_HOME_RIGHT,
            duration=1.0,
        )
        # In the middle of the trajectory, left velocity on joint 0 should be ~0.5 rad/s
        mid = traj.n_steps // 2
        assert abs(traj.qd_left[mid, 0]) > 0.0, "Left velocity should be non-zero for moving joint"
        # Right arm should have near-zero velocity (not moving)
        assert np.allclose(traj.qd_right[mid], 0.0, atol=1e-6), (
            "Right arm velocity should be ~zero when not moving"
        )

    def test_auto_duration_when_none(self, planner):
        """When duration=None, auto-duration produces a positive value and valid trajectory."""
        traj = planner.plan_joint_linear(
            Q_HOME_LEFT, Q_HOME_LEFT + 0.3,
            Q_HOME_RIGHT, Q_HOME_RIGHT + 0.3,
        )
        assert traj.duration > 0.0
        assert traj.n_steps >= 2

    def test_trajectory_shapes(self, planner):
        """All arrays in the trajectory have consistent shapes."""
        traj = planner.plan_joint_linear(
            Q_HOME_LEFT, Q_HOME_LEFT + 0.1,
            Q_HOME_RIGHT, Q_HOME_RIGHT + 0.1,
            duration=0.5,
        )
        T = traj.n_steps
        assert traj.timestamps.shape == (T,)
        assert traj.q_left.shape == (T, 6)
        assert traj.qd_left.shape == (T, 6)
        assert traj.q_right.shape == (T, 6)
        assert traj.qd_right.shape == (T, 6)


# ---------------------------------------------------------------------------
# plan_synchronized_linear
# ---------------------------------------------------------------------------

class TestPlanSynchronizedLinear:
    """Tests for task-space synchronized linear planning."""

    def test_left_endpoint_error_below_1mm(self, planner, dual_model):
        """Left EE at final step is within 1mm of the target."""
        q_l_init = Q_HOME_LEFT.copy()
        q_r_init = Q_HOME_RIGHT.copy()

        # Small displacement target for left arm
        target_left = dual_model.fk_left(q_l_init)
        target_left = pin.SE3(target_left.rotation, target_left.translation + np.array([0.05, 0.0, 0.0]))
        target_right = dual_model.fk_right(q_r_init)

        traj = planner.plan_synchronized_linear(
            target_left, target_right, q_l_init, q_r_init, duration=1.0,
        )

        ee_final = dual_model.fk_left_pos(traj.q_left[-1])
        err = np.linalg.norm(ee_final - target_left.translation)
        assert err < 1e-3, f"Left EE error {err * 1000:.3f} mm > 1 mm"

    def test_right_endpoint_error_below_1mm(self, planner, dual_model):
        """Right EE at final step is within 1mm of the target."""
        q_l_init = Q_HOME_LEFT.copy()
        q_r_init = Q_HOME_RIGHT.copy()

        target_left = dual_model.fk_left(q_l_init)
        target_right = dual_model.fk_right(q_r_init)
        target_right = pin.SE3(target_right.rotation, target_right.translation + np.array([-0.05, 0.0, 0.0]))

        traj = planner.plan_synchronized_linear(
            target_left, target_right, q_l_init, q_r_init, duration=1.0,
        )

        ee_final = dual_model.fk_right_pos(traj.q_right[-1])
        err = np.linalg.norm(ee_final - target_right.translation)
        assert err < 1e-3, f"Right EE error {err * 1000:.3f} mm > 1 mm"

    def test_both_endpoints_within_tolerance(self, planner, dual_model):
        """Both arms reach their targets simultaneously within 1mm."""
        q_l_init = Q_HOME_LEFT.copy()
        q_r_init = Q_HOME_RIGHT.copy()

        left_ee = dual_model.fk_left(q_l_init)
        right_ee = dual_model.fk_right(q_r_init)
        target_left = pin.SE3(left_ee.rotation, left_ee.translation + np.array([0.03, 0.02, 0.0]))
        target_right = pin.SE3(right_ee.rotation, right_ee.translation + np.array([-0.03, 0.02, 0.0]))

        traj = planner.plan_synchronized_linear(
            target_left, target_right, q_l_init, q_r_init, duration=1.0,
        )

        err_left = np.linalg.norm(dual_model.fk_left_pos(traj.q_left[-1]) - target_left.translation)
        err_right = np.linalg.norm(dual_model.fk_right_pos(traj.q_right[-1]) - target_right.translation)
        assert err_left < 1e-3, f"Left EE error {err_left * 1000:.3f} mm"
        assert err_right < 1e-3, f"Right EE error {err_right * 1000:.3f} mm"

    def test_output_is_synchronized_trajectory(self, planner, dual_model):
        """Return type is SynchronizedTrajectory with valid shapes."""
        q_l = Q_HOME_LEFT.copy()
        q_r = Q_HOME_RIGHT.copy()
        target_l = dual_model.fk_left(q_l)
        target_r = dual_model.fk_right(q_r)

        traj = planner.plan_synchronized_linear(target_l, target_r, q_l, q_r, duration=0.5)
        assert isinstance(traj, SynchronizedTrajectory)
        assert traj.q_left.shape[1] == 6
        assert traj.q_right.shape[1] == 6

    def test_auto_duration(self, planner, dual_model):
        """With duration=None, auto-duration yields a positive trajectory."""
        q_l = Q_HOME_LEFT.copy()
        q_r = Q_HOME_RIGHT.copy()
        left_ee = dual_model.fk_left(q_l)
        right_ee = dual_model.fk_right(q_r)
        target_l = pin.SE3(left_ee.rotation, left_ee.translation + np.array([0.05, 0.0, 0.0]))
        target_r = pin.SE3(right_ee.rotation, right_ee.translation + np.array([-0.05, 0.0, 0.0]))

        traj = planner.plan_synchronized_linear(target_l, target_r, q_l, q_r)
        assert traj.duration > 0.0
        assert traj.n_steps >= 2


# ---------------------------------------------------------------------------
# plan_master_slave
# ---------------------------------------------------------------------------

class TestPlanMasterSlave:
    """Tests for master-slave coordination planning."""

    def test_master_left_endpoint_error_below_1mm(self, planner, dual_model, object_frame):
        """With master='left', the left EE reaches the target within 1mm."""
        q_l = Q_HOME_LEFT.copy()
        q_r = Q_HOME_RIGHT.copy()
        master_ee = dual_model.fk_left(q_l)
        target = pin.SE3(master_ee.rotation, master_ee.translation + np.array([0.04, 0.0, 0.0]))

        traj = planner.plan_master_slave(
            [target], object_frame, q_l, q_r, master="left", duration=1.0,
        )

        err = np.linalg.norm(dual_model.fk_left_pos(traj.q_left[-1]) - target.translation)
        assert err < 1e-3, f"Master (left) endpoint error {err * 1000:.3f} mm"

    def test_master_right_endpoint_error_below_1mm(self, planner, dual_model, object_frame):
        """With master='right', the right EE reaches the target within 1mm."""
        q_l = Q_HOME_LEFT.copy()
        q_r = Q_HOME_RIGHT.copy()
        master_ee = dual_model.fk_right(q_r)
        target = pin.SE3(master_ee.rotation, master_ee.translation + np.array([-0.04, 0.0, 0.0]))

        traj = planner.plan_master_slave(
            [target], object_frame, q_l, q_r, master="right", duration=1.0,
        )

        err = np.linalg.norm(dual_model.fk_right_pos(traj.q_right[-1]) - target.translation)
        assert err < 1e-3, f"Master (right) endpoint error {err * 1000:.3f} mm"

    def test_slave_tracks_master(self, planner, dual_model, object_frame):
        """Slave arm endpoint is within 2mm of its derived target (from object frame)."""
        q_l = Q_HOME_LEFT.copy()
        q_r = Q_HOME_RIGHT.copy()
        master_ee = dual_model.fk_left(q_l)
        target = pin.SE3(master_ee.rotation, master_ee.translation + np.array([0.04, 0.0, 0.0]))

        traj = planner.plan_master_slave(
            [target], object_frame, q_l, q_r, master="left", duration=1.0,
        )

        # Reconstruct expected slave target from the master's final pose
        master_final = dual_model.fk_left(traj.q_left[-1])
        obj_new = master_final * object_frame.grasp_offset_left.inverse()
        slave_expected = obj_new * object_frame.grasp_offset_right

        slave_actual = dual_model.fk_right_pos(traj.q_right[-1])
        err = np.linalg.norm(slave_actual - slave_expected.translation)
        assert err < 2e-3, f"Slave tracking error {err * 1000:.3f} mm"

    def test_invalid_master_raises_valueerror(self, planner, dual_model, object_frame):
        """master='center' raises ValueError."""
        with pytest.raises(ValueError, match="master must be"):
            planner.plan_master_slave(
                [pin.SE3.Identity()], object_frame,
                Q_HOME_LEFT, Q_HOME_RIGHT, master="center",
            )

    def test_output_is_synchronized_trajectory(self, planner, dual_model, object_frame):
        """Return type is SynchronizedTrajectory."""
        q_l = Q_HOME_LEFT.copy()
        q_r = Q_HOME_RIGHT.copy()
        master_ee = dual_model.fk_left(q_l)

        traj = planner.plan_master_slave(
            [master_ee], object_frame, q_l, q_r, master="left", duration=0.5,
        )
        assert isinstance(traj, SynchronizedTrajectory)

    def test_multiple_waypoints(self, planner, dual_model, object_frame):
        """Planning with multiple master waypoints produces a valid trajectory."""
        q_l = Q_HOME_LEFT.copy()
        q_r = Q_HOME_RIGHT.copy()
        ee0 = dual_model.fk_left(q_l)

        wp1 = pin.SE3(ee0.rotation, ee0.translation + np.array([0.02, 0.0, 0.0]))
        wp2 = pin.SE3(ee0.rotation, ee0.translation + np.array([0.04, 0.0, 0.0]))

        traj = planner.plan_master_slave(
            [wp1, wp2], object_frame, q_l, q_r, master="left", duration=2.0,
        )
        assert traj.n_steps >= 2
        assert traj.duration > 0.0


# ---------------------------------------------------------------------------
# plan_symmetric
# ---------------------------------------------------------------------------

class TestPlanSymmetric:
    """Tests for symmetric (object-centric) coordination planning."""

    def test_both_arms_reach_targets(self, planner, dual_model, object_frame):
        """Both arms reach within 1mm of their derived targets after symmetric motion."""
        q_l = Q_HOME_LEFT.copy()
        q_r = Q_HOME_RIGHT.copy()

        # Move the object slightly along X
        delta = np.array([0.03, 0.0, 0.0])
        new_obj_pose = pin.SE3(np.eye(3), object_frame.pose.translation + delta)

        traj = planner.plan_symmetric(
            [new_obj_pose], object_frame, q_l, q_r, duration=1.0,
        )

        # Expected targets
        expected_left = (new_obj_pose * object_frame.grasp_offset_left).translation
        expected_right = (new_obj_pose * object_frame.grasp_offset_right).translation

        actual_left = dual_model.fk_left_pos(traj.q_left[-1])
        actual_right = dual_model.fk_right_pos(traj.q_right[-1])

        err_l = np.linalg.norm(actual_left - expected_left)
        err_r = np.linalg.norm(actual_right - expected_right)
        assert err_l < 1e-3, f"Left EE error {err_l * 1000:.3f} mm"
        assert err_r < 1e-3, f"Right EE error {err_r * 1000:.3f} mm"

    def test_output_is_synchronized_trajectory(self, planner, dual_model, object_frame):
        """Return type is SynchronizedTrajectory."""
        new_pose = pin.SE3(np.eye(3), object_frame.pose.translation + np.array([0.02, 0.0, 0.0]))
        traj = planner.plan_symmetric(
            [new_pose], object_frame, Q_HOME_LEFT, Q_HOME_RIGHT, duration=0.5,
        )
        assert isinstance(traj, SynchronizedTrajectory)

    def test_empty_waypoints_returns_hold(self, planner, dual_model, object_frame):
        """With empty object waypoints, the trajectory holds the initial configuration."""
        traj = planner.plan_symmetric(
            [], object_frame, Q_HOME_LEFT, Q_HOME_RIGHT, duration=0.1,
        )
        assert isinstance(traj, SynchronizedTrajectory)
        assert traj.n_steps >= 2
        # Should hold at the initial config
        assert np.allclose(traj.q_left[0], Q_HOME_LEFT, atol=1e-10)
        assert np.allclose(traj.q_left[-1], Q_HOME_LEFT, atol=1e-10)

    def test_multiple_object_waypoints(self, planner, dual_model, object_frame):
        """Planning with multiple object waypoints produces a valid trajectory."""
        base = object_frame.pose.translation.copy()
        wp1 = pin.SE3(np.eye(3), base + np.array([0.02, 0.0, 0.0]))
        wp2 = pin.SE3(np.eye(3), base + np.array([0.04, 0.0, 0.0]))

        traj = planner.plan_symmetric(
            [wp1, wp2], object_frame, Q_HOME_LEFT, Q_HOME_RIGHT, duration=2.0,
        )
        assert traj.n_steps >= 2
        assert traj.duration > 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_zero_displacement_joint_linear(self, planner):
        """Zero displacement: trajectory holds at the start config."""
        traj = planner.plan_joint_linear(
            Q_HOME_LEFT, Q_HOME_LEFT,
            Q_HOME_RIGHT, Q_HOME_RIGHT,
            duration=0.5,
        )
        assert np.allclose(traj.q_left[0], Q_HOME_LEFT, atol=1e-10)
        assert np.allclose(traj.q_left[-1], Q_HOME_LEFT, atol=1e-10)
        assert np.allclose(traj.q_right[0], Q_HOME_RIGHT, atol=1e-10)
        assert np.allclose(traj.q_right[-1], Q_HOME_RIGHT, atol=1e-10)

    def test_zero_displacement_auto_duration(self, planner):
        """Zero displacement with auto-duration produces min duration (>=0.1s)."""
        traj = planner.plan_joint_linear(
            Q_HOME_LEFT, Q_HOME_LEFT,
            Q_HOME_RIGHT, Q_HOME_RIGHT,
        )
        assert traj.duration >= 0.1

    def test_zero_displacement_synchronized_linear(self, planner, dual_model):
        """Task-space planning to the current pose should hold position."""
        q_l = Q_HOME_LEFT.copy()
        q_r = Q_HOME_RIGHT.copy()
        target_l = dual_model.fk_left(q_l)
        target_r = dual_model.fk_right(q_r)

        traj = planner.plan_synchronized_linear(
            target_l, target_r, q_l, q_r, duration=0.5,
        )

        # Positions should stay near home throughout
        err_l = np.max(np.abs(traj.q_left - q_l[None, :]))
        err_r = np.max(np.abs(traj.q_right - q_r[None, :]))
        assert err_l < 0.01, f"Left arm drifted: max deviation {err_l:.4f} rad"
        assert err_r < 0.01, f"Right arm drifted: max deviation {err_r:.4f} rad"

    def test_very_short_duration(self, planner):
        """Very short duration still produces at least 2 steps."""
        traj = planner.plan_joint_linear(
            Q_HOME_LEFT, Q_HOME_LEFT + 0.01,
            Q_HOME_RIGHT, Q_HOME_RIGHT + 0.01,
            duration=0.001,
        )
        assert traj.n_steps >= 2


# ---------------------------------------------------------------------------
# _auto_duration
# ---------------------------------------------------------------------------

class TestAutoDuration:
    """Tests for the _auto_duration static method."""

    def test_returns_positive(self):
        """Auto-duration returns a positive value for non-zero displacement."""
        dur = CoordinatedPlanner._auto_duration(
            np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]),
        )
        assert dur > 0.0

    def test_minimum_duration_is_0_1(self):
        """Auto-duration returns at least 0.1 s even for zero displacement."""
        dur = CoordinatedPlanner._auto_duration(np.zeros(6), np.zeros(6))
        assert np.isclose(dur, 0.1)

    def test_scales_with_displacement(self):
        """Larger displacement yields longer duration."""
        small = CoordinatedPlanner._auto_duration(
            np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.zeros(6),
        )
        large = CoordinatedPlanner._auto_duration(
            np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.zeros(6),
        )
        assert large > small

    def test_respects_velocity_limits(self):
        """Duration ensures that joint velocity stays below VEL_LIMITS * _VEL_SAFETY."""
        disp = np.array([3.0, 2.0, 1.0, 0.5, 0.5, 0.5])
        dur = CoordinatedPlanner._auto_duration(disp, np.zeros(6))
        # Check that disp / dur <= VEL_LIMITS * _VEL_SAFETY for all joints
        peak_vel = np.abs(disp) / dur
        assert np.all(peak_vel <= VEL_LIMITS * _VEL_SAFETY + 1e-9), (
            f"Peak vel {peak_vel} exceeds limits {VEL_LIMITS * _VEL_SAFETY}"
        )

    def test_considers_both_arms(self):
        """Duration is driven by the arm with the larger displacement."""
        # Left arm: small displacement, right arm: large displacement
        dur = CoordinatedPlanner._auto_duration(
            np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        # Duration should be driven by the right arm's large displacement
        expected_min = 3.0 / (VEL_LIMITS[0] * _VEL_SAFETY)
        assert dur >= expected_min - 1e-9
