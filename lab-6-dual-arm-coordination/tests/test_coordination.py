"""Lab 6 — Phase 2 tests: trajectory synchronisation and bimanual controller."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lab6_common import (
    DT,
    GRIPPER_OPEN,
    LEFT_CTRL,
    LEFT_GRIP_CTRL,
    LEFT_QPOS,
    NUM_JOINTS,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    RIGHT_CTRL,
    RIGHT_GRIP_CTRL,
    RIGHT_QPOS,
    load_both_pinocchio_models,
    load_mujoco_model,
    LEFT_GRIP_Y_OFFSET,
    RIGHT_GRIP_Y_OFFSET,
)
from coordination_layer import ObjectFrame, sync_trajectories
from bimanual_controller import BimanualController


# ---------------------------------------------------------------------------
# sync_trajectories
# ---------------------------------------------------------------------------

class TestSyncTrajectories:
    def _make_traj(self, duration: float, n: int) -> tuple:
        times = np.linspace(0.0, duration, n)
        q  = np.random.randn(n, NUM_JOINTS)
        qd = np.random.randn(n, NUM_JOINTS)
        return times, q, qd

    def test_same_length_unchanged(self):
        """If both trajectories have the same duration, output is identical length."""
        np.random.seed(0)
        t_L, q_L, qd_L = self._make_traj(2.0, 200)
        t_R, q_R, qd_R = self._make_traj(2.0, 200)
        t_s, q_Ls, qd_Ls, q_Rs, qd_Rs = sync_trajectories(
            (t_L, q_L, qd_L), (t_R, q_R, qd_R))
        assert len(t_s) == len(q_Ls) == len(q_Rs)
        np.testing.assert_allclose(t_s[-1], 2.0, atol=1e-6)

    def test_longer_duration_wins(self):
        """Output duration = max of the two input durations."""
        np.random.seed(1)
        t_L, q_L, qd_L = self._make_traj(3.0, 300)
        t_R, q_R, qd_R = self._make_traj(1.5, 150)
        t_s, _, _, _, _ = sync_trajectories(
            (t_L, q_L, qd_L), (t_R, q_R, qd_R))
        assert t_s[-1] >= 2.9  # at least as long as the longer trajectory

    def test_short_traj_held_at_end(self):
        """Short trajectory joints must be held at final config after its end."""
        np.random.seed(2)
        t_L, q_L, qd_L = self._make_traj(2.0, 200)
        t_R, q_R, qd_R = self._make_traj(1.0, 100)
        _, _, _, q_Rs, qd_Rs = sync_trajectories(
            (t_L, q_L, qd_L), (t_R, q_R, qd_R))
        # After t=1.0, right arm should hold final config
        final_R = q_R[-1]
        # Last element of synced right should equal original last
        np.testing.assert_allclose(q_Rs[-1], final_R, atol=1e-10)

    def test_zero_velocity_at_hold(self):
        """Velocity must be zero in the held portion of the short trajectory."""
        np.random.seed(3)
        t_L, q_L, qd_L = self._make_traj(2.0, 200)
        t_R, q_R, qd_R = self._make_traj(0.5, 50)
        t_s, _, _, _, qd_Rs = sync_trajectories(
            (t_L, q_L, qd_L), (t_R, q_R, qd_R))
        held_mask = t_s > 0.5
        np.testing.assert_allclose(qd_Rs[held_mask], 0.0, atol=1e-10)

    def test_output_shape(self):
        """All output arrays must have consistent shapes."""
        np.random.seed(4)
        t_L, q_L, qd_L = self._make_traj(1.5, 150)
        t_R, q_R, qd_R = self._make_traj(2.5, 250)
        t_s, q_Ls, qd_Ls, q_Rs, qd_Rs = sync_trajectories(
            (t_L, q_L, qd_L), (t_R, q_R, qd_R))
        N = len(t_s)
        assert q_Ls.shape  == (N, NUM_JOINTS)
        assert qd_Ls.shape == (N, NUM_JOINTS)
        assert q_Rs.shape  == (N, NUM_JOINTS)
        assert qd_Rs.shape == (N, NUM_JOINTS)


# ---------------------------------------------------------------------------
# ObjectFrame
# ---------------------------------------------------------------------------

class TestObjectFrame:
    def test_ee_targets_from_bar_pos(self):
        """EE targets must be at the correct world-frame positions."""
        from lab6_common import (BAR_PICK_POS, GRIPPER_TIP_OFFSET)
        frame = ObjectFrame.from_bar_pos(BAR_PICK_POS)
        ee_L = frame.ee_target_left()
        ee_R = frame.ee_target_right()

        expected_L = BAR_PICK_POS + np.array([0.0, LEFT_GRIP_Y_OFFSET,  GRIPPER_TIP_OFFSET])
        expected_R = BAR_PICK_POS + np.array([0.0, RIGHT_GRIP_Y_OFFSET, GRIPPER_TIP_OFFSET])

        np.testing.assert_allclose(ee_L, expected_L, atol=1e-10)
        np.testing.assert_allclose(ee_R, expected_R, atol=1e-10)

    def test_targets_update_with_position(self):
        """Moving the bar centre must shift both EE targets equally."""
        frame = ObjectFrame.from_bar_pos(np.array([0.45, 0.0, 0.335]))
        ee_L_0 = frame.ee_target_left().copy()

        frame.position = np.array([0.45, 0.20, 0.335])
        ee_L_1 = frame.ee_target_left()
        np.testing.assert_allclose(ee_L_1 - ee_L_0, np.array([0.0, 0.20, 0.0]), atol=1e-10)


# ---------------------------------------------------------------------------
# BimanualController — gravity hold
# ---------------------------------------------------------------------------

class TestBimanualController:
    def setup_method(self):
        import mujoco
        self.mj_model, self.mj_data = load_mujoco_model()
        (self.pin_L, self.data_L, self.ee_L), (self.pin_R, self.data_R, self.ee_R) = \
            load_both_pinocchio_models()
        self.ctrl = BimanualController(
            self.pin_L, self.data_L, self.ee_L,
            self.pin_R, self.data_R, self.ee_R,
        )
        # Initialise at Q_HOME
        self.mj_data.qpos[LEFT_QPOS]  = Q_HOME_LEFT
        self.mj_data.qpos[RIGHT_QPOS] = Q_HOME_RIGHT
        self.mj_data.ctrl[LEFT_GRIP_CTRL]  = GRIPPER_OPEN
        self.mj_data.ctrl[RIGHT_GRIP_CTRL] = GRIPPER_OPEN
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def test_joint_torques_nonzero_at_home(self):
        """With gravity, impedance torques at Q_HOME must be non-zero."""
        import mujoco
        self.ctrl.apply_joint_impedance(
            self.mj_data,
            Q_HOME_LEFT,  np.zeros(NUM_JOINTS),
            Q_HOME_RIGHT, np.zeros(NUM_JOINTS),
        )
        assert np.any(np.abs(self.mj_data.ctrl[LEFT_CTRL])  > 0.1), \
            "Left arm torques unexpectedly near zero at Q_HOME"
        assert np.any(np.abs(self.mj_data.ctrl[RIGHT_CTRL]) > 0.1), \
            "Right arm torques unexpectedly near zero at Q_HOME"

    def test_arms_hold_position(self):
        """Both arms must stay within 5 mm of Q_HOME after 1 second of holding."""
        import mujoco
        from lab6_common import get_ee_pose
        n_steps = int(1.0 / DT)
        for _ in range(n_steps):
            self.ctrl.apply_gravity_hold(
                self.mj_data, Q_HOME_LEFT, Q_HOME_RIGHT)
            mujoco.mj_step(self.mj_model, self.mj_data)

        # Joint error
        q_err_L = np.linalg.norm(
            self.mj_data.qpos[LEFT_QPOS] - Q_HOME_LEFT)
        q_err_R = np.linalg.norm(
            self.mj_data.qpos[RIGHT_QPOS] - Q_HOME_RIGHT)
        assert q_err_L < 0.05, f"Left arm drifted {np.degrees(q_err_L):.1f}° from Q_HOME"
        assert q_err_R < 0.05, f"Right arm drifted {np.degrees(q_err_R):.1f}° from Q_HOME"

    def test_grippers_unchanged_by_joint_ctrl(self):
        """apply_joint_impedance must not touch gripper ctrl slots."""
        import mujoco
        self.mj_data.ctrl[LEFT_GRIP_CTRL]  = GRIPPER_OPEN
        self.mj_data.ctrl[RIGHT_GRIP_CTRL] = GRIPPER_OPEN
        self.ctrl.apply_joint_impedance(
            self.mj_data,
            Q_HOME_LEFT,  np.zeros(NUM_JOINTS),
            Q_HOME_RIGHT, np.zeros(NUM_JOINTS),
        )
        assert self.mj_data.ctrl[LEFT_GRIP_CTRL]  == GRIPPER_OPEN
        assert self.mj_data.ctrl[RIGHT_GRIP_CTRL] == GRIPPER_OPEN
