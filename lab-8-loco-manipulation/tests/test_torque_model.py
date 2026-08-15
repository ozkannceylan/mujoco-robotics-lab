"""Lab 8 — M0 tests: torque model integrity and Pinocchio/MuJoCo parity.

Fast checks only (no long simulations); the 10 s standing behaviour is the
gate demo's job, not pytest's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from g1_torque_model import compile_g1_torque_model, torque_limits  # noqa: E402
from lab8_common import (  # noqa: E402
    DT,
    G1_MJCF_PATH,
    NQ,
    NU,
    NV,
    PELVIS_MJCF_Z,
    Q_STAND_JOINTS,
    TOTAL_MASS,
    clip_torques,
    com_position,
    dense_mass_matrix,
    foot_contact_state,
    joint_torques_to_ctrl,
    load_g1_pinocchio,
    load_g1_torque_mujoco,
    mj_state_to_pin,
)


@pytest.fixture(scope="module")
def models():
    """Torque-actuated MuJoCo model + matching Pinocchio model."""
    mj_model, mj_data = load_g1_torque_mujoco(timestep=DT)
    pin_model, pin_data = load_g1_pinocchio()
    return mj_model, mj_data, pin_model, pin_data


class TestTorqueModel:
    """The MJCF conversion must yield real torque actuators, not servos."""

    def test_dimensions(self, models):
        mj_model, _, _, _ = models
        assert (mj_model.nq, mj_model.nv, mj_model.nu) == (NQ, NV, NU)

    def test_actuators_are_motors(self, models):
        """gaintype=fixed, biastype=none, gainprm[0]=1 → force == ctrl."""
        mj_model, _, _, _ = models
        for i in range(mj_model.nu):
            assert mj_model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
            assert mj_model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
            assert mj_model.actuator_gainprm[i][0] == pytest.approx(1.0)

    def test_torque_limits_match_joint_spec(self, models):
        """ctrlrange must come from each joint's actuatorfrcrange."""
        mj_model, _, _, _ = models
        limits = torque_limits(mj_model)
        assert limits.shape == (NU, 2)
        assert np.all(limits[:, 0] < 0) and np.all(limits[:, 1] > 0)
        # Unitree G1 spec: knees/hip-roll 139, hips/yaw 88, ankles 50.
        knee = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_joint")
        ankle = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_ankle_pitch_joint"
        )
        assert limits[knee, 1] == pytest.approx(139.0)
        assert limits[ankle, 1] == pytest.approx(50.0)
        assert bool(mj_model.actuator_ctrllimited[knee])

    def test_ctrl_maps_to_joint_force(self, models):
        """A unit ctrl command must appear as a unit generalized force."""
        mj_model, _, _, _ = models
        data = mujoco.MjData(mj_model)
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        knee = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_joint")
        data.ctrl[knee] = 12.5
        mujoco.mj_forward(mj_model, data)
        assert data.actuator_force[knee] == pytest.approx(12.5, abs=1e-9)

    def test_keyframe_ctrl_is_zeroed(self, models):
        """Menagerie's keyframe ctrl holds position targets — invalid as torques."""
        mj_model, _, _, _ = models
        assert mj_model.nkey > 0
        assert np.abs(mj_model.key_ctrl[0]).max() == 0.0
        # …while the pose itself is preserved.
        assert mj_model.key_qpos[0][2] == pytest.approx(0.79, abs=1e-6)

    def test_floor_exists(self, models):
        """g1.xml has no ground plane; the builder must add one."""
        mj_model, _, _, _ = models
        assert mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0

    def test_clip_torques_respects_limits(self, models):
        mj_model, _, _, _ = models
        limits = torque_limits(mj_model)
        clipped = clip_torques(np.full(NU, 1e6), mj_model)
        assert np.allclose(clipped, limits[:, 1])
        clipped = clip_torques(np.full(NU, -1e6), mj_model)
        assert np.allclose(clipped, limits[:, 0])

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compile_g1_torque_model(tmp_path / "nope.xml")


class TestPinocchioParity:
    """The analytical model must describe the simulated body (Lab 5 L-6.1c)."""

    @staticmethod
    def _random_state(mj_model, mj_data, rng, airborne=True):
        mj_data.qpos[7:] = rng.uniform(-0.4, 0.4, NU)
        if airborne:
            mj_data.qpos[2] = 1.2
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

    def test_dimensions_match(self, models):
        mj_model, _, pin_model, _ = models
        assert (pin_model.nq, pin_model.nv) == (mj_model.nq, mj_model.nv)

    def test_total_mass(self, models):
        mj_model, _, pin_model, _ = models
        mass_pin = sum(inertia.mass for inertia in pin_model.inertias[1:])
        assert mass_pin == pytest.approx(mj_model.body_subtreemass[0], rel=1e-9)
        assert mass_pin == pytest.approx(TOTAL_MASS, abs=0.01)

    def test_gravity_parity(self, models):
        """g(q) from Pinocchio vs MuJoCo qfrc_bias at qvel = 0."""
        mj_model, mj_data, pin_model, pin_data = models
        rng = np.random.default_rng(1)
        for _ in range(5):
            self._random_state(mj_model, mj_data, rng)
            q, _ = mj_state_to_pin(mj_data)
            pin.computeGeneralizedGravity(pin_model, pin_data, q)
            assert np.allclose(pin_data.g, mj_data.qfrc_bias, atol=1e-9)

    def test_mass_matrix_parity(self, models):
        mj_model, mj_data, pin_model, pin_data = models
        rng = np.random.default_rng(2)
        for _ in range(3):
            self._random_state(mj_model, mj_data, rng)
            q, _ = mj_state_to_pin(mj_data)
            m_pin = pin.crba(pin_model, pin_data, q)
            m_pin = np.triu(m_pin) + np.triu(m_pin, 1).T
            assert np.allclose(m_pin, dense_mass_matrix(mj_model, mj_data), atol=1e-9)

    def test_com_parity_with_pelvis_offset(self, models):
        """Pinocchio CoM is in the FreeFlyer frame: add PELVIS_MJCF_Z for world."""
        mj_model, mj_data, pin_model, pin_data = models
        rng = np.random.default_rng(3)
        for _ in range(3):
            self._random_state(mj_model, mj_data, rng)
            q, _ = mj_state_to_pin(mj_data)
            com_pin = pin.centerOfMass(pin_model, pin_data, q).copy()
            com_pin[2] += PELVIS_MJCF_Z
            assert np.allclose(com_pin, com_position(mj_model, mj_data), atol=1e-9)

    def test_base_velocity_frame_conversion(self, models):
        """MuJoCo base linear velocity is world-frame, Pinocchio's is body-local."""
        mj_model, mj_data, _, _ = models
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        # Yaw the base 90° so world and body frames genuinely differ.
        mj_data.qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
        mj_data.qvel[:6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # +x in WORLD
        mujoco.mj_forward(mj_model, mj_data)
        _, v = mj_state_to_pin(mj_data)
        # A +x world velocity is -y in a frame yawed by +90°.
        assert np.allclose(v[:3], [0.0, -1.0, 0.0], atol=1e-9)

    def test_joint_torque_extraction(self, models):
        _, _, _, _ = models
        full = np.arange(NV, dtype=float)
        assert np.allclose(joint_torques_to_ctrl(full), np.arange(6, NV))
        assert np.allclose(joint_torques_to_ctrl(np.ones(NU)), np.ones(NU))
        with pytest.raises(ValueError):
            joint_torques_to_ctrl(np.zeros(7))


class TestStandingPose:
    """Sanity checks on the keyframe pose the controller holds."""

    def test_keyframe_has_both_feet_in_contact(self, models):
        mj_model, mj_data, _, _ = models
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mujoco.mj_forward(mj_model, mj_data)
        left, right = foot_contact_state(mj_model, mj_data)
        assert left and right

    def test_standing_joint_vector_shape(self):
        assert Q_STAND_JOINTS.shape == (NU,)

    def test_zero_torque_falls(self, models):
        """Torque actuators give nothing for free — the honest baseline.

        Under position servos this pose held itself; that is exactly the
        crutch M0 removes.
        """
        mj_model, _, _, _ = models
        data = mujoco.MjData(mj_model)
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        data.ctrl[:] = 0.0
        for _ in range(2000):
            mujoco.mj_step(mj_model, data)
        assert data.qpos[2] < 0.5
