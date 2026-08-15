"""Lab 8 — M1 tests: task Jacobians, drift terms, and the inverse-dynamics QP.

Every Jacobian is checked against finite differences (CLAUDE.md Pinocchio
Rules), because a sign or frame error there is silent: the QP will happily
solve the wrong problem and the robot simply falls over for no visible reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from g1_torque_model import torque_limits  # noqa: E402
from lab8_common import (  # noqa: E402
    DT,
    NU,
    NV,
    PELVIS_MJCF_Z,
    Q_STAND_JOINTS,
    TOTAL_MASS,
    load_g1_pinocchio,
    load_g1_torque_mujoco,
    mj_state_to_pin,
    pin_point_to_world,
    world_point_to_pin,
)
from wb_id_qp import ContactSpec, WholeBodyIDQP  # noqa: E402
from wb_tasks import (  # noqa: E402
    CoMTask,
    FramePositionTask,
    FramePoseTask,
    PostureTask,
    TaskStack,
)

HAND = "right_wrist_yaw_link"
LEFT_FOOT = "left_ankle_roll_link"
RIGHT_FOOT = "right_ankle_roll_link"
FD_EPS = 1e-6
FD_TOL = 1e-5


@pytest.fixture(scope="module")
def setup():
    """Torque model + Pinocchio model at the standing pose."""
    mj_model, mj_data = load_g1_torque_mujoco(timestep=DT)
    pin_model, pin_data = load_g1_pinocchio()
    q, v = mj_state_to_pin(mj_data)
    return mj_model, mj_data, pin_model, pin_data, q, v


def _finite_difference_jacobian(pin_model, pin_data, q, value_fn, rows: int):
    """dvalue/dq via `pin.integrate` perturbations (never `q += dq`)."""
    def evaluate(q_test):
        pin.forwardKinematics(pin_model, pin_data, q_test)
        pin.updateFramePlacements(pin_model, pin_data)
        pin.centerOfMass(pin_model, pin_data, q_test)
        return np.asarray(value_fn(q_test), dtype=float).copy()

    base = evaluate(q)
    jacobian = np.zeros((rows, NV))
    for i in range(NV):
        delta = np.zeros(NV)
        delta[i] = FD_EPS
        jacobian[:, i] = (evaluate(pin.integrate(pin_model, q, delta)) - base) / FD_EPS
    return jacobian


class TestTaskJacobians:
    """Analytic Jacobians must match finite differences."""

    def test_frame_position_task(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        task = FramePositionTask(HAND, pin_model)
        stack = TaskStack(pin_model, pin_data, [task])
        stack.update(q)
        analytic = task.jacobian(pin_model, pin_data, q)

        numeric = _finite_difference_jacobian(
            pin_model, pin_data, q, lambda _: task.current_position(pin_data), 3
        )
        assert np.abs(analytic - numeric).max() < FD_TOL

    def test_frame_pose_task_translation(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        task = FramePoseTask(LEFT_FOOT, pin_model)
        stack = TaskStack(pin_model, pin_data, [task])
        stack.update(q)
        analytic = task.jacobian(pin_model, pin_data, q)[:3, :]

        numeric = _finite_difference_jacobian(
            pin_model, pin_data, q, lambda _: task.current_position(pin_data), 3
        )
        assert np.abs(analytic - numeric).max() < FD_TOL

    def test_com_task(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        task = CoMTask(pin_model, axes=(0, 1, 2))
        stack = TaskStack(pin_model, pin_data, [task])
        stack.update(q)
        analytic = task.jacobian(pin_model, pin_data, q)

        numeric = _finite_difference_jacobian(
            pin_model, pin_data, q, lambda _: task.current_com(pin_data), 3
        )
        assert np.abs(analytic - numeric).max() < FD_TOL

    def test_posture_task(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        task = PostureTask(Q_STAND_JOINTS)
        analytic = task.jacobian(pin_model, pin_data, q)

        numeric = _finite_difference_jacobian(
            pin_model, pin_data, q, lambda q_test: np.asarray(q_test)[7:], NU
        )
        assert np.abs(analytic - numeric).max() < FD_TOL

    def test_com_axis_selection(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        planar = CoMTask(pin_model, axes=(0, 1))
        full = CoMTask(pin_model, axes=(0, 1, 2))
        TaskStack(pin_model, pin_data, [full]).update(q)
        assert planar.dimension() == 2
        assert np.allclose(
            planar.jacobian(pin_model, pin_data, q),
            full.jacobian(pin_model, pin_data, q)[:2, :],
        )


class TestTaskSemantics:
    """Targets, errors and feedforward behave as documented."""

    def test_targets_are_world_frame(self, setup):
        mj_model, mj_data, pin_model, pin_data, q, _ = setup
        task = FramePositionTask(HAND, pin_model)
        TaskStack(pin_model, pin_data, [task]).update(q)
        # Frame position must agree with MuJoCo's body position, i.e. the
        # 0.793 m pelvis offset is handled inside the task.
        import mujoco

        body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, HAND)
        assert np.allclose(task.current_position(pin_data), mj_data.xpos[body], atol=1e-9)

    def test_point_conversion_roundtrip(self):
        point = np.array([0.3, -0.2, 0.9])
        assert np.allclose(world_point_to_pin(pin_point_to_world(point)), point)
        assert pin_point_to_world(point)[2] == pytest.approx(point[2] + PELVIS_MJCF_Z)

    def test_zero_error_at_own_position(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        task = FramePositionTask(HAND, pin_model)
        TaskStack(pin_model, pin_data, [task]).update(q)
        task.set_target(task.current_position(pin_data))
        assert np.linalg.norm(task.error(pin_model, pin_data, q)) < 1e-12

    def test_capture_current_zeroes_pose_error(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        task = FramePoseTask(LEFT_FOOT, pin_model)
        TaskStack(pin_model, pin_data, [task]).update(q)
        task.capture_current(pin_data)
        assert np.linalg.norm(task.error(pin_model, pin_data, q)) < 1e-12

    def test_feedforward_enters_desired_acceleration(self, setup):
        _, _, pin_model, pin_data, q, v = setup
        task = FramePositionTask(HAND, pin_model, gain=100.0)
        TaskStack(pin_model, pin_data, [task]).update(q)
        task.set_target(task.current_position(pin_data))
        without = task.desired_acceleration(pin_model, pin_data, q, v)

        accel = np.array([0.0, 0.0, 2.5])
        task.set_target(task.current_position(pin_data), acceleration=accel)
        with_ff = task.desired_acceleration(pin_model, pin_data, q, v)
        assert np.allclose(with_ff - without, accel)

    def test_drift_is_zero_at_rest(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        stack = TaskStack(pin_model, pin_data, [FramePositionTask(HAND, pin_model)])
        stack.update_dynamics(q, np.zeros(NV))
        assert np.allclose(stack.tasks[0].drift(pin_model, pin_data), 0.0, atol=1e-9)

    def test_drift_nonzero_when_moving(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        stack = TaskStack(pin_model, pin_data, [FramePositionTask(HAND, pin_model)])
        v = np.zeros(NV)
        v[6:] = 1.0  # every joint spinning → centripetal drift at the hand
        stack.update_dynamics(q, v)
        assert np.linalg.norm(stack.tasks[0].drift(pin_model, pin_data)) > 1e-3

    def test_rejects_unknown_frame(self, setup):
        _, _, pin_model, _, _, _ = setup
        with pytest.raises(ValueError):
            FramePositionTask("no_such_frame", pin_model)

    def test_posture_shape_validation(self):
        with pytest.raises(ValueError):
            PostureTask(np.zeros(5))


class TestTaskStack:
    """Assembly and bookkeeping."""

    def test_assemble_dimensions_and_weights(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        stack = TaskStack(pin_model, pin_data)
        stack.add(CoMTask(pin_model, weight=1e4))
        stack.add(FramePositionTask(HAND, pin_model, weight=1e3))
        stack.add(PostureTask(Q_STAND_JOINTS, weight=1.0))
        stack.update(q)

        jacobian, xdot, weights = stack.assemble(q)
        expected_rows = 2 + 3 + NU
        assert jacobian.shape == (expected_rows, NV)
        assert xdot.shape == (expected_rows,)
        assert weights[0] == 1e4 and weights[2] == 1e3 and weights[-1] == 1.0

    def test_disabled_task_is_excluded(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        stack = TaskStack(pin_model, pin_data)
        com = stack.add(CoMTask(pin_model))
        hand = stack.add(FramePositionTask(HAND, pin_model))
        stack.update(q)
        assert len(stack.active) == 2
        hand.enabled = False
        assert stack.active == [com]
        assert stack.assemble(q)[0].shape[0] == com.dimension()

    def test_empty_stack_is_safe(self, setup):
        _, _, pin_model, pin_data, q, _ = setup
        jacobian, xdot, weights = TaskStack(pin_model, pin_data).assemble(q)
        assert jacobian.shape == (0, NV) and xdot.size == 0 and weights.size == 0


class TestInverseDynamicsQP:
    """The QP must produce physically consistent torques and forces."""

    @staticmethod
    def _build(setup):
        mj_model, _, pin_model, pin_data, q, v = setup
        stack = TaskStack(pin_model, pin_data)
        com = stack.add(CoMTask(pin_model, weight=1e4, gain=100.0))
        hand = stack.add(FramePositionTask(HAND, pin_model, weight=1e3, gain=400.0))
        stack.add(PostureTask(Q_STAND_JOINTS, weight=1.0, gain=50.0))
        stack.update_dynamics(q, v)
        com.set_target(com.current_com(pin_data))
        hand.set_target(hand.current_position(pin_data))
        qp = WholeBodyIDQP(
            pin_model, pin_data,
            [ContactSpec(LEFT_FOOT), ContactSpec(RIGHT_FOOT)],
            torque_limits(mj_model),
        )
        return stack, qp, q, v

    def test_solves_at_standing_pose(self, setup):
        stack, qp, q, v = self._build(setup)
        result = qp.solve(stack, q, v)
        assert "solved" in result.status
        assert result.tau.shape == (NU,)
        assert result.qddot.shape == (NV,)
        assert result.forces.shape == (12,)

    def test_contact_forces_support_body_weight(self, setup):
        """At rest the vertical contact forces must carry m·g."""
        stack, qp, q, v = self._build(setup)
        result = qp.solve(stack, q, v)
        total_normal = result.forces[2] + result.forces[8]
        assert total_normal == pytest.approx(TOTAL_MASS * 9.81, rel=0.05)

    def test_friction_cone_respected(self, setup):
        stack, qp, q, v = self._build(setup)
        result = qp.solve(stack, q, v)
        for start in (0, 6):
            fx, fy, fz = result.forces[start:start + 3]
            assert fz >= 1.0 - 1e-6          # unilateral + minimum load
            assert abs(fx) <= 0.6 * fz + 1e-6
            assert abs(fy) <= 0.6 * fz + 1e-6

    def test_centre_of_pressure_inside_foot(self, setup):
        stack, qp, q, v = self._build(setup)
        result = qp.solve(stack, q, v)
        spec = ContactSpec(LEFT_FOOT)
        for start in (0, 6):
            fz = result.forces[start + 2]
            mx, my = result.forces[start + 3], result.forces[start + 4]
            assert abs(mx) <= spec.half_width * fz + 1e-6
            assert abs(my) <= spec.half_length * fz + 1e-6

    def test_torques_within_actuator_limits(self, setup):
        mj_model = setup[0]
        stack, qp, q, v = self._build(setup)
        result = qp.solve(stack, q, v)
        limits = torque_limits(mj_model)
        assert np.all(result.tau >= limits[:, 0] - 1e-6)
        assert np.all(result.tau <= limits[:, 1] + 1e-6)

    def test_stance_feet_acceleration_is_cancelled(self, setup):
        """The contact constraint J_c q̈ + J̇_c q̇ = 0 must actually hold."""
        _, _, pin_model, pin_data, _, _ = setup
        stack, qp, q, v = self._build(setup)
        result = qp.solve(stack, q, v)
        contact_jac, drift = qp._contact_jacobian(q, v)
        residual = contact_jac @ result.qddot + drift
        assert np.abs(residual).max() < 1e-4

    def test_repeated_solves_are_stable(self, setup):
        """The hot-update path must keep returning the same solution.

        Tolerance is 10 mNm, not machine epsilon: OSQP is an ADMM solver that
        stops at `eps_abs = 1e-6` on its own residuals, and warm-starting the
        second call (25 iterations instead of 75) lands on a slightly
        different point inside that tolerance. On actuators rated 50–139 N·m
        a few mN·m of solver noise is irrelevant — demanding bitwise
        repeatability here would be testing the solver's internals, not the
        controller.
        """
        stack, qp, q, v = self._build(setup)
        first = qp.solve(stack, q, v)
        for _ in range(5):
            again = qp.solve(stack, q, v)
        assert np.allclose(first.tau, again.tau, atol=1e-2)
