"""Lab 8 — M2 tests: gait schedule, swing references, ZMP measurement.

Fast, deterministic checks only. The stepping behaviour itself is the gate
demo's job; what is tested here is that the *references* the controller is
handed are self-consistent — a swing trajectory that does not start where the
foot is, or a contact schedule that drops both feet, produces a fall whose
cause is invisible from the simulation alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gait_planner import GaitSchedule, Phase  # noqa: E402
from lab8_common import (  # noqa: E402
    DT,
    load_g1_torque_mujoco,
    measured_zmp,
    point_in_support_polygon,
    support_polygon_margin,
)
from wb_id_qp import ContactSpec, WholeBodyIDQP  # noqa: E402

LEFT = "left_ankle_roll_link"
RIGHT = "right_ankle_roll_link"
LEFT_HOME = np.array([0.0, 0.12, 0.03])
RIGHT_HOME = np.array([0.0, -0.12, 0.03])
COM_HOME = np.array([0.0, 0.0, 0.69])


@pytest.fixture
def schedule() -> GaitSchedule:
    return GaitSchedule(
        LEFT, RIGHT, LEFT_HOME, RIGHT_HOME, COM_HOME,
        n_steps=4, t_initial=1.0, t_double=1.0, t_single=0.5,
        step_length=0.0, step_height=0.02,
    )


class TestTimeline:
    """Phase sequencing and contact sets."""

    def test_phase_order_alternates_feet(self, schedule):
        swings = [
            p.phase for p in schedule.phases()
            if p.phase in (Phase.SINGLE_LEFT, Phase.SINGLE_RIGHT)
        ]
        assert swings == [
            Phase.SINGLE_LEFT, Phase.SINGLE_RIGHT,
            Phase.SINGLE_LEFT, Phase.SINGLE_RIGHT,
        ]

    def test_phases_are_contiguous(self, schedule):
        phases = schedule.phases()
        for previous, following in zip(phases, phases[1:]):
            assert previous.t_end == pytest.approx(following.t_start)
        assert phases[0].t_start == 0.0
        assert phases[-1].t_end == pytest.approx(schedule.total_duration)

    def test_single_support_has_exactly_one_stance_foot(self, schedule):
        for phase in schedule.phases():
            if phase.phase not in (Phase.SINGLE_LEFT, Phase.SINGLE_RIGHT):
                continue
            reference = schedule.reference(0.5 * (phase.t_start + phase.t_end))
            assert len(reference.stance_feet) == 1
            assert reference.swing_foot not in reference.stance_feet

    def test_double_support_has_both_feet(self, schedule):
        reference = schedule.reference(0.5 * schedule.t_initial)
        assert set(reference.stance_feet) == {LEFT, RIGHT}
        assert reference.swing_foot is None

    def test_swing_foot_matches_phase(self, schedule):
        for phase in schedule.phases():
            if phase.phase is Phase.SINGLE_LEFT:
                assert schedule.reference(phase.t_start + 0.1).swing_foot == LEFT
            elif phase.phase is Phase.SINGLE_RIGHT:
                assert schedule.reference(phase.t_start + 0.1).swing_foot == RIGHT

    def test_contact_set_is_never_empty(self, schedule):
        for t in np.arange(0.0, schedule.total_duration, 0.02):
            assert len(schedule.contact_frames(float(t))) >= 1


class TestSwingReference:
    """The swing trajectory must be continuous and land where it started."""

    @staticmethod
    def _first_swing(schedule):
        for index, phase in enumerate(schedule.phases()):
            if phase.phase in (Phase.SINGLE_LEFT, Phase.SINGLE_RIGHT):
                return phase
        raise AssertionError("no swing phase")

    def test_starts_and_ends_on_the_ground(self, schedule):
        phase = self._first_swing(schedule)
        start = schedule.reference(phase.t_start).swing_position
        end = schedule.reference(phase.t_end - 1e-9).swing_position
        assert start[2] == pytest.approx(LEFT_HOME[2], abs=1e-6)
        assert end[2] == pytest.approx(LEFT_HOME[2], abs=1e-6)

    def test_in_place_step_returns_to_the_same_spot(self, schedule):
        phase = self._first_swing(schedule)
        start = schedule.reference(phase.t_start).swing_position
        end = schedule.reference(phase.t_end - 1e-9).swing_position
        assert np.allclose(start[:2], end[:2], atol=1e-9)

    def test_clearance_reached_at_mid_swing(self, schedule):
        phase = self._first_swing(schedule)
        mid = schedule.reference(0.5 * (phase.t_start + phase.t_end)).swing_position
        assert mid[2] == pytest.approx(LEFT_HOME[2] + schedule.step_height, abs=1e-6)

    def test_vertical_velocity_vanishes_at_both_ends(self, schedule):
        """Zero touchdown velocity — otherwise the foot slams into the floor."""
        phase = self._first_swing(schedule)
        assert schedule.reference(phase.t_start).swing_velocity[2] == pytest.approx(
            schedule.step_height * np.pi / phase.duration, rel=1e-6
        )
        end_velocity = schedule.reference(phase.t_end - 1e-9).swing_velocity[2]
        assert end_velocity == pytest.approx(
            -schedule.step_height * np.pi / phase.duration, rel=1e-3
        )

    def test_feedforward_matches_finite_differences(self, schedule):
        """ẋ_ref and ẍ_ref must be the actual derivatives of the reference."""
        phase = self._first_swing(schedule)
        t = 0.5 * (phase.t_start + phase.t_end)
        h = 1e-6
        before = schedule.reference(t - h)
        at = schedule.reference(t)
        after = schedule.reference(t + h)

        numeric_velocity = (after.swing_position - before.swing_position) / (2 * h)
        assert np.allclose(at.swing_velocity, numeric_velocity, atol=1e-4)

        numeric_acceleration = (after.swing_velocity - before.swing_velocity) / (2 * h)
        assert np.allclose(at.swing_acceleration, numeric_acceleration, atol=1e-3)

    def test_forward_stride_advances_the_foot(self):
        """Same module must serve M3's forward walking."""
        walking = GaitSchedule(
            LEFT, RIGHT, LEFT_HOME, RIGHT_HOME, COM_HOME,
            n_steps=2, step_length=0.10,
        )
        phase = self._first_swing(walking)
        start = walking.reference(phase.t_start).swing_position
        end = walking.reference(phase.t_end - 1e-9).swing_position
        assert end[0] - start[0] == pytest.approx(0.10, abs=1e-6)


class TestCoMReference:
    """The weight shift is what makes stepping in place possible."""

    def test_com_moves_toward_the_next_stance_foot(self, schedule):
        """First swing is the left foot, so the CoM must move to the right."""
        start = schedule.reference(0.0).com_target
        end = schedule.reference(schedule.t_initial - 1e-6).com_target
        assert end[1] < start[1]
        assert end[1] == pytest.approx(RIGHT_HOME[1] * schedule.com_shift_ratio, abs=1e-6)

    def test_com_target_is_continuous(self, schedule):
        """A step change in the CoM target would jolt the whole-body QP."""
        times = np.arange(0.0, schedule.total_duration, 0.005)
        targets = np.array([schedule.reference(float(t)).com_target for t in times])
        jumps = np.linalg.norm(np.diff(targets, axis=0), axis=1)
        assert jumps.max() < 5e-3  # < 5 mm per 5 ms tick

    def test_com_stays_over_stance_during_single_support(self, schedule):
        for phase in schedule.phases():
            if phase.phase is Phase.SINGLE_LEFT:
                reference = schedule.reference(0.5 * (phase.t_start + phase.t_end))
                assert reference.com_target[1] < 0  # over the right foot
            elif phase.phase is Phase.SINGLE_RIGHT:
                reference = schedule.reference(0.5 * (phase.t_start + phase.t_end))
                assert reference.com_target[1] > 0  # over the left foot


class TestZmpMeasurement:
    """ZMP is the gate's evidence, so it must be measured, not assumed."""

    @pytest.fixture(scope="class")
    def standing(self):
        mj_model, mj_data = load_g1_torque_mujoco(timestep=DT)
        mujoco.mj_forward(mj_model, mj_data)
        return mj_model, mj_data

    def test_zmp_defined_while_in_contact(self, standing):
        mj_model, mj_data = standing
        zmp = measured_zmp(mj_model, mj_data)
        assert zmp is not None and zmp.shape == (2,)

    def test_zmp_inside_support_polygon_at_rest(self, standing):
        mj_model, mj_data = standing
        zmp = measured_zmp(mj_model, mj_data)
        assert point_in_support_polygon(mj_model, mj_data, zmp) > 0.0

    def test_zmp_near_com_when_static(self, standing):
        """With no acceleration the ZMP coincides with the ground CoM."""
        mj_model, mj_data = standing
        zmp = measured_zmp(mj_model, mj_data)
        assert np.allclose(zmp, mj_data.subtree_com[0][:2], atol=0.05)

    def test_zmp_none_when_airborne(self):
        mj_model, mj_data = load_g1_torque_mujoco(timestep=DT)
        mj_data.qpos[2] += 1.0  # lift clear of the floor
        mujoco.mj_forward(mj_model, mj_data)
        assert measured_zmp(mj_model, mj_data) is None
        assert support_polygon_margin(mj_model, mj_data) == float("-inf")


class TestContactSwitching:
    """The QP must accept a changing stance set."""

    def test_set_contacts_resizes_the_problem(self):
        from lab8_common import load_g1_pinocchio
        from g1_torque_model import torque_limits

        mj_model, _ = load_g1_torque_mujoco(timestep=DT)
        pin_model, pin_data = load_g1_pinocchio()
        qp = WholeBodyIDQP(
            pin_model, pin_data,
            [ContactSpec(LEFT), ContactSpec(RIGHT)],
            torque_limits(mj_model),
        )
        assert qp.n_forces == 12
        assert qp.contact_frames == [LEFT, RIGHT]

        qp.set_contacts([ContactSpec(RIGHT)])
        assert qp.n_forces == 6
        assert qp.contact_frames == [RIGHT]
        assert qp._solver is None  # factorisation discarded, not reused
