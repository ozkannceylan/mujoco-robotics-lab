"""Lab 8 — M3 tests: DCM planning, the DCM control law, and the foot CoP model.

The walking itself is the gate demo's job. What is checked here is that the
pieces the gate rests on are individually right, because each of them failed
silently before it failed visibly:

* the DCM plan must satisfy its own defining ODE (a plan that does not is a
  reference no controller can track, and the symptom is just "it fell");
* the control law must reduce to the planned ZMP when tracking is perfect,
  which is the property that makes the feedback term a *correction* rather
  than the whole command;
* the foot's centre-of-pressure box must match the geometry MuJoCo simulates —
  the symmetric ±0.08 m guess cost M3 half its forward CoP authority while
  promising 30 mm of rearward authority the foot does not have.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dcm_planner import DCMPlan, VRPSegment  # noqa: E402
from gait_planner import GaitSchedule, Phase  # noqa: E402
from lab8_common import GRAVITY, divergent_component, lipm_omega  # noqa: E402
from wb_id_qp import ContactSpec, WholeBodyIDQP  # noqa: E402

LEFT = "left_ankle_roll_link"
RIGHT = "right_ankle_roll_link"
LEFT_HOME = np.array([0.0, 0.12, 0.03])
RIGHT_HOME = np.array([0.0, -0.12, 0.03])
COM_HOME = np.array([0.0, 0.0, 0.69])
COM_HEIGHT = 0.66


def make_schedule(step_length: float = 0.10, n_steps: int = 6) -> GaitSchedule:
    return GaitSchedule(
        LEFT, RIGHT, LEFT_HOME, RIGHT_HOME, COM_HOME,
        # t_initial matches the demo: the settle hold is what drives the
        # initial DCM lead to zero, and it does so as e^{-ω·hold}.
        n_steps=n_steps, t_initial=1.5, t_double=0.3, t_single=0.7,
        step_length=step_length, step_width=0.18, step_height=0.05,
    )


@pytest.fixture
def plan() -> DCMPlan:
    return DCMPlan(make_schedule(), COM_HEIGHT, COM_HOME)


# ---------------------------------------------------------------------------
# LIPM primitives
# ---------------------------------------------------------------------------


class TestLIPMPrimitives:
    def test_omega_matches_definition(self):
        assert lipm_omega(0.66) == pytest.approx(np.sqrt(GRAVITY / 0.66))

    def test_omega_rejects_nonpositive_height(self):
        with pytest.raises(ValueError):
            lipm_omega(0.0)

    def test_dcm_equals_com_at_rest(self):
        com = np.array([0.1, -0.2])
        assert divergent_component(com, np.zeros(2), 3.9) == pytest.approx(com)

    def test_dcm_leads_the_com_when_moving(self):
        # Walking forward, the divergent component sits ahead of the CoM —
        # that lead is exactly what the next footstep has to catch.
        xi = divergent_component(np.zeros(2), np.array([0.4, 0.0]), 4.0)
        assert xi[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Segment algebra
# ---------------------------------------------------------------------------


class TestVRPSegment:
    def test_constant_vrp_arc_satisfies_the_dcm_ode(self):
        segment = VRPSegment(0.0, 0.7, np.array([0.1, 0.0]), np.array([0.1, 0.0]), 3.9)
        segment.solve_backward(np.array([0.25, 0.05]))
        eps = 1e-6
        for t in (0.05, 0.3, 0.65):
            xi, xi_dot = segment.evaluate(t)
            numerical = (segment.evaluate(t + eps)[0] - segment.evaluate(t - eps)[0]) / (2 * eps)
            assert xi_dot == pytest.approx(numerical, abs=1e-4)
            # ξ̇ = ω(ξ − p) is the definition the whole plan rests on.
            assert xi_dot == pytest.approx(3.9 * (xi - segment.vrp(t)), abs=1e-9)

    def test_linear_vrp_arc_satisfies_the_dcm_ode(self):
        segment = VRPSegment(1.0, 1.3, np.array([0.0, -0.1]), np.array([0.1, 0.1]), 3.9)
        segment.solve_backward(np.array([0.15, 0.12]))
        for t in (1.02, 1.15, 1.28):
            xi, xi_dot = segment.evaluate(t)
            assert xi_dot == pytest.approx(3.9 * (xi - segment.vrp(t)), abs=1e-9)

    def test_backward_solve_hits_the_requested_end_value(self):
        segment = VRPSegment(0.0, 0.5, np.array([0.0, 0.0]), np.array([0.2, -0.1]), 3.9)
        xi_end = np.array([0.3, 0.05])
        segment.solve_backward(xi_end)
        assert segment.evaluate(0.5)[0] == pytest.approx(xi_end, abs=1e-9)

    def test_backward_solve_returns_the_start_value(self):
        segment = VRPSegment(2.0, 2.7, np.array([0.1, 0.1]), np.array([0.1, 0.1]), 3.9)
        xi_start = segment.solve_backward(np.array([0.4, 0.2]))
        assert segment.evaluate(2.0)[0] == pytest.approx(xi_start, abs=1e-9)


# ---------------------------------------------------------------------------
# Whole-plan properties
# ---------------------------------------------------------------------------


class TestDCMPlan:
    def test_dcm_is_continuous_across_phase_boundaries(self, plan):
        # ξ_eos of one segment *is* ξ_ini of the next by construction. A jump
        # here would be a reference step the controller could only chase.
        for segment in plan.segments[1:]:
            before = plan.reference_dcm(segment.t_start - 1e-6)
            after = plan.reference_dcm(segment.t_start + 1e-6)
            assert after == pytest.approx(before, abs=1e-5)

    def test_vrp_is_continuous_across_phase_boundaries(self, plan):
        for segment in plan.segments[1:]:
            before = plan.reference(segment.t_start - 1e-6).vrp
            after = plan.reference(segment.t_start + 1e-6).vrp
            assert after == pytest.approx(before, abs=1e-5)

    def test_terminal_dcm_rests_over_the_final_support(self, plan):
        end = plan.reference(plan.schedule.total_duration - 1e-6)
        assert end.xi == pytest.approx(end.vrp, abs=1e-3)
        assert np.linalg.norm(end.xi_dot) < 1e-2

    def test_initial_dcm_leads_the_ramping_zmp_by_k_over_omega(self, plan):
        # With the default `settle_sweep=1.0` the settle ramps the ZMP from the
        # foot midpoint onto the first stance foot, and a DCM tracking a linear
        # ramp leads it by exactly k/ω. That lead is the lateral momentum the
        # first step needs, not an initial-condition error — measured best of
        # every alternative (see `DCMPlan._build`), so it is pinned here.
        settle = plan.segments[0]
        midpoint = 0.5 * (LEFT_HOME[:2] + RIGHT_HOME[:2]) + plan.foot_offset
        expected = midpoint + settle.slope / plan.omega
        assert plan.xi_initial == pytest.approx(expected, abs=1e-3)

    def test_holding_the_settle_removes_the_initial_lead(self):
        # The knob works, even though 1.0 is the better setting.
        held = DCMPlan(make_schedule(), COM_HEIGHT, COM_HOME, settle_sweep=0.3)
        midpoint = 0.5 * (LEFT_HOME[:2] + RIGHT_HOME[:2]) + held.foot_offset
        assert held.xi_initial == pytest.approx(midpoint, abs=2e-3)

    def test_single_support_vrp_sits_on_the_stance_foot(self, plan):
        schedule = plan.schedule
        for index, phase in enumerate(schedule.phases()):
            if phase.phase not in (Phase.SINGLE_LEFT, Phase.SINGLE_RIGHT):
                continue
            stance = schedule.stance_frame_of_phase(index)
            foot = schedule.foot_positions(index)[stance][:2] + plan.foot_offset
            middle = 0.5 * (phase.t_start + phase.t_end)
            assert plan.reference(middle).vrp == pytest.approx(foot, abs=1e-9)

    def test_vrp_targets_the_contact_patch_centre_not_the_ankle(self, plan):
        # The G1's ankle frame is 35 mm behind the middle of its sole; planning
        # the ZMP at the frame throws away a third of the forward CoP travel.
        patch = ContactSpec("")
        assert plan.foot_offset == pytest.approx([patch.center_x, patch.center_y])

    def test_nominal_com_follows_the_stable_dynamics(self, plan):
        # ċ = −ω(c − ξ): the convergent half of the LIPM. Checked by finite
        # difference against the cached integration.
        for t in (2.0, 3.5, 5.0):
            com, com_velocity = plan.nominal_com(t)
            ahead, _ = plan.nominal_com(t + 1e-3)
            behind, _ = plan.nominal_com(t - 1e-3)
            assert (ahead - behind) / 2e-3 == pytest.approx(com_velocity, abs=5e-3)
            assert com_velocity == pytest.approx(
                -plan.omega * (com - plan.reference_dcm(t)), abs=1e-9
            )

    def test_com_advances_by_the_planned_stride(self, plan):
        # Plus the one-off `foot_offset` shift: the CoM starts over the ankles
        # and finishes over the contact-patch centres.
        start, _ = plan.nominal_com(0.0)
        end, _ = plan.nominal_com(plan.schedule.total_duration)
        expected = plan.schedule.total_advance + plan.foot_offset[0]
        assert end[0] - start[0] == pytest.approx(expected, abs=0.02)

    def test_stepping_in_place_produces_no_forward_travel(self):
        stationary = DCMPlan(make_schedule(step_length=0.0), COM_HEIGHT, COM_HOME)
        start, _ = stationary.nominal_com(0.0)
        end, _ = stationary.nominal_com(stationary.schedule.total_duration)
        assert end[0] - start[0] == pytest.approx(stationary.foot_offset[0], abs=2e-3)

    def test_lateral_dcm_alternates_with_the_stance_foot(self, plan):
        schedule = plan.schedule
        signs = []
        for index, phase in enumerate(schedule.phases()):
            if phase.phase not in (Phase.SINGLE_LEFT, Phase.SINGLE_RIGHT):
                continue
            middle = 0.5 * (phase.t_start + phase.t_end)
            signs.append(np.sign(plan.reference(middle).xi[1]))
        assert len(signs) >= 4
        # Swinging the left foot means standing on the right: the DCM must be
        # on the right (negative y) and swap every step.
        assert all(a != b for a, b in zip(signs, signs[1:]))


# ---------------------------------------------------------------------------
# Control law
# ---------------------------------------------------------------------------


class _FakeComData:
    """Minimal stand-in for `pin.Data` exposing only what `DCMTask` reads."""

    def __init__(self, com_pin: np.ndarray, com_velocity: np.ndarray) -> None:
        self.com = [np.asarray(com_pin, dtype=float)]
        self.vcom = [np.asarray(com_velocity, dtype=float)]


class TestDCMControlLaw:
    @staticmethod
    def _task(omega: float = 3.9, gain: float = 3.0, integral_gain: float = 0.0):
        from lab8_common import PELVIS_MJCF_Z
        from wb_tasks import DCMTask

        task = DCMTask.__new__(DCMTask)  # no Pinocchio model needed for the law
        super(DCMTask, task).__init__(None, axes=(0, 1), weight=1e4, gain=gain, name="dcm")
        task.omega = omega
        task.integral_gain = integral_gain
        task.integral_leak = 0.5
        task.integral = np.zeros(2)
        task.xi_target = np.zeros(2)
        task.xi_dot_target = np.zeros(2)
        task.vrp_lower = None
        task.vrp_upper = None
        task.last_vrp = np.zeros(2)
        task.vrp_saturated = False
        task._pelvis_offset = PELVIS_MJCF_Z
        return task

    @staticmethod
    def _data(com_xy, com_velocity_xy):
        from lab8_common import world_point_to_pin

        com = world_point_to_pin(np.array([com_xy[0], com_xy[1], 0.66]))
        return _FakeComData(com, np.array([com_velocity_xy[0], com_velocity_xy[1], 0.0]))

    def test_measured_dcm_matches_the_definition(self):
        task = self._task()
        data = self._data([0.1, -0.05], [0.39, 0.0])
        assert task.current_dcm(data) == pytest.approx([0.1 + 0.39 / 3.9, -0.05])

    def test_perfect_tracking_commands_exactly_the_planned_zmp(self):
        # p_cmd = ξ − ξ̇_ref/ω + (k/ω)(ξ − ξ_ref). With ξ = ξ_ref and
        # ξ̇_ref = ω(ξ_ref − p), every feedback term vanishes and the command
        # collapses to p. If it did not, the feedback would be biasing the
        # nominal gait rather than correcting it.
        omega, zmp = 3.9, np.array([0.12, -0.09])
        task = self._task(omega=omega)
        xi = np.array([0.2, -0.04])
        task.set_reference(xi, omega * (xi - zmp))
        data = self._data(xi - np.array([0.0, 0.0]), [0.0, 0.0])
        data.vcom[0][:2] = omega * (xi - np.asarray(task.current_dcm(data)))
        # Re-solve: place the CoM so that c + ċ/ω lands exactly on ξ.
        data = self._data([0.1, 0.0], [(xi[0] - 0.1) * omega, xi[1] * omega])
        assert task.current_dcm(data) == pytest.approx(xi)
        assert task.commanded_vrp(data) == pytest.approx(zmp, abs=1e-9)

    def test_dcm_ahead_of_reference_pushes_the_zmp_further_out(self):
        task = self._task(gain=3.0)
        task.set_reference(np.array([0.0, 0.0]), np.zeros(2))
        data = self._data([0.05, 0.0], [0.0, 0.0])
        vrp = task.commanded_vrp(data)
        assert vrp[0] > 0.05  # beyond the DCM itself, to pull it back

    def test_commanded_acceleration_matches_the_lipm(self):
        task = self._task()
        task.set_reference(np.array([0.02, 0.0]), np.zeros(2))
        data = self._data([0.05, 0.01], [0.1, -0.02])
        acceleration = task.desired_acceleration(None, data, None, None)
        expected = task.omega**2 * (np.array([0.05, 0.01]) - task.last_vrp)
        assert acceleration == pytest.approx(expected)

    def test_vrp_clamp_bounds_the_command_and_reports_saturation(self):
        task = self._task()
        task.set_reference(np.zeros(2), np.zeros(2))
        task.set_vrp_bounds(np.array([-0.05, -0.05]), np.array([0.05, 0.05]))
        data = self._data([0.4, 0.0], [0.0, 0.0])
        vrp = task.commanded_vrp(data)
        assert vrp[0] == pytest.approx(0.05)
        assert task.vrp_saturated

    def test_integral_state_is_leaky(self):
        task = self._task(integral_gain=1.0)
        for _ in range(20000):
            task.integrate_error(np.array([0.01, 0.0]), 1e-3)
        # Steady state of ẋ = e − leak·x is e/leak, not unbounded growth.
        assert task.integral[0] == pytest.approx(0.01 / 0.5, rel=1e-2)

    def test_integral_disabled_by_default(self):
        task = self._task()
        task.integrate_error(np.array([1.0, 1.0]), 1e-3)
        assert task.integral == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# Foot centre-of-pressure model
# ---------------------------------------------------------------------------


class TestFootCoPModel:
    """The QP's CoP rows must describe the foot MuJoCo actually simulates.

    Menagerie's G1 foot is four spheres at x ∈ {−0.05, 0.12}, y ∈ {±0.025,
    ±0.03}, z = −0.03 in the ankle-roll frame, radius 5 mm.
    """

    def test_defaults_match_the_menagerie_foot_geometry(self):
        contact = ContactSpec("foot")
        assert contact.center_x - contact.half_length == pytest.approx(-0.05)
        assert contact.center_x + contact.half_length == pytest.approx(0.12)
        assert contact.half_width == pytest.approx(0.025)
        assert contact.origin_height == pytest.approx(0.035)

    @staticmethod
    def _cop_rows(contact: ContactSpec):
        qp = WholeBodyIDQP.__new__(WholeBodyIDQP)
        qp.contacts = [contact]
        qp.n_forces = 6
        matrix, lower, _ = qp._friction_constraints()
        return matrix[5:9], lower[5:9]

    @staticmethod
    def _wrench(cop_xy, force_xy, contact: ContactSpec, normal: float = 300.0):
        """Wrench about the frame origin produced by a normal force at `cop`."""
        f = np.array([force_xy[0], force_xy[1], normal])
        r = np.array([cop_xy[0], cop_xy[1], -contact.origin_height])
        moment = np.cross(r, f)
        return np.concatenate([f, moment])

    @pytest.mark.parametrize(
        "cop, inside",
        [
            ((0.035, 0.0), True),     # patch centre
            ((0.115, 0.0), True),     # just inside the toe
            ((0.125, 0.0), False),    # past the toe
            ((-0.045, 0.0), True),    # just inside the heel
            ((-0.06, 0.0), False),    # past the heel
            ((0.035, 0.02), True),
            ((0.035, 0.03), False),   # past the (0.025) inner edge
        ],
    )
    def test_cop_rows_accept_exactly_the_real_patch(self, cop, inside):
        contact = ContactSpec("foot")
        matrix, lower = self._cop_rows(contact)
        wrench = self._wrench(cop, (0.0, 0.0), contact)
        satisfied = bool(np.all(matrix @ wrench >= lower - 1e-9))
        assert satisfied is inside

    def test_shear_force_shifts_the_admissible_cop(self):
        # CoP = (−m_y − h·f_x)/f_z: the frame origin is 35 mm above the ground,
        # so ignoring f_x mislocates the CoP by h·f_x/f_z — 12 mm at the shear
        # a walking step actually uses.
        contact = ContactSpec("foot")
        matrix, lower = self._cop_rows(contact)
        wrench = self._wrench((0.115, 0.0), (100.0, 0.0), contact)
        assert np.all(matrix @ wrench >= lower - 1e-9)

        naive = wrench.copy()
        naive[0] = 0.0  # the same moment, pretending there is no shear
        shifted = self._wrench((0.115, 0.0), (0.0, 0.0), contact)
        assert not np.allclose(naive[3:], shifted[3:])
