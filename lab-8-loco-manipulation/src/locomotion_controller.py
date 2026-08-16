"""Lab 8 — M2 Step 2.3: stepping controller (gait schedule → whole-body QP → τ).

Ties `GaitSchedule` to `WholeBodyIDQP`: each tick it looks up the phase, keeps
the QP's contact set in sync with which feet are actually meant to be on the
ground, points the swing-foot task at its reference (with feedforward), and
moves the CoM target for the weight shift.

The one genuinely new thing versus M1 is the **contact switch**. Removing a
foot from the stance set changes the QP's dimension, and it also changes the
physics the controller is allowed to assume: during single support the entire
horizontal force budget comes from one foot's friction cone and its CoP is
confined to one small rectangle. Two guards make the transition survivable:

* **Swing task ramp-in.** The swing foot task's weight is faded in over the
  first fraction of the swing rather than applied as a step, so lift-off does
  not begin with a large acceleration demand fighting the freshly reduced
  contact set.
* **Landing detection.** The schedule says when a foot *should* land; contact
  says when it *did*. The stance set is restored on measured contact (or the
  scheduled time, whichever comes first), so a slightly early touchdown is
  absorbed rather than fought.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gait_planner import GaitReference, GaitSchedule, Phase
from lab8_common import DT, foot_contact_state
from wb_id_qp import ContactSpec, WholeBodyIDQP
from wb_tasks import CoMTask, FramePositionTask, TaskStack

__all__ = ["SteppingController", "StepLog"]


@dataclass
class StepLog:
    """Per-tick telemetry for the M2/M3 gates."""

    t: list[float] = field(default_factory=list)
    phase: list[str] = field(default_factory=list)
    n_stance: list[int] = field(default_factory=list)
    swing_err_mm: list[float] = field(default_factory=list)
    com_err_mm: list[float] = field(default_factory=list)
    zmp_margin_mm: list[float] = field(default_factory=list)
    tau_max: list[float] = field(default_factory=list)
    pelvis_z: list[float] = field(default_factory=list)
    swing_height_mm: list[float] = field(default_factory=list)
    # M3 (DCM path) — empty when the controller runs on a plain CoM task.
    dcm_err_mm: list[float] = field(default_factory=list)
    vrp_saturated: list[bool] = field(default_factory=list)
    com_x: list[float] = field(default_factory=list)


class SteppingController:
    """Drive a `GaitSchedule` through the whole-body inverse-dynamics QP.

    Args:
        mj_model, mj_data: Torque-actuated G1 simulation.
        pin_model, pin_data: Matching Pinocchio model.
        schedule: Gait timeline.
        qp: Whole-body ID QP (its contact set is managed here).
        stack: Task stack holding CoM, swing-foot and posture tasks.
        com_task / swing_task: Handles for retargeting each tick.
        swing_ramp: Fraction of the swing phase over which the swing task's
            weight fades in.
    """

    def __init__(
        self,
        mj_model,
        mj_data,
        pin_model,
        pin_data,
        schedule: GaitSchedule,
        qp: WholeBodyIDQP,
        stack: TaskStack,
        com_task: CoMTask,
        swing_task: FramePositionTask,
        contact_template: ContactSpec | None = None,
        swing_ramp: float = 0.25,
        swing_orientation_task=None,
        dcm_plan=None,
        vrp_shrink: float = 0.7,
    ) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.schedule = schedule
        self.qp = qp
        self.stack = stack
        self.com_task = com_task
        self.swing_task = swing_task
        self.template = contact_template or ContactSpec("")
        self.swing_ramp = float(swing_ramp)

        # M3: when a DCM plan is supplied, `com_task` must be a `DCMTask` and
        # the controller commands the divergent component instead of a CoM
        # position. `vrp_shrink` is the fraction of each foot's contact patch
        # the commanded ZMP is allowed to use — a safety band, because the QP's
        # CoP constraint is the *hard* version of the same limit and hitting it
        # exactly leaves the solver no room to satisfy anything else.
        self.dcm_plan = dcm_plan
        self.vrp_shrink = float(vrp_shrink)

        # Optional: hold the swing foot level through the swing. A foot that
        # lands tilted touches down on one edge, which is both a poor contact
        # for the next stance phase and a torque spike as the sole slaps flat.
        self.swing_orientation_task = swing_orientation_task
        self.swing_orientation_weight = (
            swing_orientation_task.weight if swing_orientation_task is not None else 0.0
        )

        self.swing_weight = swing_task.weight
        self._current_stance: tuple[str, ...] = tuple(qp.contact_frames)
        self._t = 0.0
        self.log = StepLog()
        self.contact_switches = 0
        self.flight_ticks = 0

    # -- contact management -------------------------------------------------

    def _make_contact(self, frame_name: str) -> ContactSpec:
        return ContactSpec(
            frame_name=frame_name,
            friction=self.template.friction,
            half_length=self.template.half_length,
            half_width=self.template.half_width,
            min_normal_force=self.template.min_normal_force,
        )

    def _effective_stance(self, reference: GaitReference) -> tuple[str, ...]:
        """Stance set = what the schedule intends **∩ what the ground confirms**.

        The QP's contact constraint asserts `J_c q̈ + J̇_c q̇ = 0` and lets the
        solver push through wrenches at those frames. Listing a foot that is
        actually in the air therefore hands the controller an imaginary support
        polygon and imaginary forces — measured as the M2 failure mode: the
        schedule declared double support while the landing foot was still 60 mm
        up, the QP planned against two feet, and the robot launched itself
        (foot 0.66 m in the air, torques saturated, fall at 4.0 s).

        So a foot enters the stance set only when contact is measured. The
        schedule still decides *intent* — the swing foot is excluded while it
        is meant to be swinging — but reality has the last word on support.
        """
        left_down, right_down = foot_contact_state(self.mj_model, self.mj_data)
        measured = {
            self.schedule.left_frame: left_down,
            self.schedule.right_frame: right_down,
        }

        scheduled = set(reference.stance_feet)
        if reference.swing_foot is not None and self._swing_progress(reference) > 0.6:
            # Late in the swing the foot is allowed to re-enter support as soon
            # as it touches down — an early landing should be absorbed, not
            # fought until the schedule catches up.
            scheduled.add(reference.swing_foot)

        confirmed = tuple(sorted(name for name in scheduled if measured.get(name, False)))
        if confirmed:
            return confirmed

        # No confirmed contact (brief flight or a slipping touchdown): keep the
        # previous stance rather than emit an empty, unsolvable contact set.
        self.flight_ticks += 1
        return self._current_stance

    def _swing_progress(self, reference: GaitReference) -> float:
        index, phase = self.schedule._phase_at(self._t)
        del index
        return phase.progress(self._t)

    def _sync_contacts(self, stance: tuple[str, ...]) -> None:
        if stance != self._current_stance:
            self.qp.set_contacts([self._make_contact(name) for name in stance])
            self._current_stance = stance
            self.contact_switches += 1

    # -- per-tick -----------------------------------------------------------

    def update_targets(self, t: float) -> GaitReference:
        """Apply the schedule's references to the task stack for time `t`."""
        self._t = t
        reference = self.schedule.reference(t)

        if self.dcm_plan is None:
            self.com_task.set_target(reference.com_target)
        else:
            plan = self.dcm_plan.reference(t)
            self.com_task.set_reference(plan.xi, plan.xi_dot)
            # One integrator step per control tick, on the error measured from
            # the previous tick's kinematics (this runs before the fresh
            # `update_dynamics`). Sign matches the ZMP law: a DCM that has run
            # past its reference must be answered by a ZMP further out.
            self.com_task.integrate_error(
                self.com_task.current_dcm(self.pin_data) - plan.xi, DT
            )

        if reference.swing_foot is None or reference.swing_foot in self._current_stance:
            # Either double support, or the swing foot has already landed and
            # been taken into the stance set. Driving a position task on a foot
            # the QP is simultaneously constraining to zero acceleration is a
            # direct contradiction — the solver would trade balance for it.
            self.swing_task.enabled = False
        else:
            self.swing_task.enabled = True
            self.swing_task.frame_id = self.pin_model.getFrameId(reference.swing_foot)
            self.swing_task.frame_name = reference.swing_foot
            self.swing_task.set_target(
                reference.swing_position,
                reference.swing_velocity,
                reference.swing_acceleration,
            )
            # Fade the swing task in so lift-off does not step-change the
            # acceleration demand at the same instant support is halved.
            ramp = min(self._swing_progress(reference) / max(self.swing_ramp, 1e-6), 1.0)
            self.swing_task.weight = self.swing_weight * ramp

        if self.swing_orientation_task is not None:
            if reference.swing_foot is None or reference.swing_foot in self._current_stance:
                self.swing_orientation_task.enabled = False
            else:
                self.swing_orientation_task.enabled = True
                self.swing_orientation_task.frame_id = self.pin_model.getFrameId(
                    reference.swing_foot
                )
                self.swing_orientation_task.frame_name = reference.swing_foot
                self.swing_orientation_task.weight = self.swing_orientation_weight

        self._sync_contacts(self._effective_stance(reference))
        if self.dcm_plan is not None:
            self._update_vrp_bounds()
        return reference

    def _update_vrp_bounds(self) -> None:
        """Confine the commanded ZMP to the feet that are actually loaded.

        The box is the union of the stance feet's contact patches, shrunk by
        `vrp_shrink`. Axis-aligned is exact here because the gait walks along
        +x with the feet unrotated; a turning gait would need the patch
        corners rotated into world before taking the hull.

        Without this the DCM law will happily ask for a ZMP outside the
        support polygon whenever the divergent component runs far enough
        ahead. The QP cannot deliver it — the CoP rows forbid it — so the
        request is quietly traded off in the cost and the controller believes
        it is still in charge. Clamping makes the saturation explicit
        (`DCMTask.vrp_saturated`) and leaves the QP a feasible target.
        """
        lower = np.full(2, np.inf)
        upper = np.full(2, -np.inf)
        # Same asymmetric patch the QP's CoP rows use — the clamp is only
        # useful if it agrees with the hard constraint it is protecting.
        offset = np.array([self.template.center_x, self.template.center_y])
        half = np.array(
            [self.template.half_length * self.vrp_shrink,
             self.template.half_width * self.vrp_shrink]
        )
        for name in self._current_stance:
            frame_id = self.pin_model.getFrameId(name)
            centre = (
                np.asarray(self.pin_data.oMf[frame_id].translation, dtype=float)[:2] + offset
            )
            lower = np.minimum(lower, centre - half)
            upper = np.maximum(upper, centre + half)
        if np.all(np.isfinite(lower)):
            self.com_task.set_vrp_bounds(lower, upper)

    def record(self, t: float, reference: GaitReference, tau: np.ndarray) -> None:
        """Append one row of telemetry."""
        from lab8_common import measured_zmp, point_in_support_polygon

        self.log.t.append(t)
        self.log.phase.append(reference.phase.value)
        self.log.n_stance.append(len(self._current_stance))
        self.log.tau_max.append(float(np.abs(tau).max()))
        self.log.pelvis_z.append(float(self.mj_data.qpos[2]))

        com = self.mj_data.subtree_com[0]
        self.log.com_x.append(float(com[0]))
        if self.dcm_plan is None:
            com_target = reference.com_target[:2]
            self.log.dcm_err_mm.append(0.0)
            self.log.vrp_saturated.append(False)
        else:
            plan = self.dcm_plan.reference(t)
            com_target = plan.com
            measured_dcm = self.com_task.current_dcm(self.pin_data)
            self.log.dcm_err_mm.append(
                float(np.linalg.norm(plan.xi - measured_dcm)) * 1000.0
            )
            self.log.vrp_saturated.append(bool(self.com_task.vrp_saturated))
        self.log.com_err_mm.append(float(np.linalg.norm(com_target - com[:2])) * 1000.0)

        if reference.swing_foot is not None:
            frame_id = self.pin_model.getFrameId(reference.swing_foot)
            from lab8_common import pin_point_to_world

            actual = pin_point_to_world(self.pin_data.oMf[frame_id].translation)
            self.log.swing_err_mm.append(
                float(np.linalg.norm(reference.swing_position - actual)) * 1000.0
            )
            self.log.swing_height_mm.append(float(actual[2]) * 1000.0)
        else:
            self.log.swing_err_mm.append(0.0)
            self.log.swing_height_mm.append(0.0)

        zmp = measured_zmp(self.mj_model, self.mj_data)
        if zmp is None:
            self.log.zmp_margin_mm.append(float("-inf"))
        else:
            self.log.zmp_margin_mm.append(
                point_in_support_polygon(self.mj_model, self.mj_data, zmp) * 1000.0
            )

    def stance_fraction_zmp_inside(self, margin_m: float = 0.0) -> float:
        """Fraction of *loaded* ticks whose ZMP sat inside the support polygon."""
        values = [m for m in self.log.zmp_margin_mm if np.isfinite(m)]
        if not values:
            return 0.0
        inside = sum(1 for m in values if m > margin_m * 1000.0)
        return inside / len(values)
