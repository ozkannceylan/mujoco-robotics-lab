"""Lab 8 — M3: divergent-component-of-motion (DCM) reference generation.

Why this module exists
----------------------
M2 walked in place by commanding the CoM to sit over whichever foot was about
to take the load: shift, swing, shift, swing. That rule is *quasi-static* — it
assumes the robot can come to rest over each foot in turn. Forward walking
never provides that moment. Measured directly (LESSONS L-M3-c): the same
controller with a non-zero stride reaches 3 of 10 steps and 0.22 m before
falling, at every stride length, double-support duration and CoM bias tried.

The standard answer is to stop commanding where the CoM *is* and start
commanding where it is *going*. Under the linear inverted pendulum the CoM
obeys ``c̈ = ω²(c − p)`` with ``ω = √(g/z_c)`` and ``p`` the ZMP. That
second-order system splits into

    ξ = c + ċ/ω        (divergent — unstable, ξ̇ = ω(ξ − p))
    η = c − ċ/ω        (convergent — stable, needs no control)

so balance is entirely a question of steering ξ, and ξ is steered by the ZMP,
which the whole-body QP can place anywhere inside the support polygon. Walking
forward is then not a sequence of static poses but a chain of controlled
divergences: ξ is *allowed* to run away from each stance foot, and the next
footstep is placed where it will catch it.

What is planned here
--------------------
A ZMP (equivalently, for constant height, a *virtual repellent point*)
trajectory that is **piecewise linear in time** and continuous:

    initial double support : foot midpoint → first stance foot
    single support         : held at the stance foot
    double support         : previous stance foot → next stance foot
    final double support   : previous stance foot → foot midpoint

and the DCM trajectory that this ZMP produces, obtained by integrating
``ξ̇ = ω(ξ − p)`` **backwards** from the terminal condition ``ξ(T) = p_final``
(stop over the final support). Backwards is the only stable direction: the
forward integration of an unstable system amplifies the initial-condition
error by ``e^{ωT}``, while the backward recursion contracts it by ``e^{−ωT}``.

For a segment with ``p(τ) = p₀ + kτ`` over ``τ ∈ [0, T]`` the closed-form
solution is

    ξ(τ) = A e^{ωτ} + p(τ) + k/ω,        A = ξ₀ − p₀ − k/ω
    ξ̇(τ) = ω A e^{ωτ} + k

so the backward step from a known end value is
``A = (ξ_T − p_T − k/ω) e^{−ωT}``. Constant-ZMP segments are the ``k = 0``
case of the same formula, which is why both are handled by one class.

Reference: Englsberger, Ott & Albu-Schäffer, "Three-Dimensional Bipedal
Walking Control Based on Divergent Component of Motion", T-RO 2015.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gait_planner import GaitSchedule, Phase

__all__ = ["VRPSegment", "DCMReference", "DCMPlan"]


@dataclass
class VRPSegment:
    """One piecewise-linear ZMP/VRP segment and the DCM arc it generates.

    Attributes:
        t_start / t_end: Segment interval [s].
        p_start / p_end: Horizontal ZMP at each end [m] (2,).
        omega: LIPM frequency [1/s].
        coefficient: The ``A`` of ``ξ(τ) = A e^{ωτ} + p(τ) + k/ω`` (2,), set by
            the backward recursion.
    """

    t_start: float
    t_end: float
    p_start: np.ndarray
    p_end: np.ndarray
    omega: float
    coefficient: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return max(self.t_end - self.t_start, 1e-9)

    @property
    def slope(self) -> np.ndarray:
        """ZMP rate `k = (p_end − p_start)/T` [m/s]."""
        return (self.p_end - self.p_start) / self.duration

    def vrp(self, t: float) -> np.ndarray:
        """ZMP/VRP at absolute time `t`, clamped to the segment."""
        tau = float(np.clip(t - self.t_start, 0.0, self.duration))
        return self.p_start + self.slope * tau

    def solve_backward(self, xi_end: np.ndarray) -> np.ndarray:
        """Set `coefficient` from the end-of-segment DCM; return its start value."""
        k_over_omega = self.slope / self.omega
        self.coefficient = (xi_end - self.p_end - k_over_omega) * np.exp(
            -self.omega * self.duration
        )
        return self.coefficient + self.p_start + k_over_omega

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """DCM position and velocity at absolute time `t`."""
        if self.coefficient is None:
            raise RuntimeError("segment not solved — call DCMPlan first")
        tau = float(np.clip(t - self.t_start, 0.0, self.duration))
        growth = self.coefficient * np.exp(self.omega * tau)
        k_over_omega = self.slope / self.omega
        xi = growth + self.vrp(t) + k_over_omega
        xi_dot = self.omega * growth + self.slope
        return xi, xi_dot


@dataclass
class DCMReference:
    """DCM/CoM reference at one instant (all horizontal, 2-vectors)."""

    xi: np.ndarray
    xi_dot: np.ndarray
    vrp: np.ndarray
    com: np.ndarray
    com_velocity: np.ndarray


class DCMPlan:
    """DCM trajectory for a `GaitSchedule`.

    Args:
        schedule: Gait timeline; supplies phase boundaries and footholds.
        com_height: CoM height above the contact plane [m] — measured on the
            settled robot, not assumed, because ω is what couples the plan to
            the machine.
        com_home: CoM at t=0 (world, 3,). Only the horizontal part is used; it
            seeds the nominal CoM integration.
        settle_sweep: Fraction of the initial settle spent sweeping the ZMP
            onto the first stance foot; the rest is held at the foot midpoint.
            Defaults to 1.0 — no hold — which measured best by a wide margin.
            See `_build` for the numbers and why the obvious argument for a
            hold is wrong.
        foot_offset: Contact-patch centre relative to the foot **frame** [m].
            The G1's ankle-roll frame is not the middle of its sole — the patch
            runs from 0.05 m behind it to 0.12 m ahead — so planning the ZMP at
            the frame origin puts it 35 mm rearward of the foot's centre and
            spends a third of the available forward CoP travel before the
            controller has asked for anything. Defaults to `ContactSpec`'s
            values so the plan and the QP's CoP bounds describe one foot.

    The nominal CoM trajectory is integrated forward from `com_home` through
    ``ċ = −ω(c − ξ)`` — the *stable* half of the LIPM, so forward integration
    is well posed here. It is used for logging and for the touchdown-time CoM
    estimate, never as the tracking target: the controller tracks ξ.
    """

    def __init__(
        self,
        schedule: GaitSchedule,
        com_height: float,
        com_home: np.ndarray,
        omega: float | None = None,
        foot_offset: np.ndarray | None = None,
        settle_sweep: float = 1.0,
    ) -> None:
        from lab8_common import lipm_omega
        from wb_id_qp import ContactSpec

        self.schedule = schedule
        self.com_height = float(com_height)
        self.omega = float(omega) if omega is not None else lipm_omega(com_height)
        self.com_home = np.asarray(com_home, dtype=float).copy()
        if foot_offset is None:
            patch = ContactSpec("")
            foot_offset = np.array([patch.center_x, patch.center_y])
        self.foot_offset = np.asarray(foot_offset, dtype=float)[:2].copy()
        self.settle_sweep = float(settle_sweep)
        self.segments: list[VRPSegment] = []
        self._build()

    # -- construction ------------------------------------------------------

    def _phase_vrp(self, index: int, previous_end: np.ndarray | None) -> tuple[
        np.ndarray, np.ndarray
    ]:
        """(p_start, p_end) for phase `index`.

        Continuity is imposed by construction: every segment starts where the
        last one ended. A discontinuous ZMP is physically admissible (the CoP
        *can* jump between feet) but it puts a step change into the commanded
        CoM acceleration at every phase boundary, which the QP then pays for in
        torque. Sweeping it linearly across the double support costs nothing
        and is what the foot pressure actually does.
        """
        phase = self.schedule.phases()[index]
        feet = {
            name: position[:2] + self.foot_offset
            for name, position in self.schedule.foot_positions(index).items()
        }
        midpoint = 0.5 * (
            feet[self.schedule.left_frame] + feet[self.schedule.right_frame]
        )

        if phase.phase in (Phase.SINGLE_LEFT, Phase.SINGLE_RIGHT):
            stance = self.schedule.stance_frame_of_phase(index)
            target = feet[stance][:2].copy()
            # The ZMP is already at the stance foot when the swing starts (the
            # preceding double support put it there), and stays put while the
            # other foot is in flight — one foot, one CoP region.
            start = previous_end if previous_end is not None else target
            return start.copy(), target

        # Double support (including the initial settle and the final DONE):
        # sweep to wherever the load goes next.
        next_stance = self.schedule.next_stance_frame(index)
        target = midpoint if next_stance is None else feet[next_stance][:2]
        start = previous_end if previous_end is not None else midpoint
        return np.asarray(start, dtype=float).copy(), np.asarray(target, dtype=float).copy()

    def _build(self) -> None:
        """Lay out the ZMP segments, then back-propagate the DCM through them.

        Segments need not be one-per-gait-phase, and `settle_sweep` exercises
        that: below 1.0 the initial settle is split into a **hold** at the foot
        midpoint plus a shorter sweep onto the first stance foot.

        The hold exists to answer an objection that turns out to be wrong, and
        the default is 1.0 (no hold) because the measurement said so. The
        objection: a DCM tracking a linearly ramping ZMP leads it by `k/ω` in
        steady state, so sweeping across the whole 1.5 s settle starts the plan
        with ξ about 30 mm to one side of a robot that is standing perfectly
        still — an initial-condition mismatch. The measurement:

        | settle_sweep | steps | distance | DCM RMS | ZMP clamped |
        |---|---|---|---|---|
        | 0.3 | 6/12, fell | 0.77 m | 138.2 mm | 32 % |
        | 0.5 | 8/12, fell | 0.89 m | 124.4 mm | 20 % |
        | 0.7 | 6/12, fell | 0.61 m | 159.6 mm | 27 % |
        | **1.0** | **12/12** | **1.18 m** | **6.2 mm** | 3 % |

        The lead is not a mismatch to be removed — it is the lateral momentum
        the first step needs, and 1.5 s of gentle ramp is the robot acquiring
        it. Holding still and then sweeping the ZMP across in half the time
        enters the first transfer cold and twice as fast. The parameter stays
        because the reasoning is worth being able to re-run.
        """
        previous_end: np.ndarray | None = None
        for index, phase in enumerate(self.schedule.phases()):
            p_start, p_end = self._phase_vrp(index, previous_end)
            if (
                index == 0
                and self.settle_sweep < 1.0
                and phase.duration > 2.0 * self.schedule.t_double
            ):
                hold_until = phase.t_end - self.settle_sweep * phase.duration
                self.segments.append(
                    VRPSegment(phase.t_start, hold_until, p_start, p_start.copy(), self.omega)
                )
                self.segments.append(
                    VRPSegment(hold_until, phase.t_end, p_start.copy(), p_end, self.omega)
                )
            else:
                self.segments.append(
                    VRPSegment(phase.t_start, phase.t_end, p_start, p_end, self.omega)
                )
            previous_end = p_end

        # Terminal condition: come to rest over the final ZMP. ξ = c and ċ = 0
        # there, so the robot ends statically balanced rather than still
        # diverging at the end of the last step.
        xi_end = self.segments[-1].p_end.copy()
        for segment in reversed(self.segments):
            xi_end = segment.solve_backward(xi_end)
        self.xi_initial = xi_end.copy()

    # -- evaluation --------------------------------------------------------

    def _segment_at(self, t: float) -> VRPSegment:
        for segment in self.segments:
            if t < segment.t_end:
                return segment
        return self.segments[-1]

    def reference(self, t: float) -> DCMReference:
        """DCM, its velocity, the planned ZMP, and the nominal CoM at `t`."""
        segment = self._segment_at(t)
        xi, xi_dot = segment.evaluate(t)
        com, com_velocity = self.nominal_com(t)
        return DCMReference(
            xi=xi, xi_dot=xi_dot, vrp=segment.vrp(t), com=com, com_velocity=com_velocity
        )

    def nominal_com(self, t: float, dt: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """Nominal CoM position/velocity at `t` from ``ċ = −ω(c − ξ)``.

        Cached on a fixed grid and linearly interpolated: the integration is
        only needed for plots and diagnostics, so paying for it once beats
        re-integrating from t=0 on every query.
        """
        if not hasattr(self, "_com_grid"):
            self._integrate_com(dt)
        grid, values = self._com_grid, self._com_values
        t = float(np.clip(t, grid[0], grid[-1]))
        index = int(np.searchsorted(grid, t) - 1)
        index = int(np.clip(index, 0, len(grid) - 2))
        alpha = (t - grid[index]) / (grid[index + 1] - grid[index])
        com = values[index] + alpha * (values[index + 1] - values[index])
        xi = self.reference_dcm(t)
        return com, -self.omega * (com - xi)

    def reference_dcm(self, t: float) -> np.ndarray:
        """Just the DCM at `t` (avoids the CoM recursion)."""
        return self._segment_at(t).evaluate(t)[0]

    def _integrate_com(self, dt: float) -> None:
        total = self.schedule.total_duration
        grid = np.arange(0.0, total + dt, dt)
        values = np.zeros((len(grid), 2))
        com = self.com_home[:2].copy()
        values[0] = com
        for i in range(1, len(grid)):
            # Midpoint rule: ċ = −ω(c − ξ) is stiff enough at ω≈4 that plain
            # Euler visibly lags over a 14 s walk.
            xi_half = self.reference_dcm(grid[i - 1] + 0.5 * dt)
            com_half = com + 0.5 * dt * (-self.omega * (com - self.reference_dcm(grid[i - 1])))
            com = com + dt * (-self.omega * (com_half - xi_half))
            values[i] = com
        self._com_grid = grid
        self._com_values = values
