"""Lab 8 — M2 Step 2.1/2.2: gait schedule and swing-foot references.

Produces, for any time `t`, everything the whole-body QP needs to take a step:

* which feet are in contact (the QP's stance set),
* where the swing foot should be, with velocity and acceleration feedforward,
* where the CoM should be.

Scope is deliberately **step-in-place** for M2 (`step_length = 0`); M3 sets a
non-zero stride and reuses this module unchanged.

Why the CoM reference is generated here rather than by Lab 7's LIPM preview
controller
--------------------------------------------------------------------------
Lab 7's `lipm_planner` is validated at the planning level (18 tests) and its
`swing_trajectory` shape is reused. Its `LIPMPreviewController`, however,
converts a *ZMP* reference into a CoM trajectory under the cart-table model —
useful when the controller can only command CoM position. Lab 8's QP already
enforces balance through contact wrenches with a CoP constraint, so what it
needs from the planner is simply "where should the CoM be", and the honest
answer while stepping in place is "over the foot that is about to carry the
robot". Generating that directly keeps one source of truth for the timing
instead of two planners that can disagree about phase boundaries.

The lateral weight shift is the whole difficulty of stepping in place: the CoM
must be over the stance foot *before* the swing foot leaves the ground, or the
robot tips the moment support is removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = ["Phase", "GaitPhase", "GaitReference", "GaitSchedule"]


class Phase(str, Enum):
    """Support phase of the gait cycle."""

    DOUBLE = "double"          # both feet down, weight shifting
    SINGLE_LEFT = "single_left"    # left foot swinging, right in stance
    SINGLE_RIGHT = "single_right"  # right foot swinging, left in stance
    DONE = "done"              # sequence finished, both feet down


@dataclass
class GaitPhase:
    """One entry of the gait timeline."""

    phase: Phase
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def progress(self, t: float) -> float:
        """Normalised progress through this phase, clipped to [0, 1]."""
        if self.duration <= 0.0:
            return 1.0
        return float(np.clip((t - self.t_start) / self.duration, 0.0, 1.0))


@dataclass
class GaitReference:
    """Everything the controller needs at one instant."""

    phase: Phase
    com_target: np.ndarray            # (3,) world; z is left to the posture/knees
    stance_feet: tuple[str, ...]      # frame names in contact
    swing_foot: str | None            # frame name, or None in double support
    swing_position: np.ndarray | None
    swing_velocity: np.ndarray | None
    swing_acceleration: np.ndarray | None
    step_index: int


class GaitSchedule:
    """Timeline of support phases plus the references each phase implies.

    Args:
        left_foot_frame / right_foot_frame: Pinocchio frame names.
        left_foot_home / right_foot_home: Foot positions at t=0 (world, on the
            ground).
        com_home: CoM position at t=0 (world).
        n_steps: Number of swing phases (alternating, starting with the left).
        t_initial: Settling time before the first weight shift [s].
        t_double: Double-support (weight transfer) duration [s].
        t_single: Single-support (swing) duration [s].
        step_length: Forward stride [m]. Zero → stepping in place (M2).
        step_height: Peak swing clearance [m].
        com_shift_ratio: How far toward the stance foot the CoM target moves,
            as a fraction of the distance from the midpoint. 1.0 puts it
            directly over the stance foot.
    """

    def __init__(
        self,
        left_foot_frame: str,
        right_foot_frame: str,
        left_foot_home: np.ndarray,
        right_foot_home: np.ndarray,
        com_home: np.ndarray,
        n_steps: int = 4,
        t_initial: float = 1.5,
        t_double: float = 0.6,
        t_single: float = 0.7,
        step_length: float = 0.0,
        step_height: float = 0.04,
        com_shift_ratio: float = 1.0,
    ) -> None:
        self.left_frame = left_foot_frame
        self.right_frame = right_foot_frame
        self.foot_home = {
            left_foot_frame: np.asarray(left_foot_home, dtype=float).copy(),
            right_foot_frame: np.asarray(right_foot_home, dtype=float).copy(),
        }
        self.com_home = np.asarray(com_home, dtype=float).copy()
        self.n_steps = int(n_steps)
        self.t_initial = float(t_initial)
        self.t_double = float(t_double)
        self.t_single = float(t_single)
        self.step_length = float(step_length)
        self.step_height = float(step_height)
        self.com_shift_ratio = float(com_shift_ratio)

        self._foot_position = {k: v.copy() for k, v in self.foot_home.items()}
        self._phases: list[GaitPhase] = []
        self._swing_of_phase: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
        self._build_timeline()

    # -- timeline ----------------------------------------------------------

    def _build_timeline(self) -> None:
        """Lay out DS → SS → DS → SS … and the landing target of each swing."""
        t = 0.0
        self._phases.append(GaitPhase(Phase.DOUBLE, t, t + self.t_initial))
        t += self.t_initial

        positions = {k: v.copy() for k, v in self.foot_home.items()}
        for step in range(self.n_steps):
            swing = self.left_frame if step % 2 == 0 else self.right_frame
            start = positions[swing].copy()
            landing = start.copy()
            landing[0] += self.step_length

            phase = Phase.SINGLE_LEFT if swing == self.left_frame else Phase.SINGLE_RIGHT
            self._phases.append(GaitPhase(phase, t, t + self.t_single))
            self._swing_of_phase[len(self._phases) - 1] = (swing, start, landing)
            t += self.t_single
            positions[swing] = landing

            # Double support after each swing: transfer weight to the foot that
            # will become stance for the *next* swing.
            self._phases.append(GaitPhase(Phase.DOUBLE, t, t + self.t_double))
            t += self.t_double

        self._phases.append(GaitPhase(Phase.DONE, t, t + self.t_initial))
        self.total_duration = t + self.t_initial

    def _phase_at(self, t: float) -> tuple[int, GaitPhase]:
        for index, phase in enumerate(self._phases):
            if t < phase.t_end or index == len(self._phases) - 1:
                return index, phase
        return len(self._phases) - 1, self._phases[-1]

    def _next_stance_frame(self, index: int) -> str | None:
        """Frame that must carry the robot in the next single-support phase."""
        for later in range(index + 1, len(self._phases)):
            if later in self._swing_of_phase:
                swing, _, _ = self._swing_of_phase[later]
                return self.right_frame if swing == self.left_frame else self.left_frame
        return None

    # -- references --------------------------------------------------------

    def _com_over(self, stance_frame: str | None) -> np.ndarray:
        """CoM target above a stance foot (or the midpoint when both are down)."""
        target = self.com_home.copy()
        if stance_frame is None:
            return target
        midpoint = 0.5 * (self.foot_home[self.left_frame] + self.foot_home[self.right_frame])
        offset = self.foot_home[stance_frame][:2] - midpoint[:2]
        target[:2] = self.com_home[:2] + self.com_shift_ratio * offset
        return target

    def reference(self, t: float) -> GaitReference:
        """Full reference at time `t`."""
        index, phase = self._phase_at(t)
        step_index = sum(1 for i in range(index + 1) if i in self._swing_of_phase)

        if phase.phase in (Phase.SINGLE_LEFT, Phase.SINGLE_RIGHT):
            swing_frame, start, landing = self._swing_of_phase[index]
            stance_frame = (
                self.right_frame if swing_frame == self.left_frame else self.left_frame
            )
            position, velocity, acceleration = self._swing_state(
                t - phase.t_start, phase.duration, start, landing
            )
            return GaitReference(
                phase=phase.phase,
                com_target=self._com_over(stance_frame),
                stance_feet=(stance_frame,),
                swing_foot=swing_frame,
                swing_position=position,
                swing_velocity=velocity,
                swing_acceleration=acceleration,
                step_index=step_index,
            )

        # Double support (or DONE): shift the CoM toward the foot that will be
        # stance next, so the swing can start from a balanced posture.
        next_stance = self._next_stance_frame(index)
        alpha = phase.progress(t)
        # Raised cosine → zero velocity at both ends of the transfer.
        blend = 0.5 * (1.0 - np.cos(np.pi * alpha))
        previous = self._com_target_before(index)
        goal = self._com_over(next_stance)
        return GaitReference(
            phase=phase.phase,
            com_target=previous + blend * (goal - previous),
            stance_feet=(self.left_frame, self.right_frame),
            swing_foot=None,
            swing_position=None,
            swing_velocity=None,
            swing_acceleration=None,
            step_index=step_index,
        )

    def _com_target_before(self, index: int) -> np.ndarray:
        """CoM target held at the end of the previous phase."""
        if index == 0:
            return self.com_home.copy()
        previous = self._phases[index - 1]
        if previous.phase in (Phase.SINGLE_LEFT, Phase.SINGLE_RIGHT):
            swing_frame, _, _ = self._swing_of_phase[index - 1]
            stance = self.right_frame if swing_frame == self.left_frame else self.left_frame
            return self._com_over(stance)
        return self.com_home.copy()

    def _swing_state(
        self,
        t_local: float,
        duration: float,
        start: np.ndarray,
        landing: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Swing foot position/velocity/acceleration.

        Horizontal: raised-cosine (zero velocity at lift-off and touchdown, so
        the foot neither scuffs nor slams). Vertical: a sine arc, which unlike
        Lab 7's parabola also has zero vertical velocity at both ends.

        Feedforward matters here for the same reason it did on M1's hand
        circle (L-M1-b): without ẋ_ref/ẍ_ref the swing foot is chased, and a
        late foot is a missed touchdown.
        """
        if duration <= 0.0:
            return landing.copy(), np.zeros(3), np.zeros(3)

        s = float(np.clip(t_local / duration, 0.0, 1.0))
        w = np.pi / duration

        blend = 0.5 * (1.0 - np.cos(np.pi * s))
        d_blend = 0.5 * w * np.sin(np.pi * s)
        dd_blend = 0.5 * w * w * np.cos(np.pi * s)

        delta = landing - start
        position = start + blend * delta
        velocity = d_blend * delta
        acceleration = dd_blend * delta

        # Vertical arc: h·sin(π s) — peaks at mid-swing, zero at both ends.
        position[2] = start[2] + self.step_height * np.sin(np.pi * s)
        velocity[2] = self.step_height * w * np.cos(np.pi * s)
        acceleration[2] = -self.step_height * w * w * np.sin(np.pi * s)
        return position, velocity, acceleration

    # -- introspection -----------------------------------------------------

    def phases(self) -> list[GaitPhase]:
        """The full timeline (for plots and tests)."""
        return list(self._phases)

    def contact_frames(self, t: float) -> tuple[str, ...]:
        """Stance frames at time `t` — the QP's contact set."""
        return self.reference(t).stance_feet
