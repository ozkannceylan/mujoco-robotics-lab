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
        step_width: Lateral distance between the feet after the first step [m].
            None → keep each foot on its home line. This is the single biggest
            lever on how hard the gait is to balance: the ZMP has to travel
            from one foot to the other every step, and the lateral DCM swings
            with the same amplitude, so a wide stance buys a large lateral
            excursion that must be arrested inside one foot width. The G1's
            rest stance is 0.237 m; a walking gait wants noticeably less.
        step_height: Peak swing clearance [m].
        close_stance: Append a final step that brings the trailing foot level
            with the leading one, so the gait ends with the feet together.
            Without it a walk ends mid-stride — after M3's twelve steps the
            feet sit 0.09 m apart in x — and *the next* walk therefore starts
            from a staggered stance no milestone ever validated. Measured: a
            second walk diverged right after its first step, DCM error going
            0.4 mm → 138 → 672 in half a second, at low torque (L-M5-e). A real
            walk ends with the feet together; so does this one now.
        first_swing: Frame of the foot that takes the first step. Defaults to
            the left, which is right for a gait starting from a level stance —
            and wrong for one **restarting mid-stride**. After an odd number of
            steps the left foot is the *leading* one, and asking it to step
            again produces a zero-length step (its landing is computed from the
            trailing stance foot, which is exactly where it already stands)
            while the DCM plan still expects the body to advance. Measured: a
            second walk fell every time, at low torque, with the DCM diverging
            to 218 mm RMS (L-M5-e). Callers that resume a gait should pass the
            trailing foot.
        com_shift_ratio: How far toward the stance foot the CoM target moves
            **laterally**, as a fraction of the distance from the foot
            midpoint. 1.0 puts it directly over the stance foot.
        com_forward_shift_ratio: The same bias along the walking direction.
            Defaults to 0 — the CoM tracks the midpoint forward and only
            leans sideways. See `_com_over` for why the two axes differ.
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
        step_width: float | None = None,
        step_height: float = 0.04,
        first_swing: str | None = None,
        close_stance: bool = False,
        com_shift_ratio: float = 1.0,
        com_forward_shift_ratio: float = 0.0,
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
        self.step_width = None if step_width is None else float(step_width)
        self.step_height = float(step_height)
        if first_swing is not None and first_swing not in (left_foot_frame, right_foot_frame):
            raise ValueError(f"first_swing must be a foot frame, got '{first_swing}'")
        self.first_swing = first_swing or left_foot_frame
        self.close_stance = bool(close_stance)
        self.com_shift_ratio = float(com_shift_ratio)
        self.com_forward_shift_ratio = float(com_forward_shift_ratio)

        self._home_midpoint = 0.5 * (
            self.foot_home[left_foot_frame] + self.foot_home[right_foot_frame]
        )
        self._phases: list[GaitPhase] = []
        self._swing_of_phase: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
        # Foot positions in force during each phase — the CoM reference has to
        # follow the feet as they advance, not the t=0 home poses.
        self._foot_state: dict[int, dict[str, np.ndarray]] = {}
        self._build_timeline()

    # -- timeline ----------------------------------------------------------

    def _build_timeline(self) -> None:
        """Lay out DS → SS → DS → SS …, the landing of each swing, and the
        foot positions in force during every phase.

        Footstep placement puts the swing foot `step_length` **ahead of the
        stance foot**, not ahead of its own previous position. The difference
        matters: stepping ahead of your own footprint advances the body by
        `step_length/2` per step (the feet never pass each other), which is a
        shuffle. Passing the stance foot advances it by a full `step_length`,
        which is a walk — and is what Lab 7's `generate_footsteps` does too.
        """
        t = 0.0
        positions = {k: v.copy() for k, v in self.foot_home.items()}

        self._phases.append(GaitPhase(Phase.DOUBLE, t, t + self.t_initial))
        self._foot_state[0] = {k: v.copy() for k, v in positions.items()}
        t += self.t_initial

        for step in range(self.n_steps):
            other = (
                self.right_frame if self.first_swing == self.left_frame else self.left_frame
            )
            swing = self.first_swing if step % 2 == 0 else other
            stance = self.right_frame if swing == self.left_frame else self.left_frame
            start = positions[swing].copy()
            landing = start.copy()
            landing[0] = positions[stance][0] + self.step_length
            if self.step_width is not None:
                # Place the foot beside the stance foot at the requested
                # separation, keeping it on its own side of the body.
                side = 1.0 if swing == self.left_frame else -1.0
                landing[1] = positions[stance][1] + side * self.step_width

            phase = Phase.SINGLE_LEFT if swing == self.left_frame else Phase.SINGLE_RIGHT
            self._phases.append(GaitPhase(phase, t, t + self.t_single))
            index = len(self._phases) - 1
            self._swing_of_phase[index] = (swing, start, landing)
            # During the swing the stance foot is the only support, and the
            # swing foot is in flight; record where it *will* land so the CoM
            # reference for the following double support is already correct.
            self._foot_state[index] = {k: v.copy() for k, v in positions.items()}
            t += self.t_single
            positions[swing] = landing

            # Double support after each swing: transfer weight to the foot that
            # will become stance for the *next* swing.
            self._phases.append(GaitPhase(Phase.DOUBLE, t, t + self.t_double))
            self._foot_state[len(self._phases) - 1] = {
                k: v.copy() for k, v in positions.items()
            }
            t += self.t_double

        if self.close_stance and self.n_steps > 0:
            # One extra swing: the trailing foot steps up **level** with the
            # leading one rather than past it, leaving a symmetric stance.
            leading, trailing = (
                (self.left_frame, self.right_frame)
                if positions[self.left_frame][0] > positions[self.right_frame][0]
                else (self.right_frame, self.left_frame)
            )
            if abs(positions[leading][0] - positions[trailing][0]) > 1e-4:
                start = positions[trailing].copy()
                landing = start.copy()
                landing[0] = positions[leading][0]
                phase = (
                    Phase.SINGLE_LEFT if trailing == self.left_frame
                    else Phase.SINGLE_RIGHT
                )
                self._phases.append(GaitPhase(phase, t, t + self.t_single))
                index = len(self._phases) - 1
                self._swing_of_phase[index] = (trailing, start, landing)
                self._foot_state[index] = {k: v.copy() for k, v in positions.items()}
                t += self.t_single
                positions[trailing] = landing

                self._phases.append(GaitPhase(Phase.DOUBLE, t, t + self.t_double))
                self._foot_state[len(self._phases) - 1] = {
                    k: v.copy() for k, v in positions.items()
                }
                t += self.t_double

        self._phases.append(GaitPhase(Phase.DONE, t, t + self.t_initial))
        self._foot_state[len(self._phases) - 1] = {k: v.copy() for k, v in positions.items()}
        self.total_duration = t + self.t_initial
        self.final_foot_positions = {k: v.copy() for k, v in positions.items()}
        final_midpoint = 0.5 * (
            positions[self.left_frame] + positions[self.right_frame]
        )
        self.total_advance = float(final_midpoint[0] - self._home_midpoint[0])

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

    def _com_over(self, stance_frame: str | None, index: int) -> np.ndarray:
        """CoM target for phase `index`, biased toward `stance_frame`.

        Uses the foot positions in force during that phase, so the reference
        travels with the gait instead of being pinned to the t=0 stance.

        Lateral and forward bias are **separate**, and that separation is what
        makes forward walking possible at all (L-M3-a). Biasing both axes
        toward the stance foot is right when stepping in place, where the feet
        are side by side. Once the feet straddle a stride the same rule aims
        the CoM at a foot that is also half a step *ahead*, so the robot leans
        out over a diagonal support polygon and topples — measured as a fall
        during the first transfer of every forward-walking attempt.

        Sideways the CoM still has to get over the stance foot (that is the
        whole point of the transfer). Forward it should stay near the midpoint
        and simply flow with the feet, which is what a walk actually looks
        like: the body advances steadily while support alternates beneath it.
        """
        feet = self._foot_state.get(index, self.foot_home)
        midpoint = 0.5 * (feet[self.left_frame] + feet[self.right_frame])

        target = self.com_home.copy()
        target[0] = self.com_home[0] + (midpoint[0] - self._home_midpoint[0])
        target[1] = self.com_home[1] + (midpoint[1] - self._home_midpoint[1])
        if stance_frame is not None:
            offset = feet[stance_frame][:2] - midpoint[:2]
            target[0] += self.com_forward_shift_ratio * offset[0]
            target[1] += self.com_shift_ratio * offset[1]
        return target

    def forward_progress(self, t: float) -> float:
        """Commanded forward CoM displacement at time `t` [m].

        Advance is a **continuous ramp over the whole walking interval**, not a
        per-phase quantity. Deriving it from the current foot midpoint instead
        (the natural-looking choice) freezes the CoM through every single
        support and then jumps it during double support: the body stops dead
        while a foot is in the air, and the robot walks out from under itself.
        Measured as a backwards fall on the second step — the CoM ended up
        0.11 m *behind* where it started while the feet had moved forward
        (L-M3-b).

        A steady ramp is also simply what walking is: support alternates
        underneath a body that keeps moving.
        """
        if self.total_advance <= 0.0:
            return 0.0
        start = self.t_initial
        end = self.total_duration - self.t_initial
        if t <= start:
            return 0.0
        if t >= end:
            return self.total_advance
        alpha = (t - start) / (end - start)
        return self.total_advance * alpha

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
            com_target = self._com_over(stance_frame, index)
            com_target[0] = self.com_home[0] + self.forward_progress(t)
            return GaitReference(
                phase=phase.phase,
                com_target=com_target,
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
        goal = self._com_over(next_stance, index)
        com_target = previous + blend * (goal - previous)
        com_target[0] = self.com_home[0] + self.forward_progress(t)
        return GaitReference(
            phase=phase.phase,
            com_target=com_target,
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
            return self._com_over(stance, index - 1)
        return self._com_over(None, index - 1)

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

    def foot_positions(self, index: int) -> dict[str, np.ndarray]:
        """Foot positions in force during phase `index` (world).

        During a single-support phase the swing foot is reported at its
        *lift-off* position; it is only moved to the landing once the following
        double support begins.
        """
        feet = self._foot_state.get(index, self.foot_home)
        return {k: v.copy() for k, v in feet.items()}

    def swing_frame_of_phase(self, index: int) -> str | None:
        """Swing foot of phase `index`, or None if it is a support phase."""
        entry = self._swing_of_phase.get(index)
        return None if entry is None else entry[0]

    def stance_frame_of_phase(self, index: int) -> str | None:
        """Sole stance foot of phase `index`, or None in double support."""
        swing = self.swing_frame_of_phase(index)
        if swing is None:
            return None
        return self.right_frame if swing == self.left_frame else self.left_frame

    def next_stance_frame(self, index: int) -> str | None:
        """Foot that carries the robot in the next single-support phase."""
        return self._next_stance_frame(index)

    def phase_index_at(self, t: float) -> int:
        """Index into `phases()` of the phase active at time `t`."""
        return self._phase_at(t)[0]

    def contact_frames(self, t: float) -> tuple[str, ...]:
        """Stance frames at time `t` — the QP's contact set."""
        return self.reference(t).stance_feet
