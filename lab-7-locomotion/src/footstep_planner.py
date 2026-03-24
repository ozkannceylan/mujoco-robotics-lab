"""Footstep planner for the G1 humanoid.

Generates footstep sequences, ZMP reference trajectories, and
foot swing trajectories using quintic polynomials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from lab7_common import DT, HIP_WIDTH


@dataclass
class Footstep:
    """A single footstep with position and timing.

    Attributes:
        x: X position of foot center [m].
        y: Y position of foot center [m].
        z: Z position of foot center [m].
        yaw: Foot yaw angle [rad].
        is_left: True if this is a left foot step.
        t_start: Start time of this step [s].
        t_end: End time of this step [s].
    """

    x: float
    y: float
    z: float
    yaw: float
    is_left: bool
    t_start: float
    t_end: float


class FootstepPlanner:
    """Plans footstep sequences and generates ZMP/foot trajectories.

    Generates alternating left-right footstep sequences for forward
    walking, with corresponding ZMP reference trajectories and
    quintic polynomial foot swing paths.
    """

    def __init__(
        self,
        step_length: float = 0.15,
        step_width: float = 2 * HIP_WIDTH,
        step_height: float = 0.04,
        step_duration: float = 0.5,
        double_support_duration: float = 0.2,
        foot_z: float = 0.0,
        dt: float = DT,
    ) -> None:
        """Initialize footstep planner.

        Args:
            step_length: Forward distance per step [m].
            step_width: Lateral distance between feet [m].
            step_height: Maximum swing foot height [m].
            step_duration: Duration of each step (swing phase) [s].
            double_support_duration: Duration of double support phase [s].
            foot_z: Ground height for feet [m].
            dt: Timestep [s].
        """
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height
        self.step_duration = step_duration
        self.double_support_duration = double_support_duration
        self.foot_z = foot_z
        self.dt = dt

    def plan_footsteps(
        self,
        num_steps: int = 10,
        start_x: float = 0.0,
        start_y_left: float = None,
        start_y_right: float = None,
        first_foot: str = "left",
    ) -> List[Footstep]:
        """Plan a sequence of alternating footsteps.

        The first and last steps have half step length for
        smooth start/stop.

        Args:
            num_steps: Number of steps to plan.
            start_x: Starting x position.
            start_y_left: Starting left foot y. Defaults to +step_width/2.
            start_y_right: Starting right foot y. Defaults to -step_width/2.
            first_foot: Which foot takes the first step ("left" or "right").

        Returns:
            List of Footstep objects.
        """
        if start_y_left is None:
            start_y_left = self.step_width / 2
        if start_y_right is None:
            start_y_right = -self.step_width / 2

        steps: List[Footstep] = []
        is_left = first_foot == "left"
        x = start_x
        t = self.double_support_duration  # initial double support

        for i in range(num_steps):
            # First step: half length, last step: half length
            if i == 0:
                sl = self.step_length * 0.5
            elif i == num_steps - 1:
                sl = self.step_length * 0.5
            else:
                sl = self.step_length

            x += sl
            y = start_y_left if is_left else start_y_right

            step = Footstep(
                x=x,
                y=y,
                z=self.foot_z,
                yaw=0.0,
                is_left=is_left,
                t_start=t,
                t_end=t + self.step_duration,
            )
            steps.append(step)

            t += self.step_duration + self.double_support_duration
            is_left = not is_left

        return steps

    def generate_zmp_reference(
        self,
        steps: List[Footstep],
        total_time: float | None = None,
        initial_left: np.ndarray | None = None,
        initial_right: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ZMP reference trajectory from footsteps.

        During single support: ZMP at stance foot center.
        During double support: ZMP linearly transitions between feet.
        Initial phase: ZMP at midpoint between feet.

        Args:
            steps: List of Footstep objects.
            total_time: Total trajectory time. Auto-computed if None.
            initial_left: Initial left foot position [2]. Defaults to [0, step_width/2].
            initial_right: Initial right foot position [2]. Defaults to [0, -step_width/2].

        Returns:
            Tuple of (time [N], zmp_x [N], zmp_y [N]).
        """
        if total_time is None:
            total_time = steps[-1].t_end + self.double_support_duration + 0.5

        N = int(total_time / self.dt)
        t = np.arange(N) * self.dt
        zmp_x = np.zeros(N)
        zmp_y = np.zeros(N)

        if initial_left is None:
            initial_left = np.array([0.0, self.step_width / 2])
        if initial_right is None:
            initial_right = np.array([0.0, -self.step_width / 2])

        # Track current foot positions
        left_pos = initial_left.copy()
        right_pos = initial_right.copy()

        for k in range(N):
            tk = t[k]

            # Find which step phase we're in
            active_step = None
            for step in steps:
                if step.t_start <= tk < step.t_end:
                    active_step = step
                    break

            if active_step is not None:
                # Single support: ZMP at stance foot
                if active_step.is_left:
                    # Left foot is swinging, stance = right
                    zmp_x[k] = right_pos[0]
                    zmp_y[k] = right_pos[1]
                else:
                    # Right foot is swinging, stance = left
                    zmp_x[k] = left_pos[0]
                    zmp_y[k] = left_pos[1]
            else:
                # Double support or initial phase
                # Find previous and next step for interpolation
                prev_step = None
                next_step = None
                for step in steps:
                    if step.t_end <= tk:
                        prev_step = step
                        # Update foot positions
                        if step.is_left:
                            left_pos = np.array([step.x, step.y])
                        else:
                            right_pos = np.array([step.x, step.y])
                    elif step.t_start > tk and next_step is None:
                        next_step = step

                if prev_step is None:
                    # Initial double support: midpoint
                    zmp_x[k] = (left_pos[0] + right_pos[0]) / 2
                    zmp_y[k] = (left_pos[1] + right_pos[1]) / 2
                elif next_step is None:
                    # Final double support: midpoint
                    zmp_x[k] = (left_pos[0] + right_pos[0]) / 2
                    zmp_y[k] = (left_pos[1] + right_pos[1]) / 2
                else:
                    # Transitional double support: interpolate
                    ds_start = prev_step.t_end
                    ds_end = next_step.t_start
                    ds_dur = ds_end - ds_start
                    if ds_dur > 0:
                        alpha = (tk - ds_start) / ds_dur
                    else:
                        alpha = 0.5

                    # From stance foot to next stance foot
                    if next_step.is_left:
                        # Next is left swing, so next stance is right
                        from_x, from_y = left_pos[0], left_pos[1]
                        to_x, to_y = right_pos[0], right_pos[1]
                    else:
                        # Next is right swing, so next stance is left
                        from_x, from_y = right_pos[0], right_pos[1]
                        to_x, to_y = left_pos[0], left_pos[1]

                    zmp_x[k] = from_x + alpha * (to_x - from_x)
                    zmp_y[k] = from_y + alpha * (to_y - from_y)

        return t, zmp_x, zmp_y

    def generate_foot_trajectory(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        duration: float,
        swing_height: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a quintic polynomial foot swing trajectory.

        The foot follows a smooth path from start to end with:
        - Zero velocity at start and end
        - Parabolic height profile peaking at midpoint

        Args:
            start_pos: Start position [3].
            end_pos: End position [3].
            duration: Swing duration [s].
            swing_height: Maximum swing height [m]. Defaults to self.step_height.

        Returns:
            Tuple of (times [M], positions [M, 3]).
        """
        if swing_height is None:
            swing_height = self.step_height

        M = max(int(duration / self.dt), 2)
        times = np.linspace(0, duration, M)
        positions = np.zeros((M, 3))

        for i, ti in enumerate(times):
            s = ti / duration  # normalized phase [0, 1]

            # Quintic polynomial: zero velocity and acceleration at boundaries
            # s(t) = 10*t^3 - 15*t^4 + 6*t^5
            phase = 10 * s**3 - 15 * s**4 + 6 * s**5

            # XY: linear interpolation with quintic timing
            positions[i, 0] = start_pos[0] + phase * (end_pos[0] - start_pos[0])
            positions[i, 1] = start_pos[1] + phase * (end_pos[1] - start_pos[1])

            # Z: parabolic height profile
            # h(s) = 4 * swing_height * s * (1 - s)
            height = 4 * swing_height * s * (1 - s)
            positions[i, 2] = start_pos[2] + phase * (end_pos[2] - start_pos[2]) + height

        return times, positions

    def get_foot_positions_at_time(
        self,
        t: float,
        steps: List[Footstep],
        initial_left: np.ndarray | None = None,
        initial_right: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Get left and right foot positions at a given time.

        Args:
            t: Current time [s].
            steps: List of Footstep objects.
            initial_left: Initial left foot position [3].
            initial_right: Initial right foot position [3].

        Returns:
            Tuple of (left_foot_pos [3], right_foot_pos [3], phase).
            phase is one of: "double_support", "left_swing", "right_swing"
        """
        if initial_left is None:
            initial_left = np.array([0.0, self.step_width / 2, self.foot_z])
        if initial_right is None:
            initial_right = np.array([0.0, -self.step_width / 2, self.foot_z])

        left_pos = initial_left.copy()
        right_pos = initial_right.copy()

        # Update feet from completed steps
        for step in steps:
            if step.t_end <= t:
                if step.is_left:
                    left_pos = np.array([step.x, step.y, step.z])
                else:
                    right_pos = np.array([step.x, step.y, step.z])

        # Check if any step is active
        for step in steps:
            if step.t_start <= t < step.t_end:
                # Active swing
                phase_frac = (t - step.t_start) / (step.t_end - step.t_start)
                s = 10 * phase_frac**3 - 15 * phase_frac**4 + 6 * phase_frac**5
                height = 4 * self.step_height * phase_frac * (1 - phase_frac)

                target = np.array([step.x, step.y, step.z])

                if step.is_left:
                    start = left_pos.copy()
                    swing_pos = start + s * (target - start)
                    swing_pos[2] += height
                    return swing_pos, right_pos, "left_swing"
                else:
                    start = right_pos.copy()
                    swing_pos = start + s * (target - start)
                    swing_pos[2] += height
                    return left_pos, swing_pos, "right_swing"

        return left_pos, right_pos, "double_support"
