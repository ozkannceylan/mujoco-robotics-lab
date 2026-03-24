"""Contact management for whole-body locomotion.

Defines contact states, schedules, and friction cone constraints
for the QP-based whole-body controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin


class ContactState(Enum):
    """Contact state for a foot."""
    STANCE = "stance"
    SWING = "swing"


@dataclass
class ContactInfo:
    """Information about a single contact point.

    Attributes:
        frame_name: Name of the contact frame.
        state: Current contact state (STANCE/SWING).
        pose: SE3 pose of the contact point (world frame).
        friction_coeff: Friction coefficient.
        normal: Surface normal direction [3].
    """
    frame_name: str
    state: ContactState
    pose: pin.SE3 = field(default_factory=lambda: pin.SE3.Identity())
    friction_coeff: float = 0.8
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def is_active(self) -> bool:
        """Check if this contact is in stance (active).

        Returns:
            True if contact is in STANCE.
        """
        return self.state == ContactState.STANCE


class ContactSchedule:
    """Time-indexed contact schedule for both feet.

    Manages which feet are in contact at each time instant.

    Attributes:
        dt: Time step.
        left_foot_frame: Frame name for left foot.
        right_foot_frame: Frame name for right foot.
    """

    def __init__(
        self,
        dt: float,
        left_foot_frame: str = "left_ankle_roll_link",
        right_foot_frame: str = "right_ankle_roll_link",
    ) -> None:
        """Initialize contact schedule.

        Args:
            dt: Time step [s].
            left_foot_frame: Left foot frame name.
            right_foot_frame: Right foot frame name.
        """
        self.dt = dt
        self.left_foot_frame = left_foot_frame
        self.right_foot_frame = right_foot_frame
        self._schedule: List[Dict[str, ContactState]] = []

    def add_phase(
        self,
        duration: float,
        left_state: ContactState,
        right_state: ContactState,
    ) -> None:
        """Add a contact phase to the schedule.

        Args:
            duration: Duration of this phase [s].
            left_state: Contact state for left foot.
            right_state: Contact state for right foot.
        """
        n_steps = max(1, int(duration / self.dt))
        for _ in range(n_steps):
            self._schedule.append({
                self.left_foot_frame: left_state,
                self.right_foot_frame: right_state,
            })

    def get_contacts_at(self, step: int) -> Dict[str, ContactState]:
        """Get contact states at a given time step.

        Args:
            step: Time step index.

        Returns:
            Dictionary mapping foot frame names to contact states.
        """
        if step >= len(self._schedule):
            # After schedule ends, assume double support
            return {
                self.left_foot_frame: ContactState.STANCE,
                self.right_foot_frame: ContactState.STANCE,
            }
        return self._schedule[step]

    def get_active_contacts_at(self, step: int) -> List[str]:
        """Get list of frame names in STANCE at a given step.

        Args:
            step: Time step index.

        Returns:
            List of frame names with active contacts.
        """
        contacts = self.get_contacts_at(step)
        return [name for name, state in contacts.items()
                if state == ContactState.STANCE]

    @property
    def total_steps(self) -> int:
        """Total number of time steps in the schedule.

        Returns:
            Number of steps.
        """
        return len(self._schedule)

    @property
    def total_duration(self) -> float:
        """Total duration of the schedule [s].

        Returns:
            Duration in seconds.
        """
        return len(self._schedule) * self.dt

    @staticmethod
    def build_double_support(
        dt: float,
        duration: float,
    ) -> "ContactSchedule":
        """Build a schedule with permanent double support.

        Args:
            dt: Time step.
            duration: Total duration.

        Returns:
            ContactSchedule with double support.
        """
        schedule = ContactSchedule(dt)
        schedule.add_phase(duration, ContactState.STANCE, ContactState.STANCE)
        return schedule

    @staticmethod
    def build_walking_schedule(
        dt: float,
        n_steps: int = 6,
        step_duration: float = 0.5,
        double_support_duration: float = 0.2,
        start_foot: str = "left",
    ) -> "ContactSchedule":
        """Build an alternating walk schedule.

        Args:
            dt: Time step.
            n_steps: Number of walking steps.
            step_duration: Duration of each single-support phase.
            double_support_duration: Duration of each double-support transition.
            start_foot: Which foot swings first ("left" or "right").

        Returns:
            ContactSchedule for walking.
        """
        schedule = ContactSchedule(dt)
        # Initial double support
        schedule.add_phase(double_support_duration, ContactState.STANCE, ContactState.STANCE)

        swing_left = (start_foot == "left")
        for _ in range(n_steps):
            if swing_left:
                schedule.add_phase(step_duration, ContactState.SWING, ContactState.STANCE)
            else:
                schedule.add_phase(step_duration, ContactState.STANCE, ContactState.SWING)
            # Double support transition
            schedule.add_phase(double_support_duration, ContactState.STANCE, ContactState.STANCE)
            swing_left = not swing_left

        return schedule


def build_friction_cone_constraints(
    mu: float = 0.8,
    n_facets: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build linearized friction cone inequality constraints.

    The friction cone is linearized with n_facets planes:
      A @ f <= 0  (where f = [fx, fy, fz])

    Also enforces fz >= f_min (small positive force).

    Args:
        mu: Friction coefficient.
        n_facets: Number of facets for linearization.

    Returns:
        Tuple of (A [n_facets+1, 3], b [n_facets+1]):
            A @ f <= b encodes the friction cone.
    """
    # Linearized friction cone: for each facet direction
    # cos(theta_i)*fx + sin(theta_i)*fy - mu*fz <= 0
    angles = np.linspace(0, 2 * np.pi, n_facets, endpoint=False)

    # n_facets rows for friction + 1 row for minimum normal force
    A = np.zeros((n_facets + 1, 3))
    b = np.zeros(n_facets + 1)

    for i, theta in enumerate(angles):
        A[i, 0] = np.cos(theta)   # fx coefficient
        A[i, 1] = np.sin(theta)   # fy coefficient
        A[i, 2] = -mu             # -mu * fz

    # Minimum normal force: -fz <= -f_min  =>  fz >= f_min
    f_min = 1.0  # minimum 1N normal force
    A[n_facets, 2] = -1.0
    b[n_facets] = -f_min

    return A, b
