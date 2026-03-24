"""Lab 6 — Trajectory synchronization and object-centric frame utilities.

Key abstractions:
  sync_trajectories(): ensures both arm trajectories share the same time axis
    by resampling the shorter one to match the longer duration. This guarantees
    both arms start and finish each motion phase simultaneously.

  ObjectFrame: represents the carry bar's pose in world space and derives each
    arm's EE target from it (the "object-centric" design principle).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lab6_common import LEFT_GRIP_Y_OFFSET, NUM_JOINTS, RIGHT_GRIP_Y_OFFSET


# ---------------------------------------------------------------------------
# Trajectory synchronisation
# ---------------------------------------------------------------------------

def sync_trajectories(
    traj_L: tuple[np.ndarray, np.ndarray, np.ndarray],
    traj_R: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample both arm trajectories onto a common time axis.

    The common duration is max(T_L, T_R).  The shorter trajectory is held at
    its final configuration for the remaining time by appending one extra
    sample at t = T_max.  Both trajectories are then interpolated onto the
    same uniform time grid with the same number of samples.

    Args:
        traj_L: Left arm trajectory (times [N_L], q [N_L, 6], qd [N_L, 6]).
        traj_R: Right arm trajectory (times [N_R], q [N_R, 6], qd [N_R, 6]).

    Returns:
        (times_sync, q_L, qd_L, q_R, qd_R) — all arrays share the same
        time axis; q/qd arrays have shape [N_sync, 6].
    """
    times_L, q_L, qd_L = traj_L
    times_R, q_R, qd_R = traj_R

    T_max = max(times_L[-1], times_R[-1])
    dt    = min(times_L[1] - times_L[0], times_R[1] - times_R[0])
    N_sync = int(T_max / dt) + 1
    t_sync = np.linspace(0.0, T_max, N_sync)

    def _resample(t_orig, q_orig, qd_orig):
        # Clamp extrapolation: hold final config beyond original end
        t_clamped = np.clip(t_sync, 0.0, t_orig[-1])
        q_new  = np.column_stack([
            np.interp(t_clamped, t_orig, q_orig[:, j]) for j in range(NUM_JOINTS)
        ])
        qd_new = np.column_stack([
            np.interp(t_clamped, t_orig, qd_orig[:, j]) for j in range(NUM_JOINTS)
        ])
        # Zero velocity in the held portion
        at_end = t_sync > t_orig[-1]
        qd_new[at_end] = 0.0
        return q_new, qd_new

    q_L_sync,  qd_L_sync  = _resample(times_L, q_L,  qd_L)
    q_R_sync,  qd_R_sync  = _resample(times_R, q_R,  qd_R)

    return t_sync, q_L_sync, qd_L_sync, q_R_sync, qd_R_sync


# ---------------------------------------------------------------------------
# Object-centric frame
# ---------------------------------------------------------------------------

@dataclass
class ObjectFrame:
    """World-frame pose of the cooperative carry bar.

    All arm EE targets are derived from this frame by applying fixed offsets,
    ensuring that when the bar moves both arms update automatically.

    Attributes:
        position: (3,) world position of bar centre.
        rotation: (3,3) world rotation matrix of bar (identity = axis-aligned).
        grip_offset_L: (3,) offset from bar centre to left arm tool0 target.
        grip_offset_R: (3,) offset from bar centre to right arm tool0 target.
    """
    position:      np.ndarray
    rotation:      np.ndarray = None
    grip_offset_L: np.ndarray = None
    grip_offset_R: np.ndarray = None

    def __post_init__(self) -> None:
        if self.rotation is None:
            self.rotation = np.eye(3)
        if self.grip_offset_L is None:
            # tool0 hovers GRIPPER_TIP_OFFSET above the bar surface
            from lab6_common import GRIPPER_TIP_OFFSET
            self.grip_offset_L = np.array([
                0.0, LEFT_GRIP_Y_OFFSET, GRIPPER_TIP_OFFSET
            ])
        if self.grip_offset_R is None:
            from lab6_common import GRIPPER_TIP_OFFSET
            self.grip_offset_R = np.array([
                0.0, RIGHT_GRIP_Y_OFFSET, GRIPPER_TIP_OFFSET
            ])

    def ee_target_left(self) -> np.ndarray:
        """Compute world-frame tool0 target for the left arm.

        Returns:
            Target position (3,) in world frame.
        """
        return self.position + self.rotation @ self.grip_offset_L

    def ee_target_right(self) -> np.ndarray:
        """Compute world-frame tool0 target for the right arm.

        Returns:
            Target position (3,) in world frame.
        """
        return self.position + self.rotation @ self.grip_offset_R

    @classmethod
    def from_bar_pos(cls, bar_pos: np.ndarray) -> "ObjectFrame":
        """Create an ObjectFrame from the bar's centre position.

        Args:
            bar_pos: Bar centre (3,) world position.

        Returns:
            ObjectFrame with default (identity) rotation.
        """
        return cls(position=bar_pos.copy())
