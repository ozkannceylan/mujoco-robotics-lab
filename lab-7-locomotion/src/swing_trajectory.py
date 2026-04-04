"""Swing foot trajectory generation for bipedal walking.

During single support, the swing foot moves from its current position
to the target footstep position with:
  - Horizontal (X): smooth interpolation (cubic)
  - Vertical (Z): parabolic bump with configurable peak height
  - Lateral (Y): smooth interpolation (cubic)
"""

from __future__ import annotations

import numpy as np


def _cubic_interp(s: np.ndarray) -> np.ndarray:
    """Smooth start/stop interpolation: 3s^2 - 2s^3."""
    return 3.0 * s**2 - 2.0 * s**3


def generate_swing_trajectory(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    duration: float,
    dt: float,
    step_height: float = 0.03,
) -> np.ndarray:
    """Generate 3D swing foot trajectory from start to end position.

    Args:
        start_pos: Starting foot position (3,) [x, y, z].
        end_pos: Target foot position (3,) [x, y, z].
        duration: Duration of swing phase [s].
        dt: Timestep [s].
        step_height: Maximum vertical clearance [m].

    Returns:
        foot_positions: (N, 3) array of foot positions over the swing.
    """
    N = int(duration / dt)
    s = np.linspace(0.0, 1.0, N)  # phase variable [0, 1]
    alpha = _cubic_interp(s)

    positions = np.zeros((N, 3))

    # Horizontal: cubic interpolation
    positions[:, 0] = start_pos[0] + alpha * (end_pos[0] - start_pos[0])
    positions[:, 1] = start_pos[1] + alpha * (end_pos[1] - start_pos[1])

    # Vertical: parabolic bump (peak at s=0.5)
    z_base = start_pos[2] + alpha * (end_pos[2] - start_pos[2])
    z_bump = 4.0 * step_height * s * (1.0 - s)  # parabola: max at s=0.5
    positions[:, 2] = z_base + z_bump

    return positions
