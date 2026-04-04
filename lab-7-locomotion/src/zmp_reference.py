"""ZMP Reference Generator for bipedal walking.

Given a footstep plan (list of foot positions + timings), generates the
piecewise-constant/linear ZMP reference trajectory:
  - During single support: ZMP = stance foot center
  - During double support: ZMP linearly interpolates from previous to next stance

Footstep plan convention:
  Each footstep = (x, y, foot_id) where foot_id is 'L' or 'R'.
  The plan starts and ends with double support phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Footstep:
    """A single footstep in the walking plan."""

    x: float       # foot center X [m]
    y: float       # foot center Y [m]
    foot: str      # 'L' or 'R'


@dataclass
class WalkingTiming:
    """Timing parameters for the walking gait."""

    t_ss: float = 0.8     # single support duration [s]
    t_ds: float = 0.2     # double support duration [s]
    t_init: float = 1.0   # initial double support (shift to first stance foot) [s]
    t_final: float = 1.0  # final double support (center CoM between feet) [s]


# ---------------------------------------------------------------------------
# Footstep plan generation
# ---------------------------------------------------------------------------


def generate_footstep_plan(
    n_steps: int = 12,
    stride_x: float = 0.08,
    foot_y: float = 0.075,
    start_foot: str = "R",
) -> list[Footstep]:
    """Generate an alternating footstep plan for forward walking.

    The first step shifts weight to `start_foot` (stationary), then
    the opposite foot takes the first forward step.

    Args:
        n_steps: Number of forward steps.
        stride_x: Forward stride per step [m].
        foot_y: Lateral half-spacing between feet [m].
        start_foot: Which foot to stand on first ('L' or 'R').

    Returns:
        List of Footstep objects. Length = n_steps + 2
        (first stance + n_steps forward + final alignment step).
    """
    steps: list[Footstep] = []

    # Sign convention: R foot at -foot_y, L foot at +foot_y
    y_map = {"L": +foot_y, "R": -foot_y}
    other = {"L": "R", "R": "L"}

    # Step 0: initial stance foot (no forward motion)
    steps.append(Footstep(x=0.0, y=y_map[start_foot], foot=start_foot))

    # Alternating steps
    current_foot = other[start_foot]
    for i in range(1, n_steps + 1):
        steps.append(Footstep(
            x=i * stride_x,
            y=y_map[current_foot],
            foot=current_foot,
        ))
        current_foot = other[current_foot]

    # Final alignment step: bring the trailing foot next to the lead foot
    last = steps[-1]
    align_foot = other[last.foot]
    steps.append(Footstep(
        x=last.x,
        y=y_map[align_foot],
        foot=align_foot,
    ))

    return steps


# ---------------------------------------------------------------------------
# ZMP reference generation
# ---------------------------------------------------------------------------


def generate_zmp_reference(
    footsteps: list[Footstep],
    timing: WalkingTiming,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate ZMP reference trajectory from footstep plan.

    Timeline:
      [0, t_init): DS — ZMP moves from midpoint of first two feet to first stance
      For each step i (1..n_steps):
        [t_start, t_start+t_ss): SS — ZMP at stance foot center
        [t_start+t_ss, t_start+t_ss+t_ds): DS — ZMP interpolates to next stance
      Final SS on last-but-one foot, then t_final DS to midpoint.

    Args:
        footsteps: List of Footstep objects from generate_footstep_plan().
        timing: Gait timing parameters.
        dt: Timestep [s].

    Returns:
        (zmp_ref_x, zmp_ref_y, t, total_time): ZMP references and time array.
    """
    n_steps = len(footsteps) - 2  # exclude initial stance and final alignment

    # Total time
    total_time = (
        timing.t_init
        + n_steps * (timing.t_ss + timing.t_ds)
        + timing.t_ss  # final SS on last stepping foot
        + timing.t_final  # final DS to center
    )

    N = int(total_time / dt)
    t = np.arange(N) * dt
    zmp_x = np.zeros(N)
    zmp_y = np.zeros(N)

    # Initial midpoint (between first two feet)
    mid_x = (footsteps[0].x + footsteps[1].x) / 2.0
    mid_y = (footsteps[0].y + footsteps[1].y) / 2.0

    # Phase 1: Initial DS — interpolate from midpoint to first stance foot
    mask = t < timing.t_init
    if np.any(mask):
        alpha = t[mask] / timing.t_init
        zmp_x[mask] = mid_x + alpha * (footsteps[0].x - mid_x)
        zmp_y[mask] = mid_y + alpha * (footsteps[0].y - mid_y)

    # Phase 2: Alternating SS + DS for each forward step
    for i in range(n_steps):
        # Stance foot for this SS phase = footsteps[i]
        # Next stance foot = footsteps[i+1]
        t_start = timing.t_init + i * (timing.t_ss + timing.t_ds)

        # SS: ZMP at current stance foot
        ss_mask = (t >= t_start) & (t < t_start + timing.t_ss)
        zmp_x[ss_mask] = footsteps[i].x
        zmp_y[ss_mask] = footsteps[i].y

        # DS: ZMP interpolates from current to next stance
        ds_start = t_start + timing.t_ss
        ds_mask = (t >= ds_start) & (t < ds_start + timing.t_ds)
        if np.any(ds_mask):
            alpha = (t[ds_mask] - ds_start) / timing.t_ds
            zmp_x[ds_mask] = footsteps[i].x + alpha * (footsteps[i + 1].x - footsteps[i].x)
            zmp_y[ds_mask] = footsteps[i].y + alpha * (footsteps[i + 1].y - footsteps[i].y)

    # Phase 3: Final SS on last stepping foot (footsteps[-2])
    t_final_ss_start = timing.t_init + n_steps * (timing.t_ss + timing.t_ds)
    final_ss_mask = (t >= t_final_ss_start) & (t < t_final_ss_start + timing.t_ss)
    zmp_x[final_ss_mask] = footsteps[-2].x
    zmp_y[final_ss_mask] = footsteps[-2].y

    # Phase 4: Final DS — interpolate from last stepping foot to midpoint of last two feet
    final_mid_x = (footsteps[-2].x + footsteps[-1].x) / 2.0
    final_mid_y = (footsteps[-2].y + footsteps[-1].y) / 2.0

    t_final_ds_start = t_final_ss_start + timing.t_ss
    final_ds_mask = t >= t_final_ds_start
    if np.any(final_ds_mask):
        alpha = np.clip((t[final_ds_mask] - t_final_ds_start) / timing.t_final, 0.0, 1.0)
        zmp_x[final_ds_mask] = footsteps[-2].x + alpha * (final_mid_x - footsteps[-2].x)
        zmp_y[final_ds_mask] = footsteps[-2].y + alpha * (final_mid_y - footsteps[-2].y)

    return zmp_x, zmp_y, t, total_time


# ---------------------------------------------------------------------------
# Foot phase lookup (which foot is swinging at time t)
# ---------------------------------------------------------------------------


def get_phase_at_time(
    t_query: float,
    footsteps: list[Footstep],
    timing: WalkingTiming,
) -> tuple[str, int, float]:
    """Determine walking phase at a given time.

    Args:
        t_query: Time [s].
        footsteps: Footstep plan.
        timing: Gait timing.

    Returns:
        (phase, step_idx, t_in_phase):
          phase: 'init_ds', 'ss', 'ds', 'final_ss', 'final_ds'
          step_idx: Index into footsteps for current stance
          t_in_phase: Time elapsed within current phase
    """
    n_steps = len(footsteps) - 2

    if t_query < timing.t_init:
        return "init_ds", 0, t_query

    t_walk = t_query - timing.t_init

    for i in range(n_steps):
        t_step = timing.t_ss + timing.t_ds
        t_step_start = i * t_step

        if t_walk < t_step_start + timing.t_ss:
            return "ss", i, t_walk - t_step_start

        if t_walk < t_step_start + t_step:
            return "ds", i, t_walk - t_step_start - timing.t_ss

    # Final SS
    t_final_ss_start = n_steps * (timing.t_ss + timing.t_ds)
    if t_walk < t_final_ss_start + timing.t_ss:
        return "final_ss", n_steps, t_walk - t_final_ss_start

    # Final DS
    return "final_ds", n_steps, t_walk - t_final_ss_start - timing.t_ss
