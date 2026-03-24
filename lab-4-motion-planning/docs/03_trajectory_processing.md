# Lab 4: Trajectory Processing

## Overview

Lab 4 trajectory processing still has the same two stages:

1. shortcutting a raw planner path
2. time-parameterizing the shortened path

The public API remains:

```python
short = shortcut_path(path, cc, max_iter=200)
times, pos, vel, acc = parameterize_topp_ra(short, VEL_LIMITS, ACC_LIMITS)
```

## Shortcutting

Shortcutting randomly selects two non-adjacent waypoints and removes the
intermediate segment when the straight-line joint-space edge is collision-free.

This stage is unchanged conceptually; the important difference is that the
collision queries now use the canonical executed MuJoCo geometry.

## Time Parameterization

`parameterize_topp_ra(...)` keeps the historical Lab 4 function name.

- If TOPP-RA is available, the function uses it.
- If TOPP-RA is not available in the current environment, the function falls
  back to a conservative quintic time-parameterization that respects the same
  velocity and acceleration limits under the tested Lab 4 scenarios.

This preserves the rest of the pipeline and keeps execution deterministic in
environments where TOPP-RA cannot be compiled.

## Final Validation

- path endpoints are preserved
- shortened paths stay collision-free
- velocity and acceleration limits are respected in the generated trajectory
- the full `tests/test_trajectory.py` suite passes on the canonical stack
