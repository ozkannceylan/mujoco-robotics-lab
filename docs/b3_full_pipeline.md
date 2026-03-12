# B3: Full Pipeline — Study Notes

## Chain

This module brings the pieces together:

```text
target → IK → trajectory → PD → plant → metrics/log
```

## Demos

`src/b3_full_pipeline.py` includes three demos:

1. Pick and place
   - `(0.20, 0.30) → (0.40, 0.10) → (0.20, 0.30)`
   - Waypoints are converted to joint-space via analytic IK
   - Segments are connected with quintic trajectories
2. Circle tracking
   - A Cartesian circle target is generated
   - IK is solved at every sample
   - Tracked with the PD controller
3. Singularity edge
   - Targets approaching the outer workspace boundary
   - `pinv` and `dls` solvers are compared

## Outputs

- `docs/b3_pick_place_log.csv`
- `docs/b3_circle_log.csv`
- `docs/b3_singularity_log.csv`
- `docs/b3_metrics.csv`

If `matplotlib` and an appropriate writer are available:

- `docs/b3_pipeline_paths.png`
- `docs/b3_pick_place.gif`

## Expected Behaviour

- Pick-and-place should repeat the closed loop without error
- For circle tracking, the actual path should stay close to the desired circle
- Near singularity, `dls` should be more robust than `pinv`
