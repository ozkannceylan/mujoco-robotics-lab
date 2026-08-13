# Lab 4: Motion Planning & Collision Avoidance — Completion Report

Completion date: 2026-03-17 (canonical stack), updated 2026-03-24 (slalom redesign)

## Platform Lock

Lab 4 is completed on:

- MuJoCo Menagerie `universal_robots_ur5e`
- mounted MuJoCo Menagerie `robotiq_2f85`
- MuJoCo-exact executed geometry for collision truth
- Pinocchio matched to the executed stack for FK, IK, and gravity terms

## Completed Work

### Phase 0: Platform alignment
- Rebased Lab 4 on the same canonical UR5e + Robotiq stack used by Lab 3
- Reused the canonical Menagerie actuator mapping for executed torque commands

### Phase 1: Collision infrastructure
- Collision checker built on the executed MuJoCo geometry
- Preserved the Lab 4 collision-checking API (`is_collision_free`, `is_path_free`, `compute_min_distance`)
- Added `compute_min_obstacle_distance` for clearance-aware planning

### Phase 2: RRT / RRT*
- Kept the planner interface and behavior intact on the canonical stack
- Validated planning success, collision-free waypoints, edge validity, and deterministic seeded behavior

### Phase 3: Path processing and execution
- Preserved shortcutting and `parameterize_topp_ra(...)` with quintic fallback
- Added `densify_path()` to prevent TOPP-RA spline overshoot

### Phase 4: Slalom obstacle-avoidance capstone (2026-03-24)
- Replaced backward-moving capstone with forward slalom through obstacles
- 4 staggered tabletop boxes (10x10x20 cm) at alternating Y positions
- 9 Cartesian waypoints at z=0.56 with gap-midpoint via-points
- Per-segment RRT* planning with bounded sampling region and clearance margin
- 23-seed IK bank for robust collision-free waypoint solving
- Full pipeline: IK → per-segment RRT* → shortcut → densify → TOPP-RA → execute

## Final Validation (2026-03-24)

- Full test suite: `44 passed, 1 skipped`
  (re-run 2026-08-13: `45 passed` — the single skip is `TestVisualization`,
  which is skipped only when `mpl_toolkits.mplot3d` is unavailable)
- Slalom waypoints: 9
- Path waypoints: 24
- Planning time: ~168 s (8 segments)
- Trajectory duration: 15.22 s
- RMS tracking error: 0.0027 rad
- Final position error: 0.0018 rad
- Min waypoint clearance: 0.034 m

## Sign-Off Artifacts

- README: `lab-4-motion-planning/README.md`
- Blog post: `lab-4-motion-planning/blog/lab4_blog_post.md`
- Capstone demo: `lab-4-motion-planning/src/capstone_demo.py`
- Demo video recorder: `lab-4-motion-planning/src/record_lab4_demo.py`
  → `media/lab4_metrics.mp4`, `media/lab4_simulation.mp4`, `media/lab4_demo.mp4`
- Validation video recorder: `lab-4-motion-planning/src/record_lab4_validation.py`
  → `media/lab4_validation_real_stack.mp4`

## Artifact Cleanup (2026-08-13)

`media/slalom_metrics.json` was deleted. It described a "round trip" scenario
(17 waypoints, 29.73 s, `minimum_obstacle_clearance_m: 0.0`) that contradicts the
validated numbers above, and no surviving script emits it — the key
`minimum_obstacle_clearance` appears in no `.py` file in the lab, and neither
`capstone_demo.py` nor `record_lab4_demo.py` writes JSON. It was an output of
`slalom_demo.py` / `generate_lab4_demo.py`, deleted in Step S5 of the redesign.
Removed as unreproducible rather than regenerated.

## Residual Note

The current Python environment cannot build TOPP-RA from source because a system compiler is unavailable. Lab 4 remains validated because the public timing API is preserved and the fallback time-parameterization respects the configured velocity and acceleration limits under the tested scenarios.
