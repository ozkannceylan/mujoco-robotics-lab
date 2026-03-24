# Lab 4: Motion Planning & Collision Avoidance — Completion Report

Completion date: 2026-03-17

## Platform Lock

Lab 4 is completed on:

- MuJoCo Menagerie `universal_robots_ur5e`
- mounted MuJoCo Menagerie `robotiq_2f85`
- MuJoCo-exact executed geometry for collision truth
- Pinocchio matched to the executed stack for FK and gravity terms

## Completed Work

### Phase 0: Platform alignment
- Rebased Lab 4 on the same canonical UR5e + Robotiq stack used by Lab 3
- Reused the canonical Menagerie actuator mapping for executed torque commands

### Phase 1: Collision infrastructure
- Replaced the old separate collision-model truth path with a collision checker built on the executed MuJoCo geometry
- Preserved the Lab 4 collision-checking API (`is_collision_free`, `is_path_free`, `compute_min_distance`)
- Revalidated free/colliding configurations, self-collision, and FK agreement

### Phase 2: RRT / RRT*
- Kept the planner interface and behavior intact on the canonical stack
- Revalidated planning success, collision-free waypoints, edge validity, and deterministic seeded behavior

### Phase 3: Path processing and execution
- Preserved shortcutting
- Preserved `parameterize_topp_ra(...)` and added a conservative quintic fallback for environments where TOPP-RA cannot be built
- Revalidated timed execution with Menagerie actuator mapping and gravity compensation

### Phase 4: Validation media
- Recorded a blocked-path validation video showing the actual MuJoCo scene during execution
- Used a stricter blocked-path capstone obstacle layout for the recorded artifact

## Final Validation

- Full test suite: `45 passed`
- Standard capstone RMS tracking error: `0.0125 rad`
- Standard capstone final position error: `0.0016 rad`
- Blocked-path validation scene direct path free: `False`
- Blocked-path validation scene raw path: `13` waypoints
- Blocked-path validation scene shortcut path: `3` waypoints
- Blocked-path validation scene duration: `1.498 s`
- Blocked-path validation scene RMS tracking error: `0.0124 rad`
- Blocked-path validation scene final error: `0.0041 rad`

## Sign-Off Artifacts

- README: `lab-4-motion-planning/README.md`
- Validation video: `lab-4-motion-planning/media/lab4_validation_real_stack.mp4`
- Validation recorder: `lab-4-motion-planning/src/record_lab4_validation.py`

## Residual Note

The current Python environment cannot build TOPP-RA from source because a system compiler is unavailable. Lab 4 remains validated because the public timing API is preserved and the fallback time-parameterization respects the configured velocity and acceleration limits under the tested scenarios.

## Video Production Overhaul Addendum

### Phase V1: Shared Video Standard
#### Step V1.1: Define the reusable three-phase video API
- Replace the draft shared video helper with a stable `LabVideoProducer` API in `tools/video_producer.py`
- Implement animated metrics generation, MuJoCo simulation capture, and ffmpeg-based composition
- Verify the module stays lab-agnostic and writes H.264 `1920x1080 @ 30 fps` outputs

#### Step V1.2: Add reusable overlay and composition primitives
- Support title/end cards, KPI overlays, animated plot easing, trajectory traces, and configurable cameras
- Keep Lab 4 specifics out of the shared tool
- Verify future labs can call the same API without changing the module internals

### Phase V2: Lab 4 Slalom Demo Refresh
#### Step V2.1: Update the canonical Lab 4 obstacle scene for the slalom layout
- Keep the four staggered tabletop boxes as the default Lab 4 obstacle set
- Ensure collision checking, execution, and rendered scene all use the same obstacle truth
- Verify the direct slalom corridor requires visible weaving with positive clearance

#### Step V2.2: Implement the 8-waypoint slalom planning pipeline
- Build Cartesian slalom waypoints, solve collision-free IK, plan each segment with RRT*, and smooth the full route
- Log tree expansion, path-cost convergence, obstacle clearance, and end-effector velocity metrics
- Verify the final executed trajectory remains collision-free with minimum clearance above `0.03 m`

#### Step V2.3: Build the Lab 4 demo generator
- Demo generator at `src/generate_lab4_demo.py` uses the shared video producer
- All output saved to `media/` (metrics JSON, plot PNGs, video MP4s)
- Verify the final video has title card → animated metrics → slowed simulation → end card

### Phase V3: Validation
#### Step V3.1: Re-test Lab 4 after the video overhaul
- Run the Lab 4 test suite and any new targeted checks for the demo pipeline
- Confirm the produced metrics and video artifacts match the requested structure and naming
- Record any new debugging lessons in `LESSONS.md`
