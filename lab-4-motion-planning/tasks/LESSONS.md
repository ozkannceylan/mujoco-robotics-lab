# Lab 4: Lessons Learned

## Bugs & Fixes

### 2026-03-17 — Lab 4 had to migrate to the same real stack as Lab 3
**Symptom:** The original Lab 4 implementation was centered on a separate custom collision URDF and a standalone XML scene, not the canonical Menagerie UR5e + Robotiq stack required by the project.
**Root cause:** Planning-time geometry and execution-time geometry diverged.
**Fix:** Rebased Lab 4 on the canonical UR5e + Robotiq stack and moved collision truth onto the executed MuJoCo geometry.
**Takeaway:** For motion planning, the geometry the planner trusts must match the geometry the controller actually executes.

### 2026-03-17 — Direct torque writes were wrong for the Menagerie actuator model
**Symptom:** The old execution path wrote desired torques directly into `mj_data.ctrl[:6]`, which does not match the Menagerie actuator model.
**Root cause:** The execution path still assumed a simplified direct-torque actuator layout.
**Fix:** Reused the Lab 3 actuator mapping helper so desired arm torques are converted into actuator controls before each MuJoCo step.
**Takeaway:** Menagerie actuator semantics matter just as much in planning execution as they do in controller labs.

### 2026-03-17 — Internal Robotiq linkage spacing initially looked like self-collision
**Symptom:** Minimum-distance queries at `Q_HOME` reported tiny negative values even when no meaningful collision existed.
**Root cause:** Internal gripper link proximity inside the same finger mechanism was being treated like a planning self-collision pair.
**Fix:** Excluded gripper-internal self-pairs from the monitored self-collision set while keeping arm-vs-gripper and robot-vs-environment checks active.
**Takeaway:** Real executed geometry is the right collision source, but not every mechanically adjacent pair is useful planning signal.

### 2026-03-17 — TOPP-RA could not be installed in this Python environment
**Symptom:** `toppra` failed to build because the environment has no system compiler.
**Root cause:** The available Python version did not have a ready wheel, and source build prerequisites were unavailable.
**Fix:** Preserved the `parameterize_topp_ra(...)` API and added a conservative quintic fallback that respects the configured velocity and acceleration limits under the tested scenarios.
**Takeaway:** Keep the public pipeline contract stable even when the runtime environment forces an internal fallback.

### 2026-03-17 — The recorded validation scene needed a blocked direct path
**Symptom:** The standard capstone start/goal pair produced a direct collision-free line after shortcutting, which is valid but weak as a sign-off artifact.
**Root cause:** The default comparison scene is good for planner benchmarking but not strict enough to prove obstacle-aware planning in video form.
**Fix:** Added a dedicated blocked-path validation scene (`CAPSTONE_OBSTACLES`) for the recorded artifact.
**Takeaway:** Stable demos and strict validation scenes do not always need to be the same scenario.

### 2026-03-17 — The tabletop validation case needed a new goal and smoother timing
**Symptom:** After moving the added orange blocker onto the tabletop, the old validation goal no longer produced a blocked direct path.
**Root cause:** The prior blocked-path case depended on a geometry layout that no longer intersected the chosen start-to-goal motion once every obstacle was constrained to the tabletop.
**Fix:** Chose a new blocked validation goal on the same tabletop scene and reduced the timing limits used by the recorder so the saved motion is visibly smoother while keeping the obstacle avoidance obvious.
**Takeaway:** If the validation scene changes, the sign-off start/goal pair and timing profile must be revalidated together instead of assuming the old metrics still apply.

### 2026-03-17 — Fallback 2D rendering degraded the validation artifact
**Symptom:** The previous recorder used a 700+ line matplotlib-based drawing fallback that failed to capture the real physics scene and presented confusing layout assumptions.
**Root cause:** Trying to render custom 3D pseudo-projections in matplotlib instead of using the simulator's native renderer (`mujoco.Renderer`).
**Fix:** Removed the entire matplotlib pipeline and rewrote the recorder to strictly use the native headless renderer, exactly matching the proven pipelines from Lab 1 and Lab 2.
**Takeaway:** Never build a complex secondary visualization stack if the native simulator can already record itself. Rely on the simplest native path.

## Final Validation Snapshot

- Full Lab 4 test suite: `44 passed, 1 skipped`
- Blocked-path scene start free: `True`
- Blocked-path scene goal free: `True`
- Blocked-path scene direct path free: `False`
- Blocked-path scene raw path: `35` waypoints
- Blocked-path scene shortcut path: `3` waypoints
- Blocked-path scene duration: `2.659 s`
- Blocked-path scene RMS tracking error: `0.0037 rad`
- Blocked-path scene final error: `0.0004 rad`
