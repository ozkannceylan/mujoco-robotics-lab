# Lab 4: Motion Planning & Collision Avoidance

Status: complete and signed off on 2026-03-17.

Lab 4 now runs on the canonical project stack:

- MuJoCo Menagerie `universal_robots_ur5e`
- mounted MuJoCo Menagerie `robotiq_2f85`
- exact executed MuJoCo collision geometry for planning-time collision truth
- Pinocchio retained for FK, visualization support, and gravity compensation during execution

## What Was Validated

- Collision checking on the real executed UR5e + Robotiq geometry
- RRT and RRT* planning on the canonical stack
- Path shortcutting and time parameterization
- Joint-space execution with gravity compensation through the Menagerie actuator model
- A blocked-path tabletop validation scene recorded to video

## Final Validation Summary

- Full Lab 4 test suite: `44 passed, 1 skipped`
- Standard capstone execution RMS tracking error: `0.0125 rad`
- Standard capstone final position error: `0.0016 rad`
- Blocked-path validation scene: start free `True`, goal free `True`, direct path free `False`
- Blocked-path validation scene raw path: `35` waypoints
- Blocked-path validation scene shortcut path: `3` waypoints
- Blocked-path validation scene raw cost: `9.895`
- Blocked-path validation scene shortcut cost: `7.873`
- Blocked-path validation scene trajectory duration: `2.659 s`
- Blocked-path validation scene RMS tracking error: `0.0037 rad`
- Blocked-path validation scene final position error: `0.0004 rad`

## Canonical Artifacts

- Validation video: `media/lab4_validation_real_stack.mp4`
- Capstone plots: `media/capstone_rrt_tree.png`, `media/capstone_rrt_star_tree.png`, `media/capstone_execution_comparison.png`, `media/capstone_ee_trajectory.png`
- Planning plots: `media/rrt_plan.png`, `media/rrt_star_plan.png`
- Path-processing plots: `media/raw_vs_smooth_comparison.png`

## Architecture

| Module | Role |
|------|------|
| `src/lab4_common.py` | Canonical scene/model loading, obstacle specs, actuator helpers |
| `src/collision_checker.py` | Collision queries on the executed MuJoCo geometry |
| `src/rrt_planner.py` | RRT / RRT* in joint space |
| `src/trajectory_smoother.py` | Shortcutting + time parameterization |
| `src/trajectory_executor.py` | Joint-space PD + gravity compensation execution |
| `src/capstone_demo.py` | Standard full-pipeline demo |
| `src/record_lab4_validation.py` | Blocked-path validation video recorder |

## Important Implementation Notes

- The planner no longer reasons over a separate custom collision URDF as the source of truth. Collision truth comes from the same MuJoCo robot and obstacle geometry that execution uses.
- Pinocchio is still used where it adds value: FK utilities and gravity compensation in the execution controller.
- `parameterize_topp_ra(...)` keeps the same public API. It uses TOPP-RA when available and falls back to a conservative quintic time-parameterization when TOPP-RA is not installable in the current Python environment.
- Trajectory execution no longer writes torques directly into `mj_data.ctrl[:6]`. Desired arm torques are mapped through the Menagerie actuator model first, matching the real stack used in Lab 3.
- The recorded validation artifact now uses a tabletop-only obstacle layout and a slower, smoother timing profile.
- Validation video is recorded using native `mujoco.Renderer` exclusively, matching the Lab 1 and Lab 2 recording pipelines exactly. No matplotlib 2D projection or 3D replay fallbacks are used.

## How To Run

```bash
python3 -m pytest lab-4-motion-planning/tests -q
python3 lab-4-motion-planning/src/capstone_demo.py
python3 lab-4-motion-planning/src/record_lab4_validation.py
```

## Notes

- The standard capstone demo keeps the original comparison scenario for stable RRT vs RRT* reporting.
- The recorded validation video uses a stricter blocked-path scene (`CAPSTONE_OBSTACLES`) with every obstacle on the table, so the saved artifact proves tabletop obstacle avoidance instead of a trivial straight-line path.
- Lab 4 is complete on the canonical real stack. Lab 5 is the next remaining end-to-end target.
