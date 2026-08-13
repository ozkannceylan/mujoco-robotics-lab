# Lab 2: TODO

**Status: complete.** Lab 2 (UR5e 6-DOF Arm) was built and published before the
per-lab `tasks/` convention existed in CLAUDE.md, so its step list lived in the
repo-root `tasks/todo.md`. This file backfills that record; the checklist below
is the Lab 2 section of the root TODO as it stood at completion (commit
`b5ecdb1`, "adds lab2 and organizes repo structure"), translated to English and
reconciled with what is on disk today.

## Phase A — Kinematics

- [x] Step A1: Environment setup — MuJoCo + Pinocchio load the UR5e, FK cross-validation across 6 configurations (0.000 mm error)
- [x] Step A2: Forward kinematics — DH parameter table, Pinocchio vs MuJoCo comparison
- [x] Step A3: Jacobian — geometric (with real axis detection), Pinocchio analytic, numerical, plus singularity analysis
- [x] Step A4: Inverse kinematics — pseudo-inverse and adaptive DLS solvers, FK round-trip validation
- [x] Step A5: Dynamics — M(q) via CRBA, g(q), C(q,qd), RNEA/ABA, cross-validated against MuJoCo `qfrc_bias`

## Phase B — Trajectories and Control

- [x] Step B1: Trajectory generation — cubic, quintic, trapezoidal, minimum-jerk, multi-segment
- [x] Step B2: Control hierarchy — PD + gravity compensation, computed torque, task-space impedance, OSC
- [x] Step B3: Constraints — joint limits, velocity scaling, torque saturation, self-collision checks

## Phase C — Integration

- [x] Step C1: Full pipeline — pick-and-place and circle-tracking demos with per-step metric logs
- [x] Step C2: ROS 2 bridge — `src/c2_ros2_bridge.py` scaffold plus standalone (no-ROS) demo mode
- [x] Step C3: 3D cube drawing capstone — DLS IK, quintic trajectory, gravity comp + velocity feedforward position control (0.088 mm RMS)

## Deliverables

- [x] Unit tests — 5 files, 34 tests, all passing
- [x] Docs (EN + TR) — A1, A3–C3 written (A2 authored by the engineer directly)
- [x] Video + GIF — `media/c3_draw_cube.mp4` and `media/c3_draw_cube.gif`
- [x] README — lab README plus main README entry

## Maintenance (2026-08-13 project review)

- [x] Recreated the missing MuJoCo scene as a tracked file: `models/lab_scene.xml` (+ `models/assets` symlink so the Menagerie `meshdir="assets"` resolves); `MJCF_SCENE_PATH` now points there instead of into the untracked `models/mujoco_menagerie/` checkout
- [x] Deleted `ros2_bridge/` — it held byte-identical copies of Lab 1's 2-link bridge (wrong robot, 2 joints); the real bridge is `src/c2_ros2_bridge.py`, now the only thing the README advertises
- [x] Fixed the stale `sys.path` bootstrap and the wrong `python3 src/lab-2-.../` run commands in all 12 `src/` scripts
- [x] Rewrote `media/README.md` to describe the files that are actually there
- [x] Re-encoded `media/c3_draw_cube.gif` from 39 MB to 6.4 MB (512x288, 12 fps)
- [x] Replaced the duplicate lab-local plan file with a pointer to `plan/LAB_02.md`
- [x] Backfilled this file

## Current Focus

None — lab complete. Lab 2 is published and portfolio-ready; the whole test
suite (34/34) passes and every demo script runs from the repository root.

## Blockers

None.
