# Lab 1 — TODO

**Status:** complete (published, portfolio-ready)
**Brief:** [`plan/LAB_01.md`](../../plan/LAB_01.md)
**Board backfilled:** 2026-08-13 — Lab 1 predates the per-lab `tasks/` convention, so
this file was reconstructed from the completed checklist in the root `tasks/todo.md`
(commit `b5ecdb1`) plus the on-disk evidence in `src/`, `docs/`, and `media/`.

## Current Focus

**None — lab complete.** Nothing is in flight. If Lab 1 is reopened, start from the
Backlog section below.

## Blockers

**None.**

---

## A — Foundations

- [x] **A1** MuJoCo environment setup — model loading, `mj_step()`, actuator experiments
      (`src/a1_mujoco_setup.py`, `src/a1_interactive_demo.py`, `src/a1_torque_demo.py`)
- [x] **A2** Forward kinematics — geometric + homogeneous FK, workspace analysis, MuJoCo
      `site_xpos` cross-validation (`src/a2_forward_kinematics.py`)
- [x] **A3** Jacobian — analytic vs. finite-difference vs. `mj_jacSite`, determinant sweep
      and singularity analysis; worst analytic/numeric diff < 1e-10 (`src/a3_jacobian.py`)
- [x] **A4** Inverse kinematics — two-branch analytic IK plus pseudo-inverse and DLS
      solvers, 20-target benchmark at 100% success (`src/a4_inverse_kinematics.py`)
- [x] **A5** Dynamics observation — `qM` (mass matrix) and `qfrc_bias` access from MuJoCo
      (`src/a5_dynamics_basics.py`)

## B — Control and Trajectory

- [x] **B1** Trajectory generation — cubic and quintic joint-space profiles, Cartesian
      straight-line path with per-step IK (`src/b1_trajectory_generation.py`)
- [x] **B2** PD control — joint PD with and without gravity compensation, step response
      and tracking comparison (`src/b2_pd_controller.py`)
- [x] **B3** Full pipeline demos — pick-and-place, circle tracking, singularity case
      (`src/b3_full_pipeline.py`)
- [x] **B4** Optional ROS 2 bridge skeleton — `/joint_command`, `/joint_state`, `/ee_pose`
      (`ros2_bridge/mujoco_bridge.py`, `ros2_bridge/commander.py`)

## C — Integration

- [x] **C1** Cartesian square drawing — quintic Cartesian trajectory + analytic IK +
      Jacobian velocity mapping + computed torque control, with MuJoCo viewer trail.
      Result: 0.008 mm RMS / 0.013 mm max tracking error at 0.076 Nm peak torque
      (`src/c1_draw_square.py`)
- [x] **C1** Headless video/GIF capture of the capstone demo (`src/c1_record_video.py`,
      `media/c1_draw_square.mp4`, `media/c1_draw_square.gif`)

## Documentation and Housekeeping

- [x] English module notes in `docs/` (A1–C1) with CSV artifacts
- [x] Turkish module notes in `docs-turkish/` (A1–C1)
- [x] Five-part blog series in `blog/` with index (`blog/README.md`)
- [x] Lab README with showcase GIF, results table, module tables, structure block
- [x] Root README entry for Lab 1
- [x] **Unit test suite** — added 2026-08-13. `tests/test_kinematics.py` (15 tests: FK
      known configurations, homogeneous-vs-geometric agreement, Jacobian vs. finite
      differences, singularity determinant, IK roundtrip and unreachable-target
      handling, numeric IK convergence for pinv/DLS) and `tests/test_control.py`
      (11 tests: cubic/quintic boundary conditions, trajectory sampling, Cartesian
      line adherence, PD sign/saturation, gravity-compensation error reduction).
      26 tests, all passing in ~1.5 s, no MuJoCo/rendering:
      `python3 -m pytest lab-1-2link-arm/tests/ -q`
- [x] **Duplicate plan file resolved** — 2026-08-13. `PROJECT-PLAN-lab-1-2link-arm.md`
      was byte-identical to `plan/LAB_01.md`; the lab-local copy is now a pointer note.
- [x] **Off-topic blog posts removed** — 2026-08-13. `2025-03-01-50-lessons-humanoid-vla.md`
      and `2026-03-12-hello-world-meet-rookie.md` moved to the repo-root `attic/`
      (personal-site content, never listed in `blog/README.md`).

---

## Backlog (nice-to-have, not blocking)

- [ ] Optional: MuJoCo-backed integration test for C1 (would need a headless render
      guard and is slower than the current analytical suite — deliberately excluded).
