# Lab 3: TODO

## Canonical UR5e + Robotiq Path
- [x] Step R1: Replace the Lab 3 robot baseline with MuJoCo Menagerie UR5e + mounted Robotiq 2F-85
- [x] Step R2: Rebuild the MuJoCo scenes and matching Pinocchio model path on the canonical hardware stack
- [x] Step R3: Re-run FK, gravity, and mass-matrix parity checks on the canonical stack
- [x] Step R4: Re-qualify gravity compensation and impedance control on the canonical stack
- [x] Step R5: Fix and validate contact detection on the canonical stack, including all EE-adjacent first-contact links
- [x] Step R6: Re-run hybrid force control and constant-force line tracing on the canonical stack
- [x] Step R7: Expand tests and telemetry so reviewed risks are covered by regression checks
- [x] Step R8: Update README/docs/media so only the canonical stack is presented as Lab 3 completion

## Post-Completion Review Follow-Ups (2026-08-13)
- [x] Step F1: Fix the MuJoCo 3.11 `MjData.qM` / `mj_fullM` API break — added `mj_dense_mass_matrix()` to `lab3_common.py`, suite back to `34 passed` (see LESSONS.md)
- [x] Step F2: Write the blog post required by `plan/LAB_03.md` — `blog/lab3_blog_post.md`, "Position Control Pushes Through Walls"
- [x] Step F3: Document the orphaned demo pipeline — `src/record_lab3_demo.py` and its three MP4 outputs now appear in README and ARCHITECTURE.md

## Current Focus
> Lab 3 is complete. No open implementation blockers remain inside this lab.

## Follow-On Dependency
> Labs 4 and 5 still need their own migration, review, and validation on the same hardware baseline.
