# Lab 3: Dynamics & Force Control — Completion Report

Completion date: 2026-03-17

## Platform Lock

Lab 3 is completed on:

- MuJoCo Menagerie `universal_robots_ur5e`
- mounted MuJoCo Menagerie `robotiq_2f85`
- Pinocchio model matched to that executed stack

## Completed Work

### Phase 0: Platform alignment
- Replaced the simplified local Lab 3 baseline with Menagerie UR5e + Robotiq
- Rebuilt the MuJoCo scene path around the canonical assets in `src/lab3_common.py`
- Rebuilt the Pinocchio path around a URDF that includes the mounted gripper payload

### Phase 1: Dynamics fundamentals
- Re-validated `g(q)` parity on representative configurations
- Re-validated `M(q)` parity on representative configurations
- Preserved gravity/mass consistency after payload alignment

### Phase 2: Impedance control
- Revalidated Cartesian impedance on the canonical stack
- Updated orientation error handling to `pin.log3(...)` for large-angle-safe 6D control

### Phase 3: Force control and contact
- Rebuilt contact-force reading around `mj_contactForce()`
- Expanded the EE contact set to include `wrist_3_link` and all mounted Robotiq bodies
- Revalidated hybrid force control and constant-force line tracing on the real stack

### Phase 4: Validation hardening
- Updated the Lab 3 tests to guard the canonical stack behavior
- Re-ran the full Lab 3 suite cleanly

### Phase 5: Documentation and media
- Updated Lab 3 README and task docs to the canonical final state
- Recorded a Lab 3 validation video into `media/`

## Final Validation

- Full test suite: `34 passed`
- Max gravity mismatch: `8.01e-06`
- Max mass-matrix mismatch: `3.34e-05`
- Gravity hold max error: `8.91e-06 rad`
- Gravity perturbation final speed: `0.0134 rad/s`
- Hybrid force mean: `4.89 N`
- Hybrid force in-band rate (`5 +/- 1 N`): `99.96%`
- Hybrid force max XY error: `3.60 mm`
- Line trace in-band rate (`5 +/- 1 N`): `94.07%`
- Line trace max XY error: `1.70 mm`

## Sign-Off Artifacts

- README: `lab-3-dynamics-force-control/README.md`
- Validation video: `lab-3-dynamics-force-control/media/lab3_validation_real_stack.mp4`
- Recorder: `lab-3-dynamics-force-control/src/record_lab3_validation.py`

## Residual Note

Native MuJoCo OpenGL rendering is unavailable in this environment. The saved validation video still visualizes the actual MuJoCo simulation state by projecting real MuJoCo geom poses, not by replaying an external approximation.
