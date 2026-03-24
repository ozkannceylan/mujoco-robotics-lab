# Lab 3: Lessons Learned

## Bugs & Fixes

### 2026-03-17 — Lab 3 had to be migrated back onto the canonical hardware baseline
**Symptom:** Lab 3 was previously documented as complete on a simplified local UR5e path, which did not satisfy the project-wide requirement for Menagerie UR5e + Robotiq.
**Root cause:** Lab-level implementation drifted away from the project hardware lock.
**Fix:** Rebuilt the canonical path around Menagerie `universal_robots_ur5e` + mounted `robotiq_2f85`, updated the Pinocchio model path, reran validation, and updated the docs/media to match the final stack.
**Takeaway:** A lab is not complete until the code, tests, docs, and media all refer to the same hardware baseline.

### 2026-03-17 — Menagerie servo actuators required torque-to-control mapping
**Symptom:** The canonical MuJoCo UR5e stack does not expose direct torque motors the way the earlier simplified Lab 3 path did.
**Root cause:** Menagerie UR5e uses arm actuators with their own gain/bias model and control ranges.
**Fix:** Added torque-to-control mapping in `lab3_common.py` so desired arm torques are converted into actuator controls before each MuJoCo step.
**Takeaway:** Do not assume torque commands can be written directly when migrating to Menagerie assets. Check the actuator model first.

### 2026-03-17 — Pinocchio parity depended on the mounted Robotiq payload inertial being correct
**Symptom:** Dynamics parity and gravity compensation failed until the mounted gripper payload seen by Pinocchio matched the executed MuJoCo stack closely enough.
**Root cause:** The fixed end-effector payload in the URDF did not initially match the compiled MuJoCo scene inertial properties.
**Fix:** Updated the `ee_link` inertial in `models/ur5e.urdf` to the compiled-scene mass, COM, and inertia.
**Takeaway:** For cross-engine parity, the mounted payload is not optional bookkeeping. Its inertial parameters matter.

### 2026-03-17 — Contact detection had to include the full real end-effector contact set
**Symptom:** A narrow contact filter can miss the first real table contact on the canonical stack.
**Root cause:** The first-contact link is not guaranteed to be only the terminal tool body.
**Fix:** `get_ee_and_table_ids()` now includes `wrist_3_link` and the mounted Robotiq bodies, and the force-control validation was rerun on that basis.
**Takeaway:** Contact logic should reflect the real executed geometry, not an idealized tip-only assumption.

### 2026-03-17 — 6D impedance orientation error needed the large-angle-safe formulation
**Symptom:** The earlier skew-symmetric orientation error is only reliable near small rotations.
**Root cause:** That formulation is a local approximation and can misbehave near large-angle errors.
**Fix:** Switched the canonical impedance orientation error to `pin.log3(R_des @ R_cur.T)`.
**Takeaway:** Reuse the Lie-log orientation error in shared Cartesian controllers unless there is a strong reason not to.

### 2026-03-17 — Native MuJoCo OpenGL rendering was unavailable in this environment
**Symptom:** The standard MuJoCo renderer could not create an OpenGL context here.
**Root cause:** The required GL shared libraries are not available to this runtime.
**Fix:** The validation video still uses the real MuJoCo simulation state by projecting actual MuJoCo geom poses frame by frame and overlaying measured values.
**Takeaway:** Environment-specific rendering failures should not block controller validation, but the limitation should be documented explicitly.

### 2026-03-15 — Force sensor reads articulation forces, not contact forces
**Symptom:** `<force site="tool_site"/>` style sensing reported force even without contact.
**Root cause:** MuJoCo `<force>` sensors measure body-level constraint forces, not only contact forces.
**Fix:** Switched to `mj_contactForce()` on the relevant contact pairs.
**Takeaway:** Use `mj_contactForce()` for contact control logic.

### 2026-03-15 — Force control needs filtering and damping at stiff contact
**Symptom:** Raw PI contact control chatters at the table.
**Root cause:** Stiff contact plus integral action creates oscillatory force behavior without additional damping.
**Fix:** Retained EMA force filtering and Z-velocity damping in the canonical controller.
**Takeaway:** Stable force control needs both measurement smoothing and contact-direction damping.

## Final Validation Snapshot

- Full Lab 3 test suite: `34 passed`
- Max gravity mismatch: `8.01e-06`
- Max mass-matrix mismatch: `3.34e-05`
- Gravity hold max error: `8.91e-06 rad`
- Gravity perturbation final speed: `0.0134 rad/s`
- Hybrid force mean: `4.89 N`
- Hybrid force in-band rate: `99.96%`
- Hybrid force max XY error: `3.60 mm`
- Line trace in-band rate: `94.07%`
- Line trace max XY error: `1.70 mm`
