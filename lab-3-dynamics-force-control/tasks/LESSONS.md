# Lab 3: Lessons Learned

## Bugs & Fixes

### 2026-03-15 — URDF from Lab 2 does not match Lab 3 MJCF
**Symptom:** FK cross-validation showed 105 mm error at Q_HOME between Pinocchio (Lab 2 URDF) and MuJoCo (Lab 3 MJCF).
**Root cause:** Lab 2's URDF was built for the MuJoCo Menagerie UR5e model (different frame conventions: Y-axis rotations, 180° Z base rotation, different link offsets). Lab 3's simplified MJCF uses all-Z-axis joints with X-rotation euler frames — a completely different kinematic description.
**Fix:** Rebuilt URDF from scratch by extracting body positions, quaternions, and inertias directly from the MJCF via `mj_model.body_pos`, `body_quat`, `body_mass`, `body_inertia`, `body_ipos`. Result: 0.000 mm FK error across all test configs.
**Takeaway:** Never assume a URDF from one MJCF variant matches another. Always extract kinematics directly from the MJCF you are using and regenerate the URDF.

### 2026-03-15 — Inertia tensors must account for body_iquat rotation
**Symptom:** Mass matrix cross-validation showed ~0.137 max error even after URDF kinematic chain was correct.
**Root cause:** MuJoCo's `body_inertia` stores diagonal values in the **principal axis frame** (rotated by `body_iquat`). Copying diagonals directly into URDF gives wrong results when `body_iquat` is non-identity. For upper_arm (90° Y rotation), ixx/izz were swapped. For forearm (108° Y rotation), a nonzero ixz cross-term was missing.
**Fix:** Compute full inertia tensor in body frame: `I_full = R @ diag(I_principal) @ R^T` where R is from `body_iquat`.
**Takeaway:** Always check `body_iquat` when extracting inertias from MuJoCo. Only use diagonal values directly if iquat is identity.

### 2026-03-15 — Pinocchio armature vs rotorInertia for mass matrix matching
**Symptom:** After fixing inertias, M(q) diagonal still had a constant 0.01 offset (matching MJCF armature=0.01).
**Root cause:** Pinocchio's `model.rotorInertia` does NOT affect `crba()` output. The correct field is `model.armature`, which is added directly to the CRBA diagonal — matching MuJoCo's behavior.
**Fix:** Set `model.armature[:] = 0.01` instead of `model.rotorInertia[i] = 0.01`.
**Takeaway:** In Pinocchio, use `model.armature` (not `rotorInertia`) to match MuJoCo joint armature for mass matrix agreement.

### 2026-03-15 — Force sensor reads articulation forces, not contact forces
**Symptom:** `<force site="tool_site"/>` sensor reported large forces even when no contact existed.
**Root cause:** MuJoCo's `<force>` sensor measures all constraint forces on the body, including articulation forces from joints — not just contact forces.
**Fix:** Switched to `mj_contactForce()` per-contact API, filtering for contacts between tool0/wrist_3 geoms and table_top geom.
**Takeaway:** Use `mj_contactForce()` for actual contact force measurement. The `<force>` sensor is body-level, not contact-level.

### 2026-03-15 — EE body ID too restrictive for contact detection
**Symptom:** `read_contact_force_z()` returned 0 despite `ncon > 0` contacts existing.
**Root cause:** Only tool0 body was in `ee_body_ids`, but the wrist_3 capsule geom extends further and hits the table first (wrist_3 capsule fromto="0 0 0 0 0 0.135" size="0.035").
**Fix:** Include both tool0 (body 8) and wrist_3 (body 7) in `ee_body_ids`.
**Takeaway:** When detecting EE contact, consider all link geometries near the EE, not just the final body.

### 2026-03-15 — Force oscillation in PI controller with stiff contact
**Symptom:** Mean force 4.97N but std 11.07N, 0% within ±1N band.
**Root cause:** PI force controller without damping or filtering causes chattering at stiff contact surface.
**Fix:** Three-part fix: (1) EMA low-pass filter on force (alpha=0.2), (2) Z-velocity damping K_dz=30.0, (3) reduced anti-windup. Result: std 0.30N, 97% within ±1N.
**Takeaway:** Contact force control needs velocity damping and force filtering to prevent chattering.

### 2026-03-15 — Starting config must be reachable from approach target
**Symptom:** Approach phase never made contact — EE stayed at z≈0.45m.
**Root cause:** Q_HOME places EE at (-0.233, 0.492, 0.453), far from table target (0.4, 0.0). Impedance controller couldn't traverse that distance.
**Fix:** Computed Q_ABOVE_TABLE via IK to start with EE at (0.4, 0.0, 0.35), directly above the table.
**Takeaway:** For force control demos, pre-compute a starting config near the contact surface.

## Debug Strategies

### Contact force debugging
Print `mj_data.ncon`, iterate contacts, check `geom1`/`geom2` names and body IDs. Use `mj_contactForce()` to verify force magnitudes per contact pair.

### Approach trajectory debugging
Print EE position, z_target, and contact force at regular intervals during approach to verify descent is progressing.

## Key Insights

### EMA filter alpha tuning for force control
Alpha=0.05 was too slow (took hundreds of timesteps to register contact). Alpha=0.2 gives good noise rejection while allowing fast enough response for the PI controller to track.

### Z-velocity damping is critical for contact stability
Without K_dz damping, the PI force controller and contact dynamics create an underdamped oscillation. K_dz=30 provides critical damping at the contact interface.

### Jacobian X-row near-zero at vertical tool configuration
When the EE points straight down at a low table (z=0.17), the Jacobian X-row has very small values (~0.01). A 100N Cartesian X-force produces only ~6Nm joint torque. XY motion during contact requires slow trajectories — 50mm over 6s works (3mm error), but 150mm over 4s fails (50mm error).

### Collision filtering with contype/conaffinity
Using contype=1/conaffinity=1 for arm geoms and contype=2/conaffinity=2 for the table+probe isolates EE-table contact from wrist-table collisions. This prevents false contacts and ensures clean force measurement. The probe geom must have density=0 mass=0 to avoid affecting dynamics cross-validation.

### Probe mass affects cross-validation
A probe geom with mass=0.01 kg at the EE tip causes 0.05 Nm gravity error between Pinocchio and MuJoCo (Pinocchio doesn't know about the probe). Always make contact-only geoms massless.
