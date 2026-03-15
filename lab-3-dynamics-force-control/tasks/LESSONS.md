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

## Debug Strategies

_(Document debugging techniques that prove useful during this lab.)_

## Key Insights

_(Record non-obvious learnings about dynamics, force control, MuJoCo contact, etc.)_
