# Lab 6: Lessons Learned

## Bugs & Fixes

### 2026-03-22 — Collision test configs didn't actually collide
**Symptom:** Two collision tests failed — configs expected to hit the table or cross-arm didn't trigger collisions.
**Root cause:** The left arm at origin has limited reach toward the table at x=0.5. Configs that looked like they should collide actually placed the EE in free space.
**Fix:** Used IK to find configs that reach the collision zone. For cross-arm, used IK-solved configs where both EEs meet at (0.5, 0, 0.4).
**Takeaway:** Always verify collision test configs numerically.

### 2026-03-22 — Pinocchio computeCollisions overwrites base-transformed placements
**Symptom:** `pin.computeCollisions` after applying base transforms to `geom_data.oMg` didn't work.
**Root cause:** `computeCollisions` calls `updateGeometryPlacements` internally, erasing manual offsets.
**Fix:** Used HPP-FCL directly for collision queries with world-frame placements.
**Takeaway:** For non-standard base transforms, use HPP-FCL primitives directly.

### 2026-03-22 — Gravity direction with rotated base
**Symptom:** Gravity compensation incorrect for the right arm (rotated 180° about Z).
**Root cause:** Pinocchio's gravity is in base frame; must be rotated for offset bases.
**Fix:** Transform world gravity via `R_base.T @ g_world` before `computeGeneralizedGravity`.
**Takeaway:** Always transform gravity to each model's local frame in multi-base setups.

## Debug Strategies

### Named joint lookups in MuJoCo
With multiple robots + free bodies, use `mujoco.mj_name2id()` + `jnt_qposadr`/`jnt_dofadr` by name. Never assume indexing order.

### Two-model collision checking
Separate Pinocchio models per arm + HPP-FCL for cross-arm pairs is cleaner than a combined 12-DOF model.

## Key Insights

### Object-centric frame is the right abstraction
ObjectFrame (object pose + grasp offsets) derives both arm targets from a single object trajectory, decoupling task planning from arm kinematics.

### Symmetric impedance naturally balances internal forces
Identical gains on both arms + symmetric targets = balanced internal forces without explicit force control.

### Weld constraints simplify grasp simulation
Equality weld constraints (activated at grasp, deactivated at release) sidestep friction-limited grasp mechanics entirely.
