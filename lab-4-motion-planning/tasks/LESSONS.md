# Lab 4: Lessons Learned

## Bugs & Fixes

### 2026-03-15 — Table edge collides with upper arm at all configs
**Symptom:** Every configuration (including Q_HOME, q=0) reported collision.
**Root cause:** Table at x=0.4 with half_extent 0.35 placed its near edge at x=0.05. The upper arm cylinder (r=0.055) at x=0 extended to x=0.055 — a 5mm overlap.
**Fix:** Moved table to x=0.45 and reduced X half-extent to 0.30. Near edge now at x=0.15, safely away from the arm.
**Takeaway:** Always verify obstacle placement against the robot's swept volume near the base. Marginal collisions are easy to miss visually.

## Debug Strategies

### Print colliding pair names
When `is_collision_free` returns False unexpectedly, iterate through `collisionPairs` / `collisionResults` to print which specific geometry pair is colliding. Often reveals environment setup issues rather than real planning problems.

## Key Insights

### Pinocchio GeometryObject constructor (non-deprecated)
Use: `GeometryObject(name, parent_joint, parent_frame, placement, shape)`.
The older `(name, parent_joint, parent_frame, shape, placement)` order is deprecated.

### Adjacent-link filtering for self-collision
Skip collision pairs where parent joint indices differ by ≤1 (adjacency_gap). Adjacent links physically can't collide, and checking them wastes time and produces false positives from overlapping collision geometries at joints.
