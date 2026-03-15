# Lab 4: Lessons Learned

## Bugs & Fixes

### 2026-03-15 — Table edge collides with upper arm at all configs
**Symptom:** Every configuration (including Q_HOME, q=0) reported collision.
**Root cause:** Table at x=0.4 with half_extent 0.35 placed its near edge at x=0.05. The upper arm cylinder (r=0.055) at x=0 extended to x=0.055 — a 5mm overlap.
**Fix:** Moved table to x=0.45 and reduced X half-extent to 0.30. Near edge now at x=0.15, safely away from the arm.
**Takeaway:** Always verify obstacle placement against the robot's swept volume near the base. Marginal collisions are easy to miss visually.

### 2026-03-15 — TOPP-RA crashes on duplicate waypoints
**Symptom:** `ValueError: x must be strictly increasing sequence` from scipy CubicSpline.
**Root cause:** RRT paths can have near-duplicate consecutive waypoints (distance < 1e-8), causing zero-length segments in the arc-length parameterization.
**Fix:** Filter consecutive duplicate waypoints before constructing the spline: skip any waypoint within 1e-8 of the previous one.
**Takeaway:** Always sanitize geometric data before passing to spline constructors.

## Debug Strategies

### Print colliding pair names
When `is_collision_free` returns False unexpectedly, iterate through `collisionPairs` / `collisionResults` to print which specific geometry pair is colliding. Often reveals environment setup issues rather than real planning problems.

## Key Insights

### Pinocchio GeometryObject constructor (non-deprecated)
Use: `GeometryObject(name, parent_joint, parent_frame, placement, shape)`.
The older `(name, parent_joint, parent_frame, shape, placement)` order is deprecated.

### Adjacent-link filtering for self-collision
Skip collision pairs where parent joint indices differ by ≤1 (adjacency_gap). Adjacent links physically can't collide, and checking them wastes time and produces false positives from overlapping collision geometries at joints.

### Shortcutting equalizes RRT and RRT* paths
When the direct path between start and goal is collision-free, both RRT and RRT* paths shortcut to the same 2-waypoint direct line. RRT*'s advantage shows primarily when obstacles force detours that shortcutting can't eliminate.

### Pinocchio collision uses cylinders, MuJoCo uses capsules
URDF `<cylinder>` maps to HPP-FCL Cylinder in Pinocchio, but MuJoCo `<capsule>` has rounded ends. This causes ~7% disagreement on boundary cases. Acceptable for planning — Pinocchio is slightly conservative in some cases, slightly permissive in others.
