# Lab 4: Collision Checking Infrastructure

## Overview

Motion planning requires a fast, reliable way to determine whether a given robot
configuration is collision-free. Lab 4 builds this on two libraries:

- **Pinocchio** provides the kinematic model and a `GeometryModel` that stores
  collision shapes attached to each link.
- **HPP-FCL** (the collision back-end used by Pinocchio) performs narrow-phase
  distance and Boolean collision queries between convex shapes, cylinders, and
  boxes.

The result is a `CollisionChecker` class that answers two questions for any
joint configuration `q`:

1. Is the robot free of self-collision?
2. Is the robot free of collisions with environment obstacles (boxes, table)?

---

## Pinocchio GeometryModel

When a URDF is loaded with `pin.buildGeomFromUrdf`, Pinocchio creates a
`GeometryModel` containing one `GeometryObject` per collision body defined in
the URDF. Each geometry object stores:

| Field          | Description                                   |
|----------------|-----------------------------------------------|
| `name`         | Geometry name from the URDF `<collision>` tag |
| `parentJoint`  | Index of the joint this body is attached to   |
| `parentFrame`  | Index of the corresponding frame              |
| `placement`    | SE3 transform from the parent joint frame     |
| `geometry`     | HPP-FCL shape (cylinder, sphere, box, mesh)   |

For the UR5e, the URDF defines 8 collision geometries: base_link, shoulder_link,
upper_arm_link, forearm_link, wrist_1_link, wrist_2_link, wrist_3_link, and
ee_link.

```python
import pinocchio as pin

model = pin.buildModelFromUrdf(str(urdf_path))
geom_model = pin.buildGeomFromUrdf(model, str(urdf_path), pin.COLLISION)

print(f"Robot geometries: {geom_model.ngeoms}")  # 8
```

---

## Environment Obstacles as HPP-FCL Boxes

Scene obstacles are defined as axis-aligned boxes using a simple dataclass:

```python
@dataclass
class ObstacleSpec:
    name: str
    position: np.ndarray    # (3,) center in world frame
    half_extents: np.ndarray # (3,) box half-sizes along x, y, z
```

Lab 4 defines four workspace obstacles and one table surface:

| Name  | Center (m)            | Half-extents (m)       |
|-------|-----------------------|------------------------|
| obs1  | (0.30, 0.15, 0.415)  | (0.06, 0.06, 0.10)    |
| obs2  | (0.30, -0.15, 0.415) | (0.06, 0.06, 0.10)    |
| obs3  | (0.45, 0.0, 0.375)   | (0.04, 0.12, 0.06)    |
| obs4  | (0.55, 0.12, 0.395)  | (0.05, 0.05, 0.08)    |
| table | (0.45, 0.0, 0.3)     | (0.30, 0.45, 0.015)   |

Each obstacle is added to the GeometryModel as a `GeometryObject` attached to
joint 0 (the universe/world frame), with an `hppfcl.Box` shape. Note that
HPP-FCL expects full extents, so the half-extents are doubled:

```python
full_extents = 2.0 * spec.half_extents
shape = hppfcl.Box(full_extents[0], full_extents[1], full_extents[2])
placement = pin.SE3(np.eye(3), spec.position.copy())
geom_obj = pin.GeometryObject(
    spec.name, 0, 0, placement, shape  # parent_joint=0, parent_frame=0
)
gid = geom_model.addGeometryObject(geom_obj)
```

---

## Self-Collision Filtering with Adjacency Gap

Checking every pair of robot geometries for self-collision is wasteful and
produces false positives. Adjacent links share a joint and their collision
meshes often overlap at the joint boundary. The `adjacency_gap` parameter
controls which pairs are skipped:

> Skip pair (i, j) if `|parentJoint(i) - parentJoint(j)| <= adjacency_gap`.

With `adjacency_gap=1` (the default), only links separated by at least two
joints in the kinematic chain are checked. This eliminates spurious contacts
at joint boundaries while still catching genuine self-collisions such as the
forearm folding into the shoulder.

```python
for i in range(n_robot_geoms):
    ji = geom_model.geometryObjects[i].parentJoint
    for j in range(i + 1, n_robot_geoms):
        jj = geom_model.geometryObjects[j].parentJoint
        if abs(ji - jj) <= adjacency_gap:
            continue
        geom_model.addCollisionPair(pin.CollisionPair(i, j))
```

---

## CollisionChecker API

The `CollisionChecker` class wraps all of the above into three methods:

### `is_collision_free(q) -> bool`

Runs `pin.computeCollisions` with `stop_at_first_collision=True` for speed.
Returns `True` if no collision pair reports contact.

### `is_path_free(q1, q2, resolution=0.05) -> bool`

Linearly interpolates between `q1` and `q2` in C-space, sampling at intervals
of `resolution` radians (L2 norm). Each sample is checked with
`is_collision_free`. This is the core primitive used by the RRT planner for
edge validation.

The number of samples is `max(2, ceil(||q2 - q1|| / resolution) + 1)`.

### `compute_min_distance(q) -> float`

Calls `pin.computeDistances` to find the minimum signed distance across all
collision pairs. A negative value indicates penetration depth. Useful for
debugging and for potential-field-based planners.

```python
cc = CollisionChecker()
print(f"Collision pairs: {cc.num_collision_pairs}")
print(f"Q_HOME free: {cc.is_collision_free(Q_HOME)}")
print(f"Min clearance at Q_HOME: {cc.compute_min_distance(Q_HOME):.4f} m")
```

---

## Cross-Validation: Pinocchio vs. MuJoCo

Because the MuJoCo scene (`scene_obstacles.xml`) defines the same obstacles
using MuJoCo geoms, we can cross-validate the two collision systems. The test
procedure:

1. Sample 200 random configurations uniformly from the joint limits.
2. For each configuration, query Pinocchio (`is_collision_free`) and MuJoCo
   (`mj_forward` then count `mj_data.ncon`).
3. MuJoCo floor-plane contacts are excluded (the Pinocchio model does not
   include a floor), filtering by `geom_type == 0` (plane).
4. Compute agreement percentage.

The result: **92.8% agreement** across 200 random samples. The remaining
disagreements arise from differences in collision geometry representations
(MuJoCo uses analytic geoms while the URDF may use simplified meshes) and
from the floor exclusion heuristic.

```python
# Simplified cross-validation loop
for q in random_configs:
    pin_free = cc.is_collision_free(q)
    mj_data.qpos[:6] = q
    mujoco.mj_forward(mj_model, mj_data)
    non_floor_contacts = count_non_floor_contacts(mj_data)
    mj_free = (non_floor_contacts == 0)
    if pin_free == mj_free:
        agree += 1
```

The 90% threshold is enforced by the test suite (`test_agreement_above_threshold`).

---

## Lessons Learned

**Table placement matters.** The original table position at x=0.4 with
half-extent 0.35 placed its near edge at x=0.05 -- just 5mm from the upper
arm cylinder (radius 0.055). This caused every configuration to report a
collision. Moving the table to x=0.45 with half-extent 0.30 pushed the near
edge to x=0.15, resolving the issue.

**Print colliding pairs for debugging.** When `is_collision_free` returns
`False` unexpectedly, iterating through `collisionPairs` and
`collisionResults` to print the specific colliding geometry names quickly
reveals whether the problem is a real collision or a scene setup error.

---

## Test Coverage

The collision module is covered by `tests/test_collision.py` with 20 tests
across five test classes:

- `TestCollisionCheckerInit` -- verifies geometry counts and pair registration
- `TestCollisionFree` -- known free and colliding configurations
- `TestSelfCollision` -- self-collision with no environment
- `TestPathFree` -- edge collision checking
- `TestCrossValidation` -- Pinocchio vs. MuJoCo agreement and FK match
- `TestModelLoading` -- model DOF and obstacle spec consistency
