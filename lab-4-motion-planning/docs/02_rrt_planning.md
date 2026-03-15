# Lab 4: RRT and RRT* Motion Planning

## Overview

With collision checking in place, we need an algorithm to find a collision-free
path from a start configuration to a goal configuration. Lab 4 implements two
sampling-based planners that operate in the 6-dimensional configuration space
(C-space) of the UR5e:

- **RRT (Rapidly-exploring Random Tree):** Grows a tree by sampling random
  configurations and extending toward them. Returns the first feasible path
  found.
- **RRT*:** An asymptotically optimal variant that rewires the tree as it grows,
  producing progressively shorter paths over time.

Both algorithms share the same `RRTStarPlanner` class, selected via a boolean
flag.

---

## C-Space Representation

Each configuration `q` is a 6-element vector of joint angles in radians. The
joint limits for the UR5e are set to [-2pi, 2pi] for all joints. The distance
metric used throughout is the L2 norm in C-space:

    d(q1, q2) = ||q1 - q2||_2

This is a common choice for revolute-only manipulators where all joints have
comparable ranges.

---

## The RRTNode Dataclass

Each node in the search tree stores three fields:

```python
@dataclass
class RRTNode:
    q: np.ndarray          # Joint configuration (6,)
    parent: int | None     # Index of parent node (None for root)
    cost: float            # Cumulative C-space path length from root
```

The tree is stored as a flat list (`list[RRTNode]`), with parent references by
index. This keeps memory compact and makes backtracking straightforward.

---

## RRT Algorithm

The basic RRT loop proceeds as follows:

1. **Initialize** the tree with the start node (cost = 0, no parent).
2. **Sample** a random configuration `q_rand`. With probability `goal_bias`,
   use the goal configuration instead (this accelerates convergence).
3. **Nearest neighbor:** Find the tree node closest to `q_rand` in L2 distance.
4. **Steer:** Move from `q_nearest` toward `q_rand` by at most `step_size`:

       q_new = q_nearest + min(1, step_size / d) * (q_rand - q_nearest)

5. **Collision check:** Verify the edge from `q_nearest` to `q_new` using
   `CollisionChecker.is_path_free`. If blocked, discard and loop.
6. **Add** `q_new` to the tree with `q_nearest` as parent.
7. **Goal test:** If `d(q_new, q_goal) < goal_tolerance` and the edge to
   `q_goal` is collision-free, add the goal node and return the path.

Basic RRT returns immediately upon finding the first path, without any attempt
to improve it.

---

## RRT* Algorithm

RRT* modifies steps 6 and 7 to produce shorter paths:

### Best Parent Selection

Instead of always connecting `q_new` to its nearest neighbor, RRT* searches
a neighborhood of radius `rewire_radius` for the node that yields the lowest
cumulative cost:

```python
near_indices = [i for i, node in enumerate(tree)
                if dist(node.q, q_new) <= rewire_radius]

best_parent = nearest_idx
best_cost = tree[nearest_idx].cost + dist(tree[nearest_idx].q, q_new)

for idx in near_indices:
    candidate_cost = tree[idx].cost + dist(tree[idx].q, q_new)
    if candidate_cost < best_cost:
        if collision_checker.is_path_free(tree[idx].q, q_new):
            best_parent = idx
            best_cost = candidate_cost
```

### Rewiring

After adding `q_new`, RRT* checks whether any neighbor could lower its cost
by routing through `q_new`:

```python
for idx in near_indices:
    new_cost = best_cost + dist(q_new, tree[idx].q)
    if new_cost < tree[idx].cost:
        if collision_checker.is_path_free(q_new, tree[idx].q):
            tree[idx].parent = new_idx
            propagate_cost(idx, new_cost)
```

Cost changes are propagated recursively to all descendants via
`_propagate_cost`, ensuring the entire subtree reflects the improvement.

### Continued Search

Unlike basic RRT, RRT* does not return immediately when the goal is reached.
It continues growing the tree for `max_iter` iterations, updating the goal
node's parent whenever a shorter path is found. This is what gives RRT* its
asymptotic optimality property.

---

## RRTStarPlanner API

```python
planner = RRTStarPlanner(
    collision_checker=cc,
    step_size=0.3,         # Max extension per iteration (rad)
    goal_bias=0.1,         # Probability of sampling the goal
    rewire_radius=1.0,     # Neighborhood radius for RRT*
    goal_tolerance=0.15,   # L2 distance threshold for goal
)

path = planner.plan(
    q_start=Q_HOME,
    q_goal=q_target,
    max_iter=5000,
    rrt_star=True,         # False for basic RRT
    seed=42,               # For reproducibility
)
# Returns: list[np.ndarray] of waypoints, or None if no path found
```

Key design decisions:

- **`step_size=0.3`** balances exploration speed against collision-check
  granularity. Too large and edges may clip obstacles between samples; too
  small and the tree grows slowly.
- **`goal_bias=0.1`** means 10% of samples are directed at the goal. Higher
  values converge faster but can get trapped behind obstacles.
- **`rewire_radius=1.0`** covers a generous neighborhood in the 6D C-space.
  The computational cost scales with tree size, but for trees under 5000 nodes
  this is acceptable.

---

## Path Extraction

Once the goal node exists, the path is recovered by backtracking through parent
indices:

```python
def _extract_path(self, goal_idx: int) -> list[np.ndarray]:
    path = []
    idx = goal_idx
    while idx is not None:
        path.append(self._tree[idx].q.copy())
        idx = self._tree[idx].parent
    path.reverse()
    return path
```

The result is an ordered list of waypoint configurations from start to goal.

---

## RRT vs. RRT*: Empirical Comparison

Using the capstone demo configuration (Q_HOME to a goal across the obstacle
field, seed=42, max_iter=5000):

| Metric           | RRT      | RRT*     |
|------------------|----------|----------|
| Path found       | Yes      | Yes      |
| Raw path cost    | Higher   | Lower    |
| Tree nodes       | Varies   | Varies   |

The test `test_rrt_star_shorter_than_rrt` verifies that RRT* produces a path
with cost at most 1.1x the RRT path cost (it is typically strictly shorter
due to rewiring).

---

## Visualization

The `visualize_plan` function projects the entire tree and the final path into
3D task space using Pinocchio FK. Tree edges are drawn as light blue lines
(subsampled to 2000 for large trees), and the path is drawn as a green line
with markers at each waypoint. Obstacle boxes are rendered as red wireframes.

```python
from rrt_planner import visualize_plan

visualize_plan(
    planner, path, ee_fid,
    title="RRT* Plan",
    save_path=MEDIA_DIR / "rrt_star_plan.png",
)
```

---

## Test Coverage

The planner is tested in `tests/test_planner.py` with 11 tests across three
classes:

- `TestRRTBasic` -- path existence, endpoint correctness, waypoint and edge
  collision freedom, handling of colliding start/goal
- `TestRRTStar` -- path finding, cost comparison with RRT, tree population,
  seed reproducibility
- `TestVisualization` -- file output without errors
