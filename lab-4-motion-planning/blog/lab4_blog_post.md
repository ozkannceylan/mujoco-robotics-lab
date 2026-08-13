# From Free Space to Cluttered Environments

*Lab 4 of the MuJoCo Robotics Lab series*

---

By the end of Lab 3 I had a UR5e that could do everything a textbook asks of an arm: forward kinematics, analytical Jacobians, inverse kinematics, gravity compensation, Cartesian impedance, hybrid force control against a real table. Point at a pose, and the arm goes there.

Then I put a box in the way, and the whole edifice fell over. Not metaphorically — the IK solver returned a perfectly valid joint configuration, the controller drove toward it with beautiful tracking error, and the arm walked straight through the obstacle as if it weren't there.

This is the gap Lab 4 exists to close. Everything before it operates in free space. Everything after it — grasping, dual-arm coordination, loco-manipulation — happens in a world with stuff in it.

---

## Why IK Alone Isn't Planning

The distinction sounds pedantic until you've watched it fail. **Inverse kinematics answers a question about a point:** given a Cartesian target, find joint angles that put the end-effector there. The answer is a configuration. **Motion planning answers a question about a path:** find a continuous curve from start to goal that stays inside the free subset of configuration space. The answer is a sequence.

IK has no concept of "between." A damped-least-squares solver converges by iterating on joint angles, and those intermediate iterates are numerical scratch work — not a trajectory, and nobody checked them for collision. Even when start and goal are both collision-free, the straight line joining them in joint space usually is not. That straight line is exactly what a PD controller will try to follow.

Worse, obstacles make the free space **non-convex**. To get around a box you may need to move *away* from the goal first — increase your distance metric before you can decrease it. No gradient-following method gets you there. You need something that explores. That's the argument for sampling-based planning, and the reason Lab 4 is its own lab rather than a footnote in Lab 3.

---

## Collision Truth: The Decision That Mattered Most

Before writing a line of planner code I had to answer a question that shaped the entire lab: **what does "in collision" mean, and who decides?**

The original Lab 4 implementation had a separate hand-built collision URDF and a standalone XML scene. The planner reasoned about that geometry; MuJoCo executed different geometry. This is the most seductive bug in robotics, because everything looks correct in isolation: the planner reports a collision-free path, the controller tracks it to four decimal places, and the arm still clips the obstacle.

I rebased the lab onto the canonical stack Lab 3 already used — Menagerie `universal_robots_ur5e` with a mounted `robotiq_2f85`, plus table and obstacle geoms in that *same* scene. The `CollisionChecker` keeps its original API:

```python
cc.is_collision_free(q)      -> bool
cc.is_path_free(q1, q2)      -> bool
cc.compute_min_distance(q)   -> float
```

but internally it sets `data.qpos` on the executed MuJoCo model and queries the actual contact set. Pinocchio (with HPP-FCL underneath) stays in the picture for FK and the gravity term, consistent with the project principle: **Pinocchio is the analytical brain, MuJoCo is the physics.** But collision truth comes from the geometry that will actually be simulated.

The takeaway I wrote into `LESSONS.md` that day: *for motion planning, the geometry the planner trusts must match the geometry the controller executes.* Any divergence is a bug you only discover at contact time.

---

## The Adjacent-Link False Positive

The first thing the new collision checker told me was that the robot was in collision at its home configuration. `compute_min_distance(Q_HOME)` returned a small **negative** number — penetration. The arm was standing in an ordinary pose, touching nothing, and the checker insisted it was intersecting itself.

The culprit was the Robotiq gripper. A parallel-jaw gripper is a closed linkage: the finger links, couplers and pads are *designed* to sit within millimetres of each other, and their collision meshes overlap slightly by construction. A naive all-pairs self-collision check flags every one. So does the shoulder-to-upper-arm pair on the UR5e itself, where two links share a joint and their geometry necessarily meets at the axis.

The fix is a policy decision encoded as one parameter. `CollisionChecker` takes an `adjacency_gap` (default `1`) and skips any pair whose bodies are within that many steps of each other in the kinematic tree:

```python
return self._body_tree_distance(body1, body2) > self.adjacency_gap
```

Links that share a joint cannot meaningfully collide — their relative motion is one degree of freedom the joint limits already bound. Checking them produces noise, never signal. The pairs that *do* matter — arm-vs-arm at a distance (elbow folding into the base), arm-vs-gripper, everything-vs-environment — stay active.

It cost me most of an afternoon, and the symptom (a tiny negative distance in a pose that is obviously fine) is easy to misread as a units bug or a frame bug when it's actually a modelling-policy bug.

---

## RRT and RRT*, Written Out By Hand

I wrote both planners from scratch rather than pulling in OMPL. For a portfolio lab that's the point: the algorithms are short enough to implement honestly, and implementing them is how you learn where they're fragile.

The tree is a flat `list[RRTNode]` with parent references by index:

```python
@dataclass
class RRTNode:
    q: np.ndarray          # joint configuration (6,)
    parent: int | None     # index of parent (None for root)
    cost: float            # cumulative C-space path length from root
```

Basic RRT is a five-step loop. Sample a random configuration (with probability `goal_bias`, the goal instead). Find the nearest tree node. Steer toward the sample by at most `step_size`. Check the edge with `is_path_free`; if clear, add the node. If the new node is within `goal_tolerance` and the connecting edge is free, backtrack the parent chain and return.

RRT* changes two things and gets asymptotic optimality for it. **Best-parent selection:** instead of connecting to the nearest neighbour, search a ball of radius `rewire_radius` for the neighbour yielding the lowest *cumulative* cost, subject to a collision-free edge. **Rewiring:** after inserting, check whether any neighbour would be cheaper routed through the new node, and if so reparent and propagate the cost change through its subtree.

One behavioural difference matters enormously: RRT returns the moment it reaches the goal. RRT* keeps going for the full `max_iter` budget, updating the goal's parent whenever a shorter route appears. That continued search buys the optimality — and it's also why RRT* is slow.

Of the tuned parameters (`step_size=0.16`, `goal_bias=0.20`, `rewire_radius=0.90`, `goal_tolerance=0.10`), the `step_size` / collision-resolution interaction is the one worth internalising. Edge checking discretizes the straight line between two configurations and tests each sample. If your step size is large relative to the thinnest obstacle dimension, a path can pass cleanly through a wall because no sample landed inside it. There is no error message for this. The planner returns a beautiful path and the arm drives through the box.

---

## Shortcutting: Making RRT Output Fit for a Controller

Raw RRT output is ugly. It has to be — the tree grows by random extension, so the path inherits every wrong turn the sampler took. On the blocked-path validation scene, the raw plan came out at **35 waypoints** for a motion that fundamentally requires going around one obstacle.

Shortcutting fixes this with an embarrassingly simple loop: pick two non-adjacent waypoints at random, test whether the straight joint-space edge between them is collision-free, and if it is, delete everything in between. Repeat a few hundred times (`SHORTCUT_ITER = 220`). On the validation scene that collapsed 35 waypoints to **3**, cutting path cost by about 20%. It isn't optimal — shortcutting is a local operator and can't discover a topologically different route — but it converts a jittery random walk into something a human recognises as the intended motion.

The correctness property that makes it safe: every shortcut is validated with the same `is_path_free` the planner used, so a shortcut path is collision-free by construction. You aren't trading safety for smoothness.

---

## Time Parameterization, and an Honest Note About TOPP-RA

A path is a geometric object; it has no time in it. A controller needs `q(t)`, `q̇(t)`, `q̈(t)`, and those must respect the robot's velocity and acceleration limits or the actuators saturate and tracking falls apart. Time-Optimal Path Parameterization by Reachability Analysis (TOPP-RA) is the standard answer: fix the geometric path, then solve for the fastest timing along it subject to those bounds, by propagating reachable and controllable velocity sets forward and backward along the path parameter.

Here's the honest part: **TOPP-RA does not build in my environment.** No system compiler, no prebuilt wheel for the Python version in use. I could change the pipeline, or keep the contract and fall back. I kept the contract:

```python
times, q, qd, qdd = parameterize_topp_ra(path, vel_limits, acc_limits, dt=DT)
```

The function uses TOPP-RA when importable and otherwise falls back to a conservative quintic parameterization respecting the same limits. The fallback is *not* time-optimal — deliberately slower, because the safe response to losing your optimizer is to give up speed rather than give up bounds. The limits are still enforced and still tested. I documented this in the README and `LESSONS.md` rather than papering over it, because "we used TOPP-RA" would be a lie and the difference shows up as a longer trajectory duration. Lab 5 later imported this exact function unchanged — the entire argument for a stable public API around an internal fallback.

Two real bugs came out of this stage. TOPP-RA-style parameterization builds an arc-length spline, and `scipy.interpolate.CubicSpline` requires strictly increasing knots; shortcut output can contain waypoints separated by less than `1e-8`, which crashes construction, so filter near-duplicates first. And sparse waypoints let the spline **overshoot between them** — a cubic through widely spaced knots can bulge outside the corridor you carefully validated. `densify_path(full_path, max_step=0.02)` resamples the polyline finely enough that the spline has nowhere to bulge into. Collision-free waypoints do not imply a collision-free spline through them.

---

## The Capstone, Version One: A Demo That Proved Nothing

My first capstone was a start-to-goal motion across an obstacle field. It ran. It tracked to 0.0125 rad RMS. It looked fine. It also proved nothing, and I nearly shipped it.

The problem surfaced when I sat down to record the demo video. After shortcutting, the executed path was essentially a straight line from start to goal. The obstacles were technically in the scene, technically avoided, and technically irrelevant — the direct line between those two configurations happened to be collision-free all along. A viewer sees an arm move from A to B. Nothing on screen distinguishes "this planner avoided obstacles" from "IK plus a PD controller."

A demo where the naive solution also works is not a demonstration of planning. So I redesigned the capstone around a scenario where the naive solution *provably* fails.

---

## The Capstone, Version Two: The Slalom

Four boxes, each 10 × 10 × 20 cm, standing on the table at alternating Y positions:

```
box_1 at (0.40, -0.15)
box_2 at (0.50, +0.15)
box_3 at (0.60, -0.15)
box_4 at (0.70, +0.15)
```

The end-effector weaves through them at a constant height of z = 0.56 m. Nine Cartesian via-points — start, left of box 1, midpoint of gap 1–2, right of box 2, gap 2–3, left of box 3, gap 3–4, right of box 4, exit — giving eight segments planned independently and concatenated.

The staggered layout is the whole trick. Every consecutive pair of via-points has a box sitting between them, so there is no straight line that works anywhere along the route. Shortcutting cannot degenerate the path into a direct connection because a direct connection is always blocked. The planner has to earn every segment.

Three implementation problems came out of this design.

**IK has to be obstacle-aware, indirectly.** IK solvers know nothing about obstacles, so a solution that reaches the right Cartesian point can still have the *elbow* buried in a box. The fix is a seed bank: 23 starting configurations fed to the DLS solver, each solution filtered by `cc.is_collision_free(q)` and a clearance floor, then scored by `(clearance, -continuity)` — prefer the configuration furthest from obstacles, break ties toward the one closest to the previous waypoint. Without the continuity term the arm flips elbow-up to elbow-down mid-slalom: collision-free, correct, and visually incomprehensible.

**Binary collision-freedom isn't enough.** A path that grazes an obstacle by 2 mm is technically valid and practically terrifying, and tracking error eats that margin instantly. So I wrapped the checker:

```python
class ClearanceAwareChecker:
    def is_collision_free(self, q):
        return self.base.is_collision_free(q) and (
            self.base.compute_min_obstacle_distance(q) >= self.min_clearance
        )
```

with `MIN_CLEARANCE_M = 0.03`. The planner now refuses configurations closer than 3 cm to any obstacle. This makes planning harder — you've deliberately shrunk the free space — but it's the difference between a path that is valid and a path that is safe.

**Per-segment planning needs bounded sampling.** Planning eight segments across the full 6-D joint space with RRT* is glacial. Each segment gets a sampling box derived from its own endpoints, expanded by `PLANNER_SAMPLING_MARGIN` and clipped to the joint limits. And because a single RRT* run can fail on a hard segment, each retries across a seed bank `(42, 17, 7, 100, 200, 333, 500)` until one succeeds — determinism via seeding, robustness via the bank.

---

## The Numbers, Honestly

The validated slalom run, from `tasks/PLAN.md`:

| Metric | Value |
|---|---|
| Slalom Cartesian waypoints | 9 |
| Path waypoints after shortcutting | 24 |
| Planning time (8 RRT* segments) | ~168 s |
| Trajectory duration | 15.22 s |
| RMS tracking error | 0.0027 rad |
| Final position error | 0.0018 rad |
| Min waypoint obstacle clearance | 0.034 m |
| Test suite | 45 passed |

The numbers deserve commentary.

**168 seconds of planning for 15 seconds of motion** is the honest cost of RRT*: eleven times more time thinking than moving. RRT* burns its full iteration budget improving the path instead of stopping at the first solution, eight times over. That's a genuine limitation, not a tuning failure — real systems reach for informed sampling, bidirectional trees or precomputed roadmaps precisely because raw RRT* doesn't amortize.

**0.0027 rad RMS is a controller result, not a planner result.** It beats the pre-redesign capstone's 0.0125 rad, but the reason is timing: conservative scaling (`VEL_SCALE = 0.18`, `ACC_SCALE = 0.14`) means the trajectory asks less of the controller. Slower motion tracks better. Calling it improved control quality would be misleading.

**0.034 m minimum clearance against a 0.03 m threshold** is a 4 mm margin — tight, which is exactly what you expect when the planner optimizes against a constraint. The constraint is doing real work.

Reconciling these numbers also turned up a committed `slalom_metrics.json` describing a 17-waypoint "round trip" run with `minimum_obstacle_clearance_m: 0.0`, contradicting every figure above. It was emitted by a script deleted during the redesign, so nothing could correct or regenerate it. A metrics file with no living producer is worse than none, because it reads as authoritative. Deleted.

---

## What Lab 4 Actually Taught Me

**1. Planner and controller must share one notion of collision.** Every "the path was collision-free but the arm hit something" bug traces back to two models of the world.

**2. Not every geometric contact is planning signal.** Adjacent links in a kinematic chain touch by design, and a checker that doesn't know this reports the robot as permanently broken. `adjacency_gap` is a modelling policy, and policies belong in named parameters.

**3. Collision-free waypoints do not imply a collision-free trajectory.** Between waypoints there's an edge; after parameterization there's a spline. Both can leave the corridor. Check edges at adequate resolution; densify before splining.

**4. Design demos where the naive solution provably fails.** If a straight line also works, your demo demonstrates nothing. The slalom's staggered layout guarantees every segment is blocked — that guarantee is the deliverable.

**5. Clearance beats binary feasibility.** Zero-clearance paths are valid and useless. A margin costs free space and buys a trajectory that survives tracking error.

**6. Keep the API when the implementation has to compromise.** TOPP-RA wouldn't build; `parameterize_topp_ra(...)` still exists, still enforces the limits, still returns the same four things.

---

## What's Next

Lab 5 puts a gripper on this arm and asks it to pick something up. Every piece of Lab 4 gets imported directly: `RRTStarPlanner` for approach and transport motions, `parameterize_topp_ra` for the timing, the collision checker for validating grasp configurations against the table. The shift is that the world stops being static — a grasped object moves with the arm, changing the collision geometry mid-trajectory, and that's a problem Lab 4's planner has no vocabulary for.

Lab 9 is the destination: a VLA-controlled humanoid. Lab 4's contribution to that is one piece of intuition — a learned policy proposing "move the hand there" is proposing a *point*, and something still has to find a safe *path*. That responsibility doesn't disappear when the high-level controller becomes a neural network. It just moves.

---

*Code: [github.com/ozkannceylan/mujoco-robotics-lab](https://github.com/ozkannceylan/mujoco-robotics-lab)*
