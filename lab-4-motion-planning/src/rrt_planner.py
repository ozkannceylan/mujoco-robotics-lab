"""Lab 4 — RRT and RRT* planner for UR5e in configuration space.

Implements sampling-based motion planning with:
- Basic RRT: grow tree toward random samples
- RRT*: rewiring for asymptotically optimal paths
- Goal bias: configurable fraction of samples aimed at goal
- Edge collision checking via CollisionChecker.is_path_free
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
    Axes3D = None  # visualization only — safe to skip

from collision_checker import CollisionChecker
from lab4_common import JOINT_LOWER, JOINT_UPPER, MEDIA_DIR, ObstacleSpec


@dataclass
class RRTNode:
    """Node in the RRT search tree.

    Attributes:
        q: Joint configuration (6,).
        parent: Index of parent node in the tree (None for root).
        cost: Cumulative path cost from root (L2 in C-space).
    """

    q: np.ndarray
    parent: int | None = None
    cost: float = 0.0


class RRTStarPlanner:
    """RRT / RRT* planner for collision-free C-space paths.

    Args:
        collision_checker: Collision checker instance.
        joint_limits_lower: Lower joint limits (6,).
        joint_limits_upper: Upper joint limits (6,).
        step_size: Maximum extension distance per step (rad in C-space L2).
        goal_bias: Probability of sampling the goal configuration.
        rewire_radius: Radius for RRT* rewiring neighborhood.
        goal_tolerance: L2 distance to goal for success.
    """

    def __init__(
        self,
        collision_checker: CollisionChecker,
        joint_limits_lower: np.ndarray | None = None,
        joint_limits_upper: np.ndarray | None = None,
        step_size: float = 0.3,
        goal_bias: float = 0.1,
        rewire_radius: float = 1.0,
        goal_tolerance: float = 0.15,
    ) -> None:
        self.cc = collision_checker
        self.lower = joint_limits_lower if joint_limits_lower is not None else JOINT_LOWER
        self.upper = joint_limits_upper if joint_limits_upper is not None else JOINT_UPPER
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius
        self.goal_tolerance = goal_tolerance
        self._tree: list[RRTNode] = []
        self._rng = np.random.default_rng()

    @property
    def tree(self) -> list[RRTNode]:
        """Access the search tree for visualization."""
        return self._tree

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        max_iter: int = 5000,
        rrt_star: bool = True,
        seed: int | None = None,
    ) -> list[np.ndarray] | None:
        """Plan a collision-free path from q_start to q_goal.

        Args:
            q_start: Start configuration (6,).
            q_goal: Goal configuration (6,).
            max_iter: Maximum number of iterations.
            rrt_star: If True, use RRT* with rewiring. If False, basic RRT.
            seed: Random seed for reproducibility.

        Returns:
            List of waypoint configurations from start to goal, or None if
            no path found within max_iter.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Validate start and goal
        if not self.cc.is_collision_free(q_start):
            return None
        if not self.cc.is_collision_free(q_goal):
            return None

        # Initialize tree with start node
        self._tree = [RRTNode(q=q_start.copy(), parent=None, cost=0.0)]
        goal_node_idx: int | None = None

        for _iteration in range(max_iter):
            # Sample random configuration (with goal bias)
            q_rand = self._sample(q_goal)

            # Find nearest node in tree
            nearest_idx = self._nearest(q_rand)
            q_nearest = self._tree[nearest_idx].q

            # Steer toward sample
            q_new = self._steer(q_nearest, q_rand)

            # Check edge collision
            if not self.cc.is_path_free(q_nearest, q_new):
                continue

            if rrt_star:
                # RRT*: find best parent in neighborhood
                new_idx = self._extend_rrt_star(q_new, nearest_idx)
            else:
                # Basic RRT: just add with nearest as parent
                cost = self._tree[nearest_idx].cost + self._dist(q_nearest, q_new)
                new_node = RRTNode(q=q_new, parent=nearest_idx, cost=cost)
                self._tree.append(new_node)
                new_idx = len(self._tree) - 1

            # Check if we reached the goal
            if self._dist(q_new, q_goal) < self.goal_tolerance:
                if self.cc.is_path_free(q_new, q_goal):
                    cost_to_goal = (
                        self._tree[new_idx].cost + self._dist(q_new, q_goal)
                    )
                    # Only update if better path found
                    if goal_node_idx is None or cost_to_goal < self._tree[goal_node_idx].cost:
                        if goal_node_idx is None:
                            goal_node = RRTNode(
                                q=q_goal.copy(),
                                parent=new_idx,
                                cost=cost_to_goal,
                            )
                            self._tree.append(goal_node)
                            goal_node_idx = len(self._tree) - 1
                        else:
                            self._tree[goal_node_idx].parent = new_idx
                            self._tree[goal_node_idx].cost = cost_to_goal

                    # For basic RRT, return immediately
                    if not rrt_star:
                        return self._extract_path(goal_node_idx)

        if goal_node_idx is not None:
            return self._extract_path(goal_node_idx)
        return None

    def _sample(self, q_goal: np.ndarray) -> np.ndarray:
        """Sample a random configuration with goal bias."""
        if self._rng.random() < self.goal_bias:
            return q_goal.copy()
        return self._rng.uniform(self.lower, self.upper)

    def _nearest(self, q: np.ndarray) -> int:
        """Find index of nearest node in tree to q."""
        dists = [self._dist(node.q, q) for node in self._tree]
        return int(np.argmin(dists))

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        """Steer from q_from toward q_to, limited by step_size."""
        diff = q_to - q_from
        dist = np.linalg.norm(diff)
        if dist <= self.step_size:
            return q_to.copy()
        return q_from + (diff / dist) * self.step_size

    def _dist(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """L2 distance in configuration space."""
        return float(np.linalg.norm(q1 - q2))

    def _near(self, q: np.ndarray) -> list[int]:
        """Find all nodes within rewire_radius of q."""
        return [
            i
            for i, node in enumerate(self._tree)
            if self._dist(node.q, q) <= self.rewire_radius
        ]

    def _extend_rrt_star(self, q_new: np.ndarray, nearest_idx: int) -> int:
        """Add q_new to tree with RRT* best-parent selection and rewiring.

        Args:
            q_new: New configuration to add.
            nearest_idx: Index of nearest node (default parent).

        Returns:
            Index of the new node in the tree.
        """
        # Find neighborhood
        near_indices = self._near(q_new)

        # Choose best parent from neighborhood
        best_parent = nearest_idx
        best_cost = self._tree[nearest_idx].cost + self._dist(
            self._tree[nearest_idx].q, q_new
        )

        for idx in near_indices:
            candidate_cost = self._tree[idx].cost + self._dist(
                self._tree[idx].q, q_new
            )
            if candidate_cost < best_cost:
                if self.cc.is_path_free(self._tree[idx].q, q_new):
                    best_parent = idx
                    best_cost = candidate_cost

        # Add new node
        new_node = RRTNode(q=q_new, parent=best_parent, cost=best_cost)
        self._tree.append(new_node)
        new_idx = len(self._tree) - 1

        # Rewire: check if new node provides shorter path to neighbors
        for idx in near_indices:
            if idx == best_parent:
                continue
            new_cost = best_cost + self._dist(q_new, self._tree[idx].q)
            if new_cost < self._tree[idx].cost:
                if self.cc.is_path_free(q_new, self._tree[idx].q):
                    self._tree[idx].parent = new_idx
                    self._propagate_cost(idx, new_cost)

        return new_idx

    def _propagate_cost(self, idx: int, new_cost: float) -> None:
        """Propagate updated cost to all descendants of node idx."""
        self._tree[idx].cost = new_cost
        # Find children and update recursively
        for i, node in enumerate(self._tree):
            if node.parent == idx:
                child_cost = new_cost + self._dist(
                    self._tree[idx].q, node.q
                )
                self._propagate_cost(i, child_cost)

    def _extract_path(self, goal_idx: int) -> list[np.ndarray]:
        """Extract path from root to goal by backtracking through parents."""
        path = []
        idx: int | None = goal_idx
        while idx is not None:
            path.append(self._tree[idx].q.copy())
            idx = self._tree[idx].parent
        path.reverse()
        return path


def visualize_plan(
    planner: RRTStarPlanner,
    path: list[np.ndarray],
    ee_fid: int,
    title: str = "RRT* Plan",
    save_path: Path | None = None,
) -> None:
    """Visualize the RRT tree and planned path in 3D task space.

    Projects tree nodes and path waypoints to EE positions using FK,
    then plots them in 3D alongside obstacle boxes.

    Args:
        planner: Planner with populated tree.
        path: Planned path (list of configurations).
        ee_fid: Pinocchio EE frame ID.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import pinocchio as pin
    from lab4_common import OBSTACLES, TABLE_SPEC, get_ee_pos

    model = planner.cc.model
    data = planner.cc.data

    matplotlib.use("Agg")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot tree edges (subsample for large trees)
    tree = planner.tree
    max_edges = 2000
    step = max(1, len(tree) // max_edges)
    for i in range(0, len(tree), step):
        node = tree[i]
        if node.parent is not None:
            p1 = get_ee_pos(model, data, ee_fid, tree[node.parent].q)
            p2 = get_ee_pos(model, data, ee_fid, node.q)
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color="lightblue", alpha=0.3, linewidth=0.5,
            )

    # Plot path
    if path:
        path_pos = np.array([get_ee_pos(model, data, ee_fid, q) for q in path])
        ax.plot(
            path_pos[:, 0], path_pos[:, 1], path_pos[:, 2],
            "g-o", linewidth=2.5, markersize=5, label="Path",
        )
        ax.scatter(*path_pos[0], c="blue", s=100, marker="^", label="Start")
        ax.scatter(*path_pos[-1], c="red", s=100, marker="*", label="Goal")

    # Plot obstacle boxes
    for obs in OBSTACLES + [TABLE_SPEC]:
        _draw_box(ax, obs.position, obs.half_extents, color="red" if obs.name != "table" else "brown")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _draw_box(
    ax: Axes3D,
    center: np.ndarray,
    half: np.ndarray,
    color: str = "red",
) -> None:
    """Draw a wireframe box on a 3D axes."""
    c, h = center, half
    # 8 corners
    corners = np.array([
        [c[0] - h[0], c[1] - h[1], c[2] - h[2]],
        [c[0] + h[0], c[1] - h[1], c[2] - h[2]],
        [c[0] + h[0], c[1] + h[1], c[2] - h[2]],
        [c[0] - h[0], c[1] + h[1], c[2] - h[2]],
        [c[0] - h[0], c[1] - h[1], c[2] + h[2]],
        [c[0] + h[0], c[1] - h[1], c[2] + h[2]],
        [c[0] + h[0], c[1] + h[1], c[2] + h[2]],
        [c[0] - h[0], c[1] + h[1], c[2] + h[2]],
    ])
    # 12 edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color, alpha=0.6, linewidth=1.0,
        )
