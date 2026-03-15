"""Lab 4 — Collision checker using Pinocchio + HPP-FCL.

Provides self-collision and environment-collision checking for the UR5e.
Environment obstacles are axis-aligned boxes added to Pinocchio's GeometryModel.
"""

from __future__ import annotations

from pathlib import Path

import hppfcl
import numpy as np
import pinocchio as pin

from lab4_common import OBSTACLES, TABLE_SPEC, URDF_PATH, ObstacleSpec


class CollisionChecker:
    """Collision checker wrapping Pinocchio GeometryModel + HPP-FCL.

    Checks both self-collision (with adjacent-link filtering) and
    environment collisions against box obstacles.

    Args:
        urdf_path: Path to URDF with collision geometries.
        obstacle_specs: List of box obstacles. Defaults to OBSTACLES + TABLE_SPEC.
        self_collision: Whether to check self-collision pairs.
        adjacency_gap: Skip self-collision pairs where parent joints differ
            by at most this value (adjacent links can't collide in practice).
    """

    def __init__(
        self,
        urdf_path: Path | None = None,
        obstacle_specs: list[ObstacleSpec] | None = None,
        self_collision: bool = True,
        adjacency_gap: int = 1,
    ) -> None:
        if urdf_path is None:
            urdf_path = URDF_PATH
        if obstacle_specs is None:
            obstacle_specs = OBSTACLES + [TABLE_SPEC]

        # Build kinematic + collision models from URDF
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.geom_model = pin.buildGeomFromUrdf(
            self.model, str(urdf_path), pin.COLLISION
        )
        self._n_robot_geoms = self.geom_model.ngeoms

        # Add environment obstacles as fixed boxes (attached to universe joint 0)
        self._obstacle_ids: list[int] = []
        for spec in obstacle_specs:
            full_extents = 2.0 * spec.half_extents
            shape = hppfcl.Box(full_extents[0], full_extents[1], full_extents[2])
            placement = pin.SE3(np.eye(3), spec.position.copy())
            geom_obj = pin.GeometryObject(
                spec.name,           # name
                0,                   # parent_joint (universe)
                0,                   # parent_frame (universe)
                placement,           # SE3 placement
                shape,               # collision geometry
            )
            gid = self.geom_model.addGeometryObject(geom_obj)
            self._obstacle_ids.append(gid)

        # Register collision pairs
        # 1. Robot vs obstacles
        for robot_gid in range(self._n_robot_geoms):
            for obs_gid in self._obstacle_ids:
                self.geom_model.addCollisionPair(
                    pin.CollisionPair(robot_gid, obs_gid)
                )

        # 2. Self-collision (skip adjacent links)
        if self_collision:
            for i in range(self._n_robot_geoms):
                ji = self.geom_model.geometryObjects[i].parentJoint
                for j in range(i + 1, self._n_robot_geoms):
                    jj = self.geom_model.geometryObjects[j].parentJoint
                    if abs(ji - jj) <= adjacency_gap:
                        continue  # skip adjacent or same-joint pairs
                    self.geom_model.addCollisionPair(pin.CollisionPair(i, j))

        # Create data objects
        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Check if configuration q is collision-free.

        Args:
            q: Joint angles (6,).

        Returns:
            True if no collision detected.
        """
        pin.computeCollisions(
            self.model, self.data,
            self.geom_model, self.geom_data,
            q, True,  # stop_at_first_collision=True
        )
        return not any(
            cr.isCollision() for cr in self.geom_data.collisionResults
        )

    def is_path_free(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        resolution: float = 0.05,
    ) -> bool:
        """Check if linear interpolation from q1 to q2 is collision-free.

        Samples configurations at intervals of `resolution` (rad in C-space norm).

        Args:
            q1: Start configuration (6,).
            q2: End configuration (6,).
            resolution: Maximum step size in joint-space L2 norm.

        Returns:
            True if entire path is collision-free.
        """
        diff = q2 - q1
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return self.is_collision_free(q1)

        n_steps = max(2, int(np.ceil(dist / resolution)) + 1)
        for i in range(n_steps):
            alpha = i / (n_steps - 1)
            q = q1 + alpha * diff
            if not self.is_collision_free(q):
                return False
        return True

    def compute_min_distance(self, q: np.ndarray) -> float:
        """Compute minimum distance between any collision pair at q.

        Args:
            q: Joint angles (6,).

        Returns:
            Minimum distance (negative means penetration).
        """
        pin.computeDistances(
            self.model, self.data,
            self.geom_model, self.geom_data,
            q,
        )
        return min(
            dr.min_distance for dr in self.geom_data.distanceResults
        )

    @property
    def num_collision_pairs(self) -> int:
        """Number of registered collision pairs."""
        return len(self.geom_model.collisionPairs)
