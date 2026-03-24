"""Lab 4 — Collision checker on the canonical executed MuJoCo geometry."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from lab4_common import OBSTACLES, URDF_PATH, ObstacleSpec, load_mujoco_model, load_pinocchio_model


class CollisionChecker:
    """Collision checker wrapping the canonical UR5e + Robotiq MuJoCo scene.

    Planning uses the same executed collision geometry as the simulator rather
    than a separate simplified collision model. Pinocchio is still loaded for
    FK-based visualization and compatibility with the rest of the lab.
    """

    def __init__(
        self,
        urdf_path: Path | None = None,
        obstacle_specs: list[ObstacleSpec] | tuple[ObstacleSpec, ...] | None = None,
        self_collision: bool = True,
        adjacency_gap: int = 1,
        include_table: bool | None = None,
    ) -> None:
        if obstacle_specs is None:
            obstacle_specs = list(OBSTACLES)
            _include_table = True if include_table is None else include_table
        else:
            obstacle_specs = list(obstacle_specs)
            _include_table = (len(obstacle_specs) > 0) if include_table is None else include_table
        include_table = _include_table

        self.obstacle_specs = obstacle_specs
        self.self_collision = self_collision
        self.adjacency_gap = adjacency_gap

        self.mj_model, self.mj_data = load_mujoco_model(
            obstacle_specs=tuple(obstacle_specs),
            include_table=include_table,
        )
        self.model, self.data, self.ee_fid = load_pinocchio_model(urdf_path or URDF_PATH)

        self._body_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            for body_id in range(self.mj_model.nbody)
        ]
        self._geom_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            for geom_id in range(self.mj_model.ngeom)
        ]

        obstacle_body_names = {spec.name for spec in obstacle_specs}
        if include_table:
            obstacle_body_names.add("table")

        self._environment_body_ids = {
            body_id
            for body_id, name in enumerate(self._body_names)
            if name in obstacle_body_names
        }
        self._robot_body_ids = {
            body_id
            for body_id, name in enumerate(self._body_names)
            if body_id != 0 and body_id not in self._environment_body_ids and name != "world"
        }

        self._robot_geom_ids = [
            geom_id
            for geom_id in range(self.mj_model.ngeom)
            if self.mj_model.geom_contype[geom_id] != 0
            and self.mj_model.geom_bodyid[geom_id] in self._robot_body_ids
        ]
        self._environment_geom_ids = [
            geom_id
            for geom_id in range(self.mj_model.ngeom)
            if self.mj_model.geom_contype[geom_id] != 0
            and self.mj_model.geom_bodyid[geom_id] in self._environment_body_ids
        ]
        self._table_geom_ids = [
            geom_id
            for geom_id in self._environment_geom_ids
            if self._body_names[self.mj_model.geom_bodyid[geom_id]] == "table"
        ]
        self._obstacle_geom_ids = [
            geom_id for geom_id in self._environment_geom_ids if geom_id not in self._table_geom_ids
        ]
        self._n_robot_geoms = len(self._robot_geom_ids)
        self._obstacle_ids = list(self._environment_geom_ids)

        self._contact_pairs: set[tuple[int, int]] = set()
        self._distance_pairs: list[tuple[int, int]] = []
        self._environment_pairs: list[tuple[int, int]] = []
        self._table_pairs: list[tuple[int, int]] = []
        self._obstacle_pairs: list[tuple[int, int]] = []
        self._self_pairs: list[tuple[int, int]] = []

        for robot_gid in self._robot_geom_ids:
            for env_gid in self._environment_geom_ids:
                pair = self._register_pair(robot_gid, env_gid)
                self._environment_pairs.append(pair)
                if env_gid in self._table_geom_ids:
                    self._table_pairs.append(pair)
                else:
                    self._obstacle_pairs.append(pair)

        if self_collision:
            for i, geom1 in enumerate(self._robot_geom_ids):
                for geom2 in self._robot_geom_ids[i + 1 :]:
                    if not self._is_self_pair_enabled(geom1, geom2):
                        continue
                    pair = self._register_pair(geom1, geom2)
                    self._self_pairs.append(pair)

    def _register_pair(self, geom1: int, geom2: int) -> tuple[int, int]:
        pair = (min(geom1, geom2), max(geom1, geom2))
        if pair not in self._contact_pairs:
            self._contact_pairs.add(pair)
            self._distance_pairs.append(pair)
        return pair

    def _body_tree_distance(self, body1: int, body2: int) -> int:
        """Compute tree distance between two MuJoCo bodies."""
        ancestors: dict[int, int] = {}
        cur = body1
        dist = 0
        while True:
            ancestors[cur] = dist
            if cur == 0:
                break
            cur = int(self.mj_model.body_parentid[cur])
            dist += 1

        cur = body2
        dist = 0
        while True:
            if cur in ancestors:
                return ancestors[cur] + dist
            if cur == 0:
                return dist
            cur = int(self.mj_model.body_parentid[cur])
            dist += 1

    def _is_self_pair_enabled(self, geom1: int, geom2: int) -> bool:
        """Return whether this robot-robot geom pair should be checked."""
        body1 = int(self.mj_model.geom_bodyid[geom1])
        body2 = int(self.mj_model.geom_bodyid[geom2])
        if body1 == body2:
            return False
        name1 = self._body_names[body1]
        name2 = self._body_names[body2]
        if name1.startswith("2f85_") and name2.startswith("2f85_"):
            return False
        return self._body_tree_distance(body1, body2) > self.adjacency_gap

    def _forward(self, q: np.ndarray) -> None:
        self.mj_data.qpos[: self.mj_model.nq] = 0.0
        self.mj_data.qvel[: self.mj_model.nv] = 0.0
        self.mj_data.qpos[: len(q)] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Check whether configuration q is collision-free."""
        self._forward(q)
        for contact_id in range(self.mj_data.ncon):
            contact = self.mj_data.contact[contact_id]
            pair = (min(contact.geom1, contact.geom2), max(contact.geom1, contact.geom2))
            if pair in self._contact_pairs:
                return False
        return True

    def is_path_free(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        resolution: float = 0.05,
    ) -> bool:
        """Check whether the straight-line joint-space path is collision-free."""
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

    def _compute_min_distance_for_pairs(
        self,
        q: np.ndarray,
        pairs: list[tuple[int, int]],
    ) -> float:
        """Compute the minimum signed distance over a selected pair set."""
        self._forward(q)
        if not pairs:
            return float("inf")

        fromto = np.zeros(6, dtype=float)
        min_distance = float("inf")
        for geom1, geom2 in pairs:
            distance = mujoco.mj_geomDistance(
                self.mj_model, self.mj_data, geom1, geom2, 10.0, fromto
            )
            min_distance = min(min_distance, float(distance))
        if abs(min_distance) < 1e-8:
            return 0.0
        return min_distance

    def compute_min_distance(
        self,
        q: np.ndarray,
        include_self: bool = True,
        include_table: bool = True,
        include_obstacles: bool = True,
    ) -> float:
        """Compute the minimum signed distance over selected collision buckets."""
        pairs: list[tuple[int, int]] = []
        if include_obstacles:
            pairs.extend(self._obstacle_pairs)
        if include_table:
            pairs.extend(self._table_pairs)
        if include_self:
            pairs.extend(self._self_pairs)
        return self._compute_min_distance_for_pairs(q, pairs)

    def compute_min_environment_distance(
        self,
        q: np.ndarray,
        include_table: bool = True,
    ) -> float:
        """Compute the minimum distance between robot and environment geometry."""
        return self.compute_min_distance(
            q,
            include_self=False,
            include_table=include_table,
            include_obstacles=True,
        )

    def compute_min_obstacle_distance(self, q: np.ndarray) -> float:
        """Compute the minimum distance between robot and obstacle boxes."""
        return self.compute_min_distance(
            q,
            include_self=False,
            include_table=False,
            include_obstacles=True,
        )

    @property
    def num_collision_pairs(self) -> int:
        """Number of monitored collision pairs."""
        return len(self._distance_pairs)
