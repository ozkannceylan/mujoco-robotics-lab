"""Lab 6 — Dual-arm collision checker using Pinocchio + HPP-FCL.

Architecture
------------
Two separate Pinocchio collision models are maintained (one per arm) for
self-collision and environment-collision checking.  Cross-arm collision
(left vs right) is resolved by querying HPP-FCL directly, using the
world-frame geometry placements that have already been computed for each
arm's individual model.

Collision checks performed:
  1. Left arm self-collision  (adjacent-link pairs filtered, gap=1)
  2. Right arm self-collision (adjacent-link pairs filtered, gap=1)
  3. Left vs right cross-arm collision (all left geoms vs all right geoms)
  4. Left arm vs environment (table)
  5. Right arm vs environment (table)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import hppfcl
import numpy as np
import pinocchio as pin

from lab6_common import (
    LEFT_BASE_SE3,
    RIGHT_BASE_SE3,
    TABLE_SPEC,
    URDF_PATH,
    ObstacleSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _se3_to_hppfcl(tf: pin.SE3) -> hppfcl.Transform3f:
    """Convert a Pinocchio SE3 to an HPP-FCL Transform3f."""
    return hppfcl.Transform3f(tf.rotation, tf.translation)


def _build_arm_geom_model(
    urdf_path: Path,
    kin_model: pin.Model,
    obstacle_specs: list[ObstacleSpec],
    adjacency_gap: int,
) -> tuple[pin.GeometryModel, int]:
    """Build a GeometryModel for one arm, register all collision pairs.

    Obstacles are added as fixed box geometries (parented to universe).
    Self-collision pairs are registered with adjacency filtering.
    Each robot geom is paired with each obstacle geom.

    Args:
        urdf_path: URDF with collision meshes / primitives.
        kin_model: Pinocchio kinematic model already built from the same URDF.
        obstacle_specs: Environment obstacles to include.
        adjacency_gap: Skip self-collision pairs whose parent joints differ
            by at most this value.

    Returns:
        Tuple of (geom_model, n_robot_geoms) where n_robot_geoms is the count
        of robot geometry objects (obstacle geoms come after).
    """
    geom_model = pin.buildGeomFromUrdf(kin_model, str(urdf_path), pin.COLLISION)
    n_robot_geoms = geom_model.ngeoms

    # --- add environment obstacles ---
    obstacle_gids: list[int] = []
    for spec in obstacle_specs:
        full_extents = 2.0 * spec.half_extents
        shape = hppfcl.Box(float(full_extents[0]),
                           float(full_extents[1]),
                           float(full_extents[2]))
        placement = pin.SE3(np.eye(3), spec.position.copy())
        geom_obj = pin.GeometryObject(
            spec.name,   # name
            0,           # parent_joint (universe = 0)
            0,           # parent_frame (universe = 0)
            placement,   # SE3 placement in world frame
            shape,       # HPP-FCL collision geometry
        )
        gid = geom_model.addGeometryObject(geom_obj)
        obstacle_gids.append(gid)

    # --- robot vs obstacles ---
    for robot_gid in range(n_robot_geoms):
        for obs_gid in obstacle_gids:
            geom_model.addCollisionPair(pin.CollisionPair(robot_gid, obs_gid))

    # --- self-collision (skip adjacent/same-joint pairs) ---
    for i in range(n_robot_geoms):
        ji = geom_model.geometryObjects[i].parentJoint
        for j in range(i + 1, n_robot_geoms):
            jj = geom_model.geometryObjects[j].parentJoint
            if abs(int(ji) - int(jj)) <= adjacency_gap:
                continue
            geom_model.addCollisionPair(pin.CollisionPair(i, j))

    return geom_model, n_robot_geoms


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DualCollisionChecker:
    """Collision checker for two UR5e arms in a shared workspace.

    Each arm has its own Pinocchio kinematic + geometry model.  After
    forward kinematics, geometry placements are transformed into the world
    frame by the arm's base SE3.  Cross-arm collisions are evaluated by
    calling HPP-FCL directly on the world-frame placements.

    Args:
        urdf_path: Path to the UR5e URDF with collision geometry.
            Defaults to ``lab6_common.URDF_PATH``.
        obstacle_specs: Environment obstacles shared by both arms.
            Defaults to ``[TABLE_SPEC]``.
        adjacency_gap: Self-collision pairs whose parent joints differ by
            at most this value are skipped (adjacent links cannot collide
            in practice). Default 1.
        cross_arm_margin: Extra clearance (m) used only for cross-arm
            distance reporting; does not affect binary collision result.
    """

    def __init__(
        self,
        urdf_path: Path | None = None,
        obstacle_specs: list[ObstacleSpec] | None = None,
        adjacency_gap: int = 1,
    ) -> None:
        self._urdf_path: Path = urdf_path if urdf_path is not None else URDF_PATH
        self._obstacle_specs: list[ObstacleSpec] = (
            obstacle_specs if obstacle_specs is not None else [TABLE_SPEC]
        )
        self._adjacency_gap = adjacency_gap

        # --- Left arm ---
        self.model_left: pin.Model = pin.buildModelFromUrdf(str(self._urdf_path))
        self.geom_model_left, self._n_left_robot_geoms = _build_arm_geom_model(
            self._urdf_path,
            self.model_left,
            self._obstacle_specs,
            adjacency_gap,
        )
        self.data_left: pin.Data = self.model_left.createData()
        self.geom_data_left: pin.GeometryData = pin.GeometryData(self.geom_model_left)

        # --- Right arm ---
        self.model_right: pin.Model = pin.buildModelFromUrdf(str(self._urdf_path))
        self.geom_model_right, self._n_right_robot_geoms = _build_arm_geom_model(
            self._urdf_path,
            self.model_right,
            self._obstacle_specs,
            adjacency_gap,
        )
        self.data_right: pin.Data = self.model_right.createData()
        self.geom_data_right: pin.GeometryData = pin.GeometryData(self.geom_model_right)

        # --- Base transforms ---
        self._base_left: pin.SE3 = LEFT_BASE_SE3
        self._base_right: pin.SE3 = RIGHT_BASE_SE3

    # ------------------------------------------------------------------
    # Internal update helpers
    # ------------------------------------------------------------------

    def _update_left(self, q_left: np.ndarray) -> None:
        """Run FK + geometry placement update for the left arm.

        Robot geometry placements are post-multiplied by the left base SE3
        so they live in the shared world frame.  Obstacle placements are
        already in world frame and are left unchanged.

        Args:
            q_left: Left-arm joint angles (6,).
        """
        pin.forwardKinematics(self.model_left, self.data_left, q_left)
        pin.updateGeometryPlacements(
            self.model_left, self.data_left,
            self.geom_model_left, self.geom_data_left,
        )
        # Apply base transform: world_T_geom = world_T_base * base_T_geom
        for i in range(self._n_left_robot_geoms):
            self.geom_data_left.oMg[i] = (
                self._base_left * self.geom_data_left.oMg[i]
            )

    def _update_right(self, q_right: np.ndarray) -> None:
        """Run FK + geometry placement update for the right arm.

        Args:
            q_right: Right-arm joint angles (6,).
        """
        pin.forwardKinematics(self.model_right, self.data_right, q_right)
        pin.updateGeometryPlacements(
            self.model_right, self.data_right,
            self.geom_model_right, self.geom_data_right,
        )
        for i in range(self._n_right_robot_geoms):
            self.geom_data_right.oMg[i] = (
                self._base_right * self.geom_data_right.oMg[i]
            )

    # ------------------------------------------------------------------
    # Individual collision checks
    # ------------------------------------------------------------------

    def _check_self_and_env(
        self,
        model: pin.Model,
        data: pin.Data,
        geom_model: pin.GeometryModel,
        geom_data: pin.GeometryData,
        q: np.ndarray,
        stop_at_first: bool = True,
    ) -> bool:
        """Run Pinocchio's built-in collision check (self + env pairs).

        NOTE: This call re-runs FK + placement updates internally using the
        *unshifted* model frame.  Because we already applied the base
        transform to geom_data.oMg in the _update_* helpers, we pass the
        already-updated geom_data but Pinocchio's computeCollisions will
        overwrite it.  To avoid double-applying the base transform, we
        perform a raw check by iterating collision pairs manually using the
        already-updated placements stored in geom_data.

        Args:
            model: Pinocchio kinematic model.
            data: Pinocchio data (already FK-updated).
            geom_model: GeometryModel with registered collision pairs.
            geom_data: GeometryData with world-frame placements (oMg updated).
            q: Joint angles (used to re-trigger FK inside computeCollisions).
            stop_at_first: Stop as soon as the first collision is detected.

        Returns:
            True if at least one registered pair is in collision.
        """
        # We iterate manually so we use the world-frame placements already
        # stored in geom_data (which include the base transform offset).
        req = hppfcl.CollisionRequest()
        req.enable_contact = False

        for cp in geom_model.collisionPairs:
            i, j = int(cp.first), int(cp.second)
            geom_i = geom_model.geometryObjects[i]
            geom_j = geom_model.geometryObjects[j]
            tf_i = _se3_to_hppfcl(geom_data.oMg[i])
            tf_j = _se3_to_hppfcl(geom_data.oMg[j])
            res = hppfcl.CollisionResult()
            hppfcl.collide(geom_i.geometry, tf_i, geom_j.geometry, tf_j, req, res)
            if res.isCollision():
                return True
        return False

    def _check_cross_arm(self) -> bool:
        """Check all left robot geoms vs all right robot geoms for collision.

        Uses world-frame placements that must have been computed by
        _update_left / _update_right before calling this method.

        Returns:
            True if any left-right pair is in collision.
        """
        req = hppfcl.CollisionRequest()
        req.enable_contact = False

        for i in range(self._n_left_robot_geoms):
            geom_i = self.geom_model_left.geometryObjects[i]
            tf_i = _se3_to_hppfcl(self.geom_data_left.oMg[i])
            for j in range(self._n_right_robot_geoms):
                geom_j = self.geom_model_right.geometryObjects[j]
                tf_j = _se3_to_hppfcl(self.geom_data_right.oMg[j])
                res = hppfcl.CollisionResult()
                hppfcl.collide(
                    geom_i.geometry, tf_i,
                    geom_j.geometry, tf_j,
                    req, res,
                )
                if res.isCollision():
                    return True
        return False

    def _min_distance_self_and_env(
        self,
        geom_model: pin.GeometryModel,
        geom_data: pin.GeometryData,
    ) -> float:
        """Compute minimum HPP-FCL distance for all registered pairs.

        Uses already-updated world-frame placements from geom_data.

        Returns:
            Minimum distance (negative = penetration).
        """
        min_dist = np.inf
        req = hppfcl.DistanceRequest()

        for cp in geom_model.collisionPairs:
            i, j = int(cp.first), int(cp.second)
            geom_i = geom_model.geometryObjects[i]
            geom_j = geom_model.geometryObjects[j]
            tf_i = _se3_to_hppfcl(geom_data.oMg[i])
            tf_j = _se3_to_hppfcl(geom_data.oMg[j])
            res = hppfcl.DistanceResult()
            dist = hppfcl.distance(geom_i.geometry, tf_i, geom_j.geometry, tf_j, req, res)
            if dist < min_dist:
                min_dist = dist

        return float(min_dist)

    def _min_distance_cross_arm(self) -> float:
        """Compute minimum distance between all left and right robot geoms.

        Returns:
            Minimum distance (negative = penetration).
        """
        min_dist = np.inf
        req = hppfcl.DistanceRequest()

        for i in range(self._n_left_robot_geoms):
            geom_i = self.geom_model_left.geometryObjects[i]
            tf_i = _se3_to_hppfcl(self.geom_data_left.oMg[i])
            for j in range(self._n_right_robot_geoms):
                geom_j = self.geom_model_right.geometryObjects[j]
                tf_j = _se3_to_hppfcl(self.geom_data_right.oMg[j])
                res = hppfcl.DistanceResult()
                dist = hppfcl.distance(
                    geom_i.geometry, tf_i,
                    geom_j.geometry, tf_j,
                    req, res,
                )
                if dist < min_dist:
                    min_dist = dist

        return float(min_dist)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_collision_free(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
    ) -> bool:
        """Check whether the dual-arm configuration is collision-free.

        Checks (in order, short-circuiting on first collision found):
          1. Left arm self-collision + left vs environment
          2. Right arm self-collision + right vs environment
          3. Left arm vs right arm cross-collision

        Args:
            q_left: Left-arm joint angles, shape (6,).
            q_right: Right-arm joint angles, shape (6,).

        Returns:
            True if no collision is detected in any check.
        """
        self._update_left(q_left)
        self._update_right(q_right)

        # 1. Left self + env
        if self._check_self_and_env(
            self.model_left, self.data_left,
            self.geom_model_left, self.geom_data_left,
            q_left,
        ):
            return False

        # 2. Right self + env
        if self._check_self_and_env(
            self.model_right, self.data_right,
            self.geom_model_right, self.geom_data_right,
            q_right,
        ):
            return False

        # 3. Cross-arm
        if self._check_cross_arm():
            return False

        return True

    def get_min_distance(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
    ) -> float:
        """Compute the minimum signed distance across all collision pairs.

        Covers self-collision, environment, and cross-arm pairs for both
        arms.  A negative return value indicates penetration.

        Args:
            q_left: Left-arm joint angles, shape (6,).
            q_right: Right-arm joint angles, shape (6,).

        Returns:
            Minimum signed distance in metres.
        """
        self._update_left(q_left)
        self._update_right(q_right)

        d_left = self._min_distance_self_and_env(
            self.geom_model_left, self.geom_data_left
        )
        d_right = self._min_distance_self_and_env(
            self.geom_model_right, self.geom_data_right
        )
        d_cross = self._min_distance_cross_arm()

        return float(min(d_left, d_right, d_cross))

    def is_path_free(
        self,
        q_left_start: np.ndarray,
        q_left_end: np.ndarray,
        q_right_start: np.ndarray,
        q_right_end: np.ndarray,
        resolution: float = 0.05,
    ) -> bool:
        """Check whether a linearly interpolated dual-arm path is collision-free.

        Both arms interpolate linearly in joint space simultaneously.  The
        path is discretised so that the maximum step in the combined
        joint-space L2 norm is at most ``resolution`` radians.

        Args:
            q_left_start: Left-arm start configuration (6,).
            q_left_end: Left-arm end configuration (6,).
            q_right_start: Right-arm start configuration (6,).
            q_right_end: Right-arm end configuration (6,).
            resolution: Maximum step size in joint-space L2 norm (rad).

        Returns:
            True if every sampled configuration is collision-free.
        """
        diff_left = q_left_end - q_left_start
        diff_right = q_right_end - q_right_start

        # Use the larger of the two arm distances to determine step count
        dist = max(np.linalg.norm(diff_left), np.linalg.norm(diff_right))

        if dist < 1e-9:
            return self.is_collision_free(q_left_start, q_right_start)

        n_steps = max(2, int(np.ceil(dist / resolution)) + 1)

        for i in range(n_steps):
            alpha = i / (n_steps - 1)
            q_l = q_left_start + alpha * diff_left
            q_r = q_right_start + alpha * diff_right
            if not self.is_collision_free(q_l, q_r):
                return False

        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_left_pairs(self) -> int:
        """Number of registered collision pairs for the left arm."""
        return len(self.geom_model_left.collisionPairs)

    @property
    def num_right_pairs(self) -> int:
        """Number of registered collision pairs for the right arm."""
        return len(self.geom_model_right.collisionPairs)

    @property
    def num_cross_pairs(self) -> int:
        """Number of cross-arm pairs checked (left_geoms * right_geoms)."""
        return self._n_left_robot_geoms * self._n_right_robot_geoms

    @property
    def num_total_pairs(self) -> int:
        """Total number of pairs checked across all collision categories."""
        return self.num_left_pairs + self.num_right_pairs + self.num_cross_pairs
