"""M2 — Pinocchio dual-arm FK, Jacobian, and DLS IK.

Wraps two independent Pinocchio UR5e models with world-frame base offsets
matching the MuJoCo scene_dual.xml layout.  Provides:
  - FK (position + orientation) for each arm
  - 6x6 geometric Jacobian in world frame for each arm
  - Damped Least-Squares IK for each arm
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pinocchio as pin

from lab6_common import (
    NUM_JOINTS_PER_ARM,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    URDF_PATH,
)

# Arm base positions in world frame (matching scene_dual.xml)
LEFT_BASE_POS = np.array([0.0, 0.0, 0.0])
RIGHT_BASE_POS = np.array([1.0, 0.0, 0.0])

# EE frame name in URDF
EE_FRAME_NAME = "ee_link"


class DualArmModel:
    """Pinocchio-based FK, Jacobian, and IK for dual UR5e arms.

    Each arm is an independent Pinocchio model. The URDF already contains
    the Menagerie 180-degree Z base rotation. World-frame placement is
    handled by adding the base translation to Pinocchio FK outputs.

    Args:
        urdf_path: Path to UR5e URDF. Defaults to lab6_common.URDF_PATH.
    """

    def __init__(self, urdf_path: Path | None = None) -> None:
        path = str(urdf_path or URDF_PATH)

        # Left arm
        self.model_left = pin.buildModelFromUrdf(path)
        self.data_left = self.model_left.createData()
        self.ee_frame_id_left = self.model_left.getFrameId(EE_FRAME_NAME)

        # Right arm (same URDF, different base offset)
        self.model_right = pin.buildModelFromUrdf(path)
        self.data_right = self.model_right.createData()
        self.ee_frame_id_right = self.model_right.getFrameId(EE_FRAME_NAME)

        self.base_left = LEFT_BASE_POS.copy()
        self.base_right = RIGHT_BASE_POS.copy()

    # ------------------------------------------------------------------
    # Forward Kinematics
    # ------------------------------------------------------------------

    def fk_left(self, q: np.ndarray) -> pin.SE3:
        """Compute left arm EE pose in world frame.

        Args:
            q: Joint angles (6,).

        Returns:
            SE3 pose of left EE in world frame.
        """
        pin.forwardKinematics(self.model_left, self.data_left, q)
        pin.updateFramePlacements(self.model_left, self.data_left)
        oMf = self.data_left.oMf[self.ee_frame_id_left]
        return pin.SE3(oMf.rotation.copy(), oMf.translation + self.base_left)

    def fk_right(self, q: np.ndarray) -> pin.SE3:
        """Compute right arm EE pose in world frame.

        Args:
            q: Joint angles (6,).

        Returns:
            SE3 pose of right EE in world frame.
        """
        pin.forwardKinematics(self.model_right, self.data_right, q)
        pin.updateFramePlacements(self.model_right, self.data_right)
        oMf = self.data_right.oMf[self.ee_frame_id_right]
        return pin.SE3(oMf.rotation.copy(), oMf.translation + self.base_right)

    def fk(self, arm: str, q: np.ndarray) -> pin.SE3:
        """Compute EE pose for the specified arm.

        Args:
            arm: "left" or "right".
            q: Joint angles (6,).

        Returns:
            SE3 pose in world frame.
        """
        if arm == "left":
            return self.fk_left(q)
        elif arm == "right":
            return self.fk_right(q)
        raise ValueError(f"Unknown arm: {arm!r}")

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------

    def jacobian_left(self, q: np.ndarray) -> np.ndarray:
        """Compute 6x6 geometric Jacobian for left arm in world frame.

        Args:
            q: Joint angles (6,).

        Returns:
            Jacobian (6, 6) — top 3 rows = linear, bottom 3 = angular.
        """
        pin.computeJointJacobians(self.model_left, self.data_left, q)
        pin.updateFramePlacements(self.model_left, self.data_left)
        return pin.getFrameJacobian(
            self.model_left, self.data_left, self.ee_frame_id_left,
            pin.ReferenceFrame.WORLD,
        ).copy()

    def jacobian_right(self, q: np.ndarray) -> np.ndarray:
        """Compute 6x6 geometric Jacobian for right arm in world frame.

        Args:
            q: Joint angles (6,).

        Returns:
            Jacobian (6, 6) — top 3 rows = linear, bottom 3 = angular.
        """
        pin.computeJointJacobians(self.model_right, self.data_right, q)
        pin.updateFramePlacements(self.model_right, self.data_right)
        return pin.getFrameJacobian(
            self.model_right, self.data_right, self.ee_frame_id_right,
            pin.ReferenceFrame.WORLD,
        ).copy()

    def jacobian(self, arm: str, q: np.ndarray) -> np.ndarray:
        """Compute Jacobian for the specified arm.

        Args:
            arm: "left" or "right".
            q: Joint angles (6,).

        Returns:
            Jacobian (6, 6).
        """
        if arm == "left":
            return self.jacobian_left(q)
        elif arm == "right":
            return self.jacobian_right(q)
        raise ValueError(f"Unknown arm: {arm!r}")

    # ------------------------------------------------------------------
    # Damped Least-Squares IK
    # ------------------------------------------------------------------

    def _ik_single(
        self,
        model: pin.Model,
        data: pin.Data,
        ee_fid: int,
        target_pos_local: np.ndarray,
        target_rot: np.ndarray | None,
        q_init: np.ndarray,
        damping: float,
        max_iter: int,
        tol_pos: float,
        tol_rot: float,
        dq_max: float,
    ) -> tuple[np.ndarray, bool, dict]:
        """Single-start DLS IK solve (internal)."""
        q = q_init.copy()
        pos_only = target_rot is None

        for i in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            oMf = data.oMf[ee_fid]

            e_pos = target_pos_local - oMf.translation

            if pos_only:
                err = e_pos
                if np.linalg.norm(err) < tol_pos:
                    return q, True, {"iter": i + 1, "pos_err": np.linalg.norm(err)}
                pin.computeJointJacobians(model, data, q)
                pin.updateFramePlacements(model, data)
                J = pin.getFrameJacobian(
                    model, data, ee_fid, pin.ReferenceFrame.WORLD
                )[:3, :]
            else:
                R_err = target_rot @ oMf.rotation.T
                e_rot = pin.log3(R_err)
                err = np.concatenate([e_pos, e_rot])
                if np.linalg.norm(e_pos) < tol_pos and np.linalg.norm(e_rot) < tol_rot:
                    return q, True, {
                        "iter": i + 1,
                        "pos_err": np.linalg.norm(e_pos),
                        "rot_err": np.linalg.norm(e_rot),
                    }
                pin.computeJointJacobians(model, data, q)
                pin.updateFramePlacements(model, data)
                J = pin.getFrameJacobian(
                    model, data, ee_fid, pin.ReferenceFrame.WORLD
                )

            # DLS: dq = J^T (J J^T + lambda^2 I)^{-1} e
            JJT = J @ J.T
            n = JJT.shape[0]
            dq = J.T @ np.linalg.solve(JJT + damping**2 * np.eye(n), err)

            # Clamp step size
            dq_norm = np.linalg.norm(dq)
            if dq_norm > dq_max:
                dq = dq * (dq_max / dq_norm)

            q = q + dq

        # Did not converge
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[ee_fid]
        final_pos_err = np.linalg.norm(target_pos_local - oMf.translation)
        info: dict = {"iter": max_iter, "pos_err": final_pos_err}
        if not pos_only:
            R_err = target_rot @ oMf.rotation.T
            info["rot_err"] = np.linalg.norm(pin.log3(R_err))
        return q, False, info

    def ik(
        self,
        arm: str,
        target_pos: np.ndarray,
        target_rot: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        *,
        damping: float = 0.01,
        max_iter: int = 200,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
        dq_max: float = 0.5,
        n_restarts: int = 8,
    ) -> tuple[np.ndarray, bool, dict]:
        """Solve IK using Damped Least-Squares with multi-start.

        Tries q_init (or Q_HOME) first. If that fails, retries with
        random perturbations around home up to n_restarts times.

        Args:
            arm: "left" or "right".
            target_pos: Desired EE position in world frame (3,).
            target_rot: Desired EE rotation matrix (3,3). If None, position-only IK.
            q_init: Initial joint guess (6,). Defaults to Q_HOME.
            damping: DLS damping factor lambda.
            max_iter: Maximum iterations per attempt.
            tol_pos: Position convergence tolerance (m).
            tol_rot: Orientation convergence tolerance (rad).
            dq_max: Maximum joint step per iteration (rad).
            n_restarts: Number of random restarts if first attempt fails.

        Returns:
            Tuple of (q_solution, converged, info_dict).
        """
        if arm == "left":
            model, data, ee_fid = self.model_left, self.data_left, self.ee_frame_id_left
            base = self.base_left
            q0 = Q_HOME_LEFT.copy()
        elif arm == "right":
            model, data, ee_fid = self.model_right, self.data_right, self.ee_frame_id_right
            base = self.base_right
            q0 = Q_HOME_RIGHT.copy()
        else:
            raise ValueError(f"Unknown arm: {arm!r}")

        target_pos_local = target_pos - base
        q_start = q_init.copy() if q_init is not None else q0

        # First attempt with provided/default initial guess
        q_sol, converged, info = self._ik_single(
            model, data, ee_fid, target_pos_local, target_rot,
            q_start, damping, max_iter, tol_pos, tol_rot, dq_max,
        )
        if converged:
            return q_sol, True, info

        # Multi-start: try random perturbations around home
        best_q, best_info = q_sol, info
        best_err = info["pos_err"]
        rng = np.random.default_rng()

        for _ in range(n_restarts):
            q_rand = q0 + rng.uniform(-1.0, 1.0, size=NUM_JOINTS_PER_ARM)
            q_sol, converged, info = self._ik_single(
                model, data, ee_fid, target_pos_local, target_rot,
                q_rand, damping, max_iter, tol_pos, tol_rot, dq_max,
            )
            if converged:
                return q_sol, True, info
            if info["pos_err"] < best_err:
                best_q, best_info, best_err = q_sol, info, info["pos_err"]

        return best_q, False, best_info
