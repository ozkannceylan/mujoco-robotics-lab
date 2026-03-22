"""Lab 6 — Coordinated trajectory planning for dual UR5e.

Provides three coordination modes:
1. Synchronized linear: both arms move simultaneously to targets
2. Master-slave: one arm leads, other tracks relative to object
3. Symmetric: both arms derive targets from object-centric frame

All modes produce SynchronizedTrajectory with matched timestamps.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lab6_common import NUM_JOINTS_PER_ARM, DT, VEL_LIMITS
from dual_arm_model import DualArmModel, ObjectFrame

# Sparse waypoint count used for task-space IK passes.
# Dense DT-rate trajectory is produced by interpolating between these waypoints.
_N_WAYPOINTS = 20

# Safety margin applied when auto-computing duration from velocity limits.
# Keeps peak velocity below VEL_LIMITS by this fraction.
_VEL_SAFETY = 0.7


# ---------------------------------------------------------------------------
# SynchronizedTrajectory
# ---------------------------------------------------------------------------


@dataclass
class SynchronizedTrajectory:
    """Time-synchronized joint trajectory for both arms.

    Attributes:
        timestamps: Time array (T,) in seconds.
        q_left: Left arm positions (T, 6).
        qd_left: Left arm velocities (T, 6).
        q_right: Right arm positions (T, 6).
        qd_right: Right arm velocities (T, 6).
    """

    timestamps: np.ndarray
    q_left: np.ndarray
    qd_left: np.ndarray
    q_right: np.ndarray
    qd_right: np.ndarray

    @property
    def duration(self) -> float:
        """Total trajectory duration in seconds."""
        return float(self.timestamps[-1])

    @property
    def n_steps(self) -> int:
        """Number of time steps in the trajectory."""
        return len(self.timestamps)

    def __post_init__(self) -> None:
        """Validate array shapes after construction."""
        T = len(self.timestamps)
        for name, arr in (
            ("q_left", self.q_left),
            ("qd_left", self.qd_left),
            ("q_right", self.q_right),
            ("qd_right", self.qd_right),
        ):
            if arr.shape != (T, NUM_JOINTS_PER_ARM):
                raise ValueError(
                    f"Array '{name}' has shape {arr.shape}, expected ({T}, {NUM_JOINTS_PER_ARM})"
                )


# ---------------------------------------------------------------------------
# CoordinatedPlanner
# ---------------------------------------------------------------------------


class CoordinatedPlanner:
    """Generates synchronized trajectories for dual arms.

    All trajectories are produced at the simulation timestep (DT).
    IK is solved at a sparse waypoint level, then interpolated to DT rate.

    Args:
        dual_model: Initialized DualArmModel providing FK/IK for both arms.
    """

    def __init__(self, dual_model: DualArmModel) -> None:
        self.model = dual_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_joint_linear(
        self,
        q_left_start: np.ndarray,
        q_left_end: np.ndarray,
        q_right_start: np.ndarray,
        q_right_end: np.ndarray,
        duration: float | None = None,
    ) -> SynchronizedTrajectory:
        """Joint-space linear interpolation for both arms simultaneously.

        Both arms start and end at the same time. Velocities are computed via
        central finite differences with forward/backward differences at endpoints.

        Args:
            q_left_start: Left arm start joint angles (6,).
            q_left_end: Left arm end joint angles (6,).
            q_right_start: Right arm start joint angles (6,).
            q_right_end: Right arm end joint angles (6,).
            duration: Motion duration in seconds. Auto-computed if None.

        Returns:
            SynchronizedTrajectory at DT rate.
        """
        q_left_start = np.asarray(q_left_start, dtype=float)
        q_left_end = np.asarray(q_left_end, dtype=float)
        q_right_start = np.asarray(q_right_start, dtype=float)
        q_right_end = np.asarray(q_right_end, dtype=float)

        if duration is None:
            duration = self._auto_duration(
                q_left_end - q_left_start,
                q_right_end - q_right_start,
            )

        n_steps = max(2, int(round(duration / DT)) + 1)
        timestamps = np.linspace(0.0, duration, n_steps)

        q_left, qd_left = self._joint_interpolate(q_left_start, q_left_end, n_steps)
        q_right, qd_right = self._joint_interpolate(q_right_start, q_right_end, n_steps)

        return SynchronizedTrajectory(
            timestamps=timestamps,
            q_left=q_left,
            qd_left=qd_left,
            q_right=q_right,
            qd_right=qd_right,
        )

    def plan_synchronized_linear(
        self,
        target_left: pin.SE3,
        target_right: pin.SE3,
        q_left_init: np.ndarray,
        q_right_init: np.ndarray,
        duration: float | None = None,
    ) -> SynchronizedTrajectory:
        """Task-space synchronized linear interpolation for both arms.

        Interpolates SE3 poses (position linear, rotation SLERP) at sparse
        waypoints, solves IK at each waypoint, then interpolates joint angles
        to DT rate. Both arms share the same timeline.

        Args:
            target_left: Desired left EE pose in world frame (SE3).
            target_right: Desired right EE pose in world frame (SE3).
            q_left_init: Left arm initial joint angles (6,).
            q_right_init: Right arm initial joint angles (6,).
            duration: Motion duration in seconds. Auto-computed if None.

        Returns:
            SynchronizedTrajectory at DT rate.

        Raises:
            RuntimeError: If IK fails for more than half the waypoints.
        """
        q_left_init = np.asarray(q_left_init, dtype=float)
        q_right_init = np.asarray(q_right_init, dtype=float)

        # Get start poses from FK
        start_left = self.model.fk_left(q_left_init)
        start_right = self.model.fk_right(q_right_init)

        # Solve IK at sparse waypoints
        alphas = np.linspace(0.0, 1.0, _N_WAYPOINTS)
        q_left_wp, q_right_wp = self._ik_waypoints(
            start_left, target_left, start_right, target_right,
            q_left_init, q_right_init, alphas,
        )

        # Auto-compute duration from joint displacements over the sparse path
        if duration is None:
            left_disp = np.sum(np.abs(np.diff(q_left_wp, axis=0)), axis=0)
            right_disp = np.sum(np.abs(np.diff(q_right_wp, axis=0)), axis=0)
            duration = self._auto_duration(left_disp, right_disp)

        return self._waypoints_to_trajectory(q_left_wp, q_right_wp, alphas, duration)

    def plan_master_slave(
        self,
        master_waypoints_se3: list[pin.SE3],
        object_frame: ObjectFrame,
        q_left_init: np.ndarray,
        q_right_init: np.ndarray,
        master: str = "left",
        duration: float | None = None,
    ) -> SynchronizedTrajectory:
        """Master-slave coordination: master follows waypoints, slave tracks object frame.

        At each master waypoint the object pose is reconstructed from the master
        EE pose. The slave target is then derived from the updated object pose and
        its stored grasp offset.

        Derivation:
            object_pose_new = master_waypoint * offset_master.inverse()
            slave_target    = object_pose_new * offset_slave

        Args:
            master_waypoints_se3: Ordered list of master EE SE3 targets in world frame.
                The first element is treated as the initial master pose (t=0).
                If the first element differs from FK(q_init), the trajectory starts
                at the current FK pose and moves through the given waypoints.
            object_frame: ObjectFrame with initial object pose and both grasp offsets.
            q_left_init: Left arm initial joint angles (6,).
            q_right_init: Right arm initial joint angles (6,).
            master: Which arm is the master — "left" or "right".
            duration: Total duration in seconds. Auto-computed if None.

        Returns:
            SynchronizedTrajectory at DT rate.

        Raises:
            ValueError: If ``master`` is not "left" or "right".
            RuntimeError: If IK fails for more than half the waypoints.
        """
        if master not in ("left", "right"):
            raise ValueError(f"master must be 'left' or 'right', got '{master}'")

        q_left_init = np.asarray(q_left_init, dtype=float)
        q_right_init = np.asarray(q_right_init, dtype=float)

        if master == "left":
            offset_master = object_frame.grasp_offset_left
            offset_slave = object_frame.grasp_offset_right
            ik_master = self.model.ik_left
            ik_slave = self.model.ik_right
            q_master_init = q_left_init.copy()
            q_slave_init = q_right_init.copy()
        else:
            offset_master = object_frame.grasp_offset_right
            offset_slave = object_frame.grasp_offset_left
            ik_master = self.model.ik_right
            ik_slave = self.model.ik_left
            q_master_init = q_right_init.copy()
            q_slave_init = q_left_init.copy()

        # Prepend current FK pose as waypoint[0] so motion is smooth from current config
        fk_master = (
            self.model.fk_left(q_master_init)
            if master == "left"
            else self.model.fk_right(q_master_init)
        )
        all_waypoints = [fk_master] + list(master_waypoints_se3)
        n_wp = len(all_waypoints)

        q_master_wps = np.zeros((n_wp, NUM_JOINTS_PER_ARM))
        q_slave_wps = np.zeros((n_wp, NUM_JOINTS_PER_ARM))
        q_master_wps[0] = q_master_init
        q_slave_wps[0] = q_slave_init

        n_failures_master = 0
        n_failures_slave = 0

        for i, master_wp in enumerate(all_waypoints):
            if i == 0:
                continue  # already set above

            # Solve master IK
            q_m_prev = q_master_wps[i - 1]
            q_m = ik_master(master_wp, q_m_prev)
            if q_m is None:
                warnings.warn(
                    f"Master IK failed at waypoint {i}/{n_wp - 1}; "
                    "holding previous configuration.",
                    stacklevel=2,
                )
                q_m = q_master_wps[i - 1].copy()
                n_failures_master += 1
            q_master_wps[i] = q_m

            # Reconstruct object pose from master waypoint
            # object_pose = master_wp * offset_master^{-1}
            obj_pose_new = master_wp * offset_master.inverse()

            # Derive slave target
            slave_target = obj_pose_new * offset_slave

            # Solve slave IK
            q_s_prev = q_slave_wps[i - 1]
            q_s = ik_slave(slave_target, q_s_prev)
            if q_s is None:
                warnings.warn(
                    f"Slave IK failed at waypoint {i}/{n_wp - 1}; "
                    "holding previous configuration.",
                    stacklevel=2,
                )
                q_s = q_slave_wps[i - 1].copy()
                n_failures_slave += 1
            q_slave_wps[i] = q_s

        _check_ik_failure_rate(n_failures_master, n_wp, "master")
        _check_ik_failure_rate(n_failures_slave, n_wp, "slave")

        # Auto-duration from joint path lengths
        if duration is None:
            left_disp = np.sum(np.abs(np.diff(q_master_wps, axis=0)), axis=0)
            right_disp = np.sum(np.abs(np.diff(q_slave_wps, axis=0)), axis=0)
            duration = self._auto_duration(left_disp, right_disp)

        alphas = np.linspace(0.0, 1.0, n_wp)

        if master == "left":
            q_left_wp = q_master_wps
            q_right_wp = q_slave_wps
        else:
            q_left_wp = q_slave_wps
            q_right_wp = q_master_wps

        return self._waypoints_to_trajectory(q_left_wp, q_right_wp, alphas, duration)

    def plan_symmetric(
        self,
        object_waypoints_se3: list[pin.SE3],
        object_frame: ObjectFrame,
        q_left_init: np.ndarray,
        q_right_init: np.ndarray,
        duration: float | None = None,
    ) -> SynchronizedTrajectory:
        """Symmetric coordination: both arms derive targets from an object trajectory.

        At each object waypoint:
            left_target  = obj_pose * grasp_offset_left
            right_target = obj_pose * grasp_offset_right

        The object path is interpolated (position linear, rotation SLERP) at
        sparse waypoints, IK is solved for both arms at each, then interpolated
        to DT rate.

        Args:
            object_waypoints_se3: Ordered list of object SE3 poses in world frame.
                The initial pose is taken from ``object_frame.pose``.
            object_frame: ObjectFrame with initial object pose and both grasp offsets.
            q_left_init: Left arm initial joint angles (6,).
            q_right_init: Right arm initial joint angles (6,).
            duration: Total duration in seconds. Auto-computed if None.

        Returns:
            SynchronizedTrajectory at DT rate.

        Raises:
            RuntimeError: If IK fails for more than half the waypoints.
        """
        q_left_init = np.asarray(q_left_init, dtype=float)
        q_right_init = np.asarray(q_right_init, dtype=float)

        # Build object SE3 path: start from object_frame.pose
        obj_path = [object_frame.pose] + list(object_waypoints_se3)
        n_segs = len(obj_path) - 1  # number of segments between consecutive poses

        if n_segs < 1:
            # Only one pose — return zero-duration hold at initial config
            n_steps = max(2, int(round((duration or DT) / DT)) + 1)
            ts = np.linspace(0.0, duration or DT, n_steps)
            q_left_tile = np.tile(q_left_init, (n_steps, 1))
            q_right_tile = np.tile(q_right_init, (n_steps, 1))
            qd_zero = np.zeros_like(q_left_tile)
            return SynchronizedTrajectory(
                timestamps=ts,
                q_left=q_left_tile,
                qd_left=qd_zero,
                q_right=q_right_tile,
                qd_right=qd_zero,
            )

        # Distribute _N_WAYPOINTS evenly across all segments
        wp_per_seg = max(2, _N_WAYPOINTS // n_segs)

        all_q_left: list[np.ndarray] = []
        all_q_right: list[np.ndarray] = []

        q_left_cur = q_left_init.copy()
        q_right_cur = q_right_init.copy()
        n_fail_left = 0
        n_fail_right = 0
        total_wps = 0

        for seg_idx in range(n_segs):
            obj_start = obj_path[seg_idx]
            obj_end = obj_path[seg_idx + 1]

            alphas = np.linspace(0.0, 1.0, wp_per_seg)
            # Skip alpha=0 on subsequent segments to avoid duplicates
            start_i = 0 if seg_idx == 0 else 1

            for alpha in alphas[start_i:]:
                obj_pose = self._interpolate_se3(obj_start, obj_end, alpha)
                left_target = obj_pose * object_frame.grasp_offset_left
                right_target = obj_pose * object_frame.grasp_offset_right

                q_l = self.model.ik_left(left_target, q_left_cur)
                if q_l is None:
                    warnings.warn(
                        f"Left IK failed at segment {seg_idx}, alpha={alpha:.3f}; "
                        "holding previous configuration.",
                        stacklevel=2,
                    )
                    q_l = q_left_cur.copy()
                    n_fail_left += 1
                else:
                    q_left_cur = q_l.copy()

                q_r = self.model.ik_right(right_target, q_right_cur)
                if q_r is None:
                    warnings.warn(
                        f"Right IK failed at segment {seg_idx}, alpha={alpha:.3f}; "
                        "holding previous configuration.",
                        stacklevel=2,
                    )
                    q_r = q_right_cur.copy()
                    n_fail_right += 1
                else:
                    q_right_cur = q_r.copy()

                all_q_left.append(q_l)
                all_q_right.append(q_r)
                total_wps += 1

        q_left_wp = np.array(all_q_left)
        q_right_wp = np.array(all_q_right)

        _check_ik_failure_rate(n_fail_left, total_wps, "left")
        _check_ik_failure_rate(n_fail_right, total_wps, "right")

        if duration is None:
            left_disp = np.sum(np.abs(np.diff(q_left_wp, axis=0)), axis=0)
            right_disp = np.sum(np.abs(np.diff(q_right_wp, axis=0)), axis=0)
            duration = self._auto_duration(left_disp, right_disp)

        alphas_full = np.linspace(0.0, 1.0, total_wps)
        return self._waypoints_to_trajectory(q_left_wp, q_right_wp, alphas_full, duration)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_se3(start: pin.SE3, end: pin.SE3, alpha: float) -> pin.SE3:
        """Interpolate between two SE3 poses.

        Translation is linearly interpolated; rotation uses SLERP.

        Args:
            start: Starting SE3 pose.
            end: Ending SE3 pose.
            alpha: Interpolation parameter in [0, 1].

        Returns:
            Interpolated SE3 pose.
        """
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # Linear interpolation of position
        pos = (1.0 - alpha) * start.translation + alpha * end.translation

        # SLERP for rotation
        q1 = pin.Quaternion(start.rotation)
        q2 = pin.Quaternion(end.rotation)
        # Ensure shortest-path SLERP by checking dot product sign
        if q1.dot(q2) < 0.0:
            q2 = pin.Quaternion(-q2.coeffs())
        q_interp = q1.slerp(alpha, q2)
        R_interp = q_interp.toRotationMatrix()

        return pin.SE3(R_interp, pos)

    @staticmethod
    def _joint_interpolate(
        q_start: np.ndarray,
        q_end: np.ndarray,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linear joint-space interpolation from q_start to q_end.

        Velocities are computed via central finite differences with forward/
        backward differences at the endpoints.

        Args:
            q_start: Start joint angles (6,).
            q_end: End joint angles (6,).
            n_steps: Number of output samples (including both endpoints).

        Returns:
            Tuple of (positions (n_steps, 6), velocities (n_steps, 6)).
        """
        n_steps = max(2, n_steps)
        alphas = np.linspace(0.0, 1.0, n_steps)
        # Positions: simple linear blend
        positions = q_start[None, :] + alphas[:, None] * (q_end - q_start)[None, :]

        # Velocities via finite differences
        # dt_norm is the step in alpha space, true velocity = dq/dt = dq/dalpha * dalpha/dt
        # We compute dq/dalpha here; caller scales by total duration
        d_alpha = alphas[1] - alphas[0] if n_steps > 1 else 1.0
        velocities = np.gradient(positions, d_alpha, axis=0)

        return positions, velocities

    @staticmethod
    def _auto_duration(
        q_left_disp: np.ndarray,
        q_right_disp: np.ndarray,
    ) -> float:
        """Compute minimum safe duration based on max joint displacement.

        Uses peak angular displacement divided by scaled velocity limits to
        ensure no joint exceeds ``_VEL_SAFETY * VEL_LIMITS``.

        Args:
            q_left_disp: Per-joint displacement (absolute) for left arm (6,).
            q_right_disp: Per-joint displacement (absolute) for right arm (6,).

        Returns:
            Duration in seconds (minimum 0.1 s).
        """
        max_left = np.max(np.abs(q_left_disp) / (VEL_LIMITS * _VEL_SAFETY))
        max_right = np.max(np.abs(q_right_disp) / (VEL_LIMITS * _VEL_SAFETY))
        return float(max(max_left, max_right, 0.1))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ik_waypoints(
        self,
        start_left: pin.SE3,
        end_left: pin.SE3,
        start_right: pin.SE3,
        end_right: pin.SE3,
        q_left_init: np.ndarray,
        q_right_init: np.ndarray,
        alphas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve IK at sparse SE3 waypoints along linear/SLERP paths.

        Uses sequential warm-starting: each waypoint's IK starts from the
        previous solution, which greatly helps convergence along smooth paths.

        Args:
            start_left: Left arm start SE3 in world frame.
            end_left: Left arm end SE3 in world frame.
            start_right: Right arm start SE3 in world frame.
            end_right: Right arm end SE3 in world frame.
            q_left_init: Left arm initial joint angles (6,).
            q_right_init: Right arm initial joint angles (6,).
            alphas: Array of interpolation parameters in [0, 1].

        Returns:
            Tuple of (q_left_waypoints (N, 6), q_right_waypoints (N, 6)).
        """
        n = len(alphas)
        q_left_wp = np.zeros((n, NUM_JOINTS_PER_ARM))
        q_right_wp = np.zeros((n, NUM_JOINTS_PER_ARM))

        # Seed with initial configs at alpha=0
        q_left_wp[0] = q_left_init
        q_right_wp[0] = q_right_init

        n_fail_left = 0
        n_fail_right = 0

        for i, alpha in enumerate(alphas):
            if i == 0:
                continue

            # Interpolate SE3 targets
            left_target = CoordinatedPlanner._interpolate_se3(start_left, end_left, alpha)
            right_target = CoordinatedPlanner._interpolate_se3(start_right, end_right, alpha)

            # Left IK (warm-started from previous solution)
            q_l = self.model.ik_left(left_target, q_left_wp[i - 1])
            if q_l is None:
                warnings.warn(
                    f"Left IK failed at alpha={alpha:.3f}; "
                    "holding previous configuration.",
                    stacklevel=3,
                )
                q_l = q_left_wp[i - 1].copy()
                n_fail_left += 1
            q_left_wp[i] = q_l

            # Right IK (warm-started from previous solution)
            q_r = self.model.ik_right(right_target, q_right_wp[i - 1])
            if q_r is None:
                warnings.warn(
                    f"Right IK failed at alpha={alpha:.3f}; "
                    "holding previous configuration.",
                    stacklevel=3,
                )
                q_r = q_right_wp[i - 1].copy()
                n_fail_right += 1
            q_right_wp[i] = q_r

        _check_ik_failure_rate(n_fail_left, n, "left")
        _check_ik_failure_rate(n_fail_right, n, "right")

        return q_left_wp, q_right_wp

    @staticmethod
    def _waypoints_to_trajectory(
        q_left_wp: np.ndarray,
        q_right_wp: np.ndarray,
        alphas: np.ndarray,
        duration: float,
    ) -> SynchronizedTrajectory:
        """Interpolate sparse joint waypoints to DT-rate trajectory.

        Performs piecewise linear interpolation in joint space between the
        sparse waypoints, producing a dense trajectory sampled at DT.

        Args:
            q_left_wp: Left arm waypoints (N, 6).
            q_right_wp: Right arm waypoints (N, 6).
            alphas: Normalised time values for each waypoint, in [0, 1] (N,).
            duration: Total trajectory duration in seconds.

        Returns:
            SynchronizedTrajectory sampled at DT rate.
        """
        n_dense = max(2, int(round(duration / DT)) + 1)
        ts_dense = np.linspace(0.0, duration, n_dense)
        alphas_dense = ts_dense / duration  # normalised [0, 1]

        # Per-joint linear interpolation from sparse to dense grid
        q_left_dense = np.zeros((n_dense, NUM_JOINTS_PER_ARM))
        q_right_dense = np.zeros((n_dense, NUM_JOINTS_PER_ARM))

        for j in range(NUM_JOINTS_PER_ARM):
            q_left_dense[:, j] = np.interp(alphas_dense, alphas, q_left_wp[:, j])
            q_right_dense[:, j] = np.interp(alphas_dense, alphas, q_right_wp[:, j])

        # Velocities via central finite differences (true time-domain)
        qd_left_dense = np.gradient(q_left_dense, DT, axis=0)
        qd_right_dense = np.gradient(q_right_dense, DT, axis=0)

        return SynchronizedTrajectory(
            timestamps=ts_dense,
            q_left=q_left_dense,
            qd_left=qd_left_dense,
            q_right=q_right_dense,
            qd_right=qd_right_dense,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _check_ik_failure_rate(n_failures: int, n_total: int, arm_label: str) -> None:
    """Raise RuntimeError if more than half of IK attempts failed.

    Args:
        n_failures: Number of failed IK solves.
        n_total: Total number of IK attempts.
        arm_label: Label for the arm (used in the error message).

    Raises:
        RuntimeError: If ``n_failures > n_total / 2``.
    """
    if n_total > 0 and n_failures > n_total / 2:
        raise RuntimeError(
            f"IK failed for {n_failures}/{n_total} waypoints on '{arm_label}' arm. "
            "Consider adjusting targets, increasing max_iter, or providing a better "
            "initial configuration."
        )
