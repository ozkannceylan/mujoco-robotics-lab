"""Linear Inverted Pendulum Model (LIPM) planner with preview control.

Implements the LIPM dynamics and Kajita 2003 preview controller
for generating Center of Mass (CoM) trajectories from ZMP references.

LIPM dynamics:
    x_ddot = (g / z_c) * (x - p)
where x is CoM position, p is ZMP position, g is gravity, z_c is CoM height.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from lab7_common import GRAVITY_MAGNITUDE, DT


class LIPMPlanner:
    """LIPM-based CoM trajectory planner with preview control.

    Uses discrete-time LQR preview control (Kajita 2003) to generate
    smooth CoM trajectories that track a desired ZMP reference.
    """

    def __init__(
        self,
        z_c: float = 0.55,
        dt: float = DT,
        preview_steps: int = 100,
        Q_e: float = 1.0,
        Q_x: float = 0.0,
        R: float = 1e-6,
    ) -> None:
        """Initialize LIPM planner.

        Args:
            z_c: Constant CoM height [m].
            dt: Timestep [s].
            preview_steps: Number of preview steps for controller.
            Q_e: Weight on ZMP tracking error in cost function.
            Q_x: Weight on CoM state in cost function.
            R: Weight on jerk input in cost function.
        """
        self.z_c = z_c
        self.dt = dt
        self.g = GRAVITY_MAGNITUDE
        self.omega = np.sqrt(self.g / self.z_c)
        self.preview_steps = preview_steps

        # Discrete LIPM state-space: state = [x, x_dot, x_ddot]
        # Input u = x_dddot (jerk)
        # ZMP output: p = x - (z_c/g)*x_ddot
        self.A = np.array([
            [1.0, dt, 0.5 * dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ])
        self.B = np.array([
            [dt**3 / 6.0],
            [0.5 * dt**2],
            [dt],
        ])
        self.C = np.array([[1.0, 0.0, -self.z_c / self.g]])

        # Augmented system for error integral
        # Augmented state: [e_integral, x, x_dot, x_ddot]
        # where e = p_ref - p (ZMP tracking error)
        self._compute_preview_gains(Q_e, Q_x, R)

    def _compute_preview_gains(
        self, Q_e: float, Q_x: float, R: float
    ) -> None:
        """Compute LQR preview control gains via discrete Riccati equation.

        Args:
            Q_e: Weight on integrated ZMP error.
            Q_x: Weight on state deviation.
            R: Weight on jerk input.
        """
        A = self.A
        B = self.B
        C = self.C

        # Augmented system matrices
        n = A.shape[0]
        A_aug = np.zeros((n + 1, n + 1))
        A_aug[0, 0] = 1.0
        A_aug[0, 1:] = C @ A
        A_aug[1:, 1:] = A

        B_aug = np.zeros((n + 1, 1))
        B_aug[0, :] = (C @ B).flatten()
        B_aug[1:, :] = B

        # Cost matrices
        Q_aug = np.zeros((n + 1, n + 1))
        Q_aug[0, 0] = Q_e
        for i in range(n):
            Q_aug[1 + i, 1 + i] = Q_x

        R_mat = np.array([[R]])

        # Solve discrete algebraic Riccati equation via iteration
        P = Q_aug.copy()
        for _ in range(2000):
            K_term = np.linalg.solve(
                R_mat + B_aug.T @ P @ B_aug, B_aug.T @ P @ A_aug
            )
            P_new = Q_aug + A_aug.T @ P @ A_aug - A_aug.T @ P @ B_aug @ K_term
            if np.max(np.abs(P_new - P)) < 1e-10:
                P = P_new
                break
            P = P_new

        # State feedback gain
        K_full = np.linalg.solve(
            R_mat + B_aug.T @ P @ B_aug, B_aug.T @ P @ A_aug
        )
        self.K_i = K_full[0, 0]        # integral gain
        self.K_x = K_full[0, 1:]       # state gain [3]

        # Preview gains
        F = np.zeros(self.preview_steps)
        Ac = A_aug - B_aug @ K_full
        X = Ac.T @ P
        I_col = np.array([1.0, 0.0, 0.0, 0.0])
        for j in range(self.preview_steps):
            F[j] = np.linalg.solve(
                R_mat + B_aug.T @ P @ B_aug, B_aug.T @ X @ I_col
            ).item()
            X = Ac.T @ X

        self.F = F

    def plan_com_trajectory(
        self, zmp_ref_x: np.ndarray, zmp_ref_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate CoM trajectory from ZMP reference using preview control.

        Args:
            zmp_ref_x: ZMP x-reference trajectory [N].
            zmp_ref_y: ZMP y-reference trajectory [N].

        Returns:
            Tuple of (com_x [N], com_y [N], zmp_actual_x [N], zmp_actual_y [N]).
        """
        N = len(zmp_ref_x)

        # Pad ZMP reference for preview window
        pad = self.preview_steps
        zmp_x_pad = np.concatenate([zmp_ref_x, np.full(pad, zmp_ref_x[-1])])
        zmp_y_pad = np.concatenate([zmp_ref_y, np.full(pad, zmp_ref_y[-1])])

        # State: [pos, vel, acc] for x and y
        state_x = np.zeros(3)
        state_y = np.zeros(3)

        # Integral of ZMP error
        err_x = 0.0
        err_y = 0.0

        # Output arrays
        com_x = np.zeros(N)
        com_y = np.zeros(N)
        zmp_x = np.zeros(N)
        zmp_y = np.zeros(N)

        for k in range(N):
            # Current ZMP
            p_x = (self.C @ state_x).item()
            p_y = (self.C @ state_y).item()

            # ZMP error integral
            err_x += zmp_x_pad[k] - p_x
            err_y += zmp_y_pad[k] - p_y

            # Preview sum
            preview_x = 0.0
            preview_y = 0.0
            for j in range(min(pad, N - k)):
                preview_x += self.F[j] * zmp_x_pad[k + 1 + j]
                preview_y += self.F[j] * zmp_y_pad[k + 1 + j]

            # Control input (jerk)
            u_x = (
                self.K_i * err_x
                + self.K_x @ state_x
                + preview_x
            )
            u_y = (
                self.K_i * err_y
                + self.K_x @ state_y
                + preview_y
            )

            # Store
            com_x[k] = state_x[0]
            com_y[k] = state_y[0]
            zmp_x[k] = p_x
            zmp_y[k] = p_y

            # Propagate
            state_x = self.A @ state_x + (self.B * u_x).flatten()
            state_y = self.A @ state_y + (self.B * u_y).flatten()

        return com_x, com_y, zmp_x, zmp_y

    def compute_capture_point(
        self,
        com_pos: np.ndarray,
        com_vel: np.ndarray,
    ) -> np.ndarray:
        """Compute the instantaneous capture point (ICP).

        The capture point is where the robot would need to place its
        foot to stop: xi = x + x_dot / omega

        Args:
            com_pos: CoM position [2] or [3] (only x,y used).
            com_vel: CoM velocity [2] or [3] (only x,y used).

        Returns:
            Capture point [2].
        """
        return com_pos[:2] + com_vel[:2] / self.omega

    def compute_zmp_from_state(
        self, com_pos: float, com_acc: float
    ) -> float:
        """Compute ZMP from CoM position and acceleration (1D).

        p = x - (z_c / g) * x_ddot

        Args:
            com_pos: CoM position (scalar).
            com_acc: CoM acceleration (scalar).

        Returns:
            ZMP position (scalar).
        """
        return com_pos - (self.z_c / self.g) * com_acc
