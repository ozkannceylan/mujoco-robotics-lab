"""LIPM + ZMP Preview Controller (Kajita 2003).

Robot-agnostic implementation. The only robot parameter is CoM height z_c.

LIPM continuous-time dynamics (cart-table model):
    x_ddot = (g / z_c) * (x_com - x_zmp)

Discretized with timestep dt, augmented with jerk input u:
    state = [x, x_dot, x_ddot]
    A = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]
    B = [[dt^3/6], [dt^2/2], [dt]]
    C = [1, 0, -z_c/g]   (ZMP output: p = x - (z_c/g)*x_ddot)

Preview controller solves discrete Riccati on the augmented system
(with integral of ZMP tracking error) to get:
    - Ki: integral gain on ZMP error
    - Gx: state feedback gain on [x, x_dot, x_ddot]
    - Gd[j]: preview gains for future ZMP references

Control law:
    u(k) = -Ki * sum(e) - Gx @ state(k) - sum(Gd[j] * zmp_ref(k+j))

Reference: Kajita et al., "Biped Walking Pattern Generation by Using
Preview Control of Zero-Moment Point", ICRA 2003.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PreviewControllerParams:
    """Hyperparameters for preview control gain synthesis."""

    z_c: float          # CoM height [m]
    g: float = 9.81     # gravity [m/s^2]
    Q_e: float = 1.0    # weight on integrated ZMP error
    Q_x: np.ndarray | None = None  # weight on state (3x3), default zeros
    R: float = 1e-6      # weight on jerk input
    n_preview: int = 80  # number of preview steps


@dataclass
class PreviewControllerGains:
    """Precomputed controller matrices and gains."""

    A: np.ndarray       # 3x3 state transition
    B: np.ndarray       # 3x1 input matrix
    C: np.ndarray       # 1x3 output (state -> ZMP)
    Ki: float           # integral gain
    Gx: np.ndarray      # 1x3 state feedback gain
    Gd: np.ndarray      # (n_preview-1,) preview gains


# ---------------------------------------------------------------------------
# Gain computation (offline, once)
# ---------------------------------------------------------------------------


def compute_preview_gains(
    params: PreviewControllerParams,
    dt: float,
) -> PreviewControllerGains:
    """Compute preview control gains by solving discrete Riccati equation.

    Args:
        params: Controller hyperparameters.
        dt: Discrete time step [s].

    Returns:
        PreviewControllerGains with all matrices needed for runtime.
    """
    z_c = params.z_c
    g = params.g

    # ── Discrete cart-table model with jerk input ──────────────────
    A = np.array([
        [1.0, dt, 0.5 * dt * dt],
        [0.0, 1.0, dt],
        [0.0, 0.0, 1.0],
    ])
    B = np.array([[dt**3 / 6.0], [dt**2 / 2.0], [dt]])
    C = np.array([[1.0, 0.0, -z_c / g]])

    # ── Augmented system (with integral of ZMP error) ─────────────
    # State: [e_int, x, x_dot, x_ddot]
    # A1 = [[1, C@A], [0, A]], B1 = [[C@B], [B]], I1 = [[1], [0]]
    A1 = np.block([
        [np.eye(1), C @ A],
        [np.zeros((3, 1)), A],
    ])
    B1 = np.vstack([C @ B, B])
    I1 = np.vstack([np.array([[1.0]]), np.zeros((3, 1))])
    F = np.vstack([C @ A, A])  # used for Gx extraction

    # Cost matrices
    Q_x = params.Q_x if params.Q_x is not None else np.zeros((3, 3))
    Q = np.block([
        [np.array([[params.Q_e]]), np.zeros((1, 3))],
        [np.zeros((3, 1)), Q_x],
    ])
    R_mat = np.array([[params.R]])

    # ── Solve DARE ────────────────────────────────────────────────
    K = solve_discrete_are(A1, B1, Q, R_mat)

    # ── Extract gains ─────────────────────────────────────────────
    inv_term = np.linalg.inv(R_mat + B1.T @ K @ B1)  # scalar-ish (1x1)

    Ki = float(inv_term @ B1.T @ K @ I1)
    Gx = inv_term @ B1.T @ K @ F  # (1, 3)

    # Preview gains Gd
    Ac = A1 - B1 @ inv_term @ B1.T @ K @ A1  # closed-loop A
    n_p = params.n_preview
    Gd = np.zeros(n_p - 1)
    Gd[0] = -Ki  # first preview gain = -Ki

    X = Ac.T @ K @ I1
    for j in range(n_p - 2):
        Gd[j + 1] = float(inv_term @ B1.T @ X)
        X = Ac.T @ X

    return PreviewControllerGains(
        A=A, B=B, C=C,
        Ki=Ki, Gx=Gx.ravel(),  # flatten to (3,)
        Gd=Gd,
    )


# ---------------------------------------------------------------------------
# Runtime update (called every dt)
# ---------------------------------------------------------------------------


def preview_control_step(
    gains: PreviewControllerGains,
    x_state: np.ndarray,
    current_zmp_ref: float,
    future_zmp_refs: np.ndarray,
) -> tuple[float, np.ndarray]:
    """One-step preview control update for a single axis.

    Args:
        gains: Precomputed controller gains.
        x_state: Augmented state [e_int, x, x_dot, x_ddot] (4,).
        current_zmp_ref: ZMP reference at current timestep.
        future_zmp_refs: Future ZMP references (n_preview-1,).

    Returns:
        (u, x_next): jerk command and next augmented state.
    """
    e_int = x_state[0]
    state = x_state[1:]  # [x, x_dot, x_ddot]

    # Current ZMP output
    zmp_current = float(gains.C @ state)

    # Control law: u = -Ki*e_int - Gx@state + Gd@future_zmp_refs
    u = -gains.Ki * e_int - gains.Gx @ state + gains.Gd @ future_zmp_refs

    # Update integral error
    e_int_next = e_int + (zmp_current - current_zmp_ref)

    # Propagate state
    state_next = gains.A @ state + gains.B.ravel() * u

    x_next = np.empty(4)
    x_next[0] = e_int_next
    x_next[1:] = state_next

    return float(u), x_next


# ---------------------------------------------------------------------------
# Batch trajectory generation
# ---------------------------------------------------------------------------


def generate_com_trajectory(
    zmp_ref_x: np.ndarray,
    zmp_ref_y: np.ndarray,
    z_c: float,
    dt: float,
    com_x0: float = 0.0,
    com_y0: float = 0.0,
    Q_e: float = 1.0,
    R: float = 1e-6,
    n_preview: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate CoM trajectory from ZMP reference using preview control.

    Runs the LIPM preview controller for both X and Y axes independently.

    Args:
        zmp_ref_x: ZMP reference X positions (N,).
        zmp_ref_y: ZMP reference Y positions (N,).
        z_c: CoM height [m].
        dt: Timestep [s].
        com_x0: Initial CoM X position.
        com_y0: Initial CoM Y position.
        Q_e: ZMP error weight.
        R: Jerk weight.
        n_preview: Number of preview steps.

    Returns:
        (com_x, com_y, com_dx, com_dy): CoM position and velocity arrays (N,).
    """
    N = len(zmp_ref_x)
    assert len(zmp_ref_y) == N

    params = PreviewControllerParams(
        z_c=z_c, Q_e=Q_e, R=R, n_preview=n_preview,
    )
    gains = compute_preview_gains(params, dt)

    # Pad ZMP references for preview lookahead
    pad = n_preview
    zmp_x_padded = np.concatenate([zmp_ref_x, np.full(pad, zmp_ref_x[-1])])
    zmp_y_padded = np.concatenate([zmp_ref_y, np.full(pad, zmp_ref_y[-1])])

    # Initialize augmented states: [e_int, x, x_dot, x_ddot]
    x_state = np.array([0.0, com_x0, 0.0, 0.0])
    y_state = np.array([0.0, com_y0, 0.0, 0.0])

    # Output arrays
    com_x = np.zeros(N)
    com_y = np.zeros(N)
    com_dx = np.zeros(N)
    com_dy = np.zeros(N)

    for k in range(N):
        # Future ZMP references for preview (n_preview - 1 samples)
        future_x = zmp_x_padded[k + 1: k + n_preview]
        future_y = zmp_y_padded[k + 1: k + n_preview]

        _, x_state = preview_control_step(
            gains, x_state, zmp_ref_x[k], future_x,
        )
        _, y_state = preview_control_step(
            gains, y_state, zmp_ref_y[k], future_y,
        )

        com_x[k] = x_state[1]
        com_y[k] = y_state[1]
        com_dx[k] = x_state[2]
        com_dy[k] = y_state[2]

    return com_x, com_y, com_dx, com_dy
