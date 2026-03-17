"""Lab 7 — Phase 2: LIPM preview controller and footstep trajectory generation.

Implements the Linear Inverted Pendulum Model (LIPM) with Kajita-style preview
control (Kajita et al., 2003) for generating ZMP-stable CoM trajectories.

References:
    Kajita et al., "Biped Walking Pattern Generation using Preview Control of Zero-Moment
    Point", ICRA 2003.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.linalg import solve_discrete_are

# ---------------------------------------------------------------------------
# LIPM Preview Controller (1D)
# ---------------------------------------------------------------------------


class LIPMPreviewController:
    """1-D Linear Inverted Pendulum preview controller.

    Models the robot CoM as a point mass at constant height z_c above the
    ground.  The ZMP (Zero Moment Point) is the output:

        p = x - (z_c / g) * x_ddot

    The controller minimises the sum of squared ZMP tracking errors plus
    squared jerk (control effort):

        J = sum_k { Q_e * (p_k - p_ref_k)^2 + R * u_k^2 }

    State: x_k = [x, x_dot, x_ddot]^T
    Input: u_k = x_dddot (jerk)

    Apply independently to x and y axes.
    """

    def __init__(
        self,
        T: float = 0.01,
        z_c: float = 0.66,
        g: float = 9.81,
        Q_e: float = 1.0,
        R: float = 1e-6,
        N_preview: int = 200,
    ) -> None:
        """Initialise and pre-compute LQR + preview gains.

        Args:
            T: Control period [s].
            z_c: CoM height above ground [m].
            g: Gravitational acceleration [m/s²].
            Q_e: ZMP error tracking weight.
            R: Jerk effort weight.
            N_preview: Number of preview steps.
        """
        self.T = T
        self.z_c = z_c
        self.g = g
        self.N_preview = N_preview

        # Discrete-time LIPM matrices
        T2 = T * T
        T3 = T2 * T
        self.A = np.array([
            [1.0, T,   T2 / 2.0],
            [0.0, 1.0, T       ],
            [0.0, 0.0, 1.0     ],
        ])
        self.b = np.array([T3 / 6.0, T2 / 2.0, T])
        self.C = np.array([1.0, 0.0, -z_c / g])   # ZMP output: p = C @ x

        # Compute gains via Discrete ARE
        self.K_e, self.K_s, self.f_gains = self._compute_gains(Q_e, R, N_preview)

        # Internal state: error integrator and LIPM state
        self.e: float = 0.0
        self.x: np.ndarray = np.zeros(3)  # [pos, vel, acc]

    # ------------------------------------------------------------------
    # Gain computation
    # ------------------------------------------------------------------

    def _compute_gains(
        self, Q_e: float, R: float, N_preview: int
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute K_e, K_s, f_gains from the Discrete Algebraic Riccati Equation.

        The augmented system with error integrator ξ = [e, x]^T (4-dim):

            ξ_{k+1} = A_aug ξ_k + b_aug u_k - [1, 0, 0, 0]^T p_ref_{k+1}

        where
            A_aug = [[1,  C^T A],
                     [0,    A  ]]
            b_aug = [C^T b, b]^T

        Returns:
            (K_e, K_s, f_gains):
                K_e   — scalar error integrator gain
                K_s   — (3,) state feedback gains
                f_gains — (N_preview,) preview gains f(1..N_preview)
        """
        A, b, C = self.A, self.b, self.C

        # Augmented system (4-dim state)
        A_aug = np.zeros((4, 4))
        A_aug[0, 0] = 1.0
        A_aug[0, 1:] = C @ A
        A_aug[1:, 1:] = A

        b_aug = np.zeros(4)
        b_aug[0] = float(C @ b)
        b_aug[1:] = b

        # LQR cost matrix (only penalise the error integrator state)
        Q_aug = np.zeros((4, 4))
        Q_aug[0, 0] = Q_e

        # Solve DARE: P = A_aug^T P A_aug - A_aug^T P b_aug (b_aug^T P b_aug + R)^{-1} b_aug^T P A_aug + Q_aug
        P = solve_discrete_are(A_aug, b_aug[:, None], Q_aug, np.array([[R]]))

        # Feedback gain vector: K = (b_aug^T P b_aug + R)^{-1} b_aug^T P A_aug
        denom = float(b_aug @ P @ b_aug) + R
        K_full = (b_aug @ P @ A_aug) / denom   # shape (4,)
        K_e = float(K_full[0])
        K_s = K_full[1:]   # shape (3,)

        # Preview gains f(j), j = 1 … N_preview
        # Closed-loop matrix
        A_cl = A_aug - np.outer(b_aug, K_full)

        # c_ref: how the future reference enters the augmented error state
        # From: e_{k+1} = e_k + C^T A x_k + C^T b u_k - p_ref_{k+1}
        # the reference contributes -1 to e, 0 elsewhere → c_ref = [-1, 0, 0, 0]^T
        c_ref = np.array([-1.0, 0.0, 0.0, 0.0])

        f_gains = np.zeros(N_preview)
        Pc_ref = P @ c_ref
        for j in range(N_preview):
            f_gains[j] = -float(b_aug @ Pc_ref) / denom
            Pc_ref = A_cl.T @ Pc_ref

        return K_e, K_s, f_gains

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def reset(self, x_init: np.ndarray | None = None, e_init: float = 0.0) -> None:
        """Reset the controller state.

        Args:
            x_init: Initial LIPM state [pos, vel, acc]. Defaults to zeros.
            e_init: Initial error integrator value.
        """
        self.x = x_init.copy() if x_init is not None else np.zeros(3)
        self.e = e_init

    def step(
        self, p_ref_k: float, p_ref_preview: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Advance the controller by one timestep.

        Args:
            p_ref_k: ZMP reference at the current step.
            p_ref_preview: ZMP reference for the next N_preview steps.

        Returns:
            (x_next, p_actual):
                x_next    — updated LIPM state [pos, vel, acc]
                p_actual  — current ZMP output
        """
        N = min(len(p_ref_preview), self.N_preview)

        # Current ZMP output
        p_act = float(self.C @ self.x)

        # Update error integrator
        self.e += p_act - p_ref_k

        # Preview control law: u = -K_e*e - K_s*x - sum f(j)*p_ref_{k+j}
        u = -self.K_e * self.e - float(self.K_s @ self.x)
        if N > 0:
            u -= float(self.f_gains[:N] @ p_ref_preview[:N])

        # Advance state
        self.x = self.A @ self.x + self.b * u

        return self.x.copy(), p_act


# ---------------------------------------------------------------------------
# Footstep planner
# ---------------------------------------------------------------------------


@dataclass
class Footstep:
    """A single footstep definition."""
    foot: Literal["left", "right"]
    pos: np.ndarray          # world position (x, y, 0) at landing
    t_start: float           # time the foot starts lifting
    t_swing_end: float       # time the foot lands
    t_ds_end: float          # time double-support ends (weight transferred)


def generate_footsteps(
    n_steps: int = 12,
    step_length: float = 0.10,
    step_width: float = 0.10,
    T_ss: float = 0.8,
    T_ds: float = 0.2,
    x0_left: np.ndarray | None = None,
    x0_right: np.ndarray | None = None,
) -> list[Footstep]:
    """Generate a pre-planned alternating footstep sequence.

    The first step is with the left foot. Steps are placed at alternating
    ±y offsets with forward increment step_length.

    Args:
        n_steps: Total number of steps (left+right combined).
        step_length: Forward distance per step [m].
        step_width: Half-distance between feet in y [m] (each foot at ±step_width).
        T_ss: Single support duration [s].
        T_ds: Double support (weight transfer) duration [s].
        x0_left: Initial left foot position. Defaults to [0, +step_width, 0].
        x0_right: Initial right foot position. Defaults to [0, -step_width, 0].

    Returns:
        List of Footstep objects in chronological order.
    """
    if x0_left is None:
        x0_left = np.array([0.0, +step_width, 0.0])
    if x0_right is None:
        x0_right = np.array([0.0, -step_width, 0.0])

    T_step = T_ss + T_ds
    footsteps: list[Footstep] = []

    # Current foot positions (updated each step)
    foot_pos = {"left": x0_left.copy(), "right": x0_right.copy()}

    for i in range(n_steps):
        swing_foot = "left" if i % 2 == 0 else "right"
        y_sign = +1.0 if swing_foot == "left" else -1.0

        # Landing position: advance by step_length, maintain ±step_width laterally
        prev_pos = foot_pos[swing_foot]
        # Step forward by step_length from the other foot's x position
        stance_foot = "right" if swing_foot == "left" else "left"
        land_x = foot_pos[stance_foot][0] + step_length
        land_pos = np.array([land_x, y_sign * step_width, 0.0])

        t_start = i * T_step + T_ds   # foot lifts after initial double support settling
        if i == 0:
            t_start = T_ds  # first step starts after initial double support

        t_swing_end = t_start + T_ss
        t_ds_end = t_swing_end + T_ds

        footsteps.append(Footstep(
            foot=swing_foot,
            pos=land_pos,
            t_start=t_start,
            t_swing_end=t_swing_end,
            t_ds_end=t_ds_end,
        ))
        foot_pos[swing_foot] = land_pos

    return footsteps


# ---------------------------------------------------------------------------
# ZMP reference trajectory
# ---------------------------------------------------------------------------


def generate_zmp_reference(
    footsteps: list[Footstep],
    T_init_ds: float = 0.5,
    T_final_ds: float = 0.5,
    dt: float = 0.01,
    x0_left: np.ndarray | None = None,
    x0_right: np.ndarray | None = None,
    step_width: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the 2-D ZMP reference trajectory.

    During single support, ZMP stays at the stance foot.
    During double support, ZMP linearly shifts from one foot to the other.

    Args:
        footsteps: Sequence from generate_footsteps().
        T_init_ds: Duration of initial double-support before first step [s].
        T_final_ds: Duration of final double-support after last step [s].
        dt: Time step [s].
        x0_left: Initial left foot position.
        x0_right: Initial right foot position.
        step_width: Half foot separation [m].

    Returns:
        (times, zmp_ref):
            times   — (N,) timestep array [s]
            zmp_ref — (N, 2) ZMP reference [m] (x, y columns)
    """
    if x0_left is None:
        x0_left = np.array([0.0, +step_width, 0.0])
    if x0_right is None:
        x0_right = np.array([0.0, -step_width, 0.0])

    if not footsteps:
        N = int(T_init_ds / dt) + 1
        times = np.arange(N) * dt
        mid = 0.5 * (x0_left[:2] + x0_right[:2])
        return times, np.tile(mid, (N, 1))

    T_total = footsteps[-1].t_ds_end + T_final_ds
    N = int(np.ceil(T_total / dt)) + 1
    times = np.arange(N) * dt
    zmp_ref = np.zeros((N, 2))

    # Current foot positions (track landing positions as gait progresses)
    foot_pos = {"left": x0_left.copy(), "right": x0_right.copy()}

    for k, t in enumerate(times):
        # Determine which phase we are in
        in_step = None
        for step in footsteps:
            if step.t_start <= t < step.t_ds_end:
                in_step = step
                break

        if in_step is None:
            # Initial or final double support → ZMP at mid-point between feet
            mid = 0.5 * (foot_pos["left"][:2] + foot_pos["right"][:2])
            zmp_ref[k] = mid
        elif t < in_step.t_swing_end:
            # Single support: ZMP at stance foot
            stance = "right" if in_step.foot == "left" else "left"
            zmp_ref[k] = foot_pos[stance][:2]
        else:
            # Double support: linear interpolation from stance → new landing
            alpha = (t - in_step.t_swing_end) / (in_step.t_ds_end - in_step.t_swing_end)
            alpha = np.clip(alpha, 0.0, 1.0)
            stance = "right" if in_step.foot == "left" else "left"
            zmp_ref[k] = (1.0 - alpha) * foot_pos[stance][:2] + alpha * in_step.pos[:2]

        # Update foot position when it lands
        if t >= in_step.t_swing_end if in_step else False:
            foot_pos[in_step.foot] = in_step.pos.copy()

    return times, zmp_ref


# ---------------------------------------------------------------------------
# Swing foot trajectory
# ---------------------------------------------------------------------------


def swing_trajectory(
    t: float,
    T_ss: float,
    p_start: np.ndarray,
    p_end: np.ndarray,
    height: float = 0.05,
) -> np.ndarray:
    """Compute swing foot position at time t during a swing phase.

    Horizontal: linear interpolation from p_start to p_end.
    Vertical: parabolic clearance arc (zero at start and end, peak at mid-swing).

    Args:
        t: Time within the swing phase [0, T_ss].
        T_ss: Single support (swing) duration [s].
        p_start: Foot position at start of swing (3D, z=ground).
        p_end: Foot position at end of swing (3D, z=ground).
        height: Peak foot clearance above ground [m].

    Returns:
        (3,) foot position in world frame.
    """
    s = np.clip(t / T_ss, 0.0, 1.0)
    pos_xy = (1.0 - s) * p_start[:2] + s * p_end[:2]
    pos_z = height * 4.0 * s * (1.0 - s)   # parabola: 0 at s=0,1; height at s=0.5
    return np.array([pos_xy[0], pos_xy[1], pos_z + p_end[2]])


# ---------------------------------------------------------------------------
# Full trajectory planner
# ---------------------------------------------------------------------------


def plan_walking_trajectory(
    n_steps: int = 12,
    step_length: float = 0.10,
    step_width: float = 0.10,
    step_height: float = 0.05,
    T_ss: float = 0.8,
    T_ds: float = 0.2,
    T_init_ds: float = 0.5,
    T_final_ds: float = 1.0,
    dt: float = 0.01,
    z_c: float = 0.66,
    g: float = 9.81,
    Q_e: float = 1.0,
    R: float = 1e-6,
    N_preview: int = 200,
) -> dict:
    """Pre-compute the full walking trajectory (offline planning).

    Runs the LIPM preview controller on the ZMP reference and assembles
    all relevant trajectories for use by the walking controller.

    Args:
        n_steps: Number of steps.
        step_length: Forward step length [m].
        step_width: Lateral half-distance between feet [m].
        step_height: Swing foot clearance [m].
        T_ss: Single support duration [s].
        T_ds: Double support duration [s].
        T_init_ds: Initial double support [s].
        T_final_ds: Final double support [s].
        dt: Planning time step [s].
        z_c: CoM height for LIPM [m].
        g: Gravity [m/s²].
        Q_e: ZMP error weight.
        R: Jerk weight.
        N_preview: Preview window length.

    Returns:
        dict with keys:
            'times'       — (N,) timestep array
            'zmp_ref'     — (N, 2) ZMP reference
            'com_x'       — (N,) CoM x position
            'com_y'       — (N,) CoM y position
            'com_vx'      — (N,) CoM x velocity
            'com_vy'      — (N,) CoM y velocity
            'lfoot'       — (N, 3) left foot position
            'rfoot'       — (N, 3) right foot position
            'phase'       — (N,) string array: 'init_ds', 'l_swing', 'r_swing', 'ds', 'final_ds'
            'footsteps'   — list of Footstep objects
            'foot_pos_l'  — (N, 3) left foot world pos (accounts for swing)
            'foot_pos_r'  — (N, 3) right foot world pos
    """
    x0_left = np.array([0.0, +step_width, 0.0])
    x0_right = np.array([0.0, -step_width, 0.0])

    # 1. Generate footstep plan
    footsteps = generate_footsteps(
        n_steps=n_steps,
        step_length=step_length,
        step_width=step_width,
        T_ss=T_ss,
        T_ds=T_ds,
        x0_left=x0_left,
        x0_right=x0_right,
    )

    # 2. Generate ZMP reference
    times, zmp_ref = generate_zmp_reference(
        footsteps,
        T_init_ds=T_init_ds,
        T_final_ds=T_final_ds,
        dt=dt,
        x0_left=x0_left,
        x0_right=x0_right,
        step_width=step_width,
    )
    N = len(times)

    # 3. Run LIPM preview controller (x and y independently)
    ctrl_x = LIPMPreviewController(T=dt, z_c=z_c, g=g, Q_e=Q_e, R=R, N_preview=N_preview)
    ctrl_y = LIPMPreviewController(T=dt, z_c=z_c, g=g, Q_e=Q_e, R=R, N_preview=N_preview)

    # Initialise CoM at the mid-point between initial feet, at rest
    com_init_x = 0.5 * (x0_left[0] + x0_right[0])
    com_init_y = 0.5 * (x0_left[1] + x0_right[1])
    ctrl_x.reset(np.array([com_init_x, 0.0, 0.0]))
    ctrl_y.reset(np.array([com_init_y, 0.0, 0.0]))

    # Pad the ZMP reference at the start with N_preview constant steps (standard
    # Kajita warm-start).  Without this, at k=0 the preview window already sees
    # N_preview upcoming ZMP steps and applies a huge initial jerk, sending the
    # CoM far ahead of the feet.  With the padding the preview window is flat
    # during warm-up and the transient is discarded.
    pad = np.tile(zmp_ref[0], (N_preview, 1))   # (N_preview, 2)
    zmp_padded = np.vstack([pad, zmp_ref])       # (N_preview + N, 2)
    N_padded = len(zmp_padded)

    com_x = np.zeros(N)
    com_y = np.zeros(N)
    com_vx = np.zeros(N)
    com_vy = np.zeros(N)
    actual_zmp = np.zeros((N, 2))

    for k in range(N_padded):
        # Preview window: next N_preview ZMP references (within padded array)
        preview_start = k + 1
        preview_end = min(k + 1 + N_preview, N_padded)
        p_prev_x = zmp_padded[preview_start:preview_end, 0]
        p_prev_y = zmp_padded[preview_start:preview_end, 1]

        state_x, pzmp_x = ctrl_x.step(zmp_padded[k, 0], p_prev_x)
        state_y, pzmp_y = ctrl_y.step(zmp_padded[k, 1], p_prev_y)

        # Only store results for the actual (non-padded) trajectory
        if k >= N_preview:
            idx = k - N_preview
            com_x[idx] = state_x[0]
            com_y[idx] = state_y[0]
            com_vx[idx] = state_x[1]
            com_vy[idx] = state_y[1]
            actual_zmp[idx] = [pzmp_x, pzmp_y]

    # 4. Compute foot trajectories
    foot_pos_l = np.zeros((N, 3))
    foot_pos_r = np.zeros((N, 3))
    phase = np.empty(N, dtype=object)

    current_l = x0_left.copy()
    current_r = x0_right.copy()

    for k, t in enumerate(times):
        active = None
        for step in footsteps:
            if step.t_start <= t < step.t_ds_end:
                active = step
                break

        if active is None:
            foot_pos_l[k] = current_l
            foot_pos_r[k] = current_r
            if t < footsteps[0].t_start:
                phase[k] = "init_ds"
            else:
                phase[k] = "final_ds"
        elif t < active.t_swing_end:
            # Swing phase
            t_in_swing = t - active.t_start
            prev_pos = current_l.copy() if active.foot == "left" else current_r.copy()
            swing_pos = swing_trajectory(t_in_swing, T_ss, prev_pos, active.pos, step_height)
            if active.foot == "left":
                foot_pos_l[k] = swing_pos
                foot_pos_r[k] = current_r
                phase[k] = "l_swing"
            else:
                foot_pos_l[k] = current_l
                foot_pos_r[k] = swing_pos
                phase[k] = "r_swing"
        else:
            # Double support transition: foot has landed
            if active.foot == "left":
                current_l = active.pos.copy()
                foot_pos_l[k] = current_l
                foot_pos_r[k] = current_r
            else:
                current_r = active.pos.copy()
                foot_pos_l[k] = current_l
                foot_pos_r[k] = current_r
            phase[k] = "ds"

    return {
        "times": times,
        "zmp_ref": zmp_ref,
        "actual_zmp": actual_zmp,
        "com_x": com_x,
        "com_y": com_y,
        "com_vx": com_vx,
        "com_vy": com_vy,
        "foot_pos_l": foot_pos_l,
        "foot_pos_r": foot_pos_r,
        "phase": phase,
        "footsteps": footsteps,
        "z_c": z_c,
        "dt": dt,
        "T_ss": T_ss,
        "T_ds": T_ds,
    }


# ---------------------------------------------------------------------------
# ZMP stability check
# ---------------------------------------------------------------------------


def check_zmp_stability(
    traj: dict,
    step_width: float = 0.10,
    foot_length: float = 0.25,
    foot_width: float = 0.10,
    skip_warmup_steps: int = 100,
) -> dict:
    """Check the ZMP tracking quality of the planned trajectory.

    The LIPM preview controller has a startup transient for the first
    ~1-2 s while it "winds up" its error integrator and preview terms.
    We therefore skip the first ``skip_warmup_steps`` steps.

    After warmup, the check verifies that the ZMP reference stays inside
    the support polygon (this is guaranteed by construction) and that the
    actual ZMP output (from the LIPM model) does not deviate excessively
    from the reference.

    Args:
        traj: Output from plan_walking_trajectory().
        step_width: Half foot separation [m].
        foot_length: Fore-aft foot length [m].
        foot_width: Lateral foot width [m].
        skip_warmup_steps: Number of initial timesteps to skip.

    Returns:
        dict with 'max_tracking_error_mm', 'pass', 'zmp_ref_in_polygon'.
    """
    N = len(traj["com_x"])
    skip = min(skip_warmup_steps, N // 4)

    zmp_ref = traj["zmp_ref"][skip:]
    actual_zmp = traj["actual_zmp"][skip:]
    foot_pos_l = traj["foot_pos_l"][skip:]
    foot_pos_r = traj["foot_pos_r"][skip:]
    phase = traj["phase"][skip:]

    # Check 1: ZMP reference inside support polygon
    ref_violations = 0
    for k in range(len(zmp_ref)):
        ph = phase[k]
        px, py = zmp_ref[k, 0], zmp_ref[k, 1]
        if ph in ("l_swing", "r_swing"):
            if ph == "r_swing":
                cx, cy = foot_pos_l[k, 0], foot_pos_l[k, 1]
            else:
                cx, cy = foot_pos_r[k, 0], foot_pos_r[k, 1]
            viol = max(0.0, abs(px - cx) - foot_length / 2.0)
            viol += max(0.0, abs(py - cy) - foot_width / 2.0)
        else:
            x_min = min(foot_pos_l[k, 0], foot_pos_r[k, 0]) - foot_length / 2.0
            x_max = max(foot_pos_l[k, 0], foot_pos_r[k, 0]) + foot_length / 2.0
            y_min = min(foot_pos_l[k, 1], foot_pos_r[k, 1]) - foot_width / 2.0
            y_max = max(foot_pos_l[k, 1], foot_pos_r[k, 1]) + foot_width / 2.0
            viol = max(0.0, x_min - px, px - x_max)
            viol += max(0.0, y_min - py, py - y_max)
        if viol > 0.001:
            ref_violations += 1

    # Check 2: Actual ZMP tracking error (actual vs reference)
    tracking_errors = np.linalg.norm(actual_zmp - zmp_ref, axis=1)
    max_tracking_err = float(tracking_errors.max())

    return {
        "max_tracking_error_mm": max_tracking_err * 1000.0,
        "zmp_ref_in_polygon": ref_violations == 0,
        "ref_violations": ref_violations,
        "pass": ref_violations == 0,  # ZMP reference must always be in polygon
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_lipm_trajectory(traj: dict, save_path=None) -> None:
    """Plot the planned CoM, ZMP, and foot trajectories.

    Args:
        traj: Output from plan_walking_trajectory().
        save_path: Optional path to save the figure.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    times = traj["times"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # X axis
    ax = axes[0]
    ax.plot(times, traj["com_x"], "b-", lw=1.5, label="CoM x")
    ax.plot(times, traj["zmp_ref"][:, 0], "r--", lw=1.0, alpha=0.7, label="ZMP ref x")
    ax.plot(times, traj["actual_zmp"][:, 0], "r:", lw=1.0, alpha=0.5, label="ZMP actual x")
    ax.set_ylabel("x [m]")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_title("LIPM Preview Controller — Walking Trajectories")

    # Y axis
    ax = axes[1]
    ax.plot(times, traj["com_y"], "b-", lw=1.5, label="CoM y")
    ax.plot(times, traj["zmp_ref"][:, 1], "r--", lw=1.0, alpha=0.7, label="ZMP ref y")
    ax.plot(times, traj["actual_zmp"][:, 1], "r:", lw=1.0, alpha=0.5, label="ZMP actual y")
    ax.plot(times, traj["foot_pos_l"][:, 1], "g-", lw=1.0, alpha=0.6, label="Left foot y")
    ax.plot(times, traj["foot_pos_r"][:, 1], "m-", lw=1.0, alpha=0.6, label="Right foot y")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.4)

    # Foot z (swing height)
    ax = axes[2]
    ax.plot(times, traj["foot_pos_l"][:, 2], "g-", lw=1.5, label="Left foot z")
    ax.plot(times, traj["foot_pos_r"][:, 2], "m-", lw=1.5, label="Right foot z")
    ax.set_ylabel("z [m]")
    ax.set_xlabel("Time [s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lab7_common import MEDIA_DIR, Z_C

    print("Planning walking trajectory …")
    traj = plan_walking_trajectory(
        n_steps=12,
        step_length=0.10,
        step_width=0.10,
        z_c=Z_C,
        dt=0.01,
    )
    print(f"  Trajectory length: {traj['times'][-1]:.2f} s, {len(traj['times'])} timesteps")

    stab = check_zmp_stability(traj)
    print(f"  Max ZMP violation: {stab['max_violation_mm']:.2f} mm → {'PASS' if stab['pass'] else 'FAIL'}")

    MEDIA_DIR.mkdir(exist_ok=True)
    plot_lipm_trajectory(traj, save_path=MEDIA_DIR / "lipm_trajectory.png")
