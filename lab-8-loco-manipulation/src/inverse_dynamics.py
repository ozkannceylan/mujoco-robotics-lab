"""Lab 8 — M1 Step 1.3: QP velocity → joint torques.

Bridges the velocity-level whole-body QP to the torque-actuated robot::

    q̇_des  ──integrate──►  q_des  ──joint servo──►  τ

    τ = K_p (q_des − q) + K_d (q̇_des − q̇) + g(q) − τ_constraint

Design notes, both earned rather than assumed:

* **Raw gains, not M(q)-shaped.** M0 measured that inertia-shaping the gains
  through the floating-base mass matrix makes the G1 fall at every setting
  (LESSONS L-M0-b), reversing Lab 5's fixed-base rule.
* **Contact-consistent gravity.** Free-space `g(q)` compensates weight the
  ground already carries; subtracting MuJoCo's generalized constraint forces
  drove M0's steady-state joint error to 0.00 mrad.

The reference `q_des` is *integrated and held* rather than recomputed as
`q + q̇_des·dt` each tick. With a per-tick target the proportional term would
be O(q̇·dt) — no posture stiffness at all, leaving the robot to fall the moment
the QP's velocity is imperfectly tracked. Holding the reference lets position
error accumulate and be corrected, so the servo does real work. To keep that
from becoming integrator windup when the robot cannot follow (a foot slips, a
joint saturates), the reference is clamped to stay within
`max_tracking_error` of the measurement.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from lab8_common import (
    NU,
    NV,
    clip_torques,
    joint_torques_to_ctrl,
    mj_state_to_pin,
)

__all__ = ["InverseDynamics"]


class InverseDynamics:
    """Track a QP joint-velocity command with a gravity-compensated servo.

    Args:
        mj_model: Torque-actuated MuJoCo model (for torque limits).
        pin_model: Pinocchio model.
        pin_data: Pinocchio data.
        dt: Control timestep [s].
        kp: Joint proportional gain [N·m/rad].
        kd: Joint derivative gain [N·m·s/rad].
        contact_consistent: Subtract MuJoCo constraint forces from g(q).
        max_tracking_error: Cap on |q_des − q| per joint [rad]; prevents the
            held reference from winding up away from a robot that cannot
            follow it.
    """

    def __init__(
        self,
        mj_model,
        pin_model: pin.Model,
        pin_data: pin.Data,
        dt: float,
        kp: float = 500.0,
        kd: float = 50.0,
        contact_consistent: bool = True,
        max_tracking_error: float = 0.20,
    ) -> None:
        self.mj_model = mj_model
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.dt = float(dt)
        self.kp = float(kp)
        self.kd = float(kd)
        self.contact_consistent = bool(contact_consistent)
        self.max_tracking_error = float(max_tracking_error)
        self._q_ref: np.ndarray | None = None

    # -- reference ---------------------------------------------------------

    def reset(self, q_pin: np.ndarray) -> None:
        """Reset the held reference to the current configuration."""
        self._q_ref = np.asarray(q_pin, dtype=float).copy()

    @property
    def q_ref(self) -> np.ndarray | None:
        """Current integrated reference configuration (nq,), or None."""
        return None if self._q_ref is None else self._q_ref.copy()

    def integrate_reference(self, qdot_des: np.ndarray, q_measured: np.ndarray) -> np.ndarray:
        """Advance the held reference by one tick and clamp it to the robot.

        `pin.integrate` — never `q += dq` — because the floating base carries a
        quaternion (nq=36 ≠ nv=35).
        """
        if self._q_ref is None:
            self.reset(q_measured)
        self._q_ref = pin.integrate(self.pin_model, self._q_ref, np.asarray(qdot_des) * self.dt)

        # The base is not actuated: keep its reference glued to the measurement
        # so the joint servo never chases an unreachable base pose.
        self._q_ref[:7] = np.asarray(q_measured, dtype=float)[:7]

        error = self._q_ref[7:] - q_measured[7:]
        clamped = np.clip(error, -self.max_tracking_error, self.max_tracking_error)
        self._q_ref[7:] = q_measured[7:] + clamped
        return self._q_ref

    # -- torque ------------------------------------------------------------

    def compute_torque(self, mj_data, qdot_des: np.ndarray) -> np.ndarray:
        """Return the clipped joint torque (29,) realising `qdot_des`."""
        qdot_des = np.asarray(qdot_des, dtype=float)
        if qdot_des.shape != (NV,):
            raise ValueError(f"qdot_des must be ({NV},), got {qdot_des.shape}")

        q_measured, _ = mj_state_to_pin(mj_data)
        q_ref = self.integrate_reference(qdot_des, q_measured)

        position_error = q_ref[7:] - mj_data.qpos[7:]
        velocity_error = qdot_des[6:] - mj_data.qvel[6:]

        pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q_measured)
        gravity = joint_torques_to_ctrl(self.pin_data.g)
        if self.contact_consistent:
            gravity = gravity - mj_data.qfrc_constraint[6:]

        tau = self.kp * position_error + self.kd * velocity_error + gravity
        return clip_torques(tau, self.mj_model)
