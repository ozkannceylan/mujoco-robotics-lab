"""Lab 8 — M0 torque-level standing controller.

The first controller of the lab, and deliberately the simplest one that can
hold the G1 upright while every command goes out as a **torque**:

    τ = K_p (q_nom − q) + K_d (−q̇)  +  g_comp(q)

with three selectable gravity-compensation modes so M0 can *measure* which
assumption is right instead of asserting one (see `GravityMode`).

Why this is not just Lab 7 with extra steps: Lab 7's `<position>` servos
computed an equivalent law *inside MuJoCo*, which meant the control input was a
joint angle and there was no way to inject a torque from an inverse-dynamics
pipeline. Here the same standing behaviour is produced from outside the
simulator, so M1's QP → RNEA output has somewhere to go.

Findings that shaped this module (full write-up in tasks/LESSONS.md):

* Gravity compensation **alone** does not stand: the robot is an inverted
  pendulum and g(q) cancels weight without stabilising posture. It collapses to
  a pelvis height of 0.097 m in ~2 s.
* Inertia-shaping the PD gains through M(q) — Lab 5's L-6.1b fix for a
  fixed-base arm — **makes the humanoid fall**. See `LESSONS.md` L-M0-b.
"""

from __future__ import annotations

from enum import Enum

import mujoco
import numpy as np
import pinocchio as pin

from lab8_common import (
    NU,
    Q_STAND_JOINTS,
    clip_torques,
    joint_torques_to_ctrl,
    mj_state_to_pin,
)


class GravityMode(str, Enum):
    """How (or whether) gravity is compensated.

    NONE
        Pure joint PD. What the Menagerie position servos effectively did.
    FREE_SPACE
        τ += g(q) from Pinocchio's free-space RNEA gravity term. Correct for a
        robot hanging in the air; while standing it double-counts the weight
        the ground already carries.
    CONTACT_CONSISTENT
        τ += g(q) − τ_constraint, i.e. free-space gravity minus the
        generalized constraint forces MuJoCo reports for the active foot
        contacts. This is the standing-specific special case of the
        contact-consistent inverse dynamics M1 builds properly, and it gives
        the tightest posture tracking of the three.
    """

    NONE = "none"
    FREE_SPACE = "free_space"
    CONTACT_CONSISTENT = "contact_consistent"


class StandingController:
    """Joint-space PD + selectable gravity compensation, torque output.

    Args:
        mj_model: Torque-actuated G1 model.
        pin_model: Pinocchio model built from the same MJCF.
        pin_data: Pinocchio data.
        q_nom: Nominal joint configuration (29,). Defaults to the Menagerie
            "stand" keyframe pose.
        kp: Proportional gain [N·m/rad].
        kd: Derivative gain [N·m·s/rad].
        gravity_mode: See `GravityMode`.

    Note:
        Gains are applied **raw**, not scaled by M(q). That is intentional and
        load-bearing — see the class docstring.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        pin_model: pin.Model,
        pin_data: pin.Data,
        q_nom: np.ndarray | None = None,
        kp: float = 500.0,
        kd: float = 50.0,
        gravity_mode: GravityMode = GravityMode.CONTACT_CONSISTENT,
    ) -> None:
        self.mj_model = mj_model
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.q_nom = (Q_STAND_JOINTS if q_nom is None else np.asarray(q_nom)).copy()
        if self.q_nom.shape != (NU,):
            raise ValueError(f"q_nom must have shape ({NU},), got {self.q_nom.shape}")
        self.kp = float(kp)
        self.kd = float(kd)
        self.gravity_mode = GravityMode(gravity_mode)

    def compute_torque(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Return the clipped joint torque command (29,) for the current state."""
        # Slice to the robot's own joints: a scene may append free bodies
        # after the G1 (M5's payload), which lengthens qpos/qvel beyond the
        # 29 actuated joints this controller drives.
        q_joints = mj_data.qpos[7:7 + NU]
        v_joints = mj_data.qvel[6:6 + NU]

        tau = self.kp * (self.q_nom - q_joints) + self.kd * (-v_joints)

        if self.gravity_mode is not GravityMode.NONE:
            q, _ = mj_state_to_pin(mj_data)
            pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
            gravity = joint_torques_to_ctrl(self.pin_data.g)
            if self.gravity_mode is GravityMode.CONTACT_CONSISTENT:
                gravity = gravity - mj_data.qfrc_constraint[6:6 + NU]
            tau = tau + gravity

        return clip_torques(tau, self.mj_model)

    def step(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Compute the torque, write it to ctrl, and advance the simulation."""
        tau = self.compute_torque(mj_data)
        mj_data.ctrl[:] = tau
        mujoco.mj_step(self.mj_model, mj_data)
        return tau
