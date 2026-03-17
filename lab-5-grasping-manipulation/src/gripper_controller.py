"""Lab 5 — Gripper controller for the parallel-jaw gripper.

Controls the position-actuated parallel jaw gripper (ctrl index 6).
Left and right fingers are mirrored via an equality constraint in MuJoCo,
so a single setpoint drives both symmetrically.

  ctrl[6] = 0.000  → closed (fingers squeeze object)
  ctrl[6] = 0.030  → open   (60 mm total jaw gap)
"""

from __future__ import annotations

import mujoco
import numpy as np

from lab5_common import GRIPPER_CLOSED, GRIPPER_IDX, GRIPPER_OPEN, DT


# ---------------------------------------------------------------------------
# Gripper setpoints
# ---------------------------------------------------------------------------

def open_gripper(mj_data, open_pos: float = GRIPPER_OPEN) -> None:
    """Command the gripper to the open position.

    Args:
        mj_data: MuJoCo data object (ctrl is modified in-place).
        open_pos: Target finger slide distance (m). Default: GRIPPER_OPEN.
    """
    mj_data.ctrl[GRIPPER_IDX] = open_pos


def close_gripper(mj_data, close_pos: float = GRIPPER_CLOSED) -> None:
    """Command the gripper to the closed (squeeze) position.

    Args:
        mj_data: MuJoCo data object (ctrl is modified in-place).
        close_pos: Target finger slide distance (m). Default: GRIPPER_CLOSED.
    """
    mj_data.ctrl[GRIPPER_IDX] = close_pos


# ---------------------------------------------------------------------------
# State queries
# ---------------------------------------------------------------------------

def get_finger_position(mj_model, mj_data) -> float:
    """Return current left_finger_joint position (m).

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.

    Returns:
        Current slide position of left_finger_joint (m).
    """
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_joint")
    return float(mj_data.qpos[mj_model.jnt_qposadr[jid]])


def get_finger_velocity(mj_model, mj_data) -> float:
    """Return left_finger_joint slide velocity (m/s).

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.

    Returns:
        Finger velocity (m/s).
    """
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_joint")
    return float(mj_data.qvel[mj_model.jnt_dofadr[jid]])


def is_gripper_settled(mj_model, mj_data, vel_thresh: float = 5e-4) -> bool:
    """Check if the gripper fingers have stopped moving.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        vel_thresh: Velocity threshold below which motion is considered settled (m/s).

    Returns:
        True if |finger_velocity| < vel_thresh.
    """
    return abs(get_finger_velocity(mj_model, mj_data)) < vel_thresh


def is_gripper_in_contact(mj_model, mj_data) -> bool:
    """Check if any finger geometry is in contact with any object.

    Iterates over all active contacts and checks if any finger geom
    ('left_pad', 'right_pad', 'left_finger_geom', 'right_finger_geom')
    appears in the contact pair.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.

    Returns:
        True if at least one finger geom is in contact.
    """
    finger_geom_names = (
        "left_pad", "right_pad", "left_finger_geom", "right_finger_geom"
    )
    finger_ids = {
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in finger_geom_names
    }

    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        if c.geom1 in finger_ids or c.geom2 in finger_ids:
            return True
    return False


# ---------------------------------------------------------------------------
# Blocking settlement helper (for use inside simulation loops)
# ---------------------------------------------------------------------------

def step_until_settled(
    mj_model,
    mj_data,
    max_steps: int = 500,
    vel_thresh: float = 5e-4,
) -> int:
    """Step the simulation until the gripper settles or timeout.

    Intended to be called after issuing an open or close command to wait
    for the fingers to reach equilibrium before the next state transition.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data. ctrl must already have the target set.
        max_steps: Maximum simulation steps to wait.
        vel_thresh: Finger velocity threshold for settled (m/s).

    Returns:
        Number of steps taken.
    """
    for step in range(max_steps):
        mujoco.mj_step(mj_model, mj_data)
        if is_gripper_settled(mj_model, mj_data, vel_thresh):
            return step + 1
    return max_steps
