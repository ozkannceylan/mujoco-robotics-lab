"""M3 — Grasp pose calculator for dual-arm box approach.

Computes approach and grasp standoff SE3 poses from the current
box pose in MuJoCo.  All targets are derived from the box body
position and orientation — nothing is hardcoded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np
import pinocchio as pin

from lab6_common import BOX_HALF_EXTENTS


# Standoff distances from box surface along the approach axis
GRASP_STANDOFF = 0.05    # 5 cm from box surface
APPROACH_STANDOFF = 0.10  # 10 cm from box surface


def _rotation_facing(direction: np.ndarray) -> np.ndarray:
    """Build rotation matrix with z-axis along *direction* and y-axis ~ up.

    Args:
        direction: Unit vector for EE z-axis (approach direction).

    Returns:
        Rotation matrix (3, 3).
    """
    z = direction / np.linalg.norm(direction)

    # Choose y-axis as close to world +z as possible
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, up)) > 0.99:
        # z nearly vertical — fall back to world +y
        up = np.array([0.0, 1.0, 0.0])

    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def compute_grasp_poses(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    box_body_name: str = "box",
) -> dict[str, pin.SE3]:
    """Compute approach and grasp standoff poses from box state.

    The left arm approaches from the -x side (EE z → +x toward box).
    The right arm approaches from the +x side (EE z → -x toward box).

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data (must have mj_forward called).
        box_body_name: Name of the box body in the model.

    Returns:
        Dict with keys: left_approach, left_grasp, right_approach, right_grasp.
        Each value is a pinocchio.SE3 pose in world frame.
    """
    box_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, box_body_name)
    box_pos = mj_data.xpos[box_id].copy()
    box_rot = mj_data.xmat[box_id].reshape(3, 3).copy()

    # Box x-axis in world frame (longest dimension direction)
    box_x_axis = box_rot[:, 0]
    box_half_x = BOX_HALF_EXTENTS[0]

    # Left arm: approach along -x_box direction (from -x side)
    left_dir = box_x_axis.copy()        # EE z points toward +x (toward box center)
    left_grasp_pos = box_pos - box_x_axis * (box_half_x + GRASP_STANDOFF)
    left_approach_pos = box_pos - box_x_axis * (box_half_x + APPROACH_STANDOFF)
    R_left = _rotation_facing(left_dir)

    # Right arm: approach along +x_box direction (from +x side)
    right_dir = -box_x_axis.copy()      # EE z points toward -x (toward box center)
    right_grasp_pos = box_pos + box_x_axis * (box_half_x + GRASP_STANDOFF)
    right_approach_pos = box_pos + box_x_axis * (box_half_x + APPROACH_STANDOFF)
    R_right = _rotation_facing(right_dir)

    return {
        "left_approach": pin.SE3(R_left, left_approach_pos),
        "left_grasp": pin.SE3(R_left, left_grasp_pos),
        "right_approach": pin.SE3(R_right, right_approach_pos),
        "right_grasp": pin.SE3(R_right, right_grasp_pos),
    }
