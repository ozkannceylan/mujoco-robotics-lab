"""Tests for Lab 5 gripper controller and scene setup."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

# Add src/ to path
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from lab5_common import (
    GRIPPER_CLOSED,
    GRIPPER_IDX,
    GRIPPER_OPEN,
    NUM_JOINTS,
    Q_HOME,
    load_mujoco_model,
    load_pinocchio_model,
)
from gripper_controller import (
    close_gripper,
    get_finger_position,
    get_finger_velocity,
    is_gripper_in_contact,
    is_gripper_settled,
    open_gripper,
    step_until_settled,
)


@pytest.fixture
def mj():
    """Return loaded (model, data) at Q_HOME with gripper open."""
    mj_model, mj_data = load_mujoco_model()
    mj_data.qpos[:NUM_JOINTS] = Q_HOME.copy()
    mj_data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
    mujoco.mj_forward(mj_model, mj_data)
    return mj_model, mj_data


# ---------------------------------------------------------------------------
# Scene structure
# ---------------------------------------------------------------------------

class TestSceneLoads:
    def test_nq(self, mj):
        m, _ = mj
        # 6 arm + 2 finger + 7 freejoint = 15
        assert m.nq == 15

    def test_nu(self, mj):
        m, _ = mj
        # 6 arm torque + 1 gripper position
        assert m.nu == 7

    def test_required_geoms_exist(self, mj):
        m, _ = mj
        for name in ["left_pad", "right_pad", "box_geom", "table_top"]:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert gid >= 0, f"Geom '{name}' not found in model"

    def test_gripper_site_exists(self, mj):
        m, _ = mj
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
        assert sid >= 0

    def test_no_initial_contacts(self, mj):
        m, d = mj
        assert d.ncon == 0, f"Expected 0 contacts at Q_HOME, got {d.ncon}"


# ---------------------------------------------------------------------------
# Gripper open / close commands
# ---------------------------------------------------------------------------

class TestGripperCommands:
    def test_open_sets_ctrl(self, mj):
        m, d = mj
        open_gripper(d, open_pos=0.025)
        assert d.ctrl[GRIPPER_IDX] == pytest.approx(0.025)

    def test_close_sets_ctrl(self, mj):
        m, d = mj
        close_gripper(d)
        assert d.ctrl[GRIPPER_IDX] == pytest.approx(GRIPPER_CLOSED)

    def test_open_then_close(self, mj):
        m, d = mj
        open_gripper(d)
        assert d.ctrl[GRIPPER_IDX] == pytest.approx(GRIPPER_OPEN)
        close_gripper(d)
        assert d.ctrl[GRIPPER_IDX] == pytest.approx(GRIPPER_CLOSED)


# ---------------------------------------------------------------------------
# Gripper state queries
# ---------------------------------------------------------------------------

class TestGripperState:
    def test_finger_position_at_open(self, mj):
        m, d = mj
        # ctrl already set to GRIPPER_OPEN; step to settle
        step_until_settled(m, d, max_steps=2000)
        pos = get_finger_position(m, d)
        assert abs(pos - GRIPPER_OPEN) < 1e-3, (
            f"Expected finger at {GRIPPER_OPEN:.3f} m, got {pos:.3f} m"
        )

    def test_finger_settles_after_command(self, mj):
        m, d = mj
        close_gripper(d)
        # Fingers start at open; give them time to close (no box → they reach limit)
        step_until_settled(m, d, max_steps=3000)
        assert is_gripper_settled(m, d, vel_thresh=1e-3)

    def test_no_contact_at_home_open(self, mj):
        m, d = mj
        mujoco.mj_forward(m, d)
        assert not is_gripper_in_contact(m, d)

    def test_contact_when_fingers_on_box(self, mj):
        """Place the box directly between the fingers and close; expect contact."""
        m, d = mj
        # Position box between fingers at current EE height
        gripper_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
        mujoco.mj_forward(m, d)
        site_pos = d.site_xpos[gripper_site_id].copy()

        # Box freejoint qpos starts at index 8 (6 arm + 2 finger)
        box_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        box_qadr = m.jnt_qposadr[box_jid]
        d.qpos[box_qadr:box_qadr + 3] = site_pos
        d.qpos[box_qadr + 3] = 1.0   # quaternion w
        d.qpos[box_qadr + 4:box_qadr + 7] = 0.0

        close_gripper(d)
        # Step and check for contact during the closing motion.
        # The box falls under gravity once contact is broken, so we verify
        # that contact occurs at any point within the first 200 steps.
        contact_detected = False
        for _ in range(200):
            mujoco.mj_step(m, d)
            if is_gripper_in_contact(m, d):
                contact_detected = True
                break

        assert contact_detected, "Expected finger-box contact during gripper closing"


# ---------------------------------------------------------------------------
# Pinocchio model
# ---------------------------------------------------------------------------

class TestPinocchioModel:
    def test_model_loads(self):
        model, data, fid = load_pinocchio_model()
        assert model.nq == 6
        assert fid >= 0

    def test_ee_link_frame_exists(self):
        model, data, fid = load_pinocchio_model()
        name = model.frames[fid].name
        assert name == "ee_link"
