"""Lab 6 — Phase 1 tests: IK convergence for all 8 bimanual configurations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lab6_common import (
    BAR_PICK_POS,
    BAR_PLACE_POS,
    GRIPPER_TIP_OFFSET,
    LEFT_ARM_BASE_POS,
    LEFT_GRIP_Y_OFFSET,
    PREGRASP_CLEARANCE,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    RIGHT_ARM_BASE_POS,
    RIGHT_GRIP_Y_OFFSET,
    load_both_pinocchio_models,
)
from dual_arm_ik import (
    BimanualGraspConfigs,
    compute_bimanual_configs,
    compute_ik,
    GraspConfigs,
)


@pytest.fixture(scope="module")
def pin_models():
    (pin_L, data_L, ee_L), (pin_R, data_R, ee_R) = load_both_pinocchio_models()
    return (pin_L, data_L, ee_L), (pin_R, data_R, ee_R)


@pytest.fixture(scope="module")
def bimanual_cfgs(pin_models):
    (pin_L, data_L, ee_L), (pin_R, data_R, ee_R) = pin_models
    return compute_bimanual_configs(pin_L, data_L, ee_L, pin_R, data_R, ee_R)


class TestIKConvergence:
    """DLS IK must converge for each of the 8 configurations (2 arms × 4 phases).

    Expected values are in WORLD frame (arm-local + arm_base_pos).
    Left arm base: (0, -0.35, 0); Right arm base: (0, 0.35, 0).
    """

    @pytest.mark.parametrize("arm,phase,world_x,world_y,world_z", [
        # Bar is Y-oriented; each arm grips its own Y-end (no arm crossing)
        # LEFT_GRIP_Y_OFFSET=-0.10: pick grip y=-0.10, place grip y=0.10
        ("left",  "pregrasp", 0.450, -0.100, 0.590),
        ("left",  "grasp",    0.450, -0.100, 0.440),
        ("left",  "preplace", 0.450,  0.100, 0.590),
        ("left",  "place",    0.450,  0.100, 0.440),
        # RIGHT_GRIP_Y_OFFSET=+0.10: pick grip y=+0.10, place grip y=0.30
        ("right", "pregrasp", 0.450,  0.100, 0.590),
        ("right", "grasp",    0.450,  0.100, 0.440),
        ("right", "preplace", 0.450,  0.300, 0.590),
        ("right", "place",    0.450,  0.300, 0.440),
    ])
    def test_ee_world_position(
        self, bimanual_cfgs, pin_models,
        arm, phase, world_x, world_y, world_z,
    ):
        """EE world position for each config must match expected within 5 mm.

        Pinocchio reports arm-local positions; we add the arm base offset to
        compare against the expected world-frame target.
        """
        import pinocchio as pin

        (pin_L, data_L, ee_L), (pin_R, data_R, ee_R) = pin_models
        cfgs = bimanual_cfgs.left if arm == "left" else bimanual_cfgs.right
        pm, pd, fid = (pin_L, data_L, ee_L) if arm == "left" else (pin_R, data_R, ee_R)
        base = LEFT_ARM_BASE_POS if arm == "left" else RIGHT_ARM_BASE_POS

        q_map = {
            "pregrasp": cfgs.q_pregrasp,
            "grasp":    cfgs.q_grasp,
            "preplace": cfgs.q_preplace,
            "place":    cfgs.q_place,
        }
        q = q_map[phase]

        pin.forwardKinematics(pm, pd, q)
        pin.updateFramePlacements(pm, pd)
        ee_local = pd.oMf[fid].translation
        ee_world = ee_local + base

        expected = np.array([world_x, world_y, world_z])
        err_mm = np.linalg.norm(ee_world - expected) * 1000
        assert err_mm < 5.0, (
            f"{arm} arm {phase}: world EE error {err_mm:.1f} mm > 5 mm. "
            f"Got world {ee_world.round(3)}, expected {expected}"
        )

    def test_bimanual_cfgs_type(self, bimanual_cfgs):
        """compute_bimanual_configs must return a BimanualGraspConfigs."""
        assert isinstance(bimanual_cfgs, BimanualGraspConfigs)
        assert isinstance(bimanual_cfgs.left, GraspConfigs)
        assert isinstance(bimanual_cfgs.right, GraspConfigs)

    def test_q_home_preserved(self, bimanual_cfgs):
        """q_home in configs must match the defined Q_HOME constants."""
        np.testing.assert_allclose(bimanual_cfgs.left.q_home,  Q_HOME_LEFT,  atol=1e-10)
        np.testing.assert_allclose(bimanual_cfgs.right.q_home, Q_HOME_RIGHT, atol=1e-10)

    def test_ik_returns_none_for_unreachable(self, pin_models):
        """IK must return None for an unreachable target."""
        (pin_L, data_L, ee_L), _ = pin_models
        # Target 2 m away from base — outside UR5e workspace
        q_sol = compute_ik(
            pin_L, data_L, ee_L,
            np.array([2.0, 2.0, 2.0]),
            np.eye(3),
            Q_HOME_LEFT,
            max_iter=50,
        )
        assert q_sol is None
