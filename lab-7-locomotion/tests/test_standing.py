"""Lab 7 — Phase 1 tests: G1 standing balance and model loading."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab7_common import (
    CTRL_LEFT_LEG,
    CTRL_RIGHT_LEG,
    CTRL_WAIST,
    NQ,
    NV,
    NU,
    Q_STAND_JOINTS,
    Z_C,
    build_pin_q_standing,
    load_g1_mujoco,
    load_g1_pinocchio,
    mj_qpos_to_pin,
    pin_q_to_mj,
)
from standing_controller import run_standing_demo


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def test_mujoco_model_dimensions():
    m, d = load_g1_mujoco()
    assert m.nq == NQ, f"Expected nq={NQ}, got {m.nq}"
    assert m.nv == NV, f"Expected nv={NV}, got {m.nv}"
    assert m.nu == NU, f"Expected nu={NU}, got {m.nu}"


def test_pinocchio_model_dimensions():
    model, data = load_g1_pinocchio()
    assert model.nq == NQ, f"Expected nq={NQ}, got {model.nq}"
    assert model.nv == NV, f"Expected nv={NV}, got {model.nv}"


def test_mujoco_has_foot_sites():
    import mujoco
    m, _ = load_g1_mujoco()
    site_names = [m.site(i).name for i in range(m.nsite)]
    assert "left_foot" in site_names
    assert "right_foot" in site_names


def test_pinocchio_has_foot_frames():
    model, _ = load_g1_pinocchio()
    lf_id = model.getFrameId("left_foot")
    rf_id = model.getFrameId("right_foot")
    assert lf_id < model.nframes, "left_foot frame not found"
    assert rf_id < model.nframes, "right_foot frame not found"


# ---------------------------------------------------------------------------
# Quaternion conversion
# ---------------------------------------------------------------------------


def test_quaternion_round_trip():
    """mj_qpos_to_pin and pin_q_to_mj must be inverses of each other."""
    rng = np.random.default_rng(42)
    qpos_mj = np.zeros(NQ)
    qpos_mj[:3] = rng.uniform(-1, 1, 3)
    # Random unit quaternion (MuJoCo: w,x,y,z)
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    qpos_mj[3:7] = q
    qpos_mj[7:] = rng.uniform(-0.5, 0.5, NQ - 7)

    pin_q = mj_qpos_to_pin(qpos_mj)
    recovered = pin_q_to_mj(pin_q)
    np.testing.assert_allclose(recovered, qpos_mj, atol=1e-12)


def test_upright_quaternion_convention():
    """Upright MuJoCo quat [w=1,x=0,y=0,z=0] → Pinocchio [x=0,y=0,z=0,w=1]."""
    q_mj = np.zeros(NQ)
    q_mj[3:7] = [1, 0, 0, 0]   # MuJoCo upright
    q_pin = mj_qpos_to_pin(q_mj)
    np.testing.assert_allclose(q_pin[3:7], [0, 0, 0, 1], atol=1e-12)


# ---------------------------------------------------------------------------
# Standing balance
# ---------------------------------------------------------------------------


def test_standing_robot_stays_up():
    """G1 with position servos at keyframe values should not fall."""
    result = run_standing_demo(duration_s=2.0, perturbation_force_n=0.0,
                                perturbation_time_s=99.0)
    # Pelvis should stay above 0.5 m (not fallen)
    # We use the CoM z proxy
    min_z = result["com_z"].min()
    assert min_z > 0.4, f"Robot fell! Min CoM z = {min_z:.3f} m"


def test_standing_perturbation_recovery():
    """G1 should recover within 15 mm of initial CoM after a 5 N push."""
    result = run_standing_demo(
        duration_s=6.0,
        perturbation_force_n=5.0,
        perturbation_time_s=2.0,
        perturbation_duration_s=0.2,
    )
    assert result["success"], (
        f"Recovery failed: error = {result['recovery_error_mm']:.1f} mm"
    )


def test_standing_com_height():
    """Settled CoM z should be close to Z_C (within 10 cm)."""
    result = run_standing_demo(duration_s=2.0, perturbation_force_n=0.0,
                                perturbation_time_s=99.0)
    # Take the average of the last 0.5 s
    com_z_final = float(np.mean(result["com_z"][-50:]))
    assert abs(com_z_final - Z_C) < 0.10, (
        f"Settled CoM z = {com_z_final:.3f} m, expected near {Z_C:.3f} m"
    )
