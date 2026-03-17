"""Lab 7 — Phase 2 tests: LIPM preview controller and trajectory planning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab7_common import Z_C
from lipm_planner import (
    LIPMPreviewController,
    check_zmp_stability,
    generate_footsteps,
    generate_zmp_reference,
    plan_walking_trajectory,
    swing_trajectory,
)


# ---------------------------------------------------------------------------
# LIPM Preview Controller
# ---------------------------------------------------------------------------


class TestLIPMPreviewController:
    def setup_method(self):
        self.ctrl = LIPMPreviewController(T=0.01, z_c=Z_C, Q_e=1.0, R=1e-6, N_preview=200)

    def test_gains_computed(self):
        """K_e, K_s, f_gains must all have correct shapes."""
        assert np.isfinite(self.ctrl.K_e)
        assert self.ctrl.K_s.shape == (3,)
        assert self.ctrl.f_gains.shape == (200,)

    def test_preview_gains_decay(self):
        """Preview gains should decay with preview horizon."""
        f = self.ctrl.f_gains
        assert abs(f[0]) > abs(f[-1]), "Preview gains should decay with horizon"

    def test_static_tracking(self):
        """With constant ZMP reference and no initial velocity, CoM should converge."""
        p_ref = 0.0  # ZMP reference at origin
        p_preview = np.full(200, p_ref)
        self.ctrl.reset(np.array([0.05, 0.0, 0.0]))  # start 5 cm off

        for _ in range(500):
            self.ctrl.step(p_ref, p_preview)

        # CoM should converge to ZMP reference position
        assert abs(self.ctrl.x[0] - p_ref) < 1e-3, (
            f"CoM did not converge: {self.ctrl.x[0]:.4f} (should be near {p_ref})"
        )

    def test_step_returns_correct_shapes(self):
        p_ref = 0.1
        p_preview = np.full(200, p_ref)
        x_next, p_act = self.ctrl.step(p_ref, p_preview)
        assert x_next.shape == (3,)
        assert np.isfinite(p_act)

    def test_short_preview(self):
        """Should handle preview array shorter than N_preview without error."""
        self.ctrl.step(0.0, np.array([0.0, 0.0, 0.0]))  # only 3 preview steps


# ---------------------------------------------------------------------------
# Footstep generation
# ---------------------------------------------------------------------------


class TestFootstepGeneration:
    def test_alternating_feet(self):
        """Footsteps should alternate left/right starting with left."""
        steps = generate_footsteps(n_steps=6)
        feet = [s.foot for s in steps]
        assert feet[0] == "left"
        assert feet[1] == "right"
        for i in range(len(feet) - 1):
            assert feet[i] != feet[i + 1], "Consecutive steps must alternate"

    def test_step_timing_monotonic(self):
        """t_start < t_swing_end < t_ds_end for all steps."""
        steps = generate_footsteps(n_steps=8)
        for s in steps:
            assert s.t_start < s.t_swing_end < s.t_ds_end

    def test_step_positions_forward(self):
        """Each step should be further forward (x) than the previous."""
        steps = generate_footsteps(n_steps=8, step_length=0.10)
        xs = [s.pos[0] for s in steps if s.foot == "left"]
        for i in range(len(xs) - 1):
            assert xs[i + 1] > xs[i], "Left foot steps should move forward"

    def test_lateral_symmetry(self):
        """Left foot at +step_width, right foot at -step_width."""
        sw = 0.10
        steps = generate_footsteps(n_steps=4, step_width=sw)
        for s in steps:
            if s.foot == "left":
                assert abs(s.pos[1] - sw) < 1e-9
            else:
                assert abs(s.pos[1] + sw) < 1e-9


# ---------------------------------------------------------------------------
# Swing trajectory
# ---------------------------------------------------------------------------


class TestSwingTrajectory:
    def test_start_at_ground(self):
        p_start = np.array([0.0, 0.1, 0.0])
        p_end = np.array([0.1, 0.1, 0.0])
        pos = swing_trajectory(0.0, 0.8, p_start, p_end, height=0.05)
        np.testing.assert_allclose(pos[:2], p_start[:2], atol=1e-9)
        assert pos[2] < 1e-6, f"z at t=0 should be 0, got {pos[2]}"

    def test_end_at_landing(self):
        p_start = np.array([0.0, 0.1, 0.0])
        p_end = np.array([0.1, 0.1, 0.0])
        pos = swing_trajectory(0.8, 0.8, p_start, p_end, height=0.05)
        np.testing.assert_allclose(pos[:2], p_end[:2], atol=1e-9)
        assert pos[2] < 1e-6, f"z at t=T should be 0, got {pos[2]}"

    def test_peak_at_mid(self):
        p_start = np.zeros(3)
        p_end = np.array([0.1, 0.0, 0.0])
        h = 0.05
        pos_mid = swing_trajectory(0.4, 0.8, p_start, p_end, height=h)
        assert abs(pos_mid[2] - h) < 1e-6, f"Peak height should be {h}, got {pos_mid[2]}"

    def test_z_non_negative(self):
        p_start = np.zeros(3)
        p_end = np.array([0.1, 0.0, 0.0])
        for t in np.linspace(0, 0.8, 50):
            pos = swing_trajectory(t, 0.8, p_start, p_end, height=0.05)
            assert pos[2] >= -1e-9, f"Foot z went negative at t={t}: z={pos[2]}"


# ---------------------------------------------------------------------------
# ZMP stability
# ---------------------------------------------------------------------------


class TestZMPStability:
    def test_zmp_reference_inside_support_polygon(self):
        """ZMP reference must always lie inside the support polygon."""
        traj = plan_walking_trajectory(
            n_steps=8,
            step_length=0.10,
            step_width=0.10,
            T_ss=0.8,
            T_ds=0.2,
            dt=0.01,
            z_c=Z_C,
            Q_e=1.0,
            R=1e-6,
            N_preview=200,
        )
        stab = check_zmp_stability(traj, skip_warmup_steps=100)
        assert stab["zmp_ref_in_polygon"], (
            f"ZMP reference outside support polygon: {stab['ref_violations']} violations"
        )

    def test_trajectory_forward_progress(self):
        """Last CoM x should be ahead of first CoM x."""
        traj = plan_walking_trajectory(n_steps=6, step_length=0.10)
        assert traj["com_x"][-1] > traj["com_x"][0], "CoM should advance forward"

    def test_trajectory_lateral_oscillation(self):
        """CoM y should oscillate (characteristic of walking gait)."""
        traj = plan_walking_trajectory(n_steps=8, step_width=0.10)
        y_range = traj["com_y"].max() - traj["com_y"].min()
        assert y_range > 0.01, f"Expected lateral oscillation > 1 cm, got {y_range*100:.1f} cm"

    def test_foot_z_non_negative(self):
        """No foot trajectory should go below ground (z < 0)."""
        traj = plan_walking_trajectory(n_steps=6)
        assert traj["foot_pos_l"][:, 2].min() >= -1e-6
        assert traj["foot_pos_r"][:, 2].min() >= -1e-6

    def test_zmp_ref_shape(self):
        traj = plan_walking_trajectory(n_steps=4, dt=0.01)
        N = len(traj["times"])
        assert traj["zmp_ref"].shape == (N, 2)
        assert traj["com_x"].shape == (N,)
        assert traj["foot_pos_l"].shape == (N, 3)
