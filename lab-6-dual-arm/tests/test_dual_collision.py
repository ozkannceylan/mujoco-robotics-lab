"""Tests for DualCollisionChecker — self, environment, and cross-arm collision detection."""
import sys
import math
from pathlib import Path

import numpy as np
import pytest
import pinocchio as pin

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from lab6_common import Q_HOME_LEFT, Q_HOME_RIGHT, TABLE_SURFACE_Z
from dual_collision_checker import DualCollisionChecker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def checker() -> DualCollisionChecker:
    """Shared DualCollisionChecker instance for the test module."""
    return DualCollisionChecker()


# ---------------------------------------------------------------------------
# Basic structural checks
# ---------------------------------------------------------------------------

def test_collision_pairs_registered(checker):
    """At least one collision pair is registered for each arm."""
    assert checker.num_left_pairs > 0, "No collision pairs registered for left arm"
    assert checker.num_right_pairs > 0, "No collision pairs registered for right arm"


def test_cross_pairs_count(checker):
    """Cross-arm pair count equals n_left_geoms * n_right_geoms."""
    expected = checker._n_left_robot_geoms * checker._n_right_robot_geoms
    assert checker.num_cross_pairs == expected


def test_total_pairs_count(checker):
    """Total pairs is sum of left, right, and cross-arm pairs."""
    expected = checker.num_left_pairs + checker.num_right_pairs + checker.num_cross_pairs
    assert checker.num_total_pairs == expected


# ---------------------------------------------------------------------------
# Home configuration is collision-free
# ---------------------------------------------------------------------------

def test_home_config_collision_free(checker):
    """Home configuration for both arms is collision-free."""
    assert checker.is_collision_free(Q_HOME_LEFT, Q_HOME_RIGHT), \
        "Home configuration should be collision-free"


def test_zeros_config_collision_free(checker):
    """All-zeros configuration for both arms is collision-free (arms pointing up)."""
    assert checker.is_collision_free(np.zeros(6), np.zeros(6)), \
        "Zero configuration should be collision-free"


# ---------------------------------------------------------------------------
# Environment collision (arm into table)
# ---------------------------------------------------------------------------

def test_arm_into_table_detected(checker):
    """Config that drives the left EE below table surface is detected as collision."""
    # Check the checker at home first (sanity guard)
    assert checker.is_collision_free(Q_HOME_LEFT, Q_HOME_RIGHT)

    # Rotate shoulder_pan toward table (pi/2), then extend arm downward.
    # At sp=pi/2, sl=0.5 the arm penetrates through the table.
    q_into_table_a = np.array([math.pi / 2, 0.5, 0.0, 0.0, 0.0, 0.0])
    # Even more extreme: sl=0.7
    q_into_table_b = np.array([math.pi / 2, 0.7, 0.0, 0.0, 0.0, 0.0])

    # At least one of these should be in collision with the table
    col_a = not checker.is_collision_free(q_into_table_a, Q_HOME_RIGHT)
    col_b = not checker.is_collision_free(q_into_table_b, Q_HOME_RIGHT)
    assert col_a or col_b, (
        "Expected at least one config with arm reaching through table to collide. "
        "Check table geometry or config choice."
    )


def test_min_distance_positive_at_home(checker):
    """Minimum distance is positive (no penetration) at home configuration."""
    d = checker.get_min_distance(Q_HOME_LEFT, Q_HOME_RIGHT)
    assert d > 0.0, f"Expected positive clearance at home, got {d:.4f} m"


# ---------------------------------------------------------------------------
# Self-collision
# ---------------------------------------------------------------------------

def test_self_collision_detected_folded_left(checker):
    """Fully self-colliding left arm configuration is detected."""
    # Fold the arm on itself: alternate max/min joint angles to make links cross.
    # shoulder_pan = 0, shoulder_lift = pi*0.9, elbow = -pi*0.9, wrists folded back.
    q_folded = np.array([0.0, math.pi * 0.9, -math.pi * 0.9, math.pi * 0.8, 0.0, 0.0])
    in_collision = not checker.is_collision_free(q_folded, Q_HOME_RIGHT)
    assert in_collision, (
        "Expected a severely folded left arm to be in self-collision or table collision. "
        "If the URDF collision geometry is convex-hull only, self-collision may not trigger — "
        "adjust config or tolerance."
    )


def test_self_collision_detected_folded_right(checker):
    """Fully self-colliding right arm configuration is detected."""
    q_folded = np.array([0.0, math.pi * 0.9, -math.pi * 0.9, math.pi * 0.8, 0.0, 0.0])
    in_collision = not checker.is_collision_free(Q_HOME_LEFT, q_folded)
    assert in_collision, (
        "Expected a severely folded right arm to be in self-collision or table collision."
    )


# ---------------------------------------------------------------------------
# Cross-arm collision
# ---------------------------------------------------------------------------

def test_cross_arm_collision_both_reaching_center(checker):
    """Both arms reaching to the same center point triggers cross-arm collision."""
    # Use IK-solved configs where both arms reach to the same point (0.5, 0, 0.4).
    # These configs verified to place both EEs within 0.1mm of each other, so
    # arm links must overlap/collide in the shared workspace.
    q_left_center = np.array([0.485, -2.168, -1.661, -4.025, -4.712, -5.197])
    q_right_center = np.array([0.485, 4.115, -1.661, 2.259, 1.571, -2.056])

    col_a = not checker.is_collision_free(q_left_center, q_right_center)
    assert col_a, (
        "Expected both arms reaching to the same point (0.5, 0, 0.4) "
        "to produce a cross-arm collision."
    )


# ---------------------------------------------------------------------------
# is_path_free
# ---------------------------------------------------------------------------

def test_path_free_home_to_home(checker):
    """Trivial path from home to home is free."""
    result = checker.is_path_free(
        Q_HOME_LEFT, Q_HOME_LEFT,
        Q_HOME_RIGHT, Q_HOME_RIGHT,
    )
    assert result is True


def test_path_free_home_to_zeros(checker):
    """Path from home to zeros (both arms moving away from table) is collision-free."""
    result = checker.is_path_free(
        Q_HOME_LEFT, np.zeros(6),
        Q_HOME_RIGHT, np.zeros(6),
    )
    assert result is True, (
        "Path from home to zero config should not pass through collisions. "
        "If it does, review home config placement relative to table."
    )


def test_path_through_collision_detected(checker):
    """Path that passes through a known collision returns False."""
    # Start at home (collision-free), end at a config known to be in collision
    q_folded = np.array([0.0, math.pi * 0.9, -math.pi * 0.9, math.pi * 0.8, 0.0, 0.0])

    # Only move the left arm; keep right at home
    result = checker.is_path_free(
        Q_HOME_LEFT, q_folded,
        Q_HOME_RIGHT, Q_HOME_RIGHT,
    )
    # At least some part of the interpolated path should be in collision
    # (we already verified q_folded is in collision above).
    assert result is False, (
        "Path ending at a collision config should be detected as not free."
    )


def test_path_free_resolution_parameter(checker):
    """is_path_free accepts custom resolution without error."""
    result = checker.is_path_free(
        Q_HOME_LEFT, Q_HOME_LEFT,
        Q_HOME_RIGHT, Q_HOME_RIGHT,
        resolution=0.01,
    )
    assert result is True


# ---------------------------------------------------------------------------
# get_min_distance
# ---------------------------------------------------------------------------

def test_min_distance_returns_float(checker):
    """get_min_distance returns a Python float."""
    d = checker.get_min_distance(Q_HOME_LEFT, Q_HOME_RIGHT)
    assert isinstance(d, float)


def test_min_distance_finite(checker):
    """get_min_distance returns a finite value at home."""
    d = checker.get_min_distance(Q_HOME_LEFT, Q_HOME_RIGHT)
    assert np.isfinite(d), f"Min distance is non-finite: {d}"


def test_min_distance_in_collision_is_nonpositive(checker):
    """get_min_distance returns a non-positive value when in collision."""
    q_folded = np.array([0.0, math.pi * 0.9, -math.pi * 0.9, math.pi * 0.8, 0.0, 0.0])

    # Verify collision first
    if checker.is_collision_free(q_folded, Q_HOME_RIGHT):
        pytest.skip("Config is not in collision; skip min_distance penetration check")

    d = checker.get_min_distance(q_folded, Q_HOME_RIGHT)
    assert d <= 0.0, (
        f"Expected non-positive distance for colliding config, got {d:.4f} m"
    )


def test_min_distance_decreases_as_arm_approaches_table(checker):
    """Min distance decreases as the arm approaches the table."""
    # q at home is far from table
    d_home = checker.get_min_distance(Q_HOME_LEFT, Q_HOME_RIGHT)

    # Tilt arm toward table
    q_tilted = Q_HOME_LEFT.copy()
    q_tilted[1] = -math.pi / 2  # shoulder lift tilts upper arm down

    d_tilted = checker.get_min_distance(q_tilted, Q_HOME_RIGHT)

    assert d_tilted <= d_home, (
        f"Distance should not increase as arm tilts toward table: "
        f"d_home={d_home:.4f}, d_tilted={d_tilted:.4f}"
    )


# ---------------------------------------------------------------------------
# is_collision_free output type
# ---------------------------------------------------------------------------

def test_collision_free_returns_bool(checker):
    """is_collision_free returns a Python bool."""
    result = checker.is_collision_free(Q_HOME_LEFT, Q_HOME_RIGHT)
    assert isinstance(result, bool)


def test_is_path_free_returns_bool(checker):
    """is_path_free returns a Python bool."""
    result = checker.is_path_free(
        Q_HOME_LEFT, Q_HOME_LEFT,
        Q_HOME_RIGHT, Q_HOME_RIGHT,
    )
    assert isinstance(result, bool)
