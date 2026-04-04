#!/usr/bin/env python3
"""test4.py — M4 Walking Integration Test Scaffold.

Validates that all M3e building blocks can be composed into a full
ZMP walking pipeline: footstep plan → ZMP reference → LIPM preview
control → swing trajectories → whole-body IK → MuJoCo simulation.

Modules under test:
  - lab7_common:           G1 model loading (MuJoCo + Pinocchio)
  - m3c_static_ik:         Whole-body IK (stacked Jacobian DLS)
  - lipm_preview_control:  LIPM + preview control CoM trajectory
  - zmp_reference:         Footstep plan + ZMP reference generation
  - swing_trajectory:      Cubic/parabolic swing foot trajectories
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

import mujoco
import numpy as np
import pinocchio as pin
from pathlib import Path
from lab7_common import load_g1_mujoco, load_g1_pinocchio
from m3c_static_ik import whole_body_ik
from lipm_preview_control import generate_com_trajectory
from zmp_reference import generate_zmp_reference
from swing_trajectory import generate_swing_trajectory
print('imports ok')
