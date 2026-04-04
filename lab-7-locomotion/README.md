# Lab 7: Locomotion Fundamentals

Bipedal locomotion on the **Unitree G1 humanoid** (29 DOF, 33.34 kg) using MuJoCo simulation and Pinocchio analytical computation.

## What This Lab Covers

- Floating-base robotics: nq != nv, quaternion conventions, Pinocchio FreeFlyer
- Gravity-compensated standing with position actuators
- CoM tracking, support polygon, push recovery
- Whole-body inverse kinematics (stacked Jacobian DLS)
- LIPM + ZMP preview control theory (Kajita 2003)
- Honest analysis of why classical ZMP walking fails with position actuators

## Milestone Summary

| Milestone | Description | Status |
|-----------|-------------|--------|
| M0 | Load G1, joint map, freefall demo | PASS |
| M1 | Standing balance + 5N push recovery | PASS (1.6mm deviation) |
| M2 | CoM cross-validation + support polygon | PASS (0.000mm error) |
| M3a | Pinocchio Jacobian validation | PASS (0/36 failures) |
| M3b | FK cross-validation (10 configs) | PASS (0.000mm error) |
| M3c | Whole-body IK (pure Pinocchio) | PASS (<0.51mm foot slip) |
| M3d | Weight shift in simulation | PASS (53.5mm shift, <1.4mm drift) |
| M3e | Dynamic walking attempt | FAILED (6 attempts) |
| M5 | Documentation + capstone demo | PASS |

## Quick Start

### Prerequisites

```bash
pip install mujoco numpy pinocchio scipy imageio[ffmpeg] matplotlib
```

The Unitree G1 model must be available at the path configured in `src/lab7_common.py` (default: `../../vla_zero_to_hero/third_party/mujoco_menagerie/unitree_g1/`).

### Run Demos

```bash
# M0: Model exploration + freefall
python3 src/m0_explore_g1.py

# M1: Standing balance + push recovery (10s)
python3 src/m1_standing.py

# M2: CoM tracking + support polygon
python3 src/m2_com_balance.py

# M3a: Jacobian validation (pure math)
python3 src/m3a_pinocchio_validation.py

# M3b: FK cross-validation (Pin vs MuJoCo)
python3 src/m3b_foot_fk_validation.py

# M3c: Whole-body IK validation (pure Pinocchio)
python3 src/m3c_static_ik.py

# M3d: Weight shift in simulation
python3 src/m3d_weight_shift.py

# M5: Capstone demo (standing + push + weight shift + LIPM plot)
python3 src/m5_capstone_demo.py
```

### Run Tests

```bash
pytest tests/
```

## Architecture

```
Pinocchio (analytical)          MuJoCo (simulation)
├─ FK, Jacobians                ├─ Physics stepping
├─ CoM computation              ├─ Contact dynamics
├─ Whole-body IK                ├─ Position actuators
└─ Frame placements             └─ Video rendering
```

**Key principle:** Pinocchio computes all kinematics analytically. MuJoCo runs the physics simulation. Cross-validation between the two confirms correctness.

Control law for all demos:
```
ctrl = q_target + qfrc_bias[6:35] / Kp - K_VEL * qvel[6:35] / Kp
```

## Key Files

| File | Purpose |
|------|---------|
| `src/lab7_common.py` | Constants, model loading, quaternion utilities |
| `src/m1_standing.py` | Gravity-compensated PD standing |
| `src/m3c_static_ik.py` | Whole-body IK solver (`whole_body_ik()`) |
| `src/m3d_weight_shift.py` | Pre-computed IK weight shift demo |
| `src/m5_capstone_demo.py` | Capstone: standing + push + shift + LIPM plot |
| `src/lipm_preview_control.py` | LIPM + ZMP preview controller (Kajita 2003) |
| `src/zmp_reference.py` | Footstep planner + ZMP reference generator |
| `src/swing_trajectory.py` | Swing foot trajectory (cubic + parabolic) |

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design, theory, failure analysis
- [Architecture (Turkish)](docs-turkish/ARCHITECTURE_TR.md) — Mimari (Turkce)
- [Code Walkthrough](docs/CODE_WALKTHROUGH.md) — Recommended reading order
- [Blog Post](blog/lab7_locomotion.md) — "Why Making a Humanoid Walk is Harder Than It Looks"
- [G1 Joint Map](docs/g1_joint_map.md) — Joint names, indices, ranges

## Media

| File | Content |
|------|---------|
| `media/m0_freefall.mp4` | G1 freefall (no control) |
| `media/m1_standing.mp4` | 10s standing + 5N push |
| `media/m2_com_balance.mp4` | CoM balance controller |
| `media/m3d_weight_shift.mp4` | 5cm lateral weight shift |
| `media/m5_capstone.mp4` | Capstone demo (all phases) |
| `media/m5_plots.png` | Trajectory plots + LIPM overlay |

## Connection to Prior Labs

| Lab | Pattern Reused |
|-----|---------------|
| Lab 3 | PD control, gravity compensation, Menagerie feedforward pattern |
| All | Cross-validation (Pinocchio vs MuJoCo) |

New in Lab 7: floating-base dynamics, quaternion manifold integration, whole-body IK. No code imported from Labs 1–6.

## What's Next

Lab 8 (Whole-Body Loco-Manipulation) will need to bridge the static-to-dynamic balance gap. Likely approaches: torque-controlled actuators with QP whole-body control, RL-trained walking policies, or hybrid model-based + learned methods.

## Lab Series

1. 2-Link Planar Arm — FK, Jacobian, IK, PD control
2. UR5e 6-DOF Arm — Scales to industrial arm with Pinocchio
3. Dynamics & Force Control — RNEA, gravity comp, impedance control
4. Motion Planning — RRT*, TOPP-RA, collision avoidance
5. Grasping & Manipulation — Custom gripper, pick-and-place
6. Dual-Arm Coordination
7. **Locomotion Fundamentals** (this lab)
8. Whole-Body Loco-Manipulation (planned)
9. VLA Integration (planned)
