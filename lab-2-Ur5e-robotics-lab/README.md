# Lab 2: UR5e Industrial Robot Arm

A self-contained 6-DOF robotics lab built around the **Universal Robots UR5e**. Lab 2 keeps the same end-to-end philosophy as Lab 1, but the math and software stack are now closer to what you would actually use on a full industrial manipulator: **Pinocchio** for analytical robotics and **MuJoCo** for simulation, rendering, and actuator behavior.

## Showcase

![Cube Drawing Demo](media/c3_draw_cube.gif)

> The final demo draws a 10 cm 3D cube with Damped Least Squares IK, quintic joint trajectories, and MuJoCo position servos augmented by gravity compensation and velocity feedforward.

## Key Results

| Metric | Value |
|---|---|
| Cube tracking RMS error | 0.088 mm |
| Cube tracking max error | 0.234 mm |
| Max actuator torque | 16.50 Nm |
| FK cross-validation (Pinocchio vs MuJoCo) | 0.000 mm |
| IK waypoint accuracy | < 0.1 mm |
| Tests shipped with the lab | 34 unittest checks across 5 files |

---

## Skills Demonstrated

- **FK cross-validation**: DH, Pinocchio, and MuJoCo agree to 0.000 mm.
- **Jacobian analysis**: geometric, Pinocchio, and numerical Jacobians with singularity-index sweeps.
- **Numerical IK**: pseudo-inverse and adaptive damped least-squares with sub-0.1 mm waypoint error.
- **Dynamics**: Pinocchio RNEA, ABA, and CRBA cross-validated against MuJoCo `qfrc_bias`.
- **Trajectory generation**: multi-segment cubic, quintic, trapezoidal, and minimum-jerk profiles.
- **Control suite**: PD+g, computed torque, task-space impedance, and operational space control (OSC).
- **Integration demos**: pick-and-place pipeline and 3D cube drawing at 0.088 mm RMS.

---

## Architecture

```text
Pinocchio = analytical brain (FK, Jacobian, M, C, g, IK)
MuJoCo    = simulator (step, render, actuator behavior, contacts)
```

The lab intentionally keeps those roles separate. Analytical quantities come from Pinocchio, while MuJoCo executes the motion and exposes the control reality of the Menagerie UR5e model.

---

## Modules

### A — Foundations

| Module | Topic | Script |
|---|---|---|
| A1 | Environment setup and FK cross-validation | `src/a1_model_setup.py` |
| A2 | Forward kinematics and UR5e DH parameters | `src/a2_forward_kinematics.py` |
| A3 | Jacobian derivation, numerical checks, singularities | `src/a3_jacobian.py` |
| A4 | Numerical IK: pseudo-inverse and DLS | `src/a4_inverse_kinematics.py` |
| A5 | Rigid-body dynamics with CRBA, RNEA, ABA | `src/a5_dynamics.py` |

### B — Motion and Control

| Module | Topic | Script |
|---|---|---|
| B1 | Joint-space and Cartesian trajectory generation | `src/b1_trajectory_generation.py` |
| B2 | PD+g, computed torque, impedance, OSC | `src/b2_control_hierarchy.py` |
| B3 | Joint limits, velocity scaling, torque saturation, self-collision heuristics | `src/b3_constraints.py` |

### C — Integration

| Module | Topic | Script |
|---|---|---|
| C1 | Pick-and-place and circle-tracking pipeline | `src/c1_pick_and_place.py` |
| C2 | ROS 2 bridge scaffold | `src/c2_ros2_bridge.py` |
| C3 | Final cube drawing demo and video export | `src/c3_draw_cube.py`, `src/c3_record_video.py` |

### Shared Utilities

| File | Purpose |
|---|---|
| `src/ur5e_common.py` | Paths, constants, DH table, loaders, shared limits |
| `src/mujoco_sim.py` | `UR5eSimulator` wrapper around MuJoCo |

---

## Quick Start

```bash
# Enter the lab folder
cd lab-2-Ur5e-robotics-lab

# Install the Python-side dependencies used throughout the lab
pip install numpy mujoco matplotlib "imageio[ffmpeg]" scipy

# Install Pinocchio with the package method that matches your platform/toolchain

# Run the environment check
python3 src/a1_model_setup.py

# Run the final interactive demo
python3 src/c3_draw_cube.py

# Record the demo video headlessly
python3 src/c3_record_video.py

# Run the tests
python3 -m unittest discover -s tests -p 'test_*.py'
```

Most modules require both `mujoco` and `pinocchio`. `c2_ros2_bridge.py` additionally needs a sourced ROS 2 environment for full bridge mode.

---

## How To Study This Lab

1. Start with the ordered notes in `docs/` or `docs-turkish/`.
2. Open the matching script in `src/`.
3. Inspect the generated CSV artifacts in the docs folders.
4. Read the longer-form narrative in `blog/`.
5. Re-run the script after the dependencies are installed.

That order keeps the math note, implementation, and results side by side.

---

## Structure

```text
lab-2-Ur5e-robotics-lab/
├── src/              Source scripts (A1–C3) and shared utilities
├── models/           URDF, MJCF scene, Menagerie UR5e assets
├── docs/             English study notes
├── docs-turkish/     Turkish study notes
├── blog/             Long-form blog-style writeups
├── media/            GIFs and recorded videos
├── tasks/            Lab planning and status notes
└── tests/            Unittest suite
```

The ROS 2 integration for this lab is a single script, [`src/c2_ros2_bridge.py`](src/c2_ros2_bridge.py) — there is no separate `ros2_bridge/` package.

---

## Documentation

| Topic | English | Turkish |
|---|---|---|
| Study index | [docs/README.md](docs/README.md) | [docs-turkish/README.md](docs-turkish/README.md) |
| A1 | [Environment Setup](docs/a1_environment_setup.md) | [Ortam Kurulumu](docs-turkish/a1_environment_setup.md) |
| A2 | [Forward Kinematics](docs/a2_forward_kinematics.md) | [Ileri Kinematik](docs-turkish/a2_forward_kinematics.md) |
| A3 | [Jacobian](docs/a3_jacobian.md) | [Jacobian](docs-turkish/a3_jacobian.md) |
| A4 | [Inverse Kinematics](docs/a4_inverse_kinematics.md) | [Ters Kinematik](docs-turkish/a4_inverse_kinematics.md) |
| A5 | [Dynamics](docs/a5_dynamics.md) | [Dinamik](docs-turkish/a5_dynamics.md) |
| B1 | [Trajectory Generation](docs/b1_trajectory_generation.md) | [Yorunge Uretimi](docs-turkish/b1_trajectory_generation.md) |
| B2 | [Control Hierarchy](docs/b2_control_hierarchy.md) | [Kontrol Hiyerarsisi](docs-turkish/b2_control_hierarchy.md) |
| B3 | [Constraints](docs/b3_constraints.md) | [Kisitlamalar](docs-turkish/b3_constraints.md) |
| C1 | [Integrated Pipeline](docs/c1_full_pipeline.md) | [Entegre Pipeline](docs-turkish/c1_full_pipeline.md) |
| C2 | [ROS 2 Bridge](docs/c2_ros2_bridge.md) | [ROS 2 Koprusu](docs-turkish/c2_ros2_bridge.md) |
| C3 | [Draw Cube](docs/c3_draw_cube.md) | [Kup Cizimi](docs-turkish/c3_draw_cube.md) |
| D1 | [VLA Bridge Note](docs/d1_vla_bridge.md) | [VLA Koprusu Notu](docs-turkish/d1_vla_bridge.md) |
| Extras | [Interview Cheatsheet](docs/interview_cheatsheet.md), [Portfolio Package](docs/portfolio_package.md) | [Interview Cheatsheet](docs-turkish/interview_cheatsheet.md), [Portfolio Package](docs-turkish/portfolio_package.md) |

---

## Blog Series

The long-form writeups live in [`blog/`](blog/README.md):

| Part | Focus | Post |
|---|---|---|
| 1 | Model loading, Pinocchio vs MuJoCo, validation strategy | [Part 1](blog/2026-03-14-ur5e-robotics-lab-part-1.md) |
| 2 | FK, DH frames, Jacobians, singularities | [Part 2](blog/2026-03-14-ur5e-robotics-lab-part-2.md) |
| 3 | Numerical IK and the role of damping | [Part 3](blog/2026-03-14-ur5e-robotics-lab-part-3.md) |
| 4 | Dynamics, trajectory generation, and controller design | [Part 4](blog/2026-03-14-ur5e-robotics-lab-part-4.md) |
| 5 | Full pipeline integration, ROS 2 bridge, and the cube demo | [Part 5](blog/2026-03-14-ur5e-robotics-lab-part-5.md) |

---

## Notes

- This lab is no longer dependency-light. The source tree expects `numpy`, and most modules also expect `mujoco` and `pinocchio`.
- In the current workspace, documentation and links were verified statically, but the full unittest suite was not runnable because `numpy` is not installed.
- Historical CSV snapshots are versioned in `docs/` and `docs-turkish/`. Re-running the scripts may regenerate them with slightly different numeric values depending on your environment and library versions.

---

## License

The Lab 2 source code and original documentation are covered by the repository
root [Apache-2.0 license](../LICENSE).

Bundled robot description packages and model assets in [`models/`](models/) keep
their upstream licenses. See the repository root
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the exact carve-outs,
including Universal Robots mesh directories that are not fully OSI-open-source.
