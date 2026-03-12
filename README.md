# Sandbox Robotics

A hands-on robotics lab for (re)learning robotics fundamentals through MuJoCo simulation.

Everything is built from scratch on a simple **2-link planar robot arm** — forward kinematics, Jacobians, inverse kinematics, dynamics, trajectory generation, PD control, and computed torque control — culminating in a real integrated task: **drawing a precise Cartesian square**.

## Showcase: From Scratch to First Real Task

<video src="media/c1_draw_square.mp4" autoplay loop muted playsinline width="640"></video>

> **Computed Torque Control** draws a 10 cm square with **0.008 mm RMS tracking error**.
> Quintic trajectory, analytic IK, Jacobian-based velocity mapping, all running in MuJoCo.

Run `python3 src/c1_record_video.py` to regenerate the video.

## What's Inside

The project is structured as a progressive learning path:

### A — Foundations
| Module | Topic | Script |
|--------|-------|--------|
| A1 | MuJoCo setup & interactive demo | `src/a1_mujoco_setup.py` |
| A2 | Forward kinematics & workspace | `src/a2_forward_kinematics.py` |
| A3 | Analytic Jacobian & singularities | `src/a3_jacobian.py` |
| A4 | Inverse kinematics (analytic, pinv, DLS) | `src/a4_inverse_kinematics.py` |
| A5 | Dynamics basics (M, C, g from MuJoCo) | `src/a5_dynamics_basics.py` |

### B — Control & Trajectory
| Module | Topic | Script |
|--------|-------|--------|
| B1 | Cubic & quintic trajectory generation | `src/b1_trajectory_generation.py` |
| B2 | PD control & gravity compensation | `src/b2_pd_controller.py` |
| B3 | Full pipeline demos (pick-place, circle) | `src/b3_full_pipeline.py` |

### C — Integration
| Module | Topic | Script |
|--------|-------|--------|
| C1 | **Cartesian square drawing** — computed torque control | `src/c1_draw_square.py` |

## Quick Start

```bash
# Install dependencies
pip install mujoco numpy imageio[ffmpeg]

# Run the square drawing demo (opens MuJoCo viewer)
python3 src/c1_draw_square.py

# Record a video (headless)
python3 src/c1_record_video.py
```

## Project Structure

```
models/          MuJoCo XML robot models (position & torque control variants)
src/             Python scripts for each module
docs/            Learning notes and documentation for each topic
media/           Recorded demo videos
```

## Key Results

| Metric | Value |
|--------|-------|
| Square tracking RMS error | 0.008 mm |
| Square tracking max error | 0.013 mm |
| Max torque used | 0.076 Nm |
| Jacobian analytic vs numeric error | < 1e-10 |
| IK success rate (20 targets) | 100% |

## Documentation

Detailed notes for each module are in [`docs/`](docs/):

- [MuJoCo Basics](docs/a1_mujoco_temelleri.md)
- [Forward Kinematics](docs/a2_forward_kinematics.md)
- [Jacobian](docs/a3_jacobian.md)
- [Inverse Kinematics](docs/a4_inverse_kinematics.md)
- [Dynamics](docs/a5_dynamics.md)
- [Trajectory Generation](docs/b1_trajectory_generation.md)
- [PD Control](docs/b2_pd_controller.md)
- [Full Pipeline](docs/b3_full_pipeline.md)
- [Draw Square (C1)](docs/c1_draw_square.md)
