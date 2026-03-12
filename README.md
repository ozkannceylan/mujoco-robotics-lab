# mujoco-kinematics-lab

A hands-on robotics lab for rebuilding robotics fundamentals in simulation with **MuJoCo**.

This repository starts from first principles on a simple **2-link planar robot arm** and builds up, step by step, to a complete control pipeline: **forward kinematics, Jacobians, inverse kinematics, dynamics, trajectory generation, PD control, and computed torque control**.

The final integrated task is a precise **Cartesian square drawing** demo.

## Showcase

![Square Drawing Demo](media/c1_draw_square.gif)

> The final demo draws a **10 cm Cartesian square** in MuJoCo using **computed torque control**, **quintic trajectory generation**, **analytic inverse kinematics**, and **Jacobian-based velocity mapping**.

To regenerate the video:

```bash
python3 src/c1_record_video.py
```

---

## Why this repo exists

This project was built as a compact robotics fundamentals lab:

* to revisit core manipulator concepts from scratch,
* to connect theory directly to simulation,
* and to turn isolated topics into a single working control stack.

Instead of jumping straight into complex robots, everything is developed on a minimal system where the math stays visible.

---

## What is implemented

### A — Foundations

| Module | Topic                                              | Script                         |
| ------ | -------------------------------------------------- | ------------------------------ |
| A1     | MuJoCo setup and interactive demo                  | `src/a1_mujoco_setup.py`       |
| A2     | Forward kinematics and workspace analysis          | `src/a2_forward_kinematics.py` |
| A3     | Analytic Jacobian and singularity analysis         | `src/a3_jacobian.py`           |
| A4     | Inverse kinematics (analytic, pseudo-inverse, DLS) | `src/a4_inverse_kinematics.py` |
| A5     | Dynamics basics (`M`, `C`, `g` from MuJoCo)        | `src/a5_dynamics_basics.py`    |

### B — Control and Trajectory

| Module | Topic                                                 | Script                            |
| ------ | ----------------------------------------------------- | --------------------------------- |
| B1     | Cubic and quintic trajectory generation               | `src/b1_trajectory_generation.py` |
| B2     | PD control and gravity compensation                   | `src/b2_pd_controller.py`         |
| B3     | Full pipeline demos (pick-and-place, circle tracking) | `src/b3_full_pipeline.py`         |

### C — Integration

| Module | Topic                                                 | Script                  |
| ------ | ----------------------------------------------------- | ----------------------- |
| C1     | Cartesian square drawing with computed torque control | `src/c1_draw_square.py` |

---

## Quick Start

### 1. Install dependencies

```bash
pip install mujoco numpy imageio[ffmpeg]
```

### 2. Run the final demo

```bash
python3 src/c1_draw_square.py
```

This opens the MuJoCo viewer and runs the square drawing task.

### 3. Record the demo video

```bash
python3 src/c1_record_video.py
```

This generates the showcase video headlessly.

---

## Project Structure

```text
models/   MuJoCo XML robot models
src/      Python scripts for each module and demo
docs/     Notes and explanations for each topic
media/    Recorded videos and visual outputs
```

---

## Key Results

| Metric                              | Value    |
| ----------------------------------- | -------- |
| Square tracking RMS error           | 0.008 mm |
| Square tracking max error           | 0.013 mm |
| Max torque used                     | 0.076 Nm |
| Analytic vs. numeric Jacobian error | < 1e-10  |
| IK success rate (20 random targets) | 100%     |

---

## Documentation

Detailed notes for each module are available in [`docs/`](docs/):

* [MuJoCo Basics](docs/a1_mujoco_basics.md)
* [Forward Kinematics](docs/a2_forward_kinematics.md)
* [Jacobian](docs/a3_jacobian.md)
* [Inverse Kinematics](docs/a4_inverse_kinematics.md)
* [Dynamics](docs/a5_dynamics.md)
* [Trajectory Generation](docs/b1_trajectory_generation.md)
* [PD Control](docs/b2_pd_controller.md)
* [Full Pipeline](docs/b3_full_pipeline.md)
* [Draw Square (C1)](docs/c1_draw_square.md)

---

## Core Takeaways

This repository is intentionally small in scope, but complete in structure.

It shows how to go from:

* robot model definition,
* to kinematics,
* to inverse kinematics,
* to trajectory generation,
* to feedback control,
* to a final task-space execution demo.

The same reasoning pattern scales to larger manipulators and humanoid control pipelines.

---

## Possible Next Steps

A natural continuation of this project would be:

* adding a 3-link or 6-DoF manipulator,
* comparing Jacobian transpose vs. operational-space control,
* adding contact-rich tasks,
* or bridging the simulation to ROS 2.
