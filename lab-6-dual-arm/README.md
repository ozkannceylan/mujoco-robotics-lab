# Lab 6: Dual-Arm Coordination

Cooperative bimanual manipulation with two UR5e arms in MuJoCo. The arms grasp a box from opposite sides, lift it, carry it laterally, and place it back on the table.

## Overview

Two UR5e arms are placed 1 meter apart on a shared table. A 30x15x15 cm box sits between them. The system progresses through 5 milestones (M0-M4), each with explicit gate criteria:

| Milestone | Description | Key Result |
|-----------|-------------|------------|
| M0 | Scene validation | 12 joints, EE z-dot=1.0 |
| M1 | Independent PD control | Max error 0.000247 rad |
| M2 | FK/IK cross-validation | FK error 0.000 mm, IK 20/20 |
| M3 | Coordinated approach | 0.1 mm Cartesian error, 2 ms sync |
| M4 | Cooperative carry | Lift 15 cm, carry 22 cm, rot 3 deg |
| M5 | Capstone demo + docs | End-to-end video with overlay |

## Prerequisites

```bash
pip install mujoco numpy pinocchio scipy imageio[ffmpeg] matplotlib
```

Python 3.10+ required. MuJoCo Menagerie mesh assets must be accessible via the symlink in `models/assets/`.

## Project Structure

```
lab-6-dual-arm/
  models/
    scene_dual.xml       # Main MJCF scene (table, box, weld constraints)
    ur5e_left.xml        # Left arm (base at origin)
    ur5e_right.xml       # Right arm (base at x=1.0m)
    ur5e.urdf            # Pinocchio URDF (Lab 3 Menagerie-matching)
    assets/              # Symlink to Menagerie mesh files
  src/
    lab6_common.py       # Shared constants and utilities
    dual_arm_model.py    # Pinocchio FK, Jacobian, DLS IK
    joint_pd_controller.py  # Joint PD + gravity compensation
    grasp_pose_calculator.py  # Grasp pose from box state
    bimanual_state_machine.py  # 6-state cooperative pipeline
    m0_validate_scene.py    # M0: scene structure check
    m1_independent_motion.py  # M1: per-arm PD control
    m2_fk_validation.py     # M2: FK cross-validation
    m2_ik_validation.py     # M2: IK convergence + round-trip
    m2_ik_visual.py         # M2: visual IK verification
    m3_coordinated_approach.py  # M3: collision-free approach
    m4_cooperative_carry.py    # M4: full pick-and-place
    m5_capstone_demo.py        # M5: capstone with overlay
  docs/
    ARCHITECTURE.md      # Comprehensive architecture document
    CODE_WALKTHROUGH.md  # Step-by-step reading guide
  docs-turkish/
    ARCHITECTURE_TR.md   # Turkish translation
  blog/
    lab6_dual_arm.md     # Technical blog post
  media/                 # Generated videos and plots
  tasks/
    TODO.md              # Progress tracking
    LESSONS.md           # Bug journal (L1-L12)
```

## Running Milestone Scripts

Each milestone builds on the previous. Run in order:

```bash
cd lab-6-dual-arm

# M0: Validate scene structure
python3 src/m0_validate_scene.py

# M1: Independent joint PD control
python3 src/m1_independent_motion.py

# M2: FK/IK validation
python3 src/m2_fk_validation.py
python3 src/m2_ik_validation.py
python3 src/m2_ik_visual.py

# M3: Coordinated approach
python3 src/m3_coordinated_approach.py

# M4: Cooperative carry
python3 src/m4_cooperative_carry.py

# M5: Capstone demo (runs full pipeline)
python3 src/m5_capstone_demo.py
```

## Expected Results

### M3: Coordinated Approach
Both arms approach the box from opposite sides with collision-free trajectories. Arrival synchronization within 2 ms.

### M4: Cooperative Carry
Full 6-state pipeline: APPROACH -> CLOSE -> GRASP -> LIFT -> CARRY -> PLACE. Box lifted 15 cm, carried 22 cm in +y direction, placed back on table with <3 cm z-error and <10 deg rotation.

### M5: Capstone
End-to-end video with state overlay text. Summary trajectory plot showing box position through all states.

## Architecture

The system follows the project-wide pattern:

```
Pinocchio = analytical brain (FK, Jacobian, IK)
MuJoCo   = physics simulator (step, render, contact, weld constraints)
```

Key design decisions:
- **Joint PD control only** (no Cartesian impedance) for reliability across large reconfigurations
- **Collision-free IK search** (300 random starts with MuJoCo contact check) for safe dual-arm motion
- **Weld constraints for grasping** (instead of friction-based) for deterministic box attachment
- **Smooth-step trajectory interpolation** for natural-looking motion
- **Carry in +y direction** (not +x) due to workspace reachability limits with 1m arm spacing

See `docs/ARCHITECTURE.md` for the full technical deep-dive.

## Key Lessons Learned

1. Both arms must have identical base orientation. Facing direction is handled by IK targets, not base rotation.
2. The "standard" UR5e URDF does not match MuJoCo Menagerie. Lab 3's hand-tuned version is required.
3. Weld constraint `eq_data` must be updated to the current relative pose before activation.
4. IK solutions must be collision-checked against the full scene, not just kinematic limits.
5. Large joint reconfigurations need ramped interpolation, not step commands.

See `tasks/LESSONS.md` for all 12 lessons with symptom/root-cause/fix/takeaway format.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design, module breakdown, controller design, IK pipeline, state machine walkthrough
- [Code Walkthrough](docs/CODE_WALKTHROUGH.md) - Guided reading order with line references
- [Architecture (Turkish)](docs-turkish/ARCHITECTURE_TR.md) - Turkish translation
- [Blog Post](blog/lab6_dual_arm.md) - "From One Arm to Two" technical narrative

## Videos

Generated by the milestone scripts into `media/`:
- `m0_scene.png` - Scene screenshot
- `m1_independent.mp4` - Independent PD control
- `m3_approach.mp4` - Coordinated approach
- `m4_carry.mp4` - Full cooperative carry
- `m5_capstone.mp4` - Capstone with state overlay
- `m5_trajectory.png` - Box trajectory plot

## Part of MuJoCo Robotics Lab

This is Lab 6 in a 9-lab progression:
1. 2-Link Planar Arm
2. UR5e 6-DOF
3. Dynamics & Force Control
4. Motion Planning
5. Grasping & Manipulation
6. **Dual-Arm Coordination** (this lab)
7. Locomotion Fundamentals
8. Whole-Body Loco-Manipulation
9. VLA Integration
