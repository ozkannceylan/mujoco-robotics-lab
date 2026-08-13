# A1: Environment Setup

## Goal

Confirm that the UR5e assets load correctly, that the analytical and simulation models agree on forward kinematics, and that the lab dependencies are ready before moving into FK, Jacobians, IK, and control.

## Files

- Script: `src/a1_model_setup.py`
- Shared helpers: `src/ur5e_common.py`
- MuJoCo wrapper: `src/mujoco_sim.py`
- MuJoCo scene: `models/lab_scene.xml` (includes the Menagerie `ur5e.xml`)
- URDF: `models/ur5e.urdf`
- Output snapshot: `docs/a1_environment_status.csv`

## Dependency Boundary

This script is the right place to fail fast. It imports the core packages used by the lab:

- `numpy`
- `mujoco`
- `pinocchio`
- `matplotlib`

If one of those imports is missing, fix that first. Later modules build directly on the same stack.

## What the Script Verifies

The script executes five checks in order:

1. **Python dependency check**
   Verifies that the core Python modules import successfully and reports their versions.
2. **Model file check**
   Confirms that the Menagerie MJCF scene and the URDF are present under `models/`.
3. **MuJoCo check**
   Loads the scene, prints basic model metadata, enumerates the joints, sets `Q_HOME`, and steps the simulation.
4. **Pinocchio check**
   Loads the URDF, prints joint and frame metadata, and computes FK at the home configuration.
5. **Cross-validation check**
   Compares the end-effector position from MuJoCo (`attachment_site`) and Pinocchio (`ee_link`) across multiple configurations. The pass condition is agreement within 1 mm.

## Why This Module Matters

- `ur5e_common.py` becomes the shared contract for the rest of the lab: paths, joint names, limits, and reference configurations are centralized there.
- `mujoco_sim.py` gives the rest of the modules a stable interface instead of spreading raw MuJoCo array access everywhere.
- The FK comparison is the first strong correctness signal in the project. If MuJoCo and Pinocchio disagree here, everything after A1 becomes suspicious.

## How to Run

```bash
python3 src/a1_model_setup.py
```

## What to Study

1. Compare `ee_link` in the URDF with `attachment_site` in the MuJoCo scene. That mapping is reused throughout the lab.
2. Read `ur5e_common.py` and identify the constants that later modules depend on: `Q_HOME`, `Q_ZEROS`, joint limits, velocity limits, torque limits, and the DH table.
3. Open `mujoco_sim.py` and trace how `UR5eSimulator` exposes state, torques, and end-effector pose.
4. Treat the FK cross-check as a trust anchor. Once this passes, the rest of the lab can build on it with confidence.

## Next Step

Move to A2 to study the UR5e kinematic chain from the DH side and compare that educational model to the URDF and MJCF ground-truth models.
