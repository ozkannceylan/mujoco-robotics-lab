# Lab 6 — Dual-Arm Coordination: Architecture

## Module Map

```
lab-6-dual-arm/
├── models/
│   ├── assets/              # Menagerie UR5e OBJ meshes (shared)
│   ├── scene_dual.xml       # Main MJCF: includes left + right arms, table, box
│   ├── ur5e_left.xml        # Left arm body hierarchy + motor actuators
│   ├── ur5e_right.xml       # Right arm body hierarchy + motor actuators
│   └── ur5e.urdf            # UR5e URDF for Pinocchio (per-arm)
├── src/
│   ├── lab6_common.py       # Constants, paths, model loaders, index slicing
│   ├── m0_validate_scene.py # M0: scene load + joint/actuator validation
│   └── ...                  # Further scripts added per milestone
├── tests/
├── docs/
├── docs-turkish/
├── media/
└── tasks/
```

## Scene Layout

```
    Left UR5e                         Right UR5e
    base (0,0,0)                      base (1.0,0,0)
    same orientation                  same orientation (no yaw)
          \                                 \
           \          Table                  \
            \       (0.5, 0, 0.17)            \
             \        [  box  ]                \
              ===================================
                         Floor
```
Both bases use identical Menagerie quat (no mount rotation).
Arms face each other via IK targets in M3, not base orientation.

## Key Design Decisions

1. **Motor actuators (torque control)** — direct torque commands, not position servos. Gravity compensation computed via Pinocchio RNEA.
2. **Two separate Pinocchio models** — one per arm, each with a base SE3 transform. Simpler than one composite model.
3. **Object-centric frame** — grasp targets derived from box pose, not hardcoded per-arm.
4. **Index slicing** — left arm: qpos[0:6], ctrl[0:6]. Right arm: qpos[6:12], ctrl[6:12]. Box freejoint: qpos[12:19].

## MuJoCo ↔ Pinocchio Mapping

| MuJoCo Name | Pinocchio Frame | Notes |
|---|---|---|
| left_ee_site | ee_link (left model) | Apply LEFT_BASE_SE3 to get world frame |
| right_ee_site | ee_link (right model) | Apply RIGHT_BASE_SE3 to get world frame |

## Cross-Lab Dependencies

- URDF from Lab 3: `lab-3-dynamics-force-control/models/universal_robots_ur5e/ur5e.urdf`
- Local copy at `models/ur5e.urdf`
- Pinocchio model loading pattern from `lab3_common.py`
