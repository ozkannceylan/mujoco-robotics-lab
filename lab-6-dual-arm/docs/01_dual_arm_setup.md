# Dual-Arm Setup: Kinematics, Scene, and Collision Checking

## Overview

Lab 6 coordinates two UR5e arms operating in a shared workspace. This document
covers the physical setup: how each arm is modelled mathematically, how the
MuJoCo scene is constructed so the two arms coexist, how FK results are
cross-validated between Pinocchio and MuJoCo, and how arm-arm collision
detection is implemented using HPP-FCL.

---

## Dual-Arm Kinematics: Two Pinocchio Models with Base SE3 Transforms

Rather than building a single branching 12-DOF kinematic tree, each arm is
represented as an independent 6-DOF Pinocchio model. A base SE3 transform
positions each model within the shared world frame.

```
World frame
   │
   ├─ LEFT_BASE_SE3  (identity — left arm at origin)
   │       └─ UR5e kinematic chain (6 joints)
   │              └─ ee_link
   │
   └─ RIGHT_BASE_SE3 (translated +1.0 m in X, rotated 180° about Z)
           └─ UR5e kinematic chain (6 joints)
                  └─ ee_link
```

### Loading the models

Both arms are loaded from the **same URDF**. The base transforms are applied
programmatically, so no separate left/right URDF is needed for Pinocchio:

```python
# from dual_arm_model.py
self.model_left = pin.buildModelFromUrdf(str(path))
self.data_left  = self.model_left.createData()
self.ee_fid_left = self.model_left.getFrameId("ee_link")

self.model_right = pin.buildModelFromUrdf(str(path))
self.data_right  = self.model_right.createData()
self.ee_fid_right = self.model_right.getFrameId("ee_link")
```

### FK in the world frame

After Pinocchio solves FK in the arm's local frame, the result is promoted to
the world frame by multiplying with the base SE3:

```python
def fk_left(self, q: np.ndarray) -> pin.SE3:
    pin.forwardKinematics(self.model_left, self.data_left, q)
    pin.updateFramePlacements(self.model_left, self.data_left)
    local_pose = self.data_left.oMf[self.ee_fid_left]
    return self.base_left * local_pose      # world_T_ee = world_T_base * base_T_ee
```

The same pattern applies to `fk_right`, with `self.base_right` substituted.

### Jacobians in the world frame

Pinocchio's `getFrameJacobian` returns the Jacobian in `LOCAL_WORLD_ALIGNED`
frame (origin at the EE, axes aligned with the world). To express this fully
in the world frame the base rotation is applied to both the linear and angular
rows:

```python
J_local = pin.getFrameJacobian(
    self.model_left, self.data_left,
    self.ee_fid_left,
    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
)
R = self.base_left.rotation
J_world[:3, :] = R @ J_local[:3, :]   # linear part
J_world[3:, :] = R @ J_local[3:, :]   # angular part
```

### Gravity compensation with rotated base

When the right arm's base is rotated 180° about Z, the gravity vector in the
arm's local frame must be recomputed. Pinocchio's `computeGeneralizedGravity`
uses a gravity vector stored in `model.gravity.linear`; this is temporarily
overridden with the locally expressed value:

```python
g_world = np.array([0.0, 0.0, -9.81])
g_local = self.base_right.rotation.T @ g_world   # rotate world gravity into arm frame
self.model_right.gravity.linear[:] = g_local
pin.computeGeneralizedGravity(self.model_right, self.data_right, q)
```

---

## MuJoCo Scene Setup

The scene is defined in `models/scene_dual.xml`, which includes both arm
models via `<include>`. Each arm XML uses prefixed names for all joints,
bodies, actuators, and sensors so there are no name collisions.

### Naming convention

| Element | Left arm | Right arm |
|---------|----------|-----------|
| Bodies | `left_shoulder_link`, … | `right_shoulder_link`, … |
| Joints | `left_shoulder_pan_joint`, … | `right_shoulder_pan_joint`, … |
| Actuators | `left_shoulder_pan`, … | `right_shoulder_pan`, … |

This gives 12 actuators total. In Python, the two arms are addressed by index
slices defined in `lab6_common.py`:

```python
LEFT_JOINT_SLICE  = slice(0, 6)    # mj_data.qpos[LEFT_JOINT_SLICE]
RIGHT_JOINT_SLICE = slice(6, 12)   # mj_data.qpos[RIGHT_JOINT_SLICE]
LEFT_CTRL_SLICE   = slice(0, 6)    # mj_data.ctrl[LEFT_CTRL_SLICE]
RIGHT_CTRL_SLICE  = slice(6, 12)   # mj_data.ctrl[RIGHT_CTRL_SLICE]
```

### Scene contents

- Both UR5e arms mounted on a shared table centred between them.
- A free-body box object (~30 × 15 × 15 cm, ~2 kg) placed on the table surface.
- Two MuJoCo equality (weld) constraints — `left_grasp` and `right_grasp` —
  initially disabled. They are activated at grasp time to simulate a rigid grip.
- Camera positioned to view both arms and the workspace simultaneously.

---

## FK Cross-Validation Between Pinocchio and MuJoCo

For any joint configuration, the EE position reported by Pinocchio (with base
offset applied) must agree with the body position reported by MuJoCo to within
1 mm. The cross-validation pattern:

```python
# Pinocchio side
ee_pin = dual_model.fk_left(q_left).translation

# MuJoCo side — set qpos and forward-kinematics
mj_data.qpos[LEFT_JOINT_SLICE] = q_left
mujoco.mj_kinematics(mj_model, mj_data)          # joint-only FK, no physics step
body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ee_link")
ee_mj = mj_data.xpos[body_id]

assert np.allclose(ee_pin, ee_mj, atol=1e-3), \
    f"FK mismatch: pin={ee_pin}, mj={ee_mj}, diff={np.abs(ee_pin - ee_mj)}"
```

Common causes of failure and their fixes:

| Symptom | Root cause | Fix |
|---------|------------|-----|
| Constant offset in all poses | Base translation not matching | Check `LEFT_BASE_SE3` / `RIGHT_BASE_SE3` vs XML `<body pos="…">` |
| Rotation error only | Base orientation mismatch | Verify 180° yaw in both XML and `RIGHT_BASE_SE3` |
| Error grows with joint index | Wrong joint ordering | Print `model.names` and compare to MuJoCo joint order |

---

## Arm-Arm Collision Checking with HPP-FCL

`DualCollisionChecker` maintains two independent Pinocchio geometry models and
performs four categories of collision checks for each call to
`is_collision_free(q_left, q_right)`:

1. Left arm self-collision (adjacent-link pairs filtered)
2. Right arm self-collision (adjacent-link pairs filtered)
3. Left vs. right cross-arm collision
4. Each arm vs. the table (environment obstacle)

### Base transform application to geometry

After Pinocchio updates geometry placements in the arm-local frame, the base
transform is applied to move each robot geometry into the shared world frame:

```python
def _update_left(self, q_left: np.ndarray) -> None:
    pin.forwardKinematics(self.model_left, self.data_left, q_left)
    pin.updateGeometryPlacements(
        self.model_left, self.data_left,
        self.geom_model_left, self.geom_data_left,
    )
    # Promote all robot geometries to world frame
    for i in range(self._n_left_robot_geoms):
        self.geom_data_left.oMg[i] = (
            self._base_left * self.geom_data_left.oMg[i]
        )
```

Obstacle geometries (the table) are added directly at their world-frame
positions and are not shifted by any base transform.

### Cross-arm collision

Because the two geometry models are independent, cross-arm collision cannot be
registered as Pinocchio collision pairs. Instead, HPP-FCL is called directly
on all combinations of left and right robot geometries using the world-frame
placements computed above:

```python
def _check_cross_arm(self) -> bool:
    req = hppfcl.CollisionRequest()
    for i in range(self._n_left_robot_geoms):
        tf_i = _se3_to_hppfcl(self.geom_data_left.oMg[i])
        for j in range(self._n_right_robot_geoms):
            tf_j = _se3_to_hppfcl(self.geom_data_right.oMg[j])
            res = hppfcl.CollisionResult()
            hppfcl.collide(
                geom_model_left.geometryObjects[i].geometry, tf_i,
                geom_model_right.geometryObjects[j].geometry, tf_j,
                req, res,
            )
            if res.isCollision():
                return True
    return False
```

### Path collision checking

`is_path_free()` discretises a linearly interpolated dual-arm path in joint
space and checks each sample:

```python
def is_path_free(
    self,
    q_left_start, q_left_end,
    q_right_start, q_right_end,
    resolution: float = 0.05,   # max step size in joint-space L2 norm (rad)
) -> bool:
```

The step count is chosen so that the maximum joint-space step for either arm
does not exceed `resolution` radians, preventing tunnel-through between
geometry shapes.

### Performance numbers

| Check category | Typical pair count |
|----------------|-------------------|
| Left self + env | ~45 pairs |
| Right self + env | ~45 pairs |
| Cross-arm | n_left_geoms × n_right_geoms ≈ 100–200 pairs |

Checks short-circuit on the first collision found, so typical wall time is
well under 1 ms for collision-free configurations.
