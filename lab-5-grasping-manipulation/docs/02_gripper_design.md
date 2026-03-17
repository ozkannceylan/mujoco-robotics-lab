# Parallel-Jaw Gripper Design — Lab 5

## Overview

Lab 5 uses a custom parallel-jaw gripper built as MJCF XML and attached to the UR5e's `tool0` flange. The gripper has two sliding fingers controlled by a single position actuator, with an equality constraint that mirrors finger motion symmetrically.

---

## Physical Structure

```
tool0 (wrist flange)
  └── gripper_base  [pos="0 0 0.020"]
        ├── gripper_adapter  (mounting plate, 70×70×40 mm)
        ├── left_finger      [pos="0 +0.015 0.060"]
        │     ├── left_finger_geom  (box 26×16×50 mm)
        │     └── left_pad         (friction pad, 20×10×16 mm, inner face)
        └── right_finger     [pos="0 -0.015 0.060"]
              ├── right_finger_geom
              └── right_pad
```

Key dimensions at `GRIPPER_CLOSED` (joint position = 0):
| Element | Position from center (Y axis) |
|---------|-------------------------------|
| Left pad inner face | +0.019 m |
| Right pad inner face | -0.019 m |
| **Internal gap** | **38 mm** |

At `GRIPPER_OPEN` (joint position = 0.030 m):
| Element | Position from center (Y axis) |
|---------|-------------------------------|
| Left pad inner face | +0.049 m |
| Right pad inner face | -0.049 m |
| **Internal gap** | **98 mm** |

---

## Joint Mechanics

Both fingers are `slide` joints moving along the gripper Y axis.

```xml
<joint name="left_finger_joint" type="slide"
       axis="0 1 0" range="0 0.030"
       damping="2.0" armature="0.0001"/>
```

- `range="0 0.030"`: fingers travel 30 mm in each direction (total 60 mm max opening change)
- `damping="2.0"`: viscous damping, keeps fingers stable during fast motions

### Equality Constraint (Mirror)

A single equality constraint mirrors the right finger to the left:

```xml
<equality>
  <joint name="finger_mirror"
         joint1="left_finger_joint"
         joint2="right_finger_joint"
         polycoef="0 1 0 0 0"/>
</equality>
```

`polycoef="0 1 0 0 0"` means `q_right = 1 × q_left` — both fingers open/close by the same amount simultaneously. This reduces the gripper from 2 actuated DOF to 1.

---

## Actuation

One position actuator controls the gripper:

```xml
<position name="gripper" joint="left_finger_joint"
          kp="200" ctrllimited="true" ctrlrange="0 0.030"/>
```

- **kp=200**: spring constant (N/m). Provides ~4 N closing force on a 20 mm object.
- **ctrllimited**: prevents commanding outside physical range.
- `ctrl[6] = 0.000` → **CLOSED** (grip)
- `ctrl[6] = 0.030` → **OPEN** (release)

---

## Coordinate Conventions

At Q_HOME (`[-π/2, -π/2, π/2, -π/2, -π/2, 0]`):
- `tool0` Z axis = world **-Z** (arm pointing down)
- `tool0` Y axis ≈ world **Y** (fingers slide in world Y)
- `gripper_site` (fingertip center) at tool0 local Z = 0.090 m below tool0 origin

The `gripper_site` site is used as the IK target reference point. The IK adds `GRIPPER_TIP_OFFSET = 0.090 m` to the box position to compute the tool0 target:

```
tool0_target = box_pos + [0, 0, GRIPPER_TIP_OFFSET]
```

---

## Friction Pad Design

The finger pad geometry is separate from the structural finger body, allowing independent tuning of contact properties:

```xml
<geom name="left_pad" type="box"
      size="0.010 0.005 0.008"
      pos="0 0.009 0.025"
      friction="1.5 0.005 0.0001"
      solimp="0.99 0.99 0.001"
      solref="0.002 1"
      condim="4"
      rgba="0.05 0.5 0.1 1"
      mass="0.005"/>
```

The structural finger body (`left_finger_geom`) uses default contact properties, while only the pad has high friction and stiff contact parameters. This models the rubber pad on a real gripper.

---

## Gripper Controller API

See `src/gripper_controller.py` for the full implementation.

| Function | Description |
|----------|-------------|
| `open_gripper(mj_data)` | Sets `ctrl[6] = 0.030` |
| `close_gripper(mj_data)` | Sets `ctrl[6] = 0.000` |
| `step_until_settled(m, d, max_steps)` | Steps until finger velocity < threshold |
| `get_finger_position(m, d)` | Returns left_finger_joint position (m) |
| `is_gripper_settled(m, d)` | Returns True if finger velocity < 5×10⁻⁴ m/s |
| `is_gripper_in_contact(m, d)` | Returns True if any finger geom touches an object |
