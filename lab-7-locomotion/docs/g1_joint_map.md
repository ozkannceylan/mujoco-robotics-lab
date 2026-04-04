# Unitree G1 — Joint Map & DOF Layout

Model: **Unitree G1 29-DOF** from MuJoCo Menagerie (`unitree_g1/scene.xml`)

---

## Model Summary

| Property | Value |
|----------|-------|
| nq (config dim) | 36 (7 freejoint + 29 hinge) |
| nv (velocity dim) | 35 (6 base + 29 hinge) |
| nu (actuators) | 29 (all position-controlled, Kp=500) |
| nbody | 31 |
| Total mass | 33.34 kg |
| Pelvis height (keyframe) | 0.79 m |
| CoM height (settled) | ~0.66 m |

---

## Joint Table

| idx | Name | Category | qpos_adr | Range (rad) |
|-----|------|----------|----------|-------------|
| 0 | floating_base_joint (free) | BASE | 0 | — |
| 1 | left_hip_pitch_joint | LEFT_LEG | 7 | [-2.531, 2.880] |
| 2 | left_hip_roll_joint | LEFT_LEG | 8 | [-0.524, 2.967] |
| 3 | left_hip_yaw_joint | LEFT_LEG | 9 | [-2.758, 2.758] |
| 4 | left_knee_joint | LEFT_LEG | 10 | [-0.087, 2.880] |
| 5 | left_ankle_pitch_joint | LEFT_LEG | 11 | [-0.873, 0.524] |
| 6 | left_ankle_roll_joint | LEFT_LEG | 12 | [-0.262, 0.262] |
| 7 | right_hip_pitch_joint | RIGHT_LEG | 13 | [-2.531, 2.880] |
| 8 | right_hip_roll_joint | RIGHT_LEG | 14 | [-2.967, 0.524] |
| 9 | right_hip_yaw_joint | RIGHT_LEG | 15 | [-2.758, 2.758] |
| 10 | right_knee_joint | RIGHT_LEG | 16 | [-0.087, 2.880] |
| 11 | right_ankle_pitch_joint | RIGHT_LEG | 17 | [-0.873, 0.524] |
| 12 | right_ankle_roll_joint | RIGHT_LEG | 18 | [-0.262, 0.262] |
| 13 | waist_yaw_joint | WAIST | 19 | [-2.618, 2.618] |
| 14 | waist_roll_joint | WAIST | 20 | [-0.520, 0.520] |
| 15 | waist_pitch_joint | WAIST | 21 | [-0.520, 0.520] |
| 16 | left_shoulder_pitch_joint | LEFT_ARM | 22 | [-3.089, 2.670] |
| 17 | left_shoulder_roll_joint | LEFT_ARM | 23 | [-1.588, 2.252] |
| 18 | left_shoulder_yaw_joint | LEFT_ARM | 24 | [-2.618, 2.618] |
| 19 | left_elbow_joint | LEFT_ARM | 25 | [-1.047, 2.094] |
| 20 | left_wrist_roll_joint | LEFT_ARM | 26 | [-1.972, 1.972] |
| 21 | left_wrist_pitch_joint | LEFT_ARM | 27 | [-1.614, 1.614] |
| 22 | left_wrist_yaw_joint | LEFT_ARM | 28 | [-1.614, 1.614] |
| 23 | right_shoulder_pitch_joint | RIGHT_ARM | 29 | [-3.089, 2.670] |
| 24 | right_shoulder_roll_joint | RIGHT_ARM | 30 | [-2.252, 1.588] |
| 25 | right_shoulder_yaw_joint | RIGHT_ARM | 31 | [-2.618, 2.618] |
| 26 | right_elbow_joint | RIGHT_ARM | 32 | [-1.047, 2.094] |
| 27 | right_wrist_roll_joint | RIGHT_ARM | 33 | [-1.972, 1.972] |
| 28 | right_wrist_pitch_joint | RIGHT_ARM | 34 | [-1.614, 1.614] |
| 29 | right_wrist_yaw_joint | RIGHT_ARM | 35 | [-1.614, 1.614] |

---

## DOF Summary

| Category | DOFs | Lab 7 Status |
|----------|------|-------------|
| BASE (freejoint) | 6 | unactuated |
| LEFT_LEG | 6 | **controlled** |
| RIGHT_LEG | 6 | **controlled** |
| WAIST | 3 | **controlled** |
| LEFT_ARM | 7 | locked at neutral |
| RIGHT_ARM | 7 | locked at neutral |
| **Total nv** | **35** | |
| **Actuated (nu)** | **29** | |
| **Locomotion (legs+waist)** | **15** | actively controlled |

---

## Actuator Table

All actuators are `position` type with `Kp=500` and `dampratio=1`.

| idx | Actuator | Category | ctrl range |
|-----|----------|----------|-----------|
| 0 | left_hip_pitch_joint | LEFT_LEG | [-2.531, 2.880] |
| 1 | left_hip_roll_joint | LEFT_LEG | [-0.524, 2.967] |
| 2 | left_hip_yaw_joint | LEFT_LEG | [-2.758, 2.758] |
| 3 | left_knee_joint | LEFT_LEG | [-0.087, 2.880] |
| 4 | left_ankle_pitch_joint | LEFT_LEG | [-0.873, 0.524] |
| 5 | left_ankle_roll_joint | LEFT_LEG | [-0.262, 0.262] |
| 6 | right_hip_pitch_joint | RIGHT_LEG | [-2.531, 2.880] |
| 7 | right_hip_roll_joint | RIGHT_LEG | [-2.967, 0.524] |
| 8 | right_hip_yaw_joint | RIGHT_LEG | [-2.758, 2.758] |
| 9 | right_knee_joint | RIGHT_LEG | [-0.087, 2.880] |
| 10 | right_ankle_pitch_joint | RIGHT_LEG | [-0.873, 0.524] |
| 11 | right_ankle_roll_joint | RIGHT_LEG | [-0.262, 0.262] |
| 12 | waist_yaw_joint | WAIST | [-2.618, 2.618] |
| 13 | waist_roll_joint | WAIST | [-0.520, 0.520] |
| 14 | waist_pitch_joint | WAIST | [-0.520, 0.520] |
| 15-21 | left arm (7 joints) | LEFT_ARM | (locked at neutral) |
| 22-28 | right arm (7 joints) | RIGHT_ARM | (locked at neutral) |

---

## Body Table

| idx | Name | Parent | Mass (kg) |
|-----|------|--------|-----------|
| 0 | world | — | 0.000 |
| 1 | pelvis | world | 3.813 |
| 2 | left_hip_pitch_link | pelvis | 1.350 |
| 3 | left_hip_roll_link | 2 | 1.520 |
| 4 | left_hip_yaw_link | 3 | 1.702 |
| 5 | left_knee_link | 4 | 1.932 |
| 6 | left_ankle_pitch_link | 5 | 0.074 |
| 7 | left_ankle_roll_link | 6 | 0.608 |
| 8-13 | right leg (mirror) | pelvis | 7.186 |
| 14 | waist_yaw_link | pelvis | 0.214 |
| 15 | waist_roll_link | 14 | 0.086 |
| 16 | torso_link | 15 | 7.818 |
| 17-23 | left arm (7 links) | torso | 3.519 |
| 24-30 | right arm (7 links) | torso | 3.519 |
| | **TOTAL** | | **33.341** |

---

## qpos Layout (nq=36)

```
Index   Contents
──────  ────────────────────────────────────────
0-2     base position (x, y, z)
3-6     base quaternion (w, x, y, z)  ← MuJoCo convention
7-12    left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
13-18   right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
19-21   waist: yaw, roll, pitch
22-28   left arm: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
29-35   right arm: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
```

## qvel Layout (nv=35)

```
Index   Contents
──────  ────────────────────────────────────────
0-2     base linear velocity (vx, vy, vz)
3-5     base angular velocity (wx, wy, wz)
6-11    left leg joint velocities
12-17   right leg joint velocities
18-20   waist joint velocities
21-27   left arm joint velocities
28-34   right arm joint velocities
```

## ctrl Layout (nu=29)

```
Index   Contents
──────  ────────────────────────────────────────
0-5     left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
6-11    right leg: (same pattern)
12-14   waist: yaw, roll, pitch
15-21   left arm: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
22-28   right arm: (same pattern)
```

---

## Arm Neutral Pose (from keyframe "stand")

Arms are held at their keyframe positions during locomotion:

| Joint | Left | Right |
|-------|------|-------|
| shoulder_pitch | 0.2 | 0.2 |
| shoulder_roll | 0.2 | -0.2 |
| shoulder_yaw | 0.0 | 0.0 |
| elbow | 1.28 | 1.28 |
| wrist_roll | 0.0 | 0.0 |
| wrist_pitch | 0.0 | 0.0 |
| wrist_yaw | 0.0 | 0.0 |

---

## Kinematic Chain

```
world
└── pelvis (freejoint, z=0.793m, 3.813 kg)
    ├── left_hip_pitch_link → left_hip_roll_link → left_hip_yaw_link
    │   → left_knee_link → left_ankle_pitch_link → left_ankle_roll_link (4 foot spheres)
    ├── right_hip_pitch_link → ... → right_ankle_roll_link (4 foot spheres)
    └── waist_yaw_link → waist_roll_link → torso_link (7.818 kg)
        ├── left_shoulder_pitch_link → ... → left_wrist_yaw_link → left_rubber_hand
        └── right_shoulder_pitch_link → ... → right_wrist_yaw_link → right_rubber_hand
```

---

## Foot Contact Geometry

Each foot has 4 sphere contact points (radius=0.005m, friction=0.6):
- 2 at rear: pos=(-0.05, ±0.025, -0.03)
- 2 at front: pos=(0.12, ±0.03, -0.03)

This gives a ~17cm toe-to-heel support base per foot.

---

## Key Differences from Labs 1-6

| Property | Labs 2-6 (UR5e) | Lab 7 (G1) |
|----------|----------------|------------|
| Base | Fixed (bolted) | Floating (freejoint) |
| Total DOF | 6 | 35 (6 base + 29 joints) |
| Actuated | 6 | 29 (15 for locomotion) |
| Mass | ~28 kg | 33.34 kg |
| Primary challenge | Reach targets | Don't fall |
| Gravity | Torque to compensate | Existential threat |
