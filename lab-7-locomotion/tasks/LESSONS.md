# Lab 7: Lessons Learned

## Bugs & Fixes

*(Log bugs here as they occur)*

## Debug Strategies

### Check MuJoCo vs Pinocchio quaternion convention
When FK values don't match between MuJoCo and Pinocchio:
- MuJoCo uses (w, x, y, z) for free joint quaternion (qpos[3:7])
- Pinocchio FreeFlyer uses (x, y, z, w) for the same (q[3:7])
- Always use `mj_qpos_to_pin()` and `pin_q_to_mj()` from lab7_common

### Check foot site vs foot body
G1 has `left_foot` and `right_foot` sites defined in the MJCF.
Use site positions for foot contact targets, not body positions.
In Pinocchio: getFrameId('left_foot') returns the frame ID, NOT a joint ID.

## Key Insights

### Floating base is fundamentally different
In fixed-base labs (1–6), joint 0 is the robot root. In Lab 7, the robot has a 7-DOF
unactuated freejoint at qpos[0:7] / qvel[0:6]. This means:
- You cannot directly command the pelvis position
- The pelvis moves as a result of contact forces and actuated joint motion
- IK must work in terms of "what joint angles make the feet land in the right place"

### LIPM z_c measurement
The CoM height z_c for LIPM should be measured from the settled standing pose, not
the keyframe pose. The keyframe puts feet 3.3cm above the ground, so after gravity
settling the CoM is approximately 0.033m lower than the keyframe value.
Keyframe CoM: 0.692m → settled CoM: ~0.659m → use z_c = 0.66m

### Pinocchio MJCF loading with FreeFlyer
buildModelFromMJCF with JointModelFreeFlyer() correctly creates nq=36, nv=35.
Joint velocity indices match MuJoCo actuator indices (offset by 6 for floating base):
- MuJoCo ctrl[0:6] ↔ Pinocchio v[6:12] (left leg)
- MuJoCo ctrl[6:12] ↔ Pinocchio v[12:18] (right leg)
