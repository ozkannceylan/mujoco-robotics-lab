# 02: Walking + Manipulation

## Goal

Integrate the gait generator from Lab 7 with the whole-body QP controller to achieve walking while simultaneously controlling the arms. This is the transition from static whole-body control (standing + reaching) to dynamic whole-body control (locomotion + manipulation).

## Files

- Gait generator: `src/gait_generator.py`
- Walking with fixed arms demo: `src/a2_walk_fixed_arms.py`
- Walking while reaching demo: `src/a3_walk_and_reach.py`
- QP controller: `src/whole_body_qp.py`
- Contact model: `src/contact_model.py`

## Gait Generator Integration

The gait generator from Lab 7 produces three outputs at each timestep:
1. **Desired CoM trajectory** — where the center of mass should be, derived from the Linear Inverted Pendulum Model (LIPM)
2. **Desired foot trajectories** — swing foot position and orientation over time
3. **Contact schedule** — which feet are in contact with the ground at each instant

These outputs feed directly into the QP controller's task and constraint definitions:

```
GaitGenerator.step(t)
    |
    +---> com_desired     ---> CoMTask.set_target(com_desired)
    +---> foot_desired_L  ---> FootPoseTask_L.set_target(foot_desired_L)
    +---> foot_desired_R  ---> FootPoseTask_R.set_target(foot_desired_R)
    +---> contact_states  ---> WholeBodyQP.set_contacts(active_contacts)
```

The contact schedule is critical for the QP. During double support, both feet contribute contact forces and friction cone constraints. During single support, only the stance foot contributes. The swing foot has no contact constraint — its motion is governed entirely by the FootPoseTask.

### Gait Timing

The gait generator uses a fixed-period stepping pattern:

| Parameter | Value |
|-----------|-------|
| Step period | 0.6 s |
| Double support ratio | 0.2 (120 ms) |
| Single support ratio | 0.8 (480 ms) |
| Step length | 0.15 m |
| Step height | 0.04 m |
| CoM height | 0.65 m (LIPM assumption) |

The foot swing trajectory follows a cubic spline in the XY plane and a half-sine in the Z direction, producing a smooth lift-swing-land motion. Peak foot height occurs at mid-swing.

### Contact Schedule

```
Time:   |  DSP  |    SSP_L    |  DSP  |    SSP_R    |  DSP  |
Left:   | STANCE|   STANCE    |STANCE |    SWING    |STANCE |
Right:  | STANCE|    SWING    |STANCE |   STANCE    |STANCE |
```

DSP = double support phase. SSP_L = single support on left foot (right foot swings). SSP_R = single support on right foot (left foot swings).

During transitions between DSP and SSP, the contact set changes. The QP constraint matrix must be rebuilt at these transition points to add or remove the swing foot's contact force variables and friction cone constraints.

## Task Activation and Deactivation

Not all tasks are active at all times during a gait cycle. The task activation pattern depends on the gait phase:

### Double Support Phase

All tasks are active:
- CoMTask (w=1000): track LIPM-generated CoM reference
- FootPoseTask left (w=100): hold current stance position
- FootPoseTask right (w=100): hold current stance position
- HandPoseTask left (w=10): track hand target
- HandPoseTask right (w=10): track hand target
- PostureTask (w=1): regularize toward nominal

### Single Support Phase (right foot swing)

The right foot task switches from stance-hold to swing-track:
- CoMTask (w=1000): track LIPM CoM reference (shifted over left foot)
- FootPoseTask left (w=100): hold stance position (this foot bears all weight)
- FootPoseTask right (w=100): track swing trajectory from gait generator
- HandPoseTask left (w=10): track hand target
- HandPoseTask right (w=10): track hand target
- PostureTask (w=1): regularize

The foot weight remains the same (w=100) during swing, but the contact constraint is removed. This means the swing foot is free to move without generating ground reaction forces. The QP naturally produces the swing motion because the foot task demands it and there is no contact constraint to prevent it.

### Transition Handling

At DSP-to-SSP transitions, two things change simultaneously:
1. The swing foot's contact is removed from the QP constraint set
2. The swing foot's target switches from the current stance pose to the swing trajectory's initial pose

If these are not synchronized, the QP may produce a brief infeasibility: the contact constraint requires zero foot velocity while the task demands lift-off. The implementation handles this by removing the contact constraint one timestep before changing the foot target.

## Walking with Fixed Arm Pose

The `a2_walk_fixed_arms.py` demo demonstrates the simplest form of walking manipulation: the arms hold a fixed Cartesian pose while the legs execute a 6-step forward walk.

### Task Stack

```
Priority 1 (w=1000):  CoMTask        — follow LIPM CoM trajectory
Priority 2 (w=100):   FootPoseTask   — follow gait generator foot trajectories
Priority 3 (w=10):    HandPoseTask   — hold fixed target (both arms)
Priority 4 (w=1):     PostureTask    — regularize toward standing configuration
```

### What Happens

When the robot walks, the torso sways laterally to shift the CoM over each stance foot. This lateral sway propagates to the arms through the kinematic chain — if no hand task is active, the arms swing passively with the torso motion.

With the HandPoseTask active at w=10, the QP explicitly counteracts this propagation. The arm joints adjust to keep the hands at their target positions despite the torso sway. This requires the arm joints to move in opposition to the torso motion.

The priority hierarchy is visible in the results: CoM tracking is nearly perfect (the robot does not fall), foot tracking is good (steps land within 1 cm of targets), and hand tracking has modest drift (up to 3-5 cm during aggressive weight shifts). The hand drift occurs because the arm has limited range of motion and the torso sway during single support is large enough that the arms cannot fully compensate.

### Arm Drift Analysis

Arm drift during walking comes from three sources:

1. **Torso sway coupling**: lateral CoM shifts tilt the torso, moving the shoulder origin. The arm must compensate but has limited range.

2. **Priority compromise**: when CoM and foot tasks demand conflicting accelerations from shared joints (waist, hip), the hand task is sacrificed first because of its lower weight.

3. **Dynamic effects**: the Coriolis and centrifugal terms from walking create forces on the arm joints that the QP must counteract. At lower hand task weights, these forces cause tracking drift.

Increasing the hand weight from 10 to 50 reduces drift but can destabilize walking because the arms fight harder against the natural CoM compensation. The 10:1 ratio (hand:posture) is a reasonable compromise.

### Results

For a 6-step walk covering 0.9 m:
- **Walk stability**: robot completes all 6 steps without falling
- **CoM tracking RMS**: < 1.5 cm
- **Foot placement error**: < 1 cm per step
- **Arm pose drift (left)**: max 3.2 cm, RMS 1.8 cm
- **Arm pose drift (right)**: max 2.9 cm, RMS 1.6 cm
- **QP solve time**: 1.0 - 2.5 ms (increases during SSP due to asymmetric contacts)

## Walking While Reaching

The `a3_walk_and_reach.py` demo adds a time-varying hand target: the robot walks forward 0.8 m (5 steps), and during the last 2 steps, the left hand transitions from its fixed home pose to a reaching target 20 cm forward and 10 cm left of the shoulder.

### Smooth Target Transition

The hand target is interpolated using a 5th-order (quintic) polynomial:

```python
def smooth_transition(t, t_start, duration, pose_start, pose_end):
    s = (t - t_start) / duration
    s = np.clip(s, 0.0, 1.0)
    # Quintic: 6s^5 - 15s^4 + 10s^3
    alpha = 6*s**5 - 15*s**4 + 10*s**3
    pos = (1 - alpha) * pose_start.translation + alpha * pose_end.translation
    rot = pin.exp3(alpha * pin.log3(pose_start.rotation.T @ pose_end.rotation))
    return pin.SE3(pose_start.rotation @ rot, pos)
```

The quintic polynomial ensures zero velocity and zero acceleration at both endpoints (`s=0` and `s=1`). This prevents discontinuities in the hand task's desired acceleration, which would cause torque spikes in the QP solution.

### CoM Compensation During Reaching

As the left hand extends forward, the arm's center of mass shifts forward and to the left. This displaces the robot's total CoM, potentially pushing it outside the support polygon. The CoM task compensates by slightly shifting the desired CoM in the opposite direction:

```python
# Adjust CoM target to account for arm extension
com_offset = -0.3 * (hand_target.translation - hand_home.translation)
com_offset[2] = 0.0  # no vertical adjustment
com_desired += com_offset[:2]  # only XY adjustment
```

The scale factor (0.3) is empirical — it approximates the arm's mass fraction (~4 kg out of ~45 kg total) multiplied by the fraction of the CoM shift that the arm motion produces. A more precise approach uses `compute_com_with_load()`, but the simple offset is sufficient for reaching without carrying.

### Results

- **Walk stability**: 5 steps completed without falling
- **Hand reaching error**: < 4 cm from final target (limited by simultaneous walking)
- **CoM stability**: stays within support polygon throughout
- **Transition smoothness**: no torque spikes during target handoff

### Comparison: Fixed vs. Reaching

| Metric | Fixed Arms | Reaching |
|--------|-----------|----------|
| Walk stability | 6/6 steps | 5/5 steps |
| Max arm drift | 3.2 cm | 4.8 cm (reaching arm) |
| CoM RMS error | 1.5 cm | 2.1 cm |
| Mean QP solve time | 1.4 ms | 1.6 ms |

The reaching case shows slightly worse CoM tracking because the arm extension creates a larger disturbance. The QP resolve time increases marginally because the hand task error is larger, requiring more iterations to converge.

## Stability Metrics

### Zero Moment Point (ZMP)

The ZMP is the point on the ground where the net moment of ground reaction forces is zero. For stable walking, the ZMP must remain within the support polygon at all times.

The QP's contact forces directly determine the ZMP:

```python
def compute_zmp(f_contacts, contact_positions):
    total_fz = sum(f[2] for f in f_contacts)
    if total_fz < 1.0:  # robot is airborne
        return None
    zmp_x = sum(f[2] * p[0] + f[0] * (-p[2]) for f, p in zip(f_contacts, contact_positions)) / total_fz
    zmp_y = sum(f[2] * p[1] + f[1] * (-p[2]) for f, p in zip(f_contacts, contact_positions)) / total_fz
    return np.array([zmp_x, zmp_y])
```

During double support, the ZMP can be anywhere within the convex hull of both feet. During single support, it must be within the stance foot. The LIPM gait generator plans the CoM trajectory so that the ZMP stays within these bounds.

### Support Polygon Margin

The distance from the ZMP to the nearest edge of the support polygon is the stability margin. A positive margin means the robot is stable. The margin shrinks during single support (smaller polygon) and during CoM perturbations (reaching, carrying).

Typical margins during walking:
- Double support: 3 - 6 cm
- Single support: 1 - 3 cm
- Single support + reaching: 0.5 - 2 cm

The controller does not have an explicit constraint on ZMP margin. Instead, the CoM task indirectly maintains the margin by tracking the LIPM reference, which is designed to keep the ZMP centered within the support polygon.

## Implementation Notes

### Task List Management

The QP solver accepts a list of active tasks. Rather than creating and destroying task objects, the implementation maintains a persistent list and updates targets:

```python
# At the start of each control cycle
com_task.set_target(gait_gen.com_desired(t))
foot_l_task.set_target(gait_gen.foot_desired_left(t))
foot_r_task.set_target(gait_gen.foot_desired_right(t))
hand_l_task.set_target(hand_target_left(t))
hand_r_task.set_target(hand_target_right(t))

# Solve
result = qp.solve(q, v, tasks=[com_task, foot_l_task, foot_r_task,
                                hand_l_task, hand_r_task, posture_task],
                  contacts=gait_gen.active_contacts(t))
```

### Contact Switching

When the contact set changes (DSP to SSP or vice versa), the QP constraint matrix changes size. OSQP requires re-setup when the constraint dimensions change. To avoid the overhead of a full re-setup, the implementation pre-allocates the constraint matrix for the maximum contact set (double support) and zeros out rows corresponding to inactive contacts. This keeps the matrix dimensions constant and allows warm starting across contact transitions.

### Torque Application

After the QP solve, torques are applied to MuJoCo actuators. The selection matrix `S` maps from generalized torques (nv) to actuated torques (na). The QP directly solves for `tau` (na), which is applied as:

```python
result = qp.solve(q, v, tasks, contacts)
tau_clipped = clip_torques(result.tau, TORQUE_LIMITS)
mj_data.ctrl[actuator_ids] = tau_clipped
mujoco.mj_step(mj_model, mj_data)
```

## What to Study

1. **Contact schedule correctness.** Plot the contact states over time and verify they match the expected DSP/SSP pattern. Misaligned contacts cause the QP to produce incorrect contact forces.

2. **Arm drift during walking.** Record the hand position error over time and correlate it with the gait phase. Drift is largest during single support when the torso sways most.

3. **CoM tracking vs. hand tracking trade-off.** Increase the hand task weight and observe how CoM tracking degrades. Find the weight ratio where walking becomes unstable.

4. **ZMP trajectory.** Plot the ZMP position relative to the support polygon boundary. Verify it stays within bounds at all times.

5. **Torque profiles during swing.** Compare joint torques in the swing leg (no contact) vs. stance leg (bearing full weight). The asymmetry reflects the contact constraint's effect on the QP solution.

## Next Step

Move to `03_loco_manip_pipeline.md` for the full loco-manipulation state machine that sequences walking, grasping, carrying, and placing into a complete pipeline.
