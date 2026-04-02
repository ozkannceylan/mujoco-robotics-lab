# Lab 6 Code Walkthrough

A step-by-step reading guide for the dual-arm cooperative manipulation codebase.
Two UR5e arms grasp a box from opposite sides, lift it, carry it sideways, and
place it back on the table.

If you want to understand this codebase, read the files in the order below.
Each section tells you what to pay attention to and why it matters for later files.

---

## 1. `models/scene_dual.xml` (143 lines) -- The Physical World

**Start here.** Every other file ultimately serves this scene definition.

**Compiler and option block (lines 9-10)**

```xml
<compiler angle="radian" meshdir="assets" autolimits="true"/>
<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>
```

All joint ranges in this file are in radians, not degrees. The 1 ms timestep
(`0.001`) is referenced everywhere as `DT` in the Python code. `autolimits=true`
means writing `range=` on a joint automatically creates clamped limits without
needing a separate `<range>` element.

**Shared defaults (lines 15-37)**

The `class="ur5e"` default tree defines joint properties, material appearance,
and collision geometry classes that both arms inherit. Watch for:
- Line 18: `joint axis="0 1 0" range="-6.28319 6.28319"` -- default Y-axis rotation
  with full +/-2pi range. Individual joints override this (shoulder_pan uses Z-axis).
- Line 19-23: Nested `size3`/`size3_limited` classes. The `size3_limited` subclass
  tightens the range to +/-pi (used by elbow joints).
- Lines 27-33: Visual geoms (`contype="0" conaffinity="0"`) never collide.
  Collision geoms are capsules/cylinders in group 3.

**Asset sharing (lines 42-81)**

Both arms reference the same set of 20 OBJ meshes. The meshes are loaded once
here. The two arm XML files (included on lines 92-93) reference these meshes by
name without redefining them. This is how MuJoCo avoids duplicating geometry data.

**The two includes (lines 92-93)**

```xml
<include file="ur5e_left.xml"/>
<include file="ur5e_right.xml"/>
```

These pull in the full kinematic trees and actuators for both arms. Read these
next. The include order determines joint/actuator index ordering -- left arm
joints get indices 0-5, right arm gets 6-11.

**Weld equality constraints (lines 96-99)**

```xml
<weld name="left_grasp" body1="left_wrist_3_link" body2="box" active="false"/>
<weld name="right_grasp" body1="right_wrist_3_link" body2="box" active="false"/>
```

> WATCH FOR THIS: These are `active="false"` at start. The state machine
> activates them during the GRASP state. A weld constraint locks the relative
> pose between two bodies. If you activate a weld without first writing the
> correct relative pose into `eq_data`, MuJoCo will enforce the default
> (identity) relative transform and the box will fly across the room. See
> `_set_weld_relpose` in `bimanual_state_machine.py` for how this is handled.

**Table geometry (lines 110-127)**

The table body is at `pos="0.5 0 0.15"` with top half-height 0.02, giving a
surface at z = 0.17. The x half-extent is only 0.20 (table spans x = 0.3 to
0.7). This narrow width is deliberate -- a wider table would collide with the
arm shoulder and upper-arm links near the bases at x = 0.0 and x = 1.0. This
was a lesson learned from Lab 3.

**Box body (lines 129-139)**

```xml
<body name="box" pos="0.5 0 0.245">
  <freejoint name="box_freejoint"/>
  <geom name="box_geom" type="box" size="0.15 0.075 0.075"
        ... density="296" .../>
```

The box has a freejoint (6-DOF: 3 position + 4 quaternion in qpos). Its center
is at z = 0.245 = table surface (0.17) + box half-height (0.075). The `size`
attribute gives half-extents: the box is 30 cm x 15 cm x 15 cm. `density=296`
makes it about 1 kg. `friction="2.0"` is high to prevent the box sliding out
during the grasp approach. `condim=4` enables torsional friction.

---

## 2. `models/ur5e_left.xml` (79 lines) -- Left Arm Definition

**Base placement (line 8)**

```xml
<body name="left_base" pos="0 0 0" quat="0 0 0 -1" childclass="ur5e">
```

The quaternion `(w,x,y,z) = (0,0,0,-1)` is a Menagerie convention for mesh
alignment. In MuJoCo quaternion format, this is a 180-degree rotation about
the Z-axis -- but it is NOT an intentional yaw for the arm. The matching URDF
(see file 3 below) has the same rotation baked in via `rpy="0 0 3.14159..."`.
An earlier version of this lab added an extra 180-degree yaw on top of this,
causing the arms to point backward. That bug was fixed by removing the
extra yaw and keeping only the Menagerie mesh-alignment quaternion.

**Kinematic chain (lines 12-61)**

Follow the body nesting to trace the kinematic chain:
- `left_shoulder_link` at z = 0.163 above base. Joint axis: Z (`0 0 1`).
- `left_upper_arm_link` at y = 0.138 with `quat="1 0 1 0"` (90-degree rotation).
  Joint axis: default Y from the `ur5e` class.
- `left_forearm_link` at relative offset `0 -0.131 0.425`. Uses `size3_limited` class
  (elbow range restricted to +/-pi).
- Three wrist links continue the chain.

> WATCH FOR THIS: Only two joints use Z-axis rotation -- `shoulder_pan` (line 14,
> explicit `axis="0 0 1"`) and `wrist_2` (line 46, explicit `axis="0 0 1"`).
> All other joints default to Y-axis from the `ur5e` class. This matches the
> URDF axis definitions.

**EE site (lines 59-60)**

```xml
<site name="left_ee_site" pos="0 0.1 0"
      quat="0.7071 -0.7071 0 0" .../>
```

The site is 10 cm along the body +Y direction from wrist_3_link origin. The
quaternion `(0.7071, -0.7071, 0, 0)` is a -90 degree rotation about X, which
maps the body +Y axis to the site +Z axis. Convention: **site Z-axis = tool
approach direction**. This is the direction the end-effector "points" for
grasping. All IK targets in this codebase are expressed as rotation matrices
where the third column (Z) is the approach direction.

**Motor actuators (lines 70-77)**

Six motor actuators with direct torque control. Joints 1-3 (shoulder, elbow):
`ctrlrange="-150 150"` (Nm). Joints 4-6 (wrist): `ctrlrange="-28 28"` (Nm).
These limits are mirrored in `lab6_common.py` as `TORQUE_LIMITS`.

---

## 3. `models/ur5e_right.xml` (79 lines) -- Right Arm Definition

Structurally identical to `ur5e_left.xml` with two differences:

1. **Base position**: `pos="1.0 0 0"` (line 9) -- one meter to the right.
2. **All names prefixed with `right_`** instead of `left_`.

> WATCH FOR THIS: The right arm has the SAME base quaternion `(0,0,0,-1)` as the
> left arm. Both arms point in the same direction at their home configuration.
> The arms do NOT face each other by default. Facing each other is achieved
> purely through IK targets -- the state machine computes targets on opposite
> sides of the box, and IK finds joint configurations where the EEs point inward.

The EE site has a blue tint (`rgba="0.2 0.2 1.0 1"`) vs the left arm's red
(`rgba="1 0.2 0.2 1"`) for visual identification.

---

## 4. `models/ur5e.urdf` (147 lines) -- Pinocchio Kinematic Model

This is the URDF that Pinocchio loads. It defines one arm (no left/right
prefix). The dual-arm model loads this file twice, once per arm.

**World joint (lines 86-90)**

```xml
<joint name="world_joint" type="fixed">
  <origin rpy="0 0 3.14159265358979" xyz="0 0 0"/>
```

The 180-degree Z rotation here matches the MuJoCo `quat="0 0 0 -1"`. This
ensures Pinocchio FK outputs are in the same rotated frame as MuJoCo.

**Shoulder lift joint (lines 100-106)**

```xml
<joint name="shoulder_lift_joint" type="revolute">
  <origin rpy="0 1.5707963267948966 0" xyz="0 0.138 0"/>
  <axis xyz="0 1 0"/>
```

> WATCH FOR THIS: The `rpy="0 pi/2 0"` on this joint origin is what makes the
> UR5e kinematic chain work with Y-axis joints instead of the standard DH
> Z-axis convention. This is inherited from the Menagerie model and differs from
> textbook DH parameters. The same pattern appears on `wrist_1_joint` (line 119).

**EE fixed joint (lines 140-144)**

```xml
<joint name="ee_fixed_joint" type="fixed">
  <origin rpy="-1.5707963267948966 0 0" xyz="0 0.1 0"/>
```

The `rpy` of -pi/2 about X and the `xyz` offset of 10 cm along Y match the
MuJoCo EE site placement exactly. The `ee_link` frame is what Pinocchio FK
returns. If these did not match, Pinocchio IK solutions would not place the
MuJoCo EE site at the intended position.

---

## 5. `src/lab6_common.py` (115 lines) -- Configuration Hub

Every other Python file imports from here. Read this file to understand the
constants that wire together the MuJoCo scene and Python code.

**Joint slicing (lines 27-31)**

```python
LEFT_JOINT_SLICE  = slice(0, 6)
RIGHT_JOINT_SLICE = slice(6, 12)
LEFT_CTRL_SLICE   = slice(0, 6)
RIGHT_CTRL_SLICE  = slice(6, 12)
```

These slices index into `mj_data.qpos`, `mj_data.qvel`, `mj_data.qfrc_bias`,
and `mj_data.ctrl`. The ordering comes from the include order in
`scene_dual.xml` (left first, then right). The box freejoint adds 7 more qpos
entries (3 position + 4 quaternion) after index 11, but the arm code never
indexes into those.

**Home configuration (lines 45-53)**

```python
Q_HOME_LEFT = np.array([
    -math.pi / 2, -math.pi / 2, math.pi / 2,
    -math.pi / 2, -math.pi / 2, 0.0,
])
Q_HOME_RIGHT = Q_HOME_LEFT.copy()
```

Both arms use identical home angles. This puts the EE pointing roughly downward
toward the table center. The right arm's home mirrors the left because both
bases have the same orientation -- the right arm is merely translated 1 m along
X.

**Box half-extents (line 61)**

```python
BOX_HALF_EXTENTS = np.array([0.15, 0.075, 0.075])
```

These must match the MuJoCo `size` attribute on the box geom. They are used by
`grasp_pose_calculator.py` to compute where the EEs should be placed relative
to the box center.

**Quaternion converters (lines 65-74)**

```python
def mj_quat_to_pin(quat_wxyz):  # (w,x,y,z) -> (x,y,z,w)
def pin_quat_to_mj(quat_xyzw):  # (x,y,z,w) -> (w,x,y,z)
```

MuJoCo uses `(w,x,y,z)` quaternion order. Pinocchio uses `(x,y,z,w)`. Getting
this wrong produces silently incorrect rotations. These two functions are the
single point of conversion.

---

## 6. `src/dual_arm_model.py` (312 lines) -- Pinocchio Kinematic Layer

This is the analytical brain. MuJoCo simulates physics; this module computes
FK, Jacobians, and IK.

**Constructor (lines 46-60)**

The same URDF is loaded twice -- once for each arm. Each gets its own
`pin.Model` and `pin.Data`. The key fields:

```python
self.base_left  = np.array([0.0, 0.0, 0.0])
self.base_right = np.array([1.0, 0.0, 0.0])
```

Pinocchio does not know about the world-frame base position. It computes FK in
the arm's local frame (which starts at the base). The base offset is added
manually in every FK call and subtracted in every IK call.

**FK methods (lines 66-92)**

```python
def fk_left(self, q):
    pin.forwardKinematics(self.model_left, self.data_left, q)
    pin.updateFramePlacements(self.model_left, self.data_left)
    oMf = self.data_left.oMf[self.ee_frame_id_left]
    return pin.SE3(oMf.rotation.copy(), oMf.translation + self.base_left)
```

> WATCH FOR THIS: The `.copy()` on the rotation matrix is essential. Without it,
> the returned SE3 object holds a reference to internal Pinocchio data that gets
> overwritten on the next FK call. The `+ self.base_left` on line 78 is how the
> local-frame Pinocchio result becomes a world-frame result.

**DLS IK core: `_ik_single` (lines 166-237)**

The damped least-squares formula on line 219:

```python
dq = J.T @ np.linalg.solve(JJT + damping**2 * np.eye(n), err)
```

This is equivalent to `dq = J^T (J J^T + lambda^2 I)^{-1} e`. The damping
factor (`lambda = 0.01` by default) prevents singularity blowup near joint
limits. Step clamping on lines 222-224 limits each iteration to `dq_max = 0.5`
rad to prevent overshooting.

> WATCH FOR THIS: Position-only mode (lines 191-199) uses only the top 3 rows
> of the Jacobian (`J[:3,:]`). Full 6-DOF mode (lines 201-214) uses the full
> 6x6 Jacobian with both position and orientation error. The orientation error
> is computed as `pin.log3(R_target @ R_current.T)`, which gives the axis-angle
> representation of the rotation difference.

Convergence requires BOTH `pos_err < tol_pos` AND `rot_err < tol_rot` (line 204).
This is stricter than checking either alone and prevents configurations where
position is correct but orientation is wrong.

**Multi-start wrapper: `ik` (lines 239-311)**

If the first attempt (from `q_init` or `Q_HOME`) fails, the method tries
`n_restarts` random perturbations within +/-1 rad of the home configuration
(line 301). It returns the best solution found even if none converged. The base
offset is subtracted from the target position on line 284:

```python
target_pos_local = target_pos - base
```

This converts the world-frame target to the arm's local frame before passing it
to `_ik_single`, which operates entirely in local coordinates.

---

## 7. `src/joint_pd_controller.py` (70 lines) -- Torque Control

Short but critical. This is the only file that writes to `mj_data.ctrl`.

**The control law (line 63)**

```python
tau = self.kp * (q_target - q) + self.kd * (0.0 - qd) + bias
```

Where `bias = mj_data.qfrc_bias[jslice]`. The `qfrc_bias` vector from MuJoCo
includes gravity and Coriolis forces. Adding it as feedforward means the PD
terms only need to handle tracking error, not fight gravity.

> WATCH FOR THIS: The velocity target is hardcoded to `0.0`. This means the
> controller always damps toward zero velocity, not toward some desired velocity
> profile. Large motions are handled by the smooth-step ramp in
> `_run_pd_until_settle` (see file 8), which gradually moves the position target
> rather than jumping to it.

**Dual-arm loop (lines 55-58)**

The method loops over `(q_target, joint_slice, ctrl_slice)` pairs for both arms
in a single call. This ensures both arms are controlled in lockstep within the
same simulation step.

**Saturation tracking (lines 64-67)**

The `saturated` flag is set if any torque gets clipped. This is useful for
debugging -- if an arm saturates during a motion, the gains may be too high or
the motion too aggressive.

---

## 8. `src/grasp_pose_calculator.py` (95 lines) -- Where to Grab

Generates SE3 grasp poses from the current box state in MuJoCo.

**`_rotation_facing` (lines 27-47)**

This is a "look-at" function. Given a direction vector, it builds a rotation
matrix where:
- Column 2 (Z-axis) = the direction (approach direction)
- Column 1 (Y-axis) ~ world up
- Column 0 (X-axis) = cross(up, z)

> WATCH FOR THIS: The degenerate case check on line 40 -- if the approach
> direction is nearly vertical (`abs(dot(z, up)) > 0.99`), the cross product
> with world-up would be near zero. The fallback uses world +Y instead.

**`compute_grasp_poses` (lines 50-94)**

The box's X-axis in world frame (`box_rot[:, 0]`) is the longest dimension
direction. The left arm approaches from the -X side of the box (EE Z-axis
points toward +X, toward the box center). The right arm approaches from the +X
side (EE Z-axis points toward -X).

Standoff distances:
- `APPROACH_STANDOFF = 0.10` -- 10 cm from box surface (safe approach)
- `GRASP_STANDOFF = 0.05` -- 5 cm from box surface (pre-contact position)

The positions are computed as:
```python
left_grasp_pos  = box_center - box_x_axis * (half_x + standoff)
right_grasp_pos = box_center + box_x_axis * (half_x + standoff)
```

The function returns a dict of 4 SE3 poses (approach and grasp for each arm).
Everything is derived from the live box pose -- nothing is hardcoded to the
initial box position.

---

## 9. `src/bimanual_state_machine.py` (638 lines) -- The Main Integration

This is the largest and most important file. It orchestrates the entire
manipulation pipeline. Read it in logical sections.

### 9a. State enum and constants (lines 47-70)

```python
class State(enum.Enum):
    APPROACH = 1  # Both arms move to grasp standoff
    CLOSE    = 2  # Push EEs 2 cm inside box surface
    GRASP    = 3  # Activate weld constraints
    LIFT     = 4  # Raise box 15 cm
    CARRY    = 5  # Translate box 20 cm in +y
    PLACE    = 6  # Lower box back to table
    DONE     = 7
```

Key constants:
- `CONTACT_PENETRATION = 0.02` -- during CLOSE, EE targets are placed 2 cm
  *inside* the box surface. This ensures physical contact.
- `LIFT_DZ = 0.15` -- lift 15 cm.
- `CARRY_DY = 0.20` -- carry 20 cm in the +Y direction (not +X).

> WATCH FOR THIS: The carry direction is +Y, not +X. Moving the box along X
> (toward either arm base) would push one arm toward its workspace limit while
> the other would need to over-extend. Moving along Y keeps both arms at
> symmetric configurations and avoids joint limits. This was discovered through
> experimentation.

### 9b. Joint wrapping: `_wrap_joints` (lines 73-79)

```python
q_out[i] = q_ref[i] + ((diff + np.pi) % (2 * np.pi) - np.pi)
```

**Why this matters:** IK can return equivalent angles that differ by 2*pi.
For example, if the current joint angle is 0.1 rad and IK returns 6.38 rad
(which is 0.1 + 2*pi), the PD controller would command a 6.28 rad motion
when only 0.0 rad of motion is needed. This function wraps each joint angle to
the equivalent value closest to `q_ref`, ensuring the controller takes the
shortest path.

The formula works by: (1) computing `diff = q - q_ref`, (2) mapping diff into
`[-pi, +pi)` via the modulo trick, (3) adding back `q_ref`. The result is
always within pi radians of the reference.

### 9c. Collision-free IK: `_find_collision_free_ik` (lines 82-152)

This is the most complex helper function. It runs up to 300 random IK trials
and picks the collision-free solution closest to the reference configuration.

For each trial (lines 106-151):
1. Generate initial guess: first trial uses `center` (previous config), others
   add uniform noise in `[-2.5, 2.5]` rad.
2. Run DLS IK via `dual._ik_single`.
3. If IK converged, wrap joints to avoid large motions.
4. Verify FK matches target (line 119) -- a sanity check that wrapping did not
   break the solution.
5. **Collision check** (lines 123-143): Temporarily set `mj_data.qpos[jslice]`
   to the candidate, call `mj_forward`, and scan all contacts.

> WATCH FOR THIS: The contact loop on lines 128-139 does NOT simply reject any
> contact. It explicitly skips box-table contacts (line 136-137) because the box
> is resting on the table throughout approach. It also only flags contacts where
> the current arm is involved (line 138). Without these filters, the approach
> phase would always fail because the box-table contact is permanent.

After the loop, the function restores the original qpos (line 141) and calls
`mj_forward` again to avoid corrupting the simulation state.

### 9d. `_compute_ee_targets_from_box` (lines 155-179)

A simpler version of `compute_grasp_poses` that returns raw arrays instead of
SE3 objects. Used by LIFT, CARRY, and PLACE states where the box position is
hypothetical (not yet achieved).

The standoff parameter can be negative. During CLOSE, LIFT, CARRY, and PLACE,
`standoff = -CONTACT_PENETRATION = -0.02`, which places the EE target 2 cm
inside the box surface. This ensures the weld constraints have something to
grip.

### 9e. Weld relative pose: `_set_weld_relpose` (lines 277-307)

```python
rel_pos  = mat1.T @ (pos2 - pos1)     # body2 position in body1 frame
rel_mat  = mat1.T @ mat2              # body2 rotation in body1 frame
mujoco.mju_mat2Quat(rel_quat, rel_mat.flatten())
```

> WATCH FOR THIS: This is the most critical function for preventing the box
> from flying away. MuJoCo weld `eq_data` layout is 11 floats per constraint:
>
> | Index | Meaning |
> |-------|---------|
> | 0:3   | Anchor point on body1 (local frame) |
> | 3:6   | Relative position of body2 in body1 frame |
> | 6:10  | Relative quaternion (w,x,y,z) of body2 in body1 frame |
> | 10    | Torque scale |
>
> The anchor is set to `[0,0,0]` (body1 origin). The relative pose is
> computed from the *current* simulation state. This means the weld will
> maintain whatever relative pose exists at the moment of activation.
> If you forget to call this before activating the weld, MuJoCo enforces
> the default relative pose (usually identity), which is wrong.

### 9f. Smooth-step settling: `_run_pd_until_settle` (lines 330-397)

This is the motion execution engine. Every state transition goes through here.

**Smooth-step interpolation (lines 360-363)**

```python
alpha = min(1.0, (step + 1) / n_ramp)
alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)
self.q_target_left = q_start_left + alpha_smooth * (q_final_left - q_start_left)
```

The `alpha^2 * (3 - 2*alpha)` is the Hermite smoothstep function. It has zero
derivative at both endpoints, so the motion starts and ends smoothly rather than
jerking. The ramp duration is 2 seconds by default.

**Dual exit criteria (lines 369-388)**

The function exits on either of two conditions:
1. **Position-based**: maximum joint error < 0.005 rad for both arms, sustained
   for 0.15 seconds (150 steps).
2. **Velocity-based**: maximum joint velocity < 0.02 rad/s for both arms,
   sustained for 0.15 seconds. This only activates after the ramp completes
   (`alpha >= 1.0`).

The velocity exit handles phases where weld constraints cause small residual
position errors that never fully converge. Without it, those phases would
always hit the timeout.

### 9g. The `run()` method: full state machine (lines 412-637)

Read this as a linear script. Each state follows the same pattern:
1. Compute target box position.
2. Compute EE targets from box target via `_compute_ee_targets_from_box`.
3. Solve IK (collision-free for APPROACH, simple for others).
4. Set `q_target_left` and `q_target_right`.
5. Call `_run_pd_until_settle`.
6. Print diagnostics.

**APPROACH (lines 430-474)** -- Two-phase approach. First moves from HOME to
approach standoff (10 cm from box), then from approach standoff to grasp
standoff (5 cm from box). Both phases use collision-free IK. The two-phase
design avoids large single motions that are harder to solve collision-free.

**CLOSE (lines 479-499)** -- Pushes EEs to `standoff = -0.02` (inside box
surface). Uses simple IK (`collision_free=False`) because contact is intentional.

**GRASP (lines 512-528)** -- The critical transition. Lines 517-518 compute
and set weld relative poses. Lines 520-521 activate the constraints. A brief
hold (0.15s) lets the solver stabilize.

> WATCH FOR THIS: The order matters. `_set_weld_relpose` MUST be called BEFORE
> `eq_active` is set to 1. If reversed, there is one timestep where the weld is
> active with the wrong relative pose, causing an impulse.

**LIFT (lines 533-554)** -- Target box position is current position + `[0, 0, 0.15]`.
IK uses `standoff = -CONTACT_PENETRATION` to keep EEs pressed against the box.

**CARRY (lines 559-580)** -- Target box position is current position + `[0, 0.20, 0]`.
The +Y direction keeps both arms in comfortable configurations.

**PLACE (lines 585-637)** -- Target z is the table surface + box half-height
(z = 0.245). After lowering, welds are deactivated (lines 611-612), and the
arms retract to approach standoff to avoid pushing the box.

---

## 10. `src/m4_cooperative_carry.py` (222 lines) -- Runner and Validation

This is the entry point for running the cooperative carry pipeline.

**Controller gains (lines 44-45)**

```python
KP = 300.0
KD = 40.0
```

These are lower than previous milestones (which used 500/50). The smooth-step
ramp in `_run_pd_until_settle` handles the large motions, so the gains only
need to track a slowly-changing target. Lower gains produce smoother, less
jerky motion.

**Trajectory plot: `plot_box_trajectory` (lines 48-99)**

Plots box x, y, z vs time with state transition markers (vertical red dashed
lines). Uses the project's standard dark theme (`#08111f` background). Initial
position reference lines (white dotted) show how far the box moved from its
starting position.

**Rotation error: `compute_rotation_error_deg` (lines 102-107)**

```python
R_rel = R1.T @ R2
angle_rad = arccos(clip((trace(R_rel) - 1) / 2, -1, 1))
```

This is the geodesic distance between two rotation matrices. It extracts the
angle of the axis-angle representation of `R1^T @ R2`. The double clip (first
on trace, then on the arccos argument) guards against numerical noise pushing
values outside `[-1, 1]`.

**Gate criteria (lines 197-216)**

Six pass/fail checks:
1. `lift_dz >= 0.13` -- box was lifted at least 13 cm above initial z.
2. `carry_dy >= 0.18` -- box was carried at least 18 cm in +y.
3. `place_dz < 0.03` -- box final z is within 3 cm of initial z (back on table).
4. `rot_err_deg < 10.0` -- box final orientation within 10 degrees of initial.
5. Video file exists.
6. Trajectory plot exists.

> WATCH FOR THIS: The lift and carry checks use the MAXIMUM achieved value
> during the relevant states (lines 175-189), not the final value. This accounts
> for slight settling after the motion completes.

---

## 11. `src/m5_capstone_demo.py` (273 lines) -- Final Integration

The capstone adds scene validation, state-text overlay on video frames, and a
summary trajectory plot.

**Scene validation (lines 197-203)**

Counts hinge joints and actuators, asserts both equal `NUM_JOINTS_TOTAL = 12`.
This catches model XML errors early.

**Text overlay: `burn_text` (lines 59-85)**

Burns state name and simulation time onto each video frame using a minimal
5x7 bitmap font rendered at 4x scale. This avoids any dependency on PIL/OpenCV
for text rendering. The glyph dictionary (`_get_glyphs`, lines 88-131) is a
hand-coded bitmap font covering uppercase letters, digits, space, colon, and
period.

**Frame-to-state mapping (lines 234-241)**

The state machine logs state at every simulation step (1000 Hz), but frames are
captured every `FRAME_SKIP` steps (33 steps = 30 FPS). The overlay loop maps
each frame index back to the corresponding simulation step to look up the
correct state.

---

## Non-Obvious Patterns

### Pattern 1: Base offset is applied manually, not via Pinocchio placement

Pinocchio has a concept of universe placement for the root joint, but this
codebase does not use it. Instead, `DualArmModel` adds `base_left` or
`base_right` to every FK translation result and subtracts it from every IK
target position. This keeps the Pinocchio model simple (one URDF, no
modifications) at the cost of remembering the offset convention.

**Where to look:** `dual_arm_model.py` lines 78, 92 (FK add), line 284 (IK subtract).

### Pattern 2: The URDF is from Lab 3, not standard DH

The URDF uses Menagerie-compatible kinematics with Y-axis joints and
non-standard origin offsets (e.g., 90-degree pitch on shoulder_lift and
wrist_1). This is NOT the textbook DH parameterization of UR5e. If you compare
joint axes to a DH table, they will not match. The URDF was validated
empirically against MuJoCo FK outputs in earlier labs.

**Where to look:** `ur5e.urdf` lines 103, 119 (the `rpy="0 pi/2 0"` origins).

### Pattern 3: Collision-free IK filters contacts by body name

The contact loop in `_find_collision_free_ik` does not reject all penetrating
contacts. It resolves body names for both geoms in each contact pair and:
- Skips contacts where one body contains "table" and the other contains "box".
- Only counts contacts where the current arm's name prefix appears.

This means the box resting on the table (always in contact) does not invalidate
approach configurations. And contacts between the *other* arm and the table do
not affect the current arm's IK search.

**Where to look:** `bimanual_state_machine.py` lines 128-143.

### Pattern 4: Weld relpose must be set BEFORE activation

The `_set_weld_relpose` method reads the current body poses from `mj_data` and
writes the relative transform into `mj_model.eq_data`. This must happen while
the weld is still inactive. Once activated, MuJoCo immediately starts enforcing
whatever is in `eq_data`. If it still contains the default (identity) transform,
the constraint solver will apply huge forces to move the box to the wrist origin.

**Where to look:** `bimanual_state_machine.py` lines 517-521 (set, then activate).

### Pattern 5: Negative standoff for contact phases

During CLOSE, LIFT, CARRY, and PLACE, the standoff passed to
`_compute_ee_targets_from_box` is `-CONTACT_PENETRATION = -0.02`. This places
the EE target 2 cm *inside* the box. Since the EE cannot physically penetrate
the box, the PD controller applies continuous force against the box surface,
maintaining contact pressure. This is essential for the weld constraints to feel
"natural" -- the arm is actively pushing against the box, not hovering nearby.

**Where to look:** `bimanual_state_machine.py` lines 484, 538, 564, 595.

### Pattern 6: Joint wrapping prevents unnecessary full rotations

Without wrapping, an IK solution of `q = 6.18 rad` for a joint currently at
`q = -0.10 rad` would cause the PD controller to command a 6.28 rad
(360-degree) rotation. With wrapping relative to `Q_HOME`, the solution becomes
`q = -0.10 rad` (the equivalent angle closest to the reference), and the
controller barely moves.

**Where to look:** `bimanual_state_machine.py` line 73-79. Called on lines 117,
217, 223 after every IK solve.

---

## File Dependency Graph

```
scene_dual.xml
  includes ur5e_left.xml
  includes ur5e_right.xml

lab6_common.py  (no dependencies, pure constants)
  |
  +-- dual_arm_model.py  (imports lab6_common, pinocchio; loads ur5e.urdf)
  +-- joint_pd_controller.py  (imports lab6_common)
  +-- grasp_pose_calculator.py  (imports lab6_common, mujoco, pinocchio)
  |
  +-- bimanual_state_machine.py  (imports all of the above)
      |
      +-- m4_cooperative_carry.py  (runner script)
      +-- m5_capstone_demo.py  (runner script with overlay)
```

---

## Summary: Reading Strategy

1. Start with the XML files (scene, arms, URDF) to understand what physically
   exists in the simulation.
2. Read `lab6_common.py` to learn the naming conventions and constants.
3. Read `dual_arm_model.py` to understand how Pinocchio talks to the scene.
4. Read `joint_pd_controller.py` -- it is short and straightforward.
5. Read `grasp_pose_calculator.py` to see how grasp targets are derived from
   the box state.
6. Read `bimanual_state_machine.py` section by section (helpers first, then
   `run()`). This is where everything comes together.
7. Skim the runner scripts (`m4`, `m5`) for the outer loop, gain tuning, and
   gate criteria.
