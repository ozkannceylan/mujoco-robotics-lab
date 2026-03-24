# 01: G1 Humanoid Setup and Standing Balance

## Goal

Load the Unitree G1 humanoid in both Pinocchio and MuJoCo, cross-validate forward kinematics between the two engines, and implement a standing balance controller that keeps the robot upright under gravity and external perturbations.

## Files

- Common module: `src/lab7_common.py`
- G1 model wrapper: `src/g1_model.py`
- Balance controller: `src/balance_controller.py`
- Standing demo: `src/a1_standing_balance.py`
- MuJoCo scene: `models/scene_flat.xml`
- MuJoCo model: `models/g1_humanoid.xml`
- Pinocchio model: `models/g1_humanoid.urdf`
- Tests: `tests/test_g1_model.py`, `tests/test_balance.py`

---

## The G1 Humanoid Model

### Simplified 23 DOF Structure

The G1 used in this lab is a simplified version of the Unitree G1 humanoid. The full robot has over 37 DOF including dexterous hands, but for locomotion we lock the arms and focus on the lower body:

| Joint Group | Joints | DOF |
|-------------|--------|-----|
| Waist | waist_yaw | 1 |
| Left leg | hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll | 6 |
| Right leg | hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll | 6 |
| Arms | locked via fixed joints | 0 |
| **Total actuated** | | **13** |

The floating base adds 7 configuration variables (3 position + 4 quaternion) and 6 velocity variables (3 linear + 3 angular), giving:

- Configuration dimension `nq = 7 + 13 = 20`
- Velocity dimension `nv = 6 + 13 = 19`

The distinction between `nq` and `nv` exists because quaternions have 4 components but only 3 degrees of freedom. This is a fundamental property of the SO(3) manifold — rotations live on a 3D surface embedded in 4D space.

### Joint Ordering and Naming

The 13 actuated joints follow a specific ordering defined in `lab7_common.py`:

```python
JOINT_NAMES = (
    "waist_yaw",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
)
```

Each leg has a 3-DOF hip (yaw for turning, roll for lateral motion, pitch for forward swing), a 1-DOF knee (pitch), and a 2-DOF ankle (pitch for push-off, roll for lateral balance). The waist provides an additional degree of freedom for torso rotation.

### Home Configuration

The home pose `Q_HOME` defines the nominal standing configuration:

```python
Q_HOME = np.array([
    0.0,     # waist_yaw
    0.0,     # left_hip_yaw
    0.0,     # left_hip_roll
    -0.3,    # left_hip_pitch (slight forward lean)
    0.6,     # left_knee_pitch (bent)
    -0.3,    # left_ankle_pitch (compensate)
    0.0,     # left_ankle_roll
    0.0,     # right_hip_yaw
    0.0,     # right_hip_roll
    -0.3,    # right_hip_pitch
    0.6,     # right_knee_pitch
    -0.3,    # right_ankle_pitch
    0.0,     # right_ankle_roll
])
```

The knees are bent at 0.6 rad (~34 degrees) rather than fully extended. This is deliberate: a slightly crouched stance lowers the center of mass and increases the effective workspace of the ankle joints, both of which improve balance authority. The hip and ankle pitch angles are set to compensate so the torso remains roughly vertical.

---

## Dual Model Setup: Pinocchio and MuJoCo

### Architecture Principle

The fundamental architecture of this lab series separates analytical computation from physics simulation:

```
Pinocchio = analytical brain (FK, Jacobians, CoM, mass matrix, IK)
MuJoCo   = physics simulator (step, render, contact, sensors)
```

Pinocchio computes everything that has an analytical formula. MuJoCo handles everything that requires contact resolution, friction, and time-stepping. The two are cross-validated at startup to confirm they agree.

### Pinocchio: Free-Flyer Root Joint

A fixed-base robot arm is loaded into Pinocchio with a fixed root joint — the base never moves. A humanoid is different. The pelvis is not attached to anything. It floats freely in space, and its position and orientation are part of the configuration.

Pinocchio represents this with `JointModelFreeFlyer`:

```python
import pinocchio as pin

model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
data = model.createData()
```

The free-flyer adds:
- 7 configuration variables: `[x, y, z, qx, qy, qz, qw]`
- 6 velocity variables: `[vx, vy, vz, wx, wy, wz]`

These occupy indices `q[0:7]` and `v[0:6]`. The actuated joint values start at `q[7:]` and `v[6:]`.

### MuJoCo: Floating Base via Free Joint

In MuJoCo, the same concept is a "free joint" on the root body:

```xml
<body name="pelvis" pos="0 0 0.82">
  <freejoint name="float_base"/>
  ...
</body>
```

MuJoCo stores the free joint as `qpos[0:7] = [x, y, z, qw, qx, qy, qz]`. Note the quaternion order difference from Pinocchio.

### Quaternion Convention: The Critical Difference

This is where most floating-base bugs originate:

| Engine | Quaternion Order | Configuration Layout |
|--------|-----------------|---------------------|
| Pinocchio | `(x, y, z, w)` | `[x, y, z, qx, qy, qz, qw, j1, ..., j13]` |
| MuJoCo | `(w, x, y, z)` | `[x, y, z, qw, qx, qy, qz, j1, ..., j13]` |

The conversion utilities in `lab7_common.py` handle this:

```python
def mj_quat_to_pin(quat_wxyz: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> (x,y,z,w)"""
    w, x, y, z = quat_wxyz
    return np.array([x, y, z, w])

def pin_quat_to_mj(quat_xyzw: np.ndarray) -> np.ndarray:
    """(x,y,z,w) -> (w,x,y,z)"""
    x, y, z, w = quat_xyzw
    return np.array([w, x, y, z])
```

These are called every time state is transferred between the two engines. The `get_state_from_mujoco()` function reads `mj_data.qpos` and `mj_data.qvel`, converts the quaternion, and returns a Pinocchio-convention `(q_pin, v_pin)` pair. The inverse function `set_mujoco_state()` does the opposite.

Getting this wrong is subtle. If you swap `w` into the wrong slot, the quaternion is still unit norm and the FK still returns a position — just the wrong one. The robot appears rotated by a seemingly random angle. The first time you see this, you spend an hour looking for a joint limit error before realizing it is a quaternion convention bug.

### State Conversion in Detail

The full state conversion from MuJoCo to Pinocchio:

```python
def get_state_from_mujoco(mj_model, mj_data):
    q_pin = np.zeros(NQ_PIN)  # 20
    v_pin = np.zeros(NV_PIN)  # 19

    # Position (same in both)
    q_pin[0:3] = mj_data.qpos[0:3]

    # Quaternion: MuJoCo (w,x,y,z) -> Pinocchio (x,y,z,w)
    q_pin[3:7] = mj_quat_to_pin(mj_data.qpos[3:7])

    # Actuated joints: mapped by name, not by index
    qpos_addrs = get_mj_joint_qpos_adr(mj_model)
    for i in range(NUM_ACTUATED):
        q_pin[NQ_FREEFLYER + i] = mj_data.qpos[qpos_addrs[i]]

    # Velocities: freeflyer (same convention)
    v_pin[0:6] = mj_data.qvel[0:6]

    # Actuated joint velocities
    dof_addrs = get_mj_joint_dofadr(mj_model)
    for i in range(NUM_ACTUATED):
        v_pin[NV_FREEFLYER + i] = mj_data.qvel[dof_addrs[i]]

    return q_pin, v_pin
```

The joint mapping is done by name, not by positional index. This is critical because MuJoCo and Pinocchio may order joints differently in their internal arrays. The `get_mj_joint_qpos_adr()` function looks up each joint name and returns its `qpos` address in MuJoCo, ensuring correct alignment.

---

## Cross-Validation: Pinocchio vs MuJoCo FK

Cross-validation is performed at startup by comparing FK results for multiple random configurations:

```python
pin.forwardKinematics(model, data, q_pin)
pin.updateFramePlacements(model, data)
ee_pin = data.oMf[foot_frame_id].translation

mujoco.mj_forward(mj_model, mj_data)
ee_mj = mj_data.xpos[body_id]

assert np.allclose(ee_pin, ee_mj, atol=1e-3)
```

The pass criterion is agreement within 1 mm. This is the trust anchor for the rest of the lab. If FK disagrees, the Jacobians will be wrong, the IK will compute incorrect configurations, and the balance controller will fail in confusing ways.

The `G1Model` class caches frame IDs for the left and right feet at initialization:

```python
self.left_foot_id = self.model.getFrameId("left_foot")
self.right_foot_id = self.model.getFrameId("right_foot")
```

These frame IDs map to specific bodies in MuJoCo. The mapping is verified once during setup and reused throughout.

---

## Center of Mass Computation

### Why CoM Matters for Balance

A robot remains balanced when the vertical projection of its center of mass (CoM) falls inside the support polygon — the convex hull of the contact points with the ground. If the CoM projection leaves the support polygon, the robot tips over.

For the G1 standing with both feet on the ground, the support polygon is roughly a rectangle defined by the outer edges of both feet. The CoM projection must stay inside this rectangle.

### Pinocchio CoM API

Pinocchio provides direct CoM computation:

```python
pin.centerOfMass(model, data, q)
com_position = data.com[0]  # (3,) world frame

pin.centerOfMass(model, data, q, v)
com_velocity = data.vcom[0]  # (3,) world frame

pin.jacobianCenterOfMass(model, data, q)
J_com = data.Jcom  # (3 x nv) matrix
```

The CoM Jacobian `J_com` maps joint velocities to CoM velocity: `v_com = J_com @ v`. Its transpose maps forces at the CoM to joint torques: `tau = J_com^T @ F_com`. This is the key relationship used by the balance controller.

---

## Standing Balance Controller

### Control Law

The balance controller has three components:

1. **CoM tracking force**: PD control on the center of mass position

$$F_{com} = K_p \cdot (x_{com}^{des} - x_{com}) + K_d \cdot (\dot{x}_{com}^{des} - \dot{x}_{com})$$

2. **Projection to joint torques**: via the CoM Jacobian transpose

$$\tau_{com} = J_{com}^T \cdot F_{com}$$

3. **Posture regulation**: prevents joint drift

$$\tau_{posture} = K_{p,post} \cdot (q_{home} - q) + K_{d,post} \cdot (-\dot{q})$$

4. **Gravity compensation**: counteracts gravitational forces

$$\tau_{gravity} = g(q)$$

The total torque is:

$$\tau = J_{com}^T \cdot F_{com} + g(q) + \tau_{posture}$$

The gravity vector `g(q)` is computed by Pinocchio using RNEA with zero velocity and zero acceleration. This gives the generalized forces due to gravity alone.

### Gains

The `BalanceGains` dataclass holds the PD gains:

```python
@dataclass
class BalanceGains:
    Kp_com: np.ndarray      # (3,) stiffness on CoM position
    Kd_com: np.ndarray      # (3,) damping on CoM velocity
    Kp_posture: float       # posture stiffness
    Kd_posture: float       # posture damping
```

The CoM gains are 3D vectors because lateral balance (X, Y) typically requires different stiffness than vertical (Z). Vertical stiffness can be lower because the support forces from the ground handle most of the vertical load.

### Support Polygon

The support polygon is computed from the foot contact positions:

```python
def compute_support_polygon(self, q):
    lfoot = self.g1.get_left_foot_pos(q)
    rfoot = self.g1.get_right_foot_pos(q)
    # Build convex hull from foot corner positions
    # Each foot contributes 4 corners based on foot geometry
    ...
```

During double support (both feet on the ground), the support polygon is the convex hull of all contact points from both feet. The balance controller checks that the CoM XY projection remains inside this polygon.

### Gravity Feedforward for Menagerie Position Servos

The MuJoCo model uses Menagerie-style position-controlled actuators. These are `general` actuators with the control law:

$$\tau = K_p \cdot (ctrl - q_{pos}) - K_d \cdot \dot{q}_{vel}$$

If we set `ctrl = q_{des}`, there will be a steady-state offset because the servo does not know about gravity. The robot droops. This was discovered in Lab 3 and the solution is feedforward compensation:

$$ctrl = q_{des} + \frac{q_{frc,bias}}{K_p} + \frac{K_d \cdot \dot{q}_{des}}{K_p}$$

where `qfrc_bias` is MuJoCo's precomputed gravity + Coriolis force at the current state. The `ACTUATOR_KP` and `ACTUATOR_KD` arrays in `lab7_common.py` store the per-joint gains extracted from the MJCF:

```python
ACTUATOR_KP = np.array([200, 200, 200, 300, 300, 150, 80,
                        200, 200, 300, 300, 150, 80])
ACTUATOR_KD = np.array([20, 20, 20, 30, 30, 15, 8,
                        20, 20, 30, 30, 15, 8])
```

The hip and knee joints have higher gains (300 Nm/rad) because they bear more load. Ankle roll has the lowest gain (80 Nm/rad) because it handles smaller torques.

---

## Standing Balance Demo

The demo script `src/a1_standing_balance.py` runs the following sequence:

1. Load G1 in MuJoCo scene, set to home pose
2. Initialize Pinocchio model and cross-validate FK
3. Run standing balance controller for 10 seconds
4. Apply 5 N lateral perturbation at t = 3 s (push at pelvis)
5. Apply 5 N lateral perturbation at t = 6 s (opposite direction)
6. Verify recovery within 2 seconds after each push
7. Save plots: CoM trajectory, support polygon with CoM projection, joint torques

### Success Criteria

- G1 stands for 10 seconds without falling
- CoM XY projection stays within the support polygon at all times
- After each perturbation, CoM returns to within 5 mm of its nominal position within 2 seconds
- FK cross-validation error < 1 mm for all 5 test configurations

### Output Media

- `media/standing_com_trajectory.png` — CoM position over time (X, Y, Z)
- `media/standing_support_polygon.png` — top-down view of support polygon with CoM projection trace
- `media/standing_joint_torques.png` — actuated joint torques over time
- `media/standing_perturbation_recovery.mp4` — video of perturbation and recovery

---

## What to Study

1. **Quaternion conventions** — trace the data flow from `mj_data.qpos[3:7]` through `mj_quat_to_pin()` into the Pinocchio configuration vector. This pattern repeats in every floating-base lab.

2. **Free-flyer nq vs nv** — understand why `nq = 20` but `nv = 19`. The quaternion constraint (unit norm) removes one degree of freedom. Pinocchio handles this internally via the `integrate()` function, which updates the quaternion on the manifold rather than adding a 4D vector.

3. **CoM Jacobian** — the `J_com` matrix maps joint velocities to CoM velocity. Its transpose maps CoM forces to joint torques. This is the theoretical foundation for all balance control.

4. **Gravity feedforward** — compare the tracking accuracy with and without the `qfrc_bias/Kp` term. The difference is dramatic: without feedforward, the robot droops visibly and the CoM drifts toward the edge of the support polygon.

## Next Step

Move to Phase 2: ZMP Planning. The standing balance controller keeps the robot upright statically. Walking requires dynamic balance — the CoM will intentionally leave the current support polygon and be caught by the next foot placement. The ZMP framework provides the mathematical foundation for planning this.
