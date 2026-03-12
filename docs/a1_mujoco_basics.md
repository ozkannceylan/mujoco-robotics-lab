# A1: MuJoCo Environment Setup — Study Notes

## What Is MuJoCo?

**MuJoCo** = "Multi-Joint dynamics with Contact". A physics engine built for robotics simulation.

Compared to Gazebo or Isaac Sim: lighter, directly accessible via a Python API, and optimised for headless batch simulation. Preferred in ML pipelines because of speed — you can run thousands of simulations in parallel.

---

## MJCF: The Robot Description Language

Everything in MuJoCo is defined using **MJCF** (MuJoCo XML Format). Mental model:

```
<mujoco>
  ├── <option>        ← physics parameters (gravity, timestep)
  ├── <worldbody>     ← scene hierarchy (body → joint → geom)
  ├── <actuator>      ← motors (attached to joints)
  └── <sensor>        ← measurements (position, velocity, force)
```

### Three Core Concepts

| Concept | Purpose | Analogy |
|---------|---------|---------|
| **Body** | Coordinate frame. An invisible reference point. | A skeletal joint |
| **Geom** | Physical shape (collision + visual). Attached to a body. | The bone/flesh attached to that joint |
| **Joint** | Degree of freedom between two bodies. Defined *inside* a body. | The joint's direction of motion |

**Hierarchy:** Bodies are nested (parent-child). A joint defines how the child body moves relative to its parent. A geom is the body's physical presence.

---

## Our Robot: 2-Link Planar Manipulator

```
         joint1          joint2         end-effector
  (base) ──○──── Link1 ────○──── Link2 ────●
           │    (0.3m)      │    (0.3m)
           │ z-axis         │ z-axis
           │ hinge          │ hinge
```

- **2 links**, each 0.3 m
- **2 hinge joints**, rotating about the z-axis (operates in the XY plane)
- **2 motors**, one per joint (ctrl range: [-10, 10])
- **Gravity off** — planar robot, no gravity complications (for now)

### Key Details in the MJCF

```xml
<!-- Body hierarchy: base → link1 → link2 -->
<body name="link1" pos="0 0 0">
  <joint name="joint1" type="hinge" axis="0 0 1"/>
  <geom fromto="0 0 0  0.3 0 0"/>       ← link geometry

  <body name="link2" pos="0.3 0 0">     ← starts at the tip of link1
    <joint name="joint2" type="hinge" axis="0 0 1"/>
    <geom fromto="0 0 0  0.3 0 0"/>

    <site name="end_effector" pos="0.3 0 0"/>  ← tracking point
  </body>
</body>
```

**`pos="0.3 0 0"`** → link2 begins at the tip of link1 (0.3 m). Thanks to this parent-child relationship, when joint1 rotates, link2 rotates along with it — that is the kinematic chain.

**`site`** → Has no physical effect, just a tracking point. We read the end-effector position via `data.site_xpos`.

---

## MuJoCo API — Basic Usage

```python
import mujoco

# 1. Load model
model = mujoco.MjModel.from_xml_path("models/two_link.xml")
data  = mujoco.MjData(model)

# 2. Simulation step
mujoco.mj_step(model, data)

# 3. Read data
data.qpos          # joint angles [θ₁, θ₂]
data.qvel          # joint velocities [θ̇₁, θ̇₂]
data.site_xpos[i]  # Cartesian positions of sites

# 4. Send motor commands
data.ctrl[0] = 5.0   # joint1 motor
data.ctrl[1] = -3.0  # joint2 motor

# 5. Update kinematics (without stepping)
mujoco.mj_forward(model, data)
```

### model vs data

- **model** (`MjModel`): Constant parameters — link lengths, masses, joint limits. Does not change during simulation.
- **data** (`MjData`): Instantaneous state — joint angles, velocities, forces. Updated with every `mj_step()`.

---

## Verification Results

From the test run:

| Criterion | Result |
|-----------|--------|
| `mj_step()` runs without error | ✓ 100 steps, 0.2 s simulation time |
| `data.qpos` has 2 elements | ✓ shape=(2,) |
| Motor command changes joint angles | ✓ ctrl=[5, -3] → Δθ=[+0.055, -0.055] rad |
| End-effector position is readable | ✓ site_xpos gives x=0.60, y=0.02 |

---

## Next Step: Forward Kinematics (A2)

Now we will compute the end-effector position from joint angles *ourselves* and compare against MuJoCo's `site_xpos` value.
