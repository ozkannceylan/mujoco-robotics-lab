# C1: Cartesian Square Drawing — Study Notes

## Goal

Integrate all previous modules (A2–B2) into a single executable script that makes
the 2-link planar robot draw a square in the XY plane using **torque control**
in a MuJoCo environment.

## Pipeline

```text
square corners ──► quintic Cartesian ──► analytic IK ──► Jacobian J⁻¹ ──► computed torque ──► MuJoCo sim ──► viewer trail
                   trajectory (B1)        (A4)             (A3)            M, bias (A5)
                                                                          Kp/Kd (B2)
```

Step by step:

1. **Path Planning** — The 4 corners of the square are defined. Along each side,
   `quintic_profile` interpolates x(t) and y(t) independently. The quintic profile
   guarantees zero velocity and acceleration at the start and end of each segment
   (smooth stop/start).

2. **Inverse Kinematics** — At every timestep, `analytic_ik` is called for the (x, y)
   target. The IK branch closest to the previous solution is selected (branch
   continuity). For the first point, elbow-down (q2 < 0) is preferred — this keeps
   the arm away from the base platform.

3. **Velocity Mapping via Jacobian** — The desired Cartesian velocity `[ẋ, ẏ]` is
   converted to joint velocities using the Jacobian inverse: `q̇_des = J⁻¹ · [ẋ, ẏ]`.
   Near singularity (det(J) < 1e-8) a safe zero velocity is returned — the quintic
   profile already produces zero velocity at corners.

4. **Joint Acceleration (Numerical Differentiation)** — `q̈_des` is computed via
   central finite differences: `q̈[i] = (q̇[i+1] − q̇[i−1]) / (2·dt)`. Simpler than
   an analytic J̇ computation and accurate enough with dt = 0.002 s.

5. **Computed Torque Control** — The equation of motion:

   ```
   M(q)·q̈ + C(q,q̇)·q̇ + g(q) = τ_applied + τ_passive
   ```

   In MuJoCo terms:
   - `M` → `mj_fullM(model, M, data.qM)` — configuration-dependent mass matrix
   - `qfrc_bias` → Coriolis + gravity terms
   - `qfrc_passive` → joint damping (−d·q̇)
   - `ctrl` → direct motor torque (N·m)

   Control law:
   ```
   u = q̈_des + Kp·(q_des − q) + Kd·(q̇_des − q̇)
   τ = M·u + qfrc_bias − qfrc_passive
   ```

   This cancels all nonlinearity and reduces the error dynamics to:
   ```
   ë + Kd·ė + Kp·e = 0
   ```
   Critically damped: `Kp = ωn² = 400`, `Kd = 2·ωn = 40` (natural frequency 20 rad/s).

6. **MuJoCo Simulation** — `mujoco.viewer.launch_passive` provides real-time
   visualisation. Each frame renders:
   - **Green square** — desired path (4 capsule lines via `user_scn` geoms)
   - **Red trail** — end-effector positions (a sphere every 5 steps)
   - **Yellow marker** — current EE position

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SQUARE_CENTER` | (0.30, 0.30) | Centre of the square (m) |
| `SQUARE_SIDE` | 0.10 | Side length (m) |
| `SEGMENT_DURATION` | 2.0 | Duration per side (s) |
| `KP` | [400, 400] | Proportional gain |
| `KD` | [40, 40] | Derivative gain |
| `TORQUE_LIMIT` | 5.0 | Motor torque limit (N·m) |
| `TRAIL_DECIMATE` | 5 | Trail point interval (steps) |

## Gain Design

With computed torque control the error dynamics are linearised:

```
ë + Kd·ė + Kp·e = 0
```

The characteristic equation of this second-order system: `s² + Kd·s + Kp = 0`

- Natural frequency: `ωn = √Kp = √400 = 20 rad/s`
- Damping ratio: `ζ = Kd / (2·ωn) = 40 / 40 = 1.0` → **critical damping**
- Settling time: `ts ≈ 4 / (ζ·ωn) = 0.2 s`

Critical damping means:
- No overshoot
- Fastest possible non-oscillatory convergence
- Minimises tracking error

## Issues Discovered and Solutions

### 1. MuJoCo Angle Unit

**Problem:** The model XML specified `range="-3.14 3.14"`. MuJoCo defaults to degrees,
so this was interpreted as ±3.14° (±0.055 rad). At normal operating angles (±1–2 rad)
the joint limit constraints kicked in and destabilised the controller.

**Solution:** Added `<compiler angle="radian"/>` to the XML. The range is now correctly
interpreted as ±3.14 radians.

### 2. Decorative Geom Collision

**Problem:** The base platform and joint visualiser geoms were defined with `contype=1`.
At certain configurations (especially elbow-up, q2 > 1.5 rad) they overlapped with
the link1 capsule, generating large constraint forces.

**Solution:** The script disables all geom collisions at runtime:
```python
model.geom_contype[:] = 0
model.geom_conaffinity[:] = 0
```

### 3. IK Branch Selection

**Problem:** For the first point, the branch with the smallest `|q2|` was selected.
Near the square's corners both branches have `|q2| ≈ 1.92 rad` — the positive branch
folds the arm toward the base, causing collision.

**Solution:** The first point now prefers **elbow-down** (q2 < 0), which opens the arm
away from the base.

## Verification Results

Headless simulation (4001 steps, 8.0 seconds):

| Metric | Value |
|--------|-------|
| RMS Cartesian error | 0.008 mm |
| Max Cartesian error | 0.013 mm |
| Final Cartesian error | 0.005 mm |
| Max applied torque | 0.077 N·m |
| Torque saturation | None |

These results demonstrate the effectiveness of computed torque control:
- Sub-millimetre tracking accuracy
- Less than 2% of the torque limit is used
- Smooth, oscillation-free motion

## Outputs

- `src/c1_draw_square.py` — main script
- Real-time animation in the MuJoCo viewer (green target square, red trail, yellow EE marker)

## Modules Used

| Module | Function Used | Purpose |
|--------|--------------|---------|
| A2 (`a3_jacobian.py`) | `fk_endeffector` | FK verification |
| A3 (`a3_jacobian.py`) | `analytic_jacobian` | J⁻¹ velocity mapping |
| A4 (`a4_inverse_kinematics.py`) | `analytic_ik` | Cartesian → joint-space |
| A5 (MuJoCo runtime) | `mj_fullM`, `data.qfrc_bias` | M(q), bias terms |
| B1 (`b1_trajectory_generation.py`) | `quintic_profile` | Smooth Cartesian path |
| B2 (control principle) | Kp/Kd error terms | PD inside computed torque |

## Running

```bash
cd src && python3 c1_draw_square.py
```

The MuJoCo viewer window opens. As the robot draws the square:
- **Green lines** = target square
- **Red dots** = end-effector trail
- **Yellow sphere** = current EE position

Closing the window prints summary metrics to the terminal.
