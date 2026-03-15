# B1: Cartesian Impedance Control

## Goal

Implement task-space impedance control: make the EE behave like a spring-damper system in Cartesian space, with tunable compliance.

## Files

- Script: `src/b1_impedance_controller.py`
- Compliance demo: `src/b2_compliance_demo.py`
- Tests: `tests/test_impedance.py`

## Theory

### Impedance Control Law

The impedance controller maps Cartesian spring-damper behavior to joint torques:

```
F = K_p · (x_d - x) + K_d · (ẋ_d - ẋ)
τ = J^T · F + g(q)
```

Where:
- K_p: Cartesian stiffness (N/m or Nm/rad)
- K_d: Cartesian damping (N·s/m or Nm·s/rad)
- J: Jacobian in LOCAL_WORLD_ALIGNED frame
- g(q): gravity compensation

### 6-DOF Extension

For full pose control (position + orientation):

```
F_6d = [K_p_lin · e_pos; K_p_rot · e_rot] + [K_d_lin · ė_pos; K_d_rot · ė_rot]
τ = J_6d^T · F_6d + g(q)
```

Orientation error uses skew-symmetric extraction:
```
e_R = 0.5 · vee(R_d^T · R - R^T · R_d)
```

This gives a 3-vector proportional to the angle-axis rotation error, valid for small angles.

### Compliance Tuning

The stiffness K_p determines how "stiff" or "compliant" the EE feels:

| K_p (N/m) | Behavior | Deflection (40N) |
|-----------|----------|-------------------|
| 100 | Soft | 104 mm |
| 500 | Medium | 43 mm |
| 2000 | Stiff | 17 mm |

## Results

- **Position tracking**: < 1mm error to Cartesian targets
- **Orientation tracking**: < 1° error
- **Perturbation recovery**: returns to target after external force removed
- **Stiffness scaling**: deflection inversely proportional to K_p (verified monotonic)

## Architecture

```
              ┌──────────────┐
  x_d, R_d ──▶│  Impedance   │──▶ τ ──▶ MuJoCo
  x, R    ◀───│  Controller  │         ▲
  q, q̇   ◀───│  (Pinocchio) │         │
              └──────────────┘         │
                     ▲                  │
                     │ FK, J, g(q)      │
                     └──────────────────┘
```

## How to Run

```bash
# Impedance controller demo (3D + 6D with perturbation)
python3 src/b1_impedance_controller.py

# Compliance comparison (soft/medium/stiff)
python3 src/b2_compliance_demo.py
```
