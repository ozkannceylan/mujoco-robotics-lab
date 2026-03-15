# Lab 3: Dynamics & Force Control

UR5e dynamics fundamentals, Cartesian impedance control, and hybrid position-force control using Pinocchio + MuJoCo.

## Objectives

1. Compute M(q), C(q,q̇), g(q) and cross-validate Pinocchio vs MuJoCo
2. Implement gravity compensation (τ = g(q))
3. Implement Cartesian impedance control with tunable compliance
4. Implement hybrid position-force control (position in XY, force in Z)
5. Capstone: constant-force line tracing on a table surface

## Key Results

| Controller | Metric | Result |
|------------|--------|--------|
| Gravity comp | Position drift (3s) | 0.0006 rad |
| Impedance (3D) | Position error | < 1 mm |
| Impedance (6D) | Orientation error | < 1° |
| Force (static) | Force accuracy | 4.95 ± 0.09 N (100% in ±1N) |
| Force (line trace) | Force accuracy | 4.99 ± 0.33 N (98.1% in ±1N) |
| Force (line trace) | XY tracking | 1.96 mm mean (100% < 5mm) |

## Architecture

```
Pinocchio (analytical)          MuJoCo (simulation)
├── FK, Jacobian                ├── Physics stepping
├── M(q), C(q,q̇), g(q)        ├── Contact detection
├── CRBA, RNEA                  ├── mj_contactForce()
└── IK (for start configs)      └── Torque-mode actuators
```

## Module Map

| File | Description |
|------|-------------|
| `src/lab3_common.py` | Paths, constants, model loading, utilities |
| `src/a1_dynamics_fundamentals.py` | M(q), C(q,q̇), g(q) computation & cross-validation |
| `src/a2_gravity_compensation.py` | Gravity compensation controller |
| `src/b1_impedance_controller.py` | Cartesian impedance (3D + 6D) |
| `src/b2_compliance_demo.py` | Soft/medium/stiff compliance comparison |
| `src/c1_force_control.py` | Hybrid position-force controller |
| `src/c2_line_trace.py` | Capstone: constant-force line tracing |

## How to Run

```bash
# Phase 1: Dynamics
python3 src/a1_dynamics_fundamentals.py
python3 src/a2_gravity_compensation.py

# Phase 2: Impedance
python3 src/b1_impedance_controller.py
python3 src/b2_compliance_demo.py

# Phase 3: Force Control
python3 src/c1_force_control.py
python3 src/c2_line_trace.py

# Tests (34 total)
python3 -m pytest tests/ -v
```

## Dependencies

- Python 3.10+
- MuJoCo
- Pinocchio (`pip install pin`)
- NumPy
- Matplotlib

## Documentation

- English: `docs/`
- Turkish: `docs-turkish/`
- Blog: `blog/lab_03_dynamics_force_control.md`
