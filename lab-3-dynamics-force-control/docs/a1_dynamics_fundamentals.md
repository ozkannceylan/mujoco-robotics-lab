# A1: Dynamics Fundamentals

## Goal

Compute the rigid-body dynamics quantities M(q), C(q,q̇), g(q) using Pinocchio's analytical algorithms, and cross-validate against MuJoCo.

## Files

- Script: `src/a1_dynamics_fundamentals.py`
- Common: `src/lab3_common.py`
- Analytical model: `models/ur5e.urdf`
- Executed MuJoCo stack: Menagerie UR5e + mounted Robotiq loaded through `src/lab3_common.py`

## Theory

The equations of motion for a rigid-body manipulator:

```
M(q)q̈ + C(q,q̇)q̇ + g(q) = τ
```

Where:
- **M(q)** — Mass (inertia) matrix, symmetric positive definite (6×6)
- **C(q,q̇)** — Coriolis and centrifugal effects matrix (6×6)
- **g(q)** — Gravity vector (6×1)
- **τ** — Joint torques (6×1)

### Mass Matrix M(q)

Computed via Composite Rigid Body Algorithm (CRBA): `pin.crba(model, data, q)`. CRBA only fills the upper triangle — we symmetrize manually.

Properties verified:
- Symmetric: M = M^T
- Positive definite: all eigenvalues > 0
- Configuration dependent: diagonal elements change with q

### Coriolis Matrix C(q,q̇)

Computed via `pin.computeCoriolisMatrix(model, data, q, qd)`.

Key property: Ṁ - 2C is skew-symmetric (passivity property), verified numerically.

### Gravity Vector g(q)

Computed via `pin.computeGeneralizedGravity(model, data, q)`.

Cross-validated against MuJoCo's `qfrc_bias` (at zero velocity): max error `8.01e-06`.

## Cross-Validation Results

| Quantity | Method | Max Error |
|----------|--------|-----------|
| g(q) | Pinocchio vs MuJoCo `qfrc_bias` | `8.01e-06` |
| M(q) | Pinocchio CRBA vs MuJoCo `mj_fullM()` | `3.34e-05` |

## Key Design Decisions

### Menagerie actuator mapping
The canonical UR5e stack keeps the Menagerie arm actuators. Lab 3 maps desired joint torques into actuator controls in `lab3_common.py` instead of replacing the Menagerie actuators.

### Armature matching
MuJoCo armature must be reflected in Pinocchio through `model.armature[:]`, not `rotorInertia`.

### URDF-MuJoCo alignment
The Pinocchio URDF includes the mounted Robotiq payload as a fixed inertial load so MuJoCo and Pinocchio stay aligned closely enough for dynamics parity on the canonical stack.

## How to Run

```bash
python3 src/a1_dynamics_fundamentals.py
```

Outputs: mass matrix heatmap, gravity bar chart, gravity sweep over shoulder_lift.
