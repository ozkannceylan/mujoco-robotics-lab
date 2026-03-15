# A1: Dynamics Fundamentals

## Goal

Compute the rigid-body dynamics quantities M(q), C(q,q̇), g(q) using Pinocchio's analytical algorithms, and cross-validate against MuJoCo.

## Files

- Script: `src/a1_dynamics_fundamentals.py`
- Common: `src/lab3_common.py`
- Model: `models/ur5e.xml` (torque-mode), `models/ur5e.urdf`

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

Cross-validated against MuJoCo's `qfrc_bias` (at zero velocity): max error < 0.0005 Nm.

## Cross-Validation Results

| Quantity | Method | Max Error |
|----------|--------|-----------|
| g(q) | pin vs mj qfrc_bias | < 0.0005 Nm |
| M(q) | pin CRBA vs mj fullM | < 0.00004 |

## Key Design Decisions

### Torque-mode actuators
Lab 3 uses `<motor>` actuators (gain=1, bias=0) for direct torque control, unlike Lab 2's position servos. This enables computed-torque and impedance control.

### Armature matching
MuJoCo's `armature=0.01` adds to the mass matrix diagonal. In Pinocchio, `model.armature[:] = 0.01` (not `rotorInertia`) achieves the same effect.

### URDF-MJCF alignment
The URDF was rebuilt from scratch by extracting body positions, quaternions, and inertias from the MJCF. Key corrections: `body_iquat` rotation for inertia tensors (upper_arm, forearm), which created cross-terms.

## How to Run

```bash
python3 src/a1_dynamics_fundamentals.py
```

Outputs: mass matrix heatmap, gravity bar chart, gravity sweep over shoulder_lift.
