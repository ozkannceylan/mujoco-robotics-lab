# A5: Dynamics Basics — Study Notes

## Fundamental Equation

The robot dynamics equation:

```text
tau = M(q) qdd + C(q, qd) qd + g(q)
```

- `tau`: joint torques
- `M(q)`: inertia (mass) matrix
- `C(q, qd) qd`: Coriolis + centrifugal effects
- `g(q)`: gravity-induced joint torques

MuJoCo computes these terms internally at every step. The goal of this module is not to derive them from scratch but to read them through the API.

## What Do We Read From MuJoCo?

- `data.qfrc_bias`
  - In practice, this gives `C(q,qd) qd + g(q)` combined
- `data.qM`
  - The packed inertia matrix data
  - Expanded to the full `M(q)` matrix via `mj_fullM(...)`

## Gravity Note for This Model

The 2-link arm in this repo operates in the XY plane with joint axes along `z`. As a consequence:

- Gravity along `z` produces little to no meaningful torque about the joint axes
- To observe gravity effects clearly, it is more instructive to set gravity to an in-plane direction

Therefore `src/a5_dynamics_basics.py` compares three cases:

1. Gravity off
2. Gravity along `z`
3. Gravity along `y`

## Configuration Dependence

`M(q)` is not constant. As the links change pose, their mass distribution and inertia axes shift in space. This is why the torque needed to produce a given acceleration depends on the configuration.

The script compares the full inertia matrix for two poses:

- `q = [0°, 0°]`
- `q = [90°, -90°]`

## Running

```bash
python3 src/a5_dynamics_basics.py
```

Generated outputs:

- `docs/a5_bias_cases.csv`
- `docs/a5_mass_matrix_cases.csv`
