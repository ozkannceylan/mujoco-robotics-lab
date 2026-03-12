# A4: Inverse Kinematics — Study Notes

## Problem

The IK question:

```text
Given:  (x_target, y_target)
Find:   (theta1, theta2)
```

In this repo the target is the `end_effector` site position. Therefore the effective second link length is:

```text
L2_eff = 0.3 + 0.015 = 0.315 m
```

The inner reachability radius is no longer zero:

```text
r_min = |L1 - L2_eff| = 0.015 m
r_max = L1 + L2_eff   = 0.615 m
```

## Analytic IK

The closed-form solution yields two branches:

```text
cos(theta2) = (x² + y² - L1² - L2_eff²) / (2 L1 L2_eff)
theta2 = ± acos(cos(theta2))
theta1 = atan2(y, x) - atan2(L2_eff sin(theta2), L1 + L2_eff cos(theta2))
```

Notes:
- `+acos(...)` and `-acos(...)` give two distinct elbow configurations
- In this repo the branches are labelled `elbow_down` and `elbow_up`

## Numeric IK

`src/a4_inverse_kinematics.py` implements two iterative methods:

1. `pinv`
   - Uses the Jacobian inverse in the full-rank square case
   - Sensitive near singularities
2. `dls`
   - `J^T (J J^T + λ² I)⁻¹ e`
   - More stable near singularities

## Verification

The script produces:

- Two analytic branches with FK error check for a single target
- A 20-target benchmark comparing `pinv` and `dls`
- A separate stress test near singularity
- Optional final check against MuJoCo `site_xpos` if available

Output file:

- `docs/a4_ik_benchmark.csv`

## Expected Behaviour

- Analytic IK should return two solutions with very small FK error for reachable targets
- `pinv` converges quickly with a good initial guess
- `dls` is safer with an ill-conditioned Jacobian

## Results From This Session

- Example target `(0.34, 0.28)` produced two analytic solutions:
  - `elbow_down`: `theta1=-6.166°`, `theta2=88.552°`
  - `elbow_up`: `theta1=85.111°`, `theta2=-88.552°`
- FK error for both analytic solutions: `0.0`
- 20-target benchmark:
  - `pinv`: `100%` success, `12.50` average iterations
  - `dls`: `100%` success, `13.45` average iterations
- Singularity stress test:
  - Initial guess: `(theta1, theta2) = (0, 0)`
  - Target: `(0.612, 0.020)`
  - `pinv` failed on the first step due to a singular Jacobian
  - `dls` converged in `161` iterations with `9.831e-06` error
