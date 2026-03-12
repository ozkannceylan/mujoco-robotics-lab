# A3: Jacobian — Study Notes

## What Is the Jacobian?

The Jacobian is the local linear map that converts joint-space velocities into end-effector velocities:

```text
x_dot = J(theta) * theta_dot
```

- `theta_dot = [theta1_dot, theta2_dot]`
- `x_dot = [vx, vy]`
- `J(theta)` is a 2×2 matrix derived from the partial derivatives of FK with respect to the joint angles.

## 2-Link Planar Analytic Jacobian

The effective second link length includes the site offset:

```text
L2_eff = L2 + 0.015
```

Because the `end_effector` site in the model sits 1.5 cm past the tip of link2.

```text
J = | -L1*sin(theta1) - L2_eff*sin(theta1 + theta2)   -L2_eff*sin(theta1 + theta2) |
    |  L1*cos(theta1) + L2_eff*cos(theta1 + theta2)    L2_eff*cos(theta1 + theta2) |
```

Interpretation:
- `J[0,0] = dx / dtheta1`
- `J[0,1] = dx / dtheta2`
- `J[1,0] = dy / dtheta1`
- `J[1,1] = dy / dtheta2`

## Verification Strategy

`src/a3_jacobian.py` includes three verification paths:

1. Analytic Jacobian vs finite-difference Jacobian comparison
2. `det(J)` sweep for singularity observation
3. Comparison with `mj_jacSite` if MuJoCo is installed

Determinant interpretation:
- `det(J) → 0` means the arm is approaching a singularity
- For a 2-link arm, typical singularity configurations are `theta2 ≈ 0` and `theta2 ≈ π`

## Generated Output

- `docs/a3_det_sweep.csv`
  - Determinant data along theta2 for a fixed theta1
  - Can be plotted from this CSV at any time

## Environment Note

If `numpy`, `matplotlib`, or `mujoco` are not installed, the code falls back to the standard library. The MuJoCo comparison is guarded and can be re-run once dependencies are available.
