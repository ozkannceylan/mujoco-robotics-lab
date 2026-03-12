# B1: Trajectory Generation — Study Notes

## Two Fundamental Approaches

1. Joint-space trajectory
   - Joint angles are interpolated directly
   - Simple and inexpensive
   - The end-effector generally does not follow a straight line
2. Cartesian trajectory
   - End-effector targets are interpolated directly
   - Requires solving IK at every sample
   - The end-effector path stays much closer to a straight line

## What Is in This Repo?

`src/b1_trajectory_generation.py` contains:

- Cubic joint-space trajectory
- Quintic joint-space trajectory
- Cartesian straight-line interpolation + analytic IK branch selection
- CSV output generation
- Path and profile plots if `matplotlib` is available

## Boundary Conditions

Cubic:

```text
q(0) = q0
q(T) = qf
qd(0) = 0
qd(T) = 0
```

Quintic:

```text
q(0) = q0
q(T) = qf
qd(0) = qd(T) = 0
qdd(0) = qdd(T) = 0
```

Because of the additional zero-acceleration constraints, the quintic profile is smoother than the cubic.

## Outputs

The script generates:

- `docs/b1_cubic_joint_traj.csv`
- `docs/b1_quintic_joint_traj.csv`
- `docs/b1_cartesian_traj.csv`

If `matplotlib` is available:

- `docs/b1_trajectory_paths.png`
- `docs/b1_trajectory_profiles.png`

## Expected Behaviour

- The cubic trajectory should have zero velocity at start and end
- The quintic trajectory should have zero velocity *and* zero acceleration at start and end
- The Cartesian trajectory should stay very close to a straight line
- The joint-space trajectory will trace a visible curve between the same start and end points
