# A2: Gravity Compensation

## Goal

Implement the simplest dynamics-based controller: τ = g(q). The robot should hold any configuration against gravity with minimal drift.

## Files

- Script: `src/a2_gravity_compensation.py`
- Tests: `tests/test_gravity_comp.py`

## Theory

Gravity compensation cancels the gravitational torque at each timestep:

```
τ = g(q)
```

Where g(q) is computed by Pinocchio's `computeGeneralizedGravity()`. This is the foundation for all subsequent controllers — impedance and force control both add gravity compensation as a baseline.

### Why it works

When τ = g(q), the equations of motion reduce to:

```
M(q)q̈ + C(q,q̇)q̇ = 0
```

With joint damping (1.0 Nm·s/rad in our model), any velocity decays exponentially and the robot settles at its initial configuration.

## Results

| Configuration | Max drift (3s) |
|--------------|----------------|
| Q_HOME | 0.0006 rad |
| Q_ZEROS | 0.002 rad |
| Arbitrary | < 0.01 rad |

After a 20 Nm impulse perturbation on the shoulder joint, the arm settles back to its original position within ~2 seconds.

## How to Run

```bash
python3 src/a2_gravity_compensation.py
```

Outputs: joint position vs time plots, perturbation response plot.
