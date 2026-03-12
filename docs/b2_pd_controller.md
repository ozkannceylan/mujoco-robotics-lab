# B2: PD Controller — Study Notes

## Control Law

The basic control used in this module:

```text
tau = Kp (q_des - q) + Kd (qd_des - qd) + g(q)
```

- `Kp`: response to position error
- `Kd`: damping on velocity error
- `g(q)`: gravity compensation term

## Approach in This Repo

`src/b2_pd_controller.py` has two layers:

1. Pure-Python fallback plant
   - Deterministic and dependency-free
   - Used to quickly measure overshoot and tracking error
2. Optional MuJoCo validation hook
   - Can connect to the real torque model once dependencies are available

## Tested Scenarios

1. Fixed target angle
   - Target: `q = [90°, -45°]`
   - Gravity compensation on vs off
2. Cubic joint trajectory tracking
   - Tracks the joint-space trajectory generated in `B1`
   - RMS tracking error measured

## Outputs

- `docs/b2_fixed_target_no_gc.csv`
- `docs/b2_fixed_target_gc.csv`
- `docs/b2_tracking_no_gc.csv`
- `docs/b2_tracking_gc.csv`

If `matplotlib` is available:

- `docs/b2_pd_controller.png`

## Expected Behaviour

- The PD controller reaches the target and settles
- Enabling gravity compensation should reduce both steady-state error and tracking error
- This fallback plant is not real MuJoCo dynamics; it is used to validate the control logic
