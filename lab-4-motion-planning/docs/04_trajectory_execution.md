# Lab 4: Trajectory Execution

## Final Execution Path

Trajectory execution on Lab 4 now matches the canonical stack used by Lab 3:

- MuJoCo executes Menagerie UR5e + mounted Robotiq 2F-85
- Pinocchio provides `g(q)` for gravity compensation
- desired arm torques are mapped through the Menagerie actuator model before
  being applied

## Control Law

Execution uses joint-space PD plus gravity compensation:

```text
tau = Kp * (q_d - q) + Kd * (qd_d - qd) + g(q)
```

with default gains:

- `Kp = 400`
- `Kd = 40`

## Important Change

The old Lab 4 execution path assumed direct torque writes into
`mj_data.ctrl[:6]`. On the canonical stack this is not correct. The final
implementation uses the same actuator mapping approach as Lab 3 so that the
executed commands respect the Menagerie actuator model.

## Logged Outputs

`execute_trajectory(...)` returns:

- `time`
- `q_actual`
- `q_desired`
- `tau`
- `ee_pos`

## Final Validation Metrics

Standard capstone scene:

- RMS tracking error: `0.0125 rad`
- final position error: `0.0016 rad`

Blocked-path validation scene:

- RMS tracking error: `0.0124 rad`
- final position error: `0.0041 rad`
- direct path free: `False`

These are within the Lab 4 success thresholds enforced by the test suite.
