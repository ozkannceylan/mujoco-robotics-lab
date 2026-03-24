# Lab 3: Dynamics & Force Control

Status: complete and signed off on 2026-03-17.

Lab 3 now runs on the canonical project stack:

- MuJoCo Menagerie `universal_robots_ur5e`
- mounted MuJoCo Menagerie `robotiq_2f85`
- Pinocchio analytical model matched to that executed stack

## What Was Validated

- Dynamics parity between Pinocchio and MuJoCo on representative configurations
- Gravity compensation on the real UR5e + Robotiq stack
- Cartesian impedance control on the same stack
- Hybrid position-force control with intentional table contact
- Constant-force line tracing on the same table-contact scene

## Final Validation Summary

- Full Lab 3 test suite: `34 passed`
- Max gravity-vector mismatch: `8.01e-06`
- Max mass-matrix mismatch: `3.34e-05`
- Gravity compensation hold max error: `8.91e-06 rad`
- Gravity compensation perturbation final speed: `0.0134 rad/s`
- Hybrid force control mean force: `4.89 N`
- Hybrid force control in-band rate (`5 +/- 1 N`): `99.96%`
- Hybrid force control max XY error: `3.60 mm`
- Constant-force trace in-band rate (`5 +/- 1 N`): `94.07%`
- Constant-force trace max XY error: `1.70 mm`

## Canonical Artifacts

- Validation video: `media/lab3_validation_real_stack.mp4`
- Dynamics plots: `media/mass_matrix_heatmap.png`, `media/gravity_vector_bar.png`, `media/gravity_sweep.png`
- Gravity compensation plots: `media/gravity_comp_hold.png`, `media/gravity_comp_perturb.png`
- Impedance plots: `media/impedance_3d_tracking.png`, `media/impedance_6d_tracking.png`
- Force-control plots: `media/hybrid_force_control.png`, `media/capstone_line_trace.png`

## Implementation Notes

- `src/lab3_common.py` builds the MuJoCo scene directly from Menagerie UR5e + Robotiq assets.
- The mounted Robotiq payload is reflected in `models/ur5e.urdf` so Pinocchio matches the executed MuJoCo stack closely enough for dynamics work.
- Arm torques are applied through the Menagerie arm actuators by mapping desired torques into actuator controls, rather than by replacing the Menagerie actuators.
- In Lab 3 force control, touching the table is intentional. The task is to establish gentle contact, then regulate approximately `5 N` downward force while holding or tracing in XY.

## Key Modules

| File | Role |
|------|------|
| `src/lab3_common.py` | Canonical model loading, IDs, actuator mapping, IK helpers |
| `src/a1_dynamics_fundamentals.py` | `M(q)`, `C(q,qdot)`, `g(q)` and parity checks |
| `src/a2_gravity_compensation.py` | `tau = g(q)` validation |
| `src/b1_impedance_controller.py` | Translational and 6D impedance control |
| `src/b2_compliance_demo.py` | Compliance comparison demos |
| `src/c1_force_control.py` | Hybrid position-force control |
| `src/c2_line_trace.py` | Constant-force line trace |
| `src/record_lab3_validation.py` | Validation video recorder |

## How To Run

```bash
python3 -m pytest lab-3-dynamics-force-control/tests -q
python3 lab-3-dynamics-force-control/src/a1_dynamics_fundamentals.py
python3 lab-3-dynamics-force-control/src/a2_gravity_compensation.py
python3 lab-3-dynamics-force-control/src/b1_impedance_controller.py
python3 lab-3-dynamics-force-control/src/c1_force_control.py
python3 lab-3-dynamics-force-control/src/c2_line_trace.py
python3 lab-3-dynamics-force-control/src/record_lab3_validation.py
```

## Notes

- Native MuJoCo OpenGL rendering is not available in this environment. The validation video still shows the real MuJoCo simulation state by projecting actual MuJoCo geom poses frame by frame and overlaying the measured force and tracking signals.
- Lab 3 is complete. Labs 4 and 5 still need separate review and sign-off.
