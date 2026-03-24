# Lab 3: Dynamics & Force Control — Architecture

Completion date: 2026-03-17

## Architecture Status

The canonical Lab 3 architecture is now implemented.

- MuJoCo executes the Menagerie UR5e with mounted Robotiq 2F-85
- Pinocchio analyzes a matched URDF with the mounted gripper payload
- All Lab 3 controllers run on that same hardware baseline

## Module Map

```
lab-3-dynamics-force-control/
├── src/
│   ├── lab3_common.py              # Canonical loaders, IDs, actuator mapping, IK helpers
│   ├── a1_dynamics_fundamentals.py # M(q), C(q,qdot), g(q), parity checks, plots
│   ├── a2_gravity_compensation.py  # tau = g(q) demos and perturbation checks
│   ├── b1_impedance_controller.py  # Translational and 6D impedance controllers
│   ├── b2_compliance_demo.py       # Soft/medium/stiff compliance comparisons
│   ├── c1_force_control.py         # Hybrid position-force controller + contact interpretation
│   ├── c2_line_trace.py            # Constant-force line tracing capstone
│   └── record_lab3_validation.py   # Lab 3 validation video generator
├── models/
│   └── ur5e.urdf                   # Pinocchio-side UR5e + fixed Robotiq payload model
├── tests/
│   ├── test_dynamics.py
│   ├── test_gravity_comp.py
│   ├── test_impedance.py
│   └── test_force_control.py
├── docs/
├── docs-turkish/
├── media/
└── README.md
```

## Data Flow

```
MuJoCo Menagerie UR5e + Robotiq scene
        │
        ├─ q, qdot, contacts, actuator state
        │
        ▼
lab3_common.py
        │
        ├─ programmatic scene loading
        ├─ body / geom / site IDs
        ├─ actuator torque-to-ctrl mapping
        └─ frame and IK helpers
        │
        ▼
Pinocchio matched model
        │
        ├─ FK
        ├─ Jacobians
        ├─ M(q), C(q,qdot), g(q)
        └─ frame placements
        │
        ▼
Controllers
        │
        ├─ gravity comp: tau = g(q)
        ├─ impedance: tau = J^T F + g(q)
        └─ hybrid force: tau = J^T(F_xy + F_z) + g(q)
        │
        ▼
MuJoCo step and contact telemetry
        │
        ▼
plots, tests, validation video
```

## Key Interfaces

### `lab3_common.py`

```python
def load_mujoco_model(scene_path: Path | None = None) -> tuple: ...
def load_pinocchio_model(urdf_path: Path | None = None) -> tuple: ...
def solve_dls_ik(...) -> np.ndarray | None: ...
def arm_torques_to_ctrl(...) -> np.ndarray: ...
def apply_arm_torques(...) -> None: ...
```

Responsibilities:

- load only the canonical UR5e + Robotiq baseline
- keep MuJoCo and Pinocchio frames aligned
- map desired arm torques into the Menagerie arm actuators

### `a1_dynamics_fundamentals.py`

Responsibilities:

- compute `M(q)`, `C(q,qdot)`, `g(q)`
- cross-validate gravity and mass matrix against MuJoCo

### `a2_gravity_compensation.py`

Responsibilities:

- apply `tau = g(q)` on the canonical stack
- validate hold behavior and perturbation settling

### `b1_impedance_controller.py`

Responsibilities:

- provide translational and 6D impedance control
- use `pin.log3(...)` orientation error in the canonical 6D path

### `c1_force_control.py`

Responsibilities:

- read table contact through `mj_contactForce()`
- include `wrist_3_link` and mounted gripper bodies in the contact set
- regulate intentional table contact around the target normal force

### `c2_line_trace.py`

Responsibilities:

- establish table contact
- maintain force while tracing the commanded XY line

### `record_lab3_validation.py`

Responsibilities:

- run the canonical Lab 3 validation sequence
- save the sign-off video into `media/`
- overlay measured force and tracking values over the MuJoCo scene

## Validation Rules

- dynamics parity must be checked on representative configurations
- gravity compensation must satisfy hold and perturbation criteria
- contact detection must be validated against named MuJoCo contact pairs
- every sign-off claim must have a generated artifact behind it
