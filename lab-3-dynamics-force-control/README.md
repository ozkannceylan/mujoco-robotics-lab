# Lab 3: Dynamics & Force Control

A self-contained 6-DOF dynamics and force-control lab built on top of Lab 2. Lab 3 keeps the same Pinocchio-as-brain / MuJoCo-as-simulator split, but the focus shifts from purely kinematic commands to torque-level control: rigid-body dynamics, gravity compensation, Cartesian impedance, and hybrid position-force control on a real contact scene.

## Showcase

![Capstone Line Trace](media/capstone_line_trace.png)

> The capstone demo holds a constant 5 N downward force against a table while tracing a straight line in the XY plane. Position errors stay sub-2 mm and the contact force stays inside `5 ± 1 N` for 94% of the trace.

## Key Results

| Metric | Value |
|---|---|
| Test suite | **34 passed** (4 files, dynamics / gravity / impedance / force) |
| Pinocchio vs MuJoCo gravity vector mismatch | 8.01e-06 |
| Pinocchio vs MuJoCo mass-matrix mismatch | 3.34e-05 |
| Gravity-compensation hold (max joint error) | 8.91e-06 rad |
| Gravity-compensation perturbation final speed | 0.0134 rad/s |
| Hybrid force-control mean force | 4.89 N |
| Hybrid force-control in-band rate (5 ± 1 N) | 99.96 % |
| Hybrid force-control max XY error | 3.60 mm |
| Line-trace in-band rate (5 ± 1 N) | 94.07 % |
| Line-trace max XY error | 1.70 mm |

---

## Skills Demonstrated

- **Rigid-body dynamics**: `M(q)`, `C(q,q̇)`, `g(q)` from Pinocchio (RNEA, CRBA) cross-validated against MuJoCo `qfrc_bias` and `mj_fullM` to sub-1e-4 precision.
- **Gravity compensation**: `τ = g(q)` holds any UR5e + Robotiq configuration hands-free; verified under perturbation.
- **Cartesian impedance control**: translational and full 6-D variants with tunable `K`/`D` gains, using `pin.log3` for large-angle-safe orientation error.
- **Hybrid position-force control**: independent control of XY position (impedance) and Z force (admissive PI on measured contact force) on a real MuJoCo table-contact scene.
- **Contact reading**: per-contact forces extracted via `mj_contactForce` over the full EE contact set (wrist_3_link + every mounted Robotiq body).
- **Real-stack alignment**: Pinocchio URDF rebuilt around the mounted Robotiq payload so analytic and simulated dynamics match the same physical robot.

---

## Architecture

```text
Task Spec (desired pose + desired force)
        │
        ▼
┌───────────────────────────┐
│  Impedance / Force Control │
│  F = K·Δx + D·Δẋ + F_des   │
└──────────┬────────────────┘
           │ Cartesian wrench
           ▼
┌───────────────────────────┐
│  Torque Computation        │
│  τ = J^T·F + g(q)          │
│  (Pinocchio RNEA / J)      │
└──────────┬────────────────┘
           │ joint torques
           ▼
┌───────────────────────────┐
│  MuJoCo Menagerie UR5e     │
│  + mounted Robotiq 2F-85   │
│  → q, q̇, contacts          │
└───────────────────────────┘
```

Lab 3 runs on the canonical project hardware stack: MuJoCo Menagerie `universal_robots_ur5e` with mounted `robotiq_2f85`, plus a Pinocchio model whose payload matches the executed stack.

---

## Modules

### A — Dynamics Fundamentals

| Module | Topic | Script |
|---|---|---|
| A1 | `M(q)`, `C(q,q̇)`, `g(q)` and Pinocchio↔MuJoCo parity | `src/a1_dynamics_fundamentals.py` |
| A2 | Gravity compensation (`τ = g(q)`) | `src/a2_gravity_compensation.py` |

### B — Cartesian Impedance

| Module | Topic | Script |
|---|---|---|
| B1 | Translational and 6-D Cartesian impedance | `src/b1_impedance_controller.py` |
| B2 | Stiffness sweep / compliance comparison | `src/b2_compliance_demo.py` |

### C — Force Control

| Module | Topic | Script |
|---|---|---|
| C1 | Hybrid position-force control against the table | `src/c1_force_control.py` |
| C2 | Constant-force line-trace capstone | `src/c2_line_trace.py` |

### Shared Utilities

| File | Purpose |
|---|---|
| `src/lab3_common.py` | Canonical model loaders, IDs, actuator mapping, IK helpers |
| `src/record_lab3_validation.py` | Headless validation video recorder |

---

## Quick Start

```bash
# From the repository root
pip install mujoco numpy pinocchio scipy "imageio[ffmpeg]" matplotlib

# Run the full test suite (34 tests)
python3 -m pytest lab-3-dynamics-force-control/tests -q

# Module-by-module walkthrough
python3 lab-3-dynamics-force-control/src/a1_dynamics_fundamentals.py
python3 lab-3-dynamics-force-control/src/a2_gravity_compensation.py
python3 lab-3-dynamics-force-control/src/b1_impedance_controller.py
python3 lab-3-dynamics-force-control/src/c1_force_control.py
python3 lab-3-dynamics-force-control/src/c2_line_trace.py

# Re-record the validation video
python3 lab-3-dynamics-force-control/src/record_lab3_validation.py
```

---

## Structure

```text
lab-3-dynamics-force-control/
├── src/              Source scripts (A1, A2, B1, B2, C1, C2) and shared utilities
├── models/           UR5e URDF (Pinocchio) + MuJoCo scenes (torque actuators, table contact)
├── docs/             English study notes
├── docs-turkish/     Turkish study notes
├── media/            Plots and recorded validation video
├── tests/            Pytest suite (34 tests across 4 files)
└── tasks/            PLAN / ARCHITECTURE / TODO / LESSONS
```

---

## Documentation

| Topic | English | Turkish |
|---|---|---|
| A1 — Dynamics fundamentals | [Dynamics Fundamentals](docs/a1_dynamics_fundamentals.md) | [Dinamik Temelleri](docs-turkish/a1_dinamik_temelleri.md) |
| A2 — Gravity compensation | [Gravity Compensation](docs/a2_gravity_compensation.md) | [Yerçekimi Kompanzasyonu](docs-turkish/a2_yercekimi_kompanzasyonu.md) |
| B1 — Impedance control | [Impedance Control](docs/b1_impedance_control.md) | [Empedans Kontrolü](docs-turkish/b1_empedans_kontrolu.md) |
| C1 — Force control | [Force Control](docs/c1_force_control.md) | [Kuvvet Kontrolü](docs-turkish/c1_kuvvet_kontrolu.md) |

---

## Media

- Capstone line-trace plot: [`media/capstone_line_trace.png`](media/capstone_line_trace.png)
- Hybrid force-control plot: [`media/hybrid_force_control.png`](media/hybrid_force_control.png)
- Cartesian impedance tracking: [`media/impedance_3d_tracking.png`](media/impedance_3d_tracking.png), [`media/impedance_6d_tracking.png`](media/impedance_6d_tracking.png)
- Compliance comparison: [`media/compliance_comparison.png`](media/compliance_comparison.png), [`media/compliance_per_axis.png`](media/compliance_per_axis.png)
- Gravity-compensation hold / perturb: [`media/gravity_comp_hold.png`](media/gravity_comp_hold.png), [`media/gravity_comp_perturb.png`](media/gravity_comp_perturb.png)
- Dynamics parity: [`media/mass_matrix_heatmap.png`](media/mass_matrix_heatmap.png), [`media/gravity_vector_bar.png`](media/gravity_vector_bar.png), [`media/gravity_sweep.png`](media/gravity_sweep.png)
- Validation video: [`media/lab3_validation_real_stack.mp4`](media/lab3_validation_real_stack.mp4)

---

## Notes

- In Lab 3, touching the table is intentional. The task is to establish gentle contact, then regulate ~5 N downward force while holding or tracing in XY.
- Arm torques are applied through the Menagerie arm actuators by mapping desired torques into actuator controls, preserving the real-robot actuator model rather than replacing it.
- Native MuJoCo OpenGL rendering was unavailable on the sign-off machine. `record_lab3_validation.py` still visualises the actual MuJoCo simulation state by projecting real geom poses frame by frame and overlaying the measured force and tracking signals.

---

## License

The Lab 3 source code and original documentation are covered by the repository root [Apache-2.0 license](../LICENSE).

Bundled robot description packages and model assets in [`models/`](models/) and the Menagerie assets reused from Lab 2 keep their upstream licenses. See the repository root [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the exact carve-outs, including Universal Robots mesh directories that are not fully OSI-open-source.
