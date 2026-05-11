# C1: Hybrid Position-Force Control

## Goal

Implement hybrid control: position control in XY plane, PI force control in Z. Maintain constant contact force while controlling EE position on a table surface.

## Files

- Force controller: `src/c1_force_control.py`
- Line trace capstone: `src/c2_line_trace.py`
- Scene: Menagerie UR5e + mounted Robotiq table-contact scene loaded through `src/lab3_common.py`
- Tests: `tests/test_force_control.py`

## Theory

### Hybrid Position-Force Control

The key insight: in constrained tasks (e.g., polishing, assembly), some Cartesian directions should be position-controlled and others force-controlled. Selection matrices partition the task space:

```
S_p = diag(1, 1, 0)  — position control in XY
S_f = diag(0, 0, 1)  — force control in Z
```

### Control Law

**Position (XY):**
```
F_p = K_p · S_p · (x_d - x) + K_d · S_p · (ẋ_d - ẋ)
```

**Force (Z) with PI + velocity damping:**
```
e_f = F_desired - F_measured
F_f = -(K_fp · e_f + K_fi · ∫e_f dt) - K_dz · ẋ_z
```

**Combined:**
```
τ = J^T · (F_p + F_f) + g(q)
```

The PI force controller ensures zero steady-state force error. Z-velocity damping (K_dz=30) prevents contact chattering.

### Contact Force Measurement

Forces are read via `mj_contactForce()`, filtering for contacts between `table_top` and the real executed end-effector contact set: `wrist_3_link` plus the mounted Robotiq bodies. Raw forces are smoothed with an EMA low-pass filter (α=0.2).

This is more reliable than MuJoCo's `<force>` sensor, which measures all constraint forces on the body (including articulation forces), not just contact.

### Intentional contact

In Lab 3, contacting the table is intentional. The goal is to establish gentle contact and then regulate approximately `5 N` normal force while holding or tracing in XY.

## Results

### Static Force Hold

| Metric | Value |
|--------|-------|
| Target force | 5.0 N |
| Mean force | 4.89 N |
| Within ±1N | 99.96% |
| Max XY error | 3.60 mm |

### Constant-Force Line Trace (50mm)

| Metric | Value |
|--------|-------|
| Target force | 5.0 N |
| Within ±1N | 94.07% |
| Max XY error | 1.70 mm |

## Architecture

```
              ┌───────────────────┐
  xy_d ──────▶│  Hybrid Controller│
  F_desired ─▶│                   │──▶ τ ──▶ MuJoCo
              │  Position (XY)    │         ▲
              │  + Force (Z)      │         │
              │  + g(q)           │    mj_contactForce
              └───────────────────┘    (real EE contact set ↔ table)
                     ▲
                     │ FK, J (Pinocchio)
                     └─────────────────
```

### Phase State Machine

```
APPROACH → SETTLE → TRACE → HOLD
   │          │        │        │
   ▼          ▼        ▼        ▼
 Impedance  Hybrid   Hybrid   Hybrid
 descent    at start  + line   at end
            XY       traj
```

## Key Gains

| Parameter | Value | Role |
|-----------|-------|------|
| K_p (XY) | 2000 N/m | Position stiffness |
| K_d (XY) | 100 N·s/m | Position damping |
| K_fp | 5.0 | Force proportional |
| K_fi | 10.0 | Force integral |
| K_dz | 30.0 N·s/m | Z-velocity damping |
| α_filter | 0.2 | EMA force smoothing |

## How to Run

```bash
# Static force hold demo
python3 src/c1_force_control.py

# Capstone: constant-force line tracing
python3 src/c2_line_trace.py
```

## Limitations

- The Jacobian X-row is small (~0.01) in the vertical tool configuration, limiting XY tracking bandwidth during contact. Longer lines (>50mm) require slower trajectories.
- Line traces work best near (0.4, 0.0) where the arm has good manipulability.
