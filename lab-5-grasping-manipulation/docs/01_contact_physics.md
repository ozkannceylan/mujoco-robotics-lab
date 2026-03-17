# Contact Physics in MuJoCo — Lab 5

## Overview

Reliable grasping requires robust contact models. MuJoCo exposes five key parameters that govern how contacts behave: `condim`, `friction`, `solref`, `solimp`, and `mass`. This document explains what each parameter controls and the specific values chosen for Lab 5.

---

## Contact Parameters

### `condim` — Contact Dimensionality

Defines how many forces a contact point can exert.

| Value | Forces included | Use case |
|-------|----------------|----------|
| 1 | Normal only | Frictionless surfaces |
| 3 | Normal + 2D tangential | Standard friction |
| 4 | Normal + 2D tangential + torsional | Grasping, spinning contact |
| 6 | Full 6-DOF wrench | Soft contacts |

**Lab 5 choice: `condim="4"`** — both the gripper pads and the graspable box use condim=4. The torsional component prevents the box from rotating within the grip when the arm moves.

---

### `friction` — Friction Coefficients

Three coefficients: `[μ_slide, μ_spin, μ_roll]`

- **μ_slide** (primary): tangential friction for sliding motion. Values ≥ 1.0 provide grip.
- **μ_spin**: torsional friction. Prevents axial rotation of grasped object.
- **μ_roll**: rolling resistance. Usually small (0.001–0.01).

**Lab 5 values:**
```xml
friction="1.5 0.005 0.0001"
```

μ_slide = 1.5 provides firm grip on a 150 g box. The spin coefficient 0.005 damps torsion without over-constraining the contact.

---

### `solref` — Constraint Reference

`solref="timeconst stiffness"` controls how the constraint force builds up.

- **timeconst**: rise time in seconds. Smaller = stiffer. Default 0.02 s.
- **stiffness**: damping ratio. Default 1.0 (critically damped).

**Lab 5 value: `solref="0.002 1"`** — very stiff contacts (2 ms rise time, critically damped). Prevents visible box compression under the gripper.

---

### `solimp` — Constraint Impedance

`solimp="dmin dmax width midpoint power"` — a 5-element vector controlling contact softness.

Key elements:
- **dmin, dmax**: minimum and maximum constraint force scaling (0–1). Higher = stiffer.
- **width**: penetration depth at which max stiffness is reached.

**Lab 5 value: `solimp="0.99 0.99 0.001"`** — near-rigid contacts (99% of ideal constraint). The 0.001 m width means full stiffness kicks in after 1 mm penetration.

---

### Box mass

The box mass directly affects how much gripper force is needed to lift it.

**Lab 5 value: `mass="0.15"` (150 g)** — realistic small object (e.g., a 40mm aluminium cube). Verified to be holdable with the gripper's kp=200 position actuator without slipping.

---

## Parameter Interaction

The table below shows how changing each parameter affects grasping behaviour:

| Parameter | If too low / soft | If too high / stiff |
|-----------|------------------|---------------------|
| μ_slide | Box slides during transport | — |
| condim | Box rotates in grip | Simulation slower |
| solref timeconst | Contact oscillates, box vibrates | — |
| solimp dmin/dmax | Soft contact, visible penetration | Numerical stiffness |
| box mass | No problem | Box too heavy, falls |

---

## Geometry Constraint: Minimum Gripper Gap

A common design mistake: the gripper's minimum gap at `GRIPPER_CLOSED` must be **less than** the object's cross-section.

In Lab 5:
- Finger body Y-offset from center: **±0.015 m**
- Pad inner offset: +0.009 m, half-size 0.005 m → pad inner face at **0.019 m** when joint=0
- Box half-width: **0.020 m**

The pad inner face (0.019 m) is just inside the box edge (0.020 m), producing 1 mm of overlap — enough for MuJoCo's stiff contact to generate holding force.

> **Lesson:** Always prototype gripper geometry in a static scene and verify that `pad_inner_face < object_half_width` before implementing control code.

---

## References

- [MuJoCo Contacts Documentation](https://mujoco.readthedocs.io/en/latest/computation.html#contacts)
- [MuJoCo XML Reference — geom](https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-geom)
