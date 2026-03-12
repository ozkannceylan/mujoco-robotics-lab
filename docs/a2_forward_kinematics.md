# A2: Forward Kinematics — Study Notes

## What Is FK?

The answer to: "Given joint angles, where is the end-effector?"

```
Input:  θ₁, θ₂ (joint angles, rad)
Output: (x, y) end-effector position (m)
```

FK is the most fundamental computation in robotics. IK is its inverse, the Jacobian is its derivative.

---

## Geometric FK — 2-Link Planar

DH parameters are overkill for a 2-link arm. The geometric approach is more intuitive:

```
Joint 2 position (tip of link1):
  x₁ = L₁ · cos(θ₁)
  y₁ = L₁ · sin(θ₁)

End-effector position:
  x = L₁ · cos(θ₁) + L₂ · cos(θ₁ + θ₂)
  y = L₁ · sin(θ₁) + L₂ · sin(θ₁ + θ₂)
```

**Key point:** `θ₁ + θ₂` — joint2's angle is *relative* (to link1). The absolute angle is θ₁ + θ₂.

### Hand Calculation Verification (θ₁=30°, θ₂=45°)

```
x = 0.3·cos(30°) + 0.3·cos(75°) = 0.2598 + 0.0776 = 0.3375
y = 0.3·sin(30°) + 0.3·sin(75°) = 0.1500 + 0.2898 = 0.4398
```

---

## Homogeneous Transformation

Using 3×3 matrices (2D), we obtain the same result via chain rule:

```
T₀₁ = | cos(θ₁)  -sin(θ₁)  L₁·cos(θ₁) |
      | sin(θ₁)   cos(θ₁)  L₁·sin(θ₁) |
      |    0         0          1        |

T₁₂ = | cos(θ₂)  -sin(θ₂)  L₂·cos(θ₂) |
      | sin(θ₂)   cos(θ₂)  L₂·sin(θ₂) |
      |    0         0          1        |

T₀₂ = T₀₁ · T₁₂   →   EE position = T₀₂[:2, 2]
```

**Why it matters:** When moving to 6+ DOF robots the geometric approach becomes unwieldy, but chain rule (T₀₁ · T₁₂ · ... · Tₙ₋₁,ₙ) always works.

---

## MuJoCo Verification

10 different angle combinations were tested. Result: **0.000000 m error** (exact match).

| θ₁ | θ₂ | FK x | FK y | MuJoCo x | MuJoCo y | Error |
|---|---|---|---|---|---|---|
| 0° | 0° | +0.6150 | +0.0000 | +0.6150 | +0.0000 | 0 |
| 30° | 45° | +0.3413 | +0.4543 | +0.3413 | +0.4543 | 0 |
| 90° | -45° | +0.2227 | +0.5227 | +0.2227 | +0.5227 | 0 |
| 180° | 0° | -0.6150 | +0.0000 | -0.6150 | +0.0000 | 0 |

**Note:** The `end_effector` site in the model XML has `pos="0.015 0 0"` offset (gripper tip). Adding this as `EE_OFFSET` in the FK computation gives an exact match. Without the offset, FK returns the tip of link2 — the site position is 1.5 cm further.

---

## Workspace

![Workspace](a2_fk_workspace.png)

- **Outer boundary:** r = L₁ + L₂ = 0.6 m (arms fully extended)
- **Inner boundary:** r = |L₁ - L₂| = 0.0 m (equal-length links → can reach the centre)
- Equal link lengths → workspace is a full disk (no hole)

---

## Configuration Examples

![Configurations](a2_fk_configurations.png)

---

## Lessons Learned

1. **Geometric FK is simple but powerful.** Trigonometry is sufficient for 2-link; DH is unnecessary.
2. **Watch for site offsets.** The MuJoCo `site` position may not coincide with the link tip — you need to know the offset in the model.
3. **Homogeneous transformation = chain rule.** One T matrix per link, multiply and done. Scales to 6 DOF.
4. **Workspace = reachable area.** If L₁ = L₂ it is a full disk; if L₁ ≠ L₂ it is an annulus.

---

## Next Step: Jacobian (A3)

Taking the derivative of FK gives us the Jacobian: the "joint velocities → end-effector velocity" mapping.
