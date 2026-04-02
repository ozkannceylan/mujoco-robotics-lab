# Lab 7: Locomotion Fundamentals

> **Status:** Not Started
> **Prerequisites:** Lab 3 (dynamics & force control)
> **Platform:** Unitree G1 on MuJoCo
> **Capstone Demo:** Stable bipedal walking on flat ground (10+ steps)

---

## Objectives

1. Understand the fundamental differences between fixed-base and floating-base dynamics
2. Implement standing balance using center of mass (CoM) control
3. Generate a basic walking gait using Zero Moment Point (ZMP) planning
4. Achieve stable bipedal walking on the Unitree G1 in MuJoCo

---

## Why This Lab Matters

This is the biggest paradigm shift in the series. Labs 1–6 dealt with fixed-base robots: the base is bolted to the ground, and the only moving parts are the joints. The G1 humanoid has a floating base — the entire robot can fall over. Gravity is no longer just a torque to compensate; it's an existential threat. Every concept you learned about dynamics (Lab 3) still applies, but the problem is fundamentally harder: you must simultaneously control the joints AND keep the robot from falling.

---

## Theory Scope

- Floating base dynamics: 6-DOF unactuated base + actuated joints
- Center of Mass (CoM) and support polygon
- Zero Moment Point (ZMP): the point where ground reaction forces produce zero moment
- ZMP stability criterion: ZMP must stay inside the support polygon
- Simple gait generation: pre-planned footstep sequence + ZMP trajectory
- Centroidal dynamics: linear and angular momentum at the CoM

---

## Architecture

```
Footstep Plan (predefined step positions)
        |
        v
+------------------------+
|  ZMP Trajectory         |
|  Planner                |
|  Footsteps -> desired   |
|  CoM trajectory         |
+------------+-----------+
             | desired CoM(t)
             v
+------------------------+
|  Inverse Kinematics     |
|  CoM + foot poses ->    |
|  joint angles           |
|  (Pinocchio IK)         |
+------------+-----------+
             | joint targets q_d(t)
             v
+------------------------+
|  Joint PD + Gravity     |
|  Compensation           |
|  (adapted from Lab 3)   |
+------------+-----------+
             | joint torques
             v
+------------------------+
|  MuJoCo Simulation      |
|  G1 humanoid model      |
|  Flat ground contact    |
+------------------------+
```

---

## Milestones

### M0: Load G1 and Understand the Robot

- Load Unitree G1 from MuJoCo Menagerie
- Print full joint list: name, index, range, actuator mapping
- Identify and document: which joints are legs, arms, waist, head
- Count total DOFs, actuated DOFs, leg DOFs
- Set arms to a neutral pose and lock them (zero torque or fixed position)
- Let simulation run for 2 seconds with no control (robot will collapse under gravity)

**Gate:**
- Joint table printed
- Robot DOF layout documented in `docs/g1_joint_map.md`
- Collapse video saved to `media/m0_freefall.mp4`
- Screenshot of initial T-pose saved to `media/m0_tpose.png`

---

### M1: Standing with Joint PD + Gravity Compensation

- Implement joint PD controller for all leg joints (reuse Lab 6 pattern: `tau = Kp*(q_ref - q) + Kd*(0 - qd) + qfrc_bias`)
- `q_ref` = initial standing joint config from the G1 model (Menagerie default or manually tuned)
- Robot should stand still on flat ground for 10 seconds without falling
- Apply 5N lateral push at torso at t=3s, robot must recover

**Gate:**
- Robot stands 10s without falling (base height stays within 5cm of initial)
- Recovers from 5N push
- Video saved to `media/m1_standing.mp4`

---

### M2: CoM Tracking and Support Polygon

- Compute CoM position using Pinocchio (`computeCenterOfMass`)
- Cross-validate CoM: Pinocchio vs MuJoCo (`data.subtree_com[0]`)
- Compute support polygon from foot contact points
- Implement CoM visualizer: plot CoM projection on ground vs support polygon over time
- Implement CoM-based balance controller: PD on CoM position to keep it centered over support polygon

**Gate:**
- CoM cross-validation error < 5mm
- CoM stays inside support polygon during 5N push test
- Plot saved to `media/m2_com_polygon.png`
- Video saved to `media/m2_com_balance.mp4`

---

### M3: Single Step (Weight Shift + Foot Lift)

- Implement weight shift: move CoM over stance foot before lifting swing foot
- Lift swing foot 5cm, move forward 15cm, place down
- This is ONE step only, not walking
- Use task-space IK: CoM target + swing foot target + stance foot fixed

**Gate:**
- Robot takes one step without falling
- Swing foot clears ground by > 3cm
- Robot stable after step for 2s
- Video saved to `media/m3_single_step.mp4`

---

### M4: ZMP Walking (10+ steps)

- Implement LIPM (Linear Inverted Pendulum Model) for CoM trajectory
- Hard-code footstep plan: 12 alternating steps, 15cm stride, flat ground
- Compute ZMP-stable CoM trajectory using preview control or analytical LIPM solution
- Execute full gait with whole-body IK (CoM + feet trajectories to joint angles)

**Gate:**
- Robot walks 10+ steps without falling
- ZMP stays inside support polygon (plot)
- No excessive wobble (base roll/pitch < 10 deg)
- Video saved to `media/m4_walking.mp4`
- ZMP plot saved to `media/m4_zmp.png`

---

### M5: Documentation and Capstone

- Full architecture document (`docs/ARCHITECTURE.md`) covering: floating-base dynamics explanation, controller design, IK pipeline, ZMP theory with math, lessons learned
- Turkish translation (`docs-turkish/ARCHITECTURE_TR.md`)
- Code walkthrough (`docs/CODE_WALKTHROUGH.md`)
- Capstone demo script with state overlay on video
- Blog post: "Making a Humanoid Walk: From Standing to ZMP Gait"

**Gate:**
- All docs complete
- Capstone video end-to-end
- Blog > 1000 words

---

## Key Design Decisions

- **Start with standing, not walking.** Standing balance is a prerequisite. If the robot can't stand still under perturbation, walking will fail immediately.
- **Use LIPM, not full nonlinear optimization.** The Linear Inverted Pendulum Model is the classic entry point. Full centroidal MPC (Lab 8 territory) is overkill here.
- **Pre-planned footsteps.** Don't implement footstep planning. Hard-code a sequence of alternating steps on flat ground. The goal is gait execution, not gait planning.
- **G1 has many DOFs you don't need.** Focus on the legs + waist. Lock the arms in a neutral pose (they'll be used in Lab 8). This reduces the problem to ~12 actuated DOFs.
- **Expect falls.** Lots of them. Implement a "reset to standing" function for fast iteration.

---

## Success Criteria

- [ ] G1 stands stably on flat ground under perturbation (5N push)
- [ ] ZMP trajectory stays inside support polygon during planned gait
- [ ] G1 walks 10+ steps on flat ground without falling
- [ ] Gait is visually smooth (no excessive wobble or jerking)
- [ ] Documentation complete (English + Turkish)
- [ ] Blog post published

---

## References

- Kajita et al., "Biped Walking Pattern Generation using Preview Control of ZMP" (2003)
- Vukobratovic & Borovac, "Zero-Moment Point — Thirty Five Years of Its Life" (2004)
- MuJoCo Menagerie: Unitree G1 model documentation
- Pinocchio: centroidal dynamics, computeCenterOfMass
