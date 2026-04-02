# Lab 7: Locomotion Fundamentals — Milestone Plan

> **Platform:** Unitree G1 on MuJoCo
> **Capstone:** Stable bipedal walking on flat ground (10+ steps)
> **Rule:** ONE milestone per session. No exceptions.

---

## M0: Load G1 and Understand the Robot

- [ ] Load Unitree G1 from MuJoCo Menagerie
- [ ] Print full joint list: name, index, range, actuator mapping
- [ ] Identify and document: which joints are legs, arms, waist, head
- [ ] Count total DOFs, actuated DOFs, leg DOFs
- [ ] Set arms to neutral pose and lock them
- [ ] Run 2s simulation with no control (freefall collapse)

**Gate:**
| Criterion | Target | Status |
|-----------|--------|--------|
| Joint table printed | complete | |
| DOF layout in `docs/g1_joint_map.md` | complete | |
| Collapse video | `media/m0_freefall.mp4` | |
| T-pose screenshot | `media/m0_tpose.png` | |

---

## M1: Standing with Joint PD + Gravity Compensation

- [ ] Joint PD controller for leg joints: `tau = Kp*(q_ref - q) + Kd*(0 - qd) + qfrc_bias`
- [ ] `q_ref` = G1 standing config (Menagerie default or tuned)
- [ ] Stand 10s on flat ground without falling
- [ ] Apply 5N lateral push at t=3s, robot recovers

**Gate:**
| Criterion | Target | Status |
|-----------|--------|--------|
| Stands 10s | base height within 5cm of initial | |
| Recovers from 5N push | no fall | |
| Video | `media/m1_standing.mp4` | |

---

## M2: CoM Tracking and Support Polygon

- [ ] Compute CoM via Pinocchio (`computeCenterOfMass`)
- [ ] Cross-validate: Pinocchio vs MuJoCo (`data.subtree_com[0]`)
- [ ] Compute support polygon from foot contact points
- [ ] CoM visualizer: CoM projection vs support polygon over time
- [ ] CoM-based balance controller: PD on CoM position

**Gate:**
| Criterion | Target | Status |
|-----------|--------|--------|
| CoM cross-validation error | < 5mm | |
| CoM inside support polygon during 5N push | yes | |
| Plot | `media/m2_com_polygon.png` | |
| Video | `media/m2_com_balance.mp4` | |

---

## M3: Single Step (Weight Shift + Foot Lift)

- [ ] Weight shift: move CoM over stance foot
- [ ] Lift swing foot 5cm, move forward 15cm, place down
- [ ] ONE step only, not walking
- [ ] Task-space IK: CoM target + swing foot target + stance foot fixed

**Gate:**
| Criterion | Target | Status |
|-----------|--------|--------|
| One step without falling | yes | |
| Swing foot clearance | > 3cm | |
| Stable after step for 2s | yes | |
| Video | `media/m3_single_step.mp4` | |

---

## M4: ZMP Walking (10+ steps)

- [ ] LIPM for CoM trajectory generation
- [ ] Hard-coded footstep plan: 12 alternating steps, 15cm stride
- [ ] ZMP-stable CoM trajectory (preview control or analytical LIPM)
- [ ] Whole-body IK execution (CoM + feet trajectories -> joint angles)

**Gate:**
| Criterion | Target | Status |
|-----------|--------|--------|
| Steps completed | >= 10 | |
| ZMP inside support polygon | yes (plot) | |
| Base roll/pitch | < 10 deg | |
| Walking video | `media/m4_walking.mp4` | |
| ZMP plot | `media/m4_zmp.png` | |

---

## M5: Documentation and Capstone

- [ ] `docs/ARCHITECTURE.md`: floating-base dynamics, controller design, IK pipeline, ZMP theory, lessons
- [ ] `docs-turkish/ARCHITECTURE_TR.md`: Turkish translation
- [ ] `docs/CODE_WALKTHROUGH.md`: code walkthrough
- [ ] Capstone demo script with state overlay
- [ ] Blog post: "Making a Humanoid Walk: From Standing to ZMP Gait"

**Gate:**
| Criterion | Target | Status |
|-----------|--------|--------|
| All docs complete | yes | |
| Capstone video | end-to-end | |
| Blog word count | > 1000 | |
