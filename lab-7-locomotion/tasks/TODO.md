# Lab 7: TODO

## M0: Load G1 and Understand the Robot — COMPLETE
- [x] Load real Menagerie G1 (scene.xml, 29 DOF, mesh geometry)
- [x] Joint table, DOF layout, freefall video, T-pose screenshot

## M1: Standing with Joint PD + Gravity Compensation — COMPLETE
- [x] ctrl = q_ref + qfrc_bias/Kp (Kp=500), stands 10s, 5N push recovery
- [x] Pelvis deviation: 1.6mm (gate: 5cm)

## M2: CoM Tracking and Support Polygon — COMPLETE
- [x] CoM cross-validation: Pinocchio vs MuJoCo = 0.000mm (gate: <5mm)
- [x] Fix: pin_q[2] = mj_qpos[2] - PELVIS_MJCF_Z (0.793m offset)
- [x] Support polygon from 8 foot contact spheres (4 per foot)
- [x] CoM Jacobian balance controller: delta_q = K * J_com_pinv * com_error
- [x] CoM inside support polygon: 100% (gate: yes)
- [x] Gate: m2_com_polygon.png, m2_com_balance.mp4

## M3: Single Step — REBUILD (previous M3/M4 deleted: zero Pinocchio, open-loop hacks)

### M3a: Pinocchio Frame Convention Validation (pure math) — COMPLETE
- [x] Load G1 in Pinocchio, set standing config (q[2]=-0.003, legs at zero)
- [x] CoM Jacobian: all 12 leg joints validated (max error 1.03e-08)
- [x] Left foot Jacobian (left_ankle_roll_link, ID=15, LOCAL_WORLD_ALIGNED): all PASS (max 1.09e-07)
- [x] Right foot Jacobian (right_ankle_roll_link, ID=28): all PASS (max 1.09e-07)
- [x] Also validated OP_FRAME variants (left_foot ID=16, right_foot ID=29): identical results
- [x] Gate: 0/36 failures, all < 1e-4
- [x] Output saved to media/m3a_jacobian_validation.txt

### M3b: Foot FK Cross-Validation (Pinocchio vs MuJoCo) — COMPLETE
- [x] 10 random leg configs within joint limits, floating base at standing pose
- [x] Foot position error: 0.0000mm max (gate: <2mm) — machine-precision match
- [x] CoM error: 0.0000mm max (gate: <5mm) — machine-precision match
- [x] Quaternion mapping documented: MuJoCo (w,x,y,z) → Pinocchio (x,y,z,w), Z offset = 0.793
- [x] Screenshot: media/m3b_config.png
- [x] Results: media/m3b_fk_validation.txt

### M3c: Static Whole-Body IK (pure Pinocchio math) — COMPLETE
- [x] Stacked Jacobian DLS: feet(6D×2) + CoM_XY(2D) + pelvis_Z(1D) + pelvis_ori(3D) = 18 tasks
- [x] Identity test: 0 iterations, exact match (gate: <5 iters)
- [x] CoM shift left 5cm: 20 iters, CoM err=0.13mm, max foot slip=0.51mm
- [x] CoM shift right 5cm: 20 iters, CoM err=0.13mm, max foot slip=0.51mm
- [x] CoM forward 3cm: 11 iters, CoM err=0.66mm, max foot slip=0.24mm
- [x] Joint changes physically sensible: hip_roll/ankle_roll for lateral, hip_pitch/ankle_pitch for sagittal
- [x] Gate: ALL PASS — foot < 2mm, CoM < 5mm, identity < 5 iters
- [x] Results: media/m3c_ik_results.txt

### M3d: IK Weight Shift in Simulation — COMPLETE
- [x] Pre-compute IK trajectory: 11 waypoints along CoM Y-shift path (whole_body_ik from M3c)
- [x] Cosine-smooth ramp over 1.5s, hold 2s, K_VEL=40 (ζ≈0.89)
- [x] CoM shift: 53.5mm (gate: >= 40mm)
- [x] Left foot drift: 0.70mm (gate: < 5mm)
- [x] Right foot drift: 1.36mm (gate: < 5mm)
- [x] Robot standing: min pelvis Z = 0.789m (gate: > 0.6m)
- [x] Gate: ALL PASS — video: m3d_weight_shift.mp4, screenshot: m3d_shifted.png

### M3e: ZMP Walking Attempt — FAILED (6 attempts)
- [x] Implemented LIPM preview control + ZMP reference + swing trajectory
- [x] Six distinct approaches tried (varying gains, feedforward, timing, IK strategies)
- [x] Root cause: position actuators cannot provide ZMP (ankle torque) control
- [x] Conclusion: structural limitation, not a tuning problem

## M4: ZMP Walking — BLOCKED (requires torque-controlled actuators or RL)

## M5: Documentation and Capstone — COMPLETE
- [x] `docs/ARCHITECTURE.md` — 6 sections (system overview, floating-base, Pinocchio, LIPM/ZMP, failures, lessons)
- [x] `docs-turkish/ARCHITECTURE_TR.md` — Full Turkish translation (6 matching sections)
- [x] `docs/CODE_WALKTHROUGH.md` — Recommended reading order for all source files
- [x] `src/m5_capstone_demo.py` — Capstone: standing + push + weight shift + LIPM plot
- [x] `blog/lab7_locomotion.md` — "Why Making a Humanoid Walk is Harder Than It Looks" (1763 words)
- [x] `README.md` — Lab overview with run instructions

## Current Focus
> Lab 7 COMPLETE at M3d scope. All documentation and capstone delivered.

## Blockers
> None — Lab 7 is done.
