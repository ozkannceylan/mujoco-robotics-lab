# Lab 6 — TODO

## Current Focus
**M4 complete.** Next session: M5 (Capstone demo + documentation).

## Completed
- [x] M0: Build and validate dual-arm scene
  - scene_dual.xml with left (origin) + right (x=1.0m, no yaw) UR5e arms
  - 12 arm joints + 1 freejoint (box), 12 motor actuators
  - Both EE z-axes point down (dot=1.0), screenshot: media/m0_scene.png
  - Table narrowed to x half-extent 0.20 to avoid arm-table collision
- [x] M1: Independent joint PD control with gravity compensation
  - DualArmJointPD class: tau = Kp*(q_des-q) + Kd*(0-qd) + qfrc_bias
  - Kp=100, Kd=10. 3 targets per arm, all converge.
  - Gate: max error 0.000247 rad (<0.001), max vel 0.00187 rad/s (<0.01)
  - Video: media/m1_independent.mp4
- [x] M2: Pinocchio dual-arm FK + IK
  - Replaced Lab 4 URDF with Lab 3 Menagerie-matching kinematic chain
  - DualArmModel class: FK, Jacobian, DLS IK with multi-start + step clamping
  - FK cross-validation: 0.000 mm error across 20 configs (exact match)
  - IK: 20/20 6DOF targets converge, max FK round-trip 0.100 mm (<0.5)
  - Position-only IK: 5/5 converge with n_restarts=20
  - Deliverables: dual_arm_model.py, m2_fk_validation.py, m2_ik_validation.py
- [x] M3: Coordinated approach to box
  - grasp_pose_calculator.py: computes approach/grasp SE3 from box pose
  - Collision-free IK search (300 trials, wrapping, static collision check)
  - Chained IK: grasp seeded from approach (0.68 rad transition distance)
  - Kp=500, Kd=50 for large HOME→approach reconfiguration
  - Phase 1 arrival diff: 2.0 ms (<50). Phase 2: 1.0 ms (<50)
  - Cartesian error: L=0.10mm, R=0.09mm (<5mm)
  - EE z-dot toward box: L=1.0000, R=1.0000 (>0.9)
  - Video: media/m3_approach.mp4, screenshot: media/m3_final.png
- [x] M4: Cooperative carry (bimanual grasp + transport)
  - BimanualStateMachine: 6-state pipeline (APPROACH→CLOSE→GRASP→LIFT→CARRY→PLACE)
  - Weld constraints with runtime relpose locking (_set_weld_relpose)
  - EE targets use -CONTACT_PENETRATION standoff during weld-active phases
  - Carry in +y direction (22cm) — +x infeasible due to workspace limits
  - Smooth motion: 2s ramp with smooth-step interpolation, Kp=300/Kd=40
  - Place: absolute z target (TABLE_SURFACE_Z + box_half_z) + arm retraction
  - Gate: lift 15.0cm (≥13), carry 22.0cm (≥18), place dz=0.0cm (<3), rot=4.0° (<10)
  - Video: media/m4_carry.mp4, plot: media/m4_box_trajectory.png

## Pending
- [ ] M5: Capstone demo + documentation

## Blockers
- None
