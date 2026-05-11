# Lab 6 — Dual-Arm Coordination: Milestone Plan

## Milestone-Based Approach

One milestone per session. Each ends with gate criteria + evidence.

### M0: Build and Validate Dual-Arm Scene
- Create `models/scene_dual.xml` with two UR5e arms (left at origin, right at x=1.0m with 180° yaw)
- Shared table, floor, box object
- Confirm 12 joints (6 per arm), 12 actuators
- Render screenshot
- **Gate:** Screenshot shows both arms at home, no MuJoCo warnings

### M1: Independent Joint PD Control
- Write `lab6_common.py` with dual-arm constants, model loaders, index slicing
- Implement gravity-compensated joint PD controller for each arm independently
- Move each arm to a target joint config and hold
- **Gate:** Both arms reach target within 1mm, hold steady for 2s

### M2: Pinocchio Dual-Arm FK + IK
- Load two separate Pinocchio models (one per arm) with base transforms
- Implement Cartesian IK (DLS) for each arm with world-frame targets
- Cross-validate FK: Pinocchio EE vs MuJoCo site positions
- **Gate:** FK error < 1mm, IK converges for 5 test targets per arm

### M3: Coordinated Approach
- Implement synchronized motion: both arms move to grasp poses simultaneously
- Object-centric frame: derive left/right targets from box pose
- Timing sync: both EEs arrive within 50ms of each other
- **Gate:** Both arms reach grasp poses ±5mm, arrival time difference < 50ms

### M4: Cooperative Carry
- Bimanual grasp: both arms contact box from opposite sides
- Impedance control for maintaining grasp force
- Lift, transport, place sequence
- **Gate:** Box transported 20cm without dropping, EE separation stays within ±5mm

### M5: Capstone Demo + Documentation
- Full pick-carry-place demo with metrics
- Record video with overlay
- Write docs (EN + TR)
- **Gate:** Clean demo video, all docs complete
