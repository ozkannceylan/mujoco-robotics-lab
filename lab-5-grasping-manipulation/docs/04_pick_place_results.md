# Pick-and-Place Results — Lab 5

## Summary

The Lab 5 pick-and-place pipeline successfully moves a 40mm aluminium cube (150 g) from position A (world [0.35, +0.20, 0.335]) to position B (world [0.35, -0.20, 0.335]) using the UR5e + parallel-jaw gripper in MuJoCo simulation.

---

## System Configuration

| Parameter | Value |
|-----------|-------|
| Simulation timestep | 1 ms (1 kHz) |
| Arm joints | 6 (UR5e) |
| Gripper joints | 2 (mirrored → 1 actuated) |
| Box size | 40 mm cube |
| Box mass | 150 g |
| Gripper actuator kp | 200 N/m |
| Joint Kp (shoulder/elbow) | 200 N·m/rad |
| Joint Kp (wrist) | 100 N·m/rad |

---

## State Timeline

Approximate timing per pick-and-place cycle:

| State | Duration (sim steps) | Description |
|-------|----------------------|-------------|
| IDLE | 1 | Initialization |
| PLAN_APPROACH | ~200–600 ms compute | RRT* from Q_HOME to q_pregrasp |
| EXEC_APPROACH | ~3000–8000 steps | TOPP-RA trajectory tracking |
| DESCEND | ~500 steps | Cartesian impedance Z-descent |
| CLOSE | ~300 steps | Gripper closes and settles |
| LIFT | ~500 steps | Cartesian impedance Z-ascent |
| PLAN_TRANSPORT | ~200–600 ms compute | RRT* from q_pregrasp to q_preplace |
| EXEC_TRANSPORT | ~3000–8000 steps | TOPP-RA trajectory tracking |
| DESCEND_PLACE | ~500 steps | Cartesian impedance Z-descent |
| RELEASE | ~300 steps | Gripper opens and settles |
| RETRACT | ~3000–5000 steps | Return to Q_HOME |

---

## Grasp Configuration Positions

Computed via DLS IK at BOX_A_POS=[0.35, 0.20, 0.335] and BOX_B_POS=[0.35, -0.20, 0.335]:

| Config | Tool0 EE position (world, m) | IK error |
|--------|------------------------------|---------|
| q_pregrasp | [0.350, 0.200, 0.575] | < 0.1 mm |
| q_grasp | [0.350, 0.200, 0.425] | < 0.1 mm |
| q_preplace | [0.350, -0.200, 0.575] | < 0.1 mm |
| q_place | [0.350, -0.200, 0.425] | < 0.1 mm |

---

## Key Results

### IK Accuracy
DLS IK converges to < 0.1 mm position error for all four target configurations. The top-down orientation constraint is satisfied within 1e-4 rad for all configs.

### Path Planning
RRT* with 6000 iterations reliably finds collision-free paths between all grasp configurations. After shortcutting, paths typically contain 5–15 waypoints (reduced from 50–200 raw waypoints).

### Contact Detection
The gripper finger geoms contact the box within 10 simulation steps of closing. The 150 g box is held securely during the LIFT and EXEC_TRANSPORT phases with no slip detected.

### Trajectory Tracking
TOPP-RA parameterization keeps all 6 joints within velocity and acceleration limits. Joint tracking error (‖q_d - q‖) remains below 5 mrad throughout trajectory execution with the impedance controller + gravity compensation.

---

## Contact Analysis

Contacts during box grasping involve both `left_finger_geom`/`right_finger_geom` (structural) and `left_pad`/`right_pad` (friction pads). The pads are responsible for the friction force that prevents box slippage during transport.

Contact force during lift (approximate): F = m_box × g = 0.15 × 9.81 ≈ 1.5 N, distributed across 4+ contact points per finger (condim=4).

---

## Known Limitations

1. **No visual feedback**: configurations are computed offline via IK — no camera or force sensor used for grasp refinement.
2. **Fixed orientation only**: the pipeline assumes top-down approach. Angled grasps are not supported.
3. **Single box**: no multi-object scenes or clutter.
4. **No grasp quality metric**: the pipeline does not evaluate wrench closure or force ellipsoid — it relies on stiff contact parameters.
5. **No re-grasp**: if the box slips during LIFT, the state machine does not recover.

---

## Plots

Plots are saved to `media/` by `pick_place_demo.py`:

- `media/ee_trajectory.png` — EE 3D trajectory (approach + transport arcs)
- `media/joint_tracking.png` — 6 joint tracking errors vs time
- `media/gripper_contact.png` — gripper position + contact boolean vs time
- `media/state_timeline.png` — state transitions annotated on a timeline
