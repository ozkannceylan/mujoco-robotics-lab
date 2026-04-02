# Lab 6 — Lessons Learned

## L1: Don't use base yaw to make arms face each other
- **Symptom:** 180° yaw mount on right arm base made Q_HOME mirroring intractable — three iterations of joint sign-flipping and grid search couldn't reliably produce symmetric downward-pointing EEs.
- **Root cause:** A base yaw rotation changes the relationship between joint space and Cartesian space. Mirroring joint values doesn't mirror Cartesian poses when base orientations differ. The mount R_z(pi) composed with Menagerie's R_z(pi) base quat produced confusing effective orientations.
- **Fix:** Removed 180° yaw from right arm base entirely. Both arms now have identical base orientation (Menagerie quat only). Q_HOME_RIGHT = Q_HOME_LEFT. "Arms facing each other" deferred to M3 via IK targets.
- **Takeaway:** Keep base frames identical for both arms. Handle facing direction through IK/trajectory planning, not base rotation. Simpler base = simpler everything downstream.

## L2: UR5e EE site frame needs explicit rotation for approach direction
- **Symptom:** EE site z-axis pointed along body Z (sideways), not along tool approach direction. Gate criterion dot(z, -z_world) always read ~0.
- **Root cause:** MuJoCo sites inherit body frame by default. The UR5e tool approach direction is along body +Y, not +Z.
- **Fix:** Added `quat="0.7071 -0.7071 0 0"` (R_x(-90°)) to both EE sites so site z-axis = body +Y = approach direction.
- **Takeaway:** Always define EE site orientation explicitly. Don't assume body frame axes match your task-space conventions.

## L3: Table collision blocking right arm PD convergence
- **Symptom:** Right arm steady-state joint error ~0.6 rad at shoulder_pan while left arm converged to <0.001 rad. Same Kp/Kd for both.
- **Root cause:** Table x half-extent was 0.40 (table edge at x=0.9). Right arm base at x=1.0 meant the UR5e shoulder/upper-arm collision capsules (radius 0.06) were physically inside the table. Contact forces prevented shoulder_pan from reaching its target.
- **Fix:** Narrowed table x half-extent from 0.40 to 0.20 (table spans x=0.3..0.7). Gives 0.3m clearance from each arm base.
- **Takeaway:** Always check `data.ncon` and contact geom pairs when a joint-space controller fails to converge on one arm but not the other. Collision is the likely culprit.

## L4: Lab 4 URDF uses different kinematic convention than MuJoCo Menagerie
- **Symptom:** Pinocchio FK with the Lab 4 URDF gave ~99mm offset from MuJoCo EE positions. Offset rotated with shoulder_pan (constant in arm-local frame).
- **Root cause:** Lab 4's URDF uses standard DH convention (shoulder_lift: `rpy="pi/2 0 0" xyz="0 0 0"`, elbow: `xyz="-0.425 0 0"`). MuJoCo Menagerie uses a different body layout (shoulder_lift: `pos="0 0.138 0" quat="1 0 1 0"`). These are different parameterizations of the same physical robot. Lab 3's URDF was hand-tuned to match Menagerie exactly, including: (1) 180° Z rotation in world_joint, (2) shoulder_lift `rpy="0 pi/2 0" xyz="0 0.138 0"`, (3) axes as `0 1 0` instead of `0 0 1`.
- **Fix:** Replaced Lab 6's URDF with Lab 3's Menagerie-matching kinematic chain (minus gripper payload). FK error dropped from 99mm to 0.000mm (exact match).
- **Takeaway:** Always verify which URDF matches your MuJoCo model. The "standard" DH-convention URDF from universal_robots_description does NOT match MuJoCo Menagerie. Lab 3's hand-tuned URDF is the canonical one for Pinocchio cross-validation.

## L5: DLS IK needs step clamping and multi-start for reliability
- **Symptom:** DLS IK (damping=0.01, max_iter=100) failed on 4/20 6DOF targets and all position-only targets. Solver got stuck in local minima with 500-990mm residual.
- **Root cause:** Without step clamping, large Jacobian pseudoinverse steps overshoot, especially near singularities. Single initial guess (Q_HOME) is far from some targets in joint space.
- **Fix:** Added dq_max=0.5 rad step clamping per iteration + multi-start (n_restarts=8 random perturbations around home). 6DOF: 20/20 converge. Position-only with n_restarts=20: 5/5 converge.
- **Takeaway:** DLS IK for 6-DOF arms should always include step clamping and multi-start. Position-only (underconstrained) needs more restarts than full 6DOF.

## L6: IK solutions must be collision-checked and joint-wrapped for dual-arm
- **Symptom:** Arm PD controller failed to converge to IK targets (err stuck at 1.0+ rad). Left arm collided with box during swing, right arm upper_arm_link rested on table at final config. 9-11 MuJoCo contacts.
- **Root cause:** (1) IK solver doesn't know about scene geometry — it finds kinematically valid but physically colliding configs. (2) Without joint wrapping, IK finds solutions that require 200°+ joint motion (going "the long way around" via 2*pi-equivalent angles). (3) The HOME→approach reconfiguration is genuinely large (~150° max joint change) because EE must rotate from pointing down to pointing sideways.
- **Fix:** Three-part solution: (a) Joint wrapping — after IK, wrap each joint to `q_ref ± pi` to minimize distance from reference. (b) Collision-free IK search — evaluate 300 random starts, check each converged solution for MuJoCo contacts at the static config, keep the closest collision-free one to home. (c) Chained IK — use approach solution as seed for grasp IK so Phase 2 transition is only ~0.68 rad instead of completely different configuration.
- **Takeaway:** For dual-arm scenes with obstacles, IK must be followed by collision checking. Joint wrapping is essential for PD control — a 200° motion that could be a 160° motion in the other direction may avoid collisions entirely. Always chain sequential IK targets (approach → grasp) so transitions are small.

## L7: Large reconfigurations need higher PD gains
- **Symptom:** Kp=100, Kd=10 (M1 gains) failed to drive the arm through a ~150° reconfiguration from HOME to approach pose. Arms barely moved in 8 seconds.
- **Root cause:** For small motions near equilibrium, Kp=100 produces only 100*0.005=0.5 Nm at threshold. For large motions (2.5 rad error), it produces 250 Nm — which is clipped to 150 Nm for the big joints. But the real issue is Kd=10 is too much damping relative to Kp=100 for fast tracking: the critically-damped response is too slow for these distances.
- **Fix:** Increased to Kp=500, Kd=50 for M3. Both arms settled in 0.62s with 2ms synchronization error.
- **Takeaway:** PD gains should be scaled to the task. Small-error regulation (M1: 0.001 rad) can use low gains. Large reconfigurations (M3: 2.5 rad) need higher Kp and proportionally higher Kd.

## L8: Weld constraints must have eq_data set to current relative pose before activation
- **Symptom:** Activating weld constraints (`eq_active=1`) at runtime launched the box 90+ cm away. Box went from `[0.5, 0, 0.245]` to `[1.12, 0.65, 0.37]` in 0.5s.
- **Root cause:** MuJoCo weld constraints enforce the relative pose stored in `eq_data`, which is computed at model compilation time from the initial body positions. At grasp time, the arm wrists are in completely different positions relative to the box. Activating the weld makes the solver try to teleport the box to the initial relative pose.
- **Fix:** Before activating, compute the current relative transform (pos + quat) between the two bodies and write it into `mj_model.eq_data[weld_id]`. Layout: `[anchor(3), rel_pos(3), rel_quat(4), torquescale(1)]`. The quat is (w,x,y,z) format via `mju_mat2Quat`.
- **Takeaway:** Never activate runtime weld constraints without first setting `eq_data` to the current relative pose. The default eq_data from compilation is almost never what you want at runtime.

## L9: Carry direction constrained by dual-arm workspace geometry
- **Symptom:** CARRY IK failed when attempting 20cm carry in +x. Right arm IK returned no solution.
- **Root cause:** Both arm bases are at y=0, separated by 1.0m in x. Box center at x=0.5. Carrying +x moves the box asymmetrically — the right EE (at `box_x + half_x + standoff`) approaches the right arm base (x=1.0). At z=0.41, the UR5e can't reach x>0.75 with the required -x pointing orientation due to kinematic constraints.
- **Fix:** Changed carry direction to +y, which is symmetric for both arms (both at y=0). Workspace is ample in y for 20cm displacement.
- **Takeaway:** For dual-arm setups, carry direction along the inter-base axis is severely constrained. Lateral (perpendicular to base axis) carry preserves symmetric reachability.

## L10: IK standoff during weld-active phases must match locked EE-to-box offset
- **Symptom:** LIFT/CARRY IK targets used `GRASP_STANDOFF` (5cm) but weld constraints locked the EE at `-CONTACT_PENETRATION` (-2cm, inside box). This creates 7cm offset between PD target and weld-enforced position, causing internal forces.
- **Root cause:** The weld freezes the relative transform at CLOSE time (EE 2cm inside box surface). Computing IK targets with a different standoff means the PD controller and weld constraint fight each other.
- **Fix:** Changed all weld-active phases (LIFT, CARRY, PLACE) to use `-CONTACT_PENETRATION` as the standoff when computing EE targets from desired box position.
- **Takeaway:** Once a weld locks a relative pose, all subsequent IK targets must be computed using that same relative offset.

## L11: Box placement needs absolute z target and arm retraction
- **Symptom:** After weld release, box slid off table (z went from 0.26 to 0.10, x/y drifted 10cm). Rotation error 120°.
- **Root cause:** Two issues: (1) PLACE used a relative delta (`-LIFT_DZ`) which didn't compensate for lift overshoot, leaving box 1.5cm above table. (2) After weld release, arms at CLOSE position pushed the box via contact forces.
- **Fix:** (1) Compute absolute place z as `TABLE_SURFACE_Z + BOX_HALF_EXTENTS[2]` (=0.245). (2) After weld release, immediately retract arms to APPROACH_STANDOFF (10cm) to break contact.
- **Takeaway:** Always use absolute target positions for critical placements. After releasing welds, retract immediately — arms at contact distance will push the object.

## L12: Step-command PD produces jerky motion — use ramped interpolation
- **Symptom:** Commanding the final joint target instantly to the PD controller produced fast, jerky arm movements.
- **Root cause:** A step command at high Kp (500) produces maximum torque instantly. The arm accelerates at maximum rate, overshoots, and oscillates.
- **Fix:** Added 2-second smooth-step (3α²-2α³) interpolation between current and target joint configs. Lower gains (Kp=300, Kd=40) combined with the ramp produce slow, smooth motion.
- **Takeaway:** For visually appealing robot motion, always ramp the PD target instead of step-commanding it. Smooth-step interpolation is simple and effective.
