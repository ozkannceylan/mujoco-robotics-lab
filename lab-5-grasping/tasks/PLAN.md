# Lab 5: Grasping & Manipulation — Implementation Plan

## Phase 1: Gripper Integration
### Step 1.1: Create UR5e + Parallel Gripper MJCF Model
- Extend Lab 3's ur5e.xml with a simple parallel-jaw gripper attached to tool0
- Two prismatic joints (left_finger, right_finger) with symmetric actuation
- Position-controlled via a single actuator (open/close)
- Gripper geoms with appropriate contype/conaffinity for object grasping
- Expected output: `models/ur5e_gripper.xml` loads without errors

### Step 1.2: Create Scene with Table and Graspable Objects
- Scene file: `models/scene_grasp.xml` — includes ur5e_gripper, table, box object
- Box is a free body on the table surface (6-DOF freejoint)
- Configure contact pairs: gripper fingers ↔ object, object ↔ table
- Expected output: scene loads, box sits stable on table under gravity

### Step 1.3: Create Lab 5 Common Module
- `src/lab5_common.py` with paths, constants, model loaders
- Load MuJoCo model (7 actuators: 6 arm + 1 gripper)
- Load Pinocchio model (6-DOF arm only — gripper is MuJoCo-only)
- Gripper open/close helpers
- Expected output: both models load, FK matches between engines

### Step 1.4: Implement Gripper Controller
- `src/gripper_controller.py` — open, close, set_width functions
- Position control for gripper actuator via ctrl
- Grasp detection: monitor contact forces between fingers and object
- Expected output: gripper opens/closes reliably at Q_HOME

### Step 1.5: Write Phase 1 Tests
- Test model loading (both engines)
- Test gripper open/close
- Test box stability on table
- Test contact detection when gripper closes on object

## Phase 2: Contact Physics Tuning
### Step 2.1: Contact Parameter Experiments
- Script `src/contact_tuning.py` — systematically vary condim, friction, solref, solimp
- Test object stability (box shouldn't slide or bounce on table)
- Test grip quality (object shouldn't slip during arm motion)
- Document optimal values with rationale

### Step 2.2: Grip Stability Test
- Move arm through various configurations while gripping
- Verify object doesn't slip: track object position relative to gripper
- Measure grip force via contact forces
- Expected output: object stays within 2mm of initial grip position during motion

### Step 2.3: Write Phase 2 Tests
- Test object doesn't fall through table
- Test grip holds during arm motion trajectory
- Test slip detection threshold

## Phase 3: Pick-and-Place Pipeline
### Step 3.1: Implement Grasp Pose Computation
- `src/grasp_planner.py` — compute top-down approach poses
- Pre-grasp pose: above object with gripper open, oriented downward
- Grasp pose: at object height for grasping
- Place pose: above target location
- Use Pinocchio IK to solve joint configurations for each pose

### Step 3.2: Implement Grasp State Machine
- `src/grasp_state_machine.py` — IDLE → APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → PLACE → RETREAT → IDLE
- Each state has: entry condition, controller, exit condition
- Wire controllers: impedance (from Lab 3) for approach/descend/lift/place, RRT* (from Lab 4) for transport, gripper ctrl for grasp/release

### Step 3.3: Integrate Lab 3 + Lab 4 Controllers
- Import impedance controller from Lab 3 for compliant motions
- Import RRT* planner from Lab 4 for collision-free transport
- Adapt collision checker for new scene (no obstacles, just table)
- Joint-space PD for approach, Cartesian impedance for descend/lift

### Step 3.4: Run Full Pick-and-Place
- `src/pick_and_place.py` — end-to-end demo
- Pick box from position A, place at position B
- Log all state transitions and metrics
- Expected output: box successfully relocated

### Step 3.5: Write Phase 3 Tests
- Test IK solutions for grasp poses are collision-free
- Test state machine transitions
- Test full pipeline success (object at target ± tolerance)

## Phase 4: Documentation & Blog
### Step 4.1: Write English Documentation
- `docs/01_gripper_integration.md` — gripper model, control
- `docs/02_contact_physics.md` — MuJoCo contact parameters deep dive
- `docs/03_pick_and_place.md` — state machine, pipeline

### Step 4.2: Write Turkish Documentation
- `docs-turkish/01_tutucu_entegrasyonu.md`
- `docs-turkish/02_temas_fizigi.md`
- `docs-turkish/03_al_birak_boru_hatti.md`

### Step 4.3: Write Blog Post
- `blog/lab_05_grasping.md` — narrative of building pick-and-place from scratch

### Step 4.4: Write README.md
- Lab overview, architecture diagram, usage instructions, results

## Phase 5: Capstone Demo
### Step 5.1: Design Capstone Scene
- Multi-object scene: pick 2-3 objects and place at designated targets
- Different object sizes/positions for variety

### Step 5.2: Run and Record Capstone
- `src/capstone_demo.py` — full pipeline with metrics and plots
- Generate state machine timeline visualization
- Generate EE trajectory plot
- Summary table with timing and success metrics
