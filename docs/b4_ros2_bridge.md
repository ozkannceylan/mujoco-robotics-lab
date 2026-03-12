# B4: ROS2 Bridge — Optional

## Purpose

Use the MuJoCo simulation as a ROS2 node:

- Receive commands on `/joint_command`
- Publish joint state on `/joint_state`
- Publish end-effector position on `/ee_pose`

## Skeleton Added

- `ros2_bridge/mujoco_bridge.py`
  - Loads the MuJoCo model
  - Opens a command subscriber
  - Runs `mj_step()` at every timer tick
  - Publishes `JointState` and `Pose`
- `ros2_bridge/commander.py`
  - Example command publisher

## Running

Terminal 1:

```bash
python3 ros2_bridge/mujoco_bridge.py
```

Terminal 2:

```bash
python3 ros2_bridge/commander.py
```

In a real ROS2 environment this should be packaged properly with a `setup.py`, launch files, and message dependencies. The purpose here is to make the integration flow clear.
