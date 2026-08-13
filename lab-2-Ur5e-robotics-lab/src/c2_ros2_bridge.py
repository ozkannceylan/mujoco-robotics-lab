"""Phase 4.2 — ROS 2 Bridge for UR5e MuJoCo Simulation.

Provides a ROS 2 node that bridges the MuJoCo UR5e simulation with the
ROS 2 ecosystem. This enables:
  - Publishing joint states from the simulation
  - Publishing end-effector pose
  - Subscribing to joint trajectory commands
  - Integration with MoveIt2 for motion planning

Dependencies (ROS 2 Humble):
  - rclpy
  - sensor_msgs
  - geometry_msgs
  - std_msgs

Run (requires ROS 2 sourced):
    python3 lab-2-Ur5e-robotics-lab/src/c2_ros2_bridge.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ur5e_common import JOINT_NAMES, NUM_JOINTS, Q_HOME
from mujoco_sim import UR5eSimulator


@dataclass(frozen=True)
class BridgeConfig:
    """ROS 2 topic and rate configuration."""
    joint_state_topic: str = "/joint_states"
    ee_pose_topic: str = "/ee_pose"
    joint_command_topic: str = "/joint_commands"
    rate_hz: float = 500.0


def main() -> None:
    """Start the ROS 2 bridge node.

    If ROS 2 dependencies are not available, prints usage information
    and demonstrates the bridge concept without ROS 2.
    """
    try:
        import rclpy
        from geometry_msgs.msg import Pose, Point, Quaternion
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
    except ModuleNotFoundError as exc:
        print("=" * 72)
        print("Phase 4.2 — ROS 2 Bridge (Scaffold)")
        print("=" * 72)
        print(f"\n  ROS 2 dependency not found: {exc}")
        print("\n  This module is a scaffold for ROS 2 integration.")
        print("  To use it, source your ROS 2 Humble workspace first:")
        print("    source /opt/ros/humble/setup.bash")
        print("    python3 lab-2-Ur5e-robotics-lab/src/c2_ros2_bridge.py")
        print("\n  Bridge design:")
        print("    - Publishes /joint_states (sensor_msgs/JointState)")
        print("    - Publishes /ee_pose (geometry_msgs/Pose)")
        print("    - Subscribes to /joint_commands (Float64MultiArray)")
        print("    - Runs MuJoCo simulation at 500 Hz")
        print("\n  Without ROS 2, demonstrating standalone simulation loop...")

        sim = UR5eSimulator()
        sim.set_qpos(Q_HOME)
        print(f"\n  Initial EE position: {sim.get_ee_pos()}")
        for _ in range(1000):
            sim.step()
        state = sim.get_state()
        print(f"  After 1000 steps:    {state.ee_pos}")
        print(f"  Sim time:            {state.time:.3f} s")

        print("\n" + "=" * 72)
        print("Phase 4.2 — ROS 2 Bridge: SCAFFOLD COMPLETE")
        print("=" * 72)
        return

    class UR5eBridgeNode(Node):
        """ROS 2 node bridging MuJoCo UR5e sim to ROS topics."""

        def __init__(self) -> None:
            super().__init__("ur5e_mujoco_bridge")
            self.config = BridgeConfig()
            self.sim = UR5eSimulator()
            self.sim.set_qpos(Q_HOME)
            self.command = np.zeros(NUM_JOINTS)

            self.state_pub = self.create_publisher(
                JointState, self.config.joint_state_topic, 10
            )
            self.pose_pub = self.create_publisher(
                Pose, self.config.ee_pose_topic, 10
            )
            self.create_subscription(
                Float64MultiArray, self.config.joint_command_topic,
                self._on_command, 10
            )

            period = 1.0 / self.config.rate_hz
            self.timer = self.create_timer(period, self._on_timer)
            self.get_logger().info(
                f"UR5e MuJoCo bridge started at {self.config.rate_hz} Hz"
            )

        def _on_command(self, msg: Float64MultiArray) -> None:
            """Handle incoming joint commands."""
            n = min(len(msg.data), NUM_JOINTS)
            for i in range(n):
                self.command[i] = float(msg.data[i])

        def _on_timer(self) -> None:
            """Step sim and publish state."""
            self.sim.set_ctrl(self.command)
            self.sim.step()
            state = self.sim.get_state()

            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = list(JOINT_NAMES)
            js.position = state.qpos.tolist()
            js.velocity = state.qvel.tolist()
            self.state_pub.publish(js)

            pose = Pose()
            pose.position = Point(
                x=float(state.ee_pos[0]),
                y=float(state.ee_pos[1]),
                z=float(state.ee_pos[2]),
            )
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_matrix(state.ee_rot).as_quat()
            pose.orientation = Quaternion(
                x=float(quat[0]), y=float(quat[1]),
                z=float(quat[2]), w=float(quat[3]),
            )
            self.pose_pub.publish(pose)

    rclpy.init()
    node = UR5eBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
