"""MuJoCo'yu ROS2 topic'leri ile sarmalamak icin minimal node iskeleti."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BridgeConfig:
    command_topic: str = "/joint_command"
    state_topic: str = "/joint_state"
    ee_topic: str = "/ee_pose"
    rate_hz: float = 100.0


def main() -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        from geometry_msgs.msg import Pose
        import mujoco
        import numpy as np
    except ModuleNotFoundError as exc:
        print(f"Eksik bagimlilik: {exc}. Bu dosya ROS2+MuJoCo ortami icin iskelettir.")
        return

    class MujocoBridgeNode(Node):
        def __init__(self):
            super().__init__("mujoco_bridge")
            self.cfg = BridgeConfig()
            self.model = mujoco.MjModel.from_xml_path("models/two_link_torque.xml")
            self.data = mujoco.MjData(self.model)
            self.command = np.zeros(self.model.nu)

            self.state_pub = self.create_publisher(JointState, self.cfg.state_topic, 10)
            self.ee_pub = self.create_publisher(Pose, self.cfg.ee_topic, 10)
            self.create_subscription(Float64MultiArray, self.cfg.command_topic, self.on_command, 10)
            self.timer = self.create_timer(1.0 / self.cfg.rate_hz, self.on_timer)

        def on_command(self, msg: Float64MultiArray) -> None:
            count = min(len(msg.data), self.model.nu)
            for idx in range(count):
                self.command[idx] = msg.data[idx]

        def on_timer(self) -> None:
            self.data.ctrl[:] = self.command
            mujoco.mj_step(self.model, self.data)

            state = JointState()
            state.name = ["joint1", "joint2"]
            state.position = [float(self.data.qpos[0]), float(self.data.qpos[1])]
            state.velocity = [float(self.data.qvel[0]), float(self.data.qvel[1])]
            self.state_pub.publish(state)

            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
            pose = Pose()
            pose.position.x = float(self.data.site_xpos[site_id][0])
            pose.position.y = float(self.data.site_xpos[site_id][1])
            pose.position.z = float(self.data.site_xpos[site_id][2])
            self.ee_pub.publish(pose)

    rclpy.init()
    node = MujocoBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
