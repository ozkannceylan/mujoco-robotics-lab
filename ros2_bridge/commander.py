"""Basit ROS2 komut gonderici iskeleti."""

from __future__ import annotations


def main() -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float64MultiArray
    except ModuleNotFoundError as exc:
        print(f"Eksik bagimlilik: {exc}. Bu dosya ROS2 ortami icin iskelettir.")
        return

    class CommanderNode(Node):
        def __init__(self):
            super().__init__("joint_commander")
            self.publisher = self.create_publisher(Float64MultiArray, "/joint_command", 10)
            self.timer = self.create_timer(1.0, self.on_timer)
            self.toggle = False

        def on_timer(self) -> None:
            msg = Float64MultiArray()
            msg.data = [1.0, -0.5] if not self.toggle else [0.2, 0.2]
            self.toggle = not self.toggle
            self.publisher.publish(msg)
            self.get_logger().info(f"Published command: {list(msg.data)}")

    rclpy.init()
    node = CommanderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
