"""A1 Interactive Demo: Move the robot arm directly.
We set qpos directly without using actuators.
"""
import numpy as np
import mujoco
import mujoco.viewer
import time

MODEL_PATH = "models/two_link.xml"


def run_demo():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    print("Opening viewer... Moving the robot.")
    print("Close the window or press Ctrl+C to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        dt = 0.01  # 10ms step

        while viewer.is_running():
            # Set joint angles DIRECTLY (no actuator, no physics)
            data.qpos[0] = np.radians(90) * np.sin(2 * np.pi * 0.3 * t)   # ±90° @ 0.3 Hz
            data.qpos[1] = np.radians(60) * np.sin(2 * np.pi * 0.5 * t)   # ±60° @ 0.5 Hz

            # Update kinematics (without stepping, just compute positions)
            mujoco.mj_forward(model, data)

            # Update viewer
            viewer.sync()

            # Print to terminal
            if int(t * 4) != int((t - dt) * 4):  # every 0.25s
                ee = data.site_xpos[ee_id]
                print(
                    f"t={t:5.1f}s | "
                    f"q1={np.degrees(data.qpos[0]):+7.1f}° | "
                    f"q2={np.degrees(data.qpos[1]):+7.1f}° | "
                    f"EE=({ee[0]:.3f}, {ee[1]:.3f})"
                )

            t += dt
            time.sleep(dt)


if __name__ == "__main__":
    run_demo()
