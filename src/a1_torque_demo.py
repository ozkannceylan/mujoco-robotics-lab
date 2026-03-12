"""A1 Torque Control Demo: We send torque directly via motor actuator.

Position actuator vs Motor actuator difference:
  - position: ctrl = target angle (rad), internal PD controller computes torque
  - motor:    ctrl = direct torque (N.m), physics simulates the effect

In this demo we use the motor actuator. As torque is applied the robot accelerates,
when torque is removed it slows to a stop via damping. Closer to real robot control.
"""
import numpy as np
import mujoco
import mujoco.viewer
import time

MODEL_PATH = "models/two_link_torque.xml"


def demo_constant_torque():
    """Demo 1: Constant torque → robot accelerates, reaches max speed via damping."""
    print("\n--- DEMO 1: Constant Torque ---")
    print("Applying 2 N.m torque to joint1. Robot starts to rotate.")
    print("After 3 seconds torque = 0 → slows down via damping.\n")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            t = step * model.opt.timestep

            # First 3 seconds: apply torque to joint1
            if t < 3.0:
                data.ctrl[0] = 2.0   # 2 N.m torque
                data.ctrl[1] = 0.0
            # 3-6 seconds: remove torque, slow down via damping
            elif t < 6.0:
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0
            # 6-9 seconds: torque in opposite direction
            elif t < 9.0:
                data.ctrl[0] = -3.0
                data.ctrl[1] = 1.5
            else:
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()

            if step % 500 == 0:
                print(
                    f"t={t:5.1f}s | "
                    f"torque=[{data.ctrl[0]:+5.1f}, {data.ctrl[1]:+5.1f}] N.m | "
                    f"pos=[{np.degrees(data.qpos[0]):+7.1f}, {np.degrees(data.qpos[1]):+7.1f}]° | "
                    f"vel=[{np.degrees(data.qvel[0]):+7.1f}, {np.degrees(data.qvel[1]):+7.1f}]°/s"
                )

            step += 1
            # Real-time pacing
            target_time = step * model.opt.timestep
            actual_time = time.time()
            if hasattr(demo_constant_torque, 't0'):
                sleep_time = demo_constant_torque.t0 + target_time - actual_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                demo_constant_torque.t0 = actual_time


def demo_manual_pd():
    """Demo 2: We compute torque with our own PD controller.
    This is the same thing a position actuator does internally —
    but now we are in control.
    """
    print("\n--- DEMO 2: Manual PD Controller (via torque) ---")
    print("Target: joint1=90°, joint2=-45°")
    print("tau = Kp*(q_desired - q) + Kd*(0 - qdot)\n")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # PD gains
    Kp = np.array([20.0, 15.0])   # proportional
    Kd = np.array([2.0, 1.5])     # derivative (damping)

    # Target angles
    q_desired = np.array([np.radians(90), np.radians(-45)])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        t0 = time.time()

        while viewer.is_running():
            t = step * model.opt.timestep

            # PD control law: tau = Kp * error + Kd * (-velocity)
            error = q_desired - data.qpos
            torque = Kp * error + Kd * (0 - data.qvel)

            # Torque limits (ctrlrange: -5, 5)
            torque = np.clip(torque, -5, 5)

            data.ctrl[0] = torque[0]
            data.ctrl[1] = torque[1]

            mujoco.mj_step(model, data)
            viewer.sync()

            if step % 500 == 0:
                print(
                    f"t={t:5.1f}s | "
                    f"error=[{np.degrees(error[0]):+6.1f}, {np.degrees(error[1]):+6.1f}]° | "
                    f"torque=[{torque[0]:+5.2f}, {torque[1]:+5.2f}] N.m | "
                    f"pos=[{np.degrees(data.qpos[0]):+7.1f}, {np.degrees(data.qpos[1]):+7.1f}]°"
                )

            step += 1
            elapsed = time.time() - t0
            sleep_time = (step * model.opt.timestep) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def demo_sinusoidal_torque():
    """Demo 3: Sinusoidal torque → oscillatory motion.
    Does not converge to a target like a position actuator; keeps oscillating.
    """
    print("\n--- DEMO 3: Sinusoidal Torque ---")
    print("Periodic torque → robot oscillates but doesn't 'converge' to a target.\n")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        t0 = time.time()

        while viewer.is_running():
            t = step * model.opt.timestep

            # Sinusoidal torque
            data.ctrl[0] = 3.0 * np.sin(2 * np.pi * 0.5 * t)
            data.ctrl[1] = 2.0 * np.sin(2 * np.pi * 0.8 * t)

            mujoco.mj_step(model, data)
            viewer.sync()

            if step % 500 == 0:
                print(
                    f"t={t:5.1f}s | "
                    f"torque=[{data.ctrl[0]:+5.2f}, {data.ctrl[1]:+5.2f}] N.m | "
                    f"pos=[{np.degrees(data.qpos[0]):+7.1f}, {np.degrees(data.qpos[1]):+7.1f}]° | "
                    f"vel=[{np.degrees(data.qvel[0]):+7.1f}, {np.degrees(data.qvel[1]):+7.1f}]°/s"
                )

            step += 1
            elapsed = time.time() - t0
            sleep_time = (step * model.opt.timestep) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    print("=" * 60)
    print("TORQUE CONTROL DEMO")
    print("=" * 60)
    print()
    print("Options:")
    print("  1 — Constant torque (acceleration + stop)")
    print("  2 — Manual PD controller (via torque)")
    print("  3 — Sinusoidal torque (oscillation)")
    print()

    choice = input("Which demo? (1/2/3): ").strip()

    demos = {"1": demo_constant_torque, "2": demo_manual_pd, "3": demo_sinusoidal_torque}
    if choice in demos:
        demos[choice]()
    else:
        print("Invalid choice. Enter 1, 2, or 3.")
