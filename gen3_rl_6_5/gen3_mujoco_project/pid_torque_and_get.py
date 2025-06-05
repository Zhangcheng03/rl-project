import mujoco
from mujoco import viewer
import numpy as np
import time
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

def main():
    xml_path = "/home/wqs/gen3_rl_5_26/gen3_model/gen3_with_gripper.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    pid = PIDController(kp=20.0, ki=0.1, kd=3.0)
    target_position = np.pi / 2
    joint_id = 0

    positions = []
    torques = []

    max_steps = 2000
    stable_counter = 0
    stable_threshold = 20

    with viewer.launch_passive(model, data) as v:
        for step in range(max_steps):
            current_position = data.qpos[joint_id]
            error = target_position - current_position
            torque = pid.update(error, dt)
            data.ctrl[joint_id] = torque
            mujoco.mj_step(model, data)

            v.sync()

            positions.append(current_position)
            torques.append(torque)

            if abs(error) < 0.001:
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter > stable_threshold:
                print("Target reached and stabilized.")
                break

            time.sleep(dt)
        else:
            print("Max steps reached, stopping.")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(positions, label='Position')
    plt.axhline(y=target_position, color='r', linestyle='--', label='Target')
    plt.legend()
    plt.title('Joint Position')

    plt.subplot(2, 1, 2)
    plt.plot(torques, label='Torque')
    plt.legend()
    plt.title('Control Torque')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
