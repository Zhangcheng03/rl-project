import time
import mujoco

def execute_joint_trajectory(model, data, trajectory, rate_hz=50, viewer=None):
    """
    Execute a joint trajectory in Mujoco using control mode.

    Parameters:
    - model: mujoco.MjModel object
    - data: mujoco.MjData object
    - trajectory: (T, 7) array of joint positions
    - rate_hz: control frequency in Hz
    - viewer: optional mujoco.viewer context (for real-time rendering)
    """
    dt = 1.0 / rate_hz
    n_steps = trajectory.shape[0]

    for i in range(n_steps):
        q = trajectory[i]
        data.ctrl[:7] = q
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

        time.sleep(dt)
