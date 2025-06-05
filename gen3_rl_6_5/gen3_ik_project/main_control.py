import os
import time
import numpy as np
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation as R
from kinova_fk import ForwardKinematics
from lm_solver import lm_inverse_kinematics
from execute_trajectory import execute_joint_trajectory

# -----------------------------------------------------------------------------
# Main control script for Kinova Gen3 with Mujoco 3.3.2:
# - Initializes Mujoco simulation
# - Plans joint-space trajectory via IK
# - Executes trajectory
# -----------------------------------------------------------------------------

from scipy.spatial.transform import Rotation as R, Slerp


def interpolate_pose_linear(pose_start, pose_goal, steps):
    pos_start, quat_start = pose_start[:3], pose_start[3:]
    pos_goal, quat_goal = pose_goal[:3], pose_goal[3:]

    pos_traj = np.linspace(pos_start, pos_goal, steps)

    key_times = [0, 1]
    key_rots = R.from_quat([quat_start, quat_goal])
    slerp = Slerp(key_times, key_rots)
    interp_rots = slerp(np.linspace(0, 1, steps)).as_quat()

    return np.hstack([pos_traj, interp_rots])  # shape (steps, 7)


def main():
    model_path = os.path.join("/home/wqs/gen3_rl_5_26/env/kinova_model/scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with viewer.launch_passive(model, data) as v:
        v.close()
        fk = ForwardKinematics()
        fk_func = lambda q: fk.get_end_effector_pose(q)

        q_goal = np.random.uniform(-np.pi, np.pi, size=7)
        x_goal = fk_func(q_goal)

        mujoco.mj_resetData(model, data)
        mujoco.mj_step(model, data)
        q_start = data.qpos[:7].copy()
        x_start = fk_func(q_start)

        num_points = 20
        traj_ee = interpolate_pose_linear(x_start, x_goal, steps=num_points)

        joint_traj = np.zeros((num_points, 7))
        for i, x_d in enumerate(traj_ee):
            theta_init = joint_traj[i-1] if i > 0 else q_start
            theta_sol, _, success = lm_inverse_kinematics(x_d, theta_init, fk_func, max_iter=200, tol=1e-3)
            if not success:
                print(f"Warning: IK failed at waypoint {i}")
            joint_traj[i] = theta_sol

        print("Starting trajectory execution...")
        execute_joint_trajectory(model, data, joint_traj, rate_hz=50, viewer=v)
        print("Done.")

if __name__ == '__main__':
    main()
