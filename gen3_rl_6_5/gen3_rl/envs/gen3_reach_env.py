import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from kinova_fk import ForwardKinematics
from stable_baselines3.common.callbacks import BaseCallback
import os
import mujoco


class KinovaReachEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100
    }

    def __init__(self, episode_len=2000, reset_noise_scale=1e-2, **kwargs):
        utils.EzPickle.__init__(self, reset_noise_scale, **kwargs)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)
        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            os.path.abspath("/home/wqs/gen3_rl_5_26/gen3_rl/kinova_model/scene.xml"),
            5,
            observation_space=observation_space,
            **kwargs
        )

        self.episode_len = episode_len
        self.step_number = 0
        self.goal_reached_count = 0
        self.goal_angles = np.zeros(7)
        self.fk = ForwardKinematics()

        self.workspace_limits = {
            'x': (-0.7, 0.7),
            'y': (-0.7, 0.7),
            'z': (0.8, 1.2)
        }

    def _get_obs(self):
        qpos = self.data.qpos.flat[:7]
        qvel = self.data.qvel.flat[:7]

        ee_pose = self.fk.forward_kinematics(qpos)
        ee_pos = ee_pose[:3, 3]

        goal_pose = self.fk.forward_kinematics(self.goal_angles)
        goal_pos = goal_pose[:3, 3]

        ee_to_goal = ee_pos - goal_pos

        return np.concatenate([qpos, qvel, ee_to_goal, goal_pos])

    def _set_sphere_position(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sphere")
        x = np.random.uniform(*self.workspace_limits['x'])
        y = np.random.uniform(*self.workspace_limits['y'])
        z = np.random.uniform(*self.workspace_limits['z'])
        self.model.body_pos[body_id] = np.array([x, y, z])

    def _set_goal_pose(self):
        while True:
            joints = [f"joint_{i+1}" for i in range(7)]
            angles = []
            for joint_name in joints:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                low = self.model.jnt_range[joint_id, 0] if self.model.jnt_limited[joint_id] else -np.pi
                high = self.model.jnt_range[joint_id, 1] if self.model.jnt_limited[joint_id] else np.pi
                angles.append(np.random.uniform(low, high))
            self.goal_angles = np.array(angles)
            T = self.fk.forward_kinematics(self.goal_angles)
            T[:3, 3] += np.array([0.0, 0.0, 0.6])  # table offset
            x, y, z = T[0, 3], T[1, 3], T[2, 3]
            if (self.workspace_limits['x'][0] <= x <= self.workspace_limits['x'][1] and
                self.workspace_limits['y'][0] <= y <= self.workspace_limits['y'][1] and
                self.workspace_limits['z'][0] <= z <= self.workspace_limits['z'][1]):
                self._label_goal_pose(np.array([x, y, z]))
                return self.goal_angles, np.array([x, y, z])

    def _label_goal_pose(self, position):
        goal_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.model.body_pos[goal_marker_id] = position

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.step_number += 1
        observation = self._get_obs()

        qpos = self.data.qpos.flat[:7]
        ee_pose = self.fk.forward_kinematics(qpos)
        ee_pos = ee_pose[:3, 3]

        goal_pose = self.fk.forward_kinematics(self.goal_angles)
        goal_pos = goal_pose[:3, 3]

        dist = np.linalg.norm(ee_pos - goal_pos)
        reward = 1.0 / (1.0 + dist) - 0.01 * np.square(action).sum()

        done = dist < 0.02
        truncated = self.step_number > self.episode_len

        if self.render_mode == "human":
            self.render()

        if done:
            self.goal_reached_count += 1
            reward += 10.0

        return observation, reward, done, truncated, {}

    def reset_model(self):
        self.step_number = 0
        noise = self.np_random.uniform(-self._reset_noise_scale, self._reset_noise_scale, size=self.model.nq)
        qpos = self.init_qpos + noise
        qvel = self.init_qvel + self.np_random.uniform(-self._reset_noise_scale, self._reset_noise_scale, size=self.model.nv)
        self.set_state(qpos, qvel)
        self._set_goal_pose()
        self._set_sphere_position()
        return self._get_obs()

class SaveModelCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.save(f"{self.save_path}/model_{self.num_timesteps}")
            if self.verbose > 0:
                print(f"Saving model at timestep {self.num_timesteps}")
        return True
