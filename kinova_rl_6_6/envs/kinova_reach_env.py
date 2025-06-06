import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from stable_baselines3.common.callbacks import BaseCallback
import os
import mujoco

def get_site_id(model, site_name):
    site_nameadr = model.site_nameadr
    names = model.names
    nsite = model.nsite

    for i in range(nsite):
        start = site_nameadr[i]
        end = site_nameadr[i + 1] if i + 1 < nsite else len(names)
        name = names[start:end].tobytes().decode('utf-8')
        if name == site_name:
            return i
    raise ValueError(f"Site name '{site_name}' not found.")

class KinovaReachEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100
    }

    def __init__(self, episode_len=2000, reset_noise_scale=1e-2, seed=42, **kwargs):
        utils.EzPickle.__init__(self, reset_noise_scale, **kwargs)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)
        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            os.path.abspath("/home/wqs/kinova_rl/kinova_model/kinova_scene.xml"),
            5,
            observation_space=observation_space,
            **kwargs
        )

        self.episode_len = episode_len
        self.step_number = 0
        self.goal_reached_count = 0
        self._fixed_seed = 42  # 存储种子用于 reset 时使用


        # 📌 限制一个较小、固定的目标平面区域（xy平面上 10cm × 10cm）
        self.goal_plane_center = np.array([0.4, 0.0, 0.9])
        self.goal_plane_size = 0.05  # 半边长 = 5cm

    def _get_obs(self):
        qpos = self.data.qpos.flat[:7]
        qvel = self.data.qvel.flat[:7]

        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")

        ee_pos = self.data.site_xpos[ee_site_id]
        goal_pos = self.data.site_xpos[goal_site_id]
        ee_to_goal = ee_pos - goal_pos

        return np.concatenate([qpos, qvel, ee_to_goal, goal_pos])

    def _set_goal_pose(self):
        # 📌 在 xy 平面上围绕中心点均匀采样，z 固定
        x = self.np_random.uniform(
            self.goal_plane_center[0] - self.goal_plane_size,
            self.goal_plane_center[0] + self.goal_plane_size
        )
        y = self.np_random.uniform(
            self.goal_plane_center[1] - self.goal_plane_size,
            self.goal_plane_center[1] + self.goal_plane_size
        )
        z = self.goal_plane_center[2]  # 固定 z 高度

        goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        self.data.site_xpos[goal_site_id] = np.array([x, y, z])


    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.step_number += 1
        observation = self._get_obs()

        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")

        ee_pos = self.data.site_xpos[ee_site_id]
        goal_pos = self.data.site_xpos[goal_site_id]

        dist = np.linalg.norm(ee_pos - goal_pos)

        # reward = 1.0 / (1.0 + dist) - 0.01 * np.square(action).sum()

        # done = dist < 0.03

        reward = -dist  # 主要目标：减小距离
        if dist < 0.03:
            reward += 10.0  # 成功奖励
        reward -= 0.01 * np.square(action).sum()  # 控制惩罚

        done = dist < 0.03
        truncated = self.step_number > self.episode_len
        truncated = self.step_number > self.episode_len

        if self.render_mode == "human":
            self.render()

        if done:
            self.goal_reached_count += 1
            # reward += 10.0

        return observation, reward, done, truncated, {}

    def reset_model(self):
        self.step_number = 0
        noise = self.np_random.uniform(-self._reset_noise_scale, self._reset_noise_scale, size=self.model.nq)
        qpos = self.init_qpos + noise
        qvel = self.init_qvel + self.np_random.uniform(-self._reset_noise_scale, self._reset_noise_scale, size=self.model.nv)
        self.set_state(qpos, qvel)
        self._set_goal_pose()
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
