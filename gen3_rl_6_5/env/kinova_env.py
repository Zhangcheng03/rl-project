#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym 
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium import utils
from kinova_fk import ForwardKinematics
from stable_baselines3.common.callbacks import BaseCallback
import os
import mujoco


class KinovaEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes" : [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps" : 100
    }

    def __init__(self, task="reach", frame_skip=5, reset_noise_scale=1e-2, episode_len=2000, **kwargs):


        utils.EzPickle.__init__(self, task , frame_skip, reset_noise_scale, episode_len, **kwargs)

        self.task = task
        self.reset_noise_scale = reset_noise_scale
        self.episode_len = episode_len
        self.step_number = 0


        obs_dim = 20
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)



        MujocoEnv.__init__(
            self,
            os.path.abspath("/home/wqs/gen3_rl_5_26/gen3_model/gen3_with_gripper.xml"),
            frame_skip,
            observation_space=self.observation_space,
            **kwargs
        )

        self.fk = ForwardKinematics()

        self.goal_position = np.zeros(3)
        self.goal_index = 0
        self.grid_goals = self._generate_grid_goals(x_range=(0.40, 0.45), y_range=(-0.1, 0.1), z=0.15, num_per_axis=3)
        
        self.workspace_limits = {
            'x': (-0.7, 0.7),
            'y': (-0.7, 0.7),
            'z': (0.0, 1.2)
        }

    def _generate_grid_goals(self, x_range, y_range, z, num_per_axis):
        x_vals = np.linspace(x_range[0], x_range[1], num_per_axis)
        y_vals = np.linspace(y_range[0], y_range[1], num_per_axis)
        grid_goals = []
        for x in x_vals:
            for y in y_vals:
                grid_goals.append(np.array([x, y, z]))
        return grid_goals


    def _set_goal(self):

        self.goal_position = self.grid_goals[self.goal_index]
        self.goal_index = (self.goal_index + 1) % len(self.grid_goals)

        goal_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if goal_marker_id != -1:
            self.model.body_pos[goal_marker_id] = self.goal_position
        else:
            raise ValueError("Target body 'target' not found in MuJoCo model.")

    def _get_obs(self):
        qpos = self.data.qpos[:7]
        qvel = self.data.qvel[:7]
        ee_pos = self.fk.forward_kinematics(qpos)[:3, 3]
        return np.concatenate([qpos, qvel, ee_pos, self.goal_position - ee_pos])


    def step(self, action):
        # print(f"[step] step_number={self.step_number}", flush=True)



        ctrl = np.zeros(self.model.nu)
        ctrl[:7] = 1.0 * np.tanh(action[:7]) 
        ctrl[7] = 127.5 + 127.5 * action[7]  
        self.data.ctrl[:] = ctrl

        self.do_simulation(ctrl, self.frame_skip)
        self.step_number += 1

        obs = self._get_obs()
        qpos = self.data.qpos[:7]
        ee_pos = self.fk.forward_kinematics(qpos)[:3, 3]
        ee_pos = self.fk.forward_kinematics(qpos)[:3, 3]
        # print(f"[obs] ee_pos={ee_pos}", flush=True)



        dist = np.linalg.norm(ee_pos - self.goal_position)
        reward = -dist + np.exp(-10 * dist) - 0.01 * np.square(action).sum()
        done = dist < 0.03 or self.step_number > self.episode_len
        return obs, reward, done, False, {}

    def reset_model(self):
        # print(f"[reset_model] resetting state...", flush=True)


        self.step_number = 0
        noise = self.reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=-noise, high=noise, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-noise, high=noise, size=self.model.nv)
        self.set_state(qpos, qvel)
        self._set_goal()
        observation = self._get_obs()
        return observation
    
    def print_model_info(self):
        print("qpos shape:", self.data.qpos.shape)
        print("qvel shape:", self.data.qvel.shape)
        print("ctrl shape:", self.data.ctrl.shape)
        print("Sample qpos:", self.data.qpos[:10])
        print("Sample qvel:", self.data.qvel[:10])

        a = [self.model.name_jntadr[i] for i in range(self.model.njnt)]
        b = [self.model.name_bodyadr[i] for i in range(self.model.nbody)]
        n_obj = self.model.njnt
        m_obj = self.model.nbody

        # Print joint and body names
        id2name = {i: None for i in range(n_obj)}
        name2id = {}

        id2name2 = {j: None for j in range(m_obj)}
        name2id2 = {}

        for count in a:
            name = self.model.names[count:].split(b"\x00")[0].decode()
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert 0 <= obj_id < n_obj and id2name[obj_id] is None
            name2id[name] = obj_id
            id2name[obj_id] = name

        for count2 in b:
            name = self.model.names[count2:].split(b"\x00")[0].decode()
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert 0 <= obj_id < m_obj and id2name2[obj_id] is None
            name2id2[name] = obj_id
            id2name2[obj_id] = name

        print("Joint names:", tuple(id2name[id] for id in sorted(name2id.values())))
        print("Joint name to ID mapping:", name2id)
        print("Joint ID to name mapping:", id2name)

        print("Body names:", tuple(id2name2[id] for id in sorted(name2id.values())))
        print("Body name to ID mapping:", name2id2)
        print("Body ID to name mapping:", id2name2)

    def test_joint_controls(self, steps_per_joint=50, amplitude=0.5):

        import time

        for joint_idx in range(7):
            print(f"Testing joint {joint_idx}...")
            for direction in [1, -1]:
                for step in range(steps_per_joint):
                    action = np.zeros(8)
                    action[joint_idx] = direction * amplitude  
                    action[7] = 0  
                    obs, reward, done, truncated, info = self.step(action)

                    if self.render_mode == "human":
                        self.render()
                    time.sleep(0.01)  
            print(f"Joint {joint_idx} test complete.")

    

class SaveModelCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.save(f"{self.save_path}/model_{self.num_timesteps}")
            if self.verbose > 0:
                print(f"Saving model at timestep {self.num_timesteps}")
        return True