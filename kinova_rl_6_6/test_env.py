#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from envs.kinova_reach_env import KinovaReachEnv


def test_environment():
    print("\n✅ 正在初始化 Kinova 环境...")
    env = KinovaReachEnv(render_mode="human")

    obs, _ = env.reset()
    print("\n✅ 初始观测值 shape:", obs.shape)
    print("✅ 初始观测值示例:", obs[:5], "...")

    env.print_model_info()

    print("\n🚀 执行环境 step 操作 10 步...")
    for i in range(1000):
        action = np.random.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
        obs, reward, done, truncated, info = env.step(action)
        joint_positions = obs[:7]         # 关节位置
        ee_position = obs[14:17]          # 末端位置 (根据 obs 拼接方式: qpos+qvel+ee_pos+error)
        print(f"Step {i+1}: reward={reward:.4f}, done={done}, joint_positions={joint_positions}, ee_position={ee_position}")
        if env.render_mode == "human":
            env.render()

    print("\n✅ 关闭环境")
    env.close()


if __name__ == "__main__":
    test_environment()
