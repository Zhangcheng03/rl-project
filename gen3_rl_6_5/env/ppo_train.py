#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from kinova_env import KinovaEnv
from datetime import datetime
import torch
import random
import numpy as np

SEED = 30

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def make_env():
    env = KinovaEnv()
    env = Monitor(env)  # 用于记录每个 episode 的回报和长度
    env.reset(seed=SEED)
    return env


def main():
    # 创建保存路径
    log_dir = os.path.join("logs", "ppo_kinova", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    # 启动环境
    env = DummyVecEnv([make_env])

    # 设置 TensorBoard 和模型保存回调
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_kinova"
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=SEED,
        tensorboard_log=log_dir,
    )

    print("\n🚀 开始训练 PPO 模型...\n")
    model.learn(total_timesteps=100_000, callback=checkpoint_callback)
    print("\n✅ 训练完成！模型保存在:", log_dir)

    # 保存最终模型
    model.save(os.path.join(log_dir, "ppo_kinova_final"))


if __name__ == "__main__":
    main()

