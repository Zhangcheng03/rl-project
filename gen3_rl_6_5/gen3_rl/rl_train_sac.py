import os
import time
import numpy as np
from envs.gen3_reach_env import KinovaReachEnv, SaveModelCallback
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# 使用 SAC 算法
algorithm = "SAC"
n_envs = 4
n_episodes = 150
episode_len = 2000
total_timesteps = n_episodes * episode_len * n_envs
save_path = "./models"
model_save_name = f"gen3_{algorithm}_optimized"
save_freq = 100000
render_mode = None

# 创建单环境
def make_env():
    return Monitor(KinovaReachEnv(episode_len=episode_len, render_mode=render_mode))

base_env = DummyVecEnv([make_env])
vec_env = VecNormalize(base_env, norm_obs=True, norm_reward=True)
vec_env.training = True
vec_env.norm_reward = True

save_callback = SaveModelCallback(check_freq=save_freq, save_path=save_path, verbose=1)

# 推荐的网络结构和参数设置
policy_kwargs = dict(net_arch=[256, 256])

model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=300000,
    batch_size=256,
    tau=0.005,
    gamma=0.98,
    train_freq=1,
    gradient_steps=1,
    learning_starts=10000,
    policy_kwargs=policy_kwargs,
    tensorboard_log=f"./tensorboard_{algorithm}"
)

start_time = time.time()
model.learn(total_timesteps=total_timesteps, log_interval=4, callback=save_callback, progress_bar=True)
end_time = time.time()

print(f"Training took {(end_time - start_time) / 60:.2f} minutes")

# 保存模型和归一化器
model.save(f"{save_path}/{model_save_name}")
vec_env.save(f"{save_path}/vecnorm_{algorithm}.pkl")
