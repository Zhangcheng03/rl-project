import os
import time
from envs.gen3_tracking_env import KinovaTrackingEnv, SaveModelCallback
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# 选择算法（可替换为 'PPO', 'SAC', 'TD3', 'DDPG'）
algorithm = "SAC"
n_envs = 10
n_episodes = 100
episode_len = 2500
save_path = "./models"
model_save_name = f"gen3_{algorithm}_model"
save_freq = 100000
render_mode = None  # 关闭渲染加速训练

# 构建环境
def make_env():
    return KinovaTrackingEnv(episode_len=episode_len, render_mode=render_mode)

env = make_vec_env(make_env, n_envs=n_envs)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

save_callback = SaveModelCallback(check_freq=save_freq, save_path=save_path, verbose=1)

ALGOS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG
}
algo_class = ALGOS[algorithm]

if os.path.exists(f"{save_path}/{model_save_name}.zip"):
    print("Loading existing model...")
    model = algo_class.load(f"{save_path}/{model_save_name}", env=env)
    model.learn(total_timesteps=500000, log_interval=4, callback=save_callback, progress_bar=True)
else:
    model = algo_class("MlpPolicy", env, verbose=1, tensorboard_log=f"./tensorboard_{algorithm}")
    total_timesteps = n_episodes * episode_len * n_envs
    model.learn(total_timesteps=total_timesteps, log_interval=4, callback=save_callback, progress_bar=True)

model.save(f"{save_path}/{model_save_name}")
if isinstance(env, VecNormalize):
    env.save(f"{save_path}/vecnorm_{algorithm}.pkl")
