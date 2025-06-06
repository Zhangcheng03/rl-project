import os
import time
from envs.gen3_tracking_env import KinovaTrackingEnv, SaveModelCallback
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from data_plot.TD3_data_plot import RewardStepLogger
import numpy as np

# 参数配置
algorithm = "TD3"  # 可切换为 "TD3"
model_cls = TD3 
n_episodes = 5000
episode_len = 2000
save_path = "./models"
model_save_name = f"gen3_{algorithm}_optimized"
save_freq = 100000
total_timesteps = episode_len * n_episodes
render_mode = None

# 构造环境
def make_env():
    return KinovaTrackingEnv(episode_len=episode_len, render_mode=render_mode)

raw_env = DummyVecEnv([make_env])
vec_env = VecNormalize(raw_env, norm_obs=True, norm_reward=True)
vec_env.training = True
vec_env.norm_reward = True

# 动作噪声（仅 TD3 使用）
action_noise = None
if algorithm == "TD3":
    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 回调
save_callback = SaveModelCallback(check_freq=save_freq, save_path=save_path, verbose=1)
log_callback = RewardStepLogger(log_dir=f"./logs_{algorithm}")

# 创建模型
model = model_cls(
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
    action_noise=action_noise,
    tensorboard_log=f"./tensorboard_{algorithm}"
)

start = time.time()
model.learn(
    total_timesteps=total_timesteps,
    log_interval=4,
    callback=[save_callback, log_callback],
    progress_bar=True
)
end = time.time()
print(f"Training complete in {(end - start)/60:.2f} minutes")

# 保存
model.save(f"{save_path}/{model_save_name}")
vec_env.save(f"{save_path}/vecnorm_{algorithm}.pkl")
