import os
import time
from envs.kinova_reach_env import KinovaReachEnv, SaveModelCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from data_plot.ppo_data_plot import RewardStepLogger
from stable_baselines3.common.monitor import Monitor



# 配置参数
algorithm = "PPO"
n_envs = 25
n_episodes = 200
episode_len = 2500
save_path = "./models_6_6"
model_save_name = f"gen3_{algorithm}_optimized"
save_freq = 100000
total_timesteps = n_envs * n_episodes * episode_len
render_mode = None

# 创建多进程并行环境（加速 PPO 收敛）
def make_env():
    def _init():
        env = KinovaReachEnv(episode_len=episode_len, render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    vec_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    vec_env.training = True
    vec_env.norm_reward = True

    # 回调函数
    save_callback = SaveModelCallback(check_freq=save_freq, save_path=save_path, verbose=1)
    log_callback = RewardStepLogger(log_dir=f"./logs_{algorithm}")

    # PPO 训练
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.98,
        learning_rate=3e-4,
        ent_coef=0.01,
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

    # 保存模型
    model.save(f"{save_path}/{model_save_name}")
    vec_env.save(f"{save_path}/vecnorm_{algorithm}.pkl")
