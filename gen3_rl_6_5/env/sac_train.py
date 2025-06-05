import os
import random
import numpy as np
import torch
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from kinova_env import KinovaEnv

# ✅ 固定随机种子以保证可复现性
SEED = 30
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ✅ 创建环境包装器

def make_env():
    env = KinovaEnv(render_mode=None)
    env = Monitor(env)
    env.reset(seed=SEED)
    return env


# ✅ 主训练函数

def main():
    log_dir = os.path.join("logs", "sac_kinova", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac_kinova"
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=2,
        seed=SEED,
        tensorboard_log=log_dir,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    print("\n🚀 使用 SAC 开始训练 Kinova 机械臂任务...")
    model.learn(total_timesteps=1500_000, callback=checkpoint_callback)

    model.save(os.path.join(log_dir, "sac_kinova_final"))
    print("\n✅ SAC 训练完成，模型保存在:", log_dir)


if __name__ == "__main__":
    main()