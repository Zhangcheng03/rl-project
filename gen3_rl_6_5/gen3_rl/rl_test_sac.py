import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.gen3_reach_env import KinovaReachEnv
import time




# 设置参数
algorithm = "SAC"
model_path = f"./models/model_1200000.zip"
vecnorm_path = f"./models/vecnorm_{algorithm}.pkl"
n_episodes = 10
render = True

# 创建环境包装函数（需与训练时保持一致）
def make_env():
    return KinovaReachEnv(render_mode="human" if render else None)



# 加载环境与VecNormalize
raw_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load(vecnorm_path, raw_env)
vec_env.training = False
vec_env.norm_reward = False

# 加载模型
model = SAC.load(model_path, env=vec_env)

# 测试循环
success_count = 0
all_rewards = []
all_steps = []

for ep in range(n_episodes):
    obs = vec_env.reset()
    done, truncated = False, False
    ep_reward = 0
    ep_step = 0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        ep_reward += reward
        ep_step += 1
        if render:
            time.sleep(0.01)

    all_rewards.append(ep_reward)
    all_steps.append(ep_step)
    if reward > 9:  # 成功定义：接近目标
        success_count += 1

# 打印总结结果
print("================ Evaluation Summary ================")
print(f"Algorithm: {algorithm}")
print(f"Episodes: {n_episodes}")
print(f"Success Rate: {success_count / n_episodes:.2f}")
print(f"Average Reward: {np.mean(all_rewards):.2f}")
print(f"Average Steps per Episode: {np.mean(all_steps):.1f}")
