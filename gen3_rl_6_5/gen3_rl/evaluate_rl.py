import numpy as np
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import VecNormalize
from envs.gen3_tracking_env import KinovaTrackingEnv

algorithm = "SAC"
model_path = f"./models/gen3_{algorithm}_model"
vecnorm_path = f"./models/vecnorm_{algorithm}.pkl"
n_episodes = 100

def make_env():
    return KinovaTrackingEnv(render_mode=None)

# 加载环境
env = make_env()
vec_env = VecNormalize.load(vecnorm_path, make_env())
vec_env.training = False
vec_env.norm_reward = False

ALGOS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG
}

model = ALGOS[algorithm].load(model_path, env=vec_env)

success_count = 0
total_rewards = []

for _ in range(n_episodes):
    obs = vec_env.reset()
    done, truncated = False, False
    ep_reward = 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = vec_env.step(action)
        ep_reward += reward
    total_rewards.append(ep_reward)
    if reward > 9:
        success_count += 1

success_rate = success_count / n_episodes
avg_reward = np.mean(total_rewards)

print(f"Algorithm: {algorithm}")
print(f"Success rate: {success_rate:.2f}")
print(f"Average episode reward: {avg_reward:.2f}")
