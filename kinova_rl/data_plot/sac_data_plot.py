import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class RewardStepLogger(BaseCallback):
    def __init__(self, log_dir: str, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.rewards = []
        self.steps = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                ep_len = info["episode"]["l"]
                self.rewards.append(ep_rew)
                self.steps.append(ep_len)
        return True

    def _on_training_end(self) -> None:
        rewards = np.array(self.rewards)
        steps = np.array(self.steps)

        np.save(os.path.join(self.log_dir, "episode_rewards.npy"), rewards)
        np.save(os.path.join(self.log_dir, "episode_lengths.npy"), steps)

        # 分图保存
        plt.figure(figsize=(8, 5))
        plt.plot(rewards, label="Episode Reward", color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("SAC Training: Episode Reward Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "episode_rewards.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(steps, label="Episode Length", color="green")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("SAC Training: Episode Step Count Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "episode_lengths.png"))
        plt.close()