import numpy as np
import matplotlib.pyplot as plt

# 定义起始和目标关节角（7个自由度）
q_start = np.array([0.0, -0.5, 0.3, -1.2, 1.0, -0.8, 0.5])
q_goal = np.array([0.3, -0.2, 0.6, -0.8, 0.7, -1.0, 0.8])

# 五次多项式平滑插值
def polynomial_smooth_traj(q0, qf, steps=100):
    t = np.linspace(0, 1, steps)
    s = 10*t**3 - 15*t**4 + 6*t**5
    traj = np.outer(1 - s, q0) + np.outer(s, qf)
    return traj

# 生成轨迹
trajectory = polynomial_smooth_traj(q_start, q_goal, steps=100)

# 绘图
plt.figure(figsize=(10, 6))
for i in range(7):
    plt.plot(trajectory[:, i], label=f'Joint {i+1}')
plt.title("Joint Trajectory (5th Order Interpolation)")
plt.xlabel("Step")
plt.ylabel("Joint Angle [rad]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/joint_trajectory_example.png", dpi=300)
plt.show()
