import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置Seaborn主题风格
sns.set(style="whitegrid", font_scale=1.2)

# 加载评估数据
results = np.load("results/results.npy")
pos_err = results[:, 0]
ori_err = results[:, 1]
success = results[:, 3]

# 图1: KDE密度图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.kdeplot(pos_err, fill=True, color="dodgerblue", linewidth=2)
plt.title("Position Error KDE")
plt.xlabel("Error [cm]")
plt.ylabel("Density")

plt.subplot(1, 2, 2)
sns.kdeplot(ori_err, fill=True, color="coral", linewidth=2)
plt.title("Orientation Error KDE")
plt.xlabel("Error [rad]")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("results/kde_error_distributions.png", dpi=300)
plt.close()

# 图2: 误差联合散点图
plt.figure(figsize=(6, 5))
plt.scatter(pos_err, ori_err, c='purple', alpha=0.6, edgecolors='black')
plt.title("Position vs Orientation Error")
plt.xlabel("Position Error [cm]")
plt.ylabel("Orientation Error [rad]")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/joint_error_scatter.png", dpi=300)
plt.close()

# 图3: 成功率饼图
success_count = int(success.sum())
fail_count = len(success) - success_count
labels = ["Success", "Failure"]
sizes = [success_count, fail_count]
colors = ["mediumseagreen", "lightcoral"]

plt.figure(figsize=(4.5, 4.5))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("IK Solve Success Rate")
plt.tight_layout()
plt.savefig("results/ik_success_pie_chart.png", dpi=300)
plt.close()

print("Enhanced plots saved in results/:")
print("- kde_error_distributions.png")
print("- joint_error_scatter.png")
print("- ik_success_pie_chart.png")
