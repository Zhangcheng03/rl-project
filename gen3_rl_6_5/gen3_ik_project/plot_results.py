import numpy as np
import os
import matplotlib.pyplot as plt

# load
os.makedirs("results", exist_ok=True)
results = np.load("results/results.npy")
pos_err = results[:,0]
ori_err = results[:,1]
success_count = int(results[:,3].sum())
fail_count = len(results) - success_count

# error histograms
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(pos_err, bins=20, edgecolor='black')
plt.title("Position Error Distribution")
plt.xlabel("Error [m]")
plt.ylabel("Count")
plt.grid(True)

plt.subplot(1,2,2)
plt.hist(ori_err, bins=20, edgecolor='black')
plt.title("Orientation Error Distribution")
plt.xlabel("Error [rad]")
plt.ylabel("Count")
plt.grid(True)

plt.tight_layout()
plt.savefig("results/ik_error_histograms.png", dpi=300)
plt.close()

# success bar chart
plt.figure(figsize=(10,8))
plt.bar(["Success","Fail"], [success_count, fail_count], edgecolor='black')
plt.title(f"IK Success Rate: {success_count/len(results)*100:.1f}%")
plt.ylabel("Trials")
plt.grid(axis='y')

plt.savefig("results/ik_success_rate_bar.png", dpi=300)
plt.close()

print("Plots saved to results/ directory:")
print(" - ik_error_histograms.png")
print(" - ik_success_rate_bar.png")
