#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from kinova_fk import ForwardKinematics
from lm_solver import lm_inverse_kinematics

fk_model = ForwardKinematics()
fk_func = lambda q: fk_model.get_end_effector_pose(q)

NUM_TRIALS = 100
results = []

for _ in range(NUM_TRIALS):
    theta_true = np.random.uniform(-np.pi, np.pi, size=7)
    x_d = fk_func(theta_true)
    theta_init = np.random.uniform(-np.pi, np.pi, size=7)

    theta_sol, iters, success = lm_inverse_kinematics(x_d, theta_init, fk_func)
    x_sol = fk_func(theta_sol)

    pos_error = np.linalg.norm(x_d[:3] - x_sol[:3])
    quat_dot = np.dot(x_d[3:], x_sol[3:])
    ori_error = 2 * np.arccos(np.clip(np.abs(quat_dot), -1.0, 1.0))
    results.append([pos_error, ori_error, iters, success])

results = np.array(results)
np.save("results/results.npy", results)

print(f"位置误差均值: {results[:,0].mean():.4f} m")
print(f"姿态误差均值: {results[:,1].mean():.4f} rad")
print(f"平均迭代次数: {results[:,2].mean():.2f}")
print(f"成功率: {np.mean(results[:,3]) * 100:.2f}%")
