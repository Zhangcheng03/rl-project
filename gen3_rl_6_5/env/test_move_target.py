#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from kinova_env import KinovaEnv
import mujoco


def compute_jacobian(env, qpos):
    mujoco.mj_forward(env.model, env.data)
    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))
    ee_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
    mujoco.mj_jacSite(env.model, env.data, jacp, jacr, ee_site_id)
    return jacp[:, :7]


def move_to_target_position(env, target_pos, threshold=0.02, max_steps=300):
    print("\n🎯 控制末端移动至目标位置:", target_pos)
    for step in range(max_steps):
        qpos = env.data.qpos[:7].copy()
        ee_pos = env.fk.forward_kinematics(qpos)[:3, 3]
        error = target_pos - ee_pos

        J = compute_jacobian(env, qpos)
        dq = np.linalg.pinv(J) @ error

        action = np.zeros(8, dtype=np.float32)
        action[:7] = dq * 2.0
        action = np.clip(action, -1.0, 1.0)

        obs, reward, done, truncated, info = env.step(action)
        if env.render_mode == "human":
            env.render()
        time.sleep(0.01)

        print(f"Step {step+1}: ee_pos={ee_pos}, error_norm={np.linalg.norm(error):.4f}")

        if np.linalg.norm(error) < threshold:
            print("✅ 已到达目标末端位置！")
            break


def test_joint_controls(env, steps_per_joint=50, amplitude=0.5):
    print("\n开始测试关节控制...")
    for joint_idx in range(7):
        print(f"Testing joint {joint_idx}...")
        for direction in [1, -1]:
            for step in range(steps_per_joint):
                action = np.zeros(8, dtype=np.float32)
                action[joint_idx] = direction * amplitude
                action[7] = 0
                env.step(action)
                if env.render_mode == "human":
                    env.render()
                time.sleep(0.01)
        print(f"Joint {joint_idx} test complete.\n")


def test_gripper_control(env, steps=50, amplitude=1.0):
    print("\n开始测试夹爪控制...")
    print("Closing gripper...")
    for _ in range(steps):
        action = np.zeros(8, dtype=np.float32)
        action[7] = amplitude
        env.step(action)
        if env.render_mode == "human":
            env.render()
        time.sleep(0.01)

    print("Opening gripper...")
    for _ in range(steps):
        action = np.zeros(8, dtype=np.float32)
        action[7] = -amplitude
        env.step(action)
        if env.render_mode == "human":
            env.render()
        time.sleep(0.01)

    print("Gripper control test complete.\n")


def main():
    env = KinovaEnv(render_mode="human")
    env.reset()

    print("\n开始自动关节控制测试...")
    test_joint_controls(env, steps_per_joint=30, amplitude=0.6)
    print("所有关节控制测试完成！\n")

    print("开始夹爪控制测试...")
    test_gripper_control(env, steps=30, amplitude=1.0)
    print("夹爪控制测试完成！\n")

    print("开始末端到点控制测试...")
    target = np.array([0.45, -0.1, 0.35])
    move_to_target_position(env, target)
    print("末端控制测试完成！")

    env.close()


if __name__ == "__main__":
    main()