#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from kinova_env import KinovaEnv


def test_joint_controls(env, steps_per_joint=50, amplitude=0.5):
    """
    自动测试每个关节的控制效果。
    依次控制关节运动，观察是否响应。
    """
    for joint_idx in range(7):
        print(f"Testing joint {joint_idx}...")
        for direction in [1, -1]:
            for step in range(steps_per_joint):
                action = np.zeros(8, dtype=np.float32)
                action[joint_idx] = direction * amplitude
                action[7] = 0  # 不控制夹爪

                obs, reward, done, truncated, info = env.step(action)

                if env.render_mode == "human":
                    env.render()
                time.sleep(0.01)
        print(f"Joint {joint_idx} test complete.\n")


def test_gripper_control(env, steps=50, amplitude=1.0):
    """
    测试夹爪开合控制效果。
    """
    print("Testing gripper closing...")
    for _ in range(steps):
        action = np.zeros(8, dtype=np.float32)
        action[7] = amplitude  # 夹爪闭合
        env.step(action)
        if env.render_mode == "human":
            env.render()
        time.sleep(0.01)

    print("Testing gripper opening...")
    for _ in range(steps):
        action = np.zeros(8, dtype=np.float32)
        action[7] = -amplitude  # 夹爪张开
        env.step(action)
        if env.render_mode == "human":
            env.render()
        time.sleep(0.01)

    print("Gripper control test complete.\n")


def main():
    env = KinovaEnv(render_mode="human")
    env.reset()

    print("开始自动关节控制测试...\n")
    test_joint_controls(env, steps_per_joint=40, amplitude=0.6)
    print("所有关节控制测试完成！\n")

    print("开始夹爪控制测试...\n")
    test_gripper_control(env, steps=40, amplitude=1.0)
    print("夹爪控制测试完成！")

    env.close()


if __name__ == "__main__":
    main()
