#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as R

class ForwardKinematics():
    def __init__(self):
        (self.D0, self.D1, self.a1, self.D2, self.a2, self.D3, self.a3, self.D4, 
         self.a4, self.D5, self.a5, self.D6, self.a6, self.x, self.y, self.z) = (
            0.1564, -0.1284, 0.0054, -0.0064, -0.2104,
            -0.2104, 0.0064, -0.0064, -0.2084, -0.1059,
            0.0, -0.0, -0.1059, -0.0615, 0.07, 0.12)

    def forward_kinematics(self, q):
        q1, q2, q3, q4, q5, q6, q7 = q
        TB1 = np.array([[np.cos(q1), -np.sin(q1), 0, 0],
                        [-np.sin(q1), -np.cos(q1), 0, 0], 
                        [0, 0, -1, self.D0], [0, 0, 0, 1]])
        T12 = np.array([[np.cos(q2), -np.sin(q2), 0, 0],
                        [0, 0, -1, self.a1],
                        [np.sin(q2), np.cos(q2), 0, self.D1], [0, 0, 0, 1]])
        T23 = np.array([[np.cos(q3), -np.sin(q3), 0, 0],
                        [0, 0, 1, self.a2],
                        [-np.sin(q3), -np.cos(q3), 0, self.D2], [0, 0, 0, 1]])
        T34 = np.array([[np.cos(q4), -np.sin(q4), 0, 0],
                        [0, 0, -1, self.a3],
                        [np.sin(q4), np.cos(q4), 0, self.D3], [0, 0, 0, 1]])
        T45 = np.array([[np.cos(q5), -np.sin(q5), 0, 0],
                        [0, 0, 1, self.a4],
                        [-np.sin(q5), -np.cos(q5), 0, self.D4], [0, 0, 0, 1]])
        T56 = np.array([[np.cos(q6), -np.sin(q6), 0, 0],
                        [0, 0, -1, self.a5],
                        [np.sin(q6), np.cos(q6), 0, self.D5], [0, 0, 0, 1]])
        T67 = np.array([[np.cos(q7), -np.sin(q7), 0, 0],
                        [0, 0, 1, self.a6],
                        [-np.sin(q7), -np.cos(q7), 0, self.D6], [0, 0, 0, 1]])
        T7end = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, self.x], [0, 0, 0, 1]])
        Tend2gripper = np.eye(4)
        Tend2gripper[2, 3] = self.y
        Tend2tool = np.eye(4)
        Tend2tool[2, 3] = self.z
        T = TB1 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67 @ T7end @ Tend2gripper @ Tend2tool
        return T

    def get_end_effector_pose(self, q):
        T = self.forward_kinematics(q)
        pos = T[:3, 3]
        rot = T[:3, :3]
        quat = R.from_matrix(rot).as_quat()  # [x, y, z, w]
        return np.concatenate([pos, quat])
