#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from jacobian import compute_jacobian_numeric

def lm_inverse_kinematics(x_d, theta_init, fk_func, max_iter=100, tol=1e-4, lambda_init=1e-2):
    theta = theta_init.copy()
    lam = lambda_init
    for i in range(max_iter):
        x = fk_func(theta)
        e = x_d - x
        J = compute_jacobian_numeric(theta, fk_func)
        H = J.T @ J + lam * np.eye(J.shape[1])
        g = J.T @ e
        delta_theta = np.linalg.solve(H, g)

        theta_new = theta + delta_theta
        x_new = fk_func(theta_new)
        e_new = x_d - x_new

        if np.linalg.norm(e_new) < tol:
            return theta_new, i+1, True

        if np.linalg.norm(e_new) < np.linalg.norm(e):
            theta = theta_new
            lam /= 10
        else:
            lam *= 10

    return theta, max_iter, False
