#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def compute_jacobian_numeric(q, fk_func, delta=1e-6):
    n = len(q)
    pose0 = fk_func(q)
    J = np.zeros((len(pose0), n))
    for i in range(n):
        dq = q.copy()
        dq[i] += delta
        pose1 = fk_func(dq)
        J[:, i] = (pose1 - pose0) / delta
    return J
