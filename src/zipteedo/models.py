#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models to estimate performance.

A model is any callable that takes as input a datum.
"""

import numpy as np

def ConstantModel(data, rho=1, use_gold=True):
    if use_gold:
        ys = [datum['y*'] for datum in data]
    else:
        ys = [datum['y'] for datum in data]

    # This comes from some silly algebra:
    # rho = (p - p')/sqrt(1 - pp' uu'/σ²)
    # p = ab ± sqrt[ a(b+2)(ab+2) ] + 2 / 2(ab+2);
    #   where a = rho, b = uu'/σ²
    u = np.mean(ys)
    sigma2 = np.std(ys)**2
    a, b = rho, u * (1-u) / sigma2
    p = (a * b + np.sqrt(a * (b+2) * (a*b+2)) + 2)/2 * (a*b+2)

    def ret(datum):
        y = datum['y*'] if use_gold else datum['y']
        if np.random.rand() < p:
            return y
        else:
            return 1-y

    return ret
