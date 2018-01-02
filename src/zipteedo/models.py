#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models to estimate performance.

A model is any callable that takes as input a datum.
"""

import numpy as np

def _get_flip_probability(ys, rho):
    # This comes from some silly algebra:
    # rho = (p - p')/sqrt(1 - pp' uu'/σ²)
    # p = 1/2 ± 1/2 sqrt[ 1 - 2 * (1-rho)^2 / (2 - rho^2b) ]
    #   where b = uu'/σ²
    u = np.mean(ys)
    sigma2 = np.std(ys)**2
    b = u * (1-u) / sigma2
    if rho > 0:
        p = 0.5 + 0.5 * np.sqrt(1 - 2 * (1-rho**2)/(2 - rho**2 * b))
    else:
        p = 0.5 - 0.5 * np.sqrt(1 - 2 * (1-rho**2)/(2 - rho**2 * b))
    assert p >= 0 and p <= 1

    return p

def ConstantModel(data, rho=1, use_gold=True):
    if use_gold:
        ys = [datum['y*'] for datum in data]
    else:
        ys = [datum['y'] for datum in data]

    p = _get_flip_probability(ys, rho)

    def ret(data):
        cs = np.random.binomial(1, p, size=(len(data,)))

        if use_gold:
            ys = np.array([datum['y*'] for datum in data])
        else:
            ys = np.array([datum['y'] for datum in data])
        return cs * ys + (1-cs) * ys
    return ret
