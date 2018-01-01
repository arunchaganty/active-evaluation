"""
Active evaluation methods.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from tqdm import tqdm, trange

from .util import StatCounter

def simple_estimator(data, seed=None):
    rng = np.random.RandomState(seed)
    rng.shuffle(data)

    ret, stats = [], StatCounter()
    for datum in data:
        fh = datum['y']
        stats += fh
        ret.append([stats.mean, stats.var])
    return ret

def model_baseline_estimator(data, model, seed=None):
    rng = np.random.RandomState(seed)
    rng.shuffle(data)

    ret, stats = [], StatCounter()
    for datum in data:
        fh = datum['y']
        gh = model(datum)
        stats += fh - gh
        ret.append([stats.mean, stats.var])
    return ret

def gaussian_process_estimator(X, y, X_old=None, y_old=None, kernel=None):
    """
    @data is a set of [x,y] pairs.
    @returns a list with the current estimate of \E[y].
    """
    gp = GaussianProcessRegressor(kernel=kernel)
    n = len(y)

    ret = []
    for i in trange(n):
        if X_old is not None and y_old is not None:
            gp.fit(np.vstack([X_old,X[:i+1]]), np.vstack([y_old,y[:i+1]]))
        else:
            gp.fit(X[:i+1], y[:i+1])

        y_, std_ = gp.predict(X, return_std=True)
        mu_ = StatCounter(zip(y_, 1./(1+std_)**2)).mean
        mu_ = np.mean(y_)
        ret.append(mu_)
    return ret, gp
