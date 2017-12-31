"""
Active evaluation methods.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from tqdm import tqdm, trange

def weighted_average(xs, ws):
    ret, w_acc = 0, 0
    for x, w in zip(xs, ws):
        ret += w*(x - ret)/(w_acc + w)
        w_acc += w
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
        mu_ = weighted_average(y_, 1./(1+std_)**2)
        mu_ = np.mean(y_)
        ret.append(mu_)
    return ret, gp
