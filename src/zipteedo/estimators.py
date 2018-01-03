"""
Active evaluation methods.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from tqdm import tqdm, trange

from .util import StatCounter

def simple(_, **__):
    def _ret(data, seed=None):
        rng = np.random.RandomState(seed)
        rng.shuffle(data)

        fs = np.array([datum['y'] for datum in data])
        ret = np.cumsum(fs) / np.arange(1, len(fs)+1)
        return ret
    return _ret

def model_baseline(model, data, baseline_samples=1.0, **_):
    def _ret(data, seed=None):
        rng = np.random.RandomState(seed)
        rng.shuffle(data)

        fs = np.array([datum['y'] for datum in data])
        gs = model(data)
        g0 = np.mean(gs[-int(len(data) * baseline_samples):])

        ret = g0 + np.cumsum(fs - gs) / np.arange(1, len(fs)+1)
        return ret

    return _ret

# TODO: make this into a standard estimator as above.
def gaussian_process_estimator(X, y, X_old=None, y_old=None, kernel=None):
    r"""
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
