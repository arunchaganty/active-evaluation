"""
Active evaluation methods.
"""
import pdb

import numpy as np
#from sklearn.gaussian_process import GaussianProcessRegressor, kernels
#import tensorflow as tf
#import gpflow
from tqdm import tqdm, trange
from .util import StatCounter

def simple(_, __, **___):
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

    return _ret, data

#def gaussian_process(_, wv_dim=50, **__):
#    gp = GaussianProcessRegressor(kernel=kernels.DotProduct())
#
#    def _ret(data, seed=None):
#        rng = np.random.RandomState(seed)
#        rng.shuffle(data)
#
#        X = np.array([datum['x_'] for datum in data])
#        Y = np.array([[datum['y'],] for datum in data])
#
#        ret = np.zeros(len(data))
#        i_ = 0
#        for i in tqdm([100, 1000, 2000,], desc="estimating"):
#            gp.fit(X[:i+1,:], Y[:i+1])
#            mean, std = gp.predict(X, return_std=True)
#            std2 = 1/(1 + std**2)
#            ret[i_:i] = (mean * std2/std2.sum()).sum()
#            i_ = i
#        return np.array(ret)
#    return _ret

def tf_gc():
    graph = tf.get_default_graph()
    for key in graph.get_all_collection_keys():
        graph.clear_collection(key)

def gaussian_process(_, wv_dim=50, **__):
    kernel = gpflow.kernels.RBF(wv_dim, lengthscales=0.1)
    #mf = gpflow.mean_functions.Linear()

    def _ret(data, seed=None):
        rng = np.random.RandomState(seed)
        rng.shuffle(data)

        X = np.array([datum['x_'] for datum in data])
        Y = np.array([[datum['y'],] for datum in data])

        ret = np.zeros(len(data))
        i_ = 0
        for i in tqdm([100, 1000, 2000,]):
            m = gpflow.models.GPR(X[:i+1, :], Y[:i+1], kern=kernel)#, mean_function=mf)
            mean, std2 = m.predict_y(X)
            std2 = 1/(1 + std2)
            std2 /= std2.sum() # normalize
            est = (mean * std2).sum()
            ret[i_:i] = est
            i_ = i
        return np.array(ret)
    return _ret

def test_gaussian_process_estimator():
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    N = 1000
    X = np.random.rand(N,1)
    Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*0.1# + 3
    data = np.hstack((X,Y))
    data_ = np.array(data)
    data_.sort(0)

    k = gpflow.kernels.Matern52(1, lengthscales=0.1)
    def update_figure(num, line, lower, upper):
#        pdb.set_trace()
        m = gpflow.models.SGPR(data[:num+1,0:1], data[:num+1, 1:], kern=k, feat=data[:num+1,0:1])
        m.kern_variance = 0.01
        m.likelihood_variance = 0.01
        
        mean, var = m.predict_y(data_[:,0:1])
#        pdb.set_trace()
        line.set_data(data_[:,0],  mean)
        lower.set_data(data_[:,0], mean - 2*np.sqrt(var))
        upper.set_data(data_[:,0], mean + 2*np.sqrt(var))
        return line, lower, upper,

    # Set up formatting for the movie files
    Writer = animation.writers['imagemagick']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure()
    ax = fig.gca()

    line, = ax.plot([], [], 'b-')
    lower, = ax.plot([], [], 'b:')
    upper, = ax.plot([], [], 'b:')
    points = ax.scatter(X,Y)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    line_ani = animation.FuncAnimation(fig, update_figure, trange(100), fargs=(line, lower, upper),
                                       interval=200, blit=True)
    line_ani.save('gp.gif', writer=writer)

## TODO: make this into a standard estimator as above.
#def gaussian_process_estimator(X, y, X_old=None, y_old=None, kernel=None):
#    r"""
#    @data is a set of [x,y] pairs.
#    @returns a list with the current estimate of \E[y].
#    """
#    gp = GaussianProcessRegressor(kernel=kernel)
#    n = len(y)
#
#    ret = []
#    for i in trange(n):
#        if X_old is not None and y_old is not None:
#            gp.fit(np.vstack([X_old,X[:i+1]]), np.vstack([y_old,y[:i+1]]))
#        else:
#            gp.fit(X[:i+1], y[:i+1])
#
#        y_, std_ = gp.predict(X, return_std=True)
#        mu_ = StatCounter(zip(y_, 1./(1+std_)**2)).mean
#        mu_ = np.mean(y_)
#        ret.append(mu_)
#    return ret, gp
