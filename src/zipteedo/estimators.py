"""
Active evaluation methods.
"""
import pdb

import numpy as np
from tqdm import tqdm, trange
from .util import StatCounter

def simple(_, **___):
    def _ret(data, seed=None):
        rng = np.random.RandomState(seed)
        rng.shuffle(data)

        fs = np.array([[datum['y'], 1.] for datum in data])
        ret = np.cumsum(fs) / np.arange(1, len(fs)+1)
        return ret
    return _ret

def model_baseline(model, baseline_samples=1.0, **_):
    def _ret(data, seed=None):
        rng = np.random.RandomState(seed)
        rng.shuffle(data)

        fs = np.array([datum['y'] for datum in data])
        gs = model(data)
        assert gs.shape == (len(data), 2), "Unexpected model output"
        gs = gs.T[0]
        g0 = np.mean(gs[-int(len(data) * baseline_samples):])

        ret = g0 + np.cumsum(fs - gs) / np.arange(1, len(fs)+1)
        return ret

    return _ret

def model_optimal(model, baseline_samples=1.0, estimate_scale=True, **_):
    def _ret(data, seed=None):
        rng = np.random.RandomState(seed)
        rng.shuffle(data)

        N = len(data)

        fs = np.array([datum['y'] for datum in data])
        gs = model(data)
        assert gs.shape == (len(data), 2), "Unexpected model output"
        gs = gs.T[0]
        g0 = np.mean(gs[-int(len(data) * baseline_samples):])

        # scale by rho*sigma_f/sigma_g
        z = np.arange(1, N+1)
        if estimate_scale:
            mean_f = np.cumsum(fs) / z
            mean_g = np.cumsum(gs) / z
            mean_fg = np.cumsum(fs * gs) / z
            var_g = np.cumsum(gs**2)/z - mean_g**2
            var_g[var_g < 1e-5] = 1 # exception for this one case.
            # scale factor = rho sigma_f/sigma_g
            alpha = (mean_fg - mean_f * mean_g) / var_g
            fs_t = np.array([np.mean((fs - alpha[t]*gs)[:t+1]) for t in range(N)])
        else:
            mean_f = np.mean(fs)
            mean_g = np.mean(gs)
            mean_fg = np.mean(fs * gs)
            var_g = np.std(gs)
            alpha = (mean_fg - mean_f * mean_g) / var_g
            fs_t = fs - alpha

        ret = alpha * g0 + fs_t
        return ret

    return _ret

def cumlogsumexp(a):
    return np.logaddexp.accumulate(a)

def sample_wo_replacement(ws):
    ws = ws.tolist()
    xs = list(range(len(ws)))
    ret = []

    # make a copy
    while len(ws) > 1:
        i = np.random.choice(len(xs), p=ws)
        ret.append(xs.pop(i))
        ws.pop(i)
        z = sum(ws)
        ws = [w/z for w in ws]
    ret += xs

    return np.array(ret)

def model_importance(model, baseline_samples=1.0, use_confidence=True, **_):
    def _ret(data, seed=None):
        rng = np.random.RandomState(seed)
        rng.shuffle(data)

        fs = np.array([datum['y'] for datum in data])
        gs = model(data)
        assert gs.shape == (len(data), 2), "Unexpected model output"
        if use_confidence and gs.T[1].sum() > 0:
            ws = (gs.T[1] + 1e-7) / (gs.T[1] + 1e-7).sum() # add a very very small weight
        else:
            ws = None
        # TODO: have an importance aware estimator.
        g0 = np.mean(gs.T[0][-int(len(data) * baseline_samples):])
        fs_gs = fs - gs.T[0]

        # sample according to importance weights.
        # Get the sample trajectory
        xs = rng.choice(len(data), len(data), p=ws)
        ws, fs_gs = ws[xs] if ws is not None else ws, fs_gs[xs]

        qs = ws if ws is not None else 1./len(data)
        ps = 1./len(data)
        fs_gs_ = ps/qs * fs_gs

        ret = g0 + np.cumsum(fs_gs_) / np.arange(1, len(fs)+1)
        return ret

    return _ret


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
# 
# def tf_gc():
#     graph = tf.get_default_graph()
#     for key in graph.get_all_collection_keys():
#         graph.clear_collection(key)
# 
# def gaussian_process(_, wv_dim=50, **__):
#     kernel = gpflow.kernels.RBF(wv_dim, lengthscales=0.1)
#     #mf = gpflow.mean_functions.Linear()
# 
#     def _ret(data, seed=None):
#         rng = np.random.RandomState(seed)
#         rng.shuffle(data)
# 
#         X = np.array([datum['x_'] for datum in data])
#         Y = np.array([[datum['y'],] for datum in data])
# 
#         ret = np.zeros(len(data))
#         i_ = 0
#         for i in tqdm([100, 1000, 2000,]):
#             m = gpflow.models.GPR(X[:i+1, :], Y[:i+1], kern=kernel)#, mean_function=mf)
#             mean, std2 = m.predict_y(X)
#             std2 = 1/(1 + std2)
#             std2 /= std2.sum() # normalize
#             est = (mean * std2).sum()
#             ret[i_:i] = est
#             i_ = i
#         return np.array(ret)
#     return _ret
# 
# def test_gaussian_process_estimator():
#     import matplotlib.pyplot as plt
#     import matplotlib.animation as animation
# 
#     N = 1000
#     X = np.random.rand(N,1)
#     Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*0.1# + 3
#     data = np.hstack((X,Y))
#     data_ = np.array(data)
#     data_.sort(0)
# 
#     k = gpflow.kernels.Matern52(1, lengthscales=0.1)
#     def update_figure(num, line, lower, upper):
# #        pdb.set_trace()
#         m = gpflow.models.SGPR(data[:num+1,0:1], data[:num+1, 1:], kern=k, feat=data[:num+1,0:1])
#         m.kern_variance = 0.01
#         m.likelihood_variance = 0.01
#         
#         mean, var = m.predict_y(data_[:,0:1])
# #        pdb.set_trace()
#         line.set_data(data_[:,0],  mean)
#         lower.set_data(data_[:,0], mean - 2*np.sqrt(var))
#         upper.set_data(data_[:,0], mean + 2*np.sqrt(var))
#         return line, lower, upper,
# 
#     # Set up formatting for the movie files
#     Writer = animation.writers['imagemagick']
#     writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# 
#     fig = plt.figure()
#     ax = fig.gca()
# 
#     line, = ax.plot([], [], 'b-')
#     lower, = ax.plot([], [], 'b:')
#     upper, = ax.plot([], [], 'b:')
#     points = ax.scatter(X,Y)
#     ax.set_xlim(X.min(), X.max())
#     ax.set_ylim(Y.min(), Y.max())
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     line_ani = animation.FuncAnimation(fig, update_figure, trange(100), fargs=(line, lower, upper),
#                                        interval=200, blit=True)
#     line_ani.save('gp.gif', writer=writer)
# 
# ## TODO: make this into a standard estimator as above.
# #def gaussian_process_estimator(X, y, X_old=None, y_old=None, kernel=None):
# #    r"""
# #    @data is a set of [x,y] pairs.
# #    @returns a list with the current estimate of \E[y].
# #    """
# #    gp = GaussianProcessRegressor(kernel=kernel)
# #    n = len(y)
# #
# #    ret = []
# #    for i in trange(n):
# #        if X_old is not None and y_old is not None:
# #            gp.fit(np.vstack([X_old,X[:i+1]]), np.vstack([y_old,y[:i+1]]))
# #        else:
# #            gp.fit(X[:i+1], y[:i+1])
# #
# #        y_, std_ = gp.predict(X, return_std=True)
# #        mu_ = StatCounter(zip(y_, 1./(1+std_)**2)).mean
# #        mu_ = np.mean(y_)
# #        ret.append(mu_)
# #    return ret, gp
