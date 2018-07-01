"""
Active evaluation methods.
"""
import pdb

import scipy as sc
import numpy as np
from tqdm import trange

def simple(**_):
    def _ret(_, __, hs, anns):
        N = len(hs)
        z = np.arange(1, N+1)

        ret = np.cumsum(hs) / z
        return ret
    return _ret

def model_variate(_g0=None, _var_g=None, estimate_scale=True, **_):
    def _ret(_, gs, hs, anns):
        N = len(hs)
        z = np.arange(1, N+1)

        # TODO: get mean_g, var_g from elsewhere
        g0 = _g0 if _g0 is not None else np.mean(gs)
        var_g = _var_g if _var_g is not None else np.var(gs)

        # scale by rho*sigma_f/sigma_g
        if estimate_scale:
            mean_h = np.cumsum(hs) / z
            mean_g = np.cumsum(gs) / z
            mean_hg = np.cumsum(hs * gs) / z
            #var_g[var_g < 1e-10] = 1 # exception for this one case.
            # scale factor = cov(h,g) / var_g

            alpha = (mean_hg - mean_h * mean_g) / var_g
            #pdb.set_trace()
            hs_t = np.array([np.mean((hs - alpha[t]*gs)[:t+1]) for t in range(N)])
        else:
            mean_h = np.mean(hs)
            mean_g = np.mean(gs)
            mean_hg = np.mean(hs * gs)
            var_g = np.std(gs)
            alpha = (mean_hg - mean_h * mean_g) / var_g
            hs_t = np.cumsum(hs - alpha*gs) / z

        ret = alpha * g0 + hs_t
        return ret
    return _ret

def _estimate_sigmas(data):
    annotators = sorted({a for datum in data for a in datum['as']})
    tasks  = list(range(len(data)))

    ix_a = np.array([annotators.index(a) for datum in data for a in datum['as']])
    ix_t = np.array([i for i, datum in enumerate(data) for _ in datum['ys']])

    y = np.array([y_ for datum in data for y_ in datum['ys']])
    fs = np.array([np.mean(y[ix_t == i]) for i in range(len(tasks))])
    as_ = np.array([np.mean(y[ix_a == i] - fs[ix_t[ix_a == i]]) if i in ix_a else 0 for i in range(len(annotators))])
    rs = y - fs[ix_t] - as_[ix_a]

    sigma2_f = np.var(fs, ddof=1)
    sigma2_a = np.var(as_, ddof=1)
    sigma2_r = np.var(rs, ddof=1)
    pdb.set_trace()

    return sigma2_f, sigma2_a, sigma2_r

def encode_data_linear(data, use_gold=True):
    annotators = sorted({datum['as'][i] for datum in data for i in datum['yi']})
    tasks  = list(range(len(data)))

    ix_a = np.array([annotators.index(datum['as'][i]) for datum in data for i in datum['yi']])
    ix_t = np.array([i for i, datum in enumerate(data) for _ in datum['yi']])
    ix = np.array([i for i, _ in enumerate(ix_a)])

    y = np.array([datum['ys'][i] for datum in data for i in datum['yi']])
    if use_gold:
        sigma2_f, sigma2_a, sigma2_r = _estimate_sigmas(data)
    else:
        fs = np.array([np.mean(y[ix_t == i]) for i in range(len(tasks))])
        as_ = np.array([np.mean(y[ix_a == i] - fs[ix_t[ix_a == i]]) if i in ix_a else 0 for i in range(len(annotators))])
        rs = y - fs[ix_t] - as_[ix_a]
        sigma2_f, sigma2_a, sigma2_r = np.var(fs, ddof=1), np.var(as_, ddof=1), np.var(rs, ddof=1)

    n = len(ix)
    A = np.zeros((n, n))
    A[ix, ix] = sigma2_r
    for i in range(len(annotators)):
        A[np.ix_(ix_a == i, ix_a == i)] += sigma2_a
    for i in range(len(tasks)):
        A[np.ix_(ix_t == i, ix_t == i)] += sigma2_f

    return A, y

def linear(_, use_gold=True, **__):
    def _ret(data, seed=None):
        A, y = encode_data_linear(data, use_gold=use_gold)

        rng = np.random.RandomState(seed)
        #ixs = rng.randint(len(_y), size=(len(_y),))
        ixs = rng.permutation(len(y))
        A, y = A[np.ix_(ixs, ixs)], y[ixs]
        n, _ = A.shape

        ret = []
        L = sc.linalg.cholesky(A)
        for i in trange(1, n+1, desc="estimating"):
            c = sc.linalg.cho_solve((L[:i,:i], True), np.ones(i))
            c /= c.sum()
            print(c)
            pdb.set_trace()
            ret.append(c.dot(y[:i]))
        return np.array(ret)
    return _ret

def cumlogsumexp(a):
    return np.logaddexp.accumulate(a)

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

def analyze_coefficients(data):
    annotators = sorted({a for datum in data for a in datum['as']})
    tasks  = list(range(len(data)))

    ix_a = np.array([annotators.index(a) for datum in data for a in datum['as']])
    ix_t = np.array([i for i, datum in enumerate(data) for _ in datum['ys']])
    ix = np.array([i for i, _ in enumerate(ix_a)])

    y = np.array([y for datum in data for y in datum['ys']])

    fs = np.array([np.mean(y[ix_t == i]) for i in range(len(tasks))])
    as_ = np.array([np.mean(y[ix_a == i] - fs[ix_t[ix_a == i]]) if i in ix_a else 0 for i in range(len(annotators))])
    rs = y - fs[ix_t] - as_[ix_a]
    sigma_f, sigma_a, sigma_r = np.std(fs), np.std(as_), np.std(rs)

    n = len(ix)
    A = np.zeros((n, n))
    A[ix, ix] = sigma_r**2
    for i in range(len(annotators)):
        A[np.ix_(ix_a == i, ix_a == i)] += sigma_a**2
    for i in range(len(tasks)):
        A[np.ix_(ix_t == i, ix_t == i)] += sigma_f**2

    c = sc.linalg.solve(A, np.ones(n), sym_pos=True)

    # convert responses into a matrix.
    Y = np.zeros((len(tasks), len(annotators)))
    Y[ix_t, ix_a] = y

    C = np.zeros((len(tasks), len(annotators)))
    C[ix_t, ix_a] = c

    R = np.zeros((len(tasks), len(annotators)))
    R[ix_t, ix_a] = rs

    M = np.ones((len(tasks), len(annotators)))
    M[ix_t, ix_a] = 0

    return Y, C, R, M, fs, as_
