#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main routine for zipteedo.
"""
import pdb

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from scipy.stats import mode

from zipteedo.util import GzipFileType, load_jsonl

vmap = {
    "contradiction": 0,
    "neutral": 0,
    "entailment": 1,
    }

def random_sampling(data):
    # Moving average.
    idxs = list(range(len(data)))
    np.random.shuffle(idxs)
    idxs_ = np.random.randint(0, 67, size=(len(data),)) # big enough prime
    ret, mu, var = [], 0, 0
    for i, (ix, ix_) in enumerate(zip(idxs, idxs_)):
        lst = data[ix]
        fh  = lst[ix_ % len(lst)]

        mu += (fh - mu)/(i+1)
        var += (fh**2 - var)/(i+1)
        ret.append(mu)
    return ret, var - mu**2

def random_sampling_model(data, ghs):
    # Moving average.
    data = data
    g_m = np.mean(ghs[2000:])

    idxs = list(range(len(data)))
    np.random.shuffle(idxs)
    idxs_ = np.random.randint(0, 67, size=(len(data),)) # big enough prime
    ret, mu = [], 0
    var = 0.
    for i, (ix, ix_) in enumerate(zip(idxs, idxs_)):
        lst = data[ix]
        fh  = lst[ix_ % len(lst)]
        gh = ghs[ix]

        mu += ((fh-gh) - mu)/(i+1)
        var += ((fh-gh)**2 - var)/(i+1)
        ret.append(g_m + mu)
    return ret, var - mu**2

def aggregate(data, fn, iters=500):
    ret, var = [], []
    for _ in trange(iters):
        ret_, var_ = fn(data)
        ret.append(ret_)
        var.append(var_)
    print(np.mean(var))
    ret = np.array(ret)
    return np.mean(ret, 0), np.percentile(ret, 10, 0), np.percentile(ret, 90, 0)

def plot_agg(xs, mus, lower, higher, *args, color='b', **kwargs):
    plt.plot(xs, mus, color=color, *args, **kwargs)
    plt.plot(xs, lower, linestyle='--', color=color, alpha=0.5)
    plt.plot(xs, higher, linestyle='--', color=color, alpha=0.5)

def do_snli(args):
    np.random.seed(args.seed)
    data = load_jsonl(args.input)

    # transform data
    data = [[vmap[l] for l in d['annotator_labels']] for d in data]
    #ys = [mode(lst).mode[0] for lst in data]
    ys = [np.mean(lst) for lst in data]
    data_ = [[y] for y in ys]
    #data, data_ = data_, data
    # Per example variance:
    stds = [np.std(lst) for lst in data]
    pdb.set_trace()
    plt.hist(stds)
    plt.show()
    return

    # Expected value.
    writer = csv.writer(args.output_data)
    writer.writerow(["system", "i", "mean", "lower", "upper"])

    mu_0 = np.mean(ys)
    xs = np.arange(len(data))
    writer.writerow(['mean', 0, mu_0, mu_0, mu_0])
    plt.plot(xs, mu_0 * np.ones(len(data)), '--', label="Final value")

    mu, mu_l, mu_h = aggregate(data, random_sampling)
    writer.writerows((["random", i, mu[i], mu_l[i], mu_h[i]] for i in range(len(mu))))
    plot_agg(xs, mu, mu_l, mu_h, label="Random sampling")

    for acc, color in zip([0.87, 0.5, 1.0], 'rgcym'):
        yhs = np.random.binomial(1, acc, len(data))
        ghs = [yh * y + (1-yh) * (1-y) for y, yh in zip(ys, yhs)]
        print(np.mean(ghs))
        mu, mu_l, mu_h = aggregate(data, lambda data: random_sampling_model(data, ghs))
        writer.writerows((["model%.2f"%acc, i, mu[i], mu_l[i], mu_h[i]] for i in range(len(mu))))
        plot_agg(xs, mu, mu_l, mu_h, color=color, label="Model-based random sampling (acc=%.2f)"%acc)

    #plt.xlim((0, len(data)))
    plt.xlim((0, 2000))
    plt.ylim((mu_0-0.1, mu_0+0.1))
    plt.legend()
    plt.rc('figure', figsize=(10, 10))
    plt.savefig(args.output_plot, dpi=300)
    #plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Zipteedo: fast turking.')
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('snli', help='Evaluate SNLI model')
    command_parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for experiments.")
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="Path to SNLI data.")
    command_parser.add_argument('-op', '--output-plot', type=str, default='plot.png', help="Path to SNLI data.")
    command_parser.add_argument('-od', '--output-data', type=argparse.FileType('w'), default=sys.stdout, help="Path to SNLI data.")
    command_parser.set_defaults(func=do_snli)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
