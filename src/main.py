#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main routine for zipteedo.
"""
import pdb

import sys

import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from zipteedo.util import GzipFileType, load_jsonl, dictstr
from zipteedo import estimators, models

def get_model(args, data):
    model_factory = getattr(models, args.model)
    return model_factory(data, **dict(args.model_args or []))

def get_estimator(args, model):
    estimator_factory = getattr(estimators, args.estimator)
    return estimator_factory(model, **dict(args.estimator_args or []))

def bootstrap_trajectory(data, estimator, realization_epochs=10, sampling_epochs=100):
    ret = np.zeros((realization_epochs * sampling_epochs, len(data)))

    # sufficiently large prime.
    idxs = np.random.randint(0, 137, size=(realization_epochs, len(data)))
    seeds = np.random.randint(0,2**31, size=(sampling_epochs,))
    for i in trange(realization_epochs, desc="Bootstrapping realizations of the data"):
        # Construct a realization of the data.
        for j, datum in enumerate(data):
            datum['y'] = datum['ys'][idxs[i,j] % len(datum['ys'])]

        for j in trange(sampling_epochs, desc="Bootstraping samples of the data"):
            ret[sampling_epochs*i + j, :] = estimator(data, seeds[i])
    return ret

def apply_transforms(args, data):
    if args.transform_gold_labels:
        for datum in data:
            datum['ys'] = [datum['y*']]
    return data

def do_simulate(args):
    data = load_jsonl(args.input)

    # Apply dataset transforms:
    data = apply_transforms(args, data)

    # model.
    truth = np.mean([datum['y*'] for datum in data])
    model = get_model(args, data)
    estimator = get_estimator(args, model)

    # get trajectory
    trajectory = np.array(bootstrap_trajectory(data, estimator, args.num_realizations, args.num_samples))
    summary = list(zip(np.mean(trajectory, 0), np.percentile(trajectory, 10, 0), np.percentile(trajectory, 90, 0)))

    # Save output
    ret = {
        "transforms": {
            "gold_labels": args.transform_gold_labels,
            },
        "model": args.model,
        "model_args": args.model_args,
        "estimator": args.estimator,
        "estimator_args": args.estimator_args,
        "truth": truth,
        "summary": summary,
        "trajectory": trajectory.tolist() if args.output_trajectory  else [],
        }

    json.dump(ret, args.output)


def plot_agg(xs, mus, lower, higher, *args, color='b', **kwargs):
    plt.plot(xs, mus, color=color, *args, **kwargs)
    plt.plot(xs, lower, linestyle='--', color=color, alpha=0.5)
    plt.plot(xs, higher, linestyle='--', color=color, alpha=0.5)

def do_plot(args):
    pass
    # simulate systems.

    ## Expected value.
    #writer = csv.writer(args.output_data)
    #writer.writerow(["system", "i", "mean", "lower", "upper"])

    #mu_0 = np.mean(ys)
    #xs = np.arange(len(data))
    #writer.writerow(['mean', 0, mu_0, mu_0, mu_0])
    #plt.plot(xs, mu_0 * np.ones(len(data)), '--', label="Final value")

    #mu, mu_l, mu_h = aggregate(data, random_sampling)
    #writer.writerows((["random", i, mu[i], mu_l[i], mu_h[i]] for i in range(len(mu))))
    #plot_agg(xs, mu, mu_l, mu_h, label="Random sampling")

    #for acc, color in zip([0.87, 0.5, 1.0], 'rgcym'):
    #    yhs = np.random.binomial(1, acc, len(data))
    #    ghs = [yh * y + (1-yh) * (1-y) for y, yh in zip(ys, yhs)]
    #    print(np.mean(ghs))
    #    mu, mu_l, mu_h = aggregate(data, lambda data: random_sampling_model(data, ghs))
    #    writer.writerows((["model%.2f"%acc, i, mu[i], mu_l[i], mu_h[i]] for i in range(len(mu))))
    #    plot_agg(xs, mu, mu_l, mu_h, color=color, label="Model-based random sampling (acc=%.2f)"%acc)

    ##plt.xlim((0, len(data)))
    #plt.xlim((0, 2000))
    #plt.ylim((mu_0-0.1, mu_0+0.1))
    #plt.legend()
    #plt.rc('figure', figsize=(10, 10))
    #plt.savefig(args.output_plot, dpi=300)
    ##plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Zipteedo: fast, economical human evaluation.')
    parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for experiments.")
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('simulate', help='Simulates an evaluation model on some data')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="Path to an input dataset.")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="Path to output the evaluation trajectory.")
    command_parser.add_argument('-oT', '--output-trajectory', action='store_true', default=False, help="Save the trajectories too.")
    command_parser.add_argument('-Tg', '--transform-gold-labels', action='store_true', default=False, help="Transform: no annotator noise.")
    command_parser.add_argument('-M', '--model', type=str, default=None, help="Which model to use")
    command_parser.add_argument('-E', '--estimator', type=str, default=None, help="Which estimator to use")
    command_parser.add_argument('-Xm', '--model-args', type=dictstr, nargs="+", default=None, help="Extra arguments for the model")
    command_parser.add_argument('-Xe', '--estimator-args', type=dictstr, nargs="+", default=None, help="Extra arguments for the estimator")
    command_parser.add_argument('-nR', '--num-realizations', type=int, default=10, help="Number of realizations of turker data")
    command_parser.add_argument('-nS', '--num-samples', type=int, default=10, help="Number of realizations of sampling algorithm")
    command_parser.set_defaults(func=do_simulate)

    command_parser = subparsers.add_parser('plot', help='Plots a set of evaluation trajectories')
    command_parser.add_argument('-o', '--output', type=str, default='trajectory.pdf', help="Path to output the plot of evaluation trajectories.")
    command_parser.add_argument('trajectories', type=GzipFileType('rt'), nargs='+', help="List of trajectory files to plot.")
    command_parser.set_defaults(func=do_plot)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        np.random.seed(ARGS.seed)
        ARGS.func(ARGS)
