#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main routine for zipteedo.
"""
import pdb

import sys

import json

import numpy as np
from tqdm import tqdm, trange

from zipteedo.util import GzipFileType, load_jsonl, dictstr
from zipteedo import estimators, models
from zipteedo import wv

def get_model(args, data):
    if args.model:
        model_factory = getattr(models, args.model)
        return model_factory(data, **args.model_args)

def get_estimator(args, model):
    if args.estimator:
        estimator_factory = getattr(estimators, args.estimator)
        return estimator_factory(model, **args.estimator_args)

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
    # embed the data.
    if args.transform_embed:
        wvecs = wv.load_word_vector_mapping(args.embeddings_vocab, args.embeddings_vectors)
        UNK = np.random.randn(50)
        for datum in data:
            datum['x_'] = sum(wvecs.get(word.lower(), UNK) for word in datum['x'].split(' '))

    return data

def do_simulate(args):
    args.model_args = dict(args.model_args or [])
    args.estimator_args = dict(args.estimator_args or [])

    data = load_jsonl(args.input)

    # Apply dataset transforms:
    data = apply_transforms(args, data)

    # model.
    truth = np.mean([datum['y*'] for datum in data])
    model = get_model(args, data)
    estimator = get_estimator(args, model)

    # get trajectory
    trajectory = np.array(bootstrap_trajectory(data, estimator, args.num_realizations, args.num_samples))
    summary = np.stack([np.mean(trajectory, 0), np.percentile(trajectory, 10, 0), np.percentile(trajectory, 90, 0)])

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
        "summary": summary.T.tolist(),
        "trajectory": trajectory.tolist() if args.output_trajectory  else [],
        }

    json.dump(ret, args.output)

def make_label(obj):
    ret = ""
    if obj["model"] == "ConstantModel":
        ret += "Constant (${:.2f}$)".format(obj["model_args"].get("cnst", 0.))
    elif obj["model"] == "OracleModel":
        ret += r"Oracle ($\rho={:.2f}$)".format(obj["model_args"].get("rho", 1.))

    # TODO: add information about estimator
    if obj["transforms"]["gold_labels"]:
        ret += " (gold)"
    return ret

def apply_data_transform(args, obj, data):
    if args.transform_mean:
        data -= obj["truth"]
    return data

def do_plot(args):
    import matplotlib.pyplot as plt
    from matplotlib.cm import viridis

    colors = viridis.colors[::256//(len(args.trajectories)-1)-1]

    for i, trajectory in enumerate(args.trajectories):
        trajectory = json.load(trajectory)
        summary = np.array(trajectory["summary"])
        apply_data_transform(args, trajectory, summary)

        xs = np.arange(1, len(summary)+1)
        plt.plot(xs, summary.T[0], color=colors[i], label=make_label(trajectory), linewidth=0.5)
        #plt.fill_between(xs, summary.T[1], summary.T[2], color=colors[i], alpha=0.3)
        plt.plot(xs, summary.T[1], color=colors[i], alpha=0.3, linestyle=':', linewidth=0.5)
        plt.plot(xs, summary.T[2], color=colors[i], alpha=0.3, linestyle=':', linewidth=0.5)

    plt.rc("text", usetex=True)
    plt.rc("figure", figsize=(10,10))
    plt.xlabel("Samples")
    plt.ylabel("Estimation error")
    plt.xlim(1, args.xlim)
    #plt.ylim(-0.2, 0.2)
    plt.legend()
    plt.savefig(args.output, dpi=400)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Zipteedo: fast, economical human evaluation.')
    parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for experiments.")
    parser.add_argument('-evo', '--embeddings-vocab',   type=argparse.FileType('r'), default="data/embeddings.vocab",   help="Path to word embedding vocabulary")
    parser.add_argument('-eve', '--embeddings-vectors', type=argparse.FileType('r'), default="data/embeddings.vectors", help="Path to word vectors")
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('simulate', help='Simulates an evaluation model on some data')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="Path to an input dataset.")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="Path to output the evaluation trajectory.")
    command_parser.add_argument('-oT', '--output-trajectory', action='store_true', default=False, help="Save the trajectories too.")
    command_parser.add_argument('-Tg', '--transform-gold-labels', action='store_true', default=False, help="Transform: no annotator noise.")
    command_parser.add_argument('-Te', '--transform-embed', action='store_true', default=False, help="Transform: add sentence embeddings.")
    command_parser.add_argument('-M', '--model', type=str, default=None, help="Which model to use")
    command_parser.add_argument('-E', '--estimator', type=str, default=None, help="Which estimator to use")
    command_parser.add_argument('-Xm', '--model-args', type=dictstr, nargs="+", default=None, help="Extra arguments for the model")
    command_parser.add_argument('-Xe', '--estimator-args', type=dictstr, nargs="+", default=None, help="Extra arguments for the estimator")
    command_parser.add_argument('-nR', '--num-realizations', type=int, default=10, help="Number of realizations of turker data")
    command_parser.add_argument('-nS', '--num-samples', type=int, default=10, help="Number of realizations of sampling algorithm")
    command_parser.set_defaults(func=do_simulate)

    command_parser = subparsers.add_parser('plot', help='Plots a set of evaluation trajectories')
    command_parser.add_argument('-o', '--output', type=str, default='trajectory.pdf', help="Path to output the plot of evaluation trajectories.")
    command_parser.add_argument('--xlim', type=int, default=2000, help="Extent to which to plot")
    command_parser.add_argument('-Tm', '--transform-mean', type=bool, default=True, help="Tranform data to mean")
    command_parser.add_argument('trajectories', type=GzipFileType('rt'), nargs='+', help="List of trajectory files to plot.")
    command_parser.set_defaults(func=do_plot)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        np.random.seed(ARGS.seed)
        ARGS.func(ARGS)
