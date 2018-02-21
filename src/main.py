#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main routine for zipteedo.
"""
import pdb

import os
import sys
import json

import logging
import logging.config
from collections import defaultdict

import numpy as np
from tqdm import tqdm, trange

from zipteedo import estimators
from zipteedo.util import GzipFileType, load_jsonl, dictstr, first, save_jsonl

logger = logging.getLogger(__name__)

def get_metric_ss(data):
    ret = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for datum in tqdm(data, desc="metric ss"):
        system = datum['system']
        for prompt, vs in datum["prompts"].items():
            for metric, v in vs.items():
                ret[metric][prompt][system].append(v)

    for metric in ret:
        for prompt in ret[metric]:
            for system in ret[metric][prompt]:
                ret[metric][prompt][system] = np.mean(ret[metric][prompt][system]), np.var(ret[metric][prompt][system])

    return ret

def get_estimator(args):
    if args.estimator:
        estimator_factory = getattr(estimators, args.estimator)
        return estimator_factory(**args.estimator_args)

def bootstrap_trajectory(fs, gs, hs, anns, estimator, epochs=1000):
    N = len(fs)
    ret = np.zeros((epochs, N))

    # Create a bootstrap sample of fs, gs, and hs
    for i in trange(epochs, desc="bootstrapping"):
        # Get a set of random indices with replacement
        ixs, jxs = np.random.randint(0, N, (N,)), np.random.randint(0, N, (N,))
        fs_, gs_ = fs[ixs], gs[ixs]
        hs_ = np.array([hs[i][j % len(hs[i])] for i,j in zip(ixs,jxs)])
        anns_ = np.array([anns[i][j % len(hs[i])] for i,j in zip(ixs,jxs)])

        ret[i, :] = estimator(fs_, gs_, hs_, anns_)
    return ret

def apply_transforms(args, data):
    # Restrict data to this given prompt, metric and system
    prompt = args.data_prompt
    metric = args.data_metric
    system = args.data_system

    fs, gs, hs, anns = [], [], [], []

    for i, datum in enumerate(data):
        if datum["system"] != system: continue
        vs = datum["prompts"][prompt]
        f, g, h, ann = vs["gold"], vs[metric], vs["human"], datum["annotators"]
        if args.transform_gold_labels:
            h = [f]
            ann = [i]

        fs.append(f)
        gs.append(g)
        hs.append(h)
        anns.append(ann)
    assert fs and gs and hs and anns, "Either {}, {} or {} does not exist".format(prompt, metric, system)

    return np.array(fs), np.array(gs), hs, anns

def do_simulate(args):
    args.estimator_args = dict(args.estimator_args or [])
    data = load_jsonl(args.input)
    data_means = load_jsonl(args.input_means)
    metric_ss = get_metric_ss(data_means)

    # project data.
    fs, gs, hs, anns = apply_transforms(args, data)
    if args.data_metric == "gold":
        args.estimator_args["_g0"], args.estimator_args["_var_g"] = np.mean(fs), np.var(fs)
    else:
        args.estimator_args["_g0"], args.estimator_args["_var_g"] = metric_ss[args.data_metric][args.data_prompt][args.data_system]

    # model.
    truth = np.mean(fs)
    estimator = get_estimator(args)

    # get trajectory
    trajectory = bootstrap_trajectory(fs, gs, hs, anns, estimator, args.num_epochs)
    summary = np.stack([np.mean(trajectory, 0), truth + np.percentile(truth - trajectory, 10, 0), truth + np.percentile(truth - trajectory, 90, 0)])

    # Save output
    ret = {
        "prompt": args.data_prompt,
        "metric": args.data_metric,
        "system": args.data_system,
        "use_gold": args.transform_gold_labels,
        "estimator": args.estimator,
        "estimator_args": args.estimator_args,
        "truth": truth,
        "summary": summary.T.tolist(),
        "trajectory": trajectory.tolist() if args.output_trajectory  else [],
        }

    json.dump(ret, args.output)

def report_trajectory(args, truth, summary):
    return {
        "prompt": args.data_prompt,
        "metric": args.data_metric,
        "system": args.data_system,
        "use_gold": args.transform_gold_labels,
        "estimator": args.estimator,
        "estimator_args": args.estimator_args,
        "truth": truth,
        "summary": summary.tolist(),
        }

def do_build_table(args):
    args.estimator_args = dict(args.estimator_args or [])
    data = load_jsonl(args.input)
    data_means = load_jsonl(args.input_means)
    metric_ss = get_metric_ss(data_means)

    prompts = list(first(data)["prompts"])
    metrics = list(first(first(data)["prompts"].values()))
    metrics.remove('human')
    systems = sorted({datum["system"] for datum in data})
    systems.remove('reference')

    trajectories = []
    settings = [(metric, prompt, system) for metric in metrics for prompt in prompts for system in systems]
    for metric, prompt, system in tqdm(settings, desc="settings"):
        args.data_prompt = prompt
        args.data_metric = metric
        args.data_system = system


        # project data.
        fs, gs, hs = apply_transforms(args, data)

        if metric in metric_ss:
            args.estimator_args["_g0"], args.estimator_args["_var_g"] = metric_ss[metric][prompt][system]
        else:
            args.estimator_args["_g0"], args.estimator_args["_var_g"] = np.mean(gs), np.var(gs)

        # model.
        truth = np.mean(fs)

        args.estimator = "simple"
        trajectory = bootstrap_trajectory(fs, gs, hs, get_estimator(args), args.num_epochs)
        simple = np.stack([np.mean(trajectory, 0), truth + np.percentile(truth - trajectory, 10, 0), truth + np.percentile(truth - trajectory, 90, 0)]).T
        trajectories.append(report_trajectory(args, truth, simple))

        args.estimator = "model_variate"
        trajectory = bootstrap_trajectory(fs, gs, hs, get_estimator(args), args.num_epochs)
        mv = np.stack([np.mean(trajectory, 0), truth + np.percentile(truth - trajectory, 10, 0), truth + np.percentile(truth - trajectory, 90, 0)]).T

        trajectories.append(report_trajectory(args, truth, mv))

    save_jsonl(args.output, trajectories)

if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')

    import argparse
    parser = argparse.ArgumentParser(description='Zipteedo: fast, economical human evaluation.')
    parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for experiments.")
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('simulate', help='Simulates an evaluation model on some data')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="Path to an input dataset.")
    command_parser.add_argument('-im', '--input-means', type=GzipFileType('rt'), default=sys.stdin, help="Path to an input dataset.")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="Path to output the evaluation trajectory.")
    command_parser.add_argument('-oT', '--output-trajectory', action='store_true', default=False, help="Save the trajectories too.")
    command_parser.add_argument('-Tg', '--transform-gold-labels', action='store_true', default=False, help="Transform: no annotator noise.")
    command_parser.add_argument('-E', '--estimator', type=str, default=None, help="Which estimator to use")
    command_parser.add_argument('-Xe', '--estimator-args', type=dictstr, nargs="+", default=None, help="Extra arguments for the estimator")
    command_parser.add_argument('-nE', '--num-epochs', type=int, default=100, help="Number of epochs")

    command_parser.add_argument('-Dp', '--data-prompt', type=str, help="Which prompt to compute trajectory on")
    command_parser.add_argument('-Dm', '--data-metric', type=str, help="Which automatic-metric to use")
    command_parser.add_argument('-Ds', '--data-system', type=str, help="Which system to use")
    command_parser.set_defaults(func=do_simulate)

    command_parser = subparsers.add_parser('build-table', help='Simulates an evaluation model on some data')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="Path to an input dataset.")
    command_parser.add_argument('-im', '--input-means', type=GzipFileType('rt'),  help="Path to an input dataset.")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="Path to output the evaluation trajectory.")
    command_parser.add_argument('-Tg', '--transform-gold-labels', action='store_true', default=False, help="Transform: no annotator noise.")
    command_parser.add_argument('-E', '--estimator', type=str, default=None, help="Which estimator to use")
    command_parser.add_argument('-Xe', '--estimator-args', type=dictstr, nargs="+", default=None, help="Extra arguments for the estimator")
    command_parser.add_argument('-nE', '--num-epochs', type=int, default=100, help="Number of epochs")

    command_parser.add_argument('-Dp', '--data-prompt', type=str, help="Which prompt to compute trajectory on")
    command_parser.add_argument('-Dm', '--data-metric', type=str, help="Which automatic-metric to use")
    command_parser.add_argument('-Ds', '--data-system', type=str, help="Which system to use")
    command_parser.set_defaults(func=do_build_table)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        np.random.seed(ARGS.seed)
        ARGS.func(ARGS)
