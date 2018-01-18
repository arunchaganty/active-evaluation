#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot various visualizations
"""
import pdb
import sys
import json
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

from zipteedo.util import GzipFileType, load_jsonl

def get_colors(n_colors=1):
    if n_colors > 1:
        colors = viridis.colors[::256//(n_colors-1)-1]
    else:
        colors = [viridis.colors[0]] # some color is fine.
    return colors

def do_model_correlation(args):
    colors = get_colors(len(args.systems))
    system_names = ["KenLM", "GoogleLM1B", "conv-model",]

    plt.plot([0,1], [0,1], color='k') # Axes
    for i, system in enumerate(args.systems):
        data = np.array([[datum['y*'], datum['y^']] for datum in load_jsonl(system)])
        rho, p = scipy.stats.pearsonr(data.T[0], data.T[1])
        system_name = system_names[i]

        plt.scatter(data.T[0], data.T[1], color=colors[i], alpha=0.3, label=r"{} ($\rho = {:.2f}; p={:.2e}$)".format(system_name, rho, p))

    plt.rc("text", usetex=True)
    plt.rc("figure", figsize=(10,10))
    plt.xlabel("True label")
    plt.ylabel("Guessed label")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=400)


def do_estimation_trajectory(args):
    def make_label(obj):
        ret = ""
        if obj["model"] == "ConstantModel":
            ret += "Constant (${:.2f}$)".format(obj["model_args"].get("cnst", 0.))
        elif obj["model"] == "OracleModel":
            ret += r"Oracle ($\rho={:.2f}$)".format(obj["model_args"].get("rho", 1.))
        else:
            ret += obj["model"]

        # TODO: add information about estimator
        if obj['estimator'] == "model_optimal":
            ret += " w/scaling"
        elif obj['estimator'] == "model_importance":
            ret += " w/importance"
        if obj["transforms"]["gold_labels"]:
            ret += " (gold)"
        return ret

    def apply_data_transform(args, obj, data):
        if args.transform_mean:
            data = data - obj["truth"]
        return data

    colors = get_colors(len(args.systems))
    for i, trajectory in enumerate([json.load(system) for system in args.systems]):
        summary = np.array(trajectory["summary"])
        summary = apply_data_transform(args, trajectory, summary)

        xs = np.arange(1, len(summary)+1)
        plt.plot(xs, summary.T[0], color=colors[i], label=make_label(trajectory), linewidth=0.5)
        #plt.fill_between(xs, summary.T[1], summary.T[2], color=colors[i], alpha=0.3)
        plt.plot(xs, summary.T[1], color=colors[i], linestyle=':', linewidth=0.5)
        plt.plot(xs, summary.T[2], color=colors[i], linestyle=':', linewidth=0.5)

    plt.rc("text", usetex=True)
    plt.rc("figure", figsize=(10,10))
    plt.xlabel("Samples")
    plt.ylabel("Estimation error")
    plt.xlim(1, args.xlim)
    if args.center:
        plt.ylim(-0.1, 0.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=400)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('model-correlation', help='Plot the correlation of a models output with truth')
    command_parser.add_argument('-o', '--output', type=str, default="model_correlation.pdf", help="Where to save plot")
    command_parser.add_argument('systems', nargs='+', type=argparse.FileType('r'), help="Systems to plot")
    command_parser.set_defaults(func=do_model_correlation)

    command_parser = subparsers.add_parser('estimation-trajectory', help='Plots a set of evaluation trajectories')
    command_parser.add_argument('-o', '--output', type=str, default='trajectory.pdf', help="Path to output the plot of evaluation trajectories.")
    command_parser.add_argument('--xlim', type=int, default=2000, help="Extent to which to plot")
    command_parser.add_argument('-Tm', '--transform-mean', type=bool, default=True, help="Tranform data to mean")
    command_parser.add_argument('-Xc', '--center', type=bool, default=True, help="Tranform data to mean")
    command_parser.add_argument('systems', type=GzipFileType('rt'), nargs='+', help="List of trajectory files to plot.")
    command_parser.set_defaults(func=do_estimation_trajectory)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
