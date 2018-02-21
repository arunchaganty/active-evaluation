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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

from zipteedo.util import GzipFileType, load_jsonl
import zipteedo.estimators as zpe

def get_colors(n_colors=1):
    _colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    return _colors[:n_colors]

    #if n_colors > 1:
    #    colors = viridis.colors[::256//(n_colors-1)-1]
    #else:
    #    colors = [viridis.colors[0]] # some color is fine.
    #return colors

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

def do_variance_trajectory(args):
    def make_label(obj):
        ret = ""
        if obj["model"] == "ConstantModel":
            return "Baseline: random sampling"
            ret += "Constant (${:.2f}$)".format(obj["model_args"].get("cnst", 0.))
        elif obj["model"] == "OracleModel":
            return "Using a perfect model"
            ret += r"Oracle ($\rho={:.2f}$)".format(obj["model_args"].get("rho", 1.))
        else:
            ret += obj["model"] or "No model"

        # TODO: add information about estimator
        if obj['estimator'] == "model_optimal":
            ret += " w/scaling"
        elif obj['estimator'] == "model_importance":
            ret += " w/importance"
        elif obj['estimator'] == "linear":
            ret += " w/linear"
        if obj["transforms"]["gold_labels"]:
            ret += " (gold)"

        ret += " $n_a={}$".format(obj["n_annotators"])

        return ret

    colors = get_colors(len(args.systems))
    for i, trajectory in enumerate([json.load(system) for system in args.systems]):
        summary = np.array(trajectory["summary"])
        summary = summary.T[2] - summary.T[1]

        xs = np.arange(1, len(summary)+1)
        plt.plot(xs, summary, color=colors[i], label=make_label(trajectory), linewidth=0.5)

    plt.rc("text", usetex=True)
    plt.rc("figure", figsize=(10,10))
    plt.xlabel("Samples")
    plt.ylabel("Confidence interval")
    plt.xlim(1, min(args.xlim, plt.xlim()[1]))
    #if args.center:
    plt.ylim(0.04, 0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=400)


def do_estimation_trajectory(args):
    def make_label(obj):
        ret = "{}-{}".format(obj["metric"], obj["estimator"])
        return ret

    def apply_data_transform(args, obj, data):
        if args.transform_mean:
            data = data - obj["truth"]
        if args.transform_final:
            data = data - data[-1,0]
        return data

    colors = get_colors(len(args.systems))
    trajectories = [json.load(system) for system in args.systems]
    for i, trajectory in enumerate(trajectories):
        summary = np.array(trajectory["summary"])
        summary = apply_data_transform(args, trajectory, summary)

        xs = np.arange(1, len(summary)+1)
        plt.plot(xs, summary.T[0], color=colors[i], label=make_label(trajectory), linewidth=0.5)
        #plt.fill_between(xs, summary.T[1], summary.T[2], color=colors[i], alpha=0.3)
        plt.plot(xs, summary.T[1], color=colors[i], linestyle=':', linewidth=0.5)
        plt.plot(xs, summary.T[2], color=colors[i], linestyle=':', linewidth=0.5)

    #plt.rc("text", usetex=True)
    plt.rc("figure", figsize=(10,10))
    plt.xlabel("Samples")
    plt.ylabel("Estimation error")
    plt.title("{} {}".format(trajectories[0]['prompt'], trajectories[0]['system']))
    plt.xlim(1, min(args.xlim, plt.xlim()[1]))
    if args.center:
        plt.ylim(-0.1, 0.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=400)

def do_coefficients(args):
    data = load_jsonl(args.input)

    A, y = zpe.encode_data_linear(data)

def do_bias_plot(args):
    data = load_jsonl(args.input)
    xy = np.array([[y, y_] for _, y, y_ in data[:4]])
    XY = np.array(sorted([[y, y_] for _, y, y_ in data], key=lambda l: l[0]))

    xlim = np.array([XY.T[0].min(), XY.T[0].max()])
    coeffs = np.polyfit(xy.T[0], xy.T[1])
    # Plot y == x line
    plt.plot(xlim, xlim * coeffs[0] + coeffs[1], linestyle='--', linewidth=2)
    plt.plot(XY.T[0], XY.T[1])
    plt.scatter(xy.T[0], xy.T[1], 400, marker="*")
    plt.xlabel("Human judgement")
    plt.ylabel("ROUGE-L")
    plt.tight_layout()
    plt.savefig(args.output)

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
    command_parser.add_argument('-Tf', '--transform-final', type=bool, default=False, help="Tranform data to mean")
    command_parser.add_argument('-Xc', '--center', type=bool, default=True, help="Tranform data to mean")
    command_parser.add_argument('systems', type=GzipFileType('rt'), nargs='+', help="List of trajectory files to plot.")
    command_parser.set_defaults(func=do_estimation_trajectory)

    command_parser = subparsers.add_parser('variance-trajectory', help='Plots a set of evaluation trajectories')
    command_parser.add_argument('-o', '--output', type=str, default='trajectory.pdf', help="Path to output the plot of evaluation trajectories.")
    command_parser.add_argument('--xlim', type=int, default=2000, help="Extent to which to plot")
    command_parser.add_argument('systems', type=GzipFileType('rt'), nargs='+', help="List of trajectory files to plot.")
    command_parser.set_defaults(func=do_variance_trajectory)

    command_parser = subparsers.add_parser('coefficients', help='Plots coefficients between responses')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="Path to an input dataset.")
    command_parser.add_argument('-o', '--output', type=str, default='coefficients.pdf', help="Path to output the plot of evaluation trajectories.")
    command_parser.set_defaults(func=do_coefficients)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
