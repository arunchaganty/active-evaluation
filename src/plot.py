#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot various visualizations
"""
import sys
import json
from collections import defaultdict


import numpy as np
import scipy.stats as scstats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mp

from zipteedo.util import GzipFileType, load_jsonl
import zipteedo.estimators as zpe


def get_correlation_table(data):
    prompts = list(first(data)["prompts"])
    systems = sorted({datum["system"] for datum in data})

    ret = defaultdict(dict)

    for prompt in prompts:
        n_ = 0
        sigma_f_ = []
        sigma_a_ = []
        for system in systems:
            n = 0
            sigma_f = []
            sigma_a = []
            for datum in data:
                if datum['system'] != system: continue
                hs = datum['prompts'][prompt]['human']
                if len(hs) > 1:
                    f = np.mean(hs)
                    sigma_f.extend(hs)
                    sigma_a.extend(hs - f)
                    sigma_f_.extend(hs)
                    sigma_a_.extend(hs - f)
                    n += 1
                    n_ += 1

            sigma2_a = np.var(sigma_a, ddof=n)
            sigma2_f = np.var(sigma_f, ddof=1) - sigma2_a
            nu = sigma2_a/sigma2_f
            ret[system][prompt] = {
                "sigma2_a": sigma2_a ,
                "sigma2_f": sigma2_f ,
                "nu": nu ,
                }

        sigma2_a = np.var(sigma_a_, ddof=n_)
        sigma2_f = np.var(sigma_f_, ddof=1) - sigma2_a
        nu = sigma2_a/sigma2_f

        ret["*"][prompt] = {
            "sigma2_a": sigma2_a ,
            "sigma2_f": sigma2_f ,
            "nu": nu ,
            }
    return ret


def make_correlation(data):
    metrics = list(first(first(data)["prompts"].values()))
    prompts = list(first(data)["prompts"])
    systems = sorted({datum["system"] for datum in data})
    metrics.remove("gold")
    metrics.remove("human")
    print(systems)

    sigma_table = get_correlation_table(data)

    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for datum in data:
        system = datum['system']
        for prompt in prompts:
            for metric in metrics:
                y, y_ = datum["prompts"][prompt]["gold"], datum["prompts"][prompt][metric]
                assert y is not None
                assert y_ is not None
                agg[prompt][metric][system].append([y, y_])
                agg[prompt][metric]["*"].append([y, y_])

    # Finally aggregate.
    ret = defaultdict(lambda: defaultdict(dict))
    for prompt in prompts:
        for metric in metrics:
            for system in systems + ["*"]:
                if system == "reference": continue
                xy = np.array(agg[prompt][metric][system])
                v = np.mean(xy.T[0] * xy.T[1] - np.mean(xy.T[0])*np.mean(xy.T[1]))/(np.sqrt(sigma_table[system][prompt]['sigma2_f']) * np.sqrt(np.var(xy.T[1], ddof=1)))
                #v =  scstats.pearsonr(xy.T[0], xy.T[1])[0]
                ret[prompt][metric][system] = v
    return ret

def first(x):
    return next(iter(x))

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
        rho, p = scstats.pearsonr(data.T[0], data.T[1])
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

LABELS = {
    "rouge-l": "ROUGE-L",
    "rouge-1": "ROUGE-1",
    "rouge-2": "ROUGE-2",
    "ter": "TER",
    "sim": "VecSim",
    "meteor": "METEOR",
    "bleu": "BLEU-2",
    "gold": "Upper bound",


    "fastqa": "fastqa",
    "fastqa_ext": "fastqa\_ext",
    "snet.single": "snet",
    "snet.ensemble": "snet.ens",
    "*": "Combined"
    }
SYSTEMS = {
        "lqual": ["seq2seq", "pointer", "ml", "ml+rl"],
        "msmarco": ["fastqa", "fastqa_ext", "snet.single", "snet.ensemble"],
        }

def do_system_correlation(args):
    data = [json.loads(line) for line in open(args.input)]
    systems = SYSTEMS["task"]

    colors = cm.Dark2.colors[:4]
    #ix = SYSTEMS.index(args.data_system)

    ids = [vs[0] for vs in data]
    XY = np.array([vs[1:] for vs in data])
    xy = XY[[ids.index(system) for system in systems]]
    xy_lr = XY[[ids.index("{}-{}".format(system, 'lr')) for system in systems ]]
    xy_ll = XY[[ids.index("{}-{}".format(system, 'll')) for system in systems ]]
    xy_ur = XY[[ids.index("{}-{}".format(system, 'ur')) for system in systems ]]
    xy_ul = XY[[ids.index("{}-{}".format(system, 'ul')) for system in systems ]]

    print("rho", scstats.pearsonr(xy.T[0], xy.T[1]))

    #metrics = ["rouge-l", "rouge-1", "rouge-2", "meteor", "bleu", "sim", "gold",]

    plt.rc("font", size=20)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    xlim = np.array([XY.T[0].min(), XY.T[0].max()])
    coeffs = np.polyfit(xy.T[0], xy.T[3], 1)
    # Plot y == x line
    plt.plot(xlim, xlim * coeffs[0] + coeffs[1], linestyle='--', linewidth=2, zorder=-1)
    for _xy in [xy, xy_lr, xy_ur]:
        plt.errorbar(_xy.T[0], _xy.T[3], xerr=[_xy.T[0]-_xy.T[1],_xy.T[2]-_xy.T[0]], yerr=[_xy.T[3]-_xy.T[4], _xy.T[5]-_xy.T[3]], capsize=2, alpha=0.5, linestyle='', marker="", zorder=-1)

    plt.scatter(xy_lr.T[0], xy_lr.T[3], color=colors, marker=">")
    plt.scatter(xy_ur.T[0], xy_ur.T[3], color=colors, marker="^")
    #plt.scatter(xy_ll.T[0], xy_ll.T[3], color=colors, marker="<")
    #plt.scatter(xy_ul.T[0], xy_ul.T[3], color=colors[ix], marker="<")
    pts = plt.scatter(xy.T[0], xy.T[3], 100, c=colors, marker="o")
    plt.xlabel("Human judgement")
    plt.ylabel("ROUGE-L")
    plt.tight_layout()
    plt.legend(handles=[mp.Patch(color=colors[i], label=LABELS.get(system, system)) for i, system in enumerate(systems)])
    plt.savefig(args.output)

def do_correlation_table(args):
    with open(args.input) as f:
        data = load_jsonl(f)
    data = make_correlation(data)
    data = data[args.data_prompt]

    metrics = sorted(data.keys())
    systems = ["seq2seq", "pointer", "ml", "ml+rl", "*"]

    X = np.array([[data[metric][system] for system in systems] for metric in metrics])

    plt.rc("font", size=20)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    fig, ax = plt.subplots()

    plt.imshow(abs(X), cmap="viridis", origin="lower", aspect="auto", vmin=0.1, vmax=0.5)

    # Add the text
    y_size, x_size = X.shape
    x_positions = np.linspace(start=0, stop=x_size, num=x_size, endpoint=False)
    y_positions = np.linspace(start=0, stop=y_size, num=y_size, endpoint=False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = "{:.2f}".format(X[y_index, x_index])
            text_x = x + 0  # jump_x
            text_y = y + 0  # jump_y
            ax.text(text_x, text_y, label, color='white', ha='center', va='center', fontsize=11)



    ax.set_xticks(np.arange(len(systems)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([LABELS.get(s, s) for s in systems], rotation=45)
    ax.set_yticklabels([LABELS.get(s, s) for s in metrics])
    plt.colorbar(label=r"Pearson $\rho$")
    plt.xlabel("Systems")
    plt.ylabel("Metrics")

    plt.tight_layout()
    plt.savefig(args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('system-correlation', help='Plot the system-wide correlation of a models output with truth')
    command_parser.add_argument('-i', '--input', type=str, default="lqual_bias.json", help="Bias data")
    command_parser.add_argument('-t', '--task', type=str, choices=["lqual", "msmarco"], default="lqual", help="Bias data")
    command_parser.add_argument('-o', '--output', type=str, default="model_correlation.pdf", help="Where to save plot")
    command_parser.set_defaults(func=do_system_correlation)

    command_parser = subparsers.add_parser('correlation-table', help='Plot the system-wide correlation of a models output with truth')
    command_parser.add_argument('-i', '--input', type=str, default="lqual_correlation.jsonl", help="Bias data")
    command_parser.add_argument('-Dp', '--data-prompt', type=str, default="hter", help="An example trajectory for a task")
    command_parser.add_argument('-o', '--output', type=str, default="model_correlation.pdf", help="Where to save plot")
    command_parser.set_defaults(func=do_correlation_table)

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
