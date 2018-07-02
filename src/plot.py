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

from zipteedo.util import GzipFileType, load_jsonl, first
from zipteedo.stats import get_correlations, get_data_efficiencies
from zipteedo.viz import draw_matrix
import zipteedo.estimators as zpe

LABELS = {
    "rouge-l": "ROUGE-L",
    "rouge-1": "ROUGE-1",
    "rouge-2": "ROUGE-2",
    "ter": "TER",
    "sim": "VecSim",
    "meteor": "METEOR",
    "bleu": "BLEU-2",
    "gold": "Upper bound",

    "hter": "Edit",
    "lqual": "CNN/DailyMail",
    "msmarco": "MSMARCO",

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

PROMPTS = {
    "lqual": ["hter", "overall", "redundancy", "grammar"],
    "msmarco": ["AnyCorrect", "AvgCorrect"],
    }



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
    data = get_correlations(data)
    data = data[args.data_prompt]

    prompt = args.data_prompt
    metrics = sorted(data.keys())
    task = first(key for key, values in PROMPTS.items() if prompt in values)
    systems = SYSTEMS[task] + ["*"]

    X = np.array([[data[metric][system] for system in systems] for metric in metrics])

    plt.rc("font", size=20)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    draw_matrix(X, with_values=True,
                x_labels=[LABELS.get(s, s) for s in systems],
                y_labels=[LABELS.get(m, m) for m in metrics],)

    plt.colorbar(label=r"Pearson $\rho$")
    plt.xlabel("Systems")
    plt.ylabel("Metrics")

    if args.with_title:
        task = first(key for key, values in PROMPTS.items() if prompt in values)
        plt.title(r"Correlations on {} using the \texttt{{{}}} prompt".format(
            LABELS.get(task, task),
            LABELS.get(prompt, prompt),
            ), fontsize=16)

    plt.tight_layout()
    plt.savefig(args.output)


def do_trajectory(args):
    data = [json.loads(line) for line in open(args.input, "rt")]
    data = {(obj["system"], obj["metric"], obj["prompt"], obj["estimator"]): obj for obj in data}

    if args.input_gold:
        data_gold = [json.loads(line) for line in open(args.input_gold, "rt")]
        data_gold = {(obj["system"], obj["metric"], obj["prompt"], obj["estimator"]): obj for obj in data_gold}
    else:
        data_gold = None

    colors = cm.tab10.colors

    system = args.data_system
    metric = args.data_metric
    prompt = args.data_prompt

    baseline = np.array(data[system, metric, prompt, "simple"]["summary"])
    model    = np.array(data[system, metric, prompt, "model_variate"]["summary"])
    if data_gold:
        model_gold = np.array(data_gold[system, metric, prompt, "model_variate"]["summary"])
    gold     = np.array(data[system, "gold", prompt, "model_variate"]["summary"])

    plt.rc("font", size=16)
    plt.rc("text", usetex=True)
    #plt.rc("figure", figsize=(10,10))

    plt.xlabel("Number of samples")
    plt.ylabel(r"80\% confidence interval")
    plt.plot(baseline.T[2] - baseline.T[1], color=colors[0], label="Humans")
    plt.plot(model.T[2] - model.T[1], color=colors[1], label="Humans + {}".format(LABELS.get(metric,metric)))
    if data_gold:
        plt.plot(model_gold.T[2] - model_gold.T[1], ':', color=colors[2], label="Noiseless humans + {}".format(LABELS.get(metric,metric)))
    plt.plot(gold.T[2] - gold.T[1], ':', color=colors[4], label="Humans + perfect metric")

    plt.xlim([0, 500])
    plt.ylim([0.05, 0.2])

    plt.legend()

    if args.with_title:
        task = first(key for key, values in PROMPTS.items() if prompt in values)
        plt.title(r"\texttt{{{}}} on {} using the \texttt{{{}}} prompt".format(
            LABELS.get(system, system),
            LABELS.get(task, task),
            LABELS.get(prompt, prompt),
            ), fontsize=16)

    plt.tight_layout()
    plt.savefig(args.output)


def do_data_efficiency_table(args):
    data = [json.loads(line) for line in open(args.input, "rt")]
    data = get_data_efficiencies(data)

    prompt = args.data_prompt
    metrics = sorted(data.keys())
    task = first(key for key, values in PROMPTS.items() if prompt in values)
    systems = SYSTEMS[task]

    X = np.array([[data[metric][prompt][system]**2 for system in systems] for metric in metrics])

    plt.rc("font", size=20)
    plt.rc("text", usetex=True)

    draw_matrix(X, with_values=True,
                x_labels=[LABELS.get(s, s) for s in systems],
                y_labels=[LABELS.get(m, m) for m in metrics],
                vmin=0.9, vmax=1.5)

    plt.colorbar(label="Data efficiency")
    plt.xlabel("Systems")
    plt.ylabel("Metrics")

    if args.with_title:
        plt.title(r"Data efficiencies on {} using the \texttt{{{}}} prompt".format(
            LABELS.get(task, task),
            LABELS.get(prompt, prompt),
            ), fontsize=16)


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
    command_parser.add_argument('-o', '--output', type=str, default="correlations.pdf", help="Where to save plot")
    command_parser.add_argument('-wt', '--with-title',  action="store_true", help="An example trajectory for a task")
    command_parser.set_defaults(func=do_correlation_table)

    command_parser = subparsers.add_parser('data-efficiency-table', help='Plot data efficiencies for different systems and automatic metrics')
    command_parser.add_argument('-i', '--input', type=str, default="lqual_trajectories.jsonl", help="Trajecotry data")
    command_parser.add_argument('-Dp', '--data-prompt', type=str, default="hter", help="An example trajectory for a task")
    command_parser.add_argument('-o', '--output', type=str, default="data_efficiencies.pdf", help="Where to save plot")
    command_parser.add_argument('-wt', '--with-title',  action="store_true", help="An example trajectory for a task")
    command_parser.set_defaults(func=do_data_efficiency_table)

    command_parser = subparsers.add_parser('trajectory', help='Plot a trajectory for an estimator')
    command_parser.add_argument('-i',  '--input',       type=str, default="lqual/lqual_trajectories.json", help="")
    command_parser.add_argument('-ig', '--input-gold',  type=str, help="")
    command_parser.add_argument('-o',  '--output',      type=str, default="lqual_trajectory.pdf", help="An example trajectory for a task")
    command_parser.add_argument('-Dp', '--data-prompt', type=str, default="hter", help="An example trajectory for a task")
    command_parser.add_argument('-Dm', '--data-metric', type=str, default="sim", help="An example trajectory for a task")
    command_parser.add_argument('-Ds', '--data-system', type=str, default="seq2seq", help="An example trajectory for a task")
    command_parser.add_argument('-wt', '--with-title',  action="store_true", help="An example trajectory for a task")
    command_parser.set_defaults(func=do_trajectory)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
