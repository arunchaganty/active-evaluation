"""
Plot various visualizations
"""
import sys
import json
from collections import Counter


import numpy as np
import scipy.stats as scstats
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mp



from zipteedo.util import GzipFileType, load_jsonl, first
from zipteedo.stats import get_correlations, get_data_efficiencies, make_bias_table
from zipteedo.viz import draw_matrix, violinplot
import zipteedo.estimators as zpe

LABELS = {
    "rouge-l": "ROUGE-L",
    "rouge-1": "ROUGE-1",
    "rouge-2": "ROUGE-2",
    "ter": "TER",
    "sim": "VecSim",
    "meteor": "METEOR",
    "bleu": "BLEU-2",
    "bleu-2": "BLEU-2",
    "bleu-4": "BLEU-4",
    "gold": "Upper bound",

    "hter": "Edit",
    "lqual": "CNN/DailyMail",
    "msmarco": "MSMARCO",

    "fastqa": "fastqa",
    "fastqa_ext": "fastqa_ext",
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

    plt.rc("font", size=16)
    plt.rc("text", usetex=False)
    #plt.rc("figure", figsize=(10,10))

    draw_matrix(X, with_values=True,
                x_labels=[LABELS.get(s, s) for s in systems],
                y_labels=[LABELS.get(m, m) for m in metrics],)

    plt.colorbar(label=r"Pearson ρ")
    plt.xlabel("Systems")
    plt.ylabel("Metrics")

    if args.with_title:
        task = first(key for key, values in PROMPTS.items() if prompt in values)
        plt.title(r"Correlations on {} using the {} prompt".format(
            LABELS.get(task, task),
            LABELS.get(prompt, prompt),
            ), fontsize=14)

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
    plt.rc("text", usetex=False)
    #plt.rc("figure", figsize=(10,10))

    plt.xlabel("Number of samples")
    plt.ylabel(r"80% confidence interval")
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
        plt.title(r"{} on {} using the {} prompt".format(
            LABELS.get(system, system),
            LABELS.get(task, task),
            LABELS.get(prompt, prompt),
            ), fontsize=14)

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

    plt.rc("font", size=16)
    plt.rc("text", usetex=False)

    draw_matrix(X, with_values=True,
                x_labels=[LABELS.get(s, s) for s in systems],
                y_labels=[LABELS.get(m, m) for m in metrics],
                vmin=0.9, vmax=1.3)

    plt.colorbar(label="Data efficiency")
    plt.xlabel("Systems")
    plt.ylabel("Metrics")

    if args.with_title:
        plt.title(r"Data efficiencies on {} using the {} prompt".format(
            LABELS.get(task, task),
            LABELS.get(prompt, prompt),
            ), fontsize=14)


    plt.tight_layout()
    plt.savefig(args.output)

def do_system_correlation(args):
    data = [json.loads(line) for line in open(args.input)]
    prompt, metric = args.data_prompt, args.data_metric
    task = first(key for key, values in PROMPTS.items() if prompt in values)
    systems = SYSTEMS[task]

    # Group by data by system.
    data = make_bias_table(data, prompt, metric, ["lr", "ur"])

    plt.rc("font", size=16)
    plt.rc("text", usetex=False)
    plt.rc("figure", figsize=(8,6))
    colors = cm.Dark2.colors[:len(systems)]

    def _thresh(y):
        return max(min(y, 1), -1)

    # 0. Plot the xy correlation curve.
    xy = np.array([[x, _thresh(y)] for system in systems for (x, *_), (y, *_) in [data[system]["default"]]])
    xlim = np.array([xy.T[0].min(), xy.T[0].max()])
    coeffs = np.polyfit(xy.T[0], xy.T[1], 1)
    plt.plot(xlim, xlim * coeffs[0] + coeffs[1], linestyle='--', linewidth=2, zorder=-1)

    # 1. Plot actual data points with error bars.
    xy = np.array([[x, y] for system in systems for (x, *_), (y, *_) in data[system].values()])
    xy_l = np.array([[x, y] for system in systems for (_, x, _), (_, y, _) in data[system].values()])
    xy_u = np.array([[x, y] for system in systems for (_, _, x), (_, _, y) in data[system].values()])
    plt.errorbar(xy.T[0], xy.T[1],
                 xerr=[(xy - xy_l).T[0], (xy_u - xy).T[0]],
                 yerr=[(xy - xy_l).T[1], (xy_u - xy).T[1]],
                 capsize=2, alpha=0.5, linestyle='', marker="", zorder=-1)

    # 2. Plot markers.
    xy = np.array([[x, y] for system in systems for (x, *_), (y, *_) in [data[system]["default"]]])
    xy_lr = np.array([[x, y] for system in systems for (x, *_), (y, *_) in [data[system]["lr"]]])
    xy_ur = np.array([[x, y] for system in systems for (x, *_), (y, *_) in [data[system]["ur"]]])

    plt.scatter(xy_lr.T[0], xy_lr.T[1], color=colors, marker=">")
    plt.scatter(xy_ur.T[0], xy_ur.T[1], color=colors, marker="^")
    plt.scatter(xy.T[0], xy.T[1], 100, c=colors, marker="o")
    plt.xlabel(r"Human judgement ({})".format(LABELS.get(prompt, prompt)))
    plt.ylabel(LABELS.get(metric, metric))

    if args.with_title:
        task = first(key for key, values in PROMPTS.items() if prompt in values)
        plt.title(r"System-level correlation on {}".format(
            LABELS.get(task, task),
            ), fontsize=14)

    plt.tight_layout()

    plt.legend(handles=[mp.Patch(color=colors[i], label=LABELS.get(system, system)) for i, system in enumerate(systems)])

    plt.savefig(args.output)


def _snap(vs, points):
    ret = []
    for x, y in vs:
        ret.append((first(x_ for x_ in points if x_ >= x), y))
    return np.array(ret)


def do_instance_correlation(args):
    data = [json.loads(line) for line in open(args.input)]
    prompt, metric = args.data_prompt, args.data_metric
    task = first(key for key, values in PROMPTS.items() if prompt in values)
    systems = SYSTEMS[task]

    # Group by data by system.
    plt.rc("font", size=16)
    plt.rc("text", usetex=False)
    plt.rc("figure", figsize=(6,8))
    colors = cm.Dark2.colors[:len(systems)]

    # 1. How many distinct Y values exist?
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)

    def _thresh(y):
        return max(min(y, 1), -1)

    xy = {system: np.array([[_thresh(datum["prompts"][prompt]["gold"]), datum["prompts"][prompt][metric]] for datum in data if system in datum["system"].split(";")])
            for system in systems}

    if args.bins:
        y = np.array([_thresh(datum["prompts"][prompt]["gold"]) for datum in data])
        distinct_values = np.linspace(y.min(), y.max(), args.bins)
        plt.xticks(distinct_values)

        for system in systems:
            xy[system] = _snap(xy[system], distinct_values)

        # 2. Make violin plots.
        for i, system in enumerate(systems):
            violinplot(axs[i], xy[system], distinct_values, colors[i])

    for i, system in enumerate(systems):
        x, y = xy[system].T[0], xy[system].T[1]
        axs[i].scatter(x, y, alpha=0.3, marker='.', color=colors[i])

    for i, system in enumerate(systems):
        x, y = xy[system].T[0], xy[system].T[1]
        coeffs = np.polyfit(x, y, 1)
        xlim = np.array([x.min(), x.max()])
        axs[i].plot(xlim, xlim * coeffs[0] + coeffs[1], linestyle='--', linewidth=1, zorder=-1, color=colors[i])

    for i, system in enumerate(systems):
        axs[i].text(1.2, 0.5, LABELS.get(system, system), va='center', rotation='vertical')

    plt.xlabel(r"Human judgement ({})".format(LABELS.get(prompt, prompt)))
    #plt.text(-1, 0, LABELS.get(metric, metric), va="center")
    fig.text(0.01, 0.5, LABELS.get(metric, metric), va='center', rotation='vertical')

    if args.with_title:
        task = first(key for key, values in PROMPTS.items() if prompt in values)
        axs[0].set_title(r"Instance-level correlation on {}".format(
            LABELS.get(task, task),
            ), fontsize=14)

    plt.subplots_adjust(wspace=0, hspace=0.05)
    #plt.tight_layout()

    #plt.legend(handles=[mp.Patch(color=colors[i], label=LABELS.get(system, system)) for i, system in enumerate(systems)])

    plt.savefig(args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()

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
    command_parser.add_argument('-o',  '--output',      type=str, default="trajectory.pdf", help="An example trajectory for a task")
    command_parser.add_argument('-Dp', '--data-prompt', type=str, default="hter", help="An example trajectory for a task")
    command_parser.add_argument('-Dm', '--data-metric', type=str, default="sim", help="An example trajectory for a task")
    command_parser.add_argument('-Ds', '--data-system', type=str, default="seq2seq", help="An example trajectory for a task")
    command_parser.add_argument('-wt', '--with-title',  action="store_true", help="An example trajectory for a task")
    command_parser.set_defaults(func=do_trajectory)

    command_parser = subparsers.add_parser('system-correlation', help='Plot the system-wide correlation of a models output with truth')
    command_parser.add_argument('-i', '--input', type=str, default="lqual.json", help="Bias data")
    command_parser.add_argument('-Dp', '--data-prompt', type=str, default="overall", help="An example trajectory for a task")
    command_parser.add_argument('-Dm', '--data-metric', type=str, default="sim", help="An example trajectory for a task")
    command_parser.add_argument('-o', '--output', type=str, default="system_correlation.pdf", help="Where to save plot")
    command_parser.add_argument('-wt', '--with-title',  action="store_true", help="An example trajectory for a task")
    command_parser.set_defaults(func=do_system_correlation)

    command_parser = subparsers.add_parser('instance-correlation', help='Plot the system-wide correlation of a models output with truth')
    command_parser.add_argument('-i', '--input', type=str, default="lqual.json", help="Bias data")
    command_parser.add_argument('-Dp', '--data-prompt', type=str, default="overall", help="An example trajectory for a task")
    command_parser.add_argument('-Dm', '--data-metric', type=str, default="sim", help="An example trajectory for a task")
    command_parser.add_argument('-o', '--output', type=str, default="instance_correlation.pdf", help="Where to save plot")
    command_parser.add_argument('-wt', '--with-title',  action="store_true", help="An example trajectory for a task")
    command_parser.add_argument('-b', '--bins',  type=int, help="An example trajectory for a task")
    command_parser.set_defaults(func=do_instance_correlation)


    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
