#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routines to pre-process datasets into a canonical format.
"""

import json
import sys
from collections import defaultdict
from tqdm import tqdm

from zipteedo.util import GzipFileType
from zipteedo.util import load_jsonl, save_jsonl, read_csv
from zipteedo import metrics

def pivot_data(data, input_key, metric_keys):
    """
    Returns another verison of this data with ids, systems as indices.
    """
    ret = defaultdict(dict)
    for datum in data:
        for system in datum["input"]["contents"]["system"].split(";"):
            id_ = datum["input"]["contents"]["id"]
            ret[id_][system] = {
                "input": datum["input"]["contents"][input_key],
                "metrics": {key: datum["output"][key] for key in metric_keys},
                }
    return ret

def do_snli(args):
    vmap = {
        "contradiction": 0,
        "neutral": 0,
        "entailment": 1,
        }

    save_jsonl(args.output,
               ({"x": datum,
                 "y*": vmap.get(datum['gold_label'], 0.5),
                 "ys": [vmap[l] for l in datum['annotator_labels']],
                } for datum in load_jsonl(args.input)))

def do_acceptability(args):
    save_jsonl(args.output,
               ({"x": datum.text,
                 "y*": (float(datum.mean_rating) - 1)/3,
                 "ys": [(float(l)-1)/3 for l in datum.rating_list.split(',')],
                 "system": datum.language,
                } for datum in read_csv(args.input))
              )

def compute_common(answer, ref):
    answer, ref = answer.lower().replace("\n", " "), ref.lower().replace("\n", " ")
    answer_, ref_ = metrics.word_tokenize(answer), metrics.word_tokenize(ref)

    common = {
        "bleu-2": metrics.bleu(answer_, [ref_], n=2) if answer_ else 0,
        "bleu-4": metrics.bleu(answer_, [ref_], n=4) if answer_ else 0,
        "meteor": metrics.meteor(answer, [ref]) if answer else 0,
        "ter": metrics.ter(answer_, ref_) if answer_ else 0,
        "sim": metrics.sim(answer, ref) if answer else 0,
        }
    common.update(metrics.rouge(answer_, ref_))
    return common

def do_msmarco_mean(args):
    # 1. Load data.
    data = load_jsonl(args.input)
    # Pivot the data so that we know can access other entries.
    refs = {datum["id"]: datum["answer"] for datum in data if datum["system"] == "reference"}

    for datum in tqdm(data):
        if datum["system"] == "reference": continue

        if datum["id"] not in refs: continue

        answer, ref = datum["answer"], refs[datum["id"]]
        common = compute_common(answer, ref)

        for system in datum["system"].split(";"):
            inst = {
                "id":  datum["id"],
                "system": system,
                "prompts": {
                    key: {
                        **common
                        } for key in ["AnyCorrect", "AvgCorrect",]
                    },
                }
            args.output.write(json.dumps(inst))
            args.output.write("\n")

def do_msmarco(args):
    # 1. Load data.
    data = load_jsonl(args.input)
    # Pivot the data so that we know can access other entries.
    index = pivot_data(data, "answer", ["AnyCorrect", "AvgCorrect"])

    ret = []
    for datum in tqdm(data):
        inp, out, rs = datum["input"]["contents"], datum["output"], datum["output"]["_responses"]


        entries = index[inp["id"]]
        answer, ref = inp["answer"], entries["reference"]["input"]
        common = compute_common(answer, ref)

        for system in datum["input"]["contents"]["system"].split(";"):
            inst = {
                "id":  inp["id"],
                "system": system,
                #"x": [[inp["query"], inp["passages"]], inp["answer"]],
                "annotators": rs["worker_ids"],
                "prompts": {
                    key: {
                        "gold": out[key] or 0,
                        "human": [r or 0 for r in rs[key]],
                        **common
                        } for key in ["AnyCorrect", "AvgCorrect",]
                    },
                }
            ret.append(inst)
    save_jsonl(args.output, ret)

def do_lqual(args):
    # 1. Load data.
    data = load_jsonl(args.input)
    # Pivot the data so that we know can access other entries.
    refs = {datum["input"]["contents"]["id"]: datum["input"]["contents"]["text"] for datum in data if datum["input"]["contents"]["system"] == "reference"}

    ret = []
    for datum in tqdm(data):
        inp, out, rs = datum["input"]["contents"], datum["output"], datum["output"]["_responses"]
        if not rs: continue
        if inp["id"] not in refs: continue

        answer, ref = inp["text"], refs[inp["id"]]
        # TODO: compute metrics for cross-examples.
        common = compute_common(answer, ref)

        for system in datum["input"]["contents"]["system"].split(";"):
            inst = {
                "id":  inp["id"],
                "system": system,
                #"x": inp["text"],
                "annotators": rs["worker_ids"],
                "prompts": {
                    key: {
                        "gold": out[key],
                        "human": rs[key],
                        **common
                        } for key in ["hter", "overall", "grammar", "redundancy"]
                    },
                }
        ret.append(inst)
    save_jsonl(args.output, ret)

def do_lqual_mean(args):
    # 1. Load data.
    data = load_jsonl(args.input)
    # Pivot the data so that we know can access other entries.
    refs = {datum["id"]: datum["text"] for datum in data if datum["system"] == "reference"}

    ret = []
    for datum in tqdm(data):
        if datum["system"] == "reference": continue

        if datum["id"] not in refs: continue

        answer, ref = datum["text"], refs[datum["id"]]
        common = compute_common(answer, ref)

        for system in datum["system"].split(";"):
            inst = {
                "id":  datum["id"],
                "system": system,
                "prompts": {
                    key: {
                        **common
                        } for key in ["hter", "overall", "grammar", "redundancy",]
                    },
                }
            ret.append(inst)
    save_jsonl(args.output, ret)


def do_dialog(args):
    ret = []
    for datum in load_jsonl(args.input):
        for i, elem in enumerate(datum["input"]["contents"]):
            obj = {"x": elem["context"],
                   "y*": (datum["output"]["overall"][i] + 2) / 5,
                   "ys": [(rs[i] + 2)/5 for rs in datum["output"]["_responses"]["overall"]],
                   "as": datum["output"]["_responses"]["worker_ids"],
                   "system": elem["system"],
                  }
            ret.append(obj)
    save_jsonl(args.output, ret)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess datasets.')
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('snli', help='Preprocess the SNLI dataset')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="SNLI json file")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="standardized data file")
    command_parser.set_defaults(func=do_snli)

    command_parser = subparsers.add_parser('acceptability', help='Preprocess the acceptability dataset')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="Acceptability CSV file")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="standardized data file")
    command_parser.set_defaults(func=do_acceptability)

    command_parser = subparsers.add_parser('msmarco', help='Preprocess the MSMarco dataset')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="MSMarco json file")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="standardized data file")
    command_parser.set_defaults(func=do_msmarco)

    command_parser = subparsers.add_parser('msmarco-mean', help='Preprocess the MSMarco dataset')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="MSMarco json file")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="standardized data file")
    command_parser.set_defaults(func=do_msmarco_mean)

    command_parser = subparsers.add_parser('lqual', help='Preprocess the lqual dataset')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="input json file")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="standardized data file")
    command_parser.set_defaults(func=do_lqual)

    command_parser = subparsers.add_parser('lqual-mean', help='Preprocess the lqual dataset')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="input json file")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="standardized data file")
    command_parser.set_defaults(func=do_lqual_mean)


    command_parser = subparsers.add_parser('dialog', help='Preprocess the dialog dataset')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="input json file")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="standardized data file")
    command_parser.set_defaults(func=do_dialog)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
