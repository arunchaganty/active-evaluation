#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routines to pre-process datasets into a canonical format.
"""

import sys
from zipteedo.util import GzipFileType, load_jsonl, save_jsonl

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess datasets.')
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('snli', help='Preprocess the SNLI dataset')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="SNLI json file")
    command_parser.add_argument('-o', '--output', type=GzipFileType('wt'), default=sys.stdout, help="standardized data file")
    command_parser.set_defaults(func=do_snli)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
