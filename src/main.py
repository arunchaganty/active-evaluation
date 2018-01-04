#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main routine for zipteedo.
"""
import pdb
import csv

import gc
import os
import sys
import json

import logging
import logging.config

import numpy as np
from tqdm import tqdm, trange

import torch

from zipteedo import estimators, models
from zipteedo import wv
from zipteedo.helper import Acceptability
from zipteedo.util import GzipFileType, load_jsonl, dictstr
from zipteedo.torch_utils import Dataset, train_model, run_model
from zipteedo.vis import plot_trajectories

logger = logging.getLogger(__name__)

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

def apply_data_transform(args, obj, data):
    if args.transform_mean:
        data -= obj["truth"]
    return data

def split(dataset, train=0.8):
    """
    Splits the dataset into train and test.
    """
    pivot = int(len(dataset) * train)
    return dataset[:pivot], dataset[pivot:]

def cross_validate(dataset, fn, splits=5, iters=None, start=0, **kwargs):
    block_size = len(dataset)//splits

    if iters is None:
        iters = splits

    it = trange(start, min(splits, start+iters), desc="Cross-validation")

    train_loss, dev_loss, output = [], [], []
    for i in it:
        train = dataset[:i*block_size] + dataset[(i+1)*block_size:]
        dev = dataset[i*block_size:(i+1)*block_size]

        _, output_, train_loss_, dev_loss_ = fn(train, dev, **kwargs)
        output.extend(output_)
        train_loss.append(train_loss_)
        dev_loss.append(dev_loss_)

        it.set_postfix(loss="{:.3f}".format(dev_loss_))

    return output, np.array(train_loss), np.array(dev_loss)

def run_split(train_raw, dev_raw, helper, model_class, config, use_cuda=False, n_epochs=15):
    gc.collect()
    train = Dataset(train_raw)
    dev = dev_raw and Dataset(dev_raw)

    # (d) Train model
    model = model_class(config, helper.embeddings)
    model, train_loss, dev_loss = train_model(model, train, dev, use_cuda=use_cuda, n_epochs=n_epochs)
    output = run_model(model, dev, use_cuda) if dev else []

    return model, output, train_loss, dev_loss

def write_stats(stats, f):
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["split",] + sorted([
        "train_loss",
        "dev_loss",
        ]))
    for i, row in enumerate(stats):
        writer.writerow([i,] + row.tolist())
    writer.writerow(["mean",] + np.mean(stats,0).tolist())

def do_train(args):
    # (a) Load data and embeddings
    helper_args = dict(args.helper_args or [])
    model_args = dict(args.model_args or [])
    train_data = load_jsonl(args.input)

    helper = Acceptability.build(train_data, **helper_args)
    helper.add_embeddings(args.embeddings_vocab, args.embeddings_vectors)

    # (b) Get model and configure it.
    Model = models.ConvModel
    config = Model.make_config()
    config.update(model_args)
    config.update({"n_features": helper.n_features, "vocab_dim": helper.vocab_dim, "embed_dim": helper.embed_dim})

    with open(os.path.join(args.model_path, "helper.pkl"), "wb") as f:
        helper.save(f)
    with open(os.path.join(args.model_path, "model.config"), "w") as f:
        json.dump(config, f)

    # (c) Vectorize the data.
    train = helper.vectorize(train_data)

    # Cross-validation information.
    if args.cross_validation_iters > 0:
        dev_output, train_stats, dev_stats = cross_validate(
            train, run_split,
            splits=args.cross_validation_splits,
            iters=args.cross_validation_iters,
            start=args.cross_validation_start,
            model_class=Model, config=config, helper=helper,
            n_epochs=args.n_epochs)
        logger.info("Final cross-validated train stats: %s", np.mean(train_stats,0))
        logger.info("Final cross-validated dev stats: %s", np.mean(dev_stats,0))

        with open(os.path.join(args.model_path, "scores"), "w") as f:
            write_stats(np.stack([train_stats, dev_stats]), f)

        with open(os.path.join(args.model_path, "predictions.jsonl"), 'w') as f:
            for datum, (y_,) in zip(train_data, dev_output):
                datum['y^'] = y_
                json.dump(datum, f)
                f.write("\n")

if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')

    import argparse
    parser = argparse.ArgumentParser(description='Zipteedo: fast, economical human evaluation.')
    parser.add_argument('-s', '--seed', type=int, default=42, help="Random seed for experiments.")
    parser.add_argument('-mp', '--model-path', default="out", help="Where to load/save models.")
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

    command_parser = subparsers.add_parser('train', help='Trains an evaluation model on some data')
    command_parser.add_argument('-i', '--input', type=GzipFileType('rt'), default=sys.stdin, help="Path to an input dataset.")
    command_parser.add_argument('-H', '--helper', type=str, required=True, help='Featurizer/Helper to use')
    command_parser.add_argument('-Xh', '--helper_args', type=dictstr, nargs="+", default=None, help='Features to use in the helper')
    command_parser.add_argument('-cvs', '--cross-validation-splits', type=int, default=10, help="Cross-validation splits to use")
    command_parser.add_argument('-cvi', '--cross-validation-iters', type=int, default=1, help="Cross-validation splits to use")
    command_parser.add_argument('-cvx', '--cross-validation-start', type=int, default=0, help="Cross-validation splits to start with")
    command_parser.add_argument('-n', '--n_epochs', type=int, default=10, help='How many iterations to train')
    command_parser.add_argument('-M', '--model', type=str, default=None, help="Which model to use")
    command_parser.add_argument('-Xm', '--model-args', type=dictstr, nargs="+", default=None, help="Extra arguments for the model")
    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        os.makedirs(ARGS.model_path, exist_ok=True)
        root_logger = logging.getLogger()
        root_logger.addHandler(logging.FileHandler(os.path.join(ARGS.model_path, "log")))

        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
        ARGS.func(ARGS)
