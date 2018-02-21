#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models to estimate performance.

A model is any callable that takes as input a datum.
"""
import pdb

import logging

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal

import numpy as np
import scipy.stats

logger = logging.getLogger(__name__)

def _get_flip_probability(ys, rho):
    # This comes from some silly algebra:
    # rho = (p - p')/sqrt(1 - pp' uu'/σ²)
    # p = 1/2 ± 1/2 sqrt[ 1 - 2 * (1-rho)^2 / (2 - rho^2b) ]
    #   where b = uu'/σ²
    u = np.mean(ys)
    sigma2 = np.std(ys)**2
    b = u * (1-u) / sigma2
    if rho > 0:
        p = 0.5 + 0.5 * np.sqrt(1 - 2 * (1-rho)**2/(2 - rho**2 * b))
    else:
        p = 0.5 - 0.5 * np.sqrt(1 - 2 * (1-rho)**2/(2 - rho**2 * b))
    assert p >= 0 and p <= 1

    return p

def OracleModel(data, rho=1.0, use_gold=True):
    ys = np.array([datum['y*'] for datum in data])
    sigma = np.std(ys)

    alpha, beta = rho, np.sqrt(1 - rho**2) * sigma

    ys_ = alpha * ys + beta * np.random.randn(len(ys))
    logging.info("rho = %.3f vs %.3f; alpha=%.3f, beta=%.3f", scipy.stats.pearsonr(ys, ys_)[0], rho, alpha, beta)
    logging.info("sigma_f = %.3f, sigma_g = %.3f", np.std(ys), np.std(ys_))

    sigma_a = np.mean([np.std(datum['ys']) for datum in data])

    def ret(data):
        if use_gold:
            ys = np.array([datum['y*'] for datum in data])
        else:
            ys = np.array([datum['y'] for datum in data])
        ys_ = alpha * ys + beta * np.random.randn(len(ys))

        eps_ = np.sqrt(np.abs(ys - ys_)**2 + sigma_a**2)

        return np.stack((ys_, eps_), -1)
    return ret

def OracleModelBinomial(data, rho=1.0, use_gold=True):
    if use_gold:
        ys = [datum['y*'] for datum in data]
    else:
        ys = [datum['y'] for datum in data]

    p = _get_flip_probability(ys, rho)

    cs = np.random.binomial(1, p, size=(len(data,)))
    Y = np.array([datum['y*'] for datum in data])
    Y_ = cs * Y + (1-cs) * (1-Y)
    print("rho = {:.3f} vs {}".format(scipy.stats.pearsonr(Y, Y_)[0], rho))

    def ret(data):
        cs = np.random.binomial(1, p, size=(len(data,)))

        if use_gold:
            ys = np.array([datum['y*'] for datum in data])
        else:
            ys = np.array([datum['y'] for datum in data])
        ys_ = cs * ys + (1-cs) * (1-ys)
        return np.stack((ys_, abs(ys - ys_)), -1)
    return ret


def ConstantModel(data, cnst=0., use_gold=True, lmb=0.):
    sigma_a = np.mean([np.std(datum['ys']) for datum in data])

    def ret(data):
        if use_gold:
            ys = np.array([datum['y*'] for datum in data])
        else:
            ys = np.array([datum['ys'] for datum in data])
        ys_ = cnst * np.ones(len(data))

        eps_ = np.sqrt(np.abs(ys - ys_)**2 + sigma_a**2)

        return np.stack((ys_, eps_), -1)

    return ret

class ConvModel(nn.Module):
    """
    A very simple bag-of-words model.
    """
    @classmethod
    def make_config(cls):
        return {
            "n_layers": 1,
            "hidden_dim": 50,
            "embed_layer": "conv",
            "dropout": 0.5,
            "update_L": False,

            # Should be set by main
            "n_features": 2,
            "embed_dim": 50,

            # Should never be changed.
            "output_dim": 1,
            }

    def __init__(self, config, L):
        super().__init__()

        # Setting up the ebemdding.
        if "vocab_dim" not in config:
            config["vocab_dim"] = L.shape[0]
        assert (config["vocab_dim"], config["embed_dim"]) == L.shape

        self.L = nn.Embedding(config["vocab_dim"], config["embed_dim"])
        self.L.weights = L
        self.L.requires_grad = config["update_L"]

        input_feature_dim = config["n_features"] * config["embed_dim"]

        # embed params
        if config["embed_layer"] == "lstm":
            # (//2 because bidirectional)
            self.h0 = Variable(torch.Tensor(2*config["n_layers"], config["hidden_dim"]//2))
            self.c0 = Variable(torch.Tensor(2*config["n_layers"], config["hidden_dim"]//2))
            self.W1 = nn.LSTM(input_feature_dim, config["hidden_dim"]//2,
                              num_layers=config["n_layers"],
                              dropout=0.1,
                              bidirectional=True,
                              batch_first=True,
                             )
        elif config["embed_layer"] == "conv":
            self.C0 = nn.Conv1d(input_feature_dim, config['hidden_dim'], 3, stride=1, padding=1)
            self.Cn = torch.nn.ModuleList([nn.Conv1d(config['hidden_dim'], config['hidden_dim'], 3, stride=1, padding=1) for _ in range(config["n_layers"])])
        elif config["embed_layer"] == "ff":
            self.C0 = nn.Linear(input_feature_dim, config['hidden_dim'])
        else:
            raise ValueError("Invalid embedding layer {}".format(config["embed_layer"]))

        # node params
        self.U = nn.Linear(config["hidden_dim"], config["output_dim"])
        xavier_normal(self.U.weight)

        self.config = config

    def _dropout(self, x):
        return F.dropout(x, self.config["dropout"])

    def _embed_sentence(self, x):
        batch_len, _, n_features = x.size()
        # Project onto L
        assert n_features >= self.config["n_features"]
        if n_features != self.config["n_features"]:
            logger.warning("Using only %d features when the data has %d", self.config["n_features"], n_features)

        x = torch.cat([self.L(x[:,:,i]) for i in range(self.config['n_features'])],2)

        if self.config["embed_layer"] == "lstm":
            h0 = self.h0.unsqueeze(1).repeat(1, batch_len, 1)
            c0 = self.c0.unsqueeze(1).repeat(1, batch_len, 1)
            x, _ = self.W1(x, (h0, c0))
        elif self.config["embed_layer"] == "conv":
            x = F.relu(self.C0(x.transpose(1,2)).transpose(1,2))
            for C in self.Cn:
                x = F.relu(C(x.transpose(1,2)).transpose(1,2))
        elif self.config["embed_layer"] == "ff":
            x = self.C0(x)
        return x

    def loss(self, xs, ys, ls):
        yhs = self.forward(xs, ls)
        loss = F.mse_loss(yhs, ys)
        return loss

    def forward(self, x, _):
        """
        Predicts on input graph @x, and lengths @ls.
        """
        x = self._embed_sentence(self._dropout(x))

        #pdb.set_trace()
        # x is B * L * F; average out L.
        x = torch.mean(x, 1)
        # These are logits, don't softmax
        y = self.U(self._dropout(x))
        #y = F.softmax(self.U(self._dropout(x)), -1)

        # TODO: have a better predictor of confidence.
        return y
