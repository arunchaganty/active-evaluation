"""
Useful routines for torch.
"""

import pdb
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .util import StatCounter

logger = logging.getLogger(__name__)

INF = 1e5
NINF = -1 * INF

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        xs, ys = self.data[i]
        xs = torch.from_numpy(xs)
        return xs, ys

def pack_sequence(batch, pad=0):
    """
    Assumes @batch is a list of B Tensors, each Tensor is Lx*

    Returns list of BxTx*, list[int]
    """
    lengths = torch.LongTensor([len(ex) for ex in batch])
    B, T = len(batch), max(lengths)
    shape = (B, T, *batch[0].size()[1:])

    ret = batch[0].new(*shape)
    ret.fill_(pad)
    ret = ret.view(B, T, -1)
    # Copy over data.
    for i, ex in enumerate(batch):
        ret[i, :lengths[i], :] = ex.view(lengths[i], -1)
    ret = ret.view(*shape)

    return ret, lengths

def create_batch(batch, pad=0):
    xs, ys = zip(*batch)
    xs, ls = pack_sequence(xs, pad)
    ys = torch.FloatTensor(ys)
    return [xs, ys, ls]

def length_mask(x, lengths, use_cuda=False):
    """
    Assumes @x is B, T, * and returns B*L, *
    """
    shape = x.size()

    mask = torch.ByteTensor(*shape)
    mask.fill_(0)
    mask = mask.view(*shape[:2], -1)
    for i, _ in enumerate(x):
        mask[i, :lengths[i], :] = 1

    if use_cuda:
        mask=mask.cuda()

    return x[mask].view(-1, *shape[2:])

def position_matrix(seq_len):
    """
    Returns a seq_len x seq_len matrix where position (i,j) is i - j
    """
    x = torch.arange(0, seq_len).unsqueeze(-1).expand(seq_len, seq_len)
    return x.t() - x

def to_scalarv(var):
    # returns a python float
    return var.view(-1).data.tolist()

def to_scalar(var):
    # returns a python float
    return var.view(-1).tolist()


def log_sum_exp(mat, axis=0):
    max_score, _ = mat.max(axis)

    if mat.dim() > 1:
        max_score_broadcast = max_score.unsqueeze(axis).expand_as(mat)
    else:
        max_score_broadcast = max_score

    return max_score + \
        torch.log(torch.sum(torch.exp(mat - max_score_broadcast), axis))

def softmax(x, axis=-1):
    """
    Computes softmax over the elements of x.
    """
    return torch.exp(x - log_sum_exp(x, axis).unsqueeze(axis).expand(*x.size()))

def to_one_hot(x, dim):
    """
    Converts a LongTensor @x from an integer format into one with float format and one-hot.
    """
    assert x.type() == 'torch.LongTensor'
    ret = torch.zeros(*x.size(), dim)
    ret.scatter_(-1, x, 1.)
    return ret

def to_one_hot_logits(x, dim):
    """
    Converts a LongTensor @x from an integer format into one with float format and one-hot.
    """
    assert x.type() == 'torch.LongTensor'
    ret = NINF * torch.ones(*x.size(), dim)
    ret.scatter_(-1, x.unsqueeze(-1), 0.)
    return ret

def run_epoch(model, dataset, optimizer=None, train=False, use_cuda=False, collapse_spans=True):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = StatCounter()

    for xs, ys, ls in dataset:
        if train and optimizer:
            optimizer.zero_grad()
        if use_cuda:
            xs, ys, ls = xs.cuda(), ys.cuda(), ls.cuda()

        xs, ys, ls = Variable(xs), Variable(ys), Variable(ls)

        loss = model.loss(xs, ys, ls)
        if train and optimizer:
            loss.backward()
            optimizer.step()

        batch_size = xs.size()[0]
        epoch_loss += loss.data[0], batch_size

        if hasattr(dataset, "set_postfix"):
            dataset.set_postfix(loss=epoch_loss.mean)

    return epoch_loss

def train_model(model, train_dataset, dev_dataset=None, n_epochs=15, use_cuda=False, **kwargs):
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=create_batch)
    dev_loader = dev_dataset and DataLoader(dev_dataset, batch_size=16, collate_fn=create_batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if use_cuda:
        model.cuda()

    epoch_it = trange(n_epochs, desc="Epochs")
    for epoch in epoch_it:
        train_loss = run_epoch(model, tqdm(train_loader, desc="Train batch"), optimizer, train=True, use_cuda=use_cuda)
        logger.info("Train %d loss: %.2f", epoch, train_loss.mean)

        if dev_loader:
            dev_loss = run_epoch(model, tqdm(dev_loader, desc="Dev batch"), train=False, use_cuda=use_cuda)
            logger.info("Dev %d Loss: %.2f", epoch, dev_loss.mean)
            epoch_it.set_postfix(
                loss="{:.3f}".format(dev_loss.mean),
                )
        else:
            dev_stats = StatCounter()

    return model, train_loss.mean, dev_loss.mean

def run_model(model, dataset, use_cuda=False, collapse_spans=False, **kwargs):
    loader = DataLoader(dataset, batch_size=16, collate_fn=create_batch)

    if use_cuda:
        model.cuda()

    output = []

    for xs, _, ls in tqdm(loader, desc="run batch"):
        if use_cuda:
            xs = xs.cuda()
        yhs = model(Variable(xs), Variable(ls))
        #yhs = softmax(yhs)

        output.extend(yhs.data.numpy().tolist())
    return output
