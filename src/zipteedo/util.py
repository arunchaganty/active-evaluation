"""
Utility functions for zipteedo.
"""
import csv
import logging
import json
import gzip
import argparse
from collections import namedtuple, Counter

import tqdm

def GzipFileType(*args, **kwargs):
    def _ret(path):
        try:
            if path.endswith('.gz'):
                return gzip.open(path, *args, **kwargs)
            else:
                return open(path, *args, **kwargs)
        except IOError as e:
            raise argparse.ArgumentError(path, e)
    return _ret

def load_jsonl(fstream):
    if isinstance(fstream, str):
        with open(fstream) as f:
            load_jsonl(f)
    return [json.loads(line) for line in fstream]

def save_jsonl(fstream, objs):
    if isinstance(fstream, str):
        with open(fstream, "w") as f:
            save_jsonl(f, objs)
    for obj in objs:
        fstream.write(json.dumps(obj))
        fstream.write("\n")

def read_csv(istream):
    reader = csv.reader(istream, delimiter='\t')
    header = next(reader)
    assert len(header) > 0, "Invalid header"
    Row = namedtuple('Row', header)
    return (Row(*row) for row in reader)

class StatCounter(object):
    def __init__(self, vs=None):
        self._Z = self._mean = 0

        if vs:
            self.update(vs)

    def update(self, values):
        for value in values:
            self._Z    += 1
            self._mean += (value - self._mean)/(self._Z)

    @property
    def mean(self):
        return self._mean

    @property
    def weight(self):
        return self._Z

    def clear(self):
        self._mean = self._Z = 0

    def __iadd__(self, value):
        if isinstance(value, tuple):
            value, weight = value
        else:
            value, weight = value, 1.

        self._Z    += weight
        self._mean += weight*(value - self._mean)/(self._Z)

        return self

    def __repr__(self):
        return "<Avg: {} (from {})>".format(self._mean, self._Z)

class StatVarCounter(object):
    def __init__(self, *vs):
        self._Z = self._mean = self._var = 0

        if vs:
            self.update(vs)

    def update(self, values):
        for value in values:
            if isinstance(value, tuple):
                assert len(value) == 2
                value, weight = value
            else:
                value, weight = value, 1
            self._Z    += weight
            self._mean += weight*(value - self._mean)/(self._Z)
            self._var  += weight*(value**2 - self._var)/(self._Z)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var - self._mean**2

    @property
    def weight(self):
        return self._Z

    def clear(self):
        self._mean = self._var = self._Z = 0

    def __iadd__(self, value, weight=1):
        self._Z    += weight
        self._mean += weight*(value - self._mean)/(self._Z)
        self._var  += weight*(value**2 - self._var)/(self._Z)

        return self

    def __repr__(self):
        return "<Avg: {} (from {})>".format(self._mean, self._Z)

def test_averager():
    avg = StatVarCounter()

    assert avg.mean == 0 and avg.weight == 0
    avg += 1
    assert avg.mean == 1 and avg.weight == 1
    avg += 0
    assert avg.mean == 0.5 and avg.weight == 2
    avg += 0, 2
    assert avg.mean == 0.25 and avg.weight == 4
    assert avg.var == (0.25 - 0.25**2)

def trynumber(x):
    if x.lower() == "true":
        return True
    elif x.lower() == "false":
        return False
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        pass
    return x

def dictstr(x):
    """
    A converter from string to dictionary, for use in argparse.
    """
    if "=" in x:
        k, v = x.split("=")
        # Try to parse v.
        return (k, trynumber(v))
    else:
        return (x, True)

def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}

def test_build_dict():
    words = "a a b a b b a c b a c a".split()
    tok2id = build_dict(words, max_words=2, offset=1)
    assert tok2id.get('a') == 1
    assert tok2id.get('b') == 2
    assert tok2id.get('c') is None

def invert_dict(tok2id):
    """
    Inverts a dictionary from tokens to ids.
    """
    ret = [None for _ in range(max(tok2id.values())+1)]
    for k, v in tok2id.items():
        ret[v] = k
    return ret

def test_invert_dict():
    id2tok = "a b c d e f g".split()
    tok2id = {t: i for i, t in enumerate(id2tok)}

    id2tok_ = invert_dict(tok2id)

    assert id2tok_ == id2tok

class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)
        self.flush()

    def flush(self):
        super().flush()
