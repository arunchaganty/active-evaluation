"""
Utility functions for zipteedo.
"""
import csv
import json
import gzip
import argparse
from collections import namedtuple

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
    def __init__(self, *vs):
        self._Z = self._mean = self._var = 0

        if vs:
            self.update(vs)

    def add(self, value, weight=1):
        self._Z    += weight
        self._mean += weight*(value - self._mean)/(self._Z)
        self._var  += weight*(value**2 - self._var)/(self._Z)

    def update(self, values):
        for value in values:
            if isinstance(value, tuple):
                assert len(value) == 2
                value, weight = value
            else:
                value, weight = value, 1
        self.add(value, weight)

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

    def __iadd__(self, other):
        if isinstance(other, tuple):
            assert len(other) == 2
            self.add(*other)
        else:
            self.add(other)

    def __repr__(self):
        return "<Avg: {} (from {})>".format(self._mean, self._Z)

def test_averager():
    avg = StatCounter()

    assert avg.mean == 0 and avg.weight == 0
    avg += 1
    assert avg.mean == 1 and avg.weight == 1
    avg += 0
    avg.update(0,1)
    assert avg.mean == 0.5 and avg.weight == 2
    avg += 0, 2
    avg.update(0,2)
    assert avg.mean == 0.25 and avg.weight == 4
    assert avg.var == (0.25 - 0.25**2)
