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
