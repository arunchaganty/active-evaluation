#!/usr/bin/env python

# Python wrapper for METEOR implementation

import os
import pexpect
import numpy as np

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
SENT2VEC = '/home/chaganty/Research/resources/sent2vec/fasttext'
MODEL = '/home/chaganty/Research/resources/sent2vec/wiki_bigrams.bin'

def _norm(x):
    z = np.linalg.norm(x)
    return x/z if z > 1e-10 else 0 * x

class Sent2Vec:
    def __init__(self):
        self.cmd = [SENT2VEC, 'print-sentence-vectors', MODEL]
        self.child = pexpect.spawn(
            self.cmd[0], self.cmd[1:],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            encoding="utf8",
            timeout=300)

    def _comm(self, line):
        line = line.strip()
        self.child.sendline(line)
        resp = self.child.readline().strip()
        assert resp == line, "Expected {}, got {}".format(line, resp)
        resp = self.child.readline().strip()
        return resp

    def embed(self, x):
        vec = self._comm(x)
        return np.array([float(v) for v in vec.split()])

    def score(self, x, y):
        return _norm(self.embed(x)).dot(_norm(self.embed(y)))

    def __del__(self):
        if self.child:
            self.child.close()


def test_sent2vec():
    x = "Barack Obama will be the fourth president to receive the Nobel Peace Prize"
    y = "Barack Obama becomes the fourth American president to receive the Nobel Peace Prize"
    s2v = Sent2Vec()
    print(s2v.score(x,y))
