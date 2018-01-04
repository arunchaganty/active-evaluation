"""
Helpers to vectorize and process data
"""
import pickle
import logging

import numpy as np
from tqdm import tqdm

from .util import build_dict, invert_dict
from .wv import load_word_vector_mapping

logger = logging.getLogger(__name__)

P_LEMMA = "LEMMA:"
P_POS = "POS:"
P_NER = "NER:"
P_CASE = "CASE:"
P_DEPREL = "REL:"
CASES = ["aa", "AA", "Aa", "aA"]
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK = "UNK"
NUM = "###"

def deprel(tag):
    return tag.split(":")[0]

def casing(word):
    if len(word) == 0: return word

    # all lowercase
    if word.islower(): return "aa"
    # all uppercase
    elif word.isupper(): return "AA"
    # starts with capital
    elif word[0].isupper(): return "Aa"
    # has non-initial capital
    else: return "aA"

def uncasing(word, case):
    if len(word) == 0: return word

    if case == "aa": return word.lower()
    elif case == "AA": return word.upper()
    elif case == "Aa": return word.title()
    else: return word

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    try:
        float(word)
        return NUM
    except ValueError:
        return word.lower()

class Helper(object):
    """
    Helper for acceptability
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    FEATURES = ["word", "casing",]

    def __init__(self, tok2id, features):
        self.tok2id = tok2id
        self.id2tok = invert_dict(tok2id)
        self.embeddings = None
        self.features = features

    @property
    def vocab_dim(self):
        return self.embeddings.shape[0]

    @property
    def embed_dim(self):
        return self.embeddings.shape[1]

    @property
    def n_features(self):
        return len(self.features)

    @property
    def n_features_exact(self):
        return 1 if "depparse" in self.features else 0

    def vectorize_example(self, tokens):
        ret = []

        for token, in tokens:
            cell = []
            if "word" in self.features:
                cell.append(self.tok2id.get(normalize(token), self.tok2id[UNK]))
            if "casing" in self.features:
                cell.append(self.tok2id[P_CASE + casing(token)])
            ret.append(cell)
        return np.array(ret, dtype=np.int64)

    def vectorize(self, data):
        """
        Returns vectorized sequences of the training data:
        Each returned element consists of an input, node and edge label
        sequences (each is a numpy array).
        """
        raise NotImplementedError()

    @classmethod
    def build(cls, data, features=None):
        """
        Use @data to construct a featurizer.
        """
        raise NotImplementedError()

    def save(self, f):
        raise NotImplementedError()

    @classmethod
    def load(cls, f):
        return cls(*pickle.load(f))

    def _load_embeddings(self, vocab_file, vectors_file):
        wvecs = load_word_vector_mapping(vocab_file, vectors_file)
        embed_size = len(next(iter(wvecs.values())))

        embeddings = np.array(np.random.randn(len(self.tok2id) + 1, embed_size), dtype=np.float32)
        embeddings[0,:] = 0. # (padding) zeros vector.
        for word, vec in wvecs.items():
            word = normalize(word)
            if word in self.tok2id:
                embeddings[self.tok2id[word]] = vec
        logger.info("Initialized embeddings.")

        return embeddings

    def add_embeddings(self, vocab_file, vectors_file):
        self.embeddings = self._load_embeddings(vocab_file, vectors_file)

class Acceptability(Helper):
    """
    Helper for acceptability
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    FEATURES = ["word", "casing",]

    def vectorize(self, data):
        """
        Returns vectorized sequences of the training data:
        Each returned element consists of an input, node and edge label
        sequences (each is a numpy array).
        """
        ret = []
        for datum in tqdm(data, desc="vectorizing data"):
            x = self.vectorize_example(datum['x'])
            y = datum['y*']
            ret.append([x,y])
        return ret

    @classmethod
    def build(cls, data, features=None):
        """
        Use @data to construct a featurizer.
        """
        if not features:
            features = cls.FEATURES
        else:
            assert all(f in cls.FEATURES for f in features)

        # Preprocess data to construct an embedding
        # Reserve 0 for the special NIL token.
        tok2id = {}
        if "word" in features:
            tok2id.update(build_dict((normalize(word) for datum in data for word in datum['x']), offset=len(tok2id)+1, max_words=10000))
        if "casing" in features:
            tok2id.update(build_dict([P_CASE + c for c in CASES], offset=len(tok2id)+1))
        tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id)+1))
        logger.info("Built dictionary for %d features.", len(tok2id))

        return cls(tok2id, features)

    def save(self, f):
        pickle.dump([self.tok2id, self.features], f)
