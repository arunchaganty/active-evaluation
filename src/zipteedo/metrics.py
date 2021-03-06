"""

"""
import sys
import pyter
from nltk.tokenize import word_tokenize
from nltk.translate import bleu as _bleu
from nltk.translate.bleu_score import SmoothingFunction
from pythonrouge.pythonrouge import Pythonrouge


from .meteor import Meteor
from .sent2vec import Sent2Vec

chencherry = SmoothingFunction()
_meteor = None
def _m():
    global _meteor
    if _meteor is None:
        _meteor = Meteor()
    return _meteor

_sent2vec = None
def _s():
    global _sent2vec
    if _sent2vec is None:
        _sent2vec = Sent2Vec()
    return _sent2vec

def rouge(hyp, ref, n=None):
    # 1 - 4, L
    hyp, ref = " ".join(hyp), " ".join(ref)
    ret = Pythonrouge(
        summary_file_exist=False,
        summary=[[hyp]], reference=[[[ref]]],
        n_gram=4, ROUGE_SU4=True, ROUGE_L=True,
        recall_only=True, stemming=True, stopwords=True,
        word_level=True, length_limit=True, length=50,
        use_cf=False, cf=95, scoring_formula='average',
        resampling=True, samples=1000, favor=True, p=0.5).calc_score()

    if n is None: return ret
    else: return ret["rouge-"+n]["f"]

def bleu(hyp, refs, n=2, smooth=True):
    # 1 - 4
    # nltk bleu
    smoothing_function = chencherry.method3 if smooth else chencherry.method0
    return _bleu(refs, hyp, [1./n for _ in range(n)], smoothing_function=smoothing_function)

def ter(hyp, ref):
    return pyter.ter(hyp, ref)

def meteor(hyp, refs):
    # meteor package.
    return _m().score(hyp, refs)

def sim(hyp, ref):
    return _s().score(hyp, ref)

def test_metrics():
    hyps = ["Barack Obama will be the fourth president to receive the Nobel Peace Prize", "US President Barack Obama will fly to Oslo in Norway, for 26 hours and be the fourth US President in history to receive the Nobel Peace Prize."]
    refs = ["Barack Obama becomes the fourth American president to receive the Nobel Peace Prize", "The American president Barack Obama will fly into Oslo, Norway for 26 hours to receive the Nobel Peace Prize, the fourth American president in history to do so."]
    for hyp, ref in zip(hyps, refs):
        hyp_, ref_ = word_tokenize(hyp), word_tokenize(ref)
        for n in range(1,5):
            print("bleu{}: {:.3f}".format(n, bleu(hyp_, [ref_], n, smooth=False)))
            print("bleu{}': {:.3f}".format(n, bleu(hyp_, [ref_], n, smooth=False)))
        print("rouge: {:.3f}".format(rouge(hyp, ref, "l")))
        print("meteor: {:.3f}".format(meteor(hyp, [ref])))
        print("ter: {:.3f}".format(ter(hyp_, ref_)))
        #print("sim: {:.3f}".format(sim(hyp, ref)))
