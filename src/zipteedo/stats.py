"""
Useful statistics functions
"""

from collections import defaultdict

from tqdm import tqdm
import numpy as np
import scipy.stats as scstats

from .util import first

def get_variance_stats(data):
    prompts = list(first(data)["prompts"])
    systems = sorted({datum["system"] for datum in data})

    ret = defaultdict(dict)

    for prompt in prompts:
        n_ = 0
        sigma_f_ = []
        sigma_a_ = []
        for system in systems:
            n = 0
            sigma_f = []
            sigma_a = []
            for datum in data:
                if datum['system'] != system: continue
                hs = datum['prompts'][prompt]['human']
                if len(hs) > 1:
                    f = np.mean(hs)
                    sigma_f.extend(hs)
                    sigma_a.extend(hs - f)
                    sigma_f_.extend(hs)
                    sigma_a_.extend(hs - f)
                    n += 1
                    n_ += 1

            sigma2_a = np.var(sigma_a, ddof=n)
            sigma2_f = np.var(sigma_f, ddof=1) - sigma2_a
            nu = sigma2_a/sigma2_f
            ret[system][prompt] = {
                "sigma2_a": sigma2_a ,
                "sigma2_f": sigma2_f ,
                "nu": nu ,
                }

        sigma2_a = np.var(sigma_a_, ddof=n_)
        sigma2_f = np.var(sigma_f_, ddof=1) - sigma2_a
        nu = sigma2_a/sigma2_f

        ret["*"][prompt] = {
            "sigma2_a": sigma2_a ,
            "sigma2_f": sigma2_f ,
            "nu": nu ,
            }
    return ret


def get_correlations(data):
    metrics = list(first(first(data)["prompts"].values()))
    prompts = list(first(data)["prompts"])
    systems = sorted({datum["system"] for datum in data})
    metrics.remove("gold")
    metrics.remove("human")
    print(systems)

    sigma_table = get_variance_stats(data)

    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for datum in data:
        system = datum['system']
        for prompt in prompts:
            for metric in metrics:
                y, y_ = datum["prompts"][prompt]["gold"], datum["prompts"][prompt][metric]
                assert y is not None
                assert y_ is not None
                agg[prompt][metric][system].append([y, y_])
                agg[prompt][metric]["*"].append([y, y_])

    # Finally aggregate.
    ret = defaultdict(lambda: defaultdict(dict))
    for prompt in prompts:
        for metric in metrics:
            for system in systems + ["*"]:
                if system == "reference": continue
                xy = np.array(agg[prompt][metric][system])
                v = np.mean(xy.T[0] * xy.T[1] - np.mean(xy.T[0])*np.mean(xy.T[1]))/(np.sqrt(sigma_table[system][prompt]['sigma2_f']) * np.sqrt(np.var(xy.T[1], ddof=1)))
                #v =  scstats.pearsonr(xy.T[0], xy.T[1])[0]
                ret[prompt][metric][system] = v
    return ret


def get_data_efficiencies(data):
    # Group data appropriately.
    data = {(datum["metric"], datum["prompt"], datum["system"], datum["estimator"]): datum for datum in data}
    metrics = sorted({m for m, p, s, _ in data})
    prompts = sorted({p for m, p, s, _ in data})
    systems = sorted({s for m, p, s, _ in data})

    ret = defaultdict(lambda: defaultdict(dict))
    settings = [(metric, prompt, system) for metric in metrics for prompt in prompts for system in systems]
    for metric, prompt, system in tqdm(settings, desc="settings"):
        simple = np.array(data[metric,prompt,system,"simple"]["summary"])
        model_variate = np.array(data[metric,prompt,system,"model_variate"]["summary"])
        ret[metric][prompt][system] = get_de(simple, model_variate)
    return ret

def get_de(baseline, other):
    baseline_len = (baseline.T[2] - baseline.T[1])
    other_len = (other.T[2] - other.T[1])
    N = len(baseline_len)
    a, b = int(0.1 * N), int(1 * N)

    #ret = np.mean(baseline_len[a:b]**2 / other_len[a:b]**2)
    ret = np.mean(baseline_len[a:b] / other_len[a:b])**2

    return ret
