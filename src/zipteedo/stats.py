"""
Useful statistics functions
"""

from collections import defaultdict

from tqdm import tqdm
import numpy as np
import scipy.stats as scstats

from .util import first

def _get_variance_stats(data):
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

def _get_ci(data, n=1000):
    # Get CI by bootstrapping...
    N = len(data)
    stats = []
    for _ in range(n):
        stats.append(data[np.random.randint(0, N, (N,))].mean())
    stats = data.mean() - np.array(stats)
    return data.mean() + np.percentile(stats, 90), data.mean() + np.percentile(stats, 10)

def _get_de(baseline, other):
    baseline_len = (baseline.T[2] - baseline.T[1])
    other_len = (other.T[2] - other.T[1])
    N = len(baseline_len)
    a, b = int(0.1 * N), int(1 * N)

    #ret = np.mean(baseline_len[a:b]**2 / other_len[a:b]**2)
    ret = np.mean(baseline_len[a:b] / other_len[a:b])**2

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
        ret[metric][prompt][system] = _get_de(simple, model_variate)
    return ret

def group_by_system(data, prompt, metric):
    by_system = defaultdict(dict)
    for datum in data:
        id_ = datum["id"]
        system = datum['system']

        y, y_ = datum["prompts"][prompt]["gold"], datum["prompts"][prompt][metric]
        assert y is not None and y_ is not None

        by_system[system][id_] = (y, y_)
    return by_system

def simulate_bias(data, mu, system, condition="ll"):
    systems = sorted(data)
    ret = {}

    for id_, (y, y_) in data[system].items():
        candidates = []
        for system_ in systems:
            if id_ not in data[system_]: continue
            z, z_ = data[system_][id_]
            # Take bad low-ROUGE examples and make them good
            if condition == "lr" and y_ < mu[1] and z_ < mu[1] and z > y:
                candidates.append((z, z_))
            # Take good low-ROUGE examples and make them bad
            elif condition == "ll" and y_ < mu[1] and z_ < mu[1] and z < y:
                candidates.append((z, z_))
            # Take low-ROUGE examples and make them high-ROUGE
            if condition == "ur" and y_ < mu[1] and z_ > mu[1]:
                candidates.append((z, z_))
            # Take low-ROUGE examples and make them high-ROUGE
            elif condition == "ul" and y_ < mu[1] and z_ > mu[1] and z_ < y_:
                candidates.append((z, z_))

        if condition.endswith("r"):
            ret[id_] = min(candidates) if candidates else (y, y_)
        elif condition.endswith("l"):
            ret[id_] = max(candidates) if candidates else (y, y_)

    return ret

def make_bias_table(data, prompt, metric, conditions=None):
    by_system = group_by_system(data, prompt, metric)
    if "reference" in by_system:
        del by_system["reference"]
    if conditions is None:
        conditions = ["ll", "lr", "ul", "ur"]

    overall = np.array([[y,y_] for vs in by_system.values() for y, y_ in vs.values()])
    mu = overall.mean(0)

    def _stats(vs):
        xy = np.array(list(vs.values()))
        x, (x_l, x_u) = xy.T[0].mean(), _get_ci(xy.T[0])
        y, (y_l, y_u) = xy.T[1].mean(), _get_ci(xy.T[1])
        return (x, x_l, x_u), (y, y_l, y_u)

    ret = defaultdict(dict)
    for system, vs in by_system.items():
        ret[system]["default"] = _stats(vs)
        for condition in conditions:
            ret[system][condition] = _stats(simulate_bias(by_system, mu, system, condition))

    return ret
