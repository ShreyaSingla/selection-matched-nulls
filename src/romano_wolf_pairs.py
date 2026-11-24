#!/usr/bin/env python
"""Per-pair test with Romano-Wolf stepdown control on the LiveBench top-50 slice (Sec. VI-B).

For each of the 21 task pairs, the observed within-slice Spearman correlation is compared with
its distribution under the one-factor selection-matched null (the same null as Table II: one
latent ability, independent resampled residuals, the observed missingness pattern, and the
same top-50 rule applied to every simulated leaderboard). Multiplicity across the 21 pairs is
controlled with the Romano-Wolf stepdown on the centred statistics: pairs are visited from the
most negative to the least negative departure, and each adjusted p is the share of replicates
in which the minimum centred statistic over the still-active pairs is at least as extreme,
made monotone across the stepdown.

    python src/romano_wolf_pairs.py            # B=5000, (1 + #)/(B + 1) p-values, as in the paper
    python src/romano_wolf_pairs.py --legacy   # B=2000, seed 11, plain proportion: the exploratory
                                               # run that first produced this table (for audit)

Writes results/romano_wolf_pairs_livebench_top50.csv (or ..._legacy.csv).
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RESULTS = os.path.join(os.path.dirname(HERE), "results")
import panels as P
from inference import fit_null_model

TASKS = P.LIVEBENCH_TASKS
CUT = P.CUTS["LiveBench"]
PAIRS = [(a, b) for i, a in enumerate(TASKS) for b in TASKS[i + 1:]]


def pair_rhos(X):
    """Spearman correlation for every task pair, pairwise-complete (at least 5 models)."""
    out = []
    for a, b in PAIRS:
        ia, ib = TASKS.index(a), TASKS.index(b)
        m = ~(np.isnan(X[:, ia]) | np.isnan(X[:, ib]))
        if m.sum() >= 5:
            ra, rb = rankdata(X[m, ia]), rankdata(X[m, ib])
            s = ra.std() * rb.std()
            out.append(((ra - ra.mean()) * (rb - rb.mean())).mean() / s if s > 0 else np.nan)
        else:
            out.append(np.nan)
    return np.array(out)


def top_slice(X, cut):
    avg = np.nanmean(X, axis=1)
    return X[np.argsort(-np.where(np.isnan(avg), -np.inf, avg))[:cut]]


def main(B=5000, seed=0, legacy=False):
    lb = P.load_livebench().sort_values("rank")
    M = lb.set_index("model")[TASKS]
    lam, pools, mask = fit_null_model(M)
    n, k = M.shape
    rng = np.random.default_rng(seed)
    null = np.empty((B, len(PAIRS)))
    for b in range(B):
        X = lam[None, :] * rng.normal(size=n)[:, None]
        for j in range(k):
            X[:, j] += rng.choice(pools[j], size=n, replace=True)
        X[~mask[rng.integers(0, n, size=n)]] = np.nan
        null[b] = pair_rhos(top_slice(X, CUT))

    obs = pair_rhos(top_slice(M.values.astype(float), CUT))
    med = np.nanmedian(null, axis=0)
    cent, stat = null - med, obs - med

    # Romano-Wolf stepdown, most negative departure first
    active, adj, prev = list(np.argsort(stat)), {}, 0.0
    while active:
        with warnings.catch_warnings():          # replicates where every active pair is
            warnings.simplefilter("ignore", RuntimeWarning)   # undefined give NaN, counted as no hit
            mn = np.nanmin(cent[:, active], axis=1)
        j = active[0]
        hits = np.nansum(mn <= stat[j])
        p = float(np.nanmean(mn <= stat[j])) if legacy else float((1 + hits) / (B + 1))
        adj[j] = prev = max(p, prev)
        active.pop(0)

    out = pd.DataFrame({"pair": [f"{a} / {b}" for a, b in PAIRS], "obs_rho_top50": obs,
                        "null_median": med,
                        "p_romano_wolf": [adj[i] for i in range(len(PAIRS))]}
                       ).sort_values("p_romano_wolf", kind="stable")
    os.makedirs(RESULTS, exist_ok=True)
    fn = "romano_wolf_pairs_livebench_top50" + ("_legacy" if legacy else "") + ".csv"
    out.to_csv(os.path.join(RESULTS, fn), index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    main(B=2000, seed=11, legacy=True) if a.legacy else main()
