#!/usr/bin/env python
"""Camera-ready robustness checks requested by BPOD Reviewer 2.

(a) PROPAGATED parameter uncertainty. The headline nulls are built at the fitted parameters
    (copula: the panel's rank-correlation matrix; one-factor: loadings lambda-hat and residual
    pools) as though they were known. Here every replicate first resamples the n models with
    replacement, refits the null's parameters on that resample, and only then simulates one
    leaderboard, so the band also carries the estimation error of the fitted null.

(b) ABILITY-STRATIFIED residuals (one-factor null only; the copula null has no residual pool).
    Models are binned by fitted ability theta-hat (deciles; terciles on CompassAcademic, n=48),
    and each simulated model draws its residuals from the real models in its own ability bin,
    so residual spread may shrink near a benchmark ceiling instead of being pooled. The
    'centred' variant subtracts each bin's mean residual first, so that only the SPREAD of the
    residuals varies with ability; the uncentred variant also imports each bin's mean
    residual, which adds shared structure and inflates the c=n calibration.

Writes results/robustness_r2.csv. Baseline rows recompute the paper's Table I nulls with the
same seed as run_all.py, as a check that this script reproduces them.

    python src/robustness_r2.py            # B=5000
    python src/robustness_r2.py --quick    # B=100
    python src/robustness_r2.py --synthetic   # stratification no-op check, synthetic panels
"""
from __future__ import annotations
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RESULTS = os.path.join(os.path.dirname(HERE), "results")
import panels as P
import nulls as NU
from nulls import paf_factors, _stat_from_sim
from inference import pairwise_corr

N_BINS = {"LiveBench": 10, "OLLM v2": 10, "CompassAcademic": 3}


def mc_p(null_top, obs_top):
    return float((1 + (null_top <= obs_top).sum()) / (len(null_top) + 1))


# ---------- parameter fits (vectorised one-factor; same estimator as nulls.fit_factor_null) ----
def fit_one_factor(V):
    Z = (V - np.nanmean(V, 0)) / np.nanstd(V, 0, ddof=1)
    L = paf_factors(pairwise_corr(Z), 1)[:, 0]
    obs = ~np.isnan(Z); Zf = np.where(obs, Z, 0.0)
    theta = (Zf * L).sum(1) / ((obs * L ** 2).sum(1) + 1e-6)
    theta = (theta - theta.mean()) / theta.std()
    resid = Z - theta[:, None] * L[None, :]
    return L, theta, resid, obs


def fit_copula(V):
    n, k = V.shape
    Rs = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            m = ~(np.isnan(V[:, a]) | np.isnan(V[:, b]))
            if m.sum() >= 5:
                Rs[a, b] = Rs[b, a] = np.corrcoef(rankdata(V[m, a]), rankdata(V[m, b]))[0, 1]
    Rp = 2 * np.sin(np.pi * Rs / 6); np.fill_diagonal(Rp, 1.0)
    w, v = np.linalg.eigh(Rp); Rp = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(Rp)); Rp = Rp / np.outer(d, d)
    return np.linalg.cholesky(Rp), ~np.isnan(V)


# ---------- one simulated leaderboard from given parameters ----------------------------------
def sim_copula(C, mask, n, rng):
    X = rng.normal(size=(n, C.shape[0])) @ C.T
    X[~mask[rng.integers(0, mask.shape[0], size=n)]] = np.nan
    return X


def sim_one_factor(L, theta, resid, obs, n, rng, n_bins=None, centre=False):
    k = len(L)
    th = rng.normal(size=n)
    X = th[:, None] * L[None, :]
    if n_bins is None:
        for j in range(k):
            X[:, j] += rng.choice(resid[obs[:, j], j], size=n, replace=True)
    else:
        cuts = np.quantile(theta, np.arange(1, n_bins) / n_bins)
        bin_fit, bin_sim = np.searchsorted(cuts, theta), np.searchsorted(cuts, th)
        for j in range(k):
            allpool = resid[obs[:, j], j]
            for b in range(n_bins):
                rows = bin_sim == b
                if not rows.any():
                    continue
                pool = resid[(bin_fit == b) & obs[:, j], j]
                if centre and pool.size:
                    pool = pool - pool.mean()       # spread varies with ability, mean does not
                X[rows, j] += rng.choice(pool if pool.size >= 3 else allpool,
                                         size=rows.sum(), replace=True)
    X[~obs[rng.integers(0, obs.shape[0], size=n)]] = np.nan
    return X


# ---------- the five variants ----------------------------------------------------------------
def run(task):
    name, variant, B, seed = task
    M, cut = P.all_panels()[name]
    V = M.values.astype(float); n = V.shape[0]
    rng = np.random.default_rng(seed)
    full, top = np.empty(B), np.empty(B)
    t0 = time.time()
    if variant == "copula (baseline)":
        full, top = NU.null_copula_cal(M, cut, B=B, seed=seed)
    elif variant == "one-factor (baseline)":
        full, top = NU.null_factor(M, cut, m=1, B=B, seed=seed)
    elif variant == "one-factor, ability-stratified":
        par = fit_one_factor(V)
        for b in range(B):
            full[b], top[b] = _stat_from_sim(sim_one_factor(*par, n, rng, N_BINS[name]), cut)
    elif variant == "one-factor, ability-stratified, centred":
        par = fit_one_factor(V)
        for b in range(B):
            full[b], top[b] = _stat_from_sim(sim_one_factor(*par, n, rng, N_BINS[name], True), cut)
    elif variant == "copula, propagated":
        for b in range(B):
            Vb = V[rng.integers(0, n, size=n)]
            C, mask = fit_copula(Vb)
            full[b], top[b] = _stat_from_sim(sim_copula(C, mask, n, rng), cut)
    elif variant == "one-factor, propagated":
        for b in range(B):
            Vb = V[rng.integers(0, n, size=n)]
            full[b], top[b] = _stat_from_sim(sim_one_factor(*fit_one_factor(Vb), n, rng), cut)
    else:
        raise ValueError(variant)
    of, ot = NU.observed(M, cut)
    med = float(np.median(top))
    return dict(panel=name, variant=variant, n=n, cut=cut, B=B, obs_full=of, obs_top=ot,
                null_full=float(np.median(full)), null_top=med,
                lo=float(np.percentile(top, 2.5)), hi=float(np.percentile(top, 97.5)),
                band_width=float(np.percentile(top, 97.5) - np.percentile(top, 2.5)),
                p=mc_p(top, ot), pct_drop_explained=100 * (of - med) / (of - ot),
                seconds=round(time.time() - t0, 1))


def synthetic_check(task):
    """Is stratification a no-op when it should be? Gaussian one-factor panels with
    homoscedastic residuals (lambda=0.82): any shift of the stratified null median relative to
    the pooled null is an artefact of the stratification itself."""
    n, k, cut, nb, rep, B = task
    rng = np.random.default_rng(1000 + rep)
    th = rng.normal(size=n)
    V = 0.82 * th[:, None] + np.sqrt(1 - 0.82 ** 2) * rng.normal(size=(n, k))
    par = fit_one_factor(V)
    r = np.random.default_rng(rep)
    med = {}
    for lab, kw in [("pooled", {}), ("stratified_centred", dict(n_bins=nb, centre=True))]:
        med[lab] = float(np.median([_stat_from_sim(sim_one_factor(*par, n, r, **kw), cut)[1]
                                    for _ in range(B)]))
    return dict(n=n, k=k, cut=cut, bins=nb, rep=rep, B=B, **med,
                shift=med["stratified_centred"] - med["pooled"])


SYNTH = [(152, 7, 50, 10), (4575, 6, 100, 10), (48, 6, 24, 3)]   # matched to the three panels

VARIANTS = ["copula (baseline)", "copula, propagated", "one-factor (baseline)",
            "one-factor, propagated", "one-factor, ability-stratified",
            "one-factor, ability-stratified, centred"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--B", type=int)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--synthetic", action="store_true",
                    help="only run the stratification no-op check on synthetic panels")
    a = ap.parse_args()
    if a.synthetic:
        Bs = a.B or 500
        tasks = [(*s, rep, Bs) for s in SYNTH for rep in range(5 if s[0] < 1000 else 3)]
        with Pool(a.procs) as pool:
            out = pd.DataFrame(pool.map(synthetic_check, tasks, chunksize=1))
        os.makedirs(RESULTS, exist_ok=True)
        out.to_csv(os.path.join(RESULTS, "robustness_r2_synthetic.csv"), index=False)
        print(out.round(3).to_string(index=False))
        print(out.groupby("n")["shift"].agg(["mean", "min", "max"]).round(3).to_string())
        sys.exit(0)
    B = a.B or (100 if a.quick else 5000)
    tasks = [(nm, v, B, a.seed) for nm in ["OLLM v2", "LiveBench", "CompassAcademic"] for v in VARIANTS]
    with Pool(a.procs) as pool:
        rows = pool.map(run, tasks, chunksize=1)
    os.makedirs(RESULTS, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "robustness_r2.csv" if B >= 5000 else f"robustness_r2_B{B}.csv"),
               index=False)
    pd.set_option("display.width", 200)
    print(out[["panel", "variant", "null_full", "null_top", "lo", "hi", "band_width", "p",
               "pct_drop_explained", "seconds"]].round(3).to_string(index=False))
