"""t-copula robustness check for Claim 1 (reviewer item: Gaussian copula has no tail
dependence, which is a strong assumption in precisely the upper tail under test).

Same calibration as nulls.null_copula_cal -- map the observed Spearman matrix to a
correlation parameter via rho_P = 2 sin(pi rho_S / 6) -- but simulate from a multivariate
t with nu degrees of freedom instead of a Gaussian, giving symmetric tail dependence that
rises as nu falls. The c = n calibration is reported alongside so the comparison is only
read where the null still reproduces the observed full-panel statistic.
"""
import json, sys, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "src")
import panels as PA
import nulls as NU
from inference import _spearman_offdiag_mean, pairwise_corr

B, SEED = 5000, 0
NUS = [3, 5, 10]
t0 = time.time()


def _corr_param(V):
    """Gaussian-copula calibration of the observed Spearman matrix (as in null_copula_cal)."""
    n, k = V.shape
    Rs = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            m = ~(np.isnan(V[:, a]) | np.isnan(V[:, b]))
            if m.sum() >= 5:
                Rs[a, b] = Rs[b, a] = np.corrcoef(rankdata(V[m, a]), rankdata(V[m, b]))[0, 1]
    Rp = 2 * np.sin(np.pi * Rs / 6)
    np.fill_diagonal(Rp, 1.0)
    w, v = np.linalg.eigh(Rp)
    Rp = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(Rp))
    return Rp / np.outer(d, d)


def null_tcopula(M, cut, nu, B=B, seed=SEED):
    V = M.values.astype(float)
    n, k = V.shape
    C = np.linalg.cholesky(_corr_param(V))
    mask = ~np.isnan(V)
    rng = np.random.default_rng(seed)
    full, top = np.empty(B), np.empty(B)
    for b in range(B):
        Z = rng.normal(size=(n, k)) @ C.T
        w = rng.chisquare(nu, size=n) / nu
        X = Z / np.sqrt(w)[:, None]                  # multivariate t -> t-copula
        X[~mask[rng.integers(0, n, size=n)]] = np.nan
        avg = np.nanmean(X, axis=1)
        sel = np.argsort(-np.where(np.isnan(avg), -np.inf, avg))[:cut]
        full[b] = _spearman_offdiag_mean(X)
        top[b] = _spearman_offdiag_mean(X[sel])
    return full, top


rows = []
for name, (Mp, cut) in PA.all_panels().items():
    of, ot = NU.observed(Mp, cut)
    gf, gt = NU.null_copula_cal(Mp, cut, B=B, seed=SEED)       # Gaussian, for reference
    rows.append(dict(panel=name, null="gaussian", nu=np.nan, obs_full=round(of, 4),
                     obs_top=round(ot, 4), cal_full=round(float(np.median(gf)), 4),
                     null_med=round(float(np.median(gt)), 4),
                     lo=round(float(np.percentile(gt, 2.5)), 4),
                     hi=round(float(np.percentile(gt, 97.5)), 4),
                     p=round((1 + int((gt <= ot).sum())) / (B + 1), 5),
                     drop_expl=round(100 * (of - float(np.median(gt))) / (of - ot), 1)))
    print(rows[-1], flush=True)
    for nu in NUS:
        tf, tt = null_tcopula(Mp, cut, nu)
        rows.append(dict(panel=name, null="t", nu=nu, obs_full=round(of, 4),
                         obs_top=round(ot, 4), cal_full=round(float(np.median(tf)), 4),
                         null_med=round(float(np.median(tt)), 4),
                         lo=round(float(np.percentile(tt, 2.5)), 4),
                         hi=round(float(np.percentile(tt, 97.5)), 4),
                         p=round((1 + int((tt <= ot).sum())) / (B + 1), 5),
                         drop_expl=round(100 * (of - float(np.median(tt))) / (of - ot), 1)))
        print(f"[{time.time()-t0:6.0f}s]", rows[-1], flush=True)

pd.DataFrame(rows).to_csv("results/tcopula_check_B5000.csv", index=False)
print("done", time.time() - t0, flush=True)
