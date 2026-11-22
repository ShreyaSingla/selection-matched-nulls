"""
Hardened inferential core for the benchmark-relevance paper (step 1).

Supersedes prototype_inference.py. Changes vs the prototype:
  - 1-factor loadings fitted by principal-axis factoring on the PAIRWISE-COMPLETE
    correlation matrix, so all models contribute, not just complete cases.
  - null residuals are RESAMPLED from the empirical residual distribution
    (column-wise, independently) instead of drawn Gaussian, so bounded support
    and heteroscedasticity are preserved under H0.
  - the observed missingness pattern is resampled along with the models.
  - works on any model x benchmark panel, so the three panels are directly comparable.

H0: a single latent ability + benchmark-specific independent residuals.
Test statistic: mean off-diagonal Spearman correlation inside the top-`cut` slice,
where the slice is selected on the simulated average exactly as in the real panel.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _spearman_offdiag_mean(X):
    """Mean off-diagonal Spearman correlation, NaN-aware (pairwise complete)."""
    k = X.shape[1]
    vals = []
    for a in range(k):
        for b in range(a + 1, k):
            m = ~(np.isnan(X[:, a]) | np.isnan(X[:, b]))
            if m.sum() >= 5:
                ra, rb = rankdata(X[m, a]), rankdata(X[m, b])
                sa, sb = ra.std(), rb.std()
                if sa > 0 and sb > 0:
                    vals.append(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))
    return float(np.mean(vals)) if vals else np.nan


def pairwise_corr(Z):
    k = Z.shape[1]
    R = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            m = ~(np.isnan(Z[:, a]) | np.isnan(Z[:, b]))
            R[a, b] = R[b, a] = np.corrcoef(Z[m, a], Z[m, b])[0, 1] if m.sum() >= 5 else 0.0
    return R


def paf_one_factor(R, n_iter=100):
    """Principal-axis factoring, one factor. Returns loadings (sign-fixed positive)."""
    k = R.shape[0]
    h2 = np.array([np.max(np.abs(R[i][np.arange(k) != i])) for i in range(k)])
    lam = np.sqrt(h2)
    for _ in range(n_iter):
        Rr = R.copy()
        np.fill_diagonal(Rr, h2)
        w, v = np.linalg.eigh(Rr)
        lam = v[:, -1] * np.sqrt(max(w[-1], 1e-9))
        if lam.sum() < 0:
            lam = -lam
        h2 = np.clip(lam ** 2, 1e-6, 0.999)
    return np.clip(np.abs(lam), 0, 0.999)


def fit_null_model(M):
    """M: DataFrame models x benchmarks (NaN allowed). Returns loadings, empirical
    residual pools per benchmark, and the observed missingness mask."""
    Z = ((M - M.mean()) / M.std()).values
    R = pairwise_corr(Z)
    lam = paf_one_factor(R)
    # regression (Thurstone) factor scores on standardized data, NaN-aware
    theta = np.array([np.nansum(lam * row) / max(np.nansum(~np.isnan(row) * lam ** 2), 1e-9)
                      for row in Z])
    theta = (theta - np.nanmean(theta)) / np.nanstd(theta)
    resid = Z - lam[None, :] * theta[:, None]
    pools = [resid[~np.isnan(resid[:, j]), j] for j in range(Z.shape[1])]
    return lam, pools, ~np.isnan(Z)


def null_band(M, cut, B=1000, seed=0):
    """Null distribution of the within-slice mean off-diagonal Spearman correlation."""
    lam, pools, mask = fit_null_model(M)
    n, k = M.shape
    rng = np.random.default_rng(seed)
    full, top = np.empty(B), np.empty(B)
    for b in range(B):
        th = rng.normal(size=n)
        X = lam[None, :] * th[:, None]
        for j in range(k):
            X[:, j] += rng.choice(pools[j], size=n, replace=True)
        mrows = mask[rng.integers(0, n, size=n)]          # resample observed missingness
        X[~mrows] = np.nan
        avg = np.nanmean(X, axis=1)
        sel = np.argsort(-np.where(np.isnan(avg), -np.inf, avg))[:cut]
        full[b] = _spearman_offdiag_mean(X)
        top[b] = _spearman_offdiag_mean(X[sel])
    return lam, full, top


def evaluate_panel(M, cut, B=1000, seed=0):
    """Observed vs null for one panel. Returns a one-row summary dict."""
    Xo = M.values
    avg = np.nanmean(Xo, axis=1)
    sel = np.argsort(-np.where(np.isnan(avg), -np.inf, avg))[:cut]
    obs_full = _spearman_offdiag_mean(Xo)
    obs_top = _spearman_offdiag_mean(Xo[sel])
    lam, nf, nt = null_band(M, cut, B=B, seed=seed)
    return dict(n_models=M.shape[0], k_bench=M.shape[1], cut=cut,
                cut_frac=cut / M.shape[0], obs_full=obs_full, obs_top=obs_top,
                null_full=nf.mean(), null_top=nt.mean(),
                null_lo=np.percentile(nt, 2.5), null_hi=np.percentile(nt, 97.5),
                p_one_sided=float((nt <= obs_top).mean()),
                mean_loading=float(lam.mean()), min_loading=float(lam.min()))
