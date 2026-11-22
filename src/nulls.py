"""Audit of the one-factor DGP: goodness-of-fit + two alternative nulls.

N1  one-factor + independent resampled residuals      (the paper's current null)
N2  Gaussian copula on the OBSERVED correlation matrix (no factor structure assumed)
N3  two-factor + independent resampled residuals       (Ruan et al. sensitivity)
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from inference import _spearman_offdiag_mean, pairwise_corr, paf_one_factor


def paf_factors(R, m=1, n_iter=200):
    """Principal-axis factoring with m factors. Returns Lambda (k x m)."""
    k = R.shape[0]
    h2 = np.array([np.max(np.abs(R[i][np.arange(k) != i])) for i in range(k)])
    L = np.zeros((k, m))
    for _ in range(n_iter):
        Rr = R.copy(); np.fill_diagonal(Rr, h2)
        w, v = np.linalg.eigh(Rr)
        idx = np.argsort(w)[::-1][:m]
        L = v[:, idx] * np.sqrt(np.clip(w[idx], 1e-9, None))
        h2 = np.clip((L ** 2).sum(1), 1e-6, 0.999)
    return L


def fit_factor_null(M, m=1):
    Z = ((M - M.mean()) / M.std()).values
    R = pairwise_corr(Z)
    L = paf_factors(R, m)
    Zf = np.where(np.isnan(Z), 0.0, Z)
    obs = (~np.isnan(Z)).astype(float)
    theta = np.zeros((Z.shape[0], m))
    for i in range(Z.shape[0]):
        Li = L * obs[i][:, None]
        A = Li.T @ Li + 1e-6 * np.eye(m)
        theta[i] = np.linalg.solve(A, Li.T @ Zf[i])
    theta = (theta - theta.mean(0)) / theta.std(0)
    resid = Z - theta @ L.T
    pools = [resid[~np.isnan(resid[:, j]), j] for j in range(Z.shape[1])]
    return L, pools, ~np.isnan(Z), R


def gof_one_factor(M, n_par=200, seed=0):
    """Goodness of fit of the 1-factor model on the FULL panel (no selection)."""
    Z = ((M - M.mean()) / M.std()).values
    R = pairwise_corr(Z)
    lam = paf_one_factor(R)
    k = R.shape[0]
    resid = R - np.outer(lam, lam)
    off = ~np.eye(k, dtype=bool)
    rmsr = float(np.sqrt((resid[off] ** 2).mean()))
    # parallel analysis on the reduced matrix: is a 2nd factor warranted?
    Rr = R.copy(); np.fill_diagonal(Rr, np.clip(lam ** 2, 1e-6, .999))
    ev = np.sort(np.linalg.eigvalsh(Rr))[::-1]
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    sim2 = []
    for _ in range(n_par):
        Y = rng.normal(size=(n, k))
        Rs = np.corrcoef(Y, rowvar=False)
        l2 = paf_one_factor(Rs)
        np.fill_diagonal(Rs, np.clip(l2 ** 2, 1e-6, .999))
        sim2.append(np.sort(np.linalg.eigvalsh(Rs))[::-1][1])
    return dict(rmsr=rmsr, max_abs_resid=float(np.abs(resid[off]).max()),
                ev1=float(ev[0]), ev2=float(ev[1]),
                ev2_null_p95=float(np.percentile(sim2, 95)),
                second_factor_warranted=bool(ev[1] > np.percentile(sim2, 95)),
                mean_loading=float(lam.mean()), min_loading=float(lam.min()))


def _stat_from_sim(X, cut):
    avg = np.nanmean(X, axis=1)
    sel = np.argsort(-np.where(np.isnan(avg), -np.inf, avg))[:cut]
    return _spearman_offdiag_mean(X), _spearman_offdiag_mean(X[sel])


def null_factor(M, cut, m=1, B=400, seed=0):
    L, pools, mask, _ = fit_factor_null(M, m)
    n, k = M.shape
    rng = np.random.default_rng(seed)
    full, top = np.empty(B), np.empty(B)
    for b in range(B):
        th = rng.normal(size=(n, m))
        X = th @ L.T
        for j in range(k):
            X[:, j] += rng.choice(pools[j], size=n, replace=True)
        X[~mask[rng.integers(0, n, size=n)]] = np.nan
        full[b], top[b] = _stat_from_sim(X, cut)
    return full, top


def null_copula(M, cut, B=400, seed=0):
    """Preserves the OBSERVED cross-benchmark dependence; assumes no factor model.
    Tests: is the within-slice drop more than selection applied to the real structure?"""
    V = M.values.astype(float)
    n, k = V.shape
    Y = np.full((n, k), np.nan)
    for j in range(k):
        m = ~np.isnan(V[:, j])
        Y[m, j] = norm.ppf(rankdata(V[m, j]) / (m.sum() + 1))
    R = pairwise_corr(Y)
    w, v = np.linalg.eigh(R)                      # nearest PSD
    R = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(R)); R = R / np.outer(d, d)
    C = np.linalg.cholesky(R)
    mask = ~np.isnan(V)
    rng = np.random.default_rng(seed)
    full, top = np.empty(B), np.empty(B)
    for b in range(B):
        X = rng.normal(size=(n, k)) @ C.T
        X[~mask[rng.integers(0, n, size=n)]] = np.nan
        full[b], top[b] = _stat_from_sim(X, cut)
    return full, top


def observed(M, cut):
    X = M.values.astype(float)
    avg = np.nanmean(X, axis=1)
    sel = np.argsort(-np.where(np.isnan(avg), -np.inf, avg))[:cut]
    return _spearman_offdiag_mean(X), _spearman_offdiag_mean(X[sel])


def null_copula_cal(M, cut, B=400, seed=0):
    """Gaussian copula calibrated to reproduce the OBSERVED Spearman matrix:
    for a Gaussian copula rho_S = (6/pi) asin(rho_P/2)  =>  rho_P = 2 sin(pi*rho_S/6).
    Assumes no factor structure; preserves the panel's real dependence."""
    V = M.values.astype(float); n, k = V.shape
    Rs = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            m = ~(np.isnan(V[:, a]) | np.isnan(V[:, b]))
            if m.sum() >= 5:
                Rs[a, b] = Rs[b, a] = np.corrcoef(rankdata(V[m, a]), rankdata(V[m, b]))[0, 1]
    Rp = 2 * np.sin(np.pi * Rs / 6); np.fill_diagonal(Rp, 1.0)
    w, v = np.linalg.eigh(Rp); Rp = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(Rp)); Rp = Rp / np.outer(d, d)
    C = np.linalg.cholesky(Rp); mask = ~np.isnan(V)
    rng = np.random.default_rng(seed); full, top = np.empty(B), np.empty(B)
    for b in range(B):
        X = rng.normal(size=(n, k)) @ C.T
        X[~mask[rng.integers(0, n, size=n)]] = np.nan
        full[b], top[b] = _stat_from_sim(X, cut)
    return full, top
