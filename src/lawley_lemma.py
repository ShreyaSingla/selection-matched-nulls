"""Pearson-Lawley closed form for selection on an internal composite (paper Sec. V-C),
plus the numerical check reported in Sec. V-D.

    python src/lawley_lemma.py
"""
import sys
import numpy as np
from scipy.stats import norm, spearmanr

sys.path.insert(0, __file__.rsplit("/", 1)[0])


def lawley_rho(lam, q):
    """Exact post-selection PEARSON correlation among one-factor components after keeping
    the top-q fraction on the unweighted composite. Returns (mean off-diagonal, var ratio)."""
    lam = np.asarray(lam, float)
    k = len(lam)
    Lam, Q = lam.sum(), (1 - lam ** 2).sum()
    s2 = (Lam ** 2 + Q) / k ** 2                   # Var(composite)
    c = (lam * Lam + 1 - lam ** 2) / k             # Cov(z_j, composite)
    if q >= 1.0:
        v = 1.0
    else:
        a = norm.ppf(1 - q); ph = norm.pdf(a)
        v = 1 + a * ph / q - (ph / q) ** 2         # truncated variance ratio
    R = (np.outer(lam, lam) - (1 - v) * np.outer(c, c) / s2)
    var = 1 - (1 - v) * c ** 2 / s2
    R = R / np.sqrt(np.outer(var, var))
    np.fill_diagonal(R, 1.0)
    return float(R[~np.eye(k, dtype=bool)].mean()), float(v)


def check(lam=0.82, k=7, N=400_000, seed=0):
    """Verify the closed form against simulation, and quantify the Spearman gap."""
    rng = np.random.default_rng(seed)
    th = rng.normal(size=N)
    Z = lam * th[:, None] + np.sqrt(1 - lam ** 2) * rng.normal(size=(N, k))
    order = np.argsort(-Z.mean(1))
    off = ~np.eye(k, dtype=bool)
    print(f"{'q':>6} {'lemma_P':>9} {'emp_P':>9} {'arcsin_S':>9} {'emp_S':>9}")
    for q in [0.5, 0.329, 0.20, 0.10, 0.05, 0.022, 0.01]:
        S = Z[order[:int(q * N)]]
        pred, _ = lawley_rho([lam] * k, q)
        emp_p = np.corrcoef(S, rowvar=False)[off].mean()
        emp_s = np.mean([spearmanr(S[:, a], S[:, b]).statistic
                         for a in range(k) for b in range(a + 1, k)])
        print(f"{q:>6.3f} {pred:>9.4f} {emp_p:>9.4f} "
              f"{(6/np.pi)*np.arcsin(pred/2):>9.4f} {emp_s:>9.4f}")
    print(f"\nq->0 limit with equal loadings: -1/(k-1) = {-1/(k-1):.4f}")


if __name__ == "__main__":
    check()
