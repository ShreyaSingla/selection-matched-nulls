"""Monte-Carlo intervals on the 'share of the drop explained' (reviewer item M3b).

The share is (rho_bar(n) - median(null_top)) / (rho_bar(n) - rho_bar(c)). Only the null
median is random here, so we bootstrap the B simulated within-slice statistics and
propagate the resulting spread of the median into the share.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import panels as PA, nulls as NU

B, SEED, NBOOT = 5000, 0, 4000
SPECS = [("copula",       lambda M,c: NU.null_copula_cal(M, c, B=B, seed=SEED)),
         ("one-factor",   lambda M,c: NU.null_factor(M, c, m=1, B=B, seed=SEED)),
         ("two-factor",   lambda M,c: NU.null_factor(M, c, m=2, B=B, seed=SEED)),
         ("three-factor", lambda M,c: NU.null_factor(M, c, m=3, B=B, seed=SEED))]

rng = np.random.default_rng(0)
rows, arrays = [], {}
for name, (M, cut) in PA.all_panels().items():
    of, ot = NU.observed(M, cut)
    for lbl, fn in SPECS:
        _, nt = fn(M, cut)
        arrays[f"{name}|{lbl}"] = nt
        med = float(np.median(nt))
        boot = np.median(rng.choice(nt, size=(NBOOT, len(nt)), replace=True), axis=1)
        shares = 100 * (of - boot) / (of - ot)
        rows.append(dict(panel=name, null=lbl, obs_full=round(of,4), obs_top=round(ot,4),
                         null_med=round(med,4),
                         share=round(100*(of-med)/(of-ot), 1),
                         share_lo=round(float(np.percentile(shares, 2.5)), 1),
                         share_hi=round(float(np.percentile(shares, 97.5)), 1)))
        print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv("results/share_intervals.csv", index=False)
np.savez_compressed("results/null_top_arrays.npz", **arrays)
print("done", flush=True)
