#!/usr/bin/env python
"""Driver: regenerates every number, table and figure in the paper.

    python src/run_all.py            # full run (~40 min on 8 cores, B=5000)
    python src/run_all.py --quick    # B=100 smoke test

Writes to results/. Nothing is cached; every output is recomputed from the
panels in data/.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

def mc_p(null_top, obs_top):
    """Unbiased Monte-Carlo p-value. The plain proportion can return exactly 0,
    which is not a valid p-value and is anti-conservative at small B."""
    return float((1 + (null_top <= obs_top).sum()) / (len(null_top) + 1))


import panels as P
import nulls as NU
import inference as INF
import rank_bootstrap as RB
import reliability as REL

RANK_B = 2000   # fixed: the paper reports the rank bootstrap at B=2,000

NULL_SPECS = [("copula", lambda M, c, B, s: NU.null_copula_cal(M, c, B=B, seed=s)),
              ("one-factor", lambda M, c, B, s: NU.null_factor(M, c, m=1, B=B, seed=s)),
              ("two-factor", lambda M, c, B, s: NU.null_factor(M, c, m=2, B=B, seed=s)),
              ("three-factor", lambda M, c, B, s: NU.null_factor(M, c, m=3, B=B, seed=s))]


def main(B=5000, seed=0):
    os.makedirs(RESULTS, exist_ok=True)
    pan = P.all_panels()

    # --- Table I: headline results under every null specification ----------
    rows = []
    for nm, (M, cut) in pan.items():
        of, ot = NU.observed(M, cut)
        for lbl, fn in NULL_SPECS:
            nf, nt = fn(M, cut, B, seed)
            rows.append(dict(panel=nm, null=lbl, n=M.shape[0], k=M.shape[1], cut=cut,
                             cut_frac=cut / M.shape[0], obs_full=of, obs_top=ot,
                             null_full=float(np.median(nf)), null_top=float(np.median(nt)),
                             lo=float(np.percentile(nt, 2.5)), hi=float(np.percentile(nt, 97.5)),
                             p=mc_p(nt, ot),
                             pct_drop_explained=100 * (of - np.median(nt)) / (of - ot)))
    main_tbl = pd.DataFrame(rows)
    main_tbl.to_csv(f"{RESULTS}/table1_main_results.csv", index=False)

    # --- one-factor goodness of fit ----------------------------------------
    gof = pd.DataFrame({nm: NU.gof_one_factor(M) for nm, (M, _) in pan.items()}).T
    gof.to_csv(f"{RESULTS}/one_factor_goodness_of_fit.csv")

    # --- Fig 1: cut-depth sweep under the copula null -----------------------
    rows = []
    for nm, (M, _) in pan.items():
        for c in P.CUT_GRID[nm]:
            of, ot = NU.observed(M, c)
            nf, nt = NU.null_copula_cal(M, c, B=B, seed=seed)
            rows.append(dict(panel=nm, cut=c, frac=c / M.shape[0], obs=ot, obs_full=of,
                             null_med=float(np.median(nt)),
                             lo=float(np.percentile(nt, 2.5)), hi=float(np.percentile(nt, 97.5)),
                             p=mc_p(nt, ot)))
    pd.DataFrame(rows).to_csv(f"{RESULTS}/cut_depth_sweep_copula.csv", index=False)

    # --- Table II(a,b): leave-one-benchmark-out -----------------------------
    rows = []
    for nm, (M, cut) in pan.items():
        if nm == "CompassAcademic":
            continue
        for drop in list(M.columns):
            sub = M.drop(columns=[drop])
            of, ot = NU.observed(sub, cut)
            _, nt = NU.null_factor(sub, cut, m=1, B=B, seed=seed)
            rows.append(dict(panel=nm, dropped=drop, obs_top=ot, null_top=float(np.median(nt)),
                             p=mc_p(nt, ot)))
    pd.DataFrame(rows).to_csv(f"{RESULTS}/table2_leave_one_out.csv", index=False)

    # --- Fig 2 + Sec V-C: rank intervals and reliability --------------------
    lb = P.load_livebench()
    raw = P.load_livebench_raw()
    raw = raw[raw["model"].isin(set(lb["model"]))]
    ranks, summ = RB.rank_bootstrap(raw, lb["model"].tolist(), B=RANK_B, top_m=10, seed=seed)
    ranks.to_csv(f"{RESULTS}/livebench_rank_intervals.csv", index=False)

    stats = REL.per_model_task_stats(raw)
    avg = lb[P.LIVEBENCH_TASKS].mean(axis=1, skipna=True).values
    top = lb.iloc[np.argsort(-avg)[:P.CUTS["LiveBench"]]]
    rel_top = REL.task_reliability(stats, P.LIVEBENCH_TASKS, set(top["model"].str.lower()))
    rel_all = REL.task_reliability(stats, P.LIVEBENCH_TASKS, set(lb["model"].str.lower()))
    pd.DataFrame({"task": P.LIVEBENCH_TASKS,
                  "reliability_top50": [rel_top[t] for t in P.LIVEBENCH_TASKS],
                  "reliability_full": [rel_all[t] for t in P.LIVEBENCH_TASKS]}
                 ).to_csv(f"{RESULTS}/livebench_task_reliability.csv", index=False)

    summ.update({
        "top50_obs_spearman": REL.mean_offdiag(top[P.LIVEBENCH_TASKS], "spearman")[0],
        "top50_disatt_spearman": REL.disattenuated_mean(top[P.LIVEBENCH_TASKS], rel_top, "spearman"),
        "full_obs_spearman": REL.mean_offdiag(lb[P.LIVEBENCH_TASKS], "spearman")[0],
        "full_disatt_spearman": REL.disattenuated_mean(lb[P.LIVEBENCH_TASKS], rel_all, "spearman"),
    })
    json.dump({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()},
              open(f"{RESULTS}/rank_and_reliability_summary.json", "w"), indent=2)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in summ.items()}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="B=100 smoke test")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(B=100 if a.quick else 5000, seed=a.seed)
