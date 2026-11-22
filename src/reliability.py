"""Per-task reliability from per-question scores, and disattenuated correlations.

Reliability of a task's model-level scores inside a slice is estimated as the
share of observed between-model variance that is not sampling error:

    rel_j = (Var_m[ mean_j(m) ] - E_m[ Var_q(x_mjq) / n_mj ]) / Var_m[ mean_j(m) ]

Disattenuation then divides each pair correlation by sqrt(rel_j * rel_l)
(Spearman's correction for attenuation). Values are capped at 1.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def per_model_task_stats(raw, model_col="model", task_col="task",
                         qid_col="question_id", score_col="score"):
    # Some (model, task, question) pairs appear more than once in the export
    # (e.g. every LCB_generation question is judged twice), so collapse to one
    # score per distinct question FIRST. Using raw row counts as n would inflate
    # the effective item count and overstate reliability.
    one = (raw.groupby([raw[model_col].str.lower().rename("model_key"),
                        task_col, qid_col])[score_col].mean().reset_index())
    g = one.groupby(["model_key", task_col])[score_col]
    df = g.agg(mean="mean", var=lambda s: s.var(ddof=1), n="size").reset_index()
    df.columns = ["model_key", "task", "mean", "var", "n"]
    df["se2"] = df["var"] / df["n"]
    return df


def task_reliability(stats, tasks, slice_keys):
    out = {}
    for t in tasks:
        d = stats[(stats.task == t) & (stats.model_key.isin(slice_keys))]
        if len(d) < 5:
            out[t] = np.nan; continue
        v_obs = d["mean"].var(ddof=1)
        v_err = d["se2"].mean()
        out[t] = float(np.clip((v_obs - v_err) / v_obs, 0.0, 1.0)) if v_obs > 0 else np.nan
    return out


def mean_offdiag(M, method="pearson"):
    C = M.corr(method=method)
    k = C.shape[0]
    iu = np.triu_indices(k, 1)
    return float(np.nanmean(C.values[iu])), C


def disattenuated_mean(M, rel, method="pearson"):
    _, C = mean_offdiag(M, method)
    cols = list(M.columns)
    vals = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = C.values[i, j]
            d = np.sqrt(rel[cols[i]] * rel[cols[j]])
            if np.isfinite(r) and np.isfinite(d) and d > 0:
                vals.append(min(r / d, 1.0))
    return float(np.nanmean(vals)) if vals else np.nan
