"""Question-level bootstrap for leaderboard rank uncertainty.

Resamples question identifiers WITH REPLACEMENT, stratified within task, so the
cross-model dependence induced by shared items is preserved: every model in a
replicate is scored on the same resampled question set.

Produces: per-model rank intervals, the stability of first place, the share of
replicates in which each model reaches the top-m, and mean top-m overlap.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_task_matrices(raw: pd.DataFrame, models: list[str],
                        model_col="model", task_col="task",
                        qid_col="question_id", score_col="score"):
    """-> (tasks, [n_models x n_questions] arrays with NaN for unjudged pairs)."""
    # Exact model-string match. The clean panel was built by SELECTING one
    # case-variant per model (most task coverage, ties on higher mean), not by
    # averaging variants together; lower-casing and averaging here would silently
    # change 60 of 152 published ranks. Verified: exact matching reproduces every
    # per-model per-task mean in livebench_panel_clean to 0.0 absolute error.
    key = {m: i for i, m in enumerate(models)}
    r = raw[raw[model_col].isin(key)]
    tasks, mats = [], []
    for t, g in r.groupby(task_col, sort=True):
        piv = g.pivot_table(index=model_col, columns=qid_col, values=score_col, aggfunc="mean")
        M = np.full((len(models), piv.shape[1]), np.nan)
        rows = [key[k] for k in piv.index]
        M[rows, :] = piv.values
        tasks.append(t); mats.append(M)
    return tasks, mats


def _avg_from_mats(mats, idxs):
    """Per-model leaderboard average given one resampled question index set per task."""
    cols = []
    for M, idx in zip(mats, idxs):
        with np.errstate(invalid="ignore"):
            cols.append(np.nanmean(M[:, idx], axis=1))
    T = np.vstack(cols).T                      # n_models x n_tasks
    with np.errstate(invalid="ignore"):
        return np.nanmean(T, axis=1)


def _ranks(avg):
    """1 = best. NaN averages sort last."""
    a = np.where(np.isnan(avg), -np.inf, avg)
    order = np.argsort(-a, kind="stable")
    r = np.empty(len(avg), dtype=int)
    r[order] = np.arange(1, len(avg) + 1)
    return r


def rank_bootstrap(raw, models, B=2000, top_m=10, seed=0):
    tasks, mats = build_task_matrices(raw, models)
    obs_rank = _ranks(_avg_from_mats(mats, [np.arange(M.shape[1]) for M in mats]))
    rng = np.random.default_rng(seed)
    R = np.empty((B, len(models)), dtype=int)
    for b in range(B):
        idxs = [rng.integers(0, M.shape[1], size=M.shape[1]) for M in mats]
        R[b] = _ranks(_avg_from_mats(mats, idxs))

    obs_top = set(np.where(obs_rank <= top_m)[0])
    overlap = np.array([len(obs_top & set(np.where(R[b] <= top_m)[0])) / top_m for b in range(B)])
    first = int(np.argmin(obs_rank))
    out = pd.DataFrame({
        "model": models,
        "obs_rank": obs_rank,
        "rank_lo": np.percentile(R, 2.5, axis=0).round().astype(int),
        "rank_hi": np.percentile(R, 97.5, axis=0).round().astype(int),
        "rank_median": np.median(R, axis=0),
        f"p_top{top_m}": (R <= top_m).mean(axis=0),
    }).sort_values("obs_rank").reset_index(drop=True)
    summary = {
        "B": B,
        "n_models": len(models),
        "n_tasks": len(tasks),
        "rank1_model": models[first],
        "rank1_stability": float((R[:, first] == 1).mean()),
        f"mean_top{top_m}_overlap": float(overlap.mean()),
        f"n_models_reaching_top{top_m}_ge5pct": int((out[f"p_top{top_m}"] >= 0.05).sum()),
        f"n_models_ci_includes_top{top_m}": int((out.rank_lo <= top_m).sum()),
    }
    return out, summary
