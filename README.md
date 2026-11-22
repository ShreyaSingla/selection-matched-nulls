# Selection-Matched Nulls for Benchmark Agreement Testing

Code and data for the paper *Selection-Matched Nulls for Benchmark Agreement Testing*
(Singla & Gupta, BPOD 2026, accepted). The paper itself is not distributed here; a link will
be added once it is published.

**The problem.** Benchmark Agreement Testing (BAT) validates a benchmark by correlating its
model scores against established benchmarks. Computed inside the *top slice* of a leaderboard
that agreement collapses — 0.703 → 0.136 on our LiveBench panel — and the literature reads the
collapse as evidence that a benchmark is invalid or unreliable. But the slice was selected by
ranking models on the **average of the very benchmarks being correlated**, so the statistic is
conditioned on a collider and the full-panel value is not an admissible baseline for it.

**What this package does.** Builds the missing baseline: a *selection-matched null* that
simulates panels whose benchmarks are interchangeable by construction, applies the real
leaderboard's own top-*c* rule, and returns the within-slice agreement honest benchmarks would
have produced. On three leaderboards (48 to 4,575 models) the null accounts for most of the
observed collapse, and on one it accounts for more than all of it.

## Quick start

```bash
pip install -r requirements.txt

python src/run_all.py --quick    # smoke test (B = 100), well under a minute
python src/run_all.py            # full run (B = 5,000), about half an hour on 8 cores
```

`run_all.py` recomputes the tables and the numbers reported in the paper into `results/`.
Nothing is cached; everything is recomputed from `data/`. The remaining scripts run
from the repository root:

```bash
python src/toy_selection.py       # toy demonstration figure; needs no project data
python src/lawley_lemma.py        # Pearson-Lawley closed form and its numerical check
python src/tcopula_check.py       # t-copula tail-dependence robustness check
python src/share_intervals.py     # Monte-Carlo intervals on the "share explained" figures
python src/fig_rank_intervals.py  # rank-interval figure (after run_all.py)
```

## Requirements

Python >= 3.10 with `numpy`, `pandas`, `scipy`, `pyarrow`, `matplotlib`. No GPU, no network.
Peak memory under 1 GB. See `requirements.txt`.

## Layout

```
src/panels.py           panel loaders, benchmark column lists, cut depths and cut grids
src/inference.py        the one-factor null (PAF loadings, resampled empirical residuals)
src/nulls.py            k-factor nulls, calibrated Gaussian-copula null, 1-factor goodness of fit
src/rank_bootstrap.py   question-level bootstrap -> rank intervals and top-m stability
src/reliability.py      per-task reliability from per-question scores; disattenuation
src/lawley_lemma.py     Pearson-Lawley closed form for selection on an internal composite
src/tcopula_check.py    t-copula robustness check
src/share_intervals.py  intervals on the share of the collapse explained by selection
src/toy_selection.py    standalone toy demonstration
src/fig_rank_intervals.py  rank-interval figure
src/run_all.py          driver; regenerates results/
data/                   the three frozen score panels, LiveBench per-question export, CompassAcademic raw JSON
results/                generated; see results/README.md
```

Provenance of every panel — upstream URL, revision date, checksum, derivation rule — is in
[`DATA_SOURCES.md`](DATA_SOURCES.md). All three sources are frozen; there is nothing newer to pull.

## What each module computes

**`nulls.null_copula_cal(M, c)`** — the primary null. Maps each benchmark to normal scores,
inverts the Gaussian-copula relation `rho_P = 2 sin(pi * rho_S / 6)` so the simulated
Spearman matrix matches the observed one, simulates, and applies the panel's own top-`c`
rule. Assumes no factor structure.

**`nulls.null_factor(M, c, m)`** — the `m`-factor null. `m=1` is the null for the
residual-excess test; `m=2,3` are sensitivity specifications.

**`nulls.gof_one_factor(M)`** — RMSR of the residual correlation matrix plus parallel
analysis on the second eigenvalue.

**`rank_bootstrap.rank_bootstrap(raw, models)`** — resamples `question_id` with replacement
stratified within task, so every model in a replicate is scored on the same resampled
question set and the cross-model dependence from shared items is preserved.

**`reliability.task_reliability`** — estimates each task's reliability inside a slice as the
share of between-model variance that is not sampling error, then disattenuates the pairwise
correlations.

Monte-Carlo p-values are computed as `(1 + #{null <= obs}) / (B + 1)`.

## Two implementation details that change the answer

1. **Model matching must be exact.** The clean LiveBench panel was built by *selecting* one
   case-variant per model name (most task coverage, ties on higher mean), not by averaging
   variants. Lower-casing and averaging when rebuilding from the raw export silently moves
   60 of 152 published ranks. `build_task_matrices` matches the exact model string.
2. **Repeated judgments must be collapsed first.** Some (model, task, question) triples
   appear more than once in the export — every `LCB_generation` question is judged twice.
   Using raw row counts as the item count `n` overstates reliability.
   `per_model_task_stats` collapses to one score per distinct question first.

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

Code is released under the MIT License. The score panels in `data/` are redistributed from
their upstream sources under those sources' own terms; see `DATA_SOURCES.md` for each.
