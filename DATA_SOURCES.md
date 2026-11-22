# Where the benchmark data comes from

Three leaderboards. All three rank models by an **unweighted mean of component benchmark
scores** — that is the paper's scope. Every file in `data/` is a frozen copy; two of the
three upstream sources cannot be re-fetched identically, which is why the copies are
included rather than a download script.

---

## 1. LiveBench judged export — `livebench_panel_clean.parquet` (152 x 7)

| | |
|---|---|
| Upstream | `livebench/model_judgment` on Hugging Face |
| Exact URL | `https://huggingface.co/datasets/livebench/model_judgment/resolve/main/data/leaderboard-00000-of-00001.parquet` |
| Raw copy | `data/livebench_model_judgment_raw.parquet` (60,372 rows x 7 cols) |
| Re-fetchable? | Yes, but frozen upstream: dataset `lastModified` = 2025-04-07 |
| Judged window | 2024-06-07 to 2025-04-07 |

One raw row is one (model, question) judged score — an LLM judge scores each answer, so
these are judged scores, not exact-match grades. This is the **only** panel with
per-question granularity, which is why the reliability and rank-interval analyses run on it
alone.

**Coverage, and why the panel must not be called "LiveBench".** The export covers 3 of
LiveBench's 6 categories. `math`, `reasoning` and `data_analysis` are absent entirely.

| category | task | distinct questions | models judged |
|---|---|---|---|
| coding | `LCB_generation` | 78 | 152 |
| coding | `coding_completion` | 50 | 152 |
| language | `connections` | 100 | 152 |
| language | `plot_unscrambling` | 90 | 152 |
| language | `typos` | 100 | 152 |
| instruction_following | `paraphrase` | 50 | 92 |
| instruction_following | `story_generation` | 26 | 97 |

**Cleaning applied upstream of `livebench_panel_clean`:** case-variant duplicate model names
were merged by lower-casing and keeping the row with the most task coverage (ties broken by
higher mean), then the panel was filtered to models covering >= 6 of 7 tasks — 152 of 195
models kept. The pre-filter pivot is kept as `data/livebench_task_pivot_full_195.csv` for
reference only.

**Ranking rule:** `avg_score` = unweighted mean of the (up to 7) per-task scores, each in
[0,1]; models ranked descending.

**Two traps when rebuilding from the raw export** — both are handled in `src/` and both
change published numbers if you get them wrong. See README §"Two implementation details".

---

## 2. Open LLM Leaderboard v2 — `ollm_panel.parquet` (4,575 x 6)

| | |
|---|---|
| Upstream | `open-llm-leaderboard/contents` on Hugging Face |
| Exact URL | `https://huggingface.co/datasets/open-llm-leaderboard/contents/resolve/main/data/train-00000-of-00001.parquet` |
| Dataset `lastModified` | **2025-03-20T12:17:27Z** |
| Raw copy | `data/ollm_contents.parquet` (4,576 rows x 36 cols, sha256 `f13a7061e63133bc…`) |
| Re-fetchable? | **Yes, and verified.** Re-downloaded 2026-10-14: byte-identical to the archived copy. |
| Benchmarks | IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO |
| Granularity | aggregate per-benchmark scores only — no per-question data |

**Provenance verified 2026-10-14 (this closes an earlier gap).** The pull URL and date had
not been recorded when the panel was first built. Re-downloading the live file and comparing
checksums shows the archived copy is byte-identical to what the dataset serves today, and
the dataset's final revision is dated 2025-03-20 — the leaderboard has not been updated
since. The panel is therefore the leaderboard's **terminal state**, not a snapshot of a
moving target, and there is no newer version to re-run on.

**Derivation:** of the 4,576 entries, the single row flagged by the maintainers
(`Flagged == True`) is dropped, leaving **4,575**; all of them have complete scores on all
six benchmarks.

**Row dependence:** many entries are fine-tunes and merges of a few base models, spread
across 715 organisations, so rows are not independent draws. The paper's dependence
sensitivity re-runs the test under (a) all 4,575 rows, (b) greedy near-duplicate removal in
6-benchmark score space at eps=2 (3,239 rows kept), and (c) one model per organisation
(715 rows).

**Ranking rule:** unweighted mean of the 6 benchmark scores (`avg6`).

---

## 3. OpenCompass CompassAcademic — `compassacademic_panel_clean.csv` (48 x 6)

| | |
|---|---|
| Upstream | the real-time JSON feed behind `opencompass/CompassAcademic-Leaderboard-Full-Version` |
| Exact URL pattern | `http://opencompass.oss-cn-shanghai.aliyuncs.com/assets/research-rank/research-data.REALTIME.<YYYYMMDD>.json` |
| Pull used here | `20260309` — the only date that returned 200 when probed weekly back 400 days |
| Raw copy | `data/compassacademic_raw_source.json` |
| Re-fetchable? | **No.** The feed is a single pinned file re-dated on each OpenCompass update; the 2026-09 and 2026-10 dates 404. This copy is the only durable record of the exact pull. |
| Benchmarks | HLE, AIME2025, MMLU-Pro, LiveCodeBenchV6, GPQA-Diamond, IFEval (each 0-100) |
| Missingness | none — zero missing cells across 48 models x 6 benchmarks |
| Granularity | aggregate only |

Also carries `org`, `num` (parameter count), `time` (release date), `chat_or_base`.

**Ranking rule:** `Average` = unweighted mean of the 6 columns. Confirmed by recomputation,
maximum absolute deviation 0.07 points, consistent with 2-decimal display rounding upstream.

**This panel is already a selected population.** OpenCompass curates frontier systems only,
so the 48 models are themselves a top slice of the global model population. The null's
loadings are therefore fitted to an already-restricted range and further within-panel
selection is mild — there is no unrestricted baseline inside this dataset. It is also the
smallest panel, so the null band is wide. Both facts belong beside its result in the paper,
not in a limitations list.

---

## Missingness and cut depths

| Panel | n | k | missing cells | headline cut | cut fraction |
|---|---|---|---|---|---|
| LiveBench judged | 152 | 7 | present (>= 6/7 coverage filter; `paraphrase` 92/152, `story_generation` 97/152) | 50 | 32.9% |
| Open LLM Leaderboard v2 | 4,575 | 6 | none | 100 | 2.2% |
| CompassAcademic | 48 | 6 | none | 24 | 50.0% |

Missingness is handled by pairwise-complete correlation estimation throughout, and the
observed missingness mask is resampled row-wise onto every simulated panel so the nulls
inherit the same coverage pattern. Note that per-pair slice sizes are therefore unequal:
inside the LiveBench top 50, every `story_generation` pair is computed on 22 models rather
than 50, `paraphrase` pairs on 47, and `paraphrase`/`story_generation` on 19.
