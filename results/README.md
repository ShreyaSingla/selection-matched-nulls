# Generated results

Produced by `python src/run_all.py` (and the standalone scripts in `src/`). Every file here
is overwritten on each run.

| file | what it is |
|---|---|
| `table1_main_results.csv` | observed vs null at the headline cut, under all four null specifications |
| `cut_depth_sweep_copula.csv`, `cut_depth_sweep_copula_B5000.csv` | observed vs copula null across the cut grid |
| `one_factor_goodness_of_fit.csv` | RMSR and parallel analysis per panel |
| `table2_leave_one_out.csv`, `table2ab_leave_one_out_B5000.csv` | leave-one-benchmark-out re-test |
| `table2c_row_dependence_B5000.csv` | sensitivity to row (organisation) dependence |
| `tcopula_check_B5000.csv` | t-copula tail-dependence robustness check |
| `share_intervals.csv` | Monte-Carlo intervals on the share of the collapse explained |
| `livebench_rank_intervals.csv` | per-model 95% rank interval and P(top 10) |
| `livebench_task_reliability.csv` | per-task reliability, full panel and top-50 |
| `rank_and_reliability_summary.json` | scalar rank-stability and reliability summaries |
