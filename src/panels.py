"""Panel loaders. Every analysis in the paper starts here.

Each panel is a models x benchmarks DataFrame of raw published scores, plus the
cut depth used in the paper's headline table. Scores are NOT rescaled: the
statistic is rank-based and every null applies the same standardisation
internally, so the 0--1 (LiveBench) vs 0--100 (CompassAcademic) difference is
immaterial.
"""
from __future__ import annotations
import os
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

LIVEBENCH_TASKS = ["LCB_generation", "coding_completion", "connections", "paraphrase",
                   "plot_unscrambling", "story_generation", "typos"]
OLLM_BENCHMARKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
COMPASS_BENCHMARKS = ["HLE", "AIME2025", "MMLU-Pro", "LiveCodeBenchV6",
                      "GPQA-Diamond", "IFEval"]

# cut depths used in the paper's headline table
CUTS = {"LiveBench": 50, "OLLM v2": 100, "CompassAcademic": 24}

# cut grids for the depth sweep (Fig. 1)
CUT_GRID = {"LiveBench": [15, 20, 25, 30, 40, 50, 65, 80, 100, 125, 152],
            "OLLM v2": [25, 50, 100, 200, 400, 800, 1600, 3000, 4575],
            "CompassAcademic": [10, 12, 15, 18, 21, 24, 30, 36, 42, 48]}


def load_livebench(path=None):
    return pd.read_parquet(path or os.path.join(DATA, "livebench_panel_clean.parquet"))


def load_livebench_raw(path=None):
    """Per-question judged scores. Needed only for reliability and rank CIs."""
    return pd.read_parquet(path or os.path.join(DATA, "livebench_model_judgment_raw.parquet"))


def load_ollm(path=None):
    return pd.read_parquet(path or os.path.join(DATA, "ollm_panel.parquet"))


def load_compass(path=None):
    return pd.read_csv(path or os.path.join(DATA, "compassacademic_panel_clean.csv"))


def all_panels():
    """-> {name: (score_matrix, headline_cut)}"""
    return {
        "LiveBench":       (load_livebench()[LIVEBENCH_TASKS], CUTS["LiveBench"]),
        "OLLM v2":         (load_ollm()[OLLM_BENCHMARKS], CUTS["OLLM v2"]),
        "CompassAcademic": (load_compass()[COMPASS_BENCHMARKS], CUTS["CompassAcademic"]),
    }
