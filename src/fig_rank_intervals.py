"""Fig. 3: rank intervals + top-10 membership probability for the LiveBench panel.

Reads results/livebench_rank_intervals.csv (produced by rank_bootstrap.py via
run_all.py) and writes fig2_rank_intervals.png.

The figure makes one claim: the published top ten is not a resolved set. Models
whose 95% rank interval reaches rank 10 or better are drawn in the focal colour;
the shaded band is the published top ten. The coloured block runs to published
rank 19, i.e. roughly twice the length of the band.
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOP_M = 10          # the published slice under test
N_SHOW = 25         # rows drawn

FOCAL = "#1f4e79"   # could be in the published top ten at 95%
OUT = "#9aa0a6"     # cannot be, at 95%
BAND = "#dce6f1"    # the published top ten

# Plot style used for the published figure (set here so the script is self-contained).
STYLE = {
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "axes.titlelocation": "left", "legend.fontsize": 7, "legend.frameon": False,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "patch.linewidth": 0.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
}

# Display names. Full strings are in results/livebench_rank_intervals.csv; these
# are shortened to keep the label column narrow in a single-column figure.
SHORT = {
    "gemini-2.5-pro-exp-03-25": "gemini-2.5-pro-exp",
    "o1-2024-12-17-high": "o1-high",
    "claude-3-7-sonnet-20250219-thinking-64k": "claude-3.7-think-64k",
    "o1-preview-2024-09-12": "o1-preview",
    "gpt-4.5-preview-2025-02-27": "gpt-4.5-preview",
    "o3-mini-2025-01-31-high": "o3-mini-high",
    "claude-3-7-sonnet-20250219-thinking-25k": "claude-3.7-think-25k",
    "claude-3-7-sonnet-20250219-base": "claude-3.7-base",
    "o1-2024-12-17-medium": "o1-medium",
    "claude-3-5-sonnet-20241022": "claude-3.5-1022",
    "chatgpt-4o-latest-2025-03-27": "chatgpt-4o-latest",
    "claude-3-5-sonnet-20240620": "claude-3.5-0620",
    "o1-2024-12-17-low": "o1-low",
    "o3-mini-2025-01-31-medium": "o3-mini-medium",
    "gemini-2.0-pro-exp-02-05": "gemini-2.0-pro-exp",
    "step-2-16k-202411": "step-2-16k",
    "gpt-4o-2024-08-06": "gpt-4o-0806",
}


def make_figure(ranks: pd.DataFrame, out_path: str, rank1_stability: float,
                n_ci_in_top: int, n_p5: int):
    d = ranks.sort_values("obs_rank").head(N_SHOW).reset_index(drop=True)
    y = np.arange(len(d))
    reach = (d.rank_lo <= TOP_M).values            # could be in the published top ten
    col = np.where(reach, FOCAL, OUT)
    labels = [f"{r.obs_rank:>2}  {SHORT.get(r.model, r.model)}" for r in d.itertuples()]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(3.45, 3.72), sharey=True,
        gridspec_kw=dict(width_ratios=[2.45, 1.0], wspace=0.07))

    # ---- left: published rank and its 95% interval --------------------------
    axL.axvspan(0.4, TOP_M + 0.5, color=BAND, zorder=0, lw=0)
    axL.axvline(TOP_M + 0.5, color="#7f8c99", lw=0.7, zorder=1)
    for i, r in enumerate(d.itertuples()):
        axL.plot([r.rank_lo, r.rank_hi], [i, i], color=col[i], lw=2.2,
                 solid_capstyle="round", zorder=2)
    axL.scatter(d.obs_rank, y, s=13, facecolor="white", edgecolor=col,
                linewidths=1.0, zorder=3)
    axL.set_xlim(0.4, d.rank_hi.max() + 1.5)
    axL.set_xlabel("rank  (1 = best)")
    axL.set_xticks([1, 10, 20, 30])
    axL.set_yticks(y)
    axL.set_yticklabels(labels, fontfamily="DejaVu Sans Mono")
    axL.invert_yaxis()
    axL.text(TOP_M + 0.1, len(d) - 0.45, "published\ntop ten", ha="right", va="bottom",
             fontsize=6, color="#3d5166", linespacing=1.1)

    # ---- right: how often the model actually lands in the top ten -----------
    axR.barh(y, d["p_top10"], height=0.62, color=col, lw=0, zorder=2)
    axR.axvline(0.05, color="#444444", lw=0.7, ls=(0, (2, 2)), zorder=3)
    axR.set_xlim(0, 1.04)
    axR.set_xticks([0, 0.5, 1])
    axR.set_xticklabels(["0", ".5", "1"])
    axR.set_xlabel("P(in top ten)")
    axR.text(0.09, len(d) - 1.3, f"5%: {n_p5} of 152", fontsize=6, color="#444444",
             ha="left", va="center")

    # ---- the one thing worth annotating in the panel ------------------------
    axL.annotate(
        f"[1,1]; first in {100*rank1_stability:.1f}%\nof replicates",
        xy=(1.9, 0), xytext=(11.8, 1.45), fontsize=6, color="#333333", linespacing=1.15,
        arrowprops=dict(arrowstyle="-", lw=0.6, color="#333333",
                        connectionstyle="angle,angleA=0,angleB=90,rad=2"))

    # ---- key: the colour split IS the claim ---------------------------------
    handles = [mpl.lines.Line2D([], [], color=FOCAL, lw=2.4),
               mpl.lines.Line2D([], [], color=OUT, lw=2.4)]
    fig.subplots_adjust(left=0.345, right=0.975, top=0.865, bottom=0.095)
    fig.text(0.004, 0.988, "The published top ten is statistically a top twenty",
             ha="left", va="top", fontsize=8)
    fig.legend(handles,
               [f"95% interval reaches the top ten: {n_ci_in_top} of 152 models",
                "does not reach the top ten"],
               loc="upper left", bbox_to_anchor=(0.004, 0.952), ncol=1, frameon=False,
               fontsize=6, handlelength=1.1, handletextpad=0.45, labelspacing=0.25,
               borderpad=0.0)
    fig.savefig(out_path, dpi=300)
    return fig


if __name__ == "__main__":
    import json, os, sys
    mpl.rcParams.update(STYLE)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ranks = pd.read_csv(os.path.join(root, "results", "livebench_rank_intervals.csv"))
    summ = json.load(open(os.path.join(root, "results", "rank_and_reliability_summary.json")))
    make_figure(ranks, os.path.join(root, "results", "fig2_rank_intervals.png"),
                summ["rank1_stability"], summ["n_models_ci_includes_top10"],
                summ["n_models_reaching_top10_ge5pct"])
