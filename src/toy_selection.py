"""Fig. 1: selection on a composite attenuates agreement among its components, in a world
where every benchmark measures one ability and nothing is wrong.

Standalone -- needs no project data. Writes results/fig0_toy_selection.png.

The right panel is a MEDIAN over independent replicate panels with a 95% band, not a single
draw: at a 1% cut the between-panel spread is larger than the quantity being compared, so a
single realization is not interpretable on its own.

The dashed grey reference keeps the SAME number of models chosen at random rather than from
the top, on the same replicate panels. It uses its own generator (seed offset RSEED), so the
top-c curve and every number printed for it are unchanged by its presence. It shows that the
collapse is caused by selection on the average, not by the slice being smaller.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

LAM, K, N = 0.82, 7, 4000      # LAM = mean loading fitted on the three real panels
NREP, TOPF = 500, 0.05
FRACS = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00])
FOCAL, BASE = "#C0392B", "#9AA5B1"
RANDC, RSEED = "#4D5763", 100_000


def simulate(seed):
    """One synthetic panel: a single latent ability plus independent per-benchmark error."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=N)
    Z = LAM * theta[:, None] + np.sqrt(1 - LAM ** 2) * rng.normal(size=(N, K))
    return Z, np.argsort(-Z.mean(axis=1))


def mean_offdiag(S):
    return float(np.mean([spearmanr(S[:, a], S[:, b]).statistic
                          for a in range(S.shape[1]) for b in range(a + 1, S.shape[1])]))


def curve(Z, order):
    return [mean_offdiag(Z[order[:max(int(f * N), 12)]]) for f in FRACS]


def random_curve(Z, seed):
    """Same slice sizes as curve(), but the models are drawn at random, not from the top."""
    rng = np.random.default_rng(RSEED + seed)
    return [mean_offdiag(Z[rng.choice(N, max(int(f * N), 12), replace=False)]) for f in FRACS]


curves, rcurves = [], []
for s in range(NREP):
    Z, order = simulate(s)
    curves.append(curve(Z, order))
    rcurves.append(random_curve(Z, s))
curves, rcurves = np.array(curves), np.array(rcurves)
med = np.median(curves, 0)
lo, hi = np.percentile(curves, 2.5, 0), np.percentile(curves, 97.5, 0)
rmed = np.median(rcurves, 0)
rlo, rhi = np.percentile(rcurves, 2.5, 0), np.percentile(rcurves, 97.5, 0)

Z0, order0 = simulate(0)                       # one panel, for the scatter
top0 = order0[:int(TOPF * N)]
r_full = spearmanr(Z0[:, 0], Z0[:, 1]).statistic
r_top = spearmanr(Z0[top0, 0], Z0[top0, 1]).statistic

print(f"full panel : {med[-1]:.3f}  (sd {curves[:, -1].std():.3f} over {NREP} panels)")
print(f"top 5%     : {med[2]:.3f}  95% band [{lo[2]:.3f}, {hi[2]:.3f}]")
print(f"top 1%     : {med[0]:.3f}  95% band [{lo[0]:.3f}, {hi[0]:.3f}]")
print(f"random 5%  : {rmed[2]:.3f}  95% band [{rlo[2]:.3f}, {rhi[2]:.3f}]")
print(f"random 1%  : {rmed[0]:.3f}  95% band [{rlo[0]:.3f}, {rhi[0]:.3f}]")
print(f"scatter pair (seed 0): full {r_full:.2f}, top 5% {r_top:.2f}")

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110})
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.0))

axa.scatter(Z0[:, 0], Z0[:, 1], s=4, c=BASE, alpha=.35, lw=0, rasterized=True)
axa.scatter(Z0[top0, 0], Z0[top0, 1], s=9, c=FOCAL, alpha=.85, lw=0)
axa.set_xlabel("score on benchmark 1 (z-units)")
axa.set_ylabel("score on benchmark 2 (z-units)")
axa.set_title("One simulated panel, two of its benchmarks", loc="left")
axa.margins(.06)
axa.text(.97, .14, f"all {N:,} models:  $\\rho$ = {r_full:.2f}", transform=axa.transAxes,
         ha="right", color="#67727E", fontsize=7, fontweight="bold")
axa.text(.97, .05, f"top 5% by average:  $\\rho$ = {r_top:.2f}", transform=axa.transAxes,
         ha="right", color=FOCAL, fontsize=7, fontweight="bold")

x = FRACS * 100
axb.axvspan(2.2, 50, color=BASE, alpha=.18, lw=0, zorder=0)
axb.fill_between(x, rlo, rhi, color=RANDC, alpha=.10, lw=0, zorder=1)
axb.plot(x, rmed, "--", color=RANDC, lw=1.1, zorder=2)
axb.fill_between(x, lo, hi, color=FOCAL, alpha=.20, lw=0, zorder=2)
axb.plot(x, med, "-o", color=FOCAL, ms=4.5, lw=1.6, zorder=3)
axb.set_xscale("log")
axb.set_xticks([1, 2, 5, 10, 20, 50, 100])
axb.set_xticklabels(["1%", "2%", "5%", "10%", "20%", "50%", "100%"])
axb.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
axb.set_xlabel("share of models kept (top $c$ by average)")
axb.set_ylabel("mean agreement within the slice\n(off-diagonal Spearman)")
axb.set_title(f"Median over {NREP} simulated panels, 95% bands", loc="left")
axb.set_ylim(-.10, .87)
axb.margins(x=.06)
axb.annotate(f"{med[-1]:.2f} over the\nwhole panel", (100, med[-1]), xytext=(-8, -4),
             textcoords="offset points", ha="right", va="top", fontsize=6.5, color=FOCAL)
axb.annotate(f"{med[0]:.2f} in the\ntop 1%", (1, med[0]), xytext=(2, 20),
             textcoords="offset points", ha="left", va="bottom", fontsize=6.5, color=FOCAL)
axb.annotate("same number of models\nchosen at random", (1, rmed[0]), xytext=(2, -12),
             textcoords="offset points", ha="left", va="top", fontsize=6.5, color=RANDC)
axb.text(2.5, .855, "cut depths used by real leaderboards", ha="left", va="top",
         fontsize=6.5, color="#5A6572")

fig.tight_layout()      # no in-image title: the caption carries the message
fig.savefig("results/fig0_toy_selection.png", dpi=320)
print("wrote results/fig0_toy_selection.png")
