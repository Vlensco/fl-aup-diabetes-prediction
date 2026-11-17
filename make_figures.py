"""
Regenerates the result figures of the paper from the CSV/JSON files in results/.

    Fig. 3  figures/figure_threshold_sweep_validation.png
            Precision, recall, and F1-score across decision thresholds on the validation set
    Fig. 5  figures/figure_validation_f1_loss_rounds.png
            Validation F1-score (threshold 0.50) and log loss of the aggregated global models per round
    Fig. 6  figures/figure_update_norm_clipping.png
            Client update norms and clipping behavior across communication rounds

Fig. 2 and Fig. 4 are produced by Data_Preparation_Revised.ipynb.

All curves are the mean over the five seeds; shaded bands are ± one standard deviation.

Usage:
    python make_figures.py                      # reads results/, writes figures/
    python make_figures.py --results results --out figures
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Regenerate result figures from results/")
parser.add_argument("--results", default="results", help="Folder with training_history_*, threshold_sweep_* and summary_* files")
parser.add_argument("--out", default="figures", help="Output folder for the PNG files")
parser.add_argument("--dpi", type=int, default=300)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

# Code keys -> paper names (DP = FUP, ADP = AUP)
SCENARIOS = ["STATIC", "DP", "ADP"]
LABEL = {
    "STATIC": "No perturbation (STATIC)",
    "DP": "Fixed update perturbation (FUP)",
    "ADP": "Adaptive update perturbation (AUP)",
}
COLOR = {"STATIC": "#4472C4", "DP": "#ED7D31", "ADP": "#4CAF50"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def mean_sd(df, x, y):
    """Mean and SD of column y at each value of x (one value per seed)."""
    g = df.groupby(x)[y]
    return g.mean(), g.std(ddof=1).fillna(0.0)


def band(ax, m, s, color, label=None, lw=1.6, ls="-"):
    ax.plot(m.index, m.values, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(m.index, (m - s).values, (m + s).values, color=color, alpha=0.18, lw=0)


# ------------------------------------------------------------------ load data
history = pd.concat(
    (pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.results, "training_history_*.csv")))),
    ignore_index=True,
)

sweeps = []
for f in sorted(glob.glob(os.path.join(args.results, "threshold_sweep_validation_*_seed*.csv"))):
    name = os.path.basename(f).replace("threshold_sweep_validation_", "").replace(".csv", "")
    scenario, seed = name.split("_seed")
    d = pd.read_csv(f)
    d["scenario"], d["seed"] = scenario, int(seed)
    sweeps.append(d)
sweeps = pd.concat(sweeps, ignore_index=True)

selected = {sc: [] for sc in SCENARIOS}
for f in sorted(glob.glob(os.path.join(args.results, "summary_*_seed*.json"))):
    with open(f, encoding="utf-8") as fh:
        s = json.load(fh)
    selected[s["scenario"]].append(float(s["test_at_selected_threshold"]["threshold"]))

# ------------------------------------------------------------------ Fig. 3: threshold sweep
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, sc in zip(axes, SCENARIOS):
    d = sweeps[sweeps.scenario == sc]
    for metric, style, lab in [("precision", "--", "Precision"), ("recall", ":", "Recall"), ("f1", "-", "F1-score")]:
        m, s = mean_sd(d, "threshold", metric)
        color = "black" if metric == "f1" else "0.45"
        band(ax, m, s, color, label=lab, lw=1.8 if metric == "f1" else 1.3, ls=style)
    t = np.array(selected[sc])
    ax.axvline(t.mean(), color=COLOR[sc], lw=1.5)
    ax.text(0.97, 0.05, f"selected\n{t.mean():.2f} ± {t.std(ddof=1):.2f}", transform=ax.transAxes,
            ha="right", va="bottom", color=COLOR[sc], fontsize=8.5)
    ax.set_title(LABEL[sc], fontsize=10)
    ax.set_xlabel("Decision threshold")
    ax.set_xlim(d.threshold.min(), d.threshold.max())
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Validation metric")
axes[0].legend(loc="lower left", frameon=False, fontsize=8)
fig.tight_layout()
p3 = os.path.join(args.out, "figure_threshold_sweep_validation.png")
fig.savefig(p3, dpi=args.dpi, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Fig. 5: validation F1 / loss per round
glob_rounds = history[history.entity.isin(["INITIAL_GLOBAL", "AGGREGATED"]) & (history.eval_split == "validation")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
for sc in SCENARIOS:
    d = glob_rounds[glob_rounds.scenario == sc]
    m, s = mean_sd(d, "round", "f1")
    band(a1, m, s, COLOR[sc], label=LABEL[sc])
    m, s = mean_sd(d, "round", "loss")
    band(a2, m, s, COLOR[sc], label=LABEL[sc])
a1.set_ylabel("Validation F1-score (threshold 0.50)")
a2.set_ylabel("Validation log loss")
for a in (a1, a2):
    a.set_xlabel("Communication round")
    a.set_xlim(0, glob_rounds["round"].max())
a1.set_ylim(0, 0.7)
a1.legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
p5 = os.path.join(args.out, "figure_validation_f1_loss_rounds.png")
fig.savefig(p5, dpi=args.dpi, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Fig. 6: update norms and clipping
clients = history[history.entity.isin(["RS_01", "RS_02"])]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))

aup = clients[clients.scenario == "ADP"]
# mean over both clients within a run first, then mean ± SD across seeds
norm_run = aup.groupby(["seed", "round"])["update_norm"].mean().reset_index()
m, s = mean_sd(norm_run, "round", "update_norm")
band(a1, m, s, "0.25", label="Client update L2 norm before clipping (AUP runs)")
clip_run = aup.groupby(["seed", "round"])["clip_threshold"].mean().reset_index()
m, s = mean_sd(clip_run, "round", "clip_threshold")
band(a1, m, s, COLOR["ADP"], label="Adaptive clip threshold (AUP)")
fup_clip = float(clients[clients.scenario == "DP"]["clip_threshold"].median())
a1.axhline(fup_clip, color=COLOR["DP"], ls="--", lw=1.3, label=f"Fixed clip threshold (FUP) = {fup_clip:g}")
a1.set_ylabel("L2 norm")
a1.legend(loc="upper right", frameon=False, fontsize=8)

for sc in ["DP", "ADP"]:
    d = clients[clients.scenario == sc].groupby(["seed", "round"])["clip_scale"].mean().reset_index()
    m, s = mean_sd(d, "round", "clip_scale")
    band(a2, m, s, COLOR[sc], label=LABEL[sc])
a2.set_ylabel("Clipping scale factor (1 = no clipping)")
a2.set_ylim(0, 1.05)
a2.legend(loc="lower right", frameon=False, fontsize=8)
for a in (a1, a2):
    a.set_xlabel("Communication round")
    a.set_xlim(1, clients["round"].max())
fig.tight_layout()
p6 = os.path.join(args.out, "figure_update_norm_clipping.png")
fig.savefig(p6, dpi=args.dpi, bbox_inches="tight")
plt.close(fig)

for p in (p3, p5, p6):
    print("Saved:", p)

# Clipping statistic quoted in the paper: share of client updates (rounds 15-50, both clients, all seeds)
# whose norm was reduced by more than 0.1% by clipping (clip_scale < 0.999).
late = clients[clients["round"] >= 15]
for sc in ["DP", "ADP"]:
    share = (late[late.scenario == sc]["clip_scale"] < 0.999).mean() * 100
    print(f"Clipped client updates from round 15 onward - {LABEL[sc]}: {share:.1f}%")
