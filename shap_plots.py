from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, kendalltau

BASE = Path(__file__).resolve().parent.parent / "env_encoder_comparison"
OUT = BASE / "plots"
OUT.mkdir(exist_ok=True)

FEAT = ["ALLSKY_SFC_SW_DNI", "WS2M", "ALLSKY_SFC_PAR_TOT", "PS", "ALLSKY_SFC_SW_DWN",
        "T2M_MAX", "PRECTOTCORR", "GWETTOP", "QV2M", "GWETPROF", "RH2M", "GWETROOT",
        "T2M", "T2MDEW", "T2M_MIN", "T2MWET"]
SHORT = {"ALLSKY_SFC_SW_DNI": "DNI", "ALLSKY_SFC_PAR_TOT": "PAR_TOT", "ALLSKY_SFC_SW_DWN": "DWN"}
lbl = lambda f: SHORT.get(f, f)
N = len(FEAT)

att = pd.read_csv(BASE / "all_16_variables" / "per_variable_attention_by_env.csv")[FEAT].mean(0).values
S = np.load(BASE / "shap_vs_attention" / "per_cluster" / "all_seed_shap_values.npy")
NSEED = S.shape[0]
shap_per_seed = np.abs(S).mean(axis=1)
shap_ens = shap_per_seed.mean(axis=0)
att_order = np.argsort(-att)

BLUE, ORANGE, GREY = "#4c78a8", "#f58518", "#9aa0a6"


ks = np.arange(1, N + 1)
ov = np.zeros((NSEED, N), dtype=int)
for s in range(NSEED):
    so = np.argsort(-shap_per_seed[s])
    for i, k in enumerate(ks):
        ov[s, i] = len(set(att_order[:k]) & set(so[:k]))
chance = ks ** 2 / N

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
ax.fill_between(ks, ov.min(0), ov.max(0), color=BLUE, alpha=0.2, label="20-run range")
ax.plot(ks, ov.mean(0), "o-", color=BLUE, lw=2, ms=5, label="observed (mean of 20 runs)")
ax.plot(ks, chance, "--", color=GREY, lw=2, label="expected by chance")
ax.set_xlabel("top-$k$ threshold"); ax.set_ylabel("variables in both top-$k$ sets")
ax.set_title("Attention / SHAP top-$k$ overlap"); ax.set_xticks(ks[::2])
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)

ax = axes[1]
fold = ov.mean(0) / chance
ax.axhline(1.0, ls="--", color=GREY, lw=2, label="chance (1.0x)")
ax.plot(ks, fold, "o-", color=ORANGE, lw=2, ms=5)
ax.set_xlabel("top-$k$ threshold"); ax.set_ylabel("fold enrichment over chance")
ax.set_title("Overlap enrichment"); ax.set_xticks(ks[::2])
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "A_topk_overlap.png", dpi=220); plt.close(fig)


K = 5
attention_top = set(att_order[:K])
cnt = {f: 0 for f in FEAT}
for s in range(NSEED):
    for i in attention_top & set(np.argsort(-shap_per_seed[s])[:K]):
        cnt[FEAT[i]] += 1
ser = pd.Series(cnt).sort_values()
ser = ser[ser > 1]

fig, ax = plt.subplots(figsize=(8, 0.55 * len(ser) + 2.2))
bars = ax.barh([lbl(f) for f in ser.index], ser.values, color=BLUE)
ax.bar_label(bars, labels=[f"{v}/{NSEED}" for v in ser.values], padding=4)
ax.set_xlim(0, NSEED * 1.15)
ax.set_xlabel(f"runs (of {NSEED}) where the variable is in BOTH top-{K} sets")
ax.set_title(f"Cross-method consensus variables (attention $\\cap$ SHAP, top-{K})")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "B_consensus_variables.png", dpi=220); plt.close(fig)


o = np.argsort(-shap_ens)
vals = shap_ens[o]
share = 100 * vals / vals.sum()
cut = int(np.searchsorted(np.cumsum(share), 99.0) + 1)

fig, ax = plt.subplots(figsize=(9, 5))
colors = [ORANGE if i < cut else GREY for i in range(N)]
ax.bar(range(N), vals, color=colors)
ax.set_yscale("log")
ax.set_xticks(range(N)); ax.set_xticklabels([lbl(FEAT[i]) for i in o], rotation=45, ha="right")
ax.axvline(cut - 0.5, color="black", ls="--", lw=1.5)
ax.annotate(f"top {cut} carry {share[:cut].sum():.2f}% of total |SHAP|",
            xy=(cut - 0.5, vals.max()), xytext=(cut + 0.3, vals.max() * 0.5),
            fontsize=10, color="black")
ax.set_ylabel("mean |SHAP|  (log scale)")
ax.set_title("SHAP magnitude collapses below the top of the ranking\n(20-run ensemble)")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "C_shap_magnitude_cliff.png", dpi=220); plt.close(fig)


att_rank = pd.Series(att, index=FEAT).rank(ascending=False)
shap_rank = pd.Series(shap_ens, index=FEAT).rank(ascending=False)
top = list(shap_rank.sort_values().index[:9])

K_SLOPE = 10
att_top = set(att_rank.sort_values().index[:K_SLOPE])
shap_top = set(shap_rank.sort_values().index[:K_SLOPE])
both = att_top & shap_top
shown = sorted(att_top | shap_top, key=lambda f: shap_rank[f])

fig, ax = plt.subplots(figsize=(8, 6.5))
for f in shown:
    a, s = att_rank[f], shap_rank[f]
    consensus = f in both
    col = BLUE if consensus else ORANGE
    ax.plot([0, 1], [a, s], "-o", color=col, lw=2.4 if consensus else 1.6, ms=6,
            alpha=0.9 if consensus else 0.55, zorder=3 if consensus else 2)
    ax.text(-0.04, a, lbl(f), ha="right", va="center", fontsize=10,
            fontweight="bold" if consensus else "normal")
    ax.text(1.04, s, lbl(f), ha="left", va="center", fontsize=10,
            fontweight="bold" if consensus else "normal")
ax.set_xlim(-0.55, 1.55); ax.invert_yaxis()
ax.set_xticks([0, 1]); ax.set_xticklabels(["attention rank", "SHAP rank"], fontsize=11)
ax.set_ylabel("rank (1 = most important)")
ax.set_title(f"Both methods select mostly the same variables, in a different order\n"
             f"(blue = in both top-{K_SLOPE} sets: {len(both)}/{K_SLOPE}; orange = selected by one method only)")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "D_rank_slope.png", dpi=220); plt.close(fig)


def tie_negligible(m, frac=0.001):
    m = m.copy(); m[m < frac * m.sum()] = 0.0
    return m

taus = np.array([kendalltau(tie_negligible(shap_per_seed[s]), att)[0] for s in range(NSEED)])
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(taus, bins=np.arange(-0.05, 0.65, 0.05), color=BLUE, edgecolor="white")
ax.axvline(0, color="black", lw=1.5, label="no association")
ax.axvline(taus.mean(), color=ORANGE, lw=2.5, ls="--", label=f"mean $\\tau_b$ = {taus.mean():.3f}")
ax.set_xlabel(r"Kendall $\tau_b$ (negligible SHAP values tied)")
ax.set_ylabel("runs"); ax.set_title(f"Rank agreement across {NSEED} independent retrainings\n(positive in every run)")
ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "E_kendall_distribution.png", dpi=220); plt.close(fig)

print(f"5 plots -> {OUT}")
print(f"  cliff cut at {cut}; top-5 overlap mean {ov.mean(0)[4]:.2f} vs chance {chance[4]:.2f}")
print(f"  Kendall tau_b: mean {taus.mean():+.3f}, min {taus.min():+.3f}, max {taus.max():+.3f}")
