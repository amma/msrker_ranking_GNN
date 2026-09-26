from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ranking_df = pd.read_csv(BASE_DIR / "arch_a_multienv_1000_marker_ranking.csv").sort_values("rank", kind="stable").reset_index(drop=True)
weights = ranking_df["avg_weight"]
mean_weight = float(weights.mean())

fig1, ax1 = plt.subplots(figsize=(11, 4.5))
ax1.plot(range(1, len(ranking_df) + 1), weights, color="#1f5aa6", linewidth=1.5)
ax1.axhline(mean_weight, color="#c0392b", linestyle="--", linewidth=1, label=f"Mean = {mean_weight:.9f}")
ax1.set_title("Ranked Marker Weights for 1000-Genotype Raw-Yield Run")
ax1.set_xlabel("Marker rank")
ax1.set_ylabel("Average attention weight")
ax1.grid(alpha=0.25)
ax1.legend(frameon=False)
fig1.savefig(BASE_DIR / "arch_a_multienv_1000_ranked_weight_curve.png", dpi=180, bbox_inches="tight")

fig2, ax2 = plt.subplots(figsize=(8.5, 4.5))
ax2.hist(weights, bins=60, color="#2a9d8f", edgecolor="white")
ax2.axvline(mean_weight, color="#c0392b", linestyle="--", linewidth=1, label=f"Mean = {mean_weight:.9f}")
ax2.set_title("Distribution of Marker Weights for 1000-Genotype Raw-Yield Run")
ax2.set_xlabel("Average attention weight")
ax2.set_ylabel("Number of markers")
ax2.grid(alpha=0.2)
ax2.legend(frameon=False)
fig2.savefig(BASE_DIR / "arch_a_multienv_1000_weight_histogram.png", dpi=180, bbox_inches="tight")

plot_top = ranking_df.head(50).copy().sort_values("avg_weight", ascending=True, kind="stable")
plot_top["marker_label"] = plot_top["node_label"] + " | " + plot_top["marker_id"].astype(int).astype(str)
fig3, ax3 = plt.subplots(figsize=(11, 12.5))
ax3.barh(plot_top["marker_label"], plot_top["avg_weight"], color="#e76f51")
ax3.set_title("Top 50 Marker Weights for 1000-Genotype Raw-Yield Run")
ax3.set_xlabel("Average attention weight")
ax3.set_ylabel("Marker node | marker ID")
ax3.grid(axis="x", alpha=0.25)
fig3.savefig(BASE_DIR / "arch_a_multienv_1000_top50_marker_weights.png", dpi=180, bbox_inches="tight")

print("Saved: arch_a_multienv_1000_ranked_weight_curve.png")
print("Saved: arch_a_multienv_1000_weight_histogram.png")
print("Saved: arch_a_multienv_1000_top50_marker_weights.png")
