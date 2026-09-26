from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from annotation_tools import (
    WINDOWS_BP,
    bh_adjust,
    enrichment_for_set,
    is_focused_term,
    load_crop_annotations,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CLUSTER_DIR = BASE_DIR / "all_marker_cluster_analysis_envresidual" / "maize"
DATA_MINING_DIR = BASE_DIR / "data_mining"
OUT_DIR = DATA_MINING_DIR / "new_cluster_marker_analysis"

MIN_SELECTED_GENES_FOR_TERM = 3
FDR_CUTOFF = 0.10
FOCUSED_TOP_PER_SET = 6

CLUSTER_LABELS = {
    0: "cool, temperate, windy, moderate humidity",
    1: "warm, humid, wet, low wind",
    2: "cool, humid, low wind, low radiation",
    3: "dry, high-sun, low-pressure, windy",
}


def build_marker_sets() -> pd.DataFrame:
    frames = []

    shared = pd.read_csv(CLUSTER_DIR / "top1000_markers_shared_by_all_clusters.csv")
    shared_ids = shared["marker_id"].astype(int).tolist()
    frames.append(
        pd.DataFrame(
            {
                "marker_id": shared_ids,
                "set_id": "global_shared_by_all_4_clusters",
                "set_label": "Global (shared by all 4 clusters, 27 markers)",
                "top_k": None,
            }
        )
    )

    for cluster_id, label in CLUSTER_LABELS.items():
        ranking = pd.read_csv(CLUSTER_DIR / f"cluster_{cluster_id}" / f"arch_a_all_markers_cluster{cluster_id}_marker_ranking.csv")
        for top_k in (50, 100):
            subset = ranking[ranking["rank"] <= top_k]
            frames.append(
                pd.DataFrame(
                    {
                        "marker_id": subset["marker_id"].astype(int).tolist(),
                        "set_id": f"C{cluster_id}_top{top_k}",
                        "set_label": f"C{cluster_id} ({label}) top {top_k}",
                        "top_k": top_k,
                    }
                )
            )

    all_sets = pd.concat(frames, ignore_index=True)
    all_sets.to_csv(OUT_DIR / "marker_sets.csv", index=False)
    return all_sets


GENE_COLUMNS = [
    "marker_id", "marker_chr", "marker_position", "gene_id", "gene_name",
    "gene_chr", "gene_start", "gene_end", "distance_bp", "gene_biotype", "description",
]


def build_gene_maps(all_sets: pd.DataFrame) -> dict[int, pd.DataFrame]:
    gene_maps = {}
    for window_bp in WINDOWS_BP:
        universe_map = pd.read_csv(DATA_MINING_DIR / f"maize_full_universe_marker_gene_window{window_bp}.csv")
        universe_map["marker_id"] = universe_map["marker_id"].astype(int)
        universe_map = universe_map[GENE_COLUMNS]
        merged = all_sets.merge(universe_map, on="marker_id", how="left")
        merged.to_csv(OUT_DIR / f"marker_gene_map_window{window_bp}.csv", index=False)
        gene_maps[window_bp] = merged
    return gene_maps


def build_background(window_bp: int) -> tuple[set[str], pd.DataFrame]:
    universe_map = pd.read_csv(DATA_MINING_DIR / f"maize_full_universe_marker_gene_window{window_bp}.csv")
    background_genes = set(universe_map["gene_id"].dropna().astype(str))
    return background_genes, universe_map


def run_enrichment(all_sets: pd.DataFrame, gene_maps: dict[int, pd.DataFrame], annotations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_rows = []
    summary_rows = []
    set_meta = all_sets.drop_duplicates("set_id")[["set_id", "set_label", "top_k"]]

    for window_bp in WINDOWS_BP:
        gene_map = gene_maps[window_bp]
        background_genes, universe_map = build_background(window_bp)
        for _, meta in set_meta.iterrows():
            set_id = meta["set_id"]
            subset = gene_map[gene_map["set_id"].eq(set_id)]
            selected_genes = set(subset["gene_id"].dropna().astype(str))
            marker_count = subset["marker_id"].nunique()
            markers_with_gene = subset.dropna(subset=["gene_id"])["marker_id"].nunique()
            summary_rows.append(
                {
                    "set_id": set_id,
                    "set_label": meta["set_label"],
                    "window_bp": window_bp,
                    "marker_count": marker_count,
                    "markers_with_nearby_gene": markers_with_gene,
                    "unique_nearby_genes": len(selected_genes),
                    "background_gene_count": len(background_genes),
                }
            )
            for source, annotation in annotations.items():
                if source == "coords":
                    continue
                enrichment = enrichment_for_set(selected_genes, background_genes, annotation, source)
                if enrichment.empty:
                    continue
                enrichment.insert(0, "set_id", set_id)
                enrichment.insert(1, "set_label", meta["set_label"])
                enrichment.insert(2, "window_bp", window_bp)
                all_rows.append(enrichment)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "marker_to_gene_summary.csv", index=False)

    enrichment_all = pd.concat(all_rows, ignore_index=True, sort=False) if all_rows else pd.DataFrame()
    enrichment_all.to_csv(OUT_DIR / "enrichment_all_terms.csv", index=False)
    significant = enrichment_all[enrichment_all["q_value_bh"].le(FDR_CUTOFF)] if not enrichment_all.empty else enrichment_all
    significant.to_csv(OUT_DIR / "enrichment_significant_fdr10.csv", index=False)
    return enrichment_all


def build_focused_genes(enrichment_all: pd.DataFrame, gene_maps: dict[int, pd.DataFrame]) -> pd.DataFrame:
    if enrichment_all.empty:
        pd.DataFrame().to_csv(OUT_DIR / "focused_term_contributing_genes.csv", index=False)
        return pd.DataFrame()

    sig = enrichment_all[enrichment_all["q_value_bh"].le(FDR_CUTOFF)].copy()
    focused = sig[sig.apply(is_focused_term, axis=1)].copy()
    focused = focused.sort_values(["set_id", "window_bp", "q_value_bh", "p_value"], kind="stable")
    focused["term_rank_within_set"] = focused.groupby(["set_id", "window_bp"], sort=False).cumcount() + 1
    top_per_set = focused[focused["term_rank_within_set"].le(FOCUSED_TOP_PER_SET)].copy()
    top_per_set.to_csv(OUT_DIR / "focused_enrichment_terms.csv", index=False)

    annotations = load_crop_annotations("maize")
    rows = []
    for term in top_per_set.itertuples(index=False):
        source = term.source
        if source not in annotations:
            continue
        ann = annotations[source]
        term_genes = set(ann.loc[ann["term_id"].astype(str).eq(str(term.term_id)), "gene_id"].dropna().astype(str))
        if not term_genes:
            continue
        gene_map = gene_maps[term.window_bp]
        mapped = gene_map[
            gene_map["set_id"].eq(term.set_id) & gene_map["gene_id"].astype(str).isin(term_genes)
        ].copy()
        if mapped.empty:
            continue
        mapped = mapped.sort_values(["distance_bp", "marker_id", "gene_id"], kind="stable")
        mapped = mapped.drop_duplicates("gene_id").head(10)
        for gene in mapped.itertuples(index=False):
            rows.append(
                {
                    "set_id": term.set_id,
                    "set_label": term.set_label,
                    "window_bp": int(term.window_bp),
                    "source": source,
                    "term_id": term.term_id,
                    "term_name": term.term_name,
                    "q_value_bh": float(term.q_value_bh),
                    "enrichment_ratio": float(term.enrichment_ratio),
                    "marker_id": gene.marker_id,
                    "marker_chr": int(gene.marker_chr),
                    "marker_position": int(gene.marker_position),
                    "gene_id": gene.gene_id,
                    "gene_name": gene.gene_name,
                    "distance_bp": int(gene.distance_bp),
                    "description": gene.description,
                }
            )
    contributing = pd.DataFrame(rows)
    contributing.to_csv(OUT_DIR / "focused_term_contributing_genes.csv", index=False)
    return contributing


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building 9 marker sets (global-27 + 4 clusters x top50/top100)...", flush=True)
    all_sets = build_marker_sets()
    print(all_sets.groupby(["set_id", "set_label"], sort=False)["marker_id"].count().to_string(), flush=True)

    print("\nFiltering precomputed universe gene-window maps (no new mapping needed)...", flush=True)
    gene_maps = build_gene_maps(all_sets)

    print("\nLoading cached maize functional annotations (GO/GOSlim/InterPro/Pfam/PlantReactome)...", flush=True)
    annotations = load_crop_annotations("maize")
    for source, df in annotations.items():
        if source != "coords":
            print(f"  {source}: {len(df)} gene-term rows", flush=True)

    print("\nRunning hypergeometric enrichment per set per window...", flush=True)
    enrichment_all = run_enrichment(all_sets, gene_maps, annotations)
    print(f"Total enrichment rows tested (before FDR filter): {len(enrichment_all)}", flush=True)
    sig = enrichment_all[enrichment_all["q_value_bh"].le(FDR_CUTOFF)] if not enrichment_all.empty else enrichment_all
    print(f"Significant at FDR<{FDR_CUTOFF}: {len(sig)}", flush=True)

    print("\nBuilding focused (non-generic) term + contributing gene tables...", flush=True)
    contributing = build_focused_genes(enrichment_all, gene_maps)
    print(f"Focused contributing-gene rows: {len(contributing)}", flush=True)

    manifest = {
        "marker_sets": all_sets.groupby("set_id")["marker_id"].count().to_dict(),
        "total_enrichment_tests": int(len(enrichment_all)),
        "significant_fdr10": int(len(sig)),
        "focused_contributing_gene_rows": int(len(contributing)),
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. Results written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
