import pandas as pd

from annotation_tools import (
    BASE_DIR,
    OUT_DIR,
    WINDOWS_BP,
    load_crop_annotations,
    map_markers_to_genes,
    normalize_marker_id,
)

ranking = pd.read_csv(BASE_DIR / "arch_a_multienv_1000_marker_ranking_chrpos.csv")
ranking["marker_id"] = normalize_marker_id(ranking["marker_id"])
universe = ranking[["marker_id", "chr", "position", "node_index"]].copy()
universe["chr"] = universe["chr"].astype(int)
universe["position"] = universe["position"].astype(int)

coords = load_crop_annotations("maize")["coords"]
for window_bp in WINDOWS_BP:
    gene_map = map_markers_to_genes(universe, coords, window_bp, "maize", "maize_full_universe")
    gene_map["universe_id"] = "maize_full_universe"
    out_path = OUT_DIR / f"maize_full_universe_marker_gene_window{window_bp}.csv"
    gene_map.to_csv(out_path, index=False)
    print(f"Saved {out_path.name}: {len(gene_map)} marker-gene pairs")
