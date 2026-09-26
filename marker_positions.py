from __future__ import annotations

from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
RANKING_PATH = BASE_DIR / "arch_a_multienv_1000_marker_ranking.csv"
GENOTYPE_PATH = BASE_DIR / "genotype_reduced.csv"
ZENG_TABLE_PATH = BASE_DIR / "zeng_2022_yield_related_gwas_table2.csv"

RANKING_CHRPOS_PATH = BASE_DIR / "arch_a_multienv_1000_marker_ranking_chrpos.csv"
TOP_CHRPOS_PATH = BASE_DIR / "arch_a_multienv_1000_top500_chrpos.csv"
ZENG_CHRPOS_PATH = BASE_DIR / "zeng_2022_table2_chrpos.csv"
EXACT_MATCH_PATH = BASE_DIR / "top500_chrpos_exact_zeng_matches.csv"
WINDOW_MATCH_PATH = BASE_DIR / "top500_chrpos_window_zeng_matches.csv"
NEAREST_PATH = BASE_DIR / "top500_chrpos_nearest_zeng_snp.csv"
SUMMARY_PATH = BASE_DIR / "top500_chrpos_evaluation_summary.csv"

TOP_K = 500
RANDOM_SETS = 20_000
SEED = 42


def parse_marker_position(marker_name: object) -> int:
    marker_text = str(marker_name)
    match = re.match(r"^(\d+)", marker_text)
    if not match:
        raise ValueError(f"Cannot parse position from marker name: {marker_name}")
    return int(match.group(1))


def infer_marker_coordinates(marker_columns: list[str]) -> pd.DataFrame:
    records = []
    chromosome = 1
    previous_position = None

    for node_index, marker_name in enumerate(marker_columns, start=1):
        position = parse_marker_position(marker_name)
        if previous_position is not None and position < previous_position:
            chromosome += 1
        records.append(
            {
                "node_index": node_index,
                "marker_id": str(marker_name).replace(".0", ""),
                "chr": chromosome,
                "position": position,
                "chr_pos": f"chr{chromosome}:{position}",
            }
        )
        previous_position = position

    return pd.DataFrame(records)


def load_zeng_table() -> pd.DataFrame:
    if ZENG_TABLE_PATH.exists():
        zeng_df = pd.read_csv(ZENG_TABLE_PATH).copy()
    else:
        url = "https://link.springer.com/article/10.1186/s12870-022-03812-5/tables/2"
        html = requests.get(url, timeout=60).text
        zeng_df = pd.read_html(StringIO(html))[0]

    def parse_zeng_snp(snp: object) -> tuple[int, int]:
        chrom_text, position_text = str(snp).split("_", 1)
        return int(chrom_text), int(position_text.replace(",", ""))

    if "chr" not in zeng_df.columns or "position" not in zeng_df.columns:
        zeng_df[["chr", "position"]] = zeng_df["SNP"].apply(lambda value: pd.Series(parse_zeng_snp(value)))

    zeng_df["chr"] = zeng_df["chr"].astype(int)
    zeng_df["position"] = zeng_df["position"].astype(int)
    zeng_df["chr_pos"] = "chr" + zeng_df["chr"].astype(str) + ":" + zeng_df["position"].astype(str)
    zeng_df["Trait"] = zeng_df["Trait"].astype(str)
    return zeng_df


def marker_hit_mask(all_coords: pd.DataFrame, ref_df: pd.DataFrame, window_bp: int) -> np.ndarray:
    all_chr = all_coords["chr"].to_numpy(dtype=int)
    all_pos = all_coords["position"].to_numpy(dtype=int)
    mask = np.zeros(len(all_coords), dtype=bool)
    for _, hit in ref_df.iterrows():
        mask |= (all_chr == int(hit["chr"])) & (np.abs(all_pos - int(hit["position"])) <= window_bp)
    return mask


def window_matches(marker_df: pd.DataFrame, ref_df: pd.DataFrame, label: str, window_bp: int) -> pd.DataFrame:
    rows = []
    for _, marker in marker_df.iterrows():
        same_chr = ref_df[ref_df["chr"].eq(int(marker["chr"]))].copy()
        if same_chr.empty:
            continue
        same_chr["distance_bp"] = (same_chr["position"].astype(int) - int(marker["position"])).abs()
        same_chr = same_chr[same_chr["distance_bp"].le(window_bp)]
        for _, hit in same_chr.iterrows():
            rows.append(
                {
                    "reference_set": label,
                    "window_bp": window_bp,
                    "rank": int(marker["rank"]),
                    "node_index": int(marker["node_index"]),
                    "marker_id": str(marker["marker_id"]),
                    "model_chr_pos": marker["chr_pos"],
                    "model_avg_weight": float(marker["avg_weight"]),
                    "zeng_trait": hit["Trait"],
                    "zeng_snp": hit["SNP"],
                    "zeng_chr_pos": hit["chr_pos"],
                    "distance_bp": int(hit["distance_bp"]),
                    "candidate_gene": hit.get("Candidate gene", ""),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ranking_df = pd.read_csv(RANKING_PATH).copy()
    ranking_df["marker_id"] = ranking_df["marker_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    marker_columns = list(pd.read_csv(GENOTYPE_PATH, nrows=0).columns)[1:]
    coord_df = infer_marker_coordinates(marker_columns)

    ranking_chrpos_df = ranking_df.merge(coord_df, on=["node_index", "marker_id"], how="left")
    if ranking_chrpos_df[["chr", "position"]].isna().any().any():
        missing = ranking_chrpos_df[ranking_chrpos_df["chr"].isna()].head()
        raise ValueError(f"Could not map some ranked markers to chr:pos: {missing}")

    ranking_chrpos_df.to_csv(RANKING_CHRPOS_PATH, index=False)
    top_df = ranking_chrpos_df.head(TOP_K).copy()
    top_df.to_csv(TOP_CHRPOS_PATH, index=False)

    zeng_df = load_zeng_table()
    zeng_df.to_csv(ZENG_CHRPOS_PATH, index=False)

    exact_matches = []
    reference_sets = {
        "Zeng2022_GYP_only": zeng_df[zeng_df["Trait"].eq("GYP")].copy(),
        "Zeng2022_all_yield_related_traits": zeng_df.copy(),
    }

    for label, ref_df in reference_sets.items():
        merged = top_df.merge(
            ref_df[
                [
                    "Trait",
                    "SNP",
                    "P value",
                    "R2%",
                    "Candidate gene",
                    "Gene annotation",
                    "chr",
                    "position",
                    "chr_pos",
                ]
            ],
            on="chr_pos",
            how="inner",
            suffixes=("_model", "_zeng"),
        )
        if not merged.empty:
            merged["reference_set"] = label
            exact_matches.append(merged)

    exact_matches_df = pd.concat(exact_matches, ignore_index=True) if exact_matches else pd.DataFrame()
    exact_matches_df.to_csv(EXACT_MATCH_PATH, index=False)

    all_coords = ranking_chrpos_df.sort_values("node_index").reset_index(drop=True)
    top_indices = (top_df["node_index"].to_numpy() - 1).astype(int)
    rng = np.random.default_rng(SEED)
    summary_records = []
    window_match_tables = []

    for label, ref_df in reference_sets.items():
        exact_count = int(top_df["chr_pos"].isin(set(ref_df["chr_pos"])).sum())
        exact_reference_hits = int(ref_df[ref_df["chr_pos"].isin(set(top_df["chr_pos"]))]["SNP"].nunique())
        exact_mask = marker_hit_mask(all_coords, ref_df, 0)
        random_counts = np.empty(RANDOM_SETS, dtype=np.int16)
        for random_index in range(len(random_counts)):
            chosen = rng.choice(len(all_coords), size=TOP_K, replace=False)
            random_counts[random_index] = int(exact_mask[chosen].sum())
        exact_p = (1 + np.sum(random_counts >= exact_count)) / (len(random_counts) + 1)
        summary_records.append(
            {
                "comparison": label,
                "window_bp": 0,
                "top_k": TOP_K,
                "reference_snp_count": int(ref_df["SNP"].nunique()),
                "observed_marker_hits": exact_count,
                "observed_reference_snp_hits": exact_reference_hits,
                "random_mean_marker_hits": float(random_counts.mean()),
                "random_std_marker_hits": float(random_counts.std(ddof=1)),
                "empirical_p_ge_observed": float(exact_p),
            }
        )

        for window_bp in [100_000, 1_000_000]:
            match_df = window_matches(top_df, ref_df, label, window_bp)
            if not match_df.empty:
                window_match_tables.append(match_df)

            observed_marker_hits = int(match_df["marker_id"].nunique()) if not match_df.empty else 0
            observed_reference_hits = int(match_df["zeng_snp"].nunique()) if not match_df.empty else 0
            mask = marker_hit_mask(all_coords, ref_df, window_bp)
            random_counts = np.empty(RANDOM_SETS, dtype=np.int16)
            for random_index in range(len(random_counts)):
                chosen = rng.choice(len(all_coords), size=TOP_K, replace=False)
                random_counts[random_index] = int(mask[chosen].sum())
            empirical_p = (1 + np.sum(random_counts >= observed_marker_hits)) / (len(random_counts) + 1)
            summary_records.append(
                {
                    "comparison": label,
                    "window_bp": window_bp,
                    "top_k": TOP_K,
                    "reference_snp_count": int(ref_df["SNP"].nunique()),
                    "observed_marker_hits": observed_marker_hits,
                    "observed_reference_snp_hits": observed_reference_hits,
                    "random_mean_marker_hits": float(random_counts.mean()),
                    "random_std_marker_hits": float(random_counts.std(ddof=1)),
                    "empirical_p_ge_observed": float(empirical_p),
                }
            )

    window_matches_df = pd.concat(window_match_tables, ignore_index=True) if window_match_tables else pd.DataFrame()
    window_matches_df.to_csv(WINDOW_MATCH_PATH, index=False)
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    nearest_rows = []
    for _, marker in top_df.iterrows():
        same_chr = zeng_df[zeng_df["chr"].eq(int(marker["chr"]))].copy()
        if same_chr.empty:
            nearest_rows.append(
                {
                    "rank": int(marker["rank"]),
                    "node_index": int(marker["node_index"]),
                    "marker_id": str(marker["marker_id"]),
                    "model_chr_pos": marker["chr_pos"],
                    "nearest_zeng_trait": None,
                    "nearest_zeng_snp": None,
                    "nearest_zeng_chr_pos": None,
                    "nearest_distance_bp": np.nan,
                }
            )
            continue
        same_chr["distance_bp"] = (same_chr["position"].astype(int) - int(marker["position"])).abs()
        nearest = same_chr.sort_values(["distance_bp", "Trait", "SNP"], kind="stable").iloc[0]
        nearest_rows.append(
            {
                "rank": int(marker["rank"]),
                "node_index": int(marker["node_index"]),
                "marker_id": str(marker["marker_id"]),
                "model_chr_pos": marker["chr_pos"],
                "nearest_zeng_trait": nearest["Trait"],
                "nearest_zeng_snp": nearest["SNP"],
                "nearest_zeng_chr_pos": nearest["chr_pos"],
                "nearest_distance_bp": int(nearest["distance_bp"]),
            }
        )
    nearest_df = pd.DataFrame(nearest_rows)
    nearest_df.to_csv(NEAREST_PATH, index=False)

    print("Chromosome-position top-500 evaluation summary:")
    print(summary_df.to_string(index=False))
    print("\nExact chr:pos matches:")
    if exact_matches_df.empty:
        print("No exact chr:pos matches between the GNN top 500 and Zeng 2022 SNPs.")
    else:
        print(exact_matches_df.to_string(index=False))
    print("\nFirst 10 window matches:")
    if window_matches_df.empty:
        print("No window matches found.")
    else:
        print(window_matches_df.head(10).to_string(index=False))
    print("\nFirst 10 nearest Zeng SNPs for GNN top markers:")
    print(nearest_df.head(10).to_string(index=False))
    print("\nFiles written:")
    for path in [
        RANKING_CHRPOS_PATH,
        TOP_CHRPOS_PATH,
        ZENG_CHRPOS_PATH,
        EXACT_MATCH_PATH,
        WINDOW_MATCH_PATH,
        NEAREST_PATH,
        SUMMARY_PATH,
    ]:
        print(path.name)


if __name__ == "__main__":
    main()
