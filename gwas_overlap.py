from __future__ import annotations

from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
RANKING_CHRPOS_PATH = BASE_DIR / "arch_a_multienv_1000_marker_ranking_chrpos.csv"

ZENG_URL = "https://link.springer.com/article/10.1186/s12870-022-03812-5/tables/2"
TOLLEY_URL = "https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2023.1221751/full"
MA_CAO_URL = "https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2021.690059/full"

ZENG_TABLE_PATH = BASE_DIR / "zeng_2022_yield_related_gwas_table2.csv"
TOLLEY_TABLE_PATH = BASE_DIR / "tolley_2023_g2f_yield_snps_table3.csv"
MA_CAO_TABLE_PATH = BASE_DIR / "ma_cao_2021_yield_related_snps_table1.csv"

SUMMARY_PATH = BASE_DIR / "external_gwas_reference_overlap_summary.csv"
MATCHES_PATH = BASE_DIR / "external_gwas_reference_window_matches.csv"
REFERENCE_UNION_PATH = BASE_DIR / "external_gwas_reference_union_chrpos.csv"

TOP_K_VALUES = [500, 1000]
WINDOWS_BP = [0, 100_000, 500_000, 1_000_000]
RANDOM_SETS = 20_000
SEED = 42


def parse_snp_chr_pos(snp: object) -> tuple[int, int]:
    text = str(snp).strip()
    match = re.match(r"^S?(\d+)[_:](\d+)$", text)
    if not match:
        raise ValueError(f"Cannot parse SNP coordinate from {snp!r}")
    return int(match.group(1)), int(match.group(2).replace(",", ""))


def add_chr_pos_columns(df: pd.DataFrame, snp_col: str) -> pd.DataFrame:
    out = df.copy()
    parsed = out[snp_col].apply(lambda value: pd.Series(parse_snp_chr_pos(value)))
    out["chr"] = parsed[0].astype(int)
    out["position"] = parsed[1].astype(int)
    out["chr_pos"] = "chr" + out["chr"].astype(str) + ":" + out["position"].astype(str)
    return out


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(part) for part in col if not str(part).startswith("Unnamed")).strip("_")
            for col in out.columns
        ]
    return out


def load_zeng_reference() -> pd.DataFrame:
    if ZENG_TABLE_PATH.exists():
        zeng = pd.read_csv(ZENG_TABLE_PATH)
    else:
        html = requests.get(ZENG_URL, timeout=60).text
        zeng = pd.read_html(StringIO(html))[0]
        zeng.to_csv(ZENG_TABLE_PATH, index=False)

    zeng = zeng.rename(
        columns={
            "SNP": "snp_id",
            "Trait": "trait",
            "Candidate gene": "candidate_gene",
            "Gene annotation": "annotation",
        }
    )
    if "chr" not in zeng.columns or "position" not in zeng.columns:
        zeng = add_chr_pos_columns(zeng, "snp_id")
    else:
        zeng["chr"] = zeng["chr"].astype(int)
        zeng["position"] = zeng["position"].astype(int)
        zeng["chr_pos"] = "chr" + zeng["chr"].astype(str) + ":" + zeng["position"].astype(str)
    zeng["source"] = "Zeng2022"
    zeng["source_detail"] = "BMC Plant Biology Table 2"
    zeng["trait"] = zeng["trait"].astype(str)
    return zeng[["source", "source_detail", "snp_id", "trait", "chr", "position", "chr_pos", "candidate_gene", "annotation"]]


def load_tolley_reference() -> pd.DataFrame:
    if TOLLEY_TABLE_PATH.exists():
        tolley = pd.read_csv(TOLLEY_TABLE_PATH)
    else:
        html = requests.get(TOLLEY_URL, timeout=60).text
        tables = pd.read_html(StringIO(html))
        tolley = flatten_columns(tables[2])
        tolley.to_csv(TOLLEY_TABLE_PATH, index=False)

    tolley = tolley.rename(
        columns={
            "SNP_ID": "snp_id",
            "Candidate genes": "candidate_gene",
            "Annotation": "annotation",
        }
    )
    tolley = add_chr_pos_columns(tolley, "snp_id")
    tolley["source"] = "Tolley2023"
    tolley["source_detail"] = "Frontiers in Genetics Table 3, G2F reaction norm yield SNPs"
    tolley["trait"] = "grain_yield"
    return tolley[["source", "source_detail", "snp_id", "trait", "chr", "position", "chr_pos", "candidate_gene", "annotation"]]


def load_ma_cao_reference() -> pd.DataFrame:
    if MA_CAO_TABLE_PATH.exists():
        ma_cao = pd.read_csv(MA_CAO_TABLE_PATH)
    else:
        html = requests.get(MA_CAO_URL, timeout=60).text
        tables = pd.read_html(StringIO(html))
        ma_cao = flatten_columns(tables[0])
        ma_cao.to_csv(MA_CAO_TABLE_PATH, index=False)

    ma_cao = ma_cao.rename(
        columns={
            "SNP name*": "snp_id",
            "Trait§": "trait",
            "Candidate gene": "candidate_gene",
        }
    )
    ma_cao = add_chr_pos_columns(ma_cao, "snp_id")
    ma_cao["source"] = "MaCao2021"
    ma_cao["source_detail"] = "Frontiers in Plant Science Table 1"
    ma_cao["annotation"] = ""
    return ma_cao[["source", "source_detail", "snp_id", "trait", "chr", "position", "chr_pos", "candidate_gene", "annotation"]]


def unique_reference_rows(ref: pd.DataFrame) -> pd.DataFrame:
    out = ref.copy()
    out["source"] = out["source"].astype(str)
    out["trait"] = out["trait"].astype(str)
    out["snp_id"] = out["snp_id"].astype(str)
    return out.drop_duplicates(["source", "snp_id"]).reset_index(drop=True)


def marker_hit_mask(all_markers: pd.DataFrame, reference: pd.DataFrame, window_bp: int) -> np.ndarray:
    marker_chr = all_markers["chr"].to_numpy(dtype=int)
    marker_pos = all_markers["position"].to_numpy(dtype=int)
    mask = np.zeros(len(all_markers), dtype=bool)
    for _, hit in reference.iterrows():
        mask |= (marker_chr == int(hit["chr"])) & (np.abs(marker_pos - int(hit["position"])) <= window_bp)
    return mask


def window_matches(top_markers: pd.DataFrame, reference: pd.DataFrame, label: str, top_k: int, window_bp: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, marker in top_markers.iterrows():
        same_chr = reference[reference["chr"].eq(int(marker["chr"]))].copy()
        if same_chr.empty:
            continue
        same_chr["distance_bp"] = (same_chr["position"].astype(int) - int(marker["position"])).abs()
        same_chr = same_chr[same_chr["distance_bp"].le(window_bp)]
        for _, hit in same_chr.iterrows():
            rows.append(
                {
                    "reference_set": label,
                    "top_k": top_k,
                    "window_bp": window_bp,
                    "rank": int(marker["rank"]),
                    "node_index": int(marker["node_index"]),
                    "marker_id": str(marker["marker_id"]).replace(".0", ""),
                    "model_chr_pos": marker["chr_pos"],
                    "model_avg_weight": float(marker["avg_weight"]),
                    "reference_source": hit["source"],
                    "reference_trait": hit["trait"],
                    "reference_snp": hit["snp_id"],
                    "reference_chr_pos": hit["chr_pos"],
                    "distance_bp": int(hit["distance_bp"]),
                    "candidate_gene": hit.get("candidate_gene", ""),
                    "annotation": hit.get("annotation", ""),
                }
            )
    return rows


def build_reference_sets(zeng: pd.DataFrame, tolley: pd.DataFrame, ma_cao: pd.DataFrame) -> dict[str, pd.DataFrame]:
    zeng_unique = unique_reference_rows(zeng)
    tolley_unique = unique_reference_rows(tolley)
    ma_cao_unique = unique_reference_rows(ma_cao)
    ma_cao_gyp = unique_reference_rows(ma_cao[ma_cao["trait"].eq("GYP")])
    zeng_gyp = unique_reference_rows(zeng[zeng["trait"].eq("GYP")])

    combined_yield = unique_reference_rows(pd.concat([zeng_gyp, tolley_unique, ma_cao_gyp], ignore_index=True))
    combined_all = unique_reference_rows(pd.concat([zeng_unique, tolley_unique, ma_cao_unique], ignore_index=True))

    return {
        "Zeng2022_GYP_only": zeng_gyp,
        "Zeng2022_all_yield_related": zeng_unique,
        "Tolley2023_G2F_yield": tolley_unique,
        "MaCao2021_GYP_only": ma_cao_gyp,
        "MaCao2021_all_yield_related": ma_cao_unique,
        "Combined_direct_yield": combined_yield,
        "Combined_all_yield_related": combined_all,
    }


def main() -> None:
    ranking = pd.read_csv(RANKING_CHRPOS_PATH).copy()
    ranking["marker_id"] = ranking["marker_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    ranking = ranking.sort_values("rank").reset_index(drop=True)
    all_markers = ranking.sort_values("node_index").reset_index(drop=True)

    zeng = load_zeng_reference()
    tolley = load_tolley_reference()
    ma_cao = load_ma_cao_reference()
    all_reference_rows = unique_reference_rows(pd.concat([zeng, tolley, ma_cao], ignore_index=True))
    all_reference_rows.to_csv(REFERENCE_UNION_PATH, index=False)

    reference_sets = build_reference_sets(zeng, tolley, ma_cao)

    rng = np.random.default_rng(SEED)
    summary_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []

    for label, reference in reference_sets.items():
        reference = reference.reset_index(drop=True)
        for top_k in TOP_K_VALUES:
            top_markers = ranking.head(top_k).copy()
            for window_bp in WINDOWS_BP:
                matches = window_matches(top_markers, reference, label, top_k, window_bp)
                match_rows.extend(matches)

                match_df = pd.DataFrame(matches)
                observed_marker_hits = int(match_df["marker_id"].nunique()) if not match_df.empty else 0
                observed_reference_hits = int(match_df["reference_snp"].nunique()) if not match_df.empty else 0

                hit_mask = marker_hit_mask(all_markers, reference, window_bp)
                random_counts = np.empty(RANDOM_SETS, dtype=np.int16)
                for random_index in range(RANDOM_SETS):
                    chosen = rng.choice(len(all_markers), size=top_k, replace=False)
                    random_counts[random_index] = int(hit_mask[chosen].sum())
                empirical_p = (1 + np.sum(random_counts >= observed_marker_hits)) / (RANDOM_SETS + 1)

                summary_rows.append(
                    {
                        "reference_set": label,
                        "top_k": top_k,
                        "window_bp": window_bp,
                        "reference_snp_count": int(reference["snp_id"].nunique()),
                        "observed_marker_hits": observed_marker_hits,
                        "observed_reference_snp_hits": observed_reference_hits,
                        "reference_snp_coverage_pct": round(100 * observed_reference_hits / max(int(reference["snp_id"].nunique()), 1), 2),
                        "top_marker_hit_pct": round(100 * observed_marker_hits / top_k, 2),
                        "random_mean_marker_hits": float(random_counts.mean()),
                        "random_std_marker_hits": float(random_counts.std(ddof=1)),
                        "empirical_p_ge_observed": float(empirical_p),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    matches = pd.DataFrame(match_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    matches.to_csv(MATCHES_PATH, index=False)

    print("Saved:")
    print(SUMMARY_PATH)
    print(MATCHES_PATH)
    print(REFERENCE_UNION_PATH)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
