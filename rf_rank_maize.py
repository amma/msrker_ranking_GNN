from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent.parent
GENOTYPE_PATH = BASE_DIR / "genotype_reduced.csv"
ENV_PATH = BASE_DIR / "train_env_vectors.csv"
TRAIN_PHENO_PATH = BASE_DIR / "final_genotype_1000_split80.csv"
VAL_PHENO_PATH = BASE_DIR / "final_genotype_1000_split20.csv"
REFERENCE_PATH = BASE_DIR / "external_gwas_reference_union_chrpos.csv"

RF_RANKING_PATH = BASE_DIR / "rf_feature_importance_maize_split80_marker_ranking.csv"
RF_RANKING_CHRPOS_PATH = BASE_DIR / "rf_feature_importance_maize_split80_marker_ranking_chrpos.csv"
RF_METRICS_PATH = BASE_DIR / "rf_feature_importance_maize_split80_prediction_metrics.csv"
GWAS_SUMMARY_PATH = BASE_DIR / "gwas_overlap_method_comparison_maize_split80_summary.csv"
GWAS_MATCHES_PATH = BASE_DIR / "gwas_overlap_method_comparison_maize_split80_matches.csv"
GWAS_WINDOW_TABLE_PATH = BASE_DIR / "gwas_overlap_method_comparison_maize_split80_top1000_500kb.csv"

SEED = 42
TOP_K_VALUES = [500, 1000]
WINDOWS_BP = [100_000, 500_000, 1_000_000]


def aggregate_pairs(final_df: pd.DataFrame) -> pd.DataFrame:
    return (
        final_df.groupby(["Hybrid", "Env"], as_index=False)
        .agg(
            mean_yield=("Yield_Mg_ha", "mean"),
            replicate_count=("Yield_Mg_ha", "size"),
        )
        .sort_values(["Hybrid", "Env"], kind="stable")
        .reset_index(drop=True)
    )


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def weighted_pearson_corr(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray) -> float:
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.sum() <= 0:
        return float("nan")
    weights = weights / weights.sum()
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    true_mean = np.sum(weights * y_true)
    pred_mean = np.sum(weights * y_pred)
    true_centered = y_true - true_mean
    pred_centered = y_pred - pred_mean
    cov = np.sum(weights * true_centered * pred_centered)
    true_var = np.sum(weights * true_centered * true_centered)
    pred_var = np.sum(weights * pred_centered * pred_centered)
    denom = np.sqrt(true_var * pred_var)
    if denom <= 0:
        return float("nan")
    return float(cov / denom)


def parse_marker_position(marker_name: object) -> int:
    match = re.match(r"^(\d+)", str(marker_name))
    if not match:
        raise ValueError(f"Cannot parse marker position from marker name: {marker_name}")
    return int(match.group(1))


def infer_marker_coordinates(marker_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    chromosome = 1
    previous_position: int | None = None
    for node_index, marker_name in enumerate(marker_columns, start=1):
        position = parse_marker_position(marker_name)
        if previous_position is not None and position < previous_position:
            chromosome += 1
        rows.append(
            {
                "node_index": node_index,
                "marker_id": str(marker_name).replace(".0", ""),
                "chr": chromosome,
                "position": position,
                "chr_pos": f"chr{chromosome}:{position}",
            }
        )
        previous_position = position
    return pd.DataFrame(rows)


def build_reference_sets(reference_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    zeng = reference_df[reference_df["source"].eq("Zeng2022")].copy()
    tolley = reference_df[reference_df["source"].eq("Tolley2023")].copy()
    ma_cao = reference_df[reference_df["source"].eq("MaCao2021")].copy()
    zeng_gyp = zeng[zeng["trait"].eq("GYP")].copy()
    ma_cao_gyp = ma_cao[ma_cao["trait"].eq("GYP")].copy()
    combined_direct = pd.concat([zeng_gyp, tolley, ma_cao_gyp], ignore_index=True)
    combined_all = pd.concat([zeng, tolley, ma_cao], ignore_index=True)

    reference_sets = {
        "Zeng2022_GYP_only": zeng_gyp,
        "Tolley2023_G2F_yield": tolley,
        "MaCao2021_GYP_only": ma_cao_gyp,
        "Combined_direct_yield": combined_direct,
        "Combined_all_yield_related": combined_all,
    }
    out: dict[str, pd.DataFrame] = {}
    for key, value in reference_sets.items():
        tmp = value.copy()
        tmp["reference_key"] = tmp["source"].astype(str) + ":" + tmp["snp_id"].astype(str)
        out[key] = tmp.drop_duplicates(["reference_key"]).reset_index(drop=True)
    return out


def marker_hit_mask(all_markers: pd.DataFrame, reference: pd.DataFrame, window_bp: int) -> np.ndarray:
    all_chr = all_markers["chr"].to_numpy(dtype=int)
    all_pos = all_markers["position"].to_numpy(dtype=int)
    mask = np.zeros(len(all_markers), dtype=bool)
    for _, ref_row in reference.iterrows():
        mask |= (all_chr == int(ref_row["chr"])) & (np.abs(all_pos - int(ref_row["position"])) <= window_bp)
    return mask


def window_matches(
    method_name: str,
    top_markers: pd.DataFrame,
    reference_set: str,
    reference: pd.DataFrame,
    top_k: int,
    window_bp: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, marker in top_markers.iterrows():
        same_chr = reference[reference["chr"].eq(int(marker["chr"]))].copy()
        if same_chr.empty:
            continue
        same_chr["distance_bp"] = (same_chr["position"].astype(int) - int(marker["position"])).abs()
        same_chr = same_chr[same_chr["distance_bp"].le(window_bp)]
        for _, ref_row in same_chr.iterrows():
            rows.append(
                {
                    "method": method_name,
                    "reference_set": reference_set,
                    "top_k": top_k,
                    "window_bp": window_bp,
                    "rank": int(marker["rank"]),
                    "node_index": int(marker["node_index"]),
                    "marker_id": str(marker["marker_id"]).replace(".0", ""),
                    "model_chr_pos": marker["chr_pos"],
                    "score": float(marker["score"]),
                    "reference_source": ref_row["source"],
                    "reference_trait": ref_row["trait"],
                    "reference_snp": ref_row["snp_id"],
                    "reference_chr_pos": ref_row["chr_pos"],
                    "distance_bp": int(ref_row["distance_bp"]),
                }
            )
    return rows


def build_method_ranking_chrpos(
    ranking_df: pd.DataFrame,
    coord_df: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    out = ranking_df.copy()
    out["marker_id"] = out["marker_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    out = out.drop(columns=[column for column in ["chr", "position", "chr_pos"] if column in out.columns])
    if "node_index" not in out.columns:
        out = out.merge(coord_df[["marker_id", "node_index"]], on="marker_id", how="left")
    out = out.merge(coord_df[["node_index", "chr", "position", "chr_pos"]], on="node_index", how="left")
    if out[["chr", "position", "chr_pos"]].isna().any().any():
        raise ValueError("Could not map one or more markers to chr:pos.")
    out = out.rename(columns={score_column: "score"})
    return out[["rank", "node_index", "marker_id", "score", "chr", "position", "chr_pos"]].sort_values("rank", kind="stable").reset_index(drop=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    genotype_df = pd.read_csv(GENOTYPE_PATH).copy()
    env_df = pd.read_csv(ENV_PATH).copy()
    train_df = pd.read_csv(TRAIN_PHENO_PATH).copy()
    val_df = pd.read_csv(VAL_PHENO_PATH).copy()

    genotype_df = genotype_df.rename(columns={genotype_df.columns[0]: "Hybrid"})
    env_df = env_df.rename(columns={env_df.columns[0]: "Env"})

    for frame in [genotype_df, env_df, train_df, val_df]:
        if "Hybrid" in frame.columns:
            frame["Hybrid"] = frame["Hybrid"].astype(str)
        if "Env" in frame.columns:
            frame["Env"] = frame["Env"].astype(str)

    train_pair_df = aggregate_pairs(train_df)
    val_pair_df = aggregate_pairs(val_df)

    marker_columns = [column for column in genotype_df.columns if column != "Hybrid"]
    env_feature_columns = [column for column in env_df.columns if column != "Env"]
    return genotype_df, env_df, train_pair_df, val_pair_df, marker_columns, env_feature_columns


def build_feature_matrix(
    pair_df: pd.DataFrame,
    genotype_lookup: pd.DataFrame,
    env_lookup: pd.DataFrame,
    marker_columns: list[str],
    env_feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged = (
        pair_df.merge(genotype_lookup[["Hybrid"] + marker_columns], on="Hybrid", how="inner")
        .merge(env_lookup[["Env"] + env_feature_columns], on="Env", how="inner")
        .copy()
    )
    x_markers = merged[marker_columns].to_numpy(dtype=np.float32, copy=True)
    x_env = merged[env_feature_columns].to_numpy(dtype=np.float32, copy=True)
    x = np.concatenate([x_markers, x_env], axis=1)
    y = merged["mean_yield"].to_numpy(dtype=np.float32, copy=True)
    w = merged["replicate_count"].to_numpy(dtype=np.float32, copy=True)
    return x, y, w


def train_rf_ranking() -> tuple[pd.DataFrame, pd.DataFrame]:
    genotype_df, env_df, train_pair_df, val_pair_df, marker_columns, env_feature_columns = load_data()
    genotype_lookup = genotype_df.drop_duplicates("Hybrid").reset_index(drop=True)
    env_lookup = env_df.drop_duplicates("Env").reset_index(drop=True)

    x_train, y_train, w_train = build_feature_matrix(
        train_pair_df,
        genotype_lookup,
        env_lookup,
        marker_columns,
        env_feature_columns,
    )
    x_val, y_val, w_val = build_feature_matrix(
        val_pair_df,
        genotype_lookup,
        env_lookup,
        marker_columns,
        env_feature_columns,
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    y_pred = model.predict(x_val).astype(np.float32, copy=False)

    importances = model.feature_importances_[: len(marker_columns)]
    ranking_df = pd.DataFrame(
        {
            "marker_id": [str(marker).replace(".0", "") for marker in marker_columns],
            "rf_importance": importances.astype(float),
        }
    )
    ranking_df = ranking_df.sort_values(["rf_importance", "marker_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))

    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "maize_split20_holdout",
                "pcc": pearson_corr(y_val, y_pred),
                "weighted_pcc": weighted_pearson_corr(y_val, y_pred, w_val),
                "r2": float(r2_score(y_val, y_pred)),
                "weighted_r2": float(r2_score(y_val, y_pred, sample_weight=w_val)),
                "mse": float(mean_squared_error(y_val, y_pred)),
                "weighted_mse": float(mean_squared_error(y_val, y_pred, sample_weight=w_val)),
                "mae": float(mean_absolute_error(y_val, y_pred)),
                "weighted_mae": float(mean_absolute_error(y_val, y_pred, sample_weight=w_val)),
                "n_pairs": int(len(y_val)),
                "marker_feature_count": int(len(marker_columns)),
                "env_feature_count": int(len(env_feature_columns)),
            }
        ]
    )
    return ranking_df, metrics_df


def compare_gwas_overlap(coord_df: pd.DataFrame, rf_ranking_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_df = pd.read_csv(REFERENCE_PATH).copy()
    reference_sets = build_reference_sets(reference_df)

    gnn_ranking = pd.read_csv(BASE_DIR / "arch_a_multienv_1000_split80_raw_marker_ranking.csv").copy()
    corr_ranking = pd.read_csv(BASE_DIR / "corr_baseline_maize.ranking.csv").copy()

    rf_chrpos = build_method_ranking_chrpos(rf_ranking_df, coord_df, "rf_importance")
    gnn_chrpos = build_method_ranking_chrpos(gnn_ranking, coord_df, "avg_weight")
    corr_chrpos = build_method_ranking_chrpos(corr_ranking, coord_df, "corr_abs")

    rf_chrpos.to_csv(RF_RANKING_CHRPOS_PATH, index=False)

    methods = {
        "GNN": gnn_chrpos,
        "RF": rf_chrpos,
        "Correlation": corr_chrpos,
    }

    summary_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []

    for method_name, ranking_df in methods.items():
        all_markers = ranking_df.sort_values("node_index").reset_index(drop=True)
        for reference_set, reference in reference_sets.items():
            reference = reference.reset_index(drop=True)
            reference_count = int(reference["reference_key"].nunique())
            for top_k in TOP_K_VALUES:
                top_markers = ranking_df.head(top_k).copy()
                for window_bp in WINDOWS_BP:
                    matches = window_matches(
                        method_name=method_name,
                        top_markers=top_markers,
                        reference_set=reference_set,
                        reference=reference,
                        top_k=top_k,
                        window_bp=window_bp,
                    )
                    match_rows.extend(matches)
                    match_df = pd.DataFrame(matches)
                    observed_marker_hits = int(match_df["marker_id"].nunique()) if not match_df.empty else 0
                    observed_reference_hits = int(match_df["reference_snp"].nunique()) if not match_df.empty else 0
                    mask = marker_hit_mask(all_markers, reference, window_bp)
                    summary_rows.append(
                        {
                            "method": method_name,
                            "reference_set": reference_set,
                            "top_k": top_k,
                            "window_bp": window_bp,
                            "reference_snp_count": reference_count,
                            "observed_marker_hits": observed_marker_hits,
                            "observed_reference_snp_hits": observed_reference_hits,
                            "reference_snp_coverage_pct": round(100 * observed_reference_hits / max(reference_count, 1), 2),
                            "top_marker_hit_pct": round(100 * observed_marker_hits / top_k, 2),
                            "all_marker_window_hit_count": int(mask.sum()),
                        }
                    )

    summary_df = pd.DataFrame(summary_rows).sort_values(["reference_set", "top_k", "window_bp", "method"], kind="stable").reset_index(drop=True)
    matches_df = pd.DataFrame(match_rows)
    window_table = (
        summary_df[
            summary_df["top_k"].eq(1000)
            & summary_df["window_bp"].eq(500_000)
            & summary_df["reference_set"].isin(
                ["Zeng2022_GYP_only", "Tolley2023_G2F_yield", "MaCao2021_GYP_only", "Combined_direct_yield"]
            )
        ][["reference_set", "method", "observed_reference_snp_hits", "reference_snp_count", "reference_snp_coverage_pct"]]
        .sort_values(["reference_set", "method"], kind="stable")
        .reset_index(drop=True)
    )
    return summary_df, matches_df, window_table


def main() -> None:
    ranking_df, metrics_df = train_rf_ranking()
    ranking_df.to_csv(RF_RANKING_PATH, index=False)
    metrics_df.to_csv(RF_METRICS_PATH, index=False)

    marker_columns = [column for column in pd.read_csv(GENOTYPE_PATH, nrows=0).columns[1:]]
    coord_df = infer_marker_coordinates(marker_columns)
    rf_chrpos = build_method_ranking_chrpos(ranking_df, coord_df, "rf_importance")
    rf_chrpos.to_csv(RF_RANKING_CHRPOS_PATH, index=False)

    summary_df, matches_df, window_table = compare_gwas_overlap(coord_df, ranking_df)
    summary_df.to_csv(GWAS_SUMMARY_PATH, index=False)
    matches_df.to_csv(GWAS_MATCHES_PATH, index=False)
    window_table.to_csv(GWAS_WINDOW_TABLE_PATH, index=False)

    print("Saved:")
    print(RF_RANKING_PATH)
    print(RF_RANKING_CHRPOS_PATH)
    print(RF_METRICS_PATH)
    print(GWAS_SUMMARY_PATH)
    print(GWAS_MATCHES_PATH)
    print(GWAS_WINDOW_TABLE_PATH)
    print()
    print("RF split20 prediction metrics:")
    print(metrics_df.to_string(index=False))
    print()
    print("Top-1000 / 500kb GWAS overlap comparison:")
    print(window_table.to_string(index=False))


if __name__ == "__main__":
    main()
