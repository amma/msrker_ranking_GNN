from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent.parent
SOY_DIR = BASE_DIR / "soynam_arch_a_general_6env_nasa"

GENOTYPE_PATH = SOY_DIR / "soynam_6env_nasa_genotype.csv"
ENV_PATH = SOY_DIR / "soynam_6env_nasa_lstm_env_vectors.csv"
TRAIN_PHENO_PATH = SOY_DIR / "soynam_6env_nasa_final_split80.csv"
VAL_PHENO_PATH = SOY_DIR / "soynam_6env_nasa_final_split20.csv"
REFERENCE_PATH = SOY_DIR / "soynam_6env_nasa_diers2018_yield_mta_reference.csv"

RF_RANKING_PATH = SOY_DIR / "rf_feature_importance_soynam_split80_marker_ranking.csv"
RF_METRICS_PATH = SOY_DIR / "rf_feature_importance_soynam_split80_prediction_metrics.csv"
SUMMARY_PATH = SOY_DIR / "gwas_overlap_method_comparison_soynam_split80_summary.csv"
MATCHES_PATH = SOY_DIR / "gwas_overlap_method_comparison_soynam_split80_matches.csv"
WINDOW_TABLE_PATH = SOY_DIR / "gwas_overlap_method_comparison_soynam_split80_top1000_500kb.csv"

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


def marker_hits_for_window(marker_df: pd.DataFrame, refs: pd.DataFrame, window_bp: int) -> tuple[np.ndarray, np.ndarray]:
    marker_hit = np.zeros(len(marker_df), dtype=bool)
    ref_hit = np.zeros(len(refs), dtype=bool)
    marker_chr = marker_df["chr"].to_numpy(dtype=int)
    marker_pos = marker_df["position"].to_numpy(dtype=int)
    ref_chr = refs["chr"].to_numpy(dtype=int)
    ref_pos = refs["position"].to_numpy(dtype=int)
    for ref_idx in range(len(refs)):
        same_chr = marker_chr == ref_chr[ref_idx]
        within = np.abs(marker_pos - ref_pos[ref_idx]) <= window_bp
        hits = same_chr & within
        if hits.any():
            ref_hit[ref_idx] = True
            marker_hit |= hits
    return marker_hit, ref_hit


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    genotype_df = pd.read_csv(GENOTYPE_PATH).copy()
    env_df = pd.read_csv(ENV_PATH).copy()
    train_df = pd.read_csv(TRAIN_PHENO_PATH).copy()
    val_df = pd.read_csv(VAL_PHENO_PATH).copy()

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
    x_train, y_train, w_train = build_feature_matrix(train_pair_df, genotype_df, env_df, marker_columns, env_feature_columns)
    x_val, y_val, w_val = build_feature_matrix(val_pair_df, genotype_df, env_df, marker_columns, env_feature_columns)

    model = RandomForestRegressor(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    y_pred = model.predict(x_val).astype(np.float32, copy=False)

    ranking_df = pd.DataFrame(
        {
            "marker_id": marker_columns,
            "rf_importance": model.feature_importances_[: len(marker_columns)].astype(float),
        }
    )
    ranking_df = ranking_df.sort_values(["rf_importance", "marker_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))

    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "soynam_split20_holdout",
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


def compare_gwas_overlap(rf_ranking_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    refs = pd.read_csv(REFERENCE_PATH).copy()
    refs = refs[refs["chr"].notna() & refs["position"].notna()].copy()
    refs["chr"] = refs["chr"].astype(int)
    refs["position"] = refs["position"].astype(int)

    gnn = pd.read_csv(SOY_DIR / "soynam_6env_nasa_split80_marker_ranking_chrpos.csv").copy()
    corr = pd.read_csv(BASE_DIR / "corr_baseline_soynam.ranking.csv").copy()
    rf = pd.read_csv(SOY_DIR / "soynam_6env_nasa_genotype.csv", nrows=0)
    marker_columns = [column for column in rf.columns if column != "Hybrid"]
    node_index_lookup = pd.DataFrame({"marker_id": marker_columns, "node_index": np.arange(1, len(marker_columns) + 1)})
    rf_chrpos = rf_ranking_df.merge(node_index_lookup, on="marker_id", how="left")
    rf_chrpos = rf_chrpos.merge(
        pd.read_csv(SOY_DIR / "soynam_6env_nasa_full_marker_ranking_chrpos.csv")[["node_index", "chr", "position", "chr_pos"]].drop_duplicates("node_index"),
        on="node_index",
        how="left",
    )
    rf_chrpos = rf_chrpos.rename(columns={"rf_importance": "score"})[["rank", "node_index", "marker_id", "score", "chr", "position", "chr_pos"]]

    methods = {
        "GNN": gnn.rename(columns={"avg_weight": "score"})[["rank", "node_index", "marker_id", "score", "chr", "position", "chr_pos"]],
        "RF": rf_chrpos,
        "Correlation": corr.rename(columns={"corr_abs": "score"})[["rank", "marker_id", "score", "chr", "position", "chr_pos"]],
    }

    summary_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []

    for method_name, ranking_df in methods.items():
        ranking_df = ranking_df.copy()
        ranking_df["marker_id"] = ranking_df["marker_id"].astype(str)
        if "node_index" not in ranking_df.columns:
            ranking_df["node_index"] = np.arange(1, len(ranking_df) + 1)
        ranking_df["chr"] = ranking_df["chr"].astype(int)
        ranking_df["position"] = ranking_df["position"].astype(int)
        marker_universe = ranking_df.reset_index(drop=True)

        for window_bp in WINDOWS_BP:
            universe_marker_hit, _ = marker_hits_for_window(marker_universe, refs, window_bp)
            for top_k in TOP_K_VALUES:
                top_markers = ranking_df.head(top_k).reset_index(drop=True)
                marker_hit, ref_hit = marker_hits_for_window(top_markers, refs, window_bp)
                observed_marker_hits = int(marker_hit.sum())
                observed_ref_hits = int(ref_hit.sum())

                for _, marker in top_markers.loc[marker_hit].iterrows():
                    same_chr_refs = refs[refs["chr"].eq(int(marker["chr"]))].copy()
                    same_chr_refs["distance_bp"] = (same_chr_refs["position"] - int(marker["position"])).abs()
                    for _, ref in same_chr_refs[same_chr_refs["distance_bp"].le(window_bp)].iterrows():
                        match_rows.append(
                            {
                                "method": method_name,
                                "reference_set": "Diers2018_yield",
                                "top_k": top_k,
                                "window_bp": window_bp,
                                "rank": int(marker["rank"]),
                                "marker_id": marker["marker_id"],
                                "marker_chr_pos": marker["chr_pos"],
                                "score": float(marker["score"]),
                                "reference_marker": ref["reference_marker"],
                                "reference_chr_pos": ref["chr_pos"],
                                "distance_bp": int(ref["distance_bp"]),
                            }
                        )

                summary_rows.append(
                    {
                        "method": method_name,
                        "reference_set": "Diers2018_yield",
                        "top_k": top_k,
                        "window_bp": window_bp,
                        "reference_snp_count": int(len(refs)),
                        "observed_marker_hits": observed_marker_hits,
                        "observed_reference_snp_hits": observed_ref_hits,
                        "reference_snp_coverage_pct": round(100 * observed_ref_hits / len(refs), 2),
                        "top_marker_hit_pct": round(100 * observed_marker_hits / top_k, 2),
                        "all_marker_window_hit_count": int(universe_marker_hit.sum()),
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["top_k", "window_bp", "method"], kind="stable").reset_index(drop=True)
    matches_df = pd.DataFrame(match_rows)
    window_table = (
        summary_df[
            summary_df["top_k"].eq(1000)
            & summary_df["window_bp"].eq(500_000)
        ][["reference_set", "method", "observed_reference_snp_hits", "reference_snp_count", "reference_snp_coverage_pct"]]
        .sort_values(["reference_set", "method"], kind="stable")
        .reset_index(drop=True)
    )
    return summary_df, matches_df, window_table


def main() -> None:
    ranking_df, metrics_df = train_rf_ranking()
    ranking_df.to_csv(RF_RANKING_PATH, index=False)
    metrics_df.to_csv(RF_METRICS_PATH, index=False)

    summary_df, matches_df, window_table = compare_gwas_overlap(ranking_df)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    matches_df.to_csv(MATCHES_PATH, index=False)
    window_table.to_csv(WINDOW_TABLE_PATH, index=False)

    print("Saved:")
    print(RF_RANKING_PATH)
    print(RF_METRICS_PATH)
    print(SUMMARY_PATH)
    print(MATCHES_PATH)
    print(WINDOW_TABLE_PATH)
    print()
    print("RF split20 prediction metrics:")
    print(metrics_df.to_string(index=False))
    print()
    print("Top-1000 / 500kb GWAS overlap comparison:")
    print(window_table.to_string(index=False))


if __name__ == "__main__":
    main()
