#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from gnn_model import (
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    MAE_WEIGHT,
    MICRO_BATCH_SIZE,
    MSE_WEIGHT,
    NUM_HEADS,
    NUM_LAYERS,
    VAL_FRACTION,
    WEIGHT_DECAY,
    run_gnn_experiment,
)


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "all_marker_cluster_analysis"

SEED = 42
TOP_K_VALUES = [200, 500, 1000]
MAIZE_CLUSTER_IDS = [0, 1, 2, 3]
MAIZE_TOP_N = 800
MAIZE_RANDOM_N = 200
SOYNAM_CLUSTER_COUNT = 2
KMEANS_N_INIT = 50

MAIZE_TRAIT_PATH = BASE_DIR / "train_trait.csv"
MAIZE_GENOTYPE_PATH = BASE_DIR / "genotype_reduced.csv"
MAIZE_ASSIGNMENT_PATH = BASE_DIR / "weather_kmeans_best_5feature_assignments_labeled.csv"
MAIZE_ENV_VECTOR_PATH = BASE_DIR / "weather_kmeans_best_5feature_all_clusters_lstm_env_vectors.csv"

SOYNAM_DIR = BASE_DIR / "soynam_arch_a_general_6env_nasa"
SOYNAM_TRAIT_PATH = SOYNAM_DIR / "soynam_6env_nasa_final_full.csv"
SOYNAM_GENOTYPE_PATH = SOYNAM_DIR / "soynam_6env_nasa_genotype.csv"
SOYNAM_WEATHER_PATH = SOYNAM_DIR / "soynam_6env_nasa_env_cov.csv"
SOYNAM_ENV_VECTOR_PATH = SOYNAM_DIR / "soynam_6env_nasa_lstm_env_vectors.csv"
SOYNAM_WEATHER_FEATURES = [
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "T2M",
    "PRECTOTCORR",
    "WS2M",
    "QV2M",
]


@dataclass
class ClusterInput:
    dataset: str
    cluster: int
    cluster_dir: Path
    env_names: list[str]
    env_path: Path
    trait_path: Path
    selected_genotypes_path: Path | None
    source_trait_rows: int
    source_genotypes: int
    source_pairs: int
    marker_count: int
    selection_description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Architecture A cluster rankings with all markers retained before training."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["maize", "soynam"],
        default=["maize", "soynam"],
        help="Datasets to run. Default: maize and soynam.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Architecture A epochs per cluster. Default: {EPOCHS}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain a cluster even when its ranking and history already exist.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse existing cluster ranking outputs and regenerate tables and figures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sources and print the planned runs without writing outputs or training.",
    )
    parser.add_argument(
        "--use-env-residual",
        action="store_true",
        help="Train on environment-residual yield instead of raw yield.",
    )
    return parser.parse_args()


def require_two_gpus() -> list[str]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This runner is configured to require both GPUs and will not fall back to CPU."
        )
    device_count = torch.cuda.device_count()
    if device_count < 2:
        raise RuntimeError(
            f"Expected two visible GPUs but found {device_count}. "
            "Launch with CUDA_VISIBLE_DEVICES=0,1 and verify the GPU driver."
        )
    names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    print(f"CUDA visible devices: {device_count} | {names}", flush=True)
    return names


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def marker_columns(genotype_df: pd.DataFrame, source: Path) -> list[str]:
    if genotype_df.shape[1] < 2:
        raise ValueError(f"{source} does not contain marker columns.")
    return list(genotype_df.columns[1:])


def vector_columns(env_df: pd.DataFrame, source: Path) -> list[str]:
    require_columns(env_df, ["Env"], source)
    columns = [column for column in env_df.columns if str(column).startswith("lstm_vector_")]
    if not columns:
        columns = [column for column in env_df.columns if column != "Env"]
    if not columns:
        raise ValueError(f"{source} does not contain environment-vector columns.")
    return columns


def stable_cluster_labels(raw_labels: np.ndarray) -> dict[int, int]:
    counts = pd.Series(raw_labels).value_counts()
    ordered_labels = sorted(counts.index, key=lambda label: (-counts[label], int(label)))
    return {int(label): index for index, label in enumerate(ordered_labels)}


def sanitize_marker_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def output_prefix(cluster_input: ClusterInput) -> str:
    relative_dir = cluster_input.cluster_dir.relative_to(BASE_DIR)
    return str(relative_dir / f"arch_a_all_markers_cluster{cluster_input.cluster}")


def output_paths(cluster_input: ClusterInput) -> dict[str, Path]:
    prefix = BASE_DIR / output_prefix(cluster_input)
    return {
        "model": Path(f"{prefix}_model.pt"),
        "history": Path(f"{prefix}_training_history.csv"),
        "predictions": Path(f"{prefix}_predictions.csv"),
        "weights": Path(f"{prefix}_all_weights.csv.gz"),
        "ranking": Path(f"{prefix}_marker_ranking.csv"),
        "baselines": Path(f"{prefix}_baselines.csv"),
    }


def select_top_and_random_genotypes(valid_trait_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hybrid_summary = (
        valid_trait_df.groupby("Hybrid", as_index=False)
        .agg(avg_yield=("Yield_Mg_ha", "mean"), appearances=("Yield_Mg_ha", "size"))
        .sort_values(["avg_yield", "Hybrid"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    needed = MAIZE_TOP_N + MAIZE_RANDOM_N
    if len(hybrid_summary) < needed:
        raise ValueError(
            f"Need {needed} valid maize hybrids for cluster selection, found {len(hybrid_summary)}."
        )

    top_df = hybrid_summary.head(MAIZE_TOP_N).copy()
    top_df["selection_group"] = f"top_{MAIZE_TOP_N}"
    random_df = hybrid_summary.iloc[MAIZE_TOP_N:].sample(n=MAIZE_RANDOM_N, random_state=SEED).copy()
    random_df["selection_group"] = f"random_{MAIZE_RANDOM_N}"
    selected = pd.concat([top_df, random_df], ignore_index=True)
    selected_hybrids = set(selected["Hybrid"].astype(str))
    candidate = valid_trait_df[valid_trait_df["Hybrid"].astype(str).isin(selected_hybrids)].copy()
    candidate = candidate[["Env", "Hybrid", "Yield_Mg_ha"]].sort_values(
        ["Env", "Hybrid", "Yield_Mg_ha"], kind="stable"
    ).reset_index(drop=True)
    return selected, candidate


def prepare_maize_inputs(root: Path) -> tuple[list[ClusterInput], pd.DataFrame]:
    root.mkdir(parents=True, exist_ok=True)
    trait = pd.read_csv(MAIZE_TRAIT_PATH)
    genotype = pd.read_csv(MAIZE_GENOTYPE_PATH, usecols=[0])
    assignments = pd.read_csv(MAIZE_ASSIGNMENT_PATH)
    env_vectors = pd.read_csv(MAIZE_ENV_VECTOR_PATH)

    require_columns(trait, ["Env", "Hybrid", "Yield_Mg_ha"], MAIZE_TRAIT_PATH)
    require_columns(assignments, ["Env", "cluster"], MAIZE_ASSIGNMENT_PATH)
    vector_cols = vector_columns(env_vectors, MAIZE_ENV_VECTOR_PATH)

    genotype_id_column = genotype.columns[0]
    valid_hybrids = set(genotype[genotype_id_column].astype(str))
    trait = trait.copy()
    trait["Env"] = trait["Env"].astype(str)
    trait["Hybrid"] = trait["Hybrid"].astype(str)
    trait["Yield_Mg_ha"] = pd.to_numeric(trait["Yield_Mg_ha"], errors="coerce")
    trait = trait[trait["Yield_Mg_ha"].notna()].copy()
    assignments = assignments.copy()
    assignments["Env"] = assignments["Env"].astype(str)
    env_vectors = env_vectors.copy()
    env_vectors["Env"] = env_vectors["Env"].astype(str)

    full_genotype = pd.read_csv(MAIZE_GENOTYPE_PATH, nrows=1)
    full_marker_count = len(marker_columns(full_genotype, MAIZE_GENOTYPE_PATH))
    if full_marker_count != 4050:
        raise ValueError(
            f"Expected 4,050 maize reduced markers in {MAIZE_GENOTYPE_PATH.name}, found {full_marker_count}."
        )

    inputs: list[ClusterInput] = []
    summary_rows: list[dict[str, Any]] = []
    for cluster in MAIZE_CLUSTER_IDS:
        cluster_dir = root / f"cluster_{cluster}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        cluster_assignments = assignments[assignments["cluster"].eq(cluster)].copy()
        if cluster_assignments.empty:
            raise ValueError(f"No maize environments found for existing cluster {cluster}.")
        env_names = sorted(cluster_assignments["Env"].unique().tolist())
        cluster_vectors = env_vectors[env_vectors["Env"].isin(env_names)][["Env", *vector_cols]].copy()
        if len(cluster_vectors) != len(env_names):
            missing = sorted(set(env_names) - set(cluster_vectors["Env"]))
            raise ValueError(f"Missing maize LSTM vectors for cluster {cluster}: {missing}")
        cluster_vectors = cluster_vectors.sort_values("Env", kind="stable").reset_index(drop=True)

        valid_trait = trait[
            trait["Env"].isin(env_names) & trait["Hybrid"].isin(valid_hybrids)
        ].copy().reset_index(drop=True)
        selected, candidate = select_top_and_random_genotypes(valid_trait)

        cluster_assignments.to_csv(cluster_dir / "weather_cluster_assignments.csv", index=False)
        cluster_vectors.to_csv(cluster_dir / "environment_vectors.csv", index=False)
        selected.to_csv(cluster_dir / "selected_genotypes_top800_random200.csv", index=False)
        candidate.to_csv(cluster_dir / "trait_records_top800_random200.csv", index=False)

        inputs.append(
            ClusterInput(
                dataset="maize",
                cluster=cluster,
                cluster_dir=cluster_dir,
                env_names=env_names,
                env_path=cluster_dir / "environment_vectors.csv",
                trait_path=cluster_dir / "trait_records_top800_random200.csv",
                selected_genotypes_path=cluster_dir / "selected_genotypes_top800_random200.csv",
                source_trait_rows=len(candidate),
                source_genotypes=candidate["Hybrid"].nunique(),
                source_pairs=candidate[["Hybrid", "Env"]].drop_duplicates().shape[0],
                marker_count=full_marker_count,
                selection_description="cluster-specific top-800 mean yield plus random-200 remaining valid hybrids",
            )
        )
        summary_rows.append(
            {
                "dataset": "maize",
                "cluster": cluster,
                "environment_count": len(env_names),
                "trait_rows": len(candidate),
                "selected_genotypes": candidate["Hybrid"].nunique(),
                "unique_genotype_environment_pairs": candidate[["Hybrid", "Env"]].drop_duplicates().shape[0],
                "marker_count": full_marker_count,
                "genotype_selection": "top800_random200_within_cluster",
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("cluster", kind="stable")
    summary.to_csv(root / "dataset_cluster_summary.csv", index=False)
    return inputs, summary


def prepare_soynam_inputs(root: Path) -> tuple[list[ClusterInput], pd.DataFrame, pd.DataFrame]:
    root.mkdir(parents=True, exist_ok=True)
    trait = pd.read_csv(SOYNAM_TRAIT_PATH)
    genotype = pd.read_csv(SOYNAM_GENOTYPE_PATH, usecols=[0])
    weather = pd.read_csv(SOYNAM_WEATHER_PATH)
    env_vectors = pd.read_csv(SOYNAM_ENV_VECTOR_PATH)

    require_columns(trait, ["Env", "Hybrid", "Yield_Mg_ha"], SOYNAM_TRAIT_PATH)
    require_columns(weather, ["ENV", *SOYNAM_WEATHER_FEATURES], SOYNAM_WEATHER_PATH)
    vector_cols = vector_columns(env_vectors, SOYNAM_ENV_VECTOR_PATH)

    trait = trait.copy()
    trait["Env"] = trait["Env"].astype(str)
    trait["Hybrid"] = trait["Hybrid"].astype(str)
    trait["Yield_Mg_ha"] = pd.to_numeric(trait["Yield_Mg_ha"], errors="coerce")
    trait = trait[trait["Yield_Mg_ha"].notna()].copy()
    weather = weather.copy()
    weather["ENV"] = weather["ENV"].astype(str)
    for feature in SOYNAM_WEATHER_FEATURES:
        weather[feature] = pd.to_numeric(weather[feature], errors="coerce")
    if weather[SOYNAM_WEATHER_FEATURES].isna().any().any():
        missing = weather[SOYNAM_WEATHER_FEATURES].isna().sum()
        raise ValueError(f"SoyNAM weather has missing/non-numeric features:\n{missing[missing.gt(0)]}")
    env_vectors = env_vectors.copy()
    env_vectors["Env"] = env_vectors["Env"].astype(str)

    env_means = (
        weather.groupby("ENV", as_index=False)[SOYNAM_WEATHER_FEATURES]
        .mean()
        .rename(columns={"ENV": "Env"})
        .sort_values("Env", kind="stable")
        .reset_index(drop=True)
    )
    scaled = StandardScaler().fit_transform(env_means[SOYNAM_WEATHER_FEATURES])
    silhouette_rows: list[dict[str, Any]] = []
    for k in [2, 3]:
        raw = KMeans(n_clusters=k, random_state=SEED, n_init=KMEANS_N_INIT).fit_predict(scaled)
        silhouette_rows.append({"k": k, "silhouette_score": float(silhouette_score(scaled, raw))})
    silhouette_table = pd.DataFrame(silhouette_rows)
    silhouette_table.to_csv(root / "weather_kmeans_silhouette_scan.csv", index=False)

    kmeans = KMeans(n_clusters=SOYNAM_CLUSTER_COUNT, random_state=SEED, n_init=KMEANS_N_INIT)
    raw_labels = kmeans.fit_predict(scaled)
    label_map = stable_cluster_labels(raw_labels)
    assignments = env_means.copy()
    assignments["raw_kmeans_label"] = raw_labels
    assignments["cluster"] = pd.Series(raw_labels).map(label_map).astype(int).to_numpy()
    daily_counts = weather.groupby("ENV").size().rename("daily_weather_records").reset_index()
    assignments = assignments.merge(daily_counts, left_on="Env", right_on="ENV", how="left").drop(columns="ENV")
    assignments = assignments.sort_values(["cluster", "Env"], kind="stable").reset_index(drop=True)
    assignments.to_csv(root / "weather_kmeans_k2_assignments.csv", index=False)

    full_genotype = pd.read_csv(SOYNAM_GENOTYPE_PATH, nrows=1)
    full_marker_count = len(marker_columns(full_genotype, SOYNAM_GENOTYPE_PATH))
    if full_marker_count != 4611:
        raise ValueError(
            f"Expected 4,611 SoyNAM markers in {SOYNAM_GENOTYPE_PATH.name}, found {full_marker_count}."
        )
    valid_hybrids = set(genotype[genotype.columns[0]].astype(str))

    inputs: list[ClusterInput] = []
    summary_rows: list[dict[str, Any]] = []
    for cluster in range(SOYNAM_CLUSTER_COUNT):
        cluster_dir = root / f"cluster_{cluster}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        cluster_assignments = assignments[assignments["cluster"].eq(cluster)].copy()
        env_names = sorted(cluster_assignments["Env"].unique().tolist())
        cluster_vectors = env_vectors[env_vectors["Env"].isin(env_names)][["Env", *vector_cols]].copy()
        if len(cluster_vectors) != len(env_names):
            missing = sorted(set(env_names) - set(cluster_vectors["Env"]))
            raise ValueError(f"Missing SoyNAM LSTM vectors for cluster {cluster}: {missing}")
        cluster_vectors = cluster_vectors.sort_values("Env", kind="stable").reset_index(drop=True)
        candidate = trait[
            trait["Env"].isin(env_names) & trait["Hybrid"].isin(valid_hybrids)
        ].copy()
        candidate = candidate[["Env", "Hybrid", "Yield_Mg_ha"]].sort_values(
            ["Env", "Hybrid", "Yield_Mg_ha"], kind="stable"
        ).reset_index(drop=True)

        cluster_assignments.to_csv(cluster_dir / "weather_cluster_assignments.csv", index=False)
        weather[weather["ENV"].isin(env_names)].sort_values(["ENV", "DOY"], kind="stable").to_csv(
            cluster_dir / "weather_records.csv", index=False
        )
        cluster_vectors.to_csv(cluster_dir / "environment_vectors.csv", index=False)
        candidate.to_csv(cluster_dir / "trait_records_all_genotypes.csv", index=False)

        inputs.append(
            ClusterInput(
                dataset="soynam",
                cluster=cluster,
                cluster_dir=cluster_dir,
                env_names=env_names,
                env_path=cluster_dir / "environment_vectors.csv",
                trait_path=cluster_dir / "trait_records_all_genotypes.csv",
                selected_genotypes_path=None,
                source_trait_rows=len(candidate),
                source_genotypes=candidate["Hybrid"].nunique(),
                source_pairs=candidate[["Hybrid", "Env"]].drop_duplicates().shape[0],
                marker_count=full_marker_count,
                selection_description="all available SoyNAM genotypes",
            )
        )
        summary_rows.append(
            {
                "dataset": "soynam",
                "cluster": cluster,
                "environment_count": len(env_names),
                "trait_rows": len(candidate),
                "selected_genotypes": candidate["Hybrid"].nunique(),
                "unique_genotype_environment_pairs": candidate[["Hybrid", "Env"]].drop_duplicates().shape[0],
                "marker_count": full_marker_count,
                "genotype_selection": "all_available_genotypes",
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("cluster", kind="stable")
    summary.to_csv(root / "dataset_cluster_summary.csv", index=False)
    return inputs, summary, silhouette_table


def run_cluster_model(
    cluster_input: ClusterInput,
    epochs: int,
    force: bool,
    skip_training: bool,
    use_env_residual: bool,
) -> dict[str, Path]:
    paths = output_paths(cluster_input)
    required = [paths["history"], paths["ranking"], paths["baselines"]]
    available = all(path.exists() for path in required)
    if skip_training:
        if not available:
            missing = [str(path) for path in required if not path.exists()]
            raise FileNotFoundError(
                f"Cannot skip training for {cluster_input.dataset} cluster {cluster_input.cluster}; missing {missing}"
            )
        print(f"Reusing existing {cluster_input.dataset} cluster {cluster_input.cluster}", flush=True)
        return paths
    if available and not force:
        print(f"Skipping completed {cluster_input.dataset} cluster {cluster_input.cluster}", flush=True)
        return paths

    genotype_path = MAIZE_GENOTYPE_PATH if cluster_input.dataset == "maize" else SOYNAM_GENOTYPE_PATH
    print(
        f"Training {cluster_input.dataset} cluster {cluster_input.cluster} | "
        f"envs={len(cluster_input.env_names)} rows={cluster_input.source_trait_rows} "
        f"markers={cluster_input.marker_count}",
        flush=True,
    )
    run_gnn_experiment(
        final_path=cluster_input.trait_path,
        genotype_path=genotype_path,
        env_path=cluster_input.env_path,
        output_prefix=output_prefix(cluster_input),
        use_env_residual=use_env_residual,
        epochs=epochs,
        return_weight_dataframe=False,
        seed=SEED,
        split_seed=SEED,
        save_predictions=True,
        save_all_weights=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return paths


def summarize_training(cluster_inputs: list[ClusterInput]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_input in cluster_inputs:
        paths = output_paths(cluster_input)
        history = pd.read_csv(paths["history"])
        ranking = pd.read_csv(paths["ranking"])
        baselines = pd.read_csv(paths["baselines"])
        best = history.sort_values(["val_mse", "epoch"], kind="stable").iloc[0]
        row: dict[str, Any] = {
            "dataset": cluster_input.dataset,
            "cluster": cluster_input.cluster,
            "environment_count": len(cluster_input.env_names),
            "environments": "; ".join(cluster_input.env_names),
            "trait_rows": cluster_input.source_trait_rows,
            "genotype_count": cluster_input.source_genotypes,
            "unique_genotype_environment_pairs": cluster_input.source_pairs,
            "marker_count": cluster_input.marker_count,
            "genotype_selection": cluster_input.selection_description,
            "best_epoch": int(best["epoch"]),
            "best_val_mse": float(best["val_mse"]),
            "best_val_mae": float(best["val_mae"]),
            "best_val_r2": float(best["val_r2"]),
            "ranking_marker_count": int(len(ranking)),
        }
        for _, baseline in baselines.iterrows():
            name = str(baseline["baseline"])
            for metric in ["mse", "mae", "r2"]:
                if metric in baseline.index:
                    row[f"baseline_{name}_{metric}"] = float(baseline[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster", kind="stable").reset_index(drop=True)


def load_rankings(cluster_inputs: list[ClusterInput]) -> dict[int, pd.DataFrame]:
    rankings: dict[int, pd.DataFrame] = {}
    for cluster_input in cluster_inputs:
        ranking = pd.read_csv(output_paths(cluster_input)["ranking"]).copy()
        require_columns(ranking, ["marker_id", "rank", "avg_weight"], output_paths(cluster_input)["ranking"])
        ranking["marker_id"] = sanitize_marker_ids(ranking["marker_id"])
        if ranking["marker_id"].duplicated().any():
            raise ValueError(
                f"Duplicate marker IDs found in {cluster_input.dataset} cluster {cluster_input.cluster} ranking."
            )
        rankings[cluster_input.cluster] = ranking.sort_values("rank", kind="stable").reset_index(drop=True)
    return rankings


def membership_table(rankings: dict[int, pd.DataFrame], top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    clusters = sorted(rankings)
    top_tables = {cluster: rankings[cluster].head(top_k).copy() for cluster in clusters}
    top_sets = {cluster: set(table["marker_id"]) for cluster, table in top_tables.items()}
    union = sorted(set().union(*top_sets.values()))
    rows: list[dict[str, Any]] = []
    for marker_id in union:
        present = [cluster for cluster in clusters if marker_id in top_sets[cluster]]
        row: dict[str, Any] = {
            "marker_id": marker_id,
            "membership_count": len(present),
            "membership_clusters": " + ".join(f"C{cluster}" for cluster in present),
        }
        if len(present) == len(clusters):
            row["marker_class"] = "shared_by_all_clusters"
        elif len(present) == 1:
            row["marker_class"] = "unique_to_one_cluster"
        else:
            row["marker_class"] = "partially_shared"
        for cluster in clusters:
            row[f"in_C{cluster}"] = cluster in present
            if cluster in present:
                source_row = top_tables[cluster].set_index("marker_id").loc[marker_id]
                row[f"rank_C{cluster}"] = int(source_row["rank"])
                row[f"avg_weight_C{cluster}"] = float(source_row["avg_weight"])
            else:
                row[f"rank_C{cluster}"] = np.nan
                row[f"avg_weight_C{cluster}"] = np.nan
        rows.append(row)
    membership = pd.DataFrame(rows).sort_values(
        ["membership_count", "marker_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)

    combination = (
        membership.groupby(["membership_clusters", "membership_count"], as_index=False)
        .size()
        .rename(columns={"size": "marker_count"})
        .sort_values(["membership_count", "marker_count", "membership_clusters"], ascending=[False, False, True], kind="stable")
        .reset_index(drop=True)
    )
    return membership, combination


def build_pairwise_overlap(rankings: dict[int, pd.DataFrame], top_k: int) -> pd.DataFrame:
    clusters = sorted(rankings)
    marker_sets = {
        cluster: set(rankings[cluster].head(top_k)["marker_id"].astype(str)) for cluster in clusters
    }
    rows: list[dict[str, Any]] = []
    for cluster_a, cluster_b in combinations(clusters, 2):
        intersection = marker_sets[cluster_a] & marker_sets[cluster_b]
        union = marker_sets[cluster_a] | marker_sets[cluster_b]
        rows.append(
            {
                "top_k": top_k,
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "intersection_marker_count": len(intersection),
                "intersection_pct_of_top_k": 100.0 * len(intersection) / top_k,
                "union_marker_count": len(union),
                "jaccard": len(intersection) / len(union) if union else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_upset_style(combinations_df: pd.DataFrame, clusters: list[int], title: str, output_path: Path) -> None:
    ordered = combinations_df.sort_values(
        ["membership_count", "marker_count", "membership_clusters"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    row_count = len(ordered)
    height = max(4.5, 0.48 * row_count + 1.8)
    figure = plt.figure(figsize=(13, height), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    bar_ax = figure.add_subplot(grid[0, 0])
    matrix_ax = figure.add_subplot(grid[0, 1])
    y = np.arange(row_count)
    colors = ["#2a9d8f" if count == len(clusters) else "#e76f51" if count == 1 else "#457b9d" for count in ordered["membership_count"]]
    bar_ax.barh(y, ordered["marker_count"], color=colors)
    bar_ax.set_yticks(y, ordered["membership_clusters"])
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel("Number of markers")
    bar_ax.set_title(title)
    for position, value in enumerate(ordered["marker_count"]):
        bar_ax.text(value, position, f" {int(value)}", va="center", fontsize=9)
    max_value = max(ordered["marker_count"].max(), 1)
    bar_ax.set_xlim(0, max_value * 1.12)

    matrix_ax.set_xlim(-0.5, len(clusters) - 0.5)
    matrix_ax.set_ylim(row_count - 0.5, -0.5)
    matrix_ax.set_xticks(range(len(clusters)), [f"C{cluster}" for cluster in clusters])
    matrix_ax.set_yticks(y, [""] * row_count)
    matrix_ax.set_title("Cluster membership")
    matrix_ax.grid(False)
    for row_index, row in ordered.iterrows():
        members = {
            int(part.strip().replace("C", ""))
            for part in str(row["membership_clusters"]).split("+")
        }
        positions = [index for index, cluster in enumerate(clusters) if cluster in members]
        matrix_ax.scatter(range(len(clusters)), [row_index] * len(clusters), s=28, color="#d3d3d3", zorder=2)
        matrix_ax.scatter(positions, [row_index] * len(positions), s=52, color="#1d3557", zorder=3)
        if len(positions) > 1:
            matrix_ax.plot([min(positions), max(positions)], [row_index, row_index], color="#1d3557", linewidth=1.8, zorder=1)
    matrix_ax.spines[["top", "right", "left"]].set_visible(False)
    matrix_ax.tick_params(axis="y", length=0)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_pairwise_heatmap(pairwise: pd.DataFrame, clusters: list[int], title: str, output_path: Path) -> None:
    matrix = np.full((len(clusters), len(clusters)), 100.0, dtype=float)
    index = {cluster: position for position, cluster in enumerate(clusters)}
    for _, row in pairwise.iterrows():
        a = index[int(row["cluster_a"])]
        b = index[int(row["cluster_b"])]
        matrix[a, b] = matrix[b, a] = float(row["intersection_pct_of_top_k"])
    figure, axis = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100)
    axis.set_xticks(range(len(clusters)), [f"C{cluster}" for cluster in clusters])
    axis.set_yticks(range(len(clusters)), [f"C{cluster}" for cluster in clusters])
    axis.set_title(title)
    for row in range(len(clusters)):
        for column in range(len(clusters)):
            color = "white" if matrix[row, column] >= 55 else "#111111"
            axis.text(column, row, f"{matrix[row, column]:.1f}%", ha="center", va="center", color=color, fontsize=9)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.84)
    colorbar.set_label("Shared markers (% of top-k)")
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_category_summary(summary: pd.DataFrame, title: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    x = np.arange(len(summary))
    global_counts = summary["shared_by_all_clusters"].to_numpy()
    partial_counts = summary["partially_shared"].to_numpy()
    unique_counts = summary["unique_to_one_cluster"].to_numpy()
    axis.bar(x, global_counts, label="Shared by all", color="#2a9d8f")
    axis.bar(x, partial_counts, bottom=global_counts, label="Partially shared", color="#457b9d")
    axis.bar(x, unique_counts, bottom=global_counts + partial_counts, label="Unique to one cluster", color="#e76f51")
    axis.set_xticks(x, [f"Top {int(value):,}" for value in summary["top_k"]])
    axis.set_ylabel("Number of markers in union of cluster lists")
    axis.set_title(title)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def analyze_marker_overlap(dataset: str, root: Path, cluster_inputs: list[ClusterInput]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rankings = load_rankings(cluster_inputs)
    clusters = sorted(rankings)
    if any(len(ranking) < max(TOP_K_VALUES) for ranking in rankings.values()):
        counts = {cluster: len(ranking) for cluster, ranking in rankings.items()}
        raise ValueError(f"{dataset} rankings are shorter than top {max(TOP_K_VALUES)}: {counts}")

    overlap_rows: list[dict[str, Any]] = []
    cluster_unique_rows: list[dict[str, Any]] = []
    all_pairwise: list[pd.DataFrame] = []
    for top_k in TOP_K_VALUES:
        membership, combinations_df = membership_table(rankings, top_k)
        pairwise = build_pairwise_overlap(rankings, top_k)
        all_pairwise.append(pairwise)

        membership.to_csv(root / f"top{top_k}_marker_membership.csv", index=False)
        combinations_df.to_csv(root / f"top{top_k}_membership_combinations.csv", index=False)
        membership[membership["marker_class"].eq("shared_by_all_clusters")].to_csv(
            root / f"top{top_k}_markers_shared_by_all_clusters.csv", index=False
        )
        membership[membership["marker_class"].eq("partially_shared")].to_csv(
            root / f"top{top_k}_partially_shared_markers.csv", index=False
        )
        for cluster in clusters:
            unique = membership[
                membership["membership_clusters"].eq(f"C{cluster}")
            ].copy()
            unique.to_csv(root / f"top{top_k}_cluster{cluster}_unique_markers.csv", index=False)
            cluster_unique_rows.append(
                {
                    "top_k": top_k,
                    "cluster": cluster,
                    "unique_marker_count": len(unique),
                    "unique_pct_of_top_k": 100.0 * len(unique) / top_k,
                }
            )

        class_counts = membership["marker_class"].value_counts()
        shared_count = int(class_counts.get("shared_by_all_clusters", 0))
        partial_count = int(class_counts.get("partially_shared", 0))
        unique_count = int(class_counts.get("unique_to_one_cluster", 0))
        overlap_rows.append(
            {
                "dataset": dataset,
                "top_k": top_k,
                "cluster_count": len(clusters),
                "shared_by_all_clusters": shared_count,
                "shared_pct_of_top_k": 100.0 * shared_count / top_k,
                "partially_shared": partial_count,
                "unique_to_one_cluster": unique_count,
                "union_marker_count": len(membership),
                "total_top_list_memberships": top_k * len(clusters),
            }
        )
        plot_upset_style(
            combinations_df,
            clusters,
            title=f"{dataset} top {top_k:,} marker overlap",
            output_path=root / f"top{top_k}_upset_overlap.png",
        )
        plot_pairwise_heatmap(
            pairwise,
            clusters,
            title=f"{dataset} top {top_k:,} pairwise overlap",
            output_path=root / f"top{top_k}_pairwise_overlap_heatmap.png",
        )

    overlap_summary = pd.DataFrame(overlap_rows)
    unique_summary = pd.DataFrame(cluster_unique_rows)
    pairwise_summary = pd.concat(all_pairwise, ignore_index=True)
    overlap_summary.to_csv(root / "marker_overlap_summary.csv", index=False)
    unique_summary.to_csv(root / "cluster_unique_marker_summary.csv", index=False)
    pairwise_summary.to_csv(root / "pairwise_marker_overlap_summary.csv", index=False)
    plot_category_summary(
        overlap_summary,
        title=f"{dataset} marker membership by top-k threshold",
        output_path=root / "marker_membership_category_summary.png",
    )
    return overlap_summary, unique_summary, pairwise_summary


def write_manifest(root: Path, args: argparse.Namespace, gpu_names: list[str]) -> None:
    manifest = {
        "analysis": "all_marker_cluster_analysis",
        "purpose": "Rank full marker matrices in weather clusters; classify shared markers after training.",
        "datasets": args.datasets,
        "use_env_residual": args.use_env_residual,
        "target_mode": "env_residual" if args.use_env_residual else "raw_yield",
        "seed": SEED,
        "top_k_values": TOP_K_VALUES,
        "epochs": args.epochs,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"),
        "gpu_names": gpu_names,
        "python": sys.version,
        "platform": platform.platform(),
        "architecture_a": {
            "hidden_dim": HIDDEN_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "epochs": args.epochs,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "mse_weight": MSE_WEIGHT,
            "mae_weight": MAE_WEIGHT,
            "validation_fraction": VAL_FRACTION,
            "split_seed": SEED,
        },
        "maize": {
            "marker_input": str(MAIZE_GENOTYPE_PATH),
            "marker_count_expected": 4050,
            "cluster_source": str(MAIZE_ASSIGNMENT_PATH),
            "genotype_selection": "top800_random200_within_cluster",
        },
        "soynam": {
            "marker_input": str(SOYNAM_GENOTYPE_PATH),
            "marker_count_expected": 4611,
            "weather_cluster_count": SOYNAM_CLUSTER_COUNT,
            "weather_features": SOYNAM_WEATHER_FEATURES,
        },
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")

    results_dir = BASE_DIR / "all_marker_cluster_analysis_envresidual" if args.use_env_residual else RESULTS_DIR

    gpu_names: list[str] = []
    if not args.dry_run:
        gpu_names = require_two_gpus()

    if args.dry_run:
        print("Dry run: validating input sources only; no output files will be written.", flush=True)
        maize_root = results_dir / "maize"
        soy_root = results_dir / "soynam"
        if "maize" in args.datasets:
            for path in [MAIZE_TRAIT_PATH, MAIZE_GENOTYPE_PATH, MAIZE_ASSIGNMENT_PATH, MAIZE_ENV_VECTOR_PATH]:
                if not path.exists():
                    raise FileNotFoundError(path)
            print("Maize: four weather clusters, all 4,050 markers, top-800/random-200 genotype selection.")
        if "soynam" in args.datasets:
            for path in [SOYNAM_TRAIT_PATH, SOYNAM_GENOTYPE_PATH, SOYNAM_WEATHER_PATH, SOYNAM_ENV_VECTOR_PATH]:
                if not path.exists():
                    raise FileNotFoundError(path)
            print("SoyNAM: K=2 weather clusters, all 4,611 markers, all available genotypes.")
        print(f"Planned output roots: {maize_root} and {soy_root}")
        print(f"Target mode: {'env_residual' if args.use_env_residual else 'raw_yield'}")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(results_dir, args, gpu_names)
    print(f"Results directory: {results_dir}", flush=True)
    print(f"Top-k values: {TOP_K_VALUES}", flush=True)
    print(f"Target mode: {'env_residual' if args.use_env_residual else 'raw_yield'}", flush=True)

    maize_cluster_summary = maize_training = maize_overlap = maize_unique = None
    soy_cluster_summary = soy_training = soy_overlap = soy_unique = soy_silhouette = None

    if "maize" in args.datasets:
        maize_root = results_dir / "maize"
        maize_root.mkdir(parents=True, exist_ok=True)
        maize_inputs, maize_cluster_summary = prepare_maize_inputs(maize_root)
        for cluster_input in maize_inputs:
            run_cluster_model(cluster_input, args.epochs, args.force, args.skip_training, args.use_env_residual)
        maize_training = summarize_training(maize_inputs)
        maize_training.to_csv(maize_root / "training_summary.csv", index=False)
        maize_overlap, maize_unique, _ = analyze_marker_overlap("maize", maize_root, maize_inputs)

    if "soynam" in args.datasets:
        soy_root = results_dir / "soynam"
        soy_root.mkdir(parents=True, exist_ok=True)
        soy_inputs, soy_cluster_summary, soy_silhouette = prepare_soynam_inputs(soy_root)
        for cluster_input in soy_inputs:
            run_cluster_model(cluster_input, args.epochs, args.force, args.skip_training, args.use_env_residual)
        soy_training = summarize_training(soy_inputs)
        soy_training.to_csv(soy_root / "training_summary.csv", index=False)
        soy_overlap, soy_unique, _ = analyze_marker_overlap("soynam", soy_root, soy_inputs)

    print("Analysis complete.", flush=True)


if __name__ == "__main__":
    main()
