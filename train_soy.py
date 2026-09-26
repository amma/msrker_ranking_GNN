from __future__ import annotations

from io import StringIO
from itertools import combinations
from pathlib import Path
import os
import random
import re
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "impact-ec-main" / "data"
OUT_DIR = BASE_DIR / "soynam_arch_a_general_6env_nasa"

SEED = 42
EPOCHS = 100
FORCE_RERUN = False
TOP_K_VALUES = [200, 500, 1000]
WEATHER_FEATURES = ["T2M_MAX", "T2M_MIN", "RH2M", "T2M", "PRECTOTCORR", "WS2M", "QV2M"]
VECTOR_COLUMNS = [f"lstm_vector_{i}" for i in range(1, 13)]


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def retrieve_power_csv(env_name: str, longitude: float, latitude: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "parameters": ",".join(WEATHER_FEATURES),
        "community": "AG",
        "longitude": str(longitude),
        "latitude": str(latitude),
        "start": start,
        "end": end,
        "format": "CSV",
    }
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")

    data_start = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("YEAR,DOY,"):
            data_start = index
            break
    if data_start is None:
        raise RuntimeError(f"Could not find NASA POWER CSV data header for {env_name}")

    df = pd.read_csv(StringIO("\n".join(lines[data_start:])))
    dates = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    if len(dates) != len(df):
        raise RuntimeError(f"Date count and NASA row count differ for {env_name}: {len(dates)} vs {len(df)}")

    df.insert(1, "MM", dates.month.astype(int))
    df.insert(2, "DD", dates.day.astype(int))
    df.insert(3, "YYYYMMDD", dates.strftime("%Y-%m-%d"))
    df.insert(0, "ENV", env_name)
    df.insert(0, "YearFilt", df["YEAR"].astype(str))
    return df[["YearFilt", "ENV", "YEAR", "MM", "DD", "DOY", "YYYYMMDD", *WEATHER_FEATURES]]


def build_weather_table(weather_raw: pd.DataFrame) -> pd.DataFrame:
    missing_specs = [
        {
            "ENV": "IL_2011",
            "longitude": -88.221499,
            "latitude": 40.054879,
            "start": "20110510",
            "end": "20111114",
        },
        {
            "ENV": "NE_2011",
            "longitude": -98.137824,
            "latitude": 40.575256,
            "start": "20110510",
            "end": "20111114",
        },
    ]
    retrieved = []
    for spec in missing_specs:
        print(f"Retrieving NASA POWER daily weather for {spec['ENV']}")
        retrieved.append(
            retrieve_power_csv(
                env_name=spec["ENV"],
                longitude=spec["longitude"],
                latitude=spec["latitude"],
                start=spec["start"],
                end=spec["end"],
            )
        )

    retrieved_df = pd.concat(retrieved, ignore_index=True)
    retrieved_weather_path = OUT_DIR / "soynam_6env_nasa_retrieved_2011_weather.csv"
    retrieved_df.to_csv(retrieved_weather_path, index=False)

    weather_existing = weather_raw.copy()
    if "Unnamed: 0" in weather_existing.columns:
        weather_existing = weather_existing.drop(columns=["Unnamed: 0"])
    weather_existing = weather_existing[["YearFilt", "ENV", "YEAR", "MM", "DD", "DOY", "YYYYMMDD", *WEATHER_FEATURES]]
    combined = pd.concat([weather_existing, retrieved_df], ignore_index=True)
    combined["ENV"] = combined["ENV"].astype(str)
    combined = combined.sort_values(["ENV", "YEAR", "DOY"], kind="stable").reset_index(drop=True)
    combined_path = OUT_DIR / "soynam_6env_nasa_env_cov.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved retrieved weather: {retrieved_weather_path}")
    print(f"Saved six-environment weather table: {combined_path}")
    return combined


def prepare_trait_table(y_raw: pd.DataFrame, x_raw: pd.DataFrame, weather_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather_envs = set(weather_df["ENV"].astype(str).unique())
    genotype_ids = set(x_raw.iloc[:, 0].astype(str))
    trait_df = (
        y_raw.assign(
            Env=y_raw["environ"].astype(str),
            Hybrid=y_raw["strain"].astype(str),
            Yield_Mg_ha=pd.to_numeric(y_raw["yield"], errors="coerce") / 1000.0,
        )
        .loc[:, ["Env", "Hybrid", "Yield_Mg_ha"]]
    )
    valid_trait_df = trait_df[
        trait_df["Env"].isin(weather_envs)
        & trait_df["Hybrid"].isin(genotype_ids)
        & trait_df["Yield_Mg_ha"].notna()
    ].copy()
    valid_trait_df = valid_trait_df.sort_values(["Env", "Hybrid", "Yield_Mg_ha"], kind="stable").reset_index(drop=True)
    trait_full_path = OUT_DIR / "soynam_6env_nasa_final_full.csv"
    valid_trait_df.to_csv(trait_full_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "dataset": "raw_Y",
                "rows": len(y_raw),
                "unique_hybrids": y_raw["strain"].astype(str).nunique(),
                "unique_envs": y_raw["environ"].astype(str).nunique(),
                "unique_hybrid_env_pairs": y_raw[["strain", "environ"]].drop_duplicates().shape[0],
            },
            {
                "dataset": "valid_weather_and_genotype",
                "rows": len(valid_trait_df),
                "unique_hybrids": valid_trait_df["Hybrid"].nunique(),
                "unique_envs": valid_trait_df["Env"].nunique(),
                "unique_hybrid_env_pairs": valid_trait_df[["Hybrid", "Env"]].drop_duplicates().shape[0],
            },
        ]
    )
    summary_path = OUT_DIR / "soynam_6env_nasa_dataset_summary.csv"
    summary.to_csv(summary_path, index=False)
    return valid_trait_df, summary


def prepare_genotype_table(x_raw: pd.DataFrame, valid_trait_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    genotype_df = x_raw.rename(columns={x_raw.columns[0]: "Hybrid"}).copy()
    genotype_df["Hybrid"] = genotype_df["Hybrid"].astype(str)
    valid_hybrids = set(valid_trait_df["Hybrid"].astype(str))
    genotype_df = genotype_df[genotype_df["Hybrid"].isin(valid_hybrids)].drop_duplicates("Hybrid").copy()
    genotype_df = genotype_df.sort_values("Hybrid", kind="stable").reset_index(drop=True)
    marker_columns = [col for col in genotype_df.columns if col != "Hybrid"]
    missing_before = int(genotype_df[marker_columns].isna().sum().sum())
    for col in marker_columns:
        values = pd.to_numeric(genotype_df[col], errors="coerce")
        mode_values = values.dropna().mode()
        fill_value = float(mode_values.iloc[0]) if not mode_values.empty else 0.0
        genotype_df[col] = values.fillna(fill_value).round().clip(0, 3).astype("uint8")
    missing_after = int(genotype_df[marker_columns].isna().sum().sum())

    genotype_path = OUT_DIR / "soynam_6env_nasa_genotype.csv"
    genotype_df.to_csv(genotype_path, index=False)
    imputation_summary = pd.DataFrame(
        [
            {
                "retained_genotypes": genotype_df["Hybrid"].nunique(),
                "marker_count": len(marker_columns),
                "missing_marker_cells_before_imputation": missing_before,
                "missing_marker_cells_after_imputation": missing_after,
            }
        ]
    )
    imputation_summary_path = OUT_DIR / "soynam_6env_nasa_genotype_imputation_summary.csv"
    imputation_summary.to_csv(imputation_summary_path, index=False)
    return genotype_df, imputation_summary


def build_env_tensor(data: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    env_meta_rows = []
    sequences = []
    for env, env_df in data.groupby("ENV", sort=True):
        env_df = env_df.sort_values("DOY", kind="stable")
        sequence = env_df[feature_columns].to_numpy(dtype=np.float32, copy=True)
        sequences.append(sequence)
        env_meta_rows.append({"Env": env, "record_count": int(len(env_df))})
    max_timesteps = max(sequence.shape[0] for sequence in sequences)
    tensor = np.zeros((len(sequences), max_timesteps, len(feature_columns)), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        tensor[index, : sequence.shape[0], :] = sequence
    return tensor, pd.DataFrame(env_meta_rows)


def build_lstm_env_vectors(weather_df: pd.DataFrame, valid_trait_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input, LSTM, Masking
    from tensorflow.keras.models import Sequential

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception as exc:
        print(f"TensorFlow GPU visibility was already initialized: {exc}")
    tf.random.set_seed(SEED)

    model_weather = weather_df.copy()
    model_weather["ENV"] = model_weather["ENV"].astype(str)
    model_weather = model_weather[model_weather["ENV"].isin(valid_trait_df["Env"].unique())].copy()
    model_weather = model_weather.sort_values(["ENV", "DOY"], kind="stable").reset_index(drop=True)
    for col in WEATHER_FEATURES:
        model_weather[col] = pd.to_numeric(model_weather[col], errors="coerce")
    if model_weather[WEATHER_FEATURES].isna().any().any():
        missing = model_weather[WEATHER_FEATURES].isna().sum()
        raise ValueError(f"Missing/non-numeric weather values:\n{missing[missing.gt(0)]}")

    env_tensor, env_meta = build_env_tensor(model_weather, WEATHER_FEATURES)
    model = Sequential(
        [
            Input(shape=(env_tensor.shape[1], env_tensor.shape[2])),
            Masking(mask_value=0.0),
            LSTM(50, return_sequences=False),
            Dense(27),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    random_targets = np.random.rand(len(env_tensor), 27).astype(np.float32)
    model.fit(env_tensor, random_targets, epochs=1, verbose=0)
    vectors = model.predict(env_tensor, verbose=0).astype(float)
    env_vectors_df = pd.concat([env_meta, pd.DataFrame(vectors, columns=VECTOR_COLUMNS)], axis=1)
    env_vectors_df = env_vectors_df.sort_values("Env", kind="stable").reset_index(drop=True)

    env_vector_path = OUT_DIR / "soynam_6env_nasa_lstm_env_vectors.csv"
    env_vectors_df[["Env", *VECTOR_COLUMNS]].to_csv(env_vector_path, index=False)
    weather_summary = (
        model_weather.groupby("ENV", as_index=False)
        .agg(first_doy=("DOY", "min"), last_doy=("DOY", "max"), record_count=("DOY", "size"))
        .rename(columns={"ENV": "Env"})
    )
    weather_summary_path = OUT_DIR / "soynam_6env_nasa_weather_sequence_summary.csv"
    weather_summary.to_csv(weather_summary_path, index=False)
    return env_vectors_df, weather_summary


def stratified_env_split(df: pd.DataFrame, train_fraction: float = 0.8, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    rng = np.random.default_rng(seed)
    for env, env_df in df.groupby("Env", sort=True):
        env_df = env_df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        train_count = int(round(train_fraction * len(env_df)))
        train_count = min(max(train_count, 1), len(env_df) - 1)
        train_parts.append(env_df.iloc[:train_count].copy())
        test_parts.append(env_df.iloc[train_count:].copy())
    train_part = pd.concat(train_parts, ignore_index=True).sort_values(["Env", "Hybrid", "Yield_Mg_ha"], kind="stable").reset_index(drop=True)
    holdout_part = pd.concat(test_parts, ignore_index=True).sort_values(["Env", "Hybrid", "Yield_Mg_ha"], kind="stable").reset_index(drop=True)
    return train_part, holdout_part


def prepare_splits(valid_trait_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_split_df, holdout_split_df = stratified_env_split(valid_trait_df, train_fraction=0.8, seed=SEED)
    train_split_df.to_csv(OUT_DIR / "soynam_6env_nasa_final_split80.csv", index=False)
    holdout_split_df.to_csv(OUT_DIR / "soynam_6env_nasa_final_split20.csv", index=False)
    split_summary = pd.DataFrame(
        [
            {
                "experiment": "full",
                "rows": len(valid_trait_df),
                "unique_hybrids": valid_trait_df["Hybrid"].nunique(),
                "unique_envs": valid_trait_df["Env"].nunique(),
                "unique_hybrid_env_pairs": valid_trait_df[["Hybrid", "Env"]].drop_duplicates().shape[0],
            },
            {
                "experiment": "split80",
                "rows": len(train_split_df),
                "unique_hybrids": train_split_df["Hybrid"].nunique(),
                "unique_envs": train_split_df["Env"].nunique(),
                "unique_hybrid_env_pairs": train_split_df[["Hybrid", "Env"]].drop_duplicates().shape[0],
            },
            {
                "experiment": "split20",
                "rows": len(holdout_split_df),
                "unique_hybrids": holdout_split_df["Hybrid"].nunique(),
                "unique_envs": holdout_split_df["Env"].nunique(),
                "unique_hybrid_env_pairs": holdout_split_df[["Hybrid", "Env"]].drop_duplicates().shape[0],
            },
        ]
    )
    split_summary.to_csv(OUT_DIR / "soynam_6env_nasa_split_summary.csv", index=False)
    return train_split_df, holdout_split_df, split_summary


def output_paths_for_prefix(prefix: str) -> dict[str, Path]:
    return {
        "history": BASE_DIR / f"{prefix}_training_history.csv",
        "ranking": BASE_DIR / f"{prefix}_marker_ranking.csv",
        "baselines": BASE_DIR / f"{prefix}_baselines.csv",
        "predictions": BASE_DIR / f"{prefix}_predictions.csv",
        "model": BASE_DIR / f"{prefix}_model.pt",
        "weights": BASE_DIR / f"{prefix}_all_weights.csv.gz",
    }


def run_gnn_models() -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, Path]], pd.DataFrame]:
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    import torch
    from gnn_model import run_gnn_experiment

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
    for device_index in range(torch.cuda.device_count()):
        print(f"  cuda:{device_index} - {torch.cuda.get_device_name(device_index)}")

    experiments = {
        "full": {
            "final_path": OUT_DIR / "soynam_6env_nasa_final_full.csv",
            "output_prefix": str((OUT_DIR.relative_to(BASE_DIR) / "soynam_arch_a_general_6env_nasa_full").as_posix()),
        },
        "split80": {
            "final_path": OUT_DIR / "soynam_6env_nasa_final_split80.csv",
            "output_prefix": str((OUT_DIR.relative_to(BASE_DIR) / "soynam_arch_a_general_6env_nasa_split80").as_posix()),
        },
        "split20": {
            "final_path": OUT_DIR / "soynam_6env_nasa_final_split20.csv",
            "output_prefix": str((OUT_DIR.relative_to(BASE_DIR) / "soynam_arch_a_general_6env_nasa_split20").as_posix()),
        },
    }

    genotype_path = OUT_DIR / "soynam_6env_nasa_genotype.csv"
    env_vector_path = OUT_DIR / "soynam_6env_nasa_lstm_env_vectors.csv"

    results = {}
    output_paths = {}
    for name, spec in experiments.items():
        paths = output_paths_for_prefix(spec["output_prefix"])
        output_paths[name] = paths
        if paths["ranking"].exists() and paths["history"].exists() and not FORCE_RERUN:
            print(f"Skipping existing six-env {name}: {paths['ranking']}")
            results[name] = {
                "ranking_df": pd.read_csv(paths["ranking"]),
                "history_df": pd.read_csv(paths["history"]),
                "baseline_df": pd.read_csv(paths["baselines"]) if paths["baselines"].exists() else pd.DataFrame(),
            }
        else:
            print(f"Running six-env Arch A experiment: {name}")
            results[name] = run_gnn_experiment(
                final_path=spec["final_path"],
                genotype_path=genotype_path,
                env_path=env_vector_path,
                output_prefix=spec["output_prefix"],
                use_env_residual=False,
                epochs=EPOCHS,
                return_weight_dataframe=False,
            )

    rows = []
    for name, spec in experiments.items():
        ranking_df = results[name]["ranking_df"].copy()
        history_df = results[name]["history_df"].copy()
        final_df = pd.read_csv(spec["final_path"])
        best = history_df.sort_values(["val_mse", "epoch"], kind="stable").iloc[0]
        rows.append(
            {
                "experiment": name,
                "best_epoch": int(best["epoch"]),
                "best_val_mse": float(best["val_mse"]),
                "best_val_mae": float(best["val_mae"]),
                "best_val_r2": float(best["val_r2"]),
                "min_marker_weight": float(ranking_df["avg_weight"].min()),
                "max_marker_weight": float(ranking_df["avg_weight"].max()),
                "marker_weight_sd": float(ranking_df["avg_weight"].std()),
                "rows": int(len(final_df)),
                "unique_hybrids": int(final_df["Hybrid"].nunique()),
                "unique_envs": int(final_df["Env"].nunique()),
                "unique_hybrid_env_pairs": int(final_df[["Hybrid", "Env"]].drop_duplicates().shape[0]),
            }
        )
    training_summary = pd.DataFrame(rows)
    training_summary.to_csv(OUT_DIR / "soynam_6env_nasa_arch_a_training_summary.csv", index=False)
    return results, output_paths, training_summary


def parse_soy_marker(marker_id: object) -> pd.Series:
    text = str(marker_id)
    match = re.match(r"^Gm(\d+)_(\d+)_", text)
    if not match:
        return pd.Series({"chr": np.nan, "position": np.nan, "chr_pos": np.nan})
    chrom = int(match.group(1))
    position = int(match.group(2))
    return pd.Series({"chr": chrom, "position": position, "chr_pos": f"Gm{chrom:02d}:{position}"})


def save_top_marker_outputs(results: dict[str, dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    ranking_chrpos = {}
    for name, result in results.items():
        ranking = result["ranking_df"].copy()
        ranking["marker_id"] = ranking["marker_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        coords = ranking["marker_id"].apply(parse_soy_marker)
        ranking = pd.concat([ranking, coords], axis=1)
        ranking_chrpos[name] = ranking
        ranking.to_csv(OUT_DIR / f"soynam_6env_nasa_{name}_marker_ranking_chrpos.csv", index=False)
        for top_k in TOP_K_VALUES:
            ranking.head(top_k).to_csv(OUT_DIR / f"soynam_6env_nasa_{name}_top{top_k}_markers.csv", index=False)

    full_top_weights = ranking_chrpos["full"].head(1000)[["rank", "node_index", "marker_id", "avg_weight", "node_label"]]
    full_top_weights.to_csv(OUT_DIR / "soynam_6env_nasa_full_top1000_marker_weights.csv", index=False)
    return ranking_chrpos


def marker_intersections(ranking_chrpos: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    intersection_id_rows = []
    experiment_names = ["full", "split80", "split20"]
    for top_k in TOP_K_VALUES:
        top_tables = {name: ranking_chrpos[name].head(top_k).copy() for name in experiment_names}
        top_sets = {name: set(table["marker_id"].astype(str)) for name, table in top_tables.items()}
        top_by_marker = {name: table.set_index("marker_id") for name, table in top_tables.items()}
        for name_a, name_b in combinations(experiment_names, 2):
            intersection = top_sets[name_a] & top_sets[name_b]
            union = top_sets[name_a] | top_sets[name_b]
            jaccard = len(intersection) / len(union) if union else np.nan
            comparison_rows.append(
                {
                    "top_k": top_k,
                    "ranking_a": name_a,
                    "ranking_b": name_b,
                    "intersection": len(intersection),
                    "intersection_pct_of_top_k": 100 * len(intersection) / top_k,
                    "union": len(union),
                    "jaccard": jaccard,
                    "jaccard_distance": 1 - jaccard if pd.notna(jaccard) else np.nan,
                }
            )
            for marker_id in sorted(intersection):
                row_a = top_by_marker[name_a].loc[marker_id]
                row_b = top_by_marker[name_b].loc[marker_id]
                intersection_id_rows.append(
                    {
                        "top_k": top_k,
                        "ranking_a": name_a,
                        "ranking_b": name_b,
                        "marker_id": marker_id,
                        "rank_a": int(row_a["rank"]),
                        "rank_b": int(row_b["rank"]),
                        "avg_weight_a": float(row_a["avg_weight"]),
                        "avg_weight_b": float(row_b["avg_weight"]),
                        "chr": row_a.get("chr", np.nan),
                        "position": row_a.get("position", np.nan),
                        "chr_pos": row_a.get("chr_pos", np.nan),
                    }
                )
    intersection_summary = pd.DataFrame(comparison_rows)
    intersection_ids = pd.DataFrame(intersection_id_rows)
    intersection_summary.to_csv(OUT_DIR / "soynam_6env_nasa_pairwise_top_marker_intersections.csv", index=False)
    intersection_ids.to_csv(OUT_DIR / "soynam_6env_nasa_pairwise_top_marker_intersection_ids.csv", index=False)
    return intersection_summary, intersection_ids


def plot_marker_intersections(intersection_summary: pd.DataFrame) -> None:
    plot_df = intersection_summary.copy()
    plot_df["comparison"] = plot_df["ranking_a"] + " vs " + plot_df["ranking_b"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    comparisons = plot_df["comparison"].unique().tolist()
    x = np.arange(len(TOP_K_VALUES))
    width = 0.24
    for idx, comparison in enumerate(comparisons):
        subset = plot_df[plot_df["comparison"].eq(comparison)].sort_values("top_k")
        ax.bar(x + (idx - 1) * width, subset["intersection_pct_of_top_k"], width=width, label=comparison)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in TOP_K_VALUES])
    ax.set_xlabel("Top-k marker cutoff")
    ax.set_ylabel("Intersection (% of top-k)")
    ax.set_title("SoyNAM six-env NASA Arch A top-marker overlap")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "soynam_6env_nasa_top_marker_intersection_pct.png", dpi=200)
    plt.close(fig)


def plot_weight_curves(ranking_chrpos: dict[str, pd.DataFrame]) -> None:
    full_top_markers = ranking_chrpos["full"].head(1000).copy()
    mean_weight = float(full_top_markers["avg_weight"].mean())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(full_top_markers["rank"], full_top_markers["avg_weight"], color="#2a6f97", linewidth=1.8)
    ax.axhline(mean_weight, color="#c1440e", linestyle="--", linewidth=1.5, label=f"Mean = {mean_weight:.6g}")
    ax.set_xlabel("Marker rank")
    ax.set_ylabel("Average marker weight")
    ax.set_title("SoyNAM six-env NASA full model top-1000 marker weights")
    ax.set_xticks([1, 200, 400, 600, 800, 1000])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "soynam_6env_nasa_full_top1000_marker_weight_curve.png", dpi=200)
    plt.close(fig)

    maize_path = BASE_DIR / "arch_a_multienv_1000_marker_ranking.csv"
    if maize_path.exists():
        maize = pd.read_csv(maize_path).head(1000)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(full_top_markers["rank"], full_top_markers["avg_weight"], label="SoyNAM six-env NASA", linewidth=1.8)
        ax.plot(maize["rank"], maize["avg_weight"], label="Previous maize full", linewidth=1.8)
        ax.set_xlabel("Marker rank")
        ax.set_ylabel("Average marker weight")
        ax.set_title("Full-model top-1000 marker-weight curves")
        ax.set_xticks([1, 200, 400, 600, 800, 1000])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "soynam_6env_nasa_vs_previous_maize_full_top1000_marker_weight_curve.png", dpi=200)
        plt.close(fig)


def yield_analysis(valid_trait_df: pd.DataFrame, train_split_df: pd.DataFrame, holdout_split_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    y = valid_trait_df.copy()
    overall = pd.DataFrame(
        [
            {
                "dataset": "full_valid_weather_supported",
                "rows": len(y),
                "unique_hybrids": y["Hybrid"].nunique(),
                "unique_envs": y["Env"].nunique(),
                "mean_yield_Mg_ha": y["Yield_Mg_ha"].mean(),
                "sd_yield_Mg_ha": y["Yield_Mg_ha"].std(),
                "min_yield_Mg_ha": y["Yield_Mg_ha"].min(),
                "q25_yield_Mg_ha": y["Yield_Mg_ha"].quantile(0.25),
                "median_yield_Mg_ha": y["Yield_Mg_ha"].median(),
                "q75_yield_Mg_ha": y["Yield_Mg_ha"].quantile(0.75),
                "max_yield_Mg_ha": y["Yield_Mg_ha"].max(),
            }
        ]
    )
    by_env = (
        y.groupby("Env", as_index=False)
        .agg(
            rows=("Yield_Mg_ha", "size"),
            unique_hybrids=("Hybrid", "nunique"),
            mean_yield_Mg_ha=("Yield_Mg_ha", "mean"),
            sd_yield_Mg_ha=("Yield_Mg_ha", "std"),
            min_yield_Mg_ha=("Yield_Mg_ha", "min"),
            median_yield_Mg_ha=("Yield_Mg_ha", "median"),
            max_yield_Mg_ha=("Yield_Mg_ha", "max"),
        )
        .sort_values("mean_yield_Mg_ha", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    by_hybrid = (
        y.groupby("Hybrid", as_index=False)
        .agg(
            environments=("Env", "nunique"),
            mean_yield_Mg_ha=("Yield_Mg_ha", "mean"),
            sd_yield_Mg_ha=("Yield_Mg_ha", "std"),
            min_yield_Mg_ha=("Yield_Mg_ha", "min"),
            max_yield_Mg_ha=("Yield_Mg_ha", "max"),
        )
        .sort_values("mean_yield_Mg_ha", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    by_hybrid.insert(0, "yield_rank", np.arange(1, len(by_hybrid) + 1))
    top_hybrids = by_hybrid.head(20).copy()

    split_rows = []
    for name, df in [("full", y), ("split80", train_split_df), ("split20", holdout_split_df)]:
        split_rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "unique_hybrids": df["Hybrid"].nunique(),
                "unique_envs": df["Env"].nunique(),
                "mean_yield_Mg_ha": df["Yield_Mg_ha"].mean(),
                "sd_yield_Mg_ha": df["Yield_Mg_ha"].std(),
                "median_yield_Mg_ha": df["Yield_Mg_ha"].median(),
                "min_yield_Mg_ha": df["Yield_Mg_ha"].min(),
                "max_yield_Mg_ha": df["Yield_Mg_ha"].max(),
            }
        )
    split_yield_summary = pd.DataFrame(split_rows)

    grand = y["Yield_Mg_ha"].mean()
    total_ss = float(((y["Yield_Mg_ha"] - grand) ** 2).sum())
    env_means = y.groupby("Env")["Yield_Mg_ha"].mean()
    hybrid_means = y.groupby("Hybrid")["Yield_Mg_ha"].mean()
    env_counts = y.groupby("Env")["Yield_Mg_ha"].size()
    hybrid_counts = y.groupby("Hybrid")["Yield_Mg_ha"].size()
    env_ss = float(((env_means - grand) ** 2 * env_counts).sum())
    hybrid_ss = float(((hybrid_means - grand) ** 2 * hybrid_counts).sum())
    residual_ss = float(total_ss - env_ss - hybrid_ss)
    variance_decomp = pd.DataFrame(
        [
            {"source": "environment_main_effect", "sum_squares": env_ss, "pct_total_ss": 100 * env_ss / total_ss},
            {"source": "hybrid_main_effect", "sum_squares": hybrid_ss, "pct_total_ss": 100 * hybrid_ss / total_ss},
            {"source": "residual_or_hybrid_by_env", "sum_squares": residual_ss, "pct_total_ss": 100 * residual_ss / total_ss},
            {"source": "total", "sum_squares": total_ss, "pct_total_ss": 100.0},
        ]
    )
    corr = y.pivot(index="Hybrid", columns="Env", values="Yield_Mg_ha").corr()
    corr_out = corr.reset_index().rename(columns={"Env": "environment"})

    outputs = {
        "overall": overall,
        "by_env": by_env,
        "by_hybrid": by_hybrid,
        "top20": top_hybrids,
        "split_summary": split_yield_summary,
        "variance_decomp": variance_decomp,
        "env_corr": corr_out,
    }
    output_names = {
        "overall": "soynam_6env_nasa_yield_overall_summary.csv",
        "by_env": "soynam_6env_nasa_yield_by_environment.csv",
        "by_hybrid": "soynam_6env_nasa_yield_by_hybrid.csv",
        "top20": "soynam_6env_nasa_yield_top20_hybrids.csv",
        "split_summary": "soynam_6env_nasa_yield_split_summary.csv",
        "variance_decomp": "soynam_6env_nasa_yield_additive_variance_decomposition.csv",
        "env_corr": "soynam_6env_nasa_yield_environment_correlation.csv",
    }
    for key, filename in output_names.items():
        outputs[key].to_csv(OUT_DIR / filename, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(y["Yield_Mg_ha"], bins=30, color="#4c78a8", alpha=0.85)
    axes[0, 0].set_title("Yield distribution")
    axes[0, 0].set_xlabel("Yield (Mg/ha)")
    axes[0, 0].set_ylabel("Rows")
    env_order = by_env["Env"].tolist()
    box_data = [y.loc[y["Env"].eq(env), "Yield_Mg_ha"].to_numpy() for env in env_order]
    axes[0, 1].boxplot(box_data, labels=env_order, vert=True)
    axes[0, 1].set_title("Yield by environment")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].set_ylabel("Yield (Mg/ha)")
    axes[1, 0].bar(by_env["Env"], by_env["mean_yield_Mg_ha"], color="#59a14f")
    axes[1, 0].set_title("Mean yield by environment")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].set_ylabel("Mean yield (Mg/ha)")
    axes[1, 1].bar(split_yield_summary["dataset"], split_yield_summary["mean_yield_Mg_ha"], color="#f28e2b")
    axes[1, 1].set_title("Mean yield by split")
    axes[1, 1].set_ylabel("Mean yield (Mg/ha)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "soynam_6env_nasa_yield_summary_plots.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_decomp = variance_decomp[variance_decomp["source"].ne("total")]
    ax.bar(plot_decomp["source"], plot_decomp["pct_total_ss"], color=["#4c78a8", "#59a14f", "#f28e2b"])
    ax.set_ylabel("Percent of total sum of squares")
    ax.set_title("Additive yield variation breakdown")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "soynam_6env_nasa_yield_variance_decomposition.png", dpi=200)
    plt.close(fig)
    return outputs


def marker_hits_for_window(markers: pd.DataFrame, refs: pd.DataFrame, window_bp: int) -> tuple[np.ndarray, np.ndarray]:
    marker_hit = np.zeros(len(markers), dtype=bool)
    ref_hit = np.zeros(len(refs), dtype=bool)
    marker_chr = markers["chr"].to_numpy()
    marker_pos = markers["position"].to_numpy()
    ref_chr = refs["chr"].to_numpy()
    ref_pos = refs["position"].to_numpy()
    for ref_index, (chrom, pos) in enumerate(zip(ref_chr, ref_pos)):
        same_chr = marker_chr == chrom
        close = np.abs(marker_pos - pos) <= window_bp
        hits = same_chr & close
        if hits.any():
            marker_hit |= hits
            ref_hit[ref_index] = True
    return marker_hit, ref_hit


def gwas_overlap(ranking_chrpos: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    ref_path = BASE_DIR / "soynam_arch_a_general" / "soynam_diers2018_yield_mta_reference.csv"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing Diers reference table: {ref_path}")
    refs = pd.read_csv(ref_path).copy()
    refs = refs[refs["chr"].notna() & refs["position"].notna()].copy()
    refs["chr"] = refs["chr"].astype(int)
    refs["position"] = refs["position"].astype(int)
    refs.to_csv(OUT_DIR / "soynam_6env_nasa_diers2018_yield_mta_reference.csv", index=False)

    full = ranking_chrpos["full"].copy()
    full = full[full["chr"].notna() & full["position"].notna()].copy()
    full["chr"] = full["chr"].astype(int)
    full["position"] = full["position"].astype(int)
    marker_universe = full.reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    n_random = 20000
    rows = []
    match_rows = []
    exact_rows = []

    exact_merge = full.merge(refs, on=["chr", "position"], suffixes=("_marker", "_reference"))
    if not exact_merge.empty:
        for _, row in exact_merge.iterrows():
            exact_rows.append(row.to_dict())

    for window_bp in [100000, 500000]:
        universe_marker_hit, universe_ref_hit = marker_hits_for_window(marker_universe, refs, window_bp)
        hit_indices = np.flatnonzero(universe_marker_hit)
        for top_k in TOP_K_VALUES:
            top_markers = full.head(top_k).reset_index(drop=True)
            marker_hit, ref_hit = marker_hits_for_window(top_markers, refs, window_bp)
            observed_marker_hits = int(marker_hit.sum())
            observed_ref_hits = int(ref_hit.sum())

            for marker_idx, marker in top_markers.loc[marker_hit].iterrows():
                same_chr_refs = refs[refs["chr"].eq(int(marker["chr"]))].copy()
                same_chr_refs["distance_bp"] = (same_chr_refs["position"] - int(marker["position"])).abs()
                for _, ref in same_chr_refs[same_chr_refs["distance_bp"].le(window_bp)].iterrows():
                    match_rows.append(
                        {
                            "model": "six_env_nasa_full",
                            "top_k": top_k,
                            "window_bp": window_bp,
                            "rank": int(marker["rank"]),
                            "marker_id": marker["marker_id"],
                            "marker_chr": int(marker["chr"]),
                            "marker_position": int(marker["position"]),
                            "marker_chr_pos": marker["chr_pos"],
                            "avg_weight": float(marker["avg_weight"]),
                            "reference_marker": ref["reference_marker"],
                            "reference_ssid": ref.get("ssid", np.nan),
                            "reference_chr": int(ref["chr"]),
                            "reference_position": int(ref["position"]),
                            "reference_chr_pos": ref["chr_pos"],
                            "distance_bp": int(ref["distance_bp"]),
                            "trait": ref["trait"],
                        }
                    )

            random_marker_hits = np.empty(n_random, dtype=np.int16)
            random_ref_hits = np.empty(n_random, dtype=np.int16)
            universe_indices = np.arange(len(marker_universe))
            for i in range(n_random):
                sample_idx = rng.choice(universe_indices, size=top_k, replace=False)
                random_marker_hits[i] = int(universe_marker_hit[sample_idx].sum())
                sample_markers = marker_universe.iloc[sample_idx].reset_index(drop=True)
                _, sample_ref_hit = marker_hits_for_window(sample_markers, refs, window_bp)
                random_ref_hits[i] = int(sample_ref_hit.sum())

            rows.append(
                {
                    "model": "six_env_nasa_full",
                    "reference_set": "Diers2018_yield",
                    "top_k": top_k,
                    "window_bp": window_bp,
                    "reference_snp_count": len(refs),
                    "all_model_marker_count": len(marker_universe),
                    "all_model_markers_within_window": int(universe_marker_hit.sum()),
                    "observed_marker_hits": observed_marker_hits,
                    "observed_marker_hit_pct_of_top_k": 100 * observed_marker_hits / top_k,
                    "observed_reference_snp_hits": observed_ref_hits,
                    "reference_snp_coverage_pct": 100 * observed_ref_hits / len(refs),
                    "random_mean_marker_hits": float(random_marker_hits.mean()),
                    "random_sd_marker_hits": float(random_marker_hits.std(ddof=1)),
                    "random_mean_reference_hits": float(random_ref_hits.mean()),
                    "random_sd_reference_hits": float(random_ref_hits.std(ddof=1)),
                    "empirical_p_marker_hits_ge_observed": float((random_marker_hits >= observed_marker_hits).mean()),
                    "empirical_p_reference_hits_ge_observed": float((random_ref_hits >= observed_ref_hits).mean()),
                    "random_iterations": n_random,
                }
            )

    summary = pd.DataFrame(rows)
    matches = pd.DataFrame(match_rows)
    exact = pd.DataFrame(exact_rows)
    summary.to_csv(OUT_DIR / "soynam_6env_nasa_full_arch_a_vs_diers2018_yield_window_overlap_summary.csv", index=False)
    matches.to_csv(OUT_DIR / "soynam_6env_nasa_full_arch_a_vs_diers2018_yield_window_matches.csv", index=False)
    exact.to_csv(OUT_DIR / "soynam_6env_nasa_full_arch_a_vs_diers2018_yield_exact_matches.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for window_bp, color in [(100000, "#4c78a8"), (500000, "#f28e2b")]:
        subset = summary[summary["window_bp"].eq(window_bp)].sort_values("top_k")
        label = f"{window_bp // 1000} kb observed"
        ax.plot(subset["top_k"], subset["observed_marker_hits"], marker="o", color=color, label=label)
        ax.plot(subset["top_k"], subset["random_mean_marker_hits"], linestyle="--", color=color, alpha=0.8, label=f"{window_bp // 1000} kb random mean")
    ax.set_xlabel("Top-k marker cutoff")
    ax.set_ylabel("Markers within window of Diers yield MTA")
    ax.set_title("SoyNAM six-env NASA full Arch A overlap with Diers 2018 yield MTAs")
    ax.set_xticks(TOP_K_VALUES)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "soynam_6env_nasa_full_arch_a_diers2018_yield_overlap_100kb_500kb.png", dpi=200)
    plt.close(fig)

    return {"reference": refs, "summary": summary, "matches": matches, "exact": exact}


def format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def main() -> None:
    set_seeds()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    y_raw = pd.read_csv(DATA_DIR / "Y.csv")
    x_raw = pd.read_csv(DATA_DIR / "X.csv")
    weather_raw = pd.read_csv(DATA_DIR / "env.cov.csv")

    weather_df = build_weather_table(weather_raw)
    valid_trait_df, dataset_summary = prepare_trait_table(y_raw, x_raw, weather_df)
    genotype_df, imputation_summary = prepare_genotype_table(x_raw, valid_trait_df)
    env_vectors_df, weather_summary = build_lstm_env_vectors(weather_df, valid_trait_df)
    train_split_df, holdout_split_df, split_summary = prepare_splits(valid_trait_df)

    results, output_paths, training_summary = run_gnn_models()
    ranking_chrpos = save_top_marker_outputs(results)
    intersection_summary, intersection_ids = marker_intersections(ranking_chrpos)
    plot_marker_intersections(intersection_summary)
    plot_weight_curves(ranking_chrpos)
    yield_analysis(valid_trait_df, train_split_df, holdout_split_df)
    gwas_overlap(ranking_chrpos)

    print("Six-environment SoyNAM NASA POWER analysis completed.")
    print(dataset_summary.to_string(index=False))
    print(intersection_summary.to_string(index=False))


if __name__ == "__main__":
    main()
