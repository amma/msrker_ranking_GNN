from __future__ import annotations

import os
import random
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Masking
from tensorflow.keras.models import Sequential


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_TEMPLATE = "weather_kmeans_best_5feature_cluster_{cluster}.csv"
OUTPUT_TEMPLATE = "weather_kmeans_best_5feature_cluster_{cluster}_lstm_env_vectors.csv"
ALL_OUTPUT_PATH = BASE_DIR / "weather_kmeans_best_5feature_all_clusters_lstm_env_vectors.csv"

SEED = 42
HIDDEN_DIM = 50
OUTPUT_DIM = 27
EPOCHS = 1
N_CLUSTERS = 4

EXCLUDE_COLUMNS = {"Env", "Date", "cluster"}


def set_seed(seed_value: int = SEED) -> None:
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)


def create_lstm_model(timesteps: int, num_features: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(timesteps, num_features)),
            Masking(mask_value=0.0),
            LSTM(HIDDEN_DIM, return_sequences=False),
            Dense(OUTPUT_DIM),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def load_cluster_files() -> tuple[pd.DataFrame, list[str]]:
    frames = []
    feature_columns: list[str] | None = None

    for cluster in range(N_CLUSTERS):
        input_path = BASE_DIR / INPUT_TEMPLATE.format(cluster=cluster)
        cluster_df = pd.read_csv(input_path).copy()
        cluster_df["Env"] = cluster_df["Env"].astype(str)
        cluster_df["cluster"] = cluster

        current_features = [
            column
            for column in cluster_df.columns
            if column not in EXCLUDE_COLUMNS
        ]
        if feature_columns is None:
            feature_columns = current_features
        elif feature_columns != current_features:
            raise ValueError(f"Feature columns differ in {input_path.name}")

        for column in current_features:
            cluster_df[column] = pd.to_numeric(cluster_df[column], errors="coerce")
        if cluster_df[current_features].isna().any().any():
            missing_counts = cluster_df[current_features].isna().sum()
            raise ValueError(
                f"Missing/non-numeric values in {input_path.name}:\n"
                f"{missing_counts[missing_counts.gt(0)]}"
            )
        frames.append(cluster_df)

    if feature_columns is None:
        raise ValueError("No cluster files were loaded.")
    return pd.concat(frames, ignore_index=True), feature_columns


def build_env_tensor(data: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    env_meta_rows = []
    sequences = []

    for env, env_df in data.groupby("Env", sort=True):
        cluster = int(env_df["cluster"].iloc[0])
        env_df = env_df.sort_values("Date", kind="stable") if "Date" in env_df.columns else env_df
        sequence = env_df[feature_columns].to_numpy(dtype=np.float32, copy=True)
        sequences.append(sequence)
        env_meta_rows.append({"Env": env, "cluster": cluster, "record_count": int(len(env_df))})

    max_timesteps = max(sequence.shape[0] for sequence in sequences)
    num_features = len(feature_columns)
    tensor = np.zeros((len(sequences), max_timesteps, num_features), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        tensor[index, : sequence.shape[0], :] = sequence

    return tensor, pd.DataFrame(env_meta_rows)


def main() -> None:
    set_seed(SEED)
    data, feature_columns = load_cluster_files()
    env_tensor, env_meta = build_env_tensor(data, feature_columns)

    model = create_lstm_model(timesteps=env_tensor.shape[1], num_features=env_tensor.shape[2])

    random_targets = np.random.rand(len(env_tensor), OUTPUT_DIM).astype(np.float32)
    model.fit(env_tensor, random_targets, epochs=EPOCHS, verbose=0)
    vectors = model.predict(env_tensor, verbose=0).astype(float)

    vector_columns = [f"lstm_vector_{index}" for index in range(1, OUTPUT_DIM + 1)]
    vectors_df = pd.concat([env_meta, pd.DataFrame(vectors, columns=vector_columns)], axis=1)
    vectors_df = vectors_df.sort_values(["cluster", "Env"], kind="stable").reset_index(drop=True)

    for cluster in range(N_CLUSTERS):
        cluster_vectors = vectors_df[vectors_df["cluster"].eq(cluster)].copy()
        output_path = BASE_DIR / OUTPUT_TEMPLATE.format(cluster=cluster)
        cluster_vectors.to_csv(output_path, index=False)
        print(f"Cluster {cluster}: {len(cluster_vectors)} env vectors saved to {output_path.name}")

    vectors_df.to_csv(ALL_OUTPUT_PATH, index=False)
    print(f"Combined vectors saved to {ALL_OUTPUT_PATH.name}")
    print(f"Input weather features: {len(feature_columns)}")
    print(f"Total environments: {vectors_df['Env'].nunique()}")
    print(f"Vector length: {OUTPUT_DIM}")


if __name__ == "__main__":
    main()
