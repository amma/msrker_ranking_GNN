from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from gnn_model import (
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    GRAD_ACCUM_STEPS,
    HIDDEN_DIM,
    LEARNING_RATE,
    MAE_WEIGHT,
    MICRO_BATCH_SIZE,
    MSE_WEIGHT,
    NUM_HEADS,
    NUM_LAYERS,
    VAL_FRACTION,
    WEIGHT_DECAY,
    MarkerEnvironmentGNN,
    aggregate_pairs,
    get_training_device,
    masked_regression_losses,
    set_seed,
    split_pairs,
)
from cluster_train import select_top_and_random_genotypes

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "env_encoder_comparison"

TRAIT_PATH = BASE_DIR / "train_trait.csv"
GENOTYPE_PATH = BASE_DIR / "genotype_reduced.csv"
WEATHER_PATH = BASE_DIR / "4_Training_Weather_Data_2014_2023_full_year.csv"

WEATHER_FEATURES = ["ALLSKY_SFC_SW_DNI", "WS2M", "ALLSKY_SFC_PAR_TOT", "PS", "ALLSKY_SFC_SW_DWN"]

ALL_WEATHER_FEATURES = [
    "ALLSKY_SFC_SW_DNI", "WS2M", "ALLSKY_SFC_PAR_TOT", "PS", "ALLSKY_SFC_SW_DWN",
    "T2M_MAX", "PRECTOTCORR", "GWETTOP", "QV2M", "GWETPROF",
    "RH2M", "GWETROOT", "T2M", "T2MDEW", "T2M_MIN", "T2MWET",
]

SEED = 42
ENV_DIM = 27
USE_ENV_RESIDUAL = True
EPS = 1e-6


def load_weather_tensor(env_names: list[str], weather_features: list[str]) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    weather_df = pd.read_csv(WEATHER_PATH, usecols=["Env", "Date", *weather_features]).copy()
    weather_df["Env"] = weather_df["Env"].astype(str)
    needed = set(env_names)
    weather_df = weather_df[weather_df["Env"].isin(needed)].copy()
    missing = needed - set(weather_df["Env"].unique())
    if missing:
        raise ValueError(f"Missing daily weather records for environments: {sorted(missing)[:10]}")

    values = weather_df[weather_features].to_numpy(dtype=np.float64)
    feature_mean = values.mean(axis=0)
    feature_std = values.std(axis=0)
    feature_std[feature_std < EPS] = 1.0
    weather_df.loc[:, weather_features] = (values - feature_mean) / feature_std

    env_name_to_index: dict[str, int] = {}
    sequences: list[np.ndarray] = []
    for index, (env_name, env_df) in enumerate(weather_df.groupby("Env", sort=True)):
        env_df = env_df.sort_values("Date", kind="stable")
        sequences.append(env_df[weather_features].to_numpy(dtype=np.float32, copy=True))
        env_name_to_index[str(env_name)] = index

    lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
    max_days = int(lengths.max())
    tensor = np.zeros((len(sequences), max_days, len(weather_features)), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        tensor[index, : sequence.shape[0], :] = sequence

    return torch.from_numpy(tensor), lengths, env_name_to_index


@dataclass
class GraphSample:
    hybrid: str
    marker_values: np.ndarray
    env_names: list[str]
    env_indices: np.ndarray
    targets: np.ndarray
    replicate_counts: np.ndarray
    target_baselines: np.ndarray
    split: str


def build_graphs(
    pair_df: pd.DataFrame,
    genotype_lookup: pd.DataFrame,
    marker_columns: list[str],
    env_name_to_index: dict[str, int],
    target_mean: float,
    target_std: float,
    env_target_lookup: pd.Series,
    global_mean: float,
    use_env_residual: bool,
    split: str,
) -> list[GraphSample]:
    graphs: list[GraphSample] = []
    if pair_df.empty:
        return graphs

    for hybrid_name, hybrid_df in pair_df.groupby("Hybrid", sort=True):
        marker_values = genotype_lookup.loc[hybrid_name, marker_columns].to_numpy(dtype=np.uint8, copy=True)
        env_names = hybrid_df["Env"].tolist()
        env_indices = np.array([env_name_to_index[str(name)] for name in env_names], dtype=np.int64)

        target_baselines = hybrid_df["Env"].map(env_target_lookup).fillna(global_mean).to_numpy(dtype=np.float32, copy=True)
        raw_targets = hybrid_df["mean_yield"].to_numpy(dtype=np.float32, copy=True)
        if use_env_residual:
            raw_targets = raw_targets - target_baselines
        else:
            target_baselines = np.zeros_like(raw_targets, dtype=np.float32)

        targets = ((raw_targets - target_mean) / target_std).astype(np.float32, copy=False)
        replicate_counts = hybrid_df["replicate_count"].to_numpy(dtype=np.float32, copy=True)
        graphs.append(
            GraphSample(
                hybrid=str(hybrid_name),
                marker_values=marker_values,
                env_names=[str(name) for name in env_names],
                env_indices=env_indices,
                targets=targets,
                replicate_counts=replicate_counts,
                target_baselines=target_baselines,
                split=split,
            )
        )
    return graphs


class GraphDataset(Dataset):
    def __init__(self, graphs: list[GraphSample]) -> None:
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> GraphSample:
        return self.graphs[index]


def collate_graph_batch(batch: list[GraphSample]) -> tuple[Any, ...]:
    batch_size = len(batch)
    num_markers = batch[0].marker_values.shape[0]
    max_envs = max(sample.env_indices.shape[0] for sample in batch)

    marker_values = torch.empty((batch_size, num_markers), dtype=torch.long)
    env_indices = torch.zeros((batch_size, max_envs), dtype=torch.long)
    targets = torch.zeros((batch_size, max_envs), dtype=torch.float32)
    env_mask = torch.zeros((batch_size, max_envs), dtype=torch.bool)
    pair_weights = torch.zeros((batch_size, max_envs), dtype=torch.float32)
    target_baselines = torch.zeros((batch_size, max_envs), dtype=torch.float32)
    hybrid_names: list[str] = []
    env_name_lists: list[list[str]] = []
    split_names: list[str] = []

    for batch_index, sample in enumerate(batch):
        env_count = sample.env_indices.shape[0]
        marker_values[batch_index] = torch.from_numpy(sample.marker_values.astype(np.int64, copy=False))
        env_indices[batch_index, :env_count] = torch.from_numpy(sample.env_indices)
        targets[batch_index, :env_count] = torch.from_numpy(sample.targets)
        env_mask[batch_index, :env_count] = True
        pair_weights[batch_index, :env_count] = torch.from_numpy(sample.replicate_counts)
        target_baselines[batch_index, :env_count] = torch.from_numpy(sample.target_baselines)
        hybrid_names.append(sample.hybrid)
        env_name_lists.append(list(sample.env_names))
        split_names.append(sample.split)

    return marker_values, env_indices, targets, env_mask, pair_weights, target_baselines, hybrid_names, env_name_lists, split_names


class SharedLSTMEnvEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_dim, batch_first=True)

    def forward(self, weather: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = pack_padded_sequence(weather, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        embedding = hidden[-1]
        placeholder_weights = embedding.new_zeros((embedding.size(0), 1))
        return embedding, placeholder_weights


class PerVariableLSTMEnvEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.lstms = nn.ModuleList(
            [nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True) for _ in range(num_features)]
        )
        self.attention_score = nn.Linear(hidden_dim, 1)

    def forward(self, weather: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths_cpu = lengths.cpu()
        per_variable_states = []
        for feature_index, lstm in enumerate(self.lstms):
            single = weather[:, :, feature_index : feature_index + 1]
            packed = pack_padded_sequence(single, lengths_cpu, batch_first=True, enforce_sorted=False)
            _, (hidden, _) = lstm(packed)
            per_variable_states.append(hidden[-1])
        stacked = torch.stack(per_variable_states, dim=1)
        scores = self.attention_score(stacked).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        combined = torch.einsum("ef,efd->ed", weights, stacked)
        return combined, weights


class EnvGNNModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_markers: int,
        env_dim: int,
        weather_tensor: torch.Tensor,
        weather_lengths: torch.Tensor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.gnn = MarkerEnvironmentGNN(num_markers=num_markers, env_dim=env_dim, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
        self.register_buffer("weather_tensor", weather_tensor, persistent=False)
        self.register_buffer("weather_lengths", weather_lengths, persistent=False)

    def forward(
        self,
        marker_values: torch.Tensor,
        env_indices: torch.Tensor,
        env_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        env_embedding_table, variable_weights = self.encoder(self.weather_tensor, self.weather_lengths)
        safe_indices = env_indices.clamp(min=0)
        gathered = env_embedding_table[safe_indices]
        result = self.gnn(marker_values, gathered, env_mask, return_attention=return_attention)
        return result, variable_weights


def compute_baselines(train_pair_df: pd.DataFrame, val_pair_df: pd.DataFrame) -> pd.DataFrame:
    global_mean = float(train_pair_df["mean_yield"].mean())
    env_mean = train_pair_df.groupby("Env")["mean_yield"].mean()
    hybrid_mean = train_pair_df.groupby("Hybrid")["mean_yield"].mean()

    env_pred = val_pair_df["Env"].map(env_mean).fillna(global_mean)
    hybrid_pred = val_pair_df["Hybrid"].map(hybrid_mean).fillna(global_mean)
    additive_pred = env_pred + hybrid_pred - global_mean
    mean_pred = pd.Series(np.full(len(val_pair_df), global_mean), index=val_pair_df.index)

    records = []
    for name, predictions in [
        ("global_mean", mean_pred),
        ("env_mean", env_pred),
        ("hybrid_mean", hybrid_pred),
        ("env_plus_hybrid", additive_pred),
    ]:
        mse = mean_squared_error(val_pair_df["mean_yield"], predictions)
        mae = mean_absolute_error(val_pair_df["mean_yield"], predictions)
        r2 = r2_score(val_pair_df["mean_yield"], predictions)
        records.append({"baseline": name, "mse": mse, "mae": mae, "r2": r2})
    return pd.DataFrame(records).sort_values("mse", kind="stable").reset_index(drop=True)


def build_env_model(encoder: nn.Module, num_markers: int, env_dim: int, weather_tensor: torch.Tensor, weather_lengths: torch.Tensor) -> tuple[nn.Module, nn.Module]:
    base_model = EnvGNNModel(encoder, num_markers=num_markers, env_dim=env_dim, weather_tensor=weather_tensor, weather_lengths=weather_lengths)
    if torch.cuda.is_available():
        base_model = base_model.to(torch.device("cuda:0"))
        if torch.cuda.device_count() >= 2:
            parallel_model = nn.DataParallel(base_model, device_ids=[0, 1])
            return parallel_model, parallel_model.module
    return base_model, base_model


@torch.inference_mode()
def evaluate(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> dict:
    model.eval()
    predictions_list = []
    targets_list = []
    weights_list = []

    for marker_batch, env_index_batch, target_batch, env_mask, pair_weight_batch, target_baseline_batch, *_ in loader:
        marker_batch = marker_batch.to(device, non_blocking=True)
        env_index_batch = env_index_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)
        env_mask = env_mask.to(device, non_blocking=True)
        pair_weight_batch = pair_weight_batch.to(device, non_blocking=True)
        target_baseline_batch = target_baseline_batch.to(device, non_blocking=True)

        predictions_std, _ = model(marker_batch, env_index_batch, env_mask)

        mask = env_mask
        pred = (predictions_std * target_std) + target_mean + target_baseline_batch
        target = (target_batch * target_std) + target_mean + target_baseline_batch

        predictions_list.append(pred[mask].detach().cpu().numpy())
        targets_list.append(target[mask].detach().cpu().numpy())
        weights_list.append(pair_weight_batch[mask].detach().cpu().numpy())

    all_pred = np.concatenate(predictions_list) if predictions_list else np.empty(0)
    all_target = np.concatenate(targets_list) if targets_list else np.empty(0)
    all_weights = np.concatenate(weights_list) if weights_list else np.empty(0)

    if len(all_pred) == 0:
        return {"mse": math.nan, "mae": math.nan, "r2": math.nan}

    mse = float(mean_squared_error(all_target, all_pred, sample_weight=all_weights))
    mae = float(mean_absolute_error(all_target, all_pred, sample_weight=all_weights))
    r2 = float(r2_score(all_target, all_pred, sample_weight=all_weights))
    return {"mse": mse, "mae": mae, "r2": r2}


def train_variant(
    variant_name: str,
    encoder: nn.Module,
    train_graphs: list[GraphSample],
    val_graphs: list[GraphSample],
    num_markers: int,
    weather_tensor: torch.Tensor,
    weather_lengths: torch.Tensor,
    weather_features: list[str],
    target_mean: float,
    target_std: float,
    epochs: int,
    device: torch.device,
    output_dir: Path,
) -> dict:
    set_seed(SEED)
    train_loader = DataLoader(
        GraphDataset(train_graphs), batch_size=MICRO_BATCH_SIZE, shuffle=True, num_workers=2,
        pin_memory=torch.cuda.is_available(), drop_last=False, collate_fn=collate_graph_batch,
    )
    val_loader = DataLoader(
        GraphDataset(val_graphs), batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available(), drop_last=False, collate_fn=collate_graph_batch,
    )

    training_model, base_model = build_env_model(encoder, num_markers, ENV_DIM, weather_tensor, weather_lengths)

    optimizer = torch.optim.AdamW(training_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history: list[dict] = []
    best_state = copy.deepcopy(base_model.state_dict())
    best_val_mse = math.inf
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = epoch_mse = epoch_mae = 0.0
        step_count = 0

        for step_index, batch in enumerate(train_loader, start=1):
            marker_batch, env_index_batch, target_batch, env_mask, pair_weight_batch, _, _, _, _ = batch
            marker_batch = marker_batch.to(device, non_blocking=True)
            env_index_batch = env_index_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            env_mask = env_mask.to(device, non_blocking=True)
            pair_weight_batch = pair_weight_batch.to(device, non_blocking=True)

            predictions, _ = training_model(marker_batch, env_index_batch, env_mask)
            batch_loss, batch_mse, batch_mae = masked_regression_losses(predictions, target_batch, env_mask, pair_weight_batch)
            (batch_loss / GRAD_ACCUM_STEPS).backward()

            if step_index % GRAD_ACCUM_STEPS == 0 or step_index == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(batch_loss.detach().item())
            epoch_mse += float(batch_mse.item())
            epoch_mae += float(batch_mae.item())
            step_count += 1

        train_loss = epoch_loss / max(step_count, 1)
        train_mse_std = epoch_mse / max(step_count, 1)
        train_mae_std = epoch_mae / max(step_count, 1)

        val_metrics = evaluate(val_loader, base_model, device, target_mean, target_std) if val_graphs else {"mse": math.nan, "mae": math.nan, "r2": math.nan}
        if not math.isnan(val_metrics["mse"]) and val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(base_model.state_dict())

        history.append(
            {
                "epoch": epoch,
                "train_loss_std": train_loss,
                "train_mse_std": train_mse_std,
                "train_mae_std": train_mae_std,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "best_val_mse": best_val_mse,
                "best_epoch": best_epoch,
            }
        )
        print(
            f"[{variant_name}] Epoch {epoch:03d}/{epochs} | train_loss_std={train_loss:.6f} "
            f"| val_mse={val_metrics['mse']:.6f} | val_mae={val_metrics['mae']:.6f} | val_r2={val_metrics['r2']:.6f}",
            flush=True,
        )

    base_model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / f"{variant_name}_training_history.csv", index=False)
    torch.save(base_model.state_dict(), output_dir / f"{variant_name}_model.pt")

    best_row = history_df.sort_values(["val_mse", "epoch"], kind="stable").iloc[0]

    variable_attention_df = None
    if variant_name == "per_variable":
        base_model.eval()
        with torch.inference_mode():
            _, variable_weights = base_model.encoder(base_model.weather_tensor, base_model.weather_lengths)
        variable_attention_df = pd.DataFrame(
            variable_weights.detach().cpu().numpy(), columns=weather_features
        )

    return {
        "history": history_df,
        "best_epoch": int(best_row["epoch"]),
        "best_val_mse": float(best_row["val_mse"]),
        "best_val_mae": float(best_row["val_mae"]),
        "best_val_r2": float(best_row["val_r2"]),
        "variable_attention": variable_attention_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared vs. per-variable LSTM environment encoder comparison.")
    parser.add_argument(
        "--all-weather-features",
        action="store_true",
        help="Use all 16 weather variables instead of the default 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weather_features = ALL_WEATHER_FEATURES if args.all_weather_features else WEATHER_FEATURES
    results_dir = RESULTS_DIR / "all_16_variables" if args.all_weather_features else RESULTS_DIR

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This comparison requires a GPU.")
    device = get_training_device()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpu_name}", flush=True)
    print(f"Weather features ({len(weather_features)}): {weather_features}", flush=True)

    results_dir.mkdir(parents=True, exist_ok=True)

    trait_df = pd.read_csv(TRAIT_PATH).copy()
    genotype_df = pd.read_csv(GENOTYPE_PATH).copy()
    genotype_df = genotype_df.rename(columns={genotype_df.columns[0]: "Hybrid"})
    trait_df["Hybrid"] = trait_df["Hybrid"].astype(str)
    trait_df["Env"] = trait_df["Env"].astype(str)
    genotype_df["Hybrid"] = genotype_df["Hybrid"].astype(str)

    marker_columns = [column for column in genotype_df.columns if column != "Hybrid"]
    genotype_lookup = genotype_df.drop_duplicates("Hybrid").set_index("Hybrid")

    valid_trait_df = trait_df[trait_df["Hybrid"].isin(genotype_lookup.index)].copy()
    _, candidate_df = select_top_and_random_genotypes(valid_trait_df)

    weather_env_ids = set(pd.read_csv(WEATHER_PATH, usecols=["Env"])["Env"].astype(str).unique())
    missing_weather = sorted(set(candidate_df["Env"].unique()) - weather_env_ids)
    if missing_weather:
        print(f"Dropping {len(missing_weather)} environments with no daily weather record: {missing_weather}", flush=True)
        candidate_df = candidate_df[~candidate_df["Env"].isin(missing_weather)].reset_index(drop=True)

    pair_df = aggregate_pairs(candidate_df)
    pair_df = pair_df[pair_df["Hybrid"].isin(genotype_lookup.index)].reset_index(drop=True)
    print(
        f"Maize analysis set: {pair_df['Hybrid'].nunique()} genotypes, "
        f"{pair_df['Env'].nunique()} environments, {len(pair_df)} genotype-env pairs",
        flush=True,
    )

    train_pair_df, val_pair_df = split_pairs(pair_df, val_fraction=VAL_FRACTION, split_seed=SEED)

    env_names = sorted(pair_df["Env"].unique())
    weather_tensor, weather_lengths, env_name_to_index = load_weather_tensor(env_names, weather_features)
    print(f"Loaded weather for {len(env_name_to_index)} environments, max_days={weather_tensor.shape[1]}", flush=True)

    global_mean = float(train_pair_df["mean_yield"].mean())
    env_target_lookup = train_pair_df.groupby("Env")["mean_yield"].mean()
    if USE_ENV_RESIDUAL:
        baseline_values = train_pair_df["Env"].map(env_target_lookup).fillna(global_mean).to_numpy(dtype=np.float32)
        train_targets_raw = train_pair_df["mean_yield"].to_numpy(dtype=np.float32) - baseline_values
    else:
        train_targets_raw = train_pair_df["mean_yield"].to_numpy(dtype=np.float32)
    target_mean = float(train_targets_raw.mean())
    target_std = float(train_targets_raw.std())
    if target_std < EPS:
        target_std = 1.0

    train_graphs = build_graphs(
        train_pair_df, genotype_lookup, marker_columns, env_name_to_index,
        target_mean, target_std, env_target_lookup, global_mean, USE_ENV_RESIDUAL, "train",
    )
    val_graphs = build_graphs(
        val_pair_df, genotype_lookup, marker_columns, env_name_to_index,
        target_mean, target_std, env_target_lookup, global_mean, USE_ENV_RESIDUAL, "val",
    )
    print(f"Genotype graphs: train={len(train_graphs)} val={len(val_graphs)} markers={len(marker_columns)}", flush=True)

    baselines_df = compute_baselines(train_pair_df, val_pair_df)
    baselines_df.to_csv(results_dir / "env_mean_baselines.csv", index=False)
    print("Validation baselines:\n" + baselines_df.to_string(index=False), flush=True)

    manifest = {
        "seed": SEED,
        "use_env_residual": USE_ENV_RESIDUAL,
        "env_dim": ENV_DIM,
        "epochs": EPOCHS,
        "weather_features": weather_features,
        "num_environments": len(env_name_to_index),
        "num_markers": len(marker_columns),
        "num_genotypes": int(genotype_lookup.shape[0]),
        "train_pairs": len(train_pair_df),
        "val_pairs": len(val_pair_df),
        "gpu": gpu_name,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    results: dict[str, dict] = {}

    print("=" * 80 + "\nTraining shared (single joint LSTM) encoder baseline\n" + "=" * 80, flush=True)
    shared_encoder = SharedLSTMEnvEncoder(num_features=len(weather_features), hidden_dim=ENV_DIM)
    results["shared"] = train_variant(
        "shared", shared_encoder, train_graphs, val_graphs, len(marker_columns),
        weather_tensor, weather_lengths, weather_features, target_mean, target_std, EPOCHS, device, results_dir,
    )

    print("=" * 80 + "\nTraining per-variable LSTM + attention encoder\n" + "=" * 80, flush=True)
    per_variable_encoder = PerVariableLSTMEnvEncoder(num_features=len(weather_features), hidden_dim=ENV_DIM)
    results["per_variable"] = train_variant(
        "per_variable", per_variable_encoder, train_graphs, val_graphs, len(marker_columns),
        weather_tensor, weather_lengths, weather_features, target_mean, target_std, EPOCHS, device, results_dir,
    )

    if results["per_variable"]["variable_attention"] is not None:
        attention_df = results["per_variable"]["variable_attention"].copy()
        attention_df.insert(0, "Env", sorted(env_name_to_index, key=env_name_to_index.get))
        attention_df.to_csv(results_dir / "per_variable_attention_by_env.csv", index=False)

    env_mean_row = baselines_df[baselines_df["baseline"] == "env_mean"].iloc[0]
    summary_rows = []
    for variant_name in ["shared", "per_variable"]:
        result = results[variant_name]
        summary_rows.append(
            {
                "variant": variant_name,
                "best_epoch": result["best_epoch"],
                "best_val_mse": result["best_val_mse"],
                "best_val_mae": result["best_val_mae"],
                "best_val_r2": result["best_val_r2"],
                "env_mean_baseline_mse": float(env_mean_row["mse"]),
                "env_mean_baseline_r2": float(env_mean_row["r2"]),
                "beats_env_mean": result["best_val_r2"] > float(env_mean_row["r2"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "comparison_summary.csv", index=False)
    print("\n" + "=" * 80 + "\nComparison summary\n" + "=" * 80, flush=True)
    print(summary_df.to_string(index=False), flush=True)

    print(f"\nAnalysis complete. Results written to {results_dir}", flush=True)


if __name__ == "__main__":
    main()
