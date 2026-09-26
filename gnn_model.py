from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_PATH = BASE_DIR / "final_genotype.csv"
GENOTYPE_PATH = BASE_DIR / "genotype_reduced.csv"
ENV_PATH = BASE_DIR / "train_env_vectors.csv"
DEFAULT_OUTPUT_PREFIX = "arch_a_multienv"

SEED = 42
HIDDEN_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 3
MICRO_BATCH_SIZE = 4
EFFECTIVE_BATCH_SIZE = 32
GRAD_ACCUM_STEPS = max(1, EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE)
EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MSE_WEIGHT = 0.8
MAE_WEIGHT = 0.2
VAL_FRACTION = 0.2
NUM_WORKERS = 4
EPS = 1e-6


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class GraphSample:
    hybrid: str
    marker_values: np.ndarray
    env_names: list[str]
    env_vectors: np.ndarray
    targets: np.ndarray
    replicate_counts: np.ndarray
    target_baselines: np.ndarray
    split: str


@dataclass
class PreparedData:
    train_graphs: list[GraphSample]
    val_graphs: list[GraphSample]
    all_graphs: list[GraphSample]
    train_pair_df: pd.DataFrame
    val_pair_df: pd.DataFrame
    all_pair_df: pd.DataFrame
    marker_columns: list[str]
    env_feature_columns: list[str]
    env_mean: np.ndarray
    env_std: np.ndarray
    target_mean: float
    target_std: float
    use_env_residual: bool


@dataclass
class OutputPaths:
    model: Path
    history: Path
    predictions: Path
    all_weights: Path
    ranking: Path
    node_list: Path
    baselines: Path


def build_output_paths(output_prefix: str = DEFAULT_OUTPUT_PREFIX) -> OutputPaths:
    paths = OutputPaths(
        model=BASE_DIR / f"{output_prefix}_model.pt",
        history=BASE_DIR / f"{output_prefix}_training_history.csv",
        predictions=BASE_DIR / f"{output_prefix}_predictions.csv",
        all_weights=BASE_DIR / f"{output_prefix}_all_weights.csv.gz",
        ranking=BASE_DIR / f"{output_prefix}_marker_ranking.csv",
        node_list=BASE_DIR / f"{output_prefix}_marker_nodes_desc.txt",
        baselines=BASE_DIR / f"{output_prefix}_baselines.csv",
    )
    for path in [
        paths.model,
        paths.history,
        paths.predictions,
        paths.all_weights,
        paths.ranking,
        paths.node_list,
        paths.baselines,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def aggregate_pairs(final_df: pd.DataFrame) -> pd.DataFrame:
    pair_df = (
        final_df.groupby(["Hybrid", "Env"], as_index=False)
        .agg(
            mean_yield=("Yield_Mg_ha", "mean"),
            replicate_count=("Yield_Mg_ha", "size"),
        )
        .sort_values(["Hybrid", "Env"], kind="stable")
        .reset_index(drop=True)
    )
    return pair_df


def split_pairs(
    pair_df: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    split_seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(split_seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []

    for _, hybrid_df in pair_df.groupby("Hybrid", sort=True):
        hybrid_df = hybrid_df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        num_pairs = len(hybrid_df)
        if num_pairs <= 1:
            train_parts.append(hybrid_df)
            continue

        val_count = max(1, int(round(num_pairs * val_fraction)))
        val_count = min(val_count, num_pairs - 1)
        val_parts.append(hybrid_df.iloc[:val_count].copy())
        train_parts.append(hybrid_df.iloc[val_count:].copy())

    train_pair_df = pd.concat(train_parts, ignore_index=True).sort_values(["Hybrid", "Env"], kind="stable").reset_index(drop=True)
    if val_parts:
        val_pair_df = pd.concat(val_parts, ignore_index=True).sort_values(["Hybrid", "Env"], kind="stable").reset_index(drop=True)
    else:
        val_pair_df = pair_df.iloc[0:0].copy()
    return train_pair_df, val_pair_df


def compute_standardization(
    train_pair_df: pd.DataFrame,
    env_lookup: pd.DataFrame,
    env_feature_columns: list[str],
    use_env_residual: bool,
) -> tuple[np.ndarray, np.ndarray, float, float, pd.Series, float]:
    train_env_matrix = np.stack(
        [
            env_lookup.loc[env_name, env_feature_columns].to_numpy(dtype=np.float32, copy=True)
            for env_name in train_pair_df["Env"]
        ],
        axis=0,
    )
    env_mean = train_env_matrix.mean(axis=0).astype(np.float32)
    env_std = train_env_matrix.std(axis=0).astype(np.float32)
    env_std[env_std < EPS] = 1.0

    global_mean = float(train_pair_df["mean_yield"].mean())
    env_target_lookup = train_pair_df.groupby("Env")["mean_yield"].mean()
    if use_env_residual:
        baseline_values = train_pair_df["Env"].map(env_target_lookup).fillna(global_mean).to_numpy(dtype=np.float32, copy=True)
        train_targets = train_pair_df["mean_yield"].to_numpy(dtype=np.float32, copy=True) - baseline_values
    else:
        train_targets = train_pair_df["mean_yield"].to_numpy(dtype=np.float32, copy=True)

    target_mean = float(train_targets.mean())
    target_std = float(train_targets.std())
    if target_std < EPS:
        target_std = 1.0

    return env_mean, env_std, target_mean, target_std, env_target_lookup, global_mean


def build_graphs(
    pair_df: pd.DataFrame,
    genotype_lookup: pd.DataFrame,
    env_lookup: pd.DataFrame,
    marker_columns: list[str],
    env_feature_columns: list[str],
    env_mean: np.ndarray,
    env_std: np.ndarray,
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
        env_vectors = np.stack(
            [
                env_lookup.loc[env_name, env_feature_columns].to_numpy(dtype=np.float32, copy=True)
                for env_name in env_names
            ],
            axis=0,
        )
        env_vectors = ((env_vectors - env_mean) / env_std).astype(np.float32, copy=False)
        target_baselines = hybrid_df["Env"].map(env_target_lookup).fillna(global_mean).to_numpy(dtype=np.float32, copy=True)
        raw_targets = hybrid_df["mean_yield"].to_numpy(dtype=np.float32, copy=True)
        if use_env_residual:
            raw_targets = raw_targets - target_baselines
        else:
            target_baselines = np.zeros_like(raw_targets, dtype=np.float32)

        targets = ((raw_targets - target_mean) / target_std).astype(
            np.float32,
            copy=False,
        )
        replicate_counts = hybrid_df["replicate_count"].to_numpy(dtype=np.float32, copy=True)
        graphs.append(
            GraphSample(
                hybrid=str(hybrid_name),
                marker_values=marker_values,
                env_names=[str(env_name) for env_name in env_names],
                env_vectors=env_vectors,
                targets=targets,
                replicate_counts=replicate_counts,
                target_baselines=target_baselines.astype(np.float32, copy=False),
                split=split,
            )
        )

    return graphs


def load_prepared_data(
    final_path: Path = FINAL_PATH,
    genotype_path: Path = GENOTYPE_PATH,
    env_path: Path = ENV_PATH,
    use_env_residual: bool = False,
    split_seed: int = SEED,
) -> PreparedData:
    final_df = pd.read_csv(final_path).copy()
    genotype_df = pd.read_csv(genotype_path).copy()
    env_df = pd.read_csv(env_path).copy()

    genotype_df = genotype_df.rename(columns={genotype_df.columns[0]: "Hybrid"})
    env_df = env_df.rename(columns={env_df.columns[0]: "Env"})

    final_df["Hybrid"] = final_df["Hybrid"].astype(str)
    final_df["Env"] = final_df["Env"].astype(str)
    genotype_df["Hybrid"] = genotype_df["Hybrid"].astype(str)
    env_df["Env"] = env_df["Env"].astype(str)

    marker_columns = [column for column in genotype_df.columns if column != "Hybrid"]
    env_feature_columns = [column for column in env_df.columns if column != "Env"]

    genotype_lookup = genotype_df.drop_duplicates("Hybrid").set_index("Hybrid")
    env_lookup = env_df.drop_duplicates("Env").set_index("Env")

    valid_final_df = final_df[
        final_df["Hybrid"].isin(genotype_lookup.index)
        & final_df["Env"].isin(env_lookup.index)
    ].copy()

    pair_df = aggregate_pairs(valid_final_df)
    train_pair_df, val_pair_df = split_pairs(pair_df, split_seed=split_seed)
    env_mean, env_std, target_mean, target_std, env_target_lookup, global_mean = compute_standardization(
        train_pair_df,
        env_lookup,
        env_feature_columns,
        use_env_residual=use_env_residual,
    )

    train_graphs = build_graphs(
        train_pair_df,
        genotype_lookup,
        env_lookup,
        marker_columns,
        env_feature_columns,
        env_mean,
        env_std,
        target_mean,
        target_std,
        env_target_lookup,
        global_mean,
        use_env_residual,
        split="train",
    )
    val_graphs = build_graphs(
        val_pair_df,
        genotype_lookup,
        env_lookup,
        marker_columns,
        env_feature_columns,
        env_mean,
        env_std,
        target_mean,
        target_std,
        env_target_lookup,
        global_mean,
        use_env_residual,
        split="val",
    )
    all_graphs = build_graphs(
        pair_df,
        genotype_lookup,
        env_lookup,
        marker_columns,
        env_feature_columns,
        env_mean,
        env_std,
        target_mean,
        target_std,
        env_target_lookup,
        global_mean,
        use_env_residual,
        split="all",
    )

    return PreparedData(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        all_graphs=all_graphs,
        train_pair_df=train_pair_df,
        val_pair_df=val_pair_df,
        all_pair_df=pair_df,
        marker_columns=[str(column) for column in marker_columns],
        env_feature_columns=[str(column) for column in env_feature_columns],
        env_mean=env_mean,
        env_std=env_std,
        target_mean=target_mean,
        target_std=target_std,
        use_env_residual=use_env_residual,
    )


class GraphDataset(Dataset):
    def __init__(self, graphs: list[GraphSample]) -> None:
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> GraphSample:
        return self.graphs[index]


def collate_graph_batch(batch: list[GraphSample]) -> tuple[torch.Tensor, ...]:
    batch_size = len(batch)
    num_markers = batch[0].marker_values.shape[0]
    env_dim = batch[0].env_vectors.shape[1]
    max_envs = max(sample.env_vectors.shape[0] for sample in batch)

    marker_values = torch.empty((batch_size, num_markers), dtype=torch.long)
    env_vectors = torch.zeros((batch_size, max_envs, env_dim), dtype=torch.float32)
    targets = torch.zeros((batch_size, max_envs), dtype=torch.float32)
    env_mask = torch.zeros((batch_size, max_envs), dtype=torch.bool)
    pair_weights = torch.zeros((batch_size, max_envs), dtype=torch.float32)
    target_baselines = torch.zeros((batch_size, max_envs), dtype=torch.float32)
    hybrid_names: list[str] = []
    env_name_lists: list[list[str]] = []
    split_names: list[str] = []

    for batch_index, sample in enumerate(batch):
        env_count = sample.env_vectors.shape[0]
        marker_values[batch_index] = torch.from_numpy(sample.marker_values.astype(np.int64, copy=False))
        env_vectors[batch_index, :env_count] = torch.from_numpy(sample.env_vectors.astype(np.float32, copy=False))
        targets[batch_index, :env_count] = torch.from_numpy(sample.targets.astype(np.float32, copy=False))
        env_mask[batch_index, :env_count] = True
        pair_weights[batch_index, :env_count] = torch.from_numpy(sample.replicate_counts.astype(np.float32, copy=False))
        target_baselines[batch_index, :env_count] = torch.from_numpy(sample.target_baselines.astype(np.float32, copy=False))
        hybrid_names.append(sample.hybrid)
        env_name_lists.append(list(sample.env_names))
        split_names.append(sample.split)

    return marker_values, env_vectors, targets, env_mask, pair_weights, target_baselines, hybrid_names, env_name_lists, split_names


class MaskedMultiHeadBipartiteAttention(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_marker = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_env = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_env = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_marker = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_marker_1 = nn.LayerNorm(hidden_dim)
        self.norm_marker_2 = nn.LayerNorm(hidden_dim)
        self.ff_marker = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.q_env = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_marker = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_marker = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_env = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_env_1 = nn.LayerNorm(hidden_dim)
        self.norm_env_2 = nn.LayerNorm(hidden_dim)
        self.ff_env = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = tensor.shape
        return tensor.view(batch_size, num_nodes, self.num_heads, self.head_dim)

    def forward(
        self,
        marker_states: torch.Tensor,
        env_states: torch.Tensor,
        env_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_marker = self._reshape_heads(self.q_marker(marker_states))
        k_env = self._reshape_heads(self.k_env(env_states))
        v_env = self._reshape_heads(self.v_env(env_states))

        q_env = self._reshape_heads(self.q_env(env_states))
        k_marker = self._reshape_heads(self.k_marker(marker_states))
        v_marker = self._reshape_heads(self.v_marker(marker_states))

        env_mask_expanded = env_mask[:, None, None, :]

        scores_marker_to_env = torch.einsum("bmhd,behd->bmhe", q_marker, k_env) * self.scale
        scores_marker_to_env = scores_marker_to_env.masked_fill(~env_mask_expanded, torch.finfo(scores_marker_to_env.dtype).min)
        attn_marker_to_env = torch.softmax(scores_marker_to_env, dim=-1)
        attn_marker_to_env = attn_marker_to_env.masked_fill(~env_mask_expanded, 0.0)
        marker_messages = torch.einsum("bmhe,behd->bmhd", attn_marker_to_env, v_env)
        marker_messages = marker_messages.reshape(marker_states.size(0), marker_states.size(1), self.hidden_dim)

        scores_env_to_marker = torch.einsum("behd,bmhd->behm", q_env, k_marker) * self.scale
        attn_env_to_marker = torch.softmax(scores_env_to_marker, dim=-1)
        env_mask_query = env_mask[:, :, None, None]
        attn_env_to_marker = attn_env_to_marker.masked_fill(~env_mask_query, 0.0)
        env_messages = torch.einsum("behm,bmhd->behd", attn_env_to_marker, v_marker)
        env_messages = env_messages.reshape(env_states.size(0), env_states.size(1), self.hidden_dim)

        marker_states = self.norm_marker_1(marker_states + self.out_marker(marker_messages))
        marker_states = self.norm_marker_2(marker_states + self.ff_marker(marker_states))

        env_states = self.norm_env_1(env_states + self.out_env(env_messages))
        env_states = self.norm_env_2(env_states + self.ff_env(env_states))
        env_states = env_states * env_mask.unsqueeze(-1)

        return marker_states, env_states, attn_env_to_marker


class MarkerEnvironmentGNN(nn.Module):
    def __init__(
        self,
        num_markers: int,
        env_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.value_embedding = nn.Embedding(4, hidden_dim)
        self.marker_id_embedding = nn.Embedding(num_markers, hidden_dim)
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [MaskedMultiHeadBipartiteAttention(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.register_buffer("marker_ids", torch.arange(num_markers, dtype=torch.long), persistent=False)

    def forward(
        self,
        marker_values: torch.Tensor,
        env_vectors: torch.Tensor,
        env_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        marker_values = marker_values.long().clamp(0, 3)
        marker_ids = self.marker_ids.unsqueeze(0).expand(marker_values.size(0), -1)
        marker_states = self.value_embedding(marker_values) + self.marker_id_embedding(marker_ids)
        env_states = self.env_encoder(env_vectors.float()) * env_mask.unsqueeze(-1)

        final_attention = None
        for layer in self.layers:
            marker_states, env_states, final_attention = layer(marker_states, env_states, env_mask)

        if final_attention is None:
            raise RuntimeError("Final attention tensor was not produced.")

        marker_attention = final_attention.mean(dim=2)
        marker_attention = marker_attention * env_mask.unsqueeze(-1)
        genotype_summary = torch.einsum("bem,bmd->bed", marker_attention, marker_states)
        combined = torch.cat([genotype_summary, env_states], dim=-1)
        predictions = self.mlp(combined).squeeze(-1) * env_mask

        if return_attention:
            return predictions, marker_attention
        return predictions


def get_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_model(num_markers: int, env_dim: int) -> tuple[nn.Module, nn.Module]:
    base_model = MarkerEnvironmentGNN(num_markers=num_markers, env_dim=env_dim)
    if torch.cuda.is_available():
        base_model = base_model.to(torch.device("cuda:0"))
        if torch.cuda.device_count() >= 2:
            parallel_model = nn.DataParallel(base_model, device_ids=[0, 1])
            return parallel_model, parallel_model.module
    return base_model, base_model


def masked_regression_losses(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    env_mask: torch.Tensor,
    pair_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = env_mask.float()
    weights = pair_weights.float() * mask
    weights = weights / weights.sum().clamp_min(EPS)

    squared_error = (predictions - targets) ** 2
    absolute_error = (predictions - targets).abs()

    mse = (squared_error * weights).sum()
    mae = (absolute_error * weights).sum()
    loss = (MSE_WEIGHT * mse) + (MAE_WEIGHT * mae)
    return loss, mse.detach(), mae.detach()


@torch.inference_mode()
def evaluate_model(
    loader: DataLoader,
    model: nn.Module,
    prepared: PreparedData,
) -> dict:
    device = get_training_device()
    model.eval()
    model = model.to(device)

    all_predictions_std: list[np.ndarray] = []
    all_targets_std: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_pair_weights: list[np.ndarray] = []
    rows: list[dict] = []

    for (
        marker_batch,
        env_batch,
        target_batch,
        env_mask,
        pair_weight_batch,
        target_baseline_batch,
        hybrid_names,
        env_name_lists,
        split_names,
    ) in loader:
        marker_batch = marker_batch.to(device, non_blocking=True)
        env_batch = env_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)
        env_mask = env_mask.to(device, non_blocking=True)
        pair_weight_batch = pair_weight_batch.to(device, non_blocking=True)

        predictions_std, marker_attention = model(marker_batch, env_batch, env_mask, return_attention=True)

        predictions_std_cpu = predictions_std.cpu().numpy()
        targets_std_cpu = target_batch.cpu().numpy()
        mask_cpu = env_mask.cpu().numpy().astype(bool)
        pair_weight_cpu = pair_weight_batch.cpu().numpy()
        target_baseline_cpu = target_baseline_batch.cpu().numpy()
        marker_attention_cpu = marker_attention.cpu().numpy()

        for batch_index, hybrid_name in enumerate(hybrid_names):
            env_count = int(mask_cpu[batch_index].sum())
            env_names = env_name_lists[batch_index][:env_count]
            split_name = split_names[batch_index]
            for env_index, env_name in enumerate(env_names):
                pred_std = float(predictions_std_cpu[batch_index, env_index])
                target_std = float(targets_std_cpu[batch_index, env_index])
                baseline = float(target_baseline_cpu[batch_index, env_index])
                pred = (pred_std * prepared.target_std) + prepared.target_mean + baseline
                target = (target_std * prepared.target_std) + prepared.target_mean + baseline
                rep_count = float(pair_weight_cpu[batch_index, env_index])
                attention_row = marker_attention_cpu[batch_index, env_index].astype(np.float32, copy=True)
                rows.append(
                    {
                        "Hybrid": hybrid_name,
                        "Env": env_name,
                        "split": split_name,
                        "replicate_count": rep_count,
                        "baseline_yield": baseline,
                        "mean_yield": target,
                        "predicted_yield": pred,
                        "prediction_std": pred_std,
                        "target_std": target_std,
                        "marker_attention": attention_row,
                    }
                )
                all_predictions_std.append(np.array([pred_std], dtype=np.float32))
                all_targets_std.append(np.array([target_std], dtype=np.float32))
                all_weights.append(attention_row[None, :])
                all_pair_weights.append(np.array([rep_count], dtype=np.float32))

    predictions_std_flat = np.concatenate(all_predictions_std, axis=0) if all_predictions_std else np.empty((0,), dtype=np.float32)
    targets_std_flat = np.concatenate(all_targets_std, axis=0) if all_targets_std else np.empty((0,), dtype=np.float32)
    weights_matrix = np.concatenate(all_weights, axis=0) if all_weights else np.empty((0, len(prepared.marker_columns)), dtype=np.float32)
    pair_weights_flat = np.concatenate(all_pair_weights, axis=0) if all_pair_weights else np.empty((0,), dtype=np.float32)

    if len(rows) == 0:
        metrics = {
            "loss": math.nan,
            "mse": math.nan,
            "mae": math.nan,
            "r2": math.nan,
        }
    else:
        baselines = np.array([row["baseline_yield"] for row in rows], dtype=np.float32)
        pred_original = (predictions_std_flat * prepared.target_std) + prepared.target_mean + baselines
        target_original = (targets_std_flat * prepared.target_std) + prepared.target_mean + baselines
        metrics = {
            "loss": float((MSE_WEIGHT * mean_squared_error(target_original, pred_original)) + (MAE_WEIGHT * mean_absolute_error(target_original, pred_original))),
            "mse": float(mean_squared_error(target_original, pred_original)),
            "mae": float(mean_absolute_error(target_original, pred_original)),
            "r2": float(r2_score(target_original, pred_original)),
        }

    return {
        "rows": rows,
        "weights_matrix": weights_matrix,
        "pair_weights": pair_weights_flat,
        "metrics": metrics,
    }


def compute_baselines(prepared: PreparedData) -> pd.DataFrame:
    train_df = prepared.train_pair_df.copy()
    val_df = prepared.val_pair_df.copy()
    global_mean = float(train_df["mean_yield"].mean())

    env_mean = train_df.groupby("Env")["mean_yield"].mean()
    hybrid_mean = train_df.groupby("Hybrid")["mean_yield"].mean()

    env_pred = val_df["Env"].map(env_mean).fillna(global_mean)
    hybrid_pred = val_df["Hybrid"].map(hybrid_mean).fillna(global_mean)
    additive_pred = env_pred + hybrid_pred - global_mean
    mean_pred = pd.Series(np.full(len(val_df), global_mean), index=val_df.index)

    baselines = [
        ("global_mean", mean_pred),
        ("env_mean", env_pred),
        ("hybrid_mean", hybrid_pred),
        ("env_plus_hybrid", additive_pred),
    ]

    records = []
    for baseline_name, predictions in baselines:
        mse = mean_squared_error(val_df["mean_yield"], predictions)
        mae = mean_absolute_error(val_df["mean_yield"], predictions)
        r2 = r2_score(val_df["mean_yield"], predictions)
        records.append({"baseline": baseline_name, "mse": mse, "mae": mae, "r2": r2})

    baseline_df = pd.DataFrame(records).sort_values("mse", kind="stable").reset_index(drop=True)
    return baseline_df


def train_model(prepared: PreparedData, epochs: int = EPOCHS) -> tuple[pd.DataFrame, nn.Module, pd.DataFrame]:
    train_dataset = GraphDataset(prepared.train_graphs)
    val_dataset = GraphDataset(prepared.val_graphs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_graph_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_graph_batch,
    )

    training_model, base_model = build_model(
        num_markers=len(prepared.marker_columns),
        env_dim=len(prepared.env_feature_columns),
    )
    device = get_training_device()
    optimizer = torch.optim.AdamW(training_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history: list[dict] = []
    best_state = copy.deepcopy(base_model.state_dict())
    best_val_mse = math.inf
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        training_model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_mae = 0.0
        step_count = 0

        for step_index, batch in enumerate(train_loader, start=1):
            marker_batch, env_batch, target_batch, env_mask, pair_weight_batch, _, _, _, _ = batch
            marker_batch = marker_batch.to(device, non_blocking=True)
            env_batch = env_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            env_mask = env_mask.to(device, non_blocking=True)
            pair_weight_batch = pair_weight_batch.to(device, non_blocking=True)

            predictions = training_model(marker_batch, env_batch, env_mask)
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

        if len(val_dataset) > 0:
            val_eval = evaluate_model(val_loader, base_model, prepared)
            val_metrics = val_eval["metrics"]
            val_loss = val_metrics["loss"]
            val_mse = val_metrics["mse"]
            val_mae = val_metrics["mae"]
            val_r2 = val_metrics["r2"]
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                best_state = copy.deepcopy(base_model.state_dict())
        else:
            val_loss = math.nan
            val_mse = math.nan
            val_mae = math.nan
            val_r2 = math.nan

        history.append(
            {
                "epoch": epoch,
                "train_loss_std": train_loss,
                "train_mse_std": train_mse_std,
                "train_mae_std": train_mae_std,
                "val_loss": val_loss,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "best_val_mse": best_val_mse,
                "best_epoch": best_epoch,
            }
        )
        print(
            f"Epoch {epoch:03d}/{epochs} | train_loss_std={train_loss:.6f} "
            f"| train_mse_std={train_mse_std:.6f} | train_mae_std={train_mae_std:.6f} "
            f"| val_mse={val_mse:.6f} | val_mae={val_mae:.6f} | val_r2={val_r2:.6f}",
            flush=True,
        )

    base_model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    baseline_df = compute_baselines(prepared)
    return history_df, base_model, baseline_df


def save_outputs(
    prepared: PreparedData,
    history_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    model: nn.Module,
    all_eval: dict,
    output_paths: OutputPaths,
    return_weight_dataframe: bool = False,
    save_predictions: bool = True,
    save_all_weights: bool = True,
) -> dict:
    history_df.to_csv(output_paths.history, index=False)
    baseline_df.to_csv(output_paths.baselines, index=False)
    torch.save(model.state_dict(), output_paths.model)

    prediction_rows = []
    weight_rows = []
    for row in all_eval["rows"]:
        prediction_rows.append(
            {
                "Hybrid": row["Hybrid"],
                "Env": row["Env"],
                "split": row["split"],
                "replicate_count": row["replicate_count"],
                "baseline_yield": row["baseline_yield"],
                "mean_yield": row["mean_yield"],
                "predicted_yield": row["predicted_yield"],
                "prediction_std": row["prediction_std"],
                "target_std": row["target_std"],
            }
        )
        weight_record = prediction_rows[-1].copy()
        weight_record["marker_attention"] = row["marker_attention"]
        weight_rows.append(weight_record)

    predictions_df = pd.DataFrame(prediction_rows)
    if save_predictions:
        predictions_df.to_csv(output_paths.predictions, index=False)

    if weight_rows:
        node_columns = [f"node_{index}" for index in range(1, len(prepared.marker_columns) + 1)]
        metadata_df = pd.DataFrame([{k: v for k, v in row.items() if k != "marker_attention"} for row in weight_rows])
        weights_matrix = np.stack([row["marker_attention"] for row in weight_rows], axis=0)
        weights_df = pd.concat([metadata_df, pd.DataFrame(weights_matrix, columns=node_columns)], axis=1)
    else:
        weights_df = pd.DataFrame()
        weights_matrix = np.empty((0, len(prepared.marker_columns)), dtype=np.float32)
        node_columns = [f"node_{index}" for index in range(1, len(prepared.marker_columns) + 1)]
    if save_all_weights:
        weights_df.to_csv(output_paths.all_weights, index=False, compression="gzip")

    pair_weights = all_eval["pair_weights"]
    if len(weights_matrix) == 0:
        average_weights = np.zeros(len(prepared.marker_columns), dtype=np.float32)
    else:
        normalized_pair_weights = pair_weights / np.clip(pair_weights.sum(), EPS, None)
        average_weights = (weights_matrix * normalized_pair_weights[:, None]).sum(axis=0)

    ranking_df = pd.DataFrame(
        {
            "node_index": np.arange(1, len(prepared.marker_columns) + 1),
            "marker_id": prepared.marker_columns,
            "avg_weight": average_weights,
        }
    )
    ranking_df = ranking_df.sort_values(
        ["avg_weight", "node_index"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))
    ranking_df["node_label"] = "node_" + ranking_df["node_index"].astype(str)
    ranking_df.to_csv(output_paths.ranking, index=False)
    output_paths.node_list.write_text("\n".join(ranking_df["node_label"]) + "\n")

    return {
        "history_df": history_df,
        "baseline_df": baseline_df,
        "predictions_df": predictions_df,
        "all_weights_df": weights_df if return_weight_dataframe else None,
        "ranking_df": ranking_df,
        "all_weights_path": output_paths.all_weights,
    }


def run_gnn_experiment(
    final_path: Path = FINAL_PATH,
    genotype_path: Path = GENOTYPE_PATH,
    env_path: Path = ENV_PATH,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    use_env_residual: bool = False,
    epochs: int = EPOCHS,
    return_weight_dataframe: bool = False,
    seed: int = SEED,
    split_seed: int = SEED,
    save_predictions: bool = True,
    save_all_weights: bool = True,
) -> dict:
    set_seed(seed)
    output_paths = build_output_paths(output_prefix=output_prefix)
    prepared = load_prepared_data(
        final_path=final_path,
        genotype_path=genotype_path,
        env_path=env_path,
        use_env_residual=use_env_residual,
        split_seed=split_seed,
    )
    history_df, model, baseline_df = train_model(prepared, epochs=epochs)

    all_loader = DataLoader(
        GraphDataset(prepared.all_graphs),
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_graph_batch,
    )
    all_eval = evaluate_model(all_loader, model, prepared)
    results = save_outputs(
        prepared,
        history_df,
        baseline_df,
        model,
        all_eval,
        output_paths=output_paths,
        return_weight_dataframe=return_weight_dataframe,
        save_predictions=save_predictions,
        save_all_weights=save_all_weights,
    )

    best_row = history_df.sort_values("val_mse", kind="stable").iloc[0]
    all_metrics = all_eval["metrics"]

    print(f"Using training device(s): {torch.cuda.device_count()} GPU(s) available", flush=True)
    print(f"Training seed: {seed}", flush=True)
    print(f"Validation split seed: {split_seed}", flush=True)
    print(f"Target mode: {'env_residual' if use_env_residual else 'raw_yield'}", flush=True)
    print(f"Unique genotype graphs: {len(prepared.all_graphs)}", flush=True)
    print(f"Unique genotype-env pairs: {len(prepared.all_pair_df)}", flush=True)
    print(f"Marker nodes per graph: {len(prepared.marker_columns)}", flush=True)
    print(f"Environment feature dimension: {len(prepared.env_feature_columns)}", flush=True)
    print(f"Best validation epoch: {int(best_row['epoch'])}", flush=True)
    print(
        f"Best validation metrics | mse={best_row['val_mse']:.6f} | mae={best_row['val_mae']:.6f} | r2={best_row['val_r2']:.6f}",
        flush=True,
    )
    print(
        f"All-pair metrics with best model | mse={all_metrics['mse']:.6f} | mae={all_metrics['mae']:.6f} | r2={all_metrics['r2']:.6f}",
        flush=True,
    )
    print("Validation baselines:", flush=True)
    print(baseline_df.to_string(index=False), flush=True)
    print(f"Training history saved to: {output_paths.history.name}", flush=True)
    print(f"Baselines saved to: {output_paths.baselines.name}", flush=True)
    print(f"Model checkpoint saved to: {output_paths.model.name}", flush=True)
    if save_predictions:
        print(f"Predictions saved to: {output_paths.predictions.name}", flush=True)
    if save_all_weights:
        print(f"All pair weights saved to: {output_paths.all_weights.name}", flush=True)
    print(f"Ranking saved to: {output_paths.ranking.name}", flush=True)
    print(f"Node list saved to: {output_paths.node_list.name}", flush=True)
    print(results["ranking_df"].head(10).to_string(index=False), flush=True)

    return results


def run_default_training(
    seed: int = SEED,
    split_seed: int = SEED,
) -> dict:
    return run_gnn_experiment(seed=seed, split_seed=split_seed)


if __name__ == "__main__":
    run_default_training()
