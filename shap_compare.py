from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.nn.utils.rnn import pack_padded_sequence

from gnn_model import aggregate_pairs, set_seed
from weather_encoder import (
    ALL_WEATHER_FEATURES,
    ENV_DIM,
    GENOTYPE_PATH,
    SEED,
    TRAIT_PATH,
    WEATHER_PATH,
    PerVariableLSTMEnvEncoder,
    load_weather_tensor,
)
from cluster_train import select_top_and_random_genotypes

BASE_DIR = Path(__file__).resolve().parent.parent
ENCODER_MODEL_PATH = BASE_DIR / "env_encoder_comparison" / "all_16_variables" / "per_variable_model.pt"
ATTENTION_PATH = BASE_DIR / "env_encoder_comparison" / "all_16_variables" / "per_variable_attention_by_env.csv"
RESULTS_DIR = BASE_DIR / "env_encoder_comparison" / "shap_vs_attention"

CLASSIFIER_EPOCHS = 300
CV_FOLDS = 5
WEIGHT_DECAY = 1e-2
DROPOUT = 0.3


class CompressionMLP(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.compress = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_features)])
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 8),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(8, 1),
        )

    def compress_features(self, stacked: torch.Tensor) -> torch.Tensor:
        scalars = [self.compress[i](stacked[:, i, :]) for i in range(stacked.shape[1])]
        return torch.cat(scalars, dim=1)

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.compress_features(stacked)).squeeze(-1)


def train_classifier(
    stacked: torch.Tensor,
    labels_t: torch.Tensor,
    train_idx_t: torch.Tensor,
    val_idx_t: torch.Tensor,
    num_features: int,
    hidden_dim: int,
    device: torch.device,
    epochs: int,
    seed: int = SEED,
) -> tuple[dict, int, float, float]:
    torch.manual_seed(seed)
    classifier = CompressionMLP(num_features=num_features, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(stacked[train_idx_t])
        loss = loss_fn(logits, labels_t[train_idx_t])
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.inference_mode():
            val_logits = classifier(stacked[val_idx_t])
            val_loss = float(loss_fn(val_logits, labels_t[val_idx_t]).item())
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = float((val_pred == labels_t[val_idx_t]).float().mean().item())

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in classifier.state_dict().items()}

    return best_state, best_epoch, best_val_acc, best_val_loss


def build_env_pair_df() -> pd.DataFrame:
    trait_df = pd.read_csv(TRAIT_PATH).copy()
    genotype_df = pd.read_csv(GENOTYPE_PATH).copy()
    genotype_df = genotype_df.rename(columns={genotype_df.columns[0]: "Hybrid"})
    trait_df["Hybrid"] = trait_df["Hybrid"].astype(str)
    trait_df["Env"] = trait_df["Env"].astype(str)
    genotype_df["Hybrid"] = genotype_df["Hybrid"].astype(str)
    genotype_lookup = genotype_df.drop_duplicates("Hybrid").set_index("Hybrid")

    valid_trait_df = trait_df[trait_df["Hybrid"].isin(genotype_lookup.index)].copy()
    _, candidate_df = select_top_and_random_genotypes(valid_trait_df)

    weather_env_ids = set(pd.read_csv(WEATHER_PATH, usecols=["Env"])["Env"].astype(str).unique())
    missing_weather = sorted(set(candidate_df["Env"].unique()) - weather_env_ids)
    if missing_weather:
        candidate_df = candidate_df[~candidate_df["Env"].isin(missing_weather)].reset_index(drop=True)

    pair_df = aggregate_pairs(candidate_df)
    pair_df = pair_df[pair_df["Hybrid"].isin(genotype_lookup.index)].reset_index(drop=True)
    return pair_df


@torch.inference_mode()
def extract_per_variable_hidden(
    encoder: PerVariableLSTMEnvEncoder, weather: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    lengths_cpu = lengths.cpu()
    per_variable_states = []
    for feature_index, lstm in enumerate(encoder.lstms):
        single = weather[:, :, feature_index : feature_index + 1]
        packed = pack_padded_sequence(single, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = lstm(packed)
        per_variable_states.append(hidden[-1])
    return torch.stack(per_variable_states, dim=1)


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    pair_df = build_env_pair_df()
    env_names = sorted(pair_df["Env"].unique())
    print(f"Environments: {len(env_names)}", flush=True)

    weather_tensor, weather_lengths, env_name_to_index = load_weather_tensor(env_names, ALL_WEATHER_FEATURES)
    ordered_envs = sorted(env_name_to_index, key=env_name_to_index.get)

    full_state = torch.load(ENCODER_MODEL_PATH, map_location="cpu")
    encoder_state = {k[len("encoder.") :]: v for k, v in full_state.items() if k.startswith("encoder.")}
    encoder = PerVariableLSTMEnvEncoder(num_features=len(ALL_WEATHER_FEATURES), hidden_dim=ENV_DIM)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder = encoder.to(device)

    weather_tensor = weather_tensor.to(device)
    weather_lengths = weather_lengths.to(device)

    stacked = extract_per_variable_hidden(encoder, weather_tensor, weather_lengths)
    print(f"Extracted frozen per-variable hidden states: {tuple(stacked.shape)}", flush=True)

    env_yield_mean = pair_df.groupby("Env")["mean_yield"].mean()
    median_yield = float(env_yield_mean.median())
    labels = np.array(
        [1.0 if env_yield_mean[env] > median_yield else 0.0 for env in ordered_envs], dtype=np.float32
    )
    print(
        f"Median env mean yield: {median_yield:.4f} | class balance: "
        f"{int(labels.sum())} high / {int(len(labels) - labels.sum())} low",
        flush=True,
    )

    labels_t = torch.from_numpy(labels).to(device)
    majority_baseline = float(max(labels.mean(), 1 - labels.mean()))

    print(f"\n{'=' * 80}\n{CV_FOLDS}-fold cross-validation (validity check)\n{'=' * 80}", flush=True)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_records = []
    for fold_index, (fold_train_idx, fold_val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        fold_train_idx_t = torch.from_numpy(fold_train_idx).to(device)
        fold_val_idx_t = torch.from_numpy(fold_val_idx).to(device)
        _, best_epoch, best_val_acc, best_val_loss = train_classifier(
            stacked, labels_t, fold_train_idx_t, fold_val_idx_t,
            len(ALL_WEATHER_FEATURES), ENV_DIM, device, CLASSIFIER_EPOCHS,
        )
        cv_records.append({"fold": fold_index, "best_epoch": best_epoch, "val_acc": best_val_acc, "val_loss": best_val_loss})
        print(f"Fold {fold_index}/{CV_FOLDS}: best_epoch={best_epoch} val_acc={best_val_acc:.3f} val_loss={best_val_loss:.4f}", flush=True)

    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    cv_mean_acc = float(cv_df["val_acc"].mean())
    cv_std_acc = float(cv_df["val_acc"].std())
    median_best_epoch = int(cv_df["best_epoch"].median())
    print(
        f"\nCV val accuracy: {cv_mean_acc:.3f} +/- {cv_std_acc:.3f} (majority baseline {majority_baseline:.3f}) "
        f"| median best_epoch={median_best_epoch}",
        flush=True,
    )

    indices = np.arange(len(ordered_envs))
    final_train_idx, final_val_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=labels)
    final_train_idx_t = torch.from_numpy(final_train_idx).to(device)
    final_val_idx_t = torch.from_numpy(final_val_idx).to(device)
    best_state, final_best_epoch, final_val_acc, final_val_loss = train_classifier(
        stacked, labels_t, final_train_idx_t, final_val_idx_t,
        len(ALL_WEATHER_FEATURES), ENV_DIM, device, max(median_best_epoch, 1),
    )
    classifier = CompressionMLP(num_features=len(ALL_WEATHER_FEATURES), hidden_dim=ENV_DIM).to(device)
    classifier.load_state_dict(best_state)
    classifier.eval()
    print(
        f"\nFinal classifier (for SHAP): trained {median_best_epoch} epochs on an 80/20 split, "
        f"val_acc={final_val_acc:.3f}",
        flush=True,
    )

    with torch.inference_mode():
        features_all = classifier.compress_features(stacked).detach()

    mlp = classifier.mlp
    background = features_all[final_train_idx_t]
    explainer = shap.GradientExplainer(mlp, background)
    raw_shap_values = explainer.shap_values(features_all)
    shap_values = np.asarray(raw_shap_values).reshape(len(ordered_envs), len(ALL_WEATHER_FEATURES))

    shap_df = pd.DataFrame(shap_values, columns=ALL_WEATHER_FEATURES)
    shap_df.insert(0, "Env", ordered_envs)
    shap_df.to_csv(RESULTS_DIR / "shap_values_by_env.csv", index=False)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_rank_df = pd.DataFrame({"variable": ALL_WEATHER_FEATURES, "mean_abs_shap": mean_abs_shap})
    shap_rank_df = shap_rank_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_rank_df["shap_rank"] = shap_rank_df.index + 1

    attention_df = pd.read_csv(ATTENTION_PATH)
    mean_attention = attention_df[ALL_WEATHER_FEATURES].mean(axis=0)
    attention_rank_df = pd.DataFrame({"variable": ALL_WEATHER_FEATURES, "mean_attention": mean_attention.values})
    attention_rank_df = attention_rank_df.sort_values("mean_attention", ascending=False).reset_index(drop=True)
    attention_rank_df["attention_rank"] = attention_rank_df.index + 1

    comparison_df = shap_rank_df.merge(attention_rank_df, on="variable")
    comparison_df["rank_diff"] = comparison_df["attention_rank"] - comparison_df["shap_rank"]
    comparison_df = comparison_df.sort_values("shap_rank").reset_index(drop=True)
    comparison_df.to_csv(RESULTS_DIR / "shap_vs_attention_ranking.csv", index=False)

    correlation, p_value = spearmanr(comparison_df["shap_rank"], comparison_df["attention_rank"])
    print(f"\nSpearman rank correlation (SHAP vs attention): rho={correlation:.4f}, p={p_value:.4f}", flush=True)
    print(comparison_df.to_string(index=False), flush=True)

    manifest = {
        "seed": SEED,
        "classifier_epochs_budget": CLASSIFIER_EPOCHS,
        "cv_folds": CV_FOLDS,
        "num_environments": len(ordered_envs),
        "num_final_train": int(len(final_train_idx)),
        "num_final_val": int(len(final_val_idx)),
        "median_env_yield": median_yield,
        "majority_class_baseline": majority_baseline,
        "cv_val_acc_mean": cv_mean_acc,
        "cv_val_acc_std": cv_std_acc,
        "cv_median_best_epoch": median_best_epoch,
        "final_model_epochs_trained": median_best_epoch,
        "final_model_val_acc": final_val_acc,
        "final_model_val_loss": final_val_loss,
        "spearman_rho": float(correlation),
        "spearman_p_value": float(p_value),
        "encoder_source": str(ENCODER_MODEL_PATH),
    }
    (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. Results written to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
