from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from weather_encoder import ALL_WEATHER_FEATURES, ENV_DIM
from shap_compare import (
    ATTENTION_PATH,
    CLASSIFIER_EPOCHS,
    ENCODER_MODEL_PATH,
    CompressionMLP,
    build_env_pair_df,
    extract_per_variable_hidden,
    train_classifier,
)
from weather_encoder import PerVariableLSTMEnvEncoder, load_weather_tensor

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "env_encoder_comparison" / "shap_vs_attention" / "stability"

N_SEEDS = 20
SEEDS = list(range(N_SEEDS))
MIN_VAL_ACC = 0.60


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    pair_df = build_env_pair_df()
    env_names = sorted(pair_df["Env"].unique())
    weather_tensor, weather_lengths, env_name_to_index = load_weather_tensor(env_names, ALL_WEATHER_FEATURES)
    ordered_envs = sorted(env_name_to_index, key=env_name_to_index.get)

    full_state = torch.load(ENCODER_MODEL_PATH, map_location="cpu", weights_only=False)
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

    env_yield_mean = pair_df.groupby("Env")["mean_yield"].mean()
    median_yield = float(env_yield_mean.median())
    labels = np.array(
        [1.0 if env_yield_mean[env] > median_yield else 0.0 for env in ordered_envs], dtype=np.float32
    )
    labels_t = torch.from_numpy(labels).to(device)

    attention_df = pd.read_csv(ATTENTION_PATH)
    mean_attention = attention_df[ALL_WEATHER_FEATURES].mean(axis=0)
    attention_rank_df = pd.DataFrame({"variable": ALL_WEATHER_FEATURES, "mean_attention": mean_attention.values})
    attention_rank_df = attention_rank_df.sort_values("mean_attention", ascending=False).reset_index(drop=True)
    attention_rank_df["attention_rank"] = attention_rank_df.index + 1
    attention_rank_lookup = attention_rank_df.set_index("variable")["attention_rank"]

    per_seed_records = []
    rank_rows = []
    indices = np.arange(len(ordered_envs))

    for seed in SEEDS:
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=labels)
        train_idx_t = torch.from_numpy(train_idx).to(device)
        val_idx_t = torch.from_numpy(val_idx).to(device)

        best_state, best_epoch, val_acc, val_loss = train_classifier(
            stacked, labels_t, train_idx_t, val_idx_t,
            len(ALL_WEATHER_FEATURES), ENV_DIM, device, CLASSIFIER_EPOCHS, seed=seed,
        )

        classifier = CompressionMLP(num_features=len(ALL_WEATHER_FEATURES), hidden_dim=ENV_DIM).to(device)
        classifier.load_state_dict(best_state)
        classifier.eval()

        passed_validity = val_acc >= MIN_VAL_ACC
        rho = float("nan")
        p_value = float("nan")

        if passed_validity:
            with torch.inference_mode():
                features_all = classifier.compress_features(stacked).detach()
            background = features_all[train_idx_t]
            explainer = shap.GradientExplainer(classifier.mlp, background)
            raw_shap_values = explainer.shap_values(features_all)
            shap_values = np.asarray(raw_shap_values).reshape(len(ordered_envs), len(ALL_WEATHER_FEATURES))
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            shap_rank_df = pd.DataFrame({"variable": ALL_WEATHER_FEATURES, "mean_abs_shap": mean_abs_shap})
            shap_rank_df = shap_rank_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            shap_rank_df["shap_rank"] = shap_rank_df.index + 1
            shap_rank_df["attention_rank"] = shap_rank_df["variable"].map(attention_rank_lookup)

            rho, p_value = spearmanr(shap_rank_df["shap_rank"], shap_rank_df["attention_rank"])
            for _, row in shap_rank_df.iterrows():
                rank_rows.append(
                    {"seed": seed, "variable": row["variable"], "shap_rank": int(row["shap_rank"]), "mean_abs_shap": row["mean_abs_shap"]}
                )

        per_seed_records.append(
            {
                "seed": seed, "best_epoch": best_epoch, "val_acc": val_acc, "val_loss": val_loss,
                "passed_validity": passed_validity, "spearman_rho": rho, "spearman_p": p_value,
            }
        )
        if passed_validity:
            print(f"Seed {seed:02d}: val_acc={val_acc:.3f} (OK) | rho={rho:.3f}", flush=True)
        else:
            print(f"Seed {seed:02d}: val_acc={val_acc:.3f} (DROPPED, below {MIN_VAL_ACC})", flush=True)

    per_seed_df = pd.DataFrame(per_seed_records)
    per_seed_df.to_csv(RESULTS_DIR / "per_seed_results.csv", index=False)

    valid_df = per_seed_df[per_seed_df["passed_validity"]].copy()
    print(f"\n{len(valid_df)}/{N_SEEDS} seeds passed the validity bar (val_acc >= {MIN_VAL_ACC}).", flush=True)
    print(
        f"Spearman rho across valid seeds: mean={valid_df['spearman_rho'].mean():.3f} "
        f"median={valid_df['spearman_rho'].median():.3f} "
        f"min={valid_df['spearman_rho'].min():.3f} max={valid_df['spearman_rho'].max():.3f}",
        flush=True,
    )

    rank_long_df = pd.DataFrame(rank_rows)
    rank_long_df.to_csv(RESULTS_DIR / "rank_persistence_long.csv", index=False)

    persistence_rows = []
    for variable in ALL_WEATHER_FEATURES:
        ranks = rank_long_df.loc[rank_long_df["variable"] == variable, "shap_rank"]
        persistence_rows.append(
            {
                "variable": variable,
                "attention_rank": int(attention_rank_lookup[variable]),
                "shap_rank_mean": ranks.mean(),
                "shap_rank_std": ranks.std(),
                "shap_rank_min": int(ranks.min()),
                "shap_rank_max": int(ranks.max()),
                "n_runs": int(ranks.shape[0]),
            }
        )
    persistence_df = pd.DataFrame(persistence_rows).sort_values("shap_rank_mean").reset_index(drop=True)
    persistence_df.to_csv(RESULTS_DIR / "rank_persistence_summary.csv", index=False)
    print("\nRank persistence across valid seeds (sorted by mean SHAP rank):", flush=True)
    print(persistence_df.to_string(index=False), flush=True)

    best_row = valid_df.sort_values("spearman_rho", ascending=False).iloc[0]
    best_seed = int(best_row["seed"])
    best_rho = float(best_row["spearman_rho"])
    print(f"\nBest-matching seed: {best_seed} (rho={best_rho:.3f}, val_acc={best_row['val_acc']:.3f})", flush=True)

    best_ranking_df = rank_long_df[rank_long_df["seed"] == best_seed].copy()
    best_ranking_df["attention_rank"] = best_ranking_df["variable"].map(attention_rank_lookup)
    best_ranking_df = best_ranking_df.sort_values("shap_rank").reset_index(drop=True)
    best_ranking_df.to_csv(RESULTS_DIR / "best_run_ranking.csv", index=False)

    manifest = {
        "n_seeds_attempted": N_SEEDS,
        "n_seeds_passed_validity": int(len(valid_df)),
        "min_val_acc_bar": MIN_VAL_ACC,
        "rho_mean": float(valid_df["spearman_rho"].mean()),
        "rho_median": float(valid_df["spearman_rho"].median()),
        "rho_std": float(valid_df["spearman_rho"].std()),
        "rho_min": float(valid_df["spearman_rho"].min()),
        "rho_max": float(valid_df["spearman_rho"].max()),
        "best_seed": best_seed,
        "best_rho": best_rho,
        "best_val_acc": float(best_row["val_acc"]),
        "original_seed42_rho": 0.31176470588235294,
    }
    (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. Results written to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
