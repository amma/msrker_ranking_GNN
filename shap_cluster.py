from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from weather_encoder import ALL_WEATHER_FEATURES, ENV_DIM, PerVariableLSTMEnvEncoder, load_weather_tensor
from shap_compare import (
    ATTENTION_PATH,
    CLASSIFIER_EPOCHS,
    ENCODER_MODEL_PATH,
    CompressionMLP,
    build_env_pair_df,
    extract_per_variable_hidden,
    train_classifier,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CLUSTER_PATH = BASE_DIR / "weather_kmeans_best_5feature_assignments_labeled.csv"
RESULTS_DIR = BASE_DIR / "env_encoder_comparison" / "shap_vs_attention" / "per_cluster"

N_SEEDS = 20
SEEDS = list(range(N_SEEDS))

CLUSTER_LABELS = {
    0: "cool, temperate, windy, moderate humidity",
    1: "warm, humid, wet, low wind",
    2: "cool, humid, low wind, low radiation",
    3: "dry, high-sun, low-pressure, windy",
}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This script expects a second GPU (cuda:1) to be available.")
    device = torch.device("cuda:1")
    print(f"Using device: {device} ({torch.cuda.get_device_name(1)})", flush=True)

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

    cluster_df = pd.read_csv(CLUSTER_PATH)
    cluster_df["Env"] = cluster_df["Env"].astype(str)
    cluster_lookup = cluster_df.drop_duplicates("Env").set_index("Env")["cluster"]
    env_clusters = np.array([int(cluster_lookup[env]) for env in ordered_envs])
    for c in sorted(CLUSTER_LABELS):
        print(f"Cluster {c} ({CLUSTER_LABELS[c]}): {(env_clusters == c).sum()} environments", flush=True)

    attention_df = pd.read_csv(ATTENTION_PATH)
    attention_df["Env"] = attention_df["Env"].astype(str)
    attention_df = attention_df.set_index("Env").loc[ordered_envs].reset_index()

    env_yield_mean = pair_df.groupby("Env")["mean_yield"].mean()
    median_yield = float(env_yield_mean.median())
    labels = np.array(
        [1.0 if env_yield_mean[env] > median_yield else 0.0 for env in ordered_envs], dtype=np.float32
    )
    labels_t = torch.from_numpy(labels).to(device)
    indices = np.arange(len(ordered_envs))

    print(f"\n{'=' * 80}\nTraining {N_SEEDS} classifiers, saving full per-environment SHAP each time\n{'=' * 80}", flush=True)
    all_seed_shap = np.zeros((N_SEEDS, len(ordered_envs), len(ALL_WEATHER_FEATURES)), dtype=np.float64)
    seed_val_acc = []
    for i, seed in enumerate(SEEDS):
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
        with torch.inference_mode():
            features_all = classifier.compress_features(stacked).detach()
        background = features_all[train_idx_t]
        explainer = shap.GradientExplainer(classifier.mlp, background)
        raw_shap_values = explainer.shap_values(features_all)
        shap_values = np.asarray(raw_shap_values).reshape(len(ordered_envs), len(ALL_WEATHER_FEATURES))
        all_seed_shap[i] = shap_values
        seed_val_acc.append(val_acc)
        print(f"Seed {seed:02d}/{N_SEEDS - 1}: val_acc={val_acc:.3f}", flush=True)

    np.save(RESULTS_DIR / "all_seed_shap_values.npy", all_seed_shap)
    pd.DataFrame({"seed": SEEDS, "val_acc": seed_val_acc}).to_csv(RESULTS_DIR / "seed_val_acc.csv", index=False)

    cluster_summaries = {}
    for cluster_id, cluster_label in CLUSTER_LABELS.items():
        mask = env_clusters == cluster_id
        n_envs = int(mask.sum())
        print(f"\n{'=' * 80}\nCluster {cluster_id}: {cluster_label} ({n_envs} environments)\n{'=' * 80}", flush=True)

        cluster_attention = attention_df.loc[mask, ALL_WEATHER_FEATURES].mean(axis=0)
        attn_rank_df = pd.DataFrame({"variable": ALL_WEATHER_FEATURES, "mean_attention": cluster_attention.values})
        attn_rank_df = attn_rank_df.sort_values("mean_attention", ascending=False).reset_index(drop=True)
        attn_rank_df["attention_rank"] = attn_rank_df.index + 1
        attn_rank_lookup = attn_rank_df.set_index("variable")["attention_rank"]

        seed_rhos = []
        seed_rankings = []
        for i, seed in enumerate(SEEDS):
            cluster_shap = all_seed_shap[i][mask, :]
            mean_abs_shap = np.abs(cluster_shap).mean(axis=0)
            shap_rank_df = pd.DataFrame({"variable": ALL_WEATHER_FEATURES, "mean_abs_shap": mean_abs_shap})
            shap_rank_df = shap_rank_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            shap_rank_df["shap_rank"] = shap_rank_df.index + 1
            shap_rank_df["attention_rank"] = shap_rank_df["variable"].map(attn_rank_lookup)
            rho, p_value = spearmanr(shap_rank_df["shap_rank"], shap_rank_df["attention_rank"])
            seed_rhos.append(rho)
            seed_rankings.append(shap_rank_df)

        seed_rhos_arr = np.array(seed_rhos, dtype=np.float64)
        best_i = int(np.nanargmax(seed_rhos_arr))
        best_seed = SEEDS[best_i]
        best_rho = float(seed_rhos_arr[best_i])
        best_ranking = seed_rankings[best_i].merge(attn_rank_df[["variable", "mean_attention"]], on="variable")
        best_ranking = best_ranking.sort_values("shap_rank").reset_index(drop=True)
        best_ranking.to_csv(RESULTS_DIR / f"cluster{cluster_id}_best_run_ranking.csv", index=False)

        pd.DataFrame({"seed": SEEDS, "spearman_rho": seed_rhos_arr}).to_csv(
            RESULTS_DIR / f"cluster{cluster_id}_seed_rho.csv", index=False
        )

        print(
            f"rho across 20 seeds: mean={np.nanmean(seed_rhos_arr):.3f} "
            f"min={np.nanmin(seed_rhos_arr):.3f} max={np.nanmax(seed_rhos_arr):.3f}",
            flush=True,
        )
        print(f"Best seed: {best_seed} (rho={best_rho:.3f})", flush=True)
        print(best_ranking.to_string(index=False), flush=True)

        cluster_summaries[cluster_id] = {
            "label": cluster_label,
            "n_envs": n_envs,
            "best_seed": best_seed,
            "best_rho": best_rho,
            "rho_mean": float(np.nanmean(seed_rhos_arr)),
            "rho_min": float(np.nanmin(seed_rhos_arr)),
            "rho_max": float(np.nanmax(seed_rhos_arr)),
        }

    (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(cluster_summaries, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. Results written to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
