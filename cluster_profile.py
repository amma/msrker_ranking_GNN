from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
WEATHER_PATH = BASE_DIR / "4_Training_Weather_Data_2014_2023_full_year.csv"
ASSIGNMENTS_PATH = BASE_DIR / "weather_kmeans_best_5feature_assignments.csv"

PROFILE_PATH = BASE_DIR / "weather_kmeans_best_5feature_cluster_profile_all_weather.csv"
ZSCORE_PATH = BASE_DIR / "weather_kmeans_best_5feature_cluster_profile_zscores.csv"
EVIDENCE_PATH = BASE_DIR / "weather_kmeans_best_5feature_cluster_label_evidence.csv"
HEATMAP_PATH = BASE_DIR / "weather_kmeans_best_5feature_cluster_profile_heatmap.png"
FULL_RAW_HEATMAP_PATH = (
    BASE_DIR / "weather_kmeans_best_5feature_cluster_profile_heatmap_all16_raw_features.png"
)
COUNTS_PATH = BASE_DIR / "weather_kmeans_best_5feature_cluster_counts.png"

ENV_COLUMN = "Env"
DATE_COLUMN = "Date"
CLUSTER_COLUMN = "cluster"

CLUSTER_FEATURES = [
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_PAR_TOT",
    "T2M_MAX",
    "PRECTOTCORR",
    "T2M_MIN",
]

PRESENTATION_COLUMNS = [
    "n_envs",
    "n_records",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "T2MWET",
    "T2MDEW",
    "QV2M",
    "RH2M",
    "PRECTOTCORR",
    "annual_precip_mm_est",
    "ALLSKY_SFC_PAR_TOT",
    "ALLSKY_SFC_SW_DNI",
    "PS",
    "WS2M",
    "diurnal_range_C",
]

FEATURE_LABELS = {
    "T2M": "mean temperature",
    "T2M_MAX": "maximum temperature",
    "T2M_MIN": "minimum temperature",
    "T2MWET": "wet-bulb temperature",
    "T2MDEW": "dew point",
    "QV2M": "specific humidity",
    "RH2M": "relative humidity",
    "PRECTOTCORR": "precipitation",
    "annual_precip_mm_est": "annual precipitation estimate",
    "ALLSKY_SFC_PAR_TOT": "PAR",
    "ALLSKY_SFC_SW_DNI": "direct radiation",
    "ALLSKY_SFC_SW_DWN": "shortwave radiation",
    "PS": "surface pressure",
    "WS2M": "wind speed",
    "GWETTOP": "surface soil wetness",
    "GWETROOT": "root-zone soil wetness",
    "GWETPROF": "profile soil wetness",
    "diurnal_range_C": "diurnal temperature range",
}

FEATURE_DIRECTIONS = {
    "T2M": ("cooler", "warmer"),
    "T2M_MAX": ("lower maximum temperature", "higher maximum temperature"),
    "T2M_MIN": ("lower minimum temperature", "higher minimum temperature"),
    "T2MWET": ("lower wet-bulb temperature", "higher wet-bulb temperature"),
    "T2MDEW": ("lower dew point", "higher dew point"),
    "QV2M": ("lower specific humidity", "higher specific humidity"),
    "RH2M": ("lower relative humidity", "higher relative humidity"),
    "PRECTOTCORR": ("drier", "wetter"),
    "annual_precip_mm_est": ("lower annual precipitation", "higher annual precipitation"),
    "ALLSKY_SFC_PAR_TOT": ("lower PAR", "higher PAR"),
    "ALLSKY_SFC_SW_DNI": ("lower direct radiation", "higher direct radiation"),
    "ALLSKY_SFC_SW_DWN": ("lower shortwave radiation", "higher shortwave radiation"),
    "PS": ("lower pressure", "higher pressure"),
    "WS2M": ("lower wind", "higher wind"),
    "GWETTOP": ("lower surface soil wetness", "higher surface soil wetness"),
    "GWETROOT": ("lower root-zone soil wetness", "higher root-zone soil wetness"),
    "GWETPROF": ("lower profile soil wetness", "higher profile soil wetness"),
    "diurnal_range_C": ("smaller diurnal range", "larger diurnal range"),
}

SUGGESTED_LABELS = {
    0: "Cool windy environments with moderate moisture and radiation",
    1: "Warm humid wet environments with low wind",
    2: "Cool humid low-radiation environments with very low wind",
    3: "Dry high-radiation low-pressure windy environments",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    weather = pd.read_csv(WEATHER_PATH).copy()
    assignments = pd.read_csv(ASSIGNMENTS_PATH).copy()

    weather[ENV_COLUMN] = weather[ENV_COLUMN].astype(str)
    assignments[ENV_COLUMN] = assignments[ENV_COLUMN].astype(str)

    weather_features = [
        column for column in weather.columns if column not in {ENV_COLUMN, DATE_COLUMN}
    ]
    for column in weather_features:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    if weather[weather_features].isna().any().any():
        missing = weather[weather_features].isna().sum()
        raise ValueError(f"Missing weather values:\n{missing[missing.gt(0)]}")

    return weather, assignments, weather_features


def build_environment_means(
    weather: pd.DataFrame,
    assignments: pd.DataFrame,
    weather_features: list[str],
) -> pd.DataFrame:
    env_means = (
        weather.groupby(ENV_COLUMN, as_index=False)[weather_features]
        .mean()
        .sort_values(ENV_COLUMN, kind="stable")
        .reset_index(drop=True)
    )
    env_counts = weather.groupby(ENV_COLUMN).size().rename("record_count").reset_index()
    env_means = env_means.merge(env_counts, on=ENV_COLUMN, how="left")
    env_means = env_means.merge(
        assignments[[ENV_COLUMN, CLUSTER_COLUMN]], on=ENV_COLUMN, how="left"
    )
    if env_means[CLUSTER_COLUMN].isna().any():
        missing_envs = env_means.loc[env_means[CLUSTER_COLUMN].isna(), ENV_COLUMN].tolist()
        raise ValueError(f"Missing cluster assignment for environments: {missing_envs[:10]}")
    env_means[CLUSTER_COLUMN] = env_means[CLUSTER_COLUMN].astype(int)
    env_means["annual_precip_mm_est"] = env_means["PRECTOTCORR"] * 365.25
    env_means["diurnal_range_C"] = env_means["T2M_MAX"] - env_means["T2M_MIN"]
    return env_means


def build_profile(env_means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_features = [
        column
        for column in env_means.columns
        if column not in {ENV_COLUMN, CLUSTER_COLUMN, "record_count"}
    ]

    rows = []
    for cluster, cluster_df in env_means.groupby(CLUSTER_COLUMN, sort=True):
        row = {
            CLUSTER_COLUMN: int(cluster),
            "suggested_label": SUGGESTED_LABELS.get(int(cluster), ""),
            "n_envs": int(cluster_df[ENV_COLUMN].nunique()),
            "n_records": int(cluster_df["record_count"].sum()),
        }
        for feature in numeric_features:
            row[feature] = float(cluster_df[feature].mean())
        rows.append(row)

    profile = pd.DataFrame(rows).sort_values(CLUSTER_COLUMN, kind="stable")
    overall_mean = env_means[numeric_features].mean()
    overall_std = env_means[numeric_features].std(ddof=0).replace(0, np.nan)
    zscores = profile[[CLUSTER_COLUMN, "suggested_label"]].copy()
    zscores[numeric_features] = (profile[numeric_features] - overall_mean) / overall_std
    return profile, zscores


def build_evidence(profile: pd.DataFrame, zscores: pd.DataFrame) -> pd.DataFrame:
    evidence_features = [
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "T2MWET",
        "T2MDEW",
        "QV2M",
        "RH2M",
        "PRECTOTCORR",
        "annual_precip_mm_est",
        "ALLSKY_SFC_PAR_TOT",
        "ALLSKY_SFC_SW_DNI",
        "PS",
        "WS2M",
        "diurnal_range_C",
    ]

    rows = []
    for _, cluster_row in profile.iterrows():
        cluster = int(cluster_row[CLUSTER_COLUMN])
        zrow = zscores.loc[zscores[CLUSTER_COLUMN].eq(cluster)].iloc[0]
        for feature in evidence_features:
            zscore = float(zrow[feature])
            low_text, high_text = FEATURE_DIRECTIONS[feature]
            direction = high_text if zscore > 0 else low_text
            rows.append(
                {
                    CLUSTER_COLUMN: cluster,
                    "suggested_label": cluster_row["suggested_label"],
                    "feature": feature,
                    "feature_description": FEATURE_LABELS[feature],
                    "mean": float(cluster_row[feature]),
                    "zscore_vs_all_envs": zscore,
                    "absolute_zscore": abs(zscore),
                    "interpretive_direction": direction,
                    "used_in_kmeans": feature in CLUSTER_FEATURES,
                }
            )

    evidence = pd.DataFrame(rows).sort_values(
        [CLUSTER_COLUMN, "absolute_zscore"], ascending=[True, False], kind="stable"
    )
    evidence["rank_within_cluster"] = evidence.groupby(CLUSTER_COLUMN).cumcount() + 1
    return evidence[
        [
            CLUSTER_COLUMN,
            "suggested_label",
            "rank_within_cluster",
            "feature",
            "feature_description",
            "mean",
            "zscore_vs_all_envs",
            "absolute_zscore",
            "interpretive_direction",
            "used_in_kmeans",
        ]
    ]


def save_zscore_heatmap(
    zscores: pd.DataFrame,
    heatmap_features: list[str],
    output_path: Path,
    title: str,
    figsize: tuple[float, float],
    rotation: int = 35,
) -> None:
    import matplotlib.pyplot as plt

    data = zscores.set_index(CLUSTER_COLUMN)[heatmap_features]
    labels = [FEATURE_LABELS[column] for column in heatmap_features]

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(data.to_numpy(), cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=rotation, ha="right")
    ax.set_yticks(range(len(data.index)), labels=[f"Cluster {idx}" for idx in data.index])
    ax.set_title(title)
    ax.set_xlabel("Weather variable")
    ax.set_ylabel("Cluster")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("z-score vs. all environment means")
    for row_index in range(data.shape[0]):
        for col_index in range(data.shape[1]):
            value = data.iloc[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_heatmaps(zscores: pd.DataFrame) -> None:
    compact_features = [
        "T2M",
        "T2MWET",
        "T2MDEW",
        "QV2M",
        "RH2M",
        "PRECTOTCORR",
        "ALLSKY_SFC_PAR_TOT",
        "ALLSKY_SFC_SW_DNI",
        "PS",
        "WS2M",
        "diurnal_range_C",
    ]
    all_raw_features = [
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "T2MWET",
        "T2MDEW",
        "QV2M",
        "RH2M",
        "PRECTOTCORR",
        "ALLSKY_SFC_PAR_TOT",
        "ALLSKY_SFC_SW_DNI",
        "ALLSKY_SFC_SW_DWN",
        "PS",
        "WS2M",
        "GWETTOP",
        "GWETROOT",
        "GWETPROF",
    ]
    save_zscore_heatmap(
        zscores,
        compact_features,
        HEATMAP_PATH,
        "Weather Cluster Profiles: Standardized Difference from Overall Mean",
        figsize=(12, 4.8),
    )
    save_zscore_heatmap(
        zscores,
        all_raw_features,
        FULL_RAW_HEATMAP_PATH,
        "Weather Cluster Profiles: All 16 Raw Features",
        figsize=(15, 5.2),
        rotation=40,
    )


def save_counts(profile: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(profile[CLUSTER_COLUMN].astype(str), profile["n_envs"], color="#4f7b93")
    ax.set_title("Environment Count by Weather Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of environments")
    for x, y in zip(profile[CLUSTER_COLUMN].astype(str), profile["n_envs"]):
        ax.text(x, y + 2, str(int(y)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(COUNTS_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    weather, assignments, weather_features = load_inputs()
    env_means = build_environment_means(weather, assignments, weather_features)
    profile, zscores = build_profile(env_means)
    evidence = build_evidence(profile, zscores)

    profile.to_csv(PROFILE_PATH, index=False)
    zscores.to_csv(ZSCORE_PATH, index=False)
    evidence.to_csv(EVIDENCE_PATH, index=False)
    save_heatmaps(zscores)
    save_counts(profile)

    print("Files written:")
    for path in [
        PROFILE_PATH,
        ZSCORE_PATH,
        EVIDENCE_PATH,
        HEATMAP_PATH,
        FULL_RAW_HEATMAP_PATH,
        COUNTS_PATH,
    ]:
        print(path.name)

    print("\nCompact profile:")
    print(
        profile[[CLUSTER_COLUMN, "suggested_label", *PRESENTATION_COLUMNS]]
        .round(2)
        .to_string(index=False)
    )

    print("\nTop label evidence by cluster:")
    print(
        evidence[evidence["rank_within_cluster"].le(5)][
            [
                CLUSTER_COLUMN,
                "rank_within_cluster",
                "feature_description",
                "mean",
                "zscore_vs_all_envs",
                "interpretive_direction",
                "used_in_kmeans",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
