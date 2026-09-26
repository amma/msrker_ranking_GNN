from pathlib import Path

import numpy as np
import pandas as pd

from rf_rank_maize import infer_marker_coordinates

BASE_DIR = Path(__file__).resolve().parent.parent
SOY_DIR = BASE_DIR / "soynam_arch_a_general_6env_nasa"


def weighted_abs_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    xc = x - (w[:, None] * x).sum(axis=0)
    yc = y - (w * y).sum()
    cov = (w[:, None] * xc * yc[:, None]).sum(axis=0)
    var_x = (w[:, None] * xc**2).sum(axis=0)
    var_y = (w * yc**2).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(cov / np.sqrt(var_x * var_y))


def marker_correlations(genotype_path: Path, phenotype_path: Path) -> pd.Series:
    genotype_df = pd.read_csv(genotype_path)
    genotype_df = genotype_df.rename(columns={genotype_df.columns[0]: "Hybrid"})
    genotype_df["Hybrid"] = genotype_df["Hybrid"].astype(str)
    marker_columns = [column for column in genotype_df.columns if column != "Hybrid"]

    phenotype_df = pd.read_csv(phenotype_path)
    phenotype_df["Hybrid"] = phenotype_df["Hybrid"].astype(str)
    phenotype_df["Env"] = phenotype_df["Env"].astype(str)
    pairs = phenotype_df.groupby(["Hybrid", "Env"], as_index=False).agg(
        mean_yield=("Yield_Mg_ha", "mean"),
        replicate_count=("Yield_Mg_ha", "size"),
    )
    merged = pairs.merge(genotype_df, on="Hybrid", how="inner")
    corr = weighted_abs_corr(
        merged[marker_columns].to_numpy(dtype=float),
        merged["mean_yield"].to_numpy(dtype=float),
        merged["replicate_count"].to_numpy(dtype=float),
    )
    return pd.Series(corr, index=marker_columns).fillna(0.0)


def finish(ranking: pd.DataFrame, output_path: Path) -> None:
    ranking = ranking.sort_values(["corr_abs", "marker_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking[["rank", "marker_id", "corr_abs", "chr", "position", "chr_pos"]].to_csv(output_path, index=False)
    print(f"Saved {output_path.name}: {len(ranking)} markers")


def build_maize() -> None:
    corr = marker_correlations(BASE_DIR / "genotype_reduced.csv", BASE_DIR / "final_genotype_1000.csv")
    coords = infer_marker_coordinates(list(corr.index))
    coords["corr_abs"] = corr.to_numpy()
    finish(coords, BASE_DIR / "corr_baseline_maize.ranking.csv")


def build_soynam() -> None:
    corr = marker_correlations(SOY_DIR / "soynam_6env_nasa_genotype.csv", SOY_DIR / "soynam_6env_nasa_final_full.csv")
    parts = pd.Series(corr.index).str.extract(r"^Gm(\d+)_(\d+)_")
    ranking = pd.DataFrame(
        {
            "marker_id": corr.index,
            "corr_abs": corr.to_numpy(),
            "chr": parts[0].astype(int).to_numpy(),
            "position": parts[1].astype(int).to_numpy(),
        }
    )
    ranking["chr_pos"] = "Gm" + ranking["chr"].astype(str).str.zfill(2) + ":" + ranking["position"].astype(str)
    finish(ranking, BASE_DIR / "corr_baseline_soynam.ranking.csv")


if __name__ == "__main__":
    build_maize()
    build_soynam()
