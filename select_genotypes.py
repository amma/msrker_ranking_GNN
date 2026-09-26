from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
trait = pd.read_csv(BASE_DIR / "train_trait.csv")
genotype_df = pd.read_csv(BASE_DIR / "genotype_reduced.csv")
env_df = pd.read_csv(BASE_DIR / "train_env_vectors.csv")

genotype_df = genotype_df.rename(columns={genotype_df.columns[0]: "Hybrid"})
env_df = env_df.rename(columns={env_df.columns[0]: "Env"})
trait["Hybrid"] = trait["Hybrid"].astype(str)
trait["Env"] = trait["Env"].astype(str)
genotype_df["Hybrid"] = genotype_df["Hybrid"].astype(str)
env_df["Env"] = env_df["Env"].astype(str)

valid_trait = trait[
    trait["Hybrid"].isin(genotype_df["Hybrid"])
    & trait["Env"].isin(env_df["Env"])
].copy()

hybrid_summary = (
    valid_trait.groupby("Hybrid", as_index=False)
    .agg(avg_yield=("Yield_Mg_ha", "mean"), appearances=("Yield_Mg_ha", "size"))
    .sort_values(["avg_yield", "Hybrid"], ascending=[False, True], kind="stable")
    .reset_index(drop=True)
)

top_genotypes = hybrid_summary.head(800).copy()
random_genotypes = hybrid_summary.iloc[800:].sample(n=200, random_state=42).copy()
selected_hybrids = set(pd.concat([top_genotypes, random_genotypes], ignore_index=True)["Hybrid"])

candidate_genotypes = valid_trait[valid_trait["Hybrid"].isin(selected_hybrids)].copy()
candidate_genotypes = candidate_genotypes[["Env", "Hybrid", "Yield_Mg_ha"]].reset_index(drop=True)
final_genotypes = candidate_genotypes.copy()

candidate_genotypes.to_csv(BASE_DIR / "candidate_genotypes_1000.csv", index=False)
final_genotypes.to_csv(BASE_DIR / "final_genotype_1000.csv", index=False)

print(f"candidate_rows: {len(candidate_genotypes)}")
print(f"candidate_hybrids: {candidate_genotypes['Hybrid'].nunique()}")
print(f"candidate_envs: {candidate_genotypes['Env'].nunique()}")
print(f"missing_envs: {candidate_genotypes.loc[~candidate_genotypes['Env'].isin(env_df['Env']), 'Env'].nunique()}")
print(f"missing_hybrids: {candidate_genotypes.loc[~candidate_genotypes['Hybrid'].isin(genotype_df['Hybrid']), 'Hybrid'].nunique()}")
