from pathlib import Path

import numpy as np
import pandas as pd

from gnn_model import run_gnn_experiment

base_dir = Path(__file__).resolve().parent.parent

seed = 42
split_top_n = 800
split_random_n = 200
split_total_n = split_top_n + split_random_n
rng = np.random.default_rng(seed)

trait_path = base_dir / 'train_trait.csv'
genotype_path = base_dir / 'genotype_reduced.csv'
env_path = base_dir / 'train_env_vectors.csv'

train_candidate_path = base_dir / 'candidate_genotypes_1000_split80.csv'
train_final_path = base_dir / 'final_genotype_1000_split80.csv'
holdout_candidate_path = base_dir / 'candidate_genotypes_1000_split20.csv'
holdout_final_path = base_dir / 'final_genotype_1000_split20.csv'
summary_path = base_dir / 'arch_a_multienv_1000_split_stability_summary.csv'
overlap_path = base_dir / 'arch_a_multienv_1000_split_top200_overlap.csv'

trait_df = pd.read_csv(trait_path).copy()
genotype_df = pd.read_csv(genotype_path).copy()
env_df = pd.read_csv(env_path).copy()

trait_df['Hybrid'] = trait_df['Hybrid'].astype(str)
trait_df['Env'] = trait_df['Env'].astype(str)
valid_hybrids = set(genotype_df.iloc[:, 0].astype(str))
valid_envs = set(env_df.iloc[:, 0].astype(str))

valid_trait_df = trait_df[
    trait_df['Hybrid'].isin(valid_hybrids)
    & trait_df['Env'].isin(valid_envs)
].copy()
valid_trait_df = valid_trait_df.reset_index(drop=True)

permutation = rng.permutation(len(valid_trait_df))
first_count = int(round(0.8 * len(valid_trait_df)))
train_split_df = valid_trait_df.iloc[permutation[:first_count]].copy().reset_index(drop=True)
holdout_split_df = valid_trait_df.iloc[permutation[first_count:]].copy().reset_index(drop=True)


def build_split_candidate(split_df: pd.DataFrame, top_n: int, random_n: int, random_state: int):
    hybrid_summary = (
        split_df.groupby('Hybrid', as_index=False)
        .agg(
            avg_yield=('Yield_Mg_ha', 'mean'),
            appearances=('Yield_Mg_ha', 'size'),
        )
        .sort_values(['avg_yield', 'appearances', 'Hybrid'], ascending=[False, False, True], kind='stable')
        .reset_index(drop=True)
    )
    if len(hybrid_summary) < top_n + random_n:
        raise ValueError(f'Need at least {top_n + random_n} valid hybrids, found {len(hybrid_summary)}')

    top_df = hybrid_summary.head(top_n).copy()
    remaining_df = hybrid_summary.iloc[top_n:].copy()
    random_df = remaining_df.sample(n=random_n, random_state=random_state).copy()
    selected_df = pd.concat([top_df, random_df], ignore_index=True)
    selected_df['selection_group'] = ['top_800'] * len(top_df) + ['random_200'] * len(random_df)
    selected_hybrids = set(selected_df['Hybrid'])

    candidate_df = split_df[split_df['Hybrid'].isin(selected_hybrids)].copy()
    candidate_df = candidate_df[['Env', 'Hybrid', 'Yield_Mg_ha']].sort_values(['Hybrid', 'Env', 'Yield_Mg_ha'], kind='stable').reset_index(drop=True)
    return hybrid_summary, selected_df, candidate_df


train_summary, train_selected, train_candidate_df = build_split_candidate(train_split_df, split_top_n, split_random_n, random_state=seed)
holdout_summary, holdout_selected, holdout_candidate_df = build_split_candidate(holdout_split_df, split_top_n, split_random_n, random_state=seed)

train_candidate_df.to_csv(train_candidate_path, index=False)
train_candidate_df.to_csv(train_final_path, index=False)
holdout_candidate_df.to_csv(holdout_candidate_path, index=False)
holdout_candidate_df.to_csv(holdout_final_path, index=False)

print('Split 80 selection:')
print(f'  rows={len(train_candidate_df)} | unique_hybrids={train_candidate_df["Hybrid"].nunique()} | unique_envs={train_candidate_df["Env"].nunique()}')
print('Split 20 selection:')
print(f'  rows={len(holdout_candidate_df)} | unique_hybrids={holdout_candidate_df["Hybrid"].nunique()} | unique_envs={holdout_candidate_df["Env"].nunique()}')

train_results = run_gnn_experiment(
    final_path=train_final_path,
    genotype_path=genotype_path,
    env_path=env_path,
    output_prefix='arch_a_multienv_1000_split80_raw',
    use_env_residual=False,
    epochs=100,
    return_weight_dataframe=False,
)

holdout_results = run_gnn_experiment(
    final_path=holdout_final_path,
    genotype_path=genotype_path,
    env_path=env_path,
    output_prefix='arch_a_multienv_1000_split20_raw',
    use_env_residual=False,
    epochs=100,
    return_weight_dataframe=False,
)

train_ranking = train_results['ranking_df'].copy()
holdout_ranking = holdout_results['ranking_df'].copy()

train_history = train_results['history_df'].sort_values(['val_mse', 'epoch'], kind='stable').iloc[0]
holdout_history = holdout_results['history_df'].sort_values(['val_mse', 'epoch'], kind='stable').iloc[0]


def ranking_stats(name: str, ranking_df: pd.DataFrame, best_row: pd.Series, candidate_df: pd.DataFrame) -> dict:
    return {
        'model': name,
        'best_val_mse': float(best_row['val_mse']),
        'best_val_mae': float(best_row['val_mae']),
        'best_val_r2': float(best_row['val_r2']),
        'min_weight': float(ranking_df['avg_weight'].min()),
        'max_weight': float(ranking_df['avg_weight'].max()),
        'weight_std': float(ranking_df['avg_weight'].std()),
        'num_genotypes': int(candidate_df['Hybrid'].nunique()),
        'num_envs': int(candidate_df['Env'].nunique()),
        'num_rows': int(len(candidate_df)),
    }


summary_records = [
    ranking_stats('1000_split80_raw_multienv', train_ranking, train_history, train_candidate_df),
    ranking_stats('1000_split20_raw_multienv', holdout_ranking, holdout_history, holdout_candidate_df),
]
summary_df = pd.DataFrame(summary_records)

train_top_markers = train_ranking.head(200)[['rank', 'node_index', 'marker_id', 'avg_weight']].rename(
    columns={'rank': 'rank_split80', 'node_index': 'node_index_split80', 'avg_weight': 'avg_weight_split80'}
)
holdout_top_markers = holdout_ranking.head(200)[['rank', 'node_index', 'marker_id', 'avg_weight']].rename(
    columns={'rank': 'rank_split20', 'node_index': 'node_index_split20', 'avg_weight': 'avg_weight_split20'}
)

overlap_df = train_top_markers.merge(holdout_top_markers, on='marker_id', how='inner')
overlap_df = overlap_df.sort_values(['rank_split80', 'rank_split20'], kind='stable').reset_index(drop=True)
overlap_df.to_csv(overlap_path, index=False)
summary_df.to_csv(summary_path, index=False)

overlap_count = int(len(overlap_df))
union_count = int(len(set(train_top_markers['marker_id']).union(set(holdout_top_markers['marker_id']))))
jaccard = overlap_count / union_count if union_count else float('nan')

print('\nSplit stability summary:')
print(summary_df.to_string(index=False))
print(f'\nTop-200 overlap count: {overlap_count}')
print(f'Top-200 overlap Jaccard: {jaccard:.6f}')
print(f'Summary saved to: {summary_path.name}')
print(f'Overlap table saved to: {overlap_path.name}')
if overlap_count > 0:
    print('\nFirst 10 overlapping top-200 markers:')
    print(overlap_df.head(10).to_string(index=False))
