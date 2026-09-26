from pathlib import Path

from gnn_model import run_gnn_experiment

BASE_DIR = Path(__file__).resolve().parent.parent

experiment = run_gnn_experiment(
    final_path=BASE_DIR / "final_genotype_1000.csv",
    output_prefix="arch_a_multienv_1000",
    use_env_residual=False,
    epochs=100,
)
print(experiment["ranking_df"].head(10))
