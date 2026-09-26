## Data

- Maize genotypes, yield and weather, 2014–2021: Genomes to Fields 2022 competition data,
  https://doi.org/10.25739/tq5e-ak26 (processed into `genotype_reduced.csv`, `train_trait.csv`,
  `Weather_Data_after_drop.csv`, `train_env_vectors.csv`)
- SoyNAM: https://doi.org/10.5061/dryad.2fqz6133v (`X.csv`, `Y.csv`, `env.cov.csv` in
  `data/impact-ec-main/data/`); weather for two missing environments is retrieved from NASA POWER
  (https://power.larc.nasa.gov/)
- Gene annotation: Ensembl Plants BioMart, downloaded by `annotation_tools.py`
- GWAS reference SNPs: Zeng et al. 2022, Tolley et al. 2023, Ma and Cao 2021 (maize) and
  Diers et al. 2018 (SoyNAM)

## Scripts

Maize inputs
- `preprocess_maize.py`: VCF to numeric markers, imputation, low-variance filtering and similarity pruning; weather and trait table cleanup
- `reduce_markers.py`: keeps one randomly chosen marker per window of 33 markers (`genotype_reduced.csv`)
- `select_genotypes.py`: 1,000-genotype maize set (top 800 by mean yield plus 200 random)

Global marker ranking
- `gnn_model.py`: graph attention network, training and marker-attention ranking (used by the training scripts)
- `train_maize.py` (GPU): global maize model and marker ranking
- `plot_weight_curve.py`: ranked marker-weight curve
- `train_split.py` (GPU): 80/20 phenotype split and one model per half (the split80 ranking is used by `rf_rank_maize.py`)
- `train_soy.py` (GPU): SoyNAM data preparation, training and marker ranking


External GWAS overlap
- `marker_positions.py`: converts the maize ranking to chromosome positions
- `gwas_overlap.py`: overlap with the maize GWAS reference SNPs in 100 kb, 500 kb and 1 Mb windows
- `corr_rank.py`: correlation baseline (absolute weighted Pearson correlation with yield)
- `rf_rank_maize.py`, `rf_rank_soy.py`: random-forest baseline and the GNN, random forest and correlation comparison

Weather clusters
- `cluster_profile.py`: cluster weather profiles, labels, heatmap and counts
- `cluster_vec.py`: LSTM environment vectors for the clustered environments
- `cluster_train.py` (GPU): one model per cluster and the shared and unique marker analysis (`--datasets maize`;
  `--skip-training` rebuilds the tables and figures from saved rankings; `--use-env-residual` trains on environment-residual yield)

Weather-variable interpretability
- `weather_encoder.py` (GPU): one LSTM per weather variable combined by attention (`--all-weather-features`)
- `shap_compare.py`, `shap_stability.py`, `shap_cluster.py`: SHAP attribution of the weather variables and comparison with attention
- `shap_plots.py`: figures for the SHAP and attention comparison

Functional annotation
- `annotation_tools.py`: annotation download, marker-to-gene mapping and enrichment functions (used by the scripts below)
- `gene_windows.py`: marker-to-gene maps for all 4,050 markers
- `enrichment.py`: enrichment for the cluster marker sets (reads the rankings from `cluster_train.py --use-env-residual`)
- `top5_terms.py`: top five enriched terms per marker set


## Requirements

Python 3.10 with numpy 2.0.2, pandas 2.2.3, scipy 1.14.1, scikit-learn 1.6.0, xgboost 2.1.4,
torch 2.5.1, tensorflow 2.19.0, shap 0.46.0, matplotlib 3.9.4, requests 2.32.4 and nbformat 5.10.4.
GNN training used two NVIDIA RTX 6000 Ada GPUs.
