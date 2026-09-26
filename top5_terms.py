from pathlib import Path

import pandas as pd

from annotation_tools import load_crop_annotations

BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE_DIR / "data_mining" / "new_cluster_marker_analysis"

annotations = load_crop_annotations('maize')
enr = pd.read_csv(ANALYSIS_DIR / 'enrichment_all_terms.csv')
enr = enr[enr.window_bp == 500000]
sets = ['global_shared_by_all_4_clusters', 'C0_top100', 'C1_top100', 'C2_top100', 'C3_top100']
labels = {'global_shared_by_all_4_clusters': 'Global', 'C0_top100': 'C0', 'C1_top100': 'C1', 'C2_top100': 'C2', 'C3_top100': 'C3'}
sig = enr[(enr.q_value_bh <= 0.10) & (enr.set_id.isin(sets))].copy()
gm = pd.read_csv(ANALYSIS_DIR / 'marker_gene_map_window500000.csv').dropna(subset=['gene_id'])

rows = []
for i, r in sig.iterrows():
    ann = annotations[r.source]
    term_genes = set(ann.loc[ann['term_id'].astype(str) == str(r.term_id), 'gene_id'].dropna().astype(str))
    set_genes = gm[gm.set_id == r.set_id]
    genes_in_set = set(set_genes.gene_id.astype(str)) & term_genes
    markers = set_genes[set_genes.gene_id.astype(str).isin(term_genes)].marker_id.nunique()
    if markers >= 3:
        rows.append({'set_id': r.set_id, 'term_name': r.term_name, 'q': r.q_value_bh, 'genes': len(genes_in_set), 'markers': markers})
df = pd.DataFrame(rows)

out_rows = []
for s in sets:
    sub = df[df.set_id == s].sort_values('q').head(5).reset_index(drop=True)
    top_q = sub.loc[0, 'q']
    for i, row in sub.iterrows():
        out_rows.append({
            'Marker set': labels[s], 'Rank': i + 1, 'Term': row.term_name[:40],
            'Genes': row.genes, 'Markers': row.markers,
            'Gap vs. #1': '1.0x' if i == 0 else f'{row.q / top_q:.1f}x',
        })
out = pd.DataFrame(out_rows)
out.to_csv(ANALYSIS_DIR / 'top5_terms_gap_table.csv', index=False)
for s in ['Global', 'C0', 'C1', 'C2', 'C3']:
    print(f'--- {s} ---')
    print(out[out['Marker set'] == s].drop(columns='Marker set').to_string(index=False))
    print()
