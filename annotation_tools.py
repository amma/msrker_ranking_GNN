from __future__ import annotations

from io import StringIO
from pathlib import Path
from urllib.parse import quote
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data_mining"
ANNOT_DIR = OUT_DIR / "annotations"

WINDOWS_BP = [100_000, 500_000]
MIN_SELECTED_GENES_FOR_TERM = 3
MIN_BACKGROUND_GENES_FOR_TERM = 5
MAX_BACKGROUND_FRAC_FOR_TERM = 0.5

GENERIC_TERM_NAMES = {
    "molecular_function",
    "biological_process",
    "cellular_component",
    "binding",
    "catalytic activity",
    "metabolic process",
    "cellular process",
    "primary metabolic process",
    "organic substance metabolic process",
    "nitrogen compound metabolic process",
    "cellular metabolic process",
    "nucleobase-containing compound metabolic process",
    "protein binding",
    "structural molecule activity",
}

BIOMART_URL = "https://plants.ensembl.org/biomart/martservice"

CROPS = {
    "maize": {
        "dataset": "zmays_eg_gene",
        "species_label": "Zea mays",
        "assembly": "Zm-B73-REFERENCE-NAM-5.0",
        "chrom_prefix": "chr",
    },
    "soynam": {
        "dataset": "gmax_eg_gene",
        "species_label": "Glycine max",
        "assembly": "Glycine_max_v2.1 / Wm82.a2.v1",
        "chrom_prefix": "Gm",
    },
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "/").replace("\n", " ").strip()


def bh_adjust(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    n = len(p)
    if n == 0:
        return pd.Series(dtype=float, index=p_values.index)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running_min = 1.0
    for rank_from_end, index in enumerate(order[::-1], start=1):
        original_rank = n - rank_from_end + 1
        running_min = min(running_min, p[index] * n / original_rank)
        adjusted[index] = running_min
    return pd.Series(np.clip(adjusted, 0, 1), index=p_values.index)


def biomart_query(dataset: str, attributes: list[str], cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t", dtype=str)

    attrs = "\n".join([f'    <Attribute name="{attr}" />' for attr in attributes])
    query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="plants_mart" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="{dataset}" interface="default">
{attrs}
  </Dataset>
</Query>"""
    url = f"{BIOMART_URL}?query={quote(query)}"
    print(f"Downloading BioMart table: {dataset} -> {cache_path.name}")
    with urllib.request.urlopen(url, timeout=180) as response:
        text = response.read().decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(text), sep="\t", dtype=str)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, sep="\t", index=False)
    return df


def load_crop_annotations(crop: str) -> dict[str, pd.DataFrame]:
    spec = CROPS[crop]
    dataset = spec["dataset"]
    prefix = ANNOT_DIR / crop
    prefix.mkdir(parents=True, exist_ok=True)

    coords = biomart_query(
        dataset,
        [
            "ensembl_gene_id",
            "external_gene_name",
            "chromosome_name",
            "start_position",
            "end_position",
            "strand",
            "gene_biotype",
            "description",
        ],
        prefix / f"{crop}_gene_coordinates.tsv",
    )
    coords.columns = [
        "gene_id",
        "gene_name",
        "chrom",
        "start",
        "end",
        "strand",
        "gene_biotype",
        "description",
    ]
    coords = coords[coords["chrom"].astype(str).str.match(r"^\d+$", na=False)].copy()
    coords["chrom"] = coords["chrom"].astype(int)
    coords["start"] = pd.to_numeric(coords["start"], errors="coerce")
    coords["end"] = pd.to_numeric(coords["end"], errors="coerce")
    coords = coords.dropna(subset=["start", "end"]).copy()
    coords["start"] = coords["start"].astype(int)
    coords["end"] = coords["end"].astype(int)
    coords = coords.drop_duplicates("gene_id").sort_values(["chrom", "start", "end"], kind="stable").reset_index(drop=True)
    coords.to_csv(prefix / f"{crop}_gene_coordinates_clean.csv", index=False)

    go = biomart_query(
        dataset,
        ["ensembl_gene_id", "go_id", "name_1006", "namespace_1003"],
        prefix / f"{crop}_go_annotations.tsv",
    )
    go.columns = ["gene_id", "term_id", "term_name", "term_domain"]
    go = go[go["term_id"].notna() & go["term_id"].ne("")].drop_duplicates().reset_index(drop=True)
    go["source"] = "GO"
    go.to_csv(prefix / f"{crop}_go_annotations_clean.csv", index=False)

    goslim = biomart_query(
        dataset,
        ["ensembl_gene_id", "goslim_goa_accession", "goslim_goa_description"],
        prefix / f"{crop}_goslim_annotations.tsv",
    )
    goslim.columns = ["gene_id", "term_id", "term_name"]
    goslim = goslim[goslim["term_id"].notna() & goslim["term_id"].ne("")].drop_duplicates().reset_index(drop=True)
    goslim["term_domain"] = "GOSlim"
    goslim["source"] = "GOSlim"
    goslim.to_csv(prefix / f"{crop}_goslim_annotations_clean.csv", index=False)

    interpro = biomart_query(
        dataset,
        ["ensembl_gene_id", "interpro", "interpro_short_description", "interpro_description"],
        prefix / f"{crop}_interpro_annotations.tsv",
    )
    interpro.columns = ["gene_id", "term_id", "term_short_name", "term_name"]
    interpro = interpro[interpro["term_id"].notna() & interpro["term_id"].ne("")].drop_duplicates().reset_index(drop=True)
    interpro["term_name"] = interpro["term_name"].where(interpro["term_name"].notna() & interpro["term_name"].ne(""), interpro["term_short_name"])
    interpro["term_domain"] = "InterPro"
    interpro["source"] = "InterPro"
    interpro = interpro[["gene_id", "term_id", "term_name", "term_domain", "source"]]
    interpro.to_csv(prefix / f"{crop}_interpro_annotations_clean.csv", index=False)

    pfam = biomart_query(
        dataset,
        ["ensembl_gene_id", "pfam"],
        prefix / f"{crop}_pfam_annotations.tsv",
    )
    pfam.columns = ["gene_id", "term_id"]
    pfam = pfam[pfam["term_id"].notna() & pfam["term_id"].ne("")].drop_duplicates().reset_index(drop=True)
    pfam["term_name"] = pfam["term_id"]
    pfam["term_domain"] = "Pfam"
    pfam["source"] = "Pfam"
    pfam.to_csv(prefix / f"{crop}_pfam_annotations_clean.csv", index=False)

    reactome = biomart_query(
        dataset,
        ["ensembl_gene_id", "plant_reactome_pathway"],
        prefix / f"{crop}_plant_reactome_annotations.tsv",
    )
    reactome.columns = ["gene_id", "term_id"]
    reactome = reactome[reactome["term_id"].notna() & reactome["term_id"].ne("")].drop_duplicates().reset_index(drop=True)
    reactome["term_name"] = reactome["term_id"]
    reactome["term_domain"] = "Plant Reactome"
    reactome["source"] = "PlantReactome"
    reactome.to_csv(prefix / f"{crop}_plant_reactome_annotations_clean.csv", index=False)

    output = {
        "coords": coords,
        "GO": go[["gene_id", "term_id", "term_name", "term_domain", "source"]],
        "GOSlim": goslim[["gene_id", "term_id", "term_name", "term_domain", "source"]],
        "InterPro": interpro,
        "Pfam": pfam,
        "PlantReactome": reactome,
    }

    if crop == "soynam":
        kegg = biomart_query(
            dataset,
            ["ensembl_gene_id", "kegg_enzyme"],
            prefix / f"{crop}_kegg_enzyme_annotations.tsv",
        )
        kegg.columns = ["gene_id", "term_id"]
        kegg = kegg[kegg["term_id"].notna() & kegg["term_id"].ne("")].drop_duplicates().reset_index(drop=True)
        kegg["term_name"] = kegg["term_id"]
        kegg["term_domain"] = "KEGG enzyme"
        kegg["source"] = "KEGG_Enzyme"
        kegg.to_csv(prefix / f"{crop}_kegg_enzyme_annotations_clean.csv", index=False)
        output["KEGG_Enzyme"] = kegg

    return output


def normalize_marker_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def map_markers_to_genes(markers: pd.DataFrame, genes: pd.DataFrame, window_bp: int, crop: str, set_id: str) -> pd.DataFrame:
    rows = []
    genes_by_chr = {
        chrom: chr_genes.sort_values("start", kind="stable").reset_index(drop=True)
        for chrom, chr_genes in genes.groupby("chrom", sort=True)
    }
    for marker in markers[["marker_id", "chr", "position"]].drop_duplicates().itertuples(index=False):
        marker_id = str(marker.marker_id)
        chrom = int(marker.chr)
        pos = int(marker.position)
        chr_genes = genes_by_chr.get(chrom)
        if chr_genes is None or chr_genes.empty:
            continue
        nearby = chr_genes[(chr_genes["start"].le(pos + window_bp)) & (chr_genes["end"].ge(pos - window_bp))].copy()
        if nearby.empty:
            continue
        for gene in nearby.itertuples(index=False):
            if pos < int(gene.start):
                distance = int(gene.start) - pos
            elif pos > int(gene.end):
                distance = pos - int(gene.end)
            else:
                distance = 0
            rows.append(
                {
                    "crop": crop,
                    "set_id": set_id,
                    "window_bp": window_bp,
                    "marker_id": marker_id,
                    "marker_chr": chrom,
                    "marker_position": pos,
                    "gene_id": gene.gene_id,
                    "gene_name": clean_text(gene.gene_name),
                    "gene_chr": int(gene.chrom),
                    "gene_start": int(gene.start),
                    "gene_end": int(gene.end),
                    "distance_bp": distance,
                    "gene_biotype": clean_text(gene.gene_biotype),
                    "description": clean_text(gene.description),
                }
            )
    return pd.DataFrame(rows)


def enrichment_for_set(
    selected_genes: set[str],
    background_genes: set[str],
    annotation: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    ann = annotation[annotation["gene_id"].isin(background_genes)].copy()
    if ann.empty or not selected_genes:
        return pd.DataFrame()
    selected_genes = selected_genes & background_genes
    M = len(background_genes)
    n = len(selected_genes)
    max_background = max(MIN_BACKGROUND_GENES_FOR_TERM, int(MAX_BACKGROUND_FRAC_FOR_TERM * M))
    rows = []
    for (term_id, term_name, term_domain), group in ann.groupby(["term_id", "term_name", "term_domain"], dropna=False, sort=False):
        term_genes = set(group["gene_id"].dropna().astype(str))
        K = len(term_genes)
        if K < MIN_BACKGROUND_GENES_FOR_TERM or K > max_background:
            continue
        x = len(selected_genes & term_genes)
        if x < MIN_SELECTED_GENES_FOR_TERM:
            continue
        p = float(hypergeom.sf(x - 1, M, K, n))
        expected = n * K / M if M else np.nan
        odds = ((x + 0.5) / (n - x + 0.5)) / ((K - x + 0.5) / (M - K - n + x + 0.5))
        rows.append(
            {
                "source": source,
                "term_id": term_id,
                "term_name": clean_text(term_name),
                "term_domain": clean_text(term_domain),
                "selected_term_genes": x,
                "selected_genes_with_annotation": n,
                "background_term_genes": K,
                "background_genes_with_annotation": M,
                "expected_selected_genes": expected,
                "enrichment_ratio": x / expected if expected > 0 else np.nan,
                "odds_ratio": odds,
                "p_value": p,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_value_bh"] = bh_adjust(result["p_value"])
    result = result.sort_values(["q_value_bh", "p_value", "source", "term_name"], kind="stable").reset_index(drop=True)
    return result


def is_focused_term(row: pd.Series) -> bool:
    name = clean_text(row.get("term_name", "")).lower()
    if not name or name in GENERIC_TERM_NAMES:
        return False
    if name.startswith("biological_") or name.startswith("molecular_") or name.startswith("cellular_"):
        return False
    return True
