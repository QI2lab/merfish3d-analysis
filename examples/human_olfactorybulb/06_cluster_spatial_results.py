"""
Performing clustering and marker gene extraction on spatially resolved data.

The clustering is performed without OR genes.

Shepherd 2024/01 - modified script to accept parameters.
"""


import scanpy as sc
import pandas as pd
import anndata as ad
from pathlib import Path

def cluster_spatial_results(root_path: Path):
    """Perform clustering on cell x gene matrix.
    
    Parameters
    ----------
    root_path: Path
        path to experiment
        
    """

    # Define sample paths
    samples = {
        "MERFISH001": root_path / Path(r"qi2labdatastore/mtx_output")
    }

    # Load data from all samples
    adatas = {}

    for sample_id, data_dir in samples.items():
        sample_adata = sc.read_10x_mtx(data_dir)
        sample_adata.var_names_make_unique()
        adatas[sample_id] = sample_adata

    # Add sample metadata
    for sample_name, adata in adatas.items():
        adata.obs["sample"] = sample_name

    # Combine all samples
    adata_combined = ad.concat(adatas, label="sample")
    adata_combined.obs_names_make_unique()
    adata_combined.layers["counts"] = adata_combined.X.copy()

    # Separate "OR*" genes for later analysis
    or_genes_mask = adata_combined.var_names.str.startswith("OR")
    or_genes_adata = adata_combined[:, or_genes_mask].copy()  # Retain "OR*" genes in a separate AnnData object
    adata_combined = adata_combined[:, ~or_genes_mask].copy()  # Exclude "OR*" genes from clustering

    # Normalization and clustering (ignoring "OR*" genes)
    sc.pp.normalize_per_cell(adata_combined, counts_per_cell_after=1e6)
    sc.pp.log1p(adata_combined)
    adata_combined.raw = adata_combined  # Save raw data for downstream DE analysis
    sc.pp.pca(adata_combined, n_comps=15)
    sc.pp.neighbors(adata_combined)
    sc.tl.umap(adata_combined)
    sc.tl.leiden(
        adata_combined,
        key_added="clusters",
        resolution=0.5,
        n_iterations=2,
        flavor="igraph",
        directed=False,
    )
    sc.pl.umap(adata_combined, color="clusters")

    # Find marker genes for clusters (excluding "OR*" genes)
    sc.tl.rank_genes_groups(
        adata_combined,
        groupby="clusters",
        method="t-test",
        use_raw=True  # Use raw (log-transformed) data
    )
    sc.pl.rank_genes_groups(adata_combined, n_genes=20, sharey=False)

    # Export marker genes for each cluster
    marker_path = root_path / Path(r"mtx_output/cluster_markers.csv")
    marker_results = pd.DataFrame(adata_combined.uns['rank_genes_groups']['names'])
    marker_results.to_csv(marker_path, index=False)

    # Align "OR*" genes with `adata_combined.obs` for cluster-based analysis
    or_genes_adata = or_genes_adata[adata_combined.obs_names].copy()
    or_genes_adata.obs["clusters"] = adata_combined.obs["clusters"]  # Copy cluster labels to "OR*" genes dataset

    # Normalize and logarithmize "OR*" genes data for differential expression analysis
    sc.pp.normalize_per_cell(or_genes_adata, counts_per_cell_after=1e6)
    sc.pp.log1p(or_genes_adata)

    # Perform differential expression analysis on "OR*" genes
    sc.tl.rank_genes_groups(
        or_genes_adata,
        groupby="clusters",
        method="t-test",
    )
    sc.pl.rank_genes_groups(or_genes_adata, n_genes=20, sharey=False, title="Differential Expression of OR Genes")

    # Export "OR*" genes differential expression results
    or_marker_path = root_path / Path(r"qi2labdatastore/mtx_output/or_cluster_markers.csv")
    or_marker_results = pd.DataFrame(or_genes_adata.uns['rank_genes_groups']['names'])
    or_marker_results.to_csv(or_marker_path, index=False)

    # Save datasets
    dataout_path = root_path / Path(r"qi2labdatastore/mtx_output/annotated_combined_dataset.h5ad")
    adata_combined.write(dataout_path)

    or_genes_out_path = root_path / Path(r"qi2labdatastore/mtx_output/or_genes_dataset.h5ad")
    or_genes_adata.write(or_genes_out_path)

if __name__ == "__main__":
    root_path = Path(r"/mnt/data/qi2lab/20240317_OB_MERFISH_7")
    cluster_spatial_results(root_path=root_path,run_baysor=True,fdr_target=.05)