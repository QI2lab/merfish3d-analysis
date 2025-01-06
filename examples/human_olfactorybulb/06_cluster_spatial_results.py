import scanpy as sc
import pandas as pd
import anndata as ad
from pathlib import Path

# Define sample paths
samples = {
    "MERFISH001": Path(r"/mnt/data/qi2lab/20240317_OB_MERFISH_7/qi2labdatastore/mtx_output")
}

# Load data from all samples
adatas= {}
 
for sample_id, data_dir in samples.items():
    sample_adata = sc.read_10x_mtx(data_dir)
    sample_adata.var_names_make_unique()
    adatas[sample_id] = sample_adata

# Add sample metadata
for sample_name, adata in adatas.items():
    adata.obs["sample"] = sample_name

adata_combined = ad.concat(adatas, label="sample")
adata_combined.obs_names_make_unique()
adata_combined.layers["counts"] = adata_combined.X.copy()

sc.pp.normalize_per_cell(adata_combined, counts_per_cell_after=1e6)
sc.pp.log1p(adata_combined)
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

# # Define marker genes for cell type annotation
# markers = {
#     "GR-OT": ["MAG", "PACSIN1", "SYNPR", "CD9", "CHGB", "CERS2", "SNCA", "NDRG1", "CCDC47", "CNTN2"],
#     "GL-OT": ["MAG", "SEPTIN4", "FRZB", "TPPP",  "KLK6", "RND3",  "SPTBN1", "CNTN2",  "SORCS1", "DEPP1", "CHSY3"],
#     "EPL-GL": ["CDH19", "SNCA",  "BIN1", "AHR", "ARPC5L", "FGF2", "HSPH1", "CD59", "KCNAB2", "FRZB", "VSIG4", "CNN3", "ATP5PO", "SERPING1", "TPST1"],
#     "EPL-GR": ["ADAMTS3", "CERK", "FA2H", "MCAM", "NCBP2",   "RELL1", "CACYBP", "HLA-DQA1", "ITGA7"],
#     "EPL-OT": ["MAG", "SNCA", "KLK6", "CHGB", "PLP1", "NSG2", "FA2H", "AMPH",  "HNRNPC", "MYLK", "KCNAB2", "CRTAC1"],
#     "GL-GR": ["VSIG4", "MCAM", "SNCA", "SPTBN1", "MAG", "FADS3", "BIN1", "TPST1", "PEG10", "MTCH1", "TPPP", "CDH19", "NFE2L1", "CERS2", "SEPTIN4", "CDH19",   "FA2H", "BCL2", "LGMN"]
#     }

# # Annotate cell types based on marker genes
# for cell_type, marker_genes in markers.items():
#     adata_combined.obs[cell_type] = adata_combined[:, marker_genes].X.mean(axis=1)

# # Visualize marker genes in UMAP
# sc.pl.umap(adata_combined, color=list(markers.keys()))

# Differential expression analysis
sc.tl.rank_genes_groups(adata_combined, groupby="clusters", method="t-test")
sc.pl.rank_genes_groups(adata_combined, n_genes=20, sharey=False)

# Export marker genes for each cluster
marker_path = Path(r"/mnt/data/qi2lab/20240317_OB_MERFISH_7/qi2labdatastore/mtx_output/cluster_markers.csv")
marker_results = pd.DataFrame(adata_combined.uns['rank_genes_groups']['names'])
marker_results.to_csv(marker_path, index=False)

# # Save the annotated dataset
# dataout_path = Path(r"/mnt/server2/qi2lab/20241212_OB_22bMERFISH_1/qi2labdatastore/mtx_output/annotated_combined_dataset.h5ad")
# adata_combined.write(dataout_path)