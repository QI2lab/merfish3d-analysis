import scanpy as sc
import pandas as pd
import anndata as ad
from pathlib import Path

# Define sample paths
samples = {
    "patient001": Path(r"/mnt/data2/qi2lab/human_OE_scrnaseq/patient001"),
    "patient002": Path(r"/mnt/data2/qi2lab/human_OE_scrnaseq/patient002"),
    "patient003": Path(r"/mnt/data2/qi2lab/human_OE_scrnaseq/patient003"),
    "patient004": Path(r"/mnt/data2/qi2lab/human_OE_scrnaseq/patient004")
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
adata_combined.var["mt"] = adata.var_names.str.startswith("MT-")

# Quality control
sc.pp.calculate_qc_metrics(adata_combined, qc_vars=["mt"], inplace=True)
adata_combined = adata_combined[
    (adata_combined.obs["n_genes_by_counts"] > 100) &
    (adata_combined.obs["n_genes_by_counts"] < 8000) &
    (adata_combined.obs["pct_counts_mt"] < 10)
]

adata_combined.layers["counts"] = adata_combined.X.copy()

# Normalize data
sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)

# Find variable genes
sc.pp.highly_variable_genes(adata_combined, n_top_genes=5000, flavor="seurat_v3", layer="counts")

# Scale data
sc.pp.scale(adata_combined, max_value=10)

# Principal component analysis
sc.tl.pca(adata_combined, n_comps=30)
sc.pl.pca_variance_ratio(adata_combined, log=True)

# Integration using Harmony
sc.external.pp.harmony_integrate(adata_combined, key="sample")

# UMAP visualization
sc.pp.neighbors(adata_combined, n_pcs=30)
sc.tl.umap(adata_combined)

# Clustering
sc.tl.leiden(adata_combined, resolution=1.8)
sc.pl.umap(adata_combined, color=["leiden", "sample"])

# Define marker genes for cell type annotation
markers = {
    "CD8+ T Cells": ["CD3D", "CD3E", "CD8A"],
    "CD4+ T Cells": ["CD3D", "CD3E", "CD4", "IL7R"],
    "Natural Killer Cells": ["FGFBP2", "FCGR3A", "CX3CR1"],
    "B Cells": ["CD19", "CD79A", "MS4A4A"],
    "Plasma Cells": ["MZB1", "SDC1", "CD79A"],
    "Monocytes": ["CD14", "S100A12", "CLEC10A"],
    "Macrophages": ["C1QA", "C1QB", "C1QC"],
    "Dendritic Cells": ["CD1C"],
    "Mast Cells": ["TPSB2", "TPSAB1"],
    "Fibroblasts": ["LUM", "DCN", "CLEC11A"],
    "Respiratory Ciliated Cells": ["FOXJ1", "CFAP126", "STOML3"],
    "Respiratory HBCs": ["KRT5", "TP63", "SOX2"],
    "Respiratory Gland Progenitor Cells": ["SOX9", "SCGB1A1"],
    "Respiratory Secretory Cells": ["MUC5AC", "MUC5B"],
    "Vascular Smooth Muscle Cells": ["TAGLN", "MYH11"],
    "Pericytes": ["SOX17", "ENG"],
    "Bowman’s Glands": ["SOX9", "SOX10", "MUC5AC", "MUC5B", "GPX3"],
    "Olfactory HBCs": ["TP63", "KRT5", "CXCL14", "SOX2", "MEG3"],
    "Olfactory Ensheathing Glia": ["S100B", "PLP1", "PMP2", "MPZ", "ALX3"],
    "Olfactory Microvillar Cells": ["ASCL3", "CFTR", "HEPACAM2", "FOXL1"],
    "Immature Neurons": ["GNG8", "OLIG2", "EBF2", "LHX2", "CBX8"],
    "Mature Neurons": ["GNG13", "EBF2", "CBX8", "RTP1"],
    "GBCs": ["HES6", "ASCL1", "CXCR4", "SOX2", "EZH2", "NEUROD1", "NEUROG1"],
    "Sustentacular Cells": ["CYP2A13", "CYP2J2", "GPX6", "ERMN", "SOX2"],
    "GR-OT": ["MAG", "GPD1", "GAD1", "SLC44A1", "PACSIN1", "LZTS2", "SYNPR", "CD9", "CHGB", "CERS2", "SNCA", "NDRG1", "CCDC47", "CNTN2", "CNDP1"],
    "GL-OT": ["MAG", "PTN", "FRZB", "TPPP", "PLPPR4", "CNDP1", "MIA", "KLK6", "LMO4", "TF", "RND3", "CARNS1", "SPTBN1", "CNTN2", "TMOD1", "SORCS1", "DEPP1", "CHSY3"],
    "EPL-GL": ["CDH19", "SNCA", "MIA", "BIN1", "AHR", "ARPC5L", "FGF2", "HSPH1", "CD59", "KCNAB2", "FRZB", "VSIG4", "CNN3", "ATP5PO", "SERPING1", "TPST1"],
    "EPL-GR": ["ADAMTS3", "CERK", "FA2H", "MCAM", "NCBP2", "FSTL1", "PAGR1", "RELL1", "CACYBP", "HLA-DQA1", "CNDP1", "AIMP2", "ITGA7"],
    "EPL-OT": ["MAG", "STMN2", "CNDP1", "SNCA", "KLK6", "CHGB", "PLP1", "NSG2", "FA2H", "AMPH", "ABCA2", "HNRNPC", "PAQR6", "DBP", "MYLK", "KCNAB2", "CRTAC1"],
    "GL-GR": ["VSIG4", "MCAM", "SNCA", "SPTBN1", "MAG", "FADS3", "BIN1", "TPST1", "PEG10", "MTCH1", "TPPP", "CDH19", "NFE2L1", "CERS2",  "CDH19", "ATP1B1", "HSPB8", "FA2H", "BCL2", "LGMN", "OLR1"]
}

# Annotate cell types based on marker genes
for cell_type, marker_genes in markers.items():
    adata_combined.obs[cell_type] = adata_combined[:, marker_genes].X.mean(axis=1)

# Visualize marker genes in UMAP
sc.pl.umap(adata_combined, color=list(markers.keys()))

# Differential expression analysis
sc.tl.rank_genes_groups(adata_combined, groupby="leiden", method="t-test")
sc.pl.rank_genes_groups(adata_combined, n_genes=20, sharey=False)

# Export marker genes for each cluster
marker_path = Path(r"/mnt/data2/qi2lab/human_OE_scrnaseq/cluster_markers.csv")
marker_results = pd.DataFrame(adata_combined.uns['rank_genes_groups']['names'])
marker_results.to_csv(marker_path, index=False)

# Save the annotated dataset
dataout_path = Path(r"/mnt/data2/qi2lab/human_OE_scrnaseq/annotated_combined_dataset.h5ad")
adata_combined.write(dataout_path)