import scanpy as sc
from pathlib import Path
import pandas as pd
import gzip


h5ad_path = Path(r"/mnt/data2/qi2lab/human_OE_scrnaseq/annotated_combined_dataset.h5ad")

adata = sc.read_h5ad(h5ad_path)
if isinstance(adata.X, pd.DataFrame):
    gene_matrix = adata.X
else:
    gene_matrix = pd.DataFrame(
        adata.X,
        index=adata.obs.index,  # Cell IDs
        columns=adata.var.index  # Gene names
    )

cell_type_columns = adata.obs.loc[:, 'CD8+ T Cells':'GL-GR']
adata.obs['cell_type'] = cell_type_columns.idxmax(axis=1)

matrix_path = Path(r"/mnt/data2/qi2lab/human_OE_scrnaseq/ref_sc.tsv.gz")
with gzip.open(matrix_path, "wt") as f:
    gene_matrix.to_csv(matrix_path, sep="\t", index_label="Cell_ID")

celltypes_path = Path(r"/mnt/data2/qi2lab/human_OE_scrnaseq/cell_types.tsv.gz")
with gzip.open(celltypes_path, "wt") as f:
    cell_types = adata.obs['cell_type']
    cell_types.to_csv(celltypes_path, sep="\t", header=["Cell_Type"], index_label="Cell_ID")