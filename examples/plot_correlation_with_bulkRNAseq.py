import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

bulkseq_path = Path('/mnt/opm3/merfish_panels/20240615/humanOB_bulk_RNASeq/SRR3376220/cufflinks/genes.fpkm_tracking')

merfish_path = Path('/mnt/data/qi2lab/20240317_OB_MERFISH_7')
segemented_results = merfish_path / Path('processed_v2') / Path('decoded') / Path('baysor_formatted_genes.csv')

merfish_df = pd.read_csv(segemented_results)
merfish_counts_df = merfish_df['gene_id'].value_counts().reset_index()
merfish_counts_df.columns = ['gene_id','counts']

seq_df = pd.read_csv(bulkseq_path,sep='\t')
rnaseq_counts_df = seq_df[['gene_short_name','FPKM']].copy()
rnaseq_counts_df.rename(columns={'gene_short_name': 'gene_id'},inplace=True)

merged_df = pd.merge(merfish_counts_df, rnaseq_counts_df, on='gene_id', how='left', indicator=True)
plt.scatter(np.log10(merged_df['FPKM']),np.log10(merged_df['counts']), alpha=0.5)
plt.ylabel('Log10(Counts)')
plt.xlabel('Log10(FPKM)')
plt.title('Log-Log Correlation Plot')
plt.grid(True)
plt.show()