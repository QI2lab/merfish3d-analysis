This example requires the optional installation of our OPM processing package,
pip install .[opm]

See comments in each file for specifics.

Order to run:

1. 01_convert_to_qi2lab.py (hours)
2. 02_register_and_deconvolve.py (days)
3. 03_cellpose_segmentation.py (minutes)
4. 04_pixeldecode_and_baysor.py (days)
5. 05_correlation_with_bulk_RNAseq.py (minutes)
6. 06_cluster_spatial_results.py (minutes)
7. 07_build_and_export_scRNAseq_ref.py (minutes)
8. 08_jsta_resegmentation.py (hours)