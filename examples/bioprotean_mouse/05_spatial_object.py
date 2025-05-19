import pandas as pd
import numpy as np
from anndata import AnnData
from scipy import sparse
from pathlib import Path
import json

def baysor_csv_to_anndata(
    root_path: Path,
    confidence_cutoff: float = 0.7
):
    """
    Converts a Baysor segmentation.csv to an AnnData object.

    Parameters
    ----------
    root_path : str or Path
        Path to root data folder.
    confidence_cutoff : float
        Minimum assignment confidence to include a transcript.

    Returns
    -------
    AnnData
        AnnData object with cell x gene count matrix.
    """

    # Load segmentation.csv
    segmentation_path = root_path / Path("qi2labdatastore/segmentation/segmentation.csv")
    df = pd.read_csv(segmentation_path, usecols=["gene", "cell", "assignment_confidence"])

    # Filter by confidence
    df = df[df["assignment_confidence"] >= confidence_cutoff]

    # Drop unassigned cells
    df = df[df["cell"].notna() & (df["cell"] != "")]

    # Ensure consistent string types
    df["cell"] = df["cell"].astype(str)
    df["gene"] = df["gene"].astype(str)

    # Pivot to get cells x genes matrix
    counts = df.groupby(["cell", "gene"]).size().unstack(fill_value=0)

    # Create AnnData object (X: cells x genes)
    adata = AnnData(X=sparse.csr_matrix(counts.values))
    adata.obs_names = counts.index  # cells
    adata.var_names = counts.columns  # genes

    # Loading Baysor segmentations
    polygons_json_path = root_path / Path("qi2labdatastore/segmentation/segmentation_polygons_2d.json")

    # Load the JSON file
    with open(polygons_json_path) as f:
        data = json.load(f)

    # Extract centroids
    centroids = {}
    for feature in data['features']:
        cell_id = feature['id']
        coords = np.array(feature['geometry']['coordinates'][0])
        centroid = coords.mean(axis=0)
        centroids[cell_id] = centroid

    # Create a DataFrame of centroids
    centroid_df = pd.DataFrame.from_dict(centroids, orient='index', columns=['x', 'y'])

    # Align with AnnData
    # Ensure that adata.obs_names are strings
    adata.obs_names = adata.obs_names.astype(str)

    # Reindex centroid_df to match adata.obs_names
    centroid_df = centroid_df.reindex(adata.obs_names)

    # Add to AnnData and save to file
    adata.obsm['spatial'] = centroid_df.values
    adata.write(root_path / Path("qi2labdatastore/mtx_output/spatial_ad2.h5ad"))

if __name__ == "__main__":
    baysor_csv_to_anndata(
        root_path=Path(r"/data/MERFISH/20241108_Bartelle_MouseMERFISH_LC/"),
        confidence_cutoff=0.7
        )

# /data/MERFISH/20241108_Bartelle_MouseMERFISH_LC/
# /mnt/data/bartelle/20241108_Bartelle_MouseMERFISH_LC