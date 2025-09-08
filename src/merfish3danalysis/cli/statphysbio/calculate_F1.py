"""
Calculate F1-score using known ground truth

Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2024/12 - create script to run on simulation.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
from numpy.typing import ArrayLike
import typer

def calculate_F1_with_radius(
    qi2lab_coords: ArrayLike,
    qi2lab_gene_ids: ArrayLike,
    gt_coords: ArrayLike,
    gt_gene_ids: ArrayLike,
    radius: float
) -> dict:
    """
    Compute F1 using greedy closest-first matching within a max radius, with same-gene and
    one-to-one constraints.

    Algorithm
    ---------
    1) For each gene present in both sets, find all qi2lab↔GT pairs within `radius` via KD-trees.
    2) Concatenate all candidates across genes; sort by distance (ascending).
    3) Greedily accept pairs if both endpoints are unused; remove both from consideration.
    4) TP = #accepted pairs; FP = #qi2lab unmatched; FN = #GT unmatched.

    Parameters
    ----------
    qi2lab_coords : ArrayLike
        (Nq, 3) z, y, x coordinates for found spots (microns; world coords).
    qi2lab_gene_ids : ArrayLike
        (Nq,) gene IDs for found spots.
    gt_coords : ArrayLike
        (Ng, 3) z, y, x coordinates for ground-truth spots (microns; world coords).
    gt_gene_ids : ArrayLike
        (Ng,) gene IDs for ground-truth spots.
    radius : float
        Maximum 3D distance (same units as coordinates) allowed for matching.

    Returns
    -------
    results : dict
        {
          "F1 Score": float,
          "Precision": float,
          "Recall": float,
          "True Positives": int,
          "False Positives": int,
          "False Negatives": int,
        }
    """
    qi2lab_coords = np.asarray(qi2lab_coords)
    qi2lab_gene_ids = np.asarray(qi2lab_gene_ids)
    gt_coords = np.asarray(gt_coords)
    gt_gene_ids = np.asarray(gt_gene_ids)

    Nq = qi2lab_coords.shape[0]
    Ng = gt_coords.shape[0]

    # Short-circuit trivial cases
    if Nq == 0 and Ng == 0:
        return {
            "F1 Score": 1.0,
            "Precision": 1.0,
            "Recall": 1.0,
            "True Positives": 0,
            "False Positives": 0,
            "False Negatives": 0,
        }
    if Nq == 0:
        return {
            "F1 Score": 0.0,
            "Precision": 0.0,
            "Recall": 0.0 if Ng > 0 else 1.0,
            "True Positives": 0,
            "False Positives": 0,
            "False Negatives": int(Ng),
        }
    if Ng == 0:
        return {
            "F1 Score": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "True Positives": 0,
            "False Positives": int(Nq),
            "False Negatives": 0,
        }

    # Build candidate pairs within radius, per gene; merge globally
    pair_q_idx_all: list[np.ndarray] = []
    pair_g_idx_all: list[np.ndarray] = []
    pair_dist_all:  list[np.ndarray] = []

    common_genes = np.intersect1d(np.unique(qi2lab_gene_ids), np.unique(gt_gene_ids))
    for gene in common_genes:
        q_idx = np.flatnonzero(qi2lab_gene_ids == gene)
        g_idx = np.flatnonzero(gt_gene_ids == gene)
        if q_idx.size == 0 or g_idx.size == 0:
            continue

        q_pts = qi2lab_coords[q_idx]
        g_pts = gt_coords[g_idx]

        q_tree = cKDTree(q_pts)
        g_tree = cKDTree(g_pts)
        try:
            dist_coo = q_tree.sparse_distance_matrix(
                g_tree, max_distance=radius, output_type="coo_matrix"
            )
        except TypeError:
            dist_coo = q_tree.sparse_distance_matrix(g_tree, max_distance=radius).tocoo()

        if dist_coo.nnz == 0:
            continue

        # Local -> global index mapping
        q_local = dist_coo.row
        g_local = dist_coo.col
        dists   = dist_coo.data

        pair_q_idx_all.append(q_idx[q_local])
        pair_g_idx_all.append(g_idx[g_local])
        pair_dist_all.append(dists)

    if not pair_q_idx_all:
        # No candidates within radius at all
        tp = 0
        fp = int(Nq)
        fn = int(Ng)
    else:
        pair_q_idx = np.concatenate(pair_q_idx_all)
        pair_g_idx = np.concatenate(pair_g_idx_all)
        pair_dist  = np.concatenate(pair_dist_all)

        # Sort globally by distance (closest first)
        order = np.argsort(pair_dist, kind="stable")
        pair_q_idx = pair_q_idx[order]
        pair_g_idx = pair_g_idx[order]

        # Greedy selection with one-to-one constraint
        q_used = np.zeros(Nq, dtype=bool)
        g_used = np.zeros(Ng, dtype=bool)

        tp = 0
        for qi, gi in zip(pair_q_idx, pair_g_idx):
            if q_used[qi] or g_used[gi]:
                continue
            q_used[qi] = True
            g_used[gi] = True
            tp += 1

        fp = int(Nq - tp)
        fn = int(Ng - g_used.sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "True Positives": int(tp),
        "False Positives": int(fp),
        "False Negatives": int(fn),
    }

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def calculate_F1(
    root_path: Path,
    search_radius: float = 1.0
):
    """Calculate F1 using ground truth.

    Parameters
    ----------
    root_path: Path
        path to experiment
    gt_path: Path
        path to ground truth file
    search_radius: float
        search radius for a sphere in microns. Should be 2-3x the z step,
        depending on the amount of low-pass blur applied.
        
    Returns
    -------
    results: dict
        dictionary of results for F1 score calculation 
    """

    # initialize datastore
    datastore_path = root_path / Path("sim_acquisition") / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    gene_ids, _ = datastore.load_codebook_parsed()
    decoded_spots = datastore.load_global_filtered_decoded_spots()
    gt_path = root_path / Path("GT_spots.csv")
    gt_spots = pd.read_csv(gt_path)
    gene_ids = np.array(gene_ids)
        
    # Extract coordinates and gene_ids from analyzed
    qi2lab_coords = decoded_spots[['global_z', 'global_y', 'global_x']].to_numpy()
    qi2lab_gene_ids = decoded_spots['gene_id'].to_numpy()

    test_tile_data = datastore.load_local_corrected_image(tile=0,round=0,return_future=False)


    # Extract coordinates and gene_ids from ground truth
    offset = [
        0, 
        1*test_tile_data.shape[1]/2*datastore.voxel_size_zyx_um[1]-datastore.voxel_size_zyx_um[1]/2,
        1*test_tile_data.shape[2]/2*datastore.voxel_size_zyx_um[2]-datastore.voxel_size_zyx_um[2]/2
    ]

    gt_coords = gt_spots[['Z', 'X', 'Y']].to_numpy() # note the tranpose, simulation GT is swapped X & Y
    gt_coords_offset = gt_coords + np.array(offset)
    gt_gene_ids = gene_ids[(gt_spots['Gene_label'].to_numpy(dtype=int)-1)]
    
    results = calculate_F1_with_radius(
        qi2lab_coords,
        qi2lab_gene_ids,
        gt_coords_offset,
        gt_gene_ids,
        search_radius
    )
    
    print("F1 Score Results:")
    print(results)

def main():
    app()

if __name__ == "__main__":
    main()