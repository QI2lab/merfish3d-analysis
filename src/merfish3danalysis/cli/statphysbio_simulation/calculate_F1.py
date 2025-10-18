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
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy closest-first matching within `radius`, with strict same-gene and one-to-one constraints.
    Returns F1 metrics and (TP coords, FP coords, FN coords).
    """
    # Canonicalize inputs
    qi2lab_coords = np.asarray(qi2lab_coords, float)
    gt_coords = np.asarray(gt_coords, float)
    # Force gene IDs to a common comparable dtype (strings) to avoid category/int/object mismatches
    qi2lab_gene_ids = np.asarray(qi2lab_gene_ids).astype(str)
    gt_gene_ids = np.asarray(gt_gene_ids).astype(str)

    Nq = qi2lab_coords.shape[0]
    Ng = gt_coords.shape[0]

    # Trivial cases
    if Nq == 0 and Ng == 0:
        return (
            {"F1 Score": 1.0, "Precision": 1.0, "Recall": 1.0,
             "True Positives": 0, "False Positives": 0, "False Negatives": 0},
            np.empty((0, 3), float), np.empty((0, 3), float), np.empty((0, 3), float)
        )
    if Nq == 0:
        return (
            {"F1 Score": 0.0, "Precision": 0.0, "Recall": 0.0 if Ng > 0 else 1.0,
             "True Positives": 0, "False Positives": 0, "False Negatives": int(Ng)},
            np.empty((0, 3), float), np.empty((0, 3), float), gt_coords.copy()
        )
    if Ng == 0:
        return (
            {"F1 Score": 0.0, "Precision": 0.0, "Recall": 0.0,
             "True Positives": 0, "False Positives": int(Nq), "False Negatives": 0},
            np.empty((0, 3), float), qi2lab_coords.copy(), np.empty((0, 3), float)
        )

    # Build candidate pairs within radius, per shared gene (strict same-gene pooling)
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
        except TypeError:  # SciPy<1.8 fallback
            dist_coo = q_tree.sparse_distance_matrix(g_tree, max_distance=radius).tocoo()

        if dist_coo.nnz == 0:
            continue

        # Local (per-gene) -> global indices
        pair_q_idx_all.append(q_idx[dist_coo.row])
        pair_g_idx_all.append(g_idx[dist_coo.col])
        pair_dist_all.append(dist_coo.data)

    if not pair_q_idx_all:
        q_used = np.zeros(Nq, dtype=bool)
        g_used = np.zeros(Ng, dtype=bool)
        tp_coords = np.empty((0, 3), float)
        fp_coords = qi2lab_coords.copy()
        fn_coords = gt_coords.copy()
        tp = 0
    else:
        pair_q_idx = np.concatenate(pair_q_idx_all)
        pair_g_idx = np.concatenate(pair_g_idx_all)
        pair_dist  = np.concatenate(pair_dist_all)

        # Sort by distance ascending; stable for deterministic ties
        order = np.argsort(pair_dist, kind="stable")
        pair_q_idx = pair_q_idx[order]
        pair_g_idx = pair_g_idx[order]

        # Greedy one-to-one matching with explicit gene check (belt & suspenders)
        q_used = np.zeros(Nq, dtype=bool)
        g_used = np.zeros(Ng, dtype=bool)
        matched_q = []
        matched_g = []

        for qi, gi in zip(pair_q_idx, pair_g_idx):
            if q_used[qi] or g_used[gi]:
                continue
            # This should always pass because pairs were built per gene; keep anyway:
            if qi2lab_gene_ids[qi] != gt_gene_ids[gi]:
                continue  # never match across genes
            q_used[qi] = True
            g_used[gi] = True
            matched_q.append(qi)
            matched_g.append(gi)

        matched_q = np.asarray(matched_q, dtype=int)
        matched_g = np.asarray(matched_g, dtype=int)

        # Post-condition: strictly same gene for all accepted pairs
        if matched_q.size:
            if not np.all(qi2lab_gene_ids[matched_q] == gt_gene_ids[matched_g]):
                raise RuntimeError("Gene-ID mismatch detected in accepted matches.")

        tp = int(matched_q.size)
        tp_coords = qi2lab_coords[matched_q]
        fp_coords = qi2lab_coords[~q_used]
        fn_coords = gt_coords[~g_used]

    # Counts & metrics
    fp = int((~q_used).sum())
    fn = int((~g_used).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    results = {
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
    }
    return results, tp_coords, fp_coords, fn_coords

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
    image = datastore.load_local_corrected_image(tile=0, round=0,return_future=False)
        
    # Extract coordinates and gene_ids from analyzed
    qi2lab_coords = decoded_spots[['global_z', 'global_y', 'global_x']].to_numpy()
    qi2lab_gene_ids = decoded_spots['gene_id'].to_numpy()
 
    # Extract coordinates and gene_ids from ground truth
    gt_coords = gt_spots[['Z', 'X', 'Y']].to_numpy() # note the tranpose, simulation GT is swapped X & Y

    # re-center the ground truth to start at (0,0) in (y,x)
    gt_coords_offset = [
            0, 
            (1.*image[0].shape[-2]/2)*datastore.voxel_size_zyx_um[1]-datastore.voxel_size_zyx_um[1]/2,
            (1.*image[0].shape[-1]/2)*datastore.voxel_size_zyx_um[2]-datastore.voxel_size_zyx_um[2]/2,
        ]

    gt_coords = gt_coords + gt_coords_offset
    gt_gene_ids = gene_ids[(gt_spots['Gene_label'].to_numpy(dtype=int)-1)]
    
    results, _, _, _ = calculate_F1_with_radius(
        qi2lab_coords,
        qi2lab_gene_ids,
        gt_coords,
        gt_gene_ids,
        search_radius
    )
    
    print("F1 Score Results:")
    print(results)

def main():
    app()

if __name__ == "__main__":
    main()
