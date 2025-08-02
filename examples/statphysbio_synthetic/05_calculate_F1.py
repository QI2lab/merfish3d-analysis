"""
Calculate F1-score using known ground truth

Shepherd 2024/12 - create script to run on simulation.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
from numpy.typing import ArrayLike

def calculate_F1_with_radius(
    qi2lab_coords: ArrayLike,
    qi2lab_gene_ids: ArrayLike,
    gt_coords: ArrayLike,
    gt_gene_ids: ArrayLike,
    radius: float
) -> dict:
    """Calculate F1 score based on spatial proximity and gene identity.
    
    Parameters
    ----------
    qi2lab_coords: ArrayLike
        z,y,x coordinates for found spots in microns. World coordinates.
    qi2lab_gene_ids: ArrayLike
        matched gene ids for found spots
    gt_coords: ArrayLike,
        z,y,x, coordinates for ground truth spots in microns. World coordinates.
    gt_gene_ids: ArrayLike
        match gene ids for ground truth spots
    radius: float
        search radius in 3D
    
    Returns
    -------
    resuts: dict
        results for F1 calculation.
    """
    
    gt_tree = cKDTree(gt_coords)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    matched_gt_indices = set() 
    for i, query_coord in enumerate(qi2lab_coords):
        qi2lab_gene_id = qi2lab_gene_ids[i]

        nearby_indices = gt_tree.query_ball_point(query_coord, r=radius)

        if not nearby_indices:
            false_positives += 1
            continue
        
        match_found = False
        for idx in nearby_indices:
            if idx in matched_gt_indices:
                continue
            
            if gt_gene_ids[idx] == qi2lab_gene_id:
                match_found = True
                true_positives += 1
                matched_gt_indices.add(idx)
                break

        if not match_found:
            false_positives += 1

    false_negatives = len(gt_gene_ids) - len(matched_gt_indices)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "True Positives": true_positives,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
    }

def calculate_F1(
    root_path: Path,
    gt_path: Path,
    search_radius: float
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
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    gene_ids, _ = datastore.load_codebook_parsed()
    decoded_spots = datastore.load_global_filtered_decoded_spots()
    gt_spots = pd.read_csv(gt_path)
    gene_ids = np.array(gene_ids)
        
    # Extract coordinates and gene_ids from analyzed
    qi2lab_coords = decoded_spots[['global_z', 'global_y', 'global_x']].to_numpy()
    qi2lab_gene_ids = decoded_spots['gene_id'].to_numpy()

    # Extract coordinates and gene_ids from ground truth
    offset = [0,256/2*.1625,256/2*.1625]
    gt_coords = gt_spots[['Z', 'X', 'Y']].to_numpy()
    gt_coords_offset = gt_coords + offset
    gt_gene_ids = gene_ids[(gt_spots['Gene_label'].to_numpy(dtype=int)-1)]
    
    results = calculate_F1_with_radius(
        qi2lab_coords,
        qi2lab_gene_ids,
        gt_coords_offset,
        gt_gene_ids,
        search_radius
    )
    
    return results
    
if __name__ == "__main__":
    root_path = Path(r"/home/max/codes/BiFISH/results/16bit_example/sim_acquisition")
    gt_path = Path(r"/home/max/codes/BiFISH/results/16bit_example/GT_spots.csv")
    results = calculate_F1(root_path=root_path,gt_path=gt_path,search_radius=.75)
    print(results)