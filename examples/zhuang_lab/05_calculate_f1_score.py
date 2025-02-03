"""
Calculate F1-score between merfish3d-analysis and MERLIN.

We have to swap yx for merfish3d-analysis to match MERLIN results, due to our
choice on swapping coordinates when creating the datastore.

Shepherd 2025/01 - create script.
"""

import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from numpy.typing import ArrayLike
from pprint import pp

def calculate_F1_with_radius(
    qi2lab_coords: ArrayLike,
    qi2lab_gene_ids: ArrayLike,
    merlin_coords: ArrayLike,
    merlin_gene_ids: ArrayLike,
    radius: float
) -> dict:
    """Calculate F1 score based on spatial proximity and gene identity.
    
    Parameters
    ----------
    qi2lab_coords: ArrayLike
        z,y,x coordinates for found spots in microns. World coordinates.
    qi2lab_gene_ids: ArrayLike
        matched gene ids for found spots
    merlin_coords: ArrayLike,
        z,y,x, coordinates for ground truth spots in microns. World coordinates.
    merlin_gene_ids: ArrayLike
        match gene ids for ground truth spots
    radius: float
        search radius in 3D
    
    Returns
    -------
    resuts: dict
        results for F1 calculation.
    """
    
    merlin_tree = cKDTree(merlin_coords)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    matched_gt_indices = set() 
    for i, query_coord in enumerate(qi2lab_coords):
        qi2lab_gene_id = qi2lab_gene_ids[i]

        nearby_indices = merlin_tree.query_ball_point(query_coord, r=radius)

        if not nearby_indices:
            false_positives += 1
            continue
        
        match_found = False
        for idx in nearby_indices:
            if idx in matched_gt_indices:
                continue
            
            if merlin_gene_ids[idx] == qi2lab_gene_id:
                match_found = True
                true_positives += 1
                matched_gt_indices.add(idx)
                break

        if not match_found:
            false_positives += 1

    false_negatives = len(merlin_gene_ids) - len(matched_gt_indices)

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
    qi2lab_path: Path,
    merlin_path: Path,
    search_radius: float
):
    """Calculate F1 using ground truth.

    Parameters
    ----------
    qi2lab_path: Path
        path to qi2lab merfish3d-analysis spot file
    merlin_path: Path
        path to Zhuang MERLIN spot file
    search_radius: float
        search radius for a sphere in microns. Should be 2-3x the z step,
        depending on the amount of low-pass blur applied.
        
    Returns
    -------
    results: dict
        dictionary of results for F1 score calculation 
    """

    # Extract coordinates and gene_ids from qi2lab merfish3d-analysis
    qi2lab_spots = pd.read_csv(qi2lab_path)
    qi2lab_coords = qi2lab_spots[['z', 'x', 'y']].to_numpy()
    qi2lab_gene_ids = qi2lab_spots['gene'].to_numpy()

    # Extract coordinates and gene_ids from Zhuang MERLIN analysis
    merlin_spots = pd.read_csv(merlin_path)
    merlin_coords = merlin_spots[['global_z', 'global_y', 'global_x']].to_numpy()
    merlin_gene_ids = merlin_spots['target_molecule_name'].to_numpy()

    results = calculate_F1_with_radius(
        qi2lab_coords,
        qi2lab_gene_ids,
        merlin_coords,
        merlin_gene_ids,
        search_radius
    )
    
    print(f"Number of qi2lab spots: {len(qi2lab_spots)}")
    print(f"Number of Zhuang spots: {len(merlin_spots)}")
    
    return results
    
if __name__ == "__main__":
    root_path = Path(r"/mnt/data/zhuang/")
    qi2lab_spots_path = root_path / Path(r"qi2labdatastore/segmentation/baysor/segmentation.csv")
    merlin_spots_path = root_path / Path(r"mop/mouse_sample1_raw/zhuang_decoded_codewords/spots_mouse1sample1.csv")
    results = calculate_F1(qi2lab_spots_path,merlin_spots_path,search_radius=1.5)
    pp(results)