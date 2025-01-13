"""
Sweep through decoding parameters and calculate F1-score using known ground truth.

Shepherd 2024/12 - create script to run on simulation.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.PixelDecoder import PixelDecoder
from pathlib import Path
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
from numpy.typing import ArrayLike
import json

def calculate_F1_with_radius(
    qi2lab_coords: ArrayLike,
    qi2lab_gene_ids: ArrayLike,
    gt_coords: ArrayLike,
    gt_gene_ids: ArrayLike,
    radius: float
) -> dict:
    gt_tree = cKDTree(gt_coords)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_gt_indices = set()
    """Calculate F1 score using a radius search.
    
    Parameters
    ----------
    qi2lab_coords : ArrayLike
        The coordinates of the spots to be evaluated.
    qi2lab_gene_ids : ArrayLike
        The gene IDs of the spots to be evaluated.
    gt_coords : ArrayLike
        The coordinates of the ground truth spots.
    gt_gene_ids : ArrayLike
        The gene IDs of the ground truth spots.
    radius : float
        The radius to search for ground truth spots.
    
    Returns
    -------
    dict
        A dictionary containing the F1 score, precision, recall, true positives, false positives, and false negatives.
    """

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

def calculate_F1(root_path: Path, gt_path: Path, search_radius: float) -> dict:
    """Helper function to calculate F1 score using a radius search.
    
    Parameters
    ----------  
    root_path : Path
        The root path of the experiment.
    gt_path : Path
        The path to the ground truth spots.
    search_radius : float
        The radius to search for ground truth spots.
    
    Returns
    -------
    dict
        A dictionary containing the F1 score, precision, recall, true positives, false positives, and false negatives.
    """

    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    gene_ids, _ = datastore.load_codebook_parsed()
    decoded_spots = datastore.load_global_filtered_decoded_spots()
    gt_spots = pd.read_csv(gt_path)

    qi2lab_coords = decoded_spots[['global_z', 'global_y', 'global_x']].to_numpy()
    qi2lab_gene_ids = decoded_spots['gene_id'].to_numpy()

    gt_coords = gt_spots[['Z', 'Y', 'X']].to_numpy()
    gt_gene_ids = np.array(gene_ids)[(gt_spots['Gene_label'].to_numpy(dtype=int)-1)]

    return calculate_F1_with_radius(
        qi2lab_coords,
        qi2lab_gene_ids,
        gt_coords,
        gt_gene_ids,
        search_radius
    )

def decode_pixels(
    root_path: Path, 
    minimum_pixels_per_RNA: int, 
    ufish_threshold: float, 
    fdr_target: float
):
    """Run pixel decoding with the given parameters.
    
    Parameters
    ----------
    root_path : Path
        The root path of the experiment.
    minimum_pixels_per_RNA : int
        The minimum number of pixels per RNA.
    ufish_threshold : float
        The ufish threshold.
    fdr_target : float
        The FDR target.
    """

    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    merfish_bits = datastore.num_bits

    decoder = PixelDecoder(
        datastore=datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        verbose=1
    )

    decoder._global_normalization_vectors()
    decoder.optimize_normalization_by_decoding(
        n_random_tiles=1,
        n_iterations=2,
        minimum_pixels=minimum_pixels_per_RNA,
        ufish_threshold=ufish_threshold
    )

    decoder.decode_all_tiles(
        assign_to_cells=False,
        prep_for_baysor=False,
        minimum_pixels=minimum_pixels_per_RNA,
        fdr_target=fdr_target,
        ufish_threshold=ufish_threshold
    )

def sweep_decode_params(
    root_path: Path,
    gt_path: Path,
    ufish_threshold_range=(0.1, 0.6),
    ufish_threshold_step=0.05,
    min_pixels_range=(2, 14),
    min_pixels_step=1,
    fdr_range=(0.05, 0.3),
    fdr_step=0.05
):
    """Sweep through decoding parameters and calculate F1 scores.
    
    Parameters
    ----------
    root_path : Path
        The root path of the experiment.
    gt_path : Path
        The path to the ground truth spots.
    ufish_threshold_range : tuple, optional
        The range of ufish thresholds to sweep through, by default (0.1, 0.6).
    ufish_threshold_step : float, optional
        The step size for the ufish threshold sweep, by default 0.05.
    min_pixels_range : tuple, optional
        The range of minimum pixels per RNA to sweep through, by default (2, 14).
    min_pixels_step : int, optional
        The step size for the minimum pixels per RNA sweep, by default 1.
    fdr_range : tuple, optional
        The range of FDR targets to sweep through, by default (0.05, 0.3).
    fdr_step : float, optional
        The step size for the FDR target sweep, by default 0.05.
    """

    fdr_values = np.arange(fdr_range[0], fdr_range[1], fdr_step, dtype=np.float32).tolist()
    min_pixels_values = np.arange(min_pixels_range[0], min_pixels_range[1], min_pixels_step, dtype=np.int32).tolist()
    ufish_values = np.arange(ufish_threshold_range[0], ufish_threshold_range[1], ufish_threshold_step, dtype=np.float32).tolist()

    results = {}

    for fdr in fdr_values:
        for min_pixels in min_pixels_values:
            for ufish in ufish_values:
                params = {
                    "fdr": round(fdr, 2),
                    "min_pixels": min_pixels,
                    "ufish_threshold": round(ufish, 2)
                }

                try:
                    decode_pixels(
                        root_path=root_path,
                        minimum_pixels_per_RNA=min_pixels,
                        ufish_threshold=ufish,
                        fdr_target=fdr
                    )

                    result = calculate_F1(
                        root_path=root_path,
                        gt_path=gt_path,
                        search_radius=0.75
                    )
                except Exception as e:
                    result = {"error": str(e)}

                results[str(params)] = result

    save_path = root_path / "decode_params_results.json"
    with save_path.open(mode='w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)

if __name__ == "__main__":
    root_path = Path(r"/mnt/opm3/20241218_statphysbio/sim_acquisition")
    gt_path = Path(r"/mnt/opm3/20241218_statphysbio/GT_spots.csv")
    sweep_decode_params(root_path=root_path, gt_path=gt_path)