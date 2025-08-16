"""
Sweep through decoding parameters and calculate F1-score using known ground truth.

Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2024/12 - create script to run on simulation.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.PixelDecoder import PixelDecoder, time_stamp
from pathlib import Path
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
from numpy.typing import ArrayLike
import json
from tqdm import tqdm

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

    test_tile_data = datastore.load_local_corrected_image(tile=0,round=0,return_future=False)


    # Extract coordinates and gene_ids from ground truth
    offset = [
        0, 
        test_tile_data.shape[1]/2*datastore.voxel_size_zyx_um[1],
        test_tile_data.shape[2]/2*datastore.voxel_size_zyx_um[2]
    ]

    gt_coords = gt_spots[['Z', 'X', 'Y']].to_numpy() # note the tranpose, simulation GT is swapped X & Y
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

def decode_pixels(
    root_path: Path, 
    mag_threshold: float, 
    ufish_threshold: float, 
):
    """Run pixel decoding with the given parameters.
    
    Parameters
    ----------
    root_path : Path
        The root path of the experiment.
    mag_threshold : float
        The magnitude threshold
    ufish_threshold : float
        The ufish threshold.
    """

    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    merfish_bits = datastore.num_bits

    decoder = PixelDecoder(
        datastore=datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        verbose=0
    )

    decoder._global_normalization_vectors()
    decoder.optimize_normalization_by_decoding(
        n_random_tiles=1,
        n_iterations=1,
        ufish_threshold=ufish_threshold,
        magnitude_threshold=mag_threshold
    )

    decoder.decode_all_tiles(
        assign_to_cells=False,
        prep_for_baysor=False,
        magnitude_threshold=mag_threshold,
        ufish_threshold=ufish_threshold
    )

def sweep_decode_params(
    root_path: Path,
    gt_path: Path,
    ufish_threshold_range=(0.1, 0.4),
    ufish_threshold_step=0.1,
    mag_threshold_range=(0.9,2.6),
    mag_threshold_step=.1,
):
    """Sweep through decoding parameters and calculate F1 scores.
    
    Parameters
    ----------
    root_path : Path
        The root path of the experiment.
    gt_path : Path
        The path to the ground truth spots.
    ufish_threshold_range : tuple, default [0.1,0.4)
        The range of ufish thresholds to sweep through.
    ufish_threshold_step : float, default .05
        The step size for the ufish threshold sweep
    mag_threshold_range : tuple, default [0.9,2.6)
        The range of minimum magnitude threshold to sweep through.
    mag_threshold_step : float, default 0.1
        The step size for the magnitude threshold.
    """

    mag_values = np.arange(
        mag_threshold_range[0], 
        mag_threshold_range[1], 
        mag_threshold_step, 
        dtype=np.float32
    ).tolist()

    ufish_values = np.arange(
        ufish_threshold_range[0], 
        ufish_threshold_range[1], 
        ufish_threshold_step, 
        dtype=np.float32
    ).tolist()

    results = {}

    for ufish in ufish_values:
        for mag in mag_values:
            params = {
                "fdr": .05,
                "min_pixels": 9,
                "mag_thresh": round(mag,1),
                "ufish_threshold": round(ufish, 1)
            }

            try:
                print(time_stamp(), f"ufish threshold: {round(ufish,1)}; magnitude threshold: {round(mag,1)}")
                decode_pixels(
                    root_path=root_path,
                    mag_threshold=round(mag,1),
                    ufish_threshold=round(ufish,1),
                )

                result = calculate_F1(
                    root_path=root_path,
                    gt_path=gt_path,
                    search_radius=0.5
                )
            except Exception as e:
                result = {"error": str(e)}

            results[str(params)] = result

    save_path = root_path / "decode_params_results.json"
    with save_path.open(mode='w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)

if __name__ == "__main__":
    root_path = Path(r"/home/dps/Documents/2025_merfish3d_paper/example_16bit_flat/0.315/sim_acquisition")
    gt_path = Path(r"/home/dps/Documents/2025_merfish3d_paper/example_16bit_flat/0.315/GT_spots.csv")
    sweep_decode_params(root_path=root_path, gt_path=gt_path)