"""
Calculate F1-score using known ground truth

Shepherd 2024/12 - create script to run on simulation.
"""


from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.postprocess.PixelDecoder import PixelDecoder
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
    gt_coords = gt_spots[['Z', 'Y', 'X']].to_numpy()
    gt_gene_ids = gene_ids[(gt_spots['Gene_label'].to_numpy(dtype=int)-1)]
    
    results = calculate_F1_with_radius(
        qi2lab_coords,
        qi2lab_gene_ids,
        gt_coords,
        gt_gene_ids,
        search_radius
    )
    
    return results


def decode_pixels(
    root_path: Path,
    minimum_pixels_per_RNA: int = 5,
    ufish_threshold: float = 0.5,
    fdr_target: float = .05
):
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    minimum_pixels_per_RNA : int
        minimum pixels with same barcode ID required to call a spot. Default = 3.
    ufish_threshold : float
        threshold to accept ufish prediction. Default = 0.5
    fdr_target : float
        false discovery rate (FDR) target. Default = .05
    """

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    merfish_bits = datastore.num_bits
    

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore, 
        use_mask=False, 
        merfish_bits=merfish_bits, 
        verbose=1
    )
    
    # reset global vectors to ensure they are recalculated
    decoder._global_normalization_vectors()
    
    # optimize normalization weights through iterative decoding and update
    decoder.optimize_normalization_by_decoding(
        n_random_tiles=1,
        n_iterations=2,
        minimum_pixels=minimum_pixels_per_RNA,
        ufish_threshold=ufish_threshold
    )
    datastore.iterative_background_vector
    
    decoder.decode_all_tiles(
        assign_to_cells=False,
        prep_for_baysor=False,
        minimum_pixels=minimum_pixels_per_RNA,
        fdr_target=fdr_target,
        ufish_threshold=ufish_threshold
    )
    
def test_decode_params(
    root_path: Path,
    gt_path: Path,
    ufish_threshold_range = [0.1,0.6],
    ufish_threshold_step = 0.05,
    min_pixels_range = [2,14],
    min_pixels_step = 1,
    fdr_range = [.05,.3],
    fdr_range_step = .05
):
    fdr_range_float = np.arange(fdr_range[0],fdr_range[1],fdr_range_step,dtype=np.float32).tolist()
    min_pixels_range_int = np.arange(min_pixels_range[0],min_pixels_range[1],min_pixels_step,dtype=np.float32).tolist()
    ufish_threshold_range_float = np.arange(ufish_threshold_range[0],ufish_threshold_range[1],ufish_threshold_step,dtype=np.float32).tolist()
    for fdr in fdr_range_float:
        for min_pixels in min_pixels_range_int:
            for ufish in ufish_threshold_range_float:
                try:
                    decode_pixels(
                        root_path = root_path,
                        minimum_pixels_per_RNA = min_pixels,
                        ufish_threshold = ufish,
                        fdr_target = fdr
                    )
                    
                    result = calculate_F1(
                        root_path=root_path,
                        gt_path=gt_path,
                        search_radius=0.75
                    )
                except:
                    result = {}
                
                file_path = Path("fdr: "+str(np.round(fdr,2))+"; min pixels: "+str(np.round(min_pixels,1))+"; ufish: "+str(np.round(ufish,2))+".json")
                save_result_path = root_path / file_path
                with save_result_path.open(mode='w', encoding='utf-8') as file:
                    json.dump(result,file,indent=2)
                                                                                   

if __name__ == "__main__":
    root_path = Path(r"/mnt/opm3/20241218_statphysbio/sim_acquisition")
    gt_path = Path(r"/mnt/opm3/20241218_statphysbio/GT_spots.csv")
    test_decode_params(root_path=root_path,gt_path=gt_path)
    
