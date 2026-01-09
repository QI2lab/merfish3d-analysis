"""
Decode using qi2lab GPU decoder and (re)-segment cells based on decoded RNA.

Shepherd 2025/07 - refactor for multiple GPU suport.
Shepherd 2024/12 - refactor
Shepherd 2024/11 - modified script to accept parameters with sensible defaults.
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from merfish3danalysis.PixelDecoder import PixelDecoder
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


def calculate_F1_with_radius(
    qi2lab_coords: np.ndarray,
    qi2lab_gene_ids: np.ndarray,
    gt_coords: np.ndarray,
    gt_gene_ids: np.ndarray,
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

def decode_pixels(
    root_path: Path,
    minimum_pixels_per_RNA: int = 2,
    ufish_threshold: float = 0.1,
    magnitude_threshold: float = (.5,10.),
):
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    merfish_bits : int
        number of bits in codebook
    minimum_pixels_per_RNA : int
        minimum pixels with same barcode ID required to call a spot. Default = 9.
    ufish_threshold : float
        threshold to accept ufish prediction. Default = 0.1
    magnitude_threshold: tuple[float,float], default = (1.,5.)
        lower and upper magnitude threshold to accept a spot. We allow for >2 on upper because 
        spots are normalized to median spot value, not maximum.
    """

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    merfish_bits = 22

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore, 
        use_mask=False, 
        merfish_bits=merfish_bits, 
        num_gpus=1,
        verbose=1,
    )


    tile_idx = 0

    image, scaled, mag, distance, decoded = decoder.decode_one_tile(
        tile_idx=tile_idx,
        display_results=False,
        return_results=True,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        use_normalization=True,
        ufish_threshold=ufish_threshold,
        lowpass_sigma=(0,0,0)
    )
    stage_zyx_um, affine_xform_um = datastore.load_local_stage_position_zyx_um(round=0, tile=tile_idx)

    RNA_file_name = Path("spots_mouse1sample1.csv")
    file_name = root_path / RNA_file_name
    df_RNA = pd.read_csv(file_name)

    # x_min = stage_zyx_um[0] #- datastore.voxel_size_zyx_um[1]*1024
    # x_max = stage_zyx_um[0] + datastore.voxel_size_zyx_um[1]*2048

    # y_min = stage_zyx_um[1] #- datastore.voxel_size_zyx_um[1]*1024
    # y_max = stage_zyx_um[1] + datastore.voxel_size_zyx_um[1]*2048
    # print(f"tile: {tile_idx}")
    # print(f"x_min: {x_min}, x_max: {x_max}")
    # print(f"y_min: {y_min}, y_max: {y_max}")
    # affine_xform_um, _, _ = datastore.load_global_coord_xforms_um(tile=tile_idx)

    x_min = 3804.87 
    x_max = x_min + 2048 * .1085 
    y_min = 1366.11
    y_max = y_min + 2048 * .1085

    mask = (
        df_RNA["global_x"].between(x_min+10, x_max)
        & df_RNA["global_y"].between(y_min+10, y_max)
    )
    df_RNA_filtered = df_RNA[mask].copy()
    
    df_RNA_filtered = df_RNA_filtered[~df_RNA_filtered["target_molecule_name"].str.startswith("Blank-")]
    # df_RNA_filtered['global_x'] += affine_xform_um[1][3]#- 5*datastore.voxel_size_zyx_um[1]
    # df_RNA_filtered['global_y'] += affine_xform_um[2][3]#- 7*datastore.voxel_size_zyx_um[2]
    #df_RNA_filtered = df_RNA_filtered[df_RNA_filtered["global_z"]>0]
    merlin_coords = df_RNA_filtered[['global_z','global_x', 'global_y']].to_numpy()
    merlin_gene_ids = df_RNA_filtered['target_molecule_name'].to_numpy()
    merlin_coords[:,1] = merlin_coords[:,1]-x_min
    merlin_coords[:,2] = merlin_coords[:,2]-y_min

    merlin_coords[:,1] = merlin_coords[:,1] / .1085
    merlin_coords[:,2] = merlin_coords[:,2] / .1085
    merlin_coords[:,0] = merlin_coords[:,0] / 1.5

    decoded_spots = datastore.load_local_decoded_spots(tile=tile_idx)
    decoded_spots["gene_id"] = decoded_spots["gene_id"].str.strip().str.replace("1-Mar","March1",regex=False)
    decoded_spots = decoded_spots[~decoded_spots["gene_id"].str.startswith("Blank-")]

    mask_qil2ab = (
        decoded_spots["tile_x"].between(100, 1948)
        & decoded_spots["tile_y"].between(100, 1948)
    )
    decoded_spots_filter = decoded_spots[mask_qil2ab].copy()
    qi2lab_coords = decoded_spots_filter[['tile_z','tile_y', 'tile_x']].to_numpy()
    qi2lab_gene_ids = decoded_spots_filter['gene_id'].to_numpy()
    #qi2lab_coords[:,0] = qi2lab_coords[:,0] * 1.5

    # polyDT_image = datastore.load_local_registered_image(
    #     tile=tile_idx,
    #     round=0,
    #     return_future=False
    # )

    decoded_spots = decoder._df_barcodes
    qi2lab_coords = decoded_spots[['tile_z','tile_y', 'tile_x']].to_numpy()
    qi2lab_gene_ids = decoded_spots['gene_id'].to_numpy()

    results = calculate_F1_with_radius(
        qi2lab_coords,
        qi2lab_gene_ids,
        merlin_coords,
        merlin_gene_ids,
        15.0
    )

    #distance[distance>0.7] = -1

    print(f"F1 score: {results["F1 Score"]}")

    import napari
    viewer = napari.Viewer()

    viewer.add_image(
        decoded,
        #scale=[1.5,datastore.voxel_size_zyx_um[1],datastore.voxel_size_zyx_um[2]],
        #translate=(stage_zyx_um[0]+affine_xform_um[1][3],stage_zyx_um[1]+affine_xform_um[2][3])
    )
    # viewer.add_image(
    #     # np.max(scaled,axis=0),
    #     # scale=[1.5,datastore.voxel_size_zyx_um[1],datastore.voxel_size_zyx_um[2]],
    #     # translate=(stage_zyx_um[0]+affine_xform_um[1][3],stage_zyx_um[1]+affine_xform_um[2][3])
    # )
    # viewer.add_image(
    #     polyDT_image,
    #     scale=[1.5,datastore.voxel_size_zyx_um[1],datastore.voxel_size_zyx_um[2]],
    #     translate=(stage_zyx_um[0]+affine_xform_um[1][3],stage_zyx_um[1]+affine_xform_um[2][3])
    # )
    viewer.add_image(
        mag,
        #scale=[1.5,datastore.voxel_size_zyx_um[1],datastore.voxel_size_zyx_um[2]],
        #translate=(stage_zyx_um[0]+affine_xform_um[1][3],stage_zyx_um[1]+affine_xform_um[2][3])
    )
    viewer.add_image(
        distance,
        #scale=[1.5,datastore.voxel_size_zyx_um[1],datastore.voxel_size_zyx_um[2]],
        #translate=(stage_zyx_um[0]+affine_xform_um[1][3],stage_zyx_um[1]+affine_xform_um[2][3])
    )
    viewer.add_points(merlin_coords,size=5,face_color="cyan")
    viewer.add_points(qi2lab_coords,size=5,symbol='s',face_color="orange")
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'px'
    napari.run()




if __name__ == "__main__":
    root_path = Path(r"/media/dps/data/zhuang")
    decode_pixels(root_path=root_path)