import napari
from tifffile import imread
import pandas as pd
import numpy as np
from pathlib import Path
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
import typer

app = typer.Typer()
app.pretty_exceptions_enable = False

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


@app.command()
def build_figure(root_path: Path, sim_path: Path = "example_16bit_cells"):
    """Build figure for different z spacings.
    
    Parameters
    ----------
    root_path: Path
        path to experiment
    sim_path: Path, default "example_16bit_cells"
        path to simulation within experiment
    """

    z_spacings = ["0.315", "1.0", "1.5"]

    top_path = root_path / Path(sim_path)
    images = []
    status = []
    points = []
    f1 = []
    for z in z_spacings:
        temp = imread(top_path / Path(str(z)) / Path("sim_acquisition") / Path("data_r0001_tile0000_1") / Path("data_r0001_tile0000.tif"))
        metadata = pd.read_csv(top_path / Path(str(z)) / Path("scan_metadata.csv"))
        print(float(metadata["axial_step_size [micron]"]))
        print(float(metadata["pixel_size [micron]"]))
        if not(z == "0.315"):
            temp_filled = np.zeros_like(images[0])
            if z == "1.0":
                image_10_voxel = np.array([
                    float(metadata["axial_step_size [micron]"]), 
                    float(metadata["pixel_size [micron]"]), 
                    float(metadata["pixel_size [micron]"])
                ])
                for z_idx in range(temp.shape[1]):
                    temp_filled[:, int(np.floor(z_idx*image_10_voxel[0]/image_0315_voxel[0])),:,:] = temp[:,z_idx,:,:]
            elif z == "1.5":
                image_15_voxel = np.array([
                    float(metadata["axial_step_size [micron]"]), 
                    float(metadata["pixel_size [micron]"]), 
                    float(metadata["pixel_size [micron]"])
                ])
                for z_idx in range(temp.shape[1]):
                    temp_filled[:, int(np.floor(z_idx*image_15_voxel[0]/image_0315_voxel[0])),:,:] = temp[:,z_idx,:,:]
            images.append(temp_filled)
        else:
            image_0315_voxel = np.array([
                float(metadata["axial_step_size [micron]"]), 
                float(metadata["pixel_size [micron]"]), 
                float(metadata["pixel_size [micron]"])
            ])
            images.append(temp)

        gt_spots = pd.read_csv(top_path / Path(str(z)) / Path("GT_spots.csv"))
        codebook = pd.read_csv(top_path / Path(str(z)) / Path("codebook.csv"))
        codebook_genes = codebook['gene_id'].to_numpy()
        
        decoded_spots = pd.read_parquet(top_path / Path(str(z)) / Path("sim_acquisition") / Path("qi2labdatastore") / Path("all_tiles_filtered_decoded_features") / Path("decoded_features.parquet"))
        qi2lab_coords = decoded_spots[['global_z', 'global_y', 'global_x']].to_numpy()
        qi2lab_gene_ids = decoded_spots['gene_id'].to_numpy()
        gt_coords = gt_spots[['Z','X','Y']].to_numpy()
        gt_gene_ids = codebook_genes[(gt_spots['Gene_label'].to_numpy(dtype=int)-1)]
        gt_coords_offset = [
            0, 
            (1.*images[0].shape[-2]/2)*image_0315_voxel[1]-image_0315_voxel[1]/2,
            (1.*images[0].shape[-1]/2)*image_0315_voxel[2]-image_0315_voxel[2]/2,
        ]

        gt_coords = gt_coords + gt_coords_offset

        if z == "0.315":
            z_f1, mq, q_fp, g_fn = calculate_F1_with_radius(
                qi2lab_coords,
                qi2lab_gene_ids,
                gt_coords,
                gt_gene_ids,
                1.0
            )

        elif z == "1.0":
            z_f1, mq, q_fp, g_fn = calculate_F1_with_radius(
                qi2lab_coords,
                qi2lab_gene_ids,
                gt_coords,
                gt_gene_ids,
                1.0
            )

        elif z == "1.5":
            z_f1, mq, q_fp, g_fn = calculate_F1_with_radius(
                qi2lab_coords,
                qi2lab_gene_ids,
                gt_coords,
                gt_gene_ids,
                1.0
            )

        # Build one points array + one categorical status property
        z_parts = []
        z_labels = []

        if mq.size:
            z_parts.append(mq)  # TP coords already
            z_labels.append(np.full(mq.shape[0], "TP", dtype=object))

        if q_fp.size:
            z_parts.append(q_fp)  # FP coords already
            z_labels.append(np.full(q_fp.shape[0], "FP", dtype=object))

        if g_fn.size:
            z_parts.append(g_fn)  # FN coords already (from GT)
            z_labels.append(np.full(g_fn.shape[0], "FN", dtype=object))

        if z_parts:
            z_points = np.vstack(z_parts)
            z_status = np.concatenate(z_labels)
        else:
            z_points = np.empty((0, 3), dtype=float)
            z_status = np.empty((0,), dtype=object)

        status.append(z_status)
        points.append(z_points)
        f1.append(z_f1)

    print(f"F1 results z=0.315: {f1[0]}")
    print(f"F1 results z=1.0: {f1[1]}")
    print(f"F1 results z=1.5: {f1[2]}")

    viewer = napari.Viewer()

    layer6 = viewer.add_points(points[2][:, [0, 2]], name="1.5 RNA XZ", scale=[1,1], size=.75, properties={"status": status[2]}, face_color="status")
    layer6.face_color_cycle = {"TP": "gray", "FP": "cyan", "FN": "orange"}
    layer5 = viewer.add_points(points[1][:, [0, 2]], name="1.0 RNA XZ", scale=[1,1], size=.75, properties={"status": status[1]}, face_color="status")
    layer5.face_color_cycle = {"TP": "gray", "FP": "cyan", "FN": "orange"}
    layer4 = viewer.add_points(points[0][:, [0, 2]], name=".315 RNA XZ", scale=[1,1], size=.75, properties={"status": status[0]}, face_color="status")
    layer4.face_color_cycle = {"TP": "gray", "FP": "cyan", "FN": "orange"}

    viewer.add_image(np.squeeze(np.max(np.swapaxes(images[2],1,2),axis=1)), name="1.5 bit 1 XZ", scale=[image_0315_voxel[0],image_0315_voxel[1]], colormap='gray', blending='additive', contrast_limits=[0,4000])
    viewer.add_image(np.squeeze(np.max(np.swapaxes(images[1],1,2),axis=1)), name="1.0 bit 1 XZ", scale=[image_0315_voxel[0],image_0315_voxel[1]], colormap='gray', blending='additive', contrast_limits=[0,4000])
    viewer.add_image(np.squeeze(np.max(np.swapaxes(images[0],1,2),axis=1)), name="0.315 bit 1 XZ", scale=[image_0315_voxel[0],image_0315_voxel[1]], colormap='gray', blending='additive', contrast_limits=[0,4000])

    layer3 = viewer.add_points(points[2][:, [1, 2]], name="1.5 RNA XY", scale=[1,1], size=.75, properties={"status": status[2]}, face_color="status")
    layer3.face_color_cycle = {"TP": "gray", "FP": "cyan", "FN": "orange"}
    layer2 = viewer.add_points(points[1][:, [1, 2]], name="1.0 RNA XY", scale=[1,1], size=.75, properties={"status": status[1]}, face_color="status")
    layer2.face_color_cycle = {"TP": "gray", "FP": "cyan", "FN": "orange"}
    layer1 = viewer.add_points(points[0][:, [1, 2]], name=".315 RNA XY", scale=[1,1], size=.75, properties={"status": status[0]}, face_color="status")
    layer1.face_color_cycle = {"TP": "gray", "FP": "cyan", "FN": "orange"}

    viewer.add_image(np.max(images[2],axis=1), name="1.5 bit 1 XY", scale=image_0315_voxel[1:], colormap='gray', blending='additive', contrast_limits=[0,4000])
    viewer.add_image(np.max(images[1],axis=1), name="1.0 bit 1 XY", scale=image_0315_voxel[1:], colormap='gray', blending='additive', contrast_limits=[0,4000])
    viewer.add_image(np.max(images[0],axis=1), name="0.315 bit 1 XY", scale=image_0315_voxel[1:], colormap='gray', blending='additive', contrast_limits=[0,4000])

    viewer.scale_bar.unit = 'um'
    viewer.scale_bar.visible = True

    napari.run()

def main():
    app()

if __name__ == "__main__":
    main()