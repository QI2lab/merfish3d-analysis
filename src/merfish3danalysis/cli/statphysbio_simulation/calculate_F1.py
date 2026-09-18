"""
Calculate F1-score using known ground truth.

Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2024/12 - create script to run on simulation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import resolve_datastore_path
from merfish3danalysis.utils.metrics import calculate_F1_with_radius

app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def calculate_F1(root_path: Path, search_radius: float = 1.0) -> dict:
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
    datastore_path = resolve_datastore_path(root_path / "sim_acquisition")
    datastore = qi2labDataStore(datastore_path)
    gene_ids, _ = datastore.load_codebook_parsed()
    decoded_spots = datastore.load_global_filtered_decoded_spots()
    gt_path = root_path / Path("GT_spots.csv")
    gt_spots = pd.read_csv(gt_path)
    gene_ids = np.array(gene_ids)
    image = datastore.load_local_corrected_image(tile=0, round=0, return_future=False)

    # Decoded coordinates are in world micrometers. The simulation's centered
    # ground truth must undergo the same stage and affine mapping before matching.
    affine, origin, spacing = datastore.load_global_coord_xforms_um(tile=0)
    _, camera = datastore.load_local_stage_position_zyx_um(tile=0, round=0)
    local_to_world = np.asarray(affine) @ np.asarray(camera)
    qi2lab_coords = decoded_spots[["global_z", "global_y", "global_x"]].to_numpy()
    local_coords = (
        np.column_stack((qi2lab_coords, np.ones(len(qi2lab_coords))))
        @ np.linalg.inv(local_to_world).T
    )[:, :3] - origin
    # Compare voxel centers; the half-plane shift follows the transformed Z axis.
    qi2lab_coords = qi2lab_coords + local_to_world[:3, 0] * spacing[0] / 2
    qi2lab_gene_ids = decoded_spots["gene_id"].to_numpy()

    # Extract coordinates and gene_ids from ground truth
    gt_coords = gt_spots[
        ["Z", "X", "Y"]
    ].to_numpy()  # note the transpose, simulation GT is swapped X & Y

    # re-center the ground truth to start at (0,0) in (y,x)
    gt_coords_offset = [
        0,
        (1.0 * image[0].shape[-2] / 2) * datastore.voxel_size_zyx_um[1]
        - datastore.voxel_size_zyx_um[1] / 2,
        (1.0 * image[0].shape[-1] / 2) * datastore.voxel_size_zyx_um[2]
        - datastore.voxel_size_zyx_um[2] / 2,
    ]

    gt_coords = gt_coords + gt_coords_offset
    gt_gene_ids = gene_ids[(gt_spots["Gene_label"].to_numpy(dtype=int) - 1)]

    image_shape_zyx = np.squeeze(image).shape[-3:]
    z_min_um = datastore.voxel_size_zyx_um[0] / 2
    z_max_um = (image_shape_zyx[0] - 0.5) * datastore.voxel_size_zyx_um[0]
    decoded_z_centers = local_coords[:, 0] + spacing[0] / 2
    qi2lab_keep = (decoded_z_centers >= z_min_um) & (decoded_z_centers <= z_max_um)
    gt_keep = (gt_coords[:, 0] >= z_min_um) & (gt_coords[:, 0] <= z_max_um)
    gt_coords = (gt_coords + origin) @ local_to_world[:3, :3].T + local_to_world[:3, 3]

    results, _, _, _ = calculate_F1_with_radius(
        qi2lab_coords[qi2lab_keep],
        qi2lab_gene_ids[qi2lab_keep],
        gt_coords[gt_keep],
        gt_gene_ids[gt_keep],
        search_radius,
    )

    print("F1 Score Results:")
    print(results)
    return results


def main() -> None:
    """Run the F1 calculation CLI."""
    app()


if __name__ == "__main__":
    main()
