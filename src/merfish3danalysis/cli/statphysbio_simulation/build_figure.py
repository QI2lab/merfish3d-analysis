"""Build simulation benchmark figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from tifffile import imread

from merfish3danalysis.utils.metrics import calculate_F1_with_radius
from merfish3danalysis.utils.spacing import round_pixel_size_um

app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def build_figure(root_path: Path, sim_path: Path = "example_16bit_cells") -> None:
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
        temp = imread(
            top_path
            / Path(str(z))
            / Path("sim_acquisition")
            / Path("data_r0001_tile0000_1")
            / Path("data_r0001_tile0000.tif")
        )
        metadata = pd.read_csv(top_path / Path(str(z)) / Path("scan_metadata.csv"))
        print(round_pixel_size_um(metadata["axial_step_size [micron]"]))
        print(round_pixel_size_um(metadata["pixel_size [micron]"]))
        if z == "0.315":
            image_0315_voxel = np.array(
                [
                    round_pixel_size_um(metadata["axial_step_size [micron]"]),
                    round_pixel_size_um(metadata["pixel_size [micron]"]),
                    round_pixel_size_um(metadata["pixel_size [micron]"]),
                ]
            )
            images.append(temp)
        else:
            temp_filled = np.zeros_like(images[0])
            if z == "1.0":
                image_10_voxel = np.array(
                    [
                        round_pixel_size_um(metadata["axial_step_size [micron]"]),
                        round_pixel_size_um(metadata["pixel_size [micron]"]),
                        round_pixel_size_um(metadata["pixel_size [micron]"]),
                    ]
                )
                for z_idx in range(temp.shape[1]):
                    temp_filled[
                        :,
                        int(np.floor(z_idx * image_10_voxel[0] / image_0315_voxel[0])),
                        :,
                        :,
                    ] = temp[:, z_idx, :, :]
            elif z == "1.5":
                image_15_voxel = np.array(
                    [
                        round_pixel_size_um(metadata["axial_step_size [micron]"]),
                        round_pixel_size_um(metadata["pixel_size [micron]"]),
                        round_pixel_size_um(metadata["pixel_size [micron]"]),
                    ]
                )
                for z_idx in range(temp.shape[1]):
                    temp_filled[
                        :,
                        int(np.floor(z_idx * image_15_voxel[0] / image_0315_voxel[0])),
                        :,
                        :,
                    ] = temp[:, z_idx, :, :]
            images.append(temp_filled)

        gt_spots = pd.read_csv(top_path / Path(str(z)) / Path("GT_spots.csv"))
        codebook = pd.read_csv(top_path / Path(str(z)) / Path("codebook.csv"))
        codebook_genes = codebook["gene_id"].to_numpy()

        decoded_spots = pd.read_parquet(
            top_path
            / Path(str(z))
            / Path("sim_acquisition")
            / Path("qi2labdatastore")
            / Path("all_tiles_filtered_decoded_features")
            / Path("decoded_features.parquet")
        )
        qi2lab_coords = decoded_spots[["global_z", "global_y", "global_x"]].to_numpy()
        qi2lab_gene_ids = decoded_spots["gene_id"].to_numpy()
        gt_coords = gt_spots[["Z", "X", "Y"]].to_numpy()
        gt_gene_ids = codebook_genes[(gt_spots["Gene_label"].to_numpy(dtype=int) - 1)]
        gt_coords_offset = [
            0,
            (1.0 * images[0].shape[-2] / 2) * image_0315_voxel[1]
            - image_0315_voxel[1] / 2,
            (1.0 * images[0].shape[-1] / 2) * image_0315_voxel[2]
            - image_0315_voxel[2] / 2,
        ]

        gt_coords = gt_coords + gt_coords_offset

        if z == "0.315":
            z_f1, mq, q_fp, g_fn = calculate_F1_with_radius(
                qi2lab_coords, qi2lab_gene_ids, gt_coords, gt_gene_ids, 1.0
            )

        elif z == "1.0":
            z_f1, mq, q_fp, g_fn = calculate_F1_with_radius(
                qi2lab_coords, qi2lab_gene_ids, gt_coords, gt_gene_ids, 1.0
            )

        elif z == "1.5":
            z_f1, mq, q_fp, g_fn = calculate_F1_with_radius(
                qi2lab_coords, qi2lab_gene_ids, gt_coords, gt_gene_ids, 1.0
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

    _figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for column, z_spacing in enumerate(z_spacings):
        image = images[column][0]
        for row, (projection, coordinate_axes, spacing, vertical_label) in enumerate(
            [
                (image.max(axis=0), (1, 2), image_0315_voxel[1:], "Y"),
                (image.max(axis=1), (0, 2), image_0315_voxel[[0, 2]], "Z"),
            ]
        ):
            axis = axes[row, column]
            height, width = projection.shape
            axis.imshow(
                projection,
                cmap="gray",
                vmin=0,
                vmax=4000,
                extent=(
                    -spacing[1] / 2,
                    (width - 0.5) * spacing[1],
                    (height - 0.5) * spacing[0],
                    -spacing[0] / 2,
                ),
            )
            for label, color in [("TP", "gray"), ("FP", "cyan"), ("FN", "orange")]:
                selected = points[column][status[column] == label]
                axis.scatter(
                    selected[:, coordinate_axes[1]],
                    selected[:, coordinate_axes[0]],
                    s=2,
                    c=color,
                    label=label,
                )
            axis.set_title(f"Z step {z_spacing} um")
            axis.set_xlabel("X (um)")
            axis.set_ylabel(f"{vertical_label} (um)")
            axis.legend()
    plt.show()


def main() -> None:
    """Run the figure-building CLI."""
    app()


if __name__ == "__main__":
    main()
