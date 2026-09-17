"""Simulation comparison figures preserve projections and physical point locations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from tifffile import imwrite

from merfish3danalysis.cli.statphysbio_simulation.build_figure import build_figure


@pytest.mark.integration
def test_simulation_figure_projects_each_spacing_on_the_common_physical_grid(
    tmp_path, monkeypatch, capsys
):
    images = []
    for label, planes, step in [("0.315", 10, 0.315), ("1.0", 3, 1.0), ("1.5", 2, 1.5)]:
        root = tmp_path / "simulation" / label
        acquisition = root / "sim_acquisition"
        image_dir = acquisition / "data_r0001_tile0000_1"
        image_dir.mkdir(parents=True)
        image = np.arange(planes * 4 * 6, dtype=np.uint16).reshape(1, planes, 4, 6)
        images.append(image)
        imwrite(image_dir / "data_r0001_tile0000.tif", image, photometric="minisblack")
        pd.DataFrame(
            {"axial_step_size [micron]": [step], "pixel_size [micron]": [0.1]}
        ).to_csv(root / "scan_metadata.csv", index=False)
        pd.DataFrame({"Z": [0.0], "X": [0.0], "Y": [0.0], "Gene_label": [1]}).to_csv(
            root / "GT_spots.csv", index=False
        )
        pd.DataFrame({"gene_id": ["A"]}).to_csv(root / "codebook.csv", index=False)
        decoded_dir = (
            acquisition / "qi2labdatastore/all_tiles_filtered_decoded_features"
        )
        decoded_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "global_z": [0.0],
                "global_y": [0.15],
                "global_x": [0.25],
                "gene_id": ["A"],
            }
        ).to_parquet(decoded_dir / "decoded_features.parquet")
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        build_figure(tmp_path, "simulation")
        figure = plt.gcf()
        axes = np.asarray(figure.axes).reshape(2, 3)
        for column, image in enumerate(images):
            np.testing.assert_array_equal(
                axes[0, column].images[0].get_array(), image[0].max(axis=0)
            )
            np.testing.assert_allclose(
                axes[0, column].images[0].get_extent(), [-0.05, 0.55, 0.35, -0.05]
            )
            np.testing.assert_allclose(
                axes[0, column].collections[0].get_offsets(), [[0.25, 0.15]]
            )
            np.testing.assert_allclose(
                axes[1, column].collections[0].get_offsets(), [[0.25, 0.0]]
            )
        # The 1 um planes occupy floor([0, 1, 2] / 0.315) on the reference grid.
        expected = np.zeros((10, 6), dtype=np.uint16)
        expected[[0, 3, 6]] = images[1][0].max(axis=1)
        np.testing.assert_array_equal(axes[1, 1].images[0].get_array(), expected)
        assert capsys.readouterr().out.count("'F1 Score': 1.0") == 3
    finally:
        plt.close("all")
