"""
Decode using qi2lab GPU decoder and (re)-segment cells based on decoded RNA.

Shepherd 2025/07 - refactor for multiple GPU support.
Shepherd 2024/12 - refactor
Shepherd 2024/11 - modified script to accept parameters with sensible defaults.
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from pathlib import Path

import pandas as pd
import typer

from merfish3danalysis.PixelDecoder import PixelDecoder
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import resolve_datastore_path
from merfish3danalysis.utils.metrics import calculate_F1_with_radius

app = typer.Typer(pretty_exceptions_enable=False)


def decode_pixels(
    root_path: Path,
    minimum_pixels_per_RNA: int = 7,
    feature_predictor_threshold: float = 0.5,
    magnitude_threshold: tuple[float, float] = (0.2, 10.0),
) -> None:
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    minimum_pixels_per_RNA : int
        minimum pixels with same barcode ID required to call a spot. Default = 7.
    feature_predictor_threshold : float
        threshold to accept feature_predictor prediction. Default = 0.5
    magnitude_threshold: tuple[float,float], default = (0.2, 10.0)
        lower and upper magnitude threshold to accept a spot.
    """
    # initialize datastore
    datastore_path = resolve_datastore_path(root_path)
    datastore = qi2labDataStore(datastore_path, validate=False)
    merfish_bits = 22

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore,
        decode_mode="2d",  # Required for the 1.5 micron axial spacing.
        use_mask=False,
        merfish_bits=merfish_bits,
        num_gpus=1,
        verbose=1,
    )

    tile_idx = 0

    _image, _scaled, mag, distance, decoded = decoder.decode_one_tile(
        tile_idx=tile_idx,
        display_results=False,
        return_results=True,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        use_normalization=True,
        feature_predictor_threshold=feature_predictor_threshold,
        lowpass_sigma=(3, 1, 1),
    )
    file_name = root_path / (
        "mop/mouse_sample1_raw/zhuang_decoded_codewords/spots_mouse1sample1.csv"
    )
    df_RNA = pd.read_csv(file_name)

    x_min = 3804.87
    x_max = x_min + 2048 * 0.1085
    y_min = 1366.11
    y_max = y_min + 2048 * 0.1085

    mask = df_RNA["global_x"].between(x_min + 10, x_max) & df_RNA["global_y"].between(
        y_min + 10, y_max
    )
    df_RNA_filtered = df_RNA[mask].copy()

    df_RNA_filtered = df_RNA_filtered[
        ~df_RNA_filtered["target_molecule_name"].str.startswith("Blank-")
    ]
    merlin_coords = df_RNA_filtered[["global_z", "global_x", "global_y"]].to_numpy(
        copy=True
    )
    merlin_gene_ids = df_RNA_filtered["target_molecule_name"].to_numpy()
    merlin_coords[:, 1] = merlin_coords[:, 1] - x_min
    merlin_coords[:, 2] = merlin_coords[:, 2] - y_min

    merlin_coords[:, 1] = merlin_coords[:, 1] / 0.1085
    merlin_coords[:, 2] = merlin_coords[:, 2] / 0.1085
    merlin_coords[:, 0] = merlin_coords[:, 0] / 1.5

    fiducial_image = datastore.load_local_fiducial_image(
        tile=tile_idx, round=0, return_future=False
    )

    decoded_spots = decoder._df_barcodes
    qi2lab_coords = decoded_spots[["tile_z", "tile_y", "tile_x"]].to_numpy()
    qi2lab_gene_ids = decoded_spots["gene_id"].to_numpy()

    results, _, _, _ = calculate_F1_with_radius(
        qi2lab_coords, qi2lab_gene_ids, merlin_coords, merlin_gene_ids, 15.0
    )

    print(f"F1 score: {results['F1 Score']}")

    from merfish3danalysis.viewer.diagnostics import show_diagnostic_images

    show_diagnostic_images(
        {
            "decoded": decoded,
            "fiducial": fiducial_image,
            "magnitude": mag,
            "distance": distance,
        },
        [1, 1, 1],
        points=(
            ("MERlin", merlin_coords, "cyan", "o"),
            ("qi2lab", qi2lab_coords, "orange", "s"),
        ),
        units="px",
    )


@app.command()
def main(root_path: Path) -> None:
    """Decode and compare one Zhuang tile with MERLIN."""
    root_path = root_path.expanduser().resolve()
    decode_pixels(root_path=root_path)


if __name__ == "__main__":
    app()
