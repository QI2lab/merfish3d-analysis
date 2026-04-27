"""
Decode using qi2lab GPU decoder.

Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2024/12 - create script to run on simulation.
"""

from pathlib import Path

import typer

from merfish3danalysis.PixelDecoder import PixelDecoder
from merfish3danalysis.qi2labDataStore import qi2labDataStore

app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def decode_pixels(
    root_path: Path,
    minimum_pixels_per_RNA: int = 2,
    feature_predictor_threshold: float = 0.2,
    magnitude_threshold: tuple[float, float] = [0.9, 10.0],
    filter_method: str = "blank_fraction",
    target_gross_misid_rate: float = 0.05,
    lr_fdr_target: float = 0.05,
) -> None:
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    minimum_pixels_per_RNA : int, default 2
        minimum pixels with same barcode ID required to call a spot.
    feature_predictor_threshold : float, default 0.2
        threshold to accept feature_predictor prediction.
    magnitude_threshold: tuple[float,float], default (0.9,10.0)
        minimum magnitude across all normalized bits required to accept a spot.
    filter_method : str, default "blank_fraction"
        downstream transcript filter. Supported values are "blank_fraction" and "lr".
    target_gross_misid_rate : float, default .05
        gross misidentification-rate target for blank-fraction filtering.
    lr_fdr_target : float, default .05
        false discovery rate target for LR filtering.
    """

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path, validate=False)
    merfish_bits = datastore.num_bits

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        verbose=1,
    )

    decoder.optimize_normalization_by_decoding(
        n_random_tiles=1,
        n_iterations=3,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        feature_predictor_threshold=feature_predictor_threshold,
    )

    decoder.decode_all_tiles(
        assign_to_cells=False,
        prep_for_baysor=False,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        feature_predictor_threshold=feature_predictor_threshold,
        filter_method=filter_method,
        target_gross_misid_rate=target_gross_misid_rate,
        lr_fdr_target=lr_fdr_target,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
