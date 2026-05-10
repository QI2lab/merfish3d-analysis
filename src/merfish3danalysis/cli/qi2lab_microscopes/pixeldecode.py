"""
Decode using qi2lab GPU decoder and (re)-segment cells based on decoded RNA.

Shepherd 2025/07 - refactor for multiple GPU support.
Shepherd 2024/12 - refactor
Shepherd 2024/11 - modified script to accept parameters with sensible defaults.
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from pathlib import Path

import typer

from merfish3danalysis.PixelDecoder import PixelDecoder
from merfish3danalysis.qi2labDataStore import qi2labDataStore

app = typer.Typer()
app.pretty_exceptions_enable = False

QI2LAB_3D_DEFAULT_MAGNITUDE_THRESHOLD = (0.9, 10.0)
QI2LAB_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST = {
    3.0: 0.7,
    5.0: 0.2,
}
QI2LAB_AXIAL_NYQUIST_STEP_UM = 0.315


def _default_qi2lab_minimum_pixels(datastore: qi2labDataStore) -> int:
    """Return the default minimum-pixel threshold for qi2lab decoding."""

    return 7 if datastore.microscope_type == "2D" else 28


def _default_qi2lab_magnitude_threshold(
    datastore: qi2labDataStore,
) -> tuple[float, float]:
    """Return the sampling-aware default magnitude threshold for qi2lab decoding."""

    if datastore.microscope_type != "2D":
        return QI2LAB_3D_DEFAULT_MAGNITUDE_THRESHOLD

    z_step_um = float(datastore.voxel_size_zyx_um[0])
    nyquist_multiple = z_step_um / QI2LAB_AXIAL_NYQUIST_STEP_UM
    nearest_multiple = min(
        QI2LAB_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST,
        key=lambda value: abs(value - nyquist_multiple),
    )
    lower_threshold = QI2LAB_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST[nearest_multiple]
    return (lower_threshold, QI2LAB_3D_DEFAULT_MAGNITUDE_THRESHOLD[1])


def _validate_filter_arguments(
    filter_method: str,
    target_gross_misid_rate: float,
    lr_fdr_target: float,
) -> None:
    """Validate that the selected filter uses the matching control parameter."""

    if filter_method in ("blank_fraction", "blank_bit_enrichment"):
        if lr_fdr_target != 0.05:
            raise typer.BadParameter(
                "--lr-fdr-target only applies with --filter-method lr. "
                "Use --target-gross-misid-rate with --filter-method "
                "blank_fraction or blank_bit_enrichment."
            )
        return

    if filter_method == "lr":
        if target_gross_misid_rate != 0.05:
            raise typer.BadParameter(
                "--target-gross-misid-rate only applies with "
                "--filter-method blank_fraction. Use --lr-fdr-target with "
                "--filter-method lr."
            )
        return

    raise typer.BadParameter(
        "filter_method must be one of 'blank_fraction', "
        "'blank_bit_enrichment', or 'lr'."
    )


@app.command()
def decode_pixels(
    root_path: Path,
    num_gpus: int = 1,
    minimum_pixels_per_RNA: int | None = None,
    feature_predictor_threshold: float = 0.5,
    magnitude_threshold: tuple[float, float] | None = None,
    filter_method: str = "blank_fraction",
    target_gross_misid_rate: float = 0.05,
    lr_fdr_target: float = 0.05,
    merfish_bits: int | None = None,
    skip_optimization: bool = False,
    reprocess_existing: bool = False,
    zstride_level: int = 0,
) -> None:
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    num_gpus : int
        number of gpus to use. Default = 1.
    minimum_pixels_per_RNA : int, optional
        minimum pixels with same barcode ID required to call a spot.
        Defaults to 7 for 2D data and 28 for 3D data.
    feature_predictor_threshold : float
        threshold to accept feature_predictor prediction. Default = 0.5
    magnitude_threshold : tuple[float,float], optional
        list of two floats [min, max] magnitude thresholds to accept a decoded
        pixel. Defaults to (0.9, 10.0) for 3D data and a 2D lookup keyed by
        axial sampling relative to the 0.315 um Nyquist reference:
        ~3x Nyquist -> 0.7 and ~5x Nyquist -> 0.2.
    filter_method : str, default "blank_fraction"
        downstream transcript filter. Supported values are "blank_fraction",
        "blank_bit_enrichment", and "lr".
    target_gross_misid_rate : float
        gross misidentification-rate target for blank-fraction filtering. Default = .05
    lr_fdr_target : float
        false discovery rate target for LR filtering. Default = .05
    merfish_bits : int. default = None
        number of bits in codebook. By default uses all bits in codebook.
    skip_optimization: bool, default = False
        skip running iterative optimization.
    reprocess_existing : bool, default = False
        flag to reprocess existing exact-called decoded data. Legacy decoded
        parquet files from the old caller are not supported.
    zstride_level: int, default = 0
        look for a skip z dataset.
    """

    # initialize datastore
    if zstride_level == 0:
        datastore_path = root_path / Path(r"qi2labdatastore")
    else:
        datastore_path = root_path / Path(f"qi2labdatastore_zstride0{zstride_level}")
    datastore = qi2labDataStore(datastore_path, validate=False)
    print(f"Using datastore at {datastore_path}")
    if merfish_bits is None:
        merfish_bits = datastore.num_bits
    if minimum_pixels_per_RNA is None:
        minimum_pixels_per_RNA = _default_qi2lab_minimum_pixels(datastore)
    if magnitude_threshold is None:
        magnitude_threshold = _default_qi2lab_magnitude_threshold(datastore)
    _validate_filter_arguments(
        filter_method=filter_method,
        target_gross_misid_rate=target_gross_misid_rate,
        lr_fdr_target=lr_fdr_target,
    )

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        num_gpus=num_gpus,
        verbose=1,
    )

    if not (reprocess_existing):
        if not skip_optimization:
            # optimize normalization weights through iterative decoding and update
            decoder.optimize_normalization_by_decoding(
                n_random_tiles=20,
                n_iterations=5,
                minimum_pixels=minimum_pixels_per_RNA,
                feature_predictor_threshold=feature_predictor_threshold,
                magnitude_threshold=magnitude_threshold,
            )

        # decode all tiles using iterative normalization weights
        decoder.decode_all_tiles(
            assign_to_cells=True,
            magnitude_threshold=magnitude_threshold,
            minimum_pixels=minimum_pixels_per_RNA,
            feature_predictor_threshold=feature_predictor_threshold,
            filter_method=filter_method,
            target_gross_misid_rate=target_gross_misid_rate,
            lr_fdr_target=lr_fdr_target,
        )
    else:
        decoder._verbose = 2
        decoder.optimize_filtering(
            assign_to_cells=True,
            filter_method=filter_method,
            target_gross_misid_rate=target_gross_misid_rate,
            lr_fdr_target=lr_fdr_target,
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
