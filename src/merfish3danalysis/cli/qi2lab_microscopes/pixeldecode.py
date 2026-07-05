"""
Decode using qi2lab GPU decoder and (re)-segment cells based on decoded RNA.

Shepherd 2025/07 - refactor for multiple GPU support.
Shepherd 2024/12 - refactor
Shepherd 2024/11 - modified script to accept parameters with sensible defaults.
Shepherd 2024/08 - rework script to utilize qi2labdatastore object.
"""

from pathlib import Path
from typing import Literal

import typer

from merfish3danalysis.cli.qi2lab_microscopes._common import qi2lab_datastore_path
from merfish3danalysis.PixelDecoder import PixelDecoder
from merfish3danalysis.qi2labDataStore import qi2labDataStore

app = typer.Typer()
app.pretty_exceptions_enable = False

QI2LAB_3D_DEFAULT_MAGNITUDE_THRESHOLD = (1.5, 10.0)
QI2LAB_2D_DEFAULT_MINIMUM_PIXELS = 7
QI2LAB_3D_DEFAULT_MINIMUM_PIXELS = 16
QI2LAB_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST = {
    3.0: 0.7,
    5.0: 0.2,
}
QI2LAB_2D_DECON_FEATURE_PREDICTOR_THRESHOLD_BY_NYQUIST = {
    3.0: 0.3,
    5.0: 0.2,
}
QI2LAB_AXIAL_NYQUIST_STEP_UM = 0.315
QI2LAB_DEFAULT_FEATURE_PREDICTOR_THRESHOLD = 0.5


def _nearest_nyquist_multiple(
    thresholds_by_multiple: dict[float, float],
    nyquist_multiple: float,
) -> float:
    """
    Return the configured Nyquist multiple nearest to a measured multiple.

    Parameters
    ----------
    thresholds_by_multiple : dict[float, float]
        Threshold table keyed by Nyquist sampling multiple.
    nyquist_multiple : float
        Measured axial step divided by the axial Nyquist step.

    Returns
    -------
    float
        Key from ``thresholds_by_multiple`` nearest to ``nyquist_multiple``.
    """

    best_multiple = next(iter(thresholds_by_multiple))
    best_distance = abs(best_multiple - nyquist_multiple)
    for multiple in thresholds_by_multiple:
        distance = abs(multiple - nyquist_multiple)
        if distance < best_distance:
            best_multiple = multiple
            best_distance = distance
    return best_multiple


def _effective_decode_mode(
    datastore: qi2labDataStore,
    decode_mode: Literal["auto", "2d", "3d"],
) -> Literal["2d", "3d"]:
    """
    Resolve the decode mode used for defaults and connected components.

    Parameters
    ----------
    datastore : qi2labDataStore
        Datastore used to infer microscope type when ``decode_mode`` is "auto".
    decode_mode : {'auto', '2d', '3d'}
        Requested decode mode.

    Returns
    -------
    {'2d', '3d'}
        Effective decode mode.
    """

    if decode_mode == "auto":
        return "2d" if datastore.microscope_type == "2D" else "3d"
    if decode_mode in {"2d", "3d"}:
        return decode_mode
    raise typer.BadParameter("decode_mode must be one of 'auto', '2d', or '3d'.")


def _default_qi2lab_minimum_pixels(
    datastore: qi2labDataStore,
    decode_mode: Literal["auto", "2d", "3d"] = "auto",
) -> int:
    """
    Return the default minimum-pixel threshold for qi2lab decoding.

    Parameters
    ----------
    datastore : qi2labDataStore
        Datastore used to infer microscope type when ``decode_mode`` is "auto".
    decode_mode : {'auto', '2d', '3d'}, default 'auto'
        Decode mode used for default selection.

    Returns
    -------
    int
        Default minimum-pixel threshold.
    """

    if _effective_decode_mode(datastore, decode_mode) == "2d":
        return QI2LAB_2D_DEFAULT_MINIMUM_PIXELS
    return QI2LAB_3D_DEFAULT_MINIMUM_PIXELS


def _default_qi2lab_magnitude_threshold(
    datastore: qi2labDataStore,
    decode_mode: Literal["auto", "2d", "3d"] = "auto",
) -> tuple[float, float]:
    """
    Return the sampling-aware default magnitude threshold for qi2lab decoding.

    Parameters
    ----------
    datastore : qi2labDataStore
        Datastore used to infer microscope type and axial sampling.
    decode_mode : {'auto', '2d', '3d'}, default 'auto'
        Decode mode used for default selection.

    Returns
    -------
    tuple[float, float]
        Default magnitude threshold range.
    """

    if _effective_decode_mode(datastore, decode_mode) != "2d":
        return QI2LAB_3D_DEFAULT_MAGNITUDE_THRESHOLD

    z_step_um = float(datastore.voxel_size_zyx_um[0])
    nyquist_multiple = z_step_um / QI2LAB_AXIAL_NYQUIST_STEP_UM
    nearest_multiple = _nearest_nyquist_multiple(
        QI2LAB_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST,
        nyquist_multiple,
    )
    lower_threshold = QI2LAB_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST[nearest_multiple]
    return (lower_threshold, QI2LAB_3D_DEFAULT_MAGNITUDE_THRESHOLD[1])


def _readouts_are_deconvolved(datastore: qi2labDataStore) -> bool:
    """
    Return whether registered readout data were saved after deconvolution.

    Parameters
    ----------
    datastore : qi2labDataStore
        Datastore containing registered readout metadata.

    Returns
    -------
    bool
        True if the first registered readout records deconvolution metadata.
    """

    tile_ids = datastore.tile_ids
    bit_ids = datastore.bit_ids
    if tile_ids is None or bit_ids is None:
        return False
    tile_ids = list(tile_ids)
    bit_ids = list(bit_ids)
    if not tile_ids or not bit_ids:
        return False

    entity_root = datastore._readouts_root_path / tile_ids[0] / bit_ids[0]
    attributes = datastore._load_entity_attributes(
        entity_root,
        image_names=("decon_data",),
    )
    return bool(attributes.get("deconvolution", False))


def _default_qi2lab_feature_predictor_threshold(
    datastore: qi2labDataStore,
    decode_mode: Literal["auto", "2d", "3d"] = "auto",
) -> float:
    """
    Return the sampling-aware default U-FISH mask threshold.

    Parameters
    ----------
    datastore : qi2labDataStore
        Datastore used to infer microscope type and axial sampling.
    decode_mode : {'auto', '2d', '3d'}, default 'auto'
        Decode mode used for default selection.

    Returns
    -------
    float
        Default feature-predictor threshold.
    """

    if _effective_decode_mode(
        datastore, decode_mode
    ) != "2d" or not _readouts_are_deconvolved(datastore):
        return QI2LAB_DEFAULT_FEATURE_PREDICTOR_THRESHOLD

    z_step_um = float(datastore.voxel_size_zyx_um[0])
    nyquist_multiple = z_step_um / QI2LAB_AXIAL_NYQUIST_STEP_UM
    nearest_multiple = _nearest_nyquist_multiple(
        QI2LAB_2D_DECON_FEATURE_PREDICTOR_THRESHOLD_BY_NYQUIST,
        nyquist_multiple,
    )
    return QI2LAB_2D_DECON_FEATURE_PREDICTOR_THRESHOLD_BY_NYQUIST[nearest_multiple]


def _validate_filter_arguments(
    filter_method: Literal["blank_fraction", "lr"],
    target_gross_misid_rate: float,
    lr_fdr_target: float,
) -> None:
    """
    Validate that the selected filter uses the matching control parameter.

    Parameters
    ----------
    filter_method : {'blank_fraction', 'lr'}
        Transcript filtering method.
    target_gross_misid_rate : float
        Gross misidentification-rate target for blank-fraction filtering.
    lr_fdr_target : float
        False-discovery-rate target for LR filtering.

    Returns
    -------
    None
        This function raises when arguments are inconsistent.
    """

    if filter_method == "blank_fraction":
        if lr_fdr_target != 0.05:
            raise typer.BadParameter(
                "--lr-fdr-target only applies with --filter-method lr. "
                "Use --target-gross-misid-rate with --filter-method "
                "blank_fraction."
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

    raise typer.BadParameter("filter_method must be one of 'blank_fraction' or 'lr'.")


@app.command()
def decode_pixels(
    root_path: Path,
    num_gpus: int = 1,
    minimum_pixels_per_RNA: int | None = None,
    feature_predictor_threshold: float | None = None,
    magnitude_threshold: tuple[float, float] | None = None,
    filter_method: Literal["blank_fraction", "lr"] = "blank_fraction",
    target_gross_misid_rate: float = 0.05,
    lr_fdr_target: float = 0.05,
    merfish_bits: int | None = None,
    skip_optimization: bool = False,
    normalization_method: Literal["iterative", "global", "none"] = "iterative",
    estimate_chromatic_affines: bool = False,
    reprocess_existing: bool = False,
    decode_mode: Literal["auto", "2d", "3d"] = "auto",
) -> None:
    """Perform pixel decoding.

    Parameters
    ----------
    root_path : Path
        Experiment root directory.
    num_gpus : int, default=1
        Number of GPUs to use.
    minimum_pixels_per_RNA : int, optional
        minimum pixels with same barcode ID required to call a spot.
        Defaults to 7 for 2D data and 16 for 3D data.
    feature_predictor_threshold : float, optional
        Legacy option retained for compatibility. Readout images are now
        weighted by the feature-predictor image before lowpass filtering rather
        than thresholded by this value.
    magnitude_threshold : tuple[float, float], optional
        Magnitude threshold range to accept a decoded pixel. Defaults to
        (1.5, 10.0) for 3D data and a 2D lookup keyed by
        axial sampling relative to the 0.315 um Nyquist reference:
        ~3x Nyquist -> 0.7 and ~5x Nyquist -> 0.2.
    filter_method : {"blank_fraction", "lr"}, default "blank_fraction"
        downstream transcript filter. Supported values are "blank_fraction" and "lr".
    target_gross_misid_rate : float, default=0.05
        Gross misidentification-rate target for blank-fraction filtering.
    lr_fdr_target : float, default=0.05
        False discovery rate target for LR filtering.
    merfish_bits : int | None, default=None
        Number of bits in codebook. By default uses all bits in codebook.
    skip_optimization : bool, default=False
        Skip running iterative optimization.
    normalization_method : {"iterative", "global", "none"}, default "iterative"
        normalization source for pixel decoding.
    estimate_chromatic_affines : bool, default=False
        If True, estimate chromatic affine transforms during iterative
        normalization. Existing datastore calibration is still used by default.
    reprocess_existing : bool, default=False
        Reprocess existing exact-called decoded data. Legacy decoded
        parquet files from the old caller are not supported.
    decode_mode : {"auto", "2d", "3d"}, default "auto"
        Decode mode. ``auto`` follows the datastore microscope type; explicit
        values control connected-component extraction and default thresholds.
    """

    # initialize datastore
    datastore_path = qi2lab_datastore_path(root_path)
    datastore = qi2labDataStore(datastore_path, validate=False)
    _effective_decode_mode(datastore, decode_mode)
    print(f"Using datastore at {datastore_path}")
    if merfish_bits is None:
        merfish_bits = datastore.num_bits
    if minimum_pixels_per_RNA is None:
        minimum_pixels_per_RNA = _default_qi2lab_minimum_pixels(
            datastore,
            decode_mode=decode_mode,
        )
    if feature_predictor_threshold is None:
        feature_predictor_threshold = _default_qi2lab_feature_predictor_threshold(
            datastore,
            decode_mode=decode_mode,
        )
    if magnitude_threshold is None:
        magnitude_threshold = _default_qi2lab_magnitude_threshold(
            datastore,
            decode_mode=decode_mode,
        )
    _validate_filter_arguments(
        filter_method=filter_method,
        target_gross_misid_rate=target_gross_misid_rate,
        lr_fdr_target=lr_fdr_target,
    )
    # initialize decoder class
    decoder = PixelDecoder(
        datastore=datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        num_gpus=num_gpus,
        verbose=1,
        decode_mode=decode_mode,
        estimate_chromatic_affines=estimate_chromatic_affines,
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
            normalization_method=normalization_method,
            filter_method=filter_method,
            target_gross_misid_rate=target_gross_misid_rate,
            lr_fdr_target=lr_fdr_target,
        )
    else:
        decoder.optimize_filtering(
            assign_to_cells=True,
            filter_method=filter_method,
            target_gross_misid_rate=target_gross_misid_rate,
            lr_fdr_target=lr_fdr_target,
        )


def main() -> None:
    """Run the Typer app."""
    app()


if __name__ == "__main__":
    main()
