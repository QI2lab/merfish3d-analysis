"""
Decode using qi2lab GPU decoder and (re)-segment cells based on decoded RNA.

Shepherd 2025/07 - refactor for multiple GPU support.
Shepherd 2024/12 - refactor
Shepherd 2024/11 - modified script to accept parameters with sensible defaults.
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from pathlib import Path
from typing import Literal

import typer

from merfish3danalysis.PixelDecoder import PixelDecoder
from merfish3danalysis.qi2labDataStore import qi2labDataStore

app = typer.Typer()
app.pretty_exceptions_enable = False

QI2LAB_3D_DEFAULT_MAGNITUDE_THRESHOLD = (1.5, 10.0)
QI2LAB_2D_DEFAULT_MINIMUM_PIXELS = 7
QI2LAB_3D_DEFAULT_MINIMUM_PIXELS = 16
QI2LAB_ZSTRIDE_3D_DEFAULT_MINIMUM_PIXELS = 10
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
        Function argument.
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
    zstride_level: int = 0,
) -> int:
    """
    Return the default minimum-pixel threshold for qi2lab decoding.

    Parameters
    ----------
    datastore : qi2labDataStore
        Function argument.
    decode_mode : {'auto', '2d', '3d'}, default 'auto'
        Decode mode used for default selection.
    zstride_level : int, default 0
        Decode-time z stride.

    Returns
    -------
    int
        Function result.
    """

    if _effective_decode_mode(datastore, decode_mode) == "2d":
        return QI2LAB_2D_DEFAULT_MINIMUM_PIXELS
    if zstride_level > 1:
        return QI2LAB_ZSTRIDE_3D_DEFAULT_MINIMUM_PIXELS
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
        Function argument.
    decode_mode : {'auto', '2d', '3d'}, default 'auto'
        Decode mode used for default selection.

    Returns
    -------
    tuple[float, float]
        Function result.
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
        Function argument.
    decode_mode : {'auto', '2d', '3d'}, default 'auto'
        Decode mode used for default selection.

    Returns
    -------
    bool
        Function result.
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
        Function argument.
    decode_mode : {'auto', '2d', '3d'}, default 'auto'
        Decode mode used for default selection.

    Returns
    -------
    float
        Function result.
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
    filter_method: str,
    target_gross_misid_rate: float,
    lr_fdr_target: float,
) -> None:
    """
    Validate that the selected filter uses the matching control parameter.

    Parameters
    ----------
    filter_method : str
        Function argument.
    target_gross_misid_rate : float
        Function argument.
    lr_fdr_target : float
        Function argument.

    Returns
    -------
    None
        Function result.
    """

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
    feature_predictor_threshold: float | None = None,
    magnitude_threshold: tuple[float, float] | None = None,
    filter_method: str = "blank_fraction",
    target_gross_misid_rate: float = 0.05,
    lr_fdr_target: float = 0.05,
    merfish_bits: int | None = None,
    skip_optimization: bool = False,
    normalization_method: str = "iterative",
    reprocess_existing: bool = False,
    zstride_level: int = 0,
    decode_mode: Literal["auto", "2d", "3d"] = "auto",
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
        Defaults to 7 for 2D data, 16 for 3D data, and 10 for strided 3D data.
    feature_predictor_threshold : float, optional
        Legacy option retained for compatibility. Readout images are now
        weighted by the feature-predictor image before lowpass filtering rather
        than thresholded by this value.
    magnitude_threshold : tuple[float,float], optional
        list of two floats [min, max] magnitude thresholds to accept a decoded
        pixel. Defaults to (1.5, 10.0) for 3D data and a 2D lookup keyed by
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
    normalization_method : {"iterative", "global", "none"}, default "iterative"
        normalization source for pixel decoding.
    reprocess_existing : bool, default = False
        flag to reprocess existing exact-called decoded data. Legacy decoded
        parquet files from the old caller are not supported.
    zstride_level: int, default = 0
        Decode-time z stride. Values 0 and 1 keep all planes; values >= 2
        decode planes 0, N, 2N...
    decode_mode : {"auto", "2d", "3d"}, default "auto"
        Decode mode. ``auto`` follows the datastore microscope type; explicit
        values control connected-component extraction and default thresholds.
    """

    # initialize datastore
    if zstride_level < 0:
        raise typer.BadParameter("zstride_level must be greater than or equal to 0.")
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path, validate=False)
    _effective_decode_mode(datastore, decode_mode)
    print(f"Using datastore at {datastore_path}")
    if merfish_bits is None:
        merfish_bits = datastore.num_bits
    if minimum_pixels_per_RNA is None:
        minimum_pixels_per_RNA = _default_qi2lab_minimum_pixels(
            datastore,
            decode_mode=decode_mode,
            zstride_level=zstride_level,
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
    if normalization_method not in {"iterative", "global", "none"}:
        raise typer.BadParameter(
            "normalization_method must be one of 'iterative', 'global', or 'none'."
        )

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        num_gpus=num_gpus,
        verbose=1,
        zstride_level=zstride_level,
        decode_mode=decode_mode,
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
        decoder._verbose = 2
        decoder.optimize_filtering(
            assign_to_cells=True,
            filter_method=filter_method,
            target_gross_misid_rate=target_gross_misid_rate,
            lr_fdr_target=lr_fdr_target,
        )


def main() -> None:
    """
    Main.

    Returns
    -------
    None
        Function result.
    """
    app()


if __name__ == "__main__":
    main()
