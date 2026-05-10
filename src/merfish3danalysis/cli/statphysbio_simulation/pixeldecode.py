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

SIMULATION_3D_DEFAULT_MAGNITUDE_THRESHOLD = (0.9, 10.0)
SIMULATION_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST = {
    3.0: 0.7,
    5.0: 0.2,
}
SIMULATION_AXIAL_NYQUIST_STEP_UM = 0.315


def _default_simulation_magnitude_threshold(
    datastore: qi2labDataStore,
) -> tuple[float, float]:
    """Return the sampling-aware default magnitude threshold for simulations."""

    if datastore.microscope_type != "2D":
        return SIMULATION_3D_DEFAULT_MAGNITUDE_THRESHOLD

    z_step_um = float(datastore.voxel_size_zyx_um[0])
    nyquist_multiple = z_step_um / SIMULATION_AXIAL_NYQUIST_STEP_UM
    nearest_multiple = min(
        SIMULATION_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST,
        key=lambda value: abs(value - nyquist_multiple),
    )
    lower_threshold = SIMULATION_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST[nearest_multiple]
    return (lower_threshold, 10.0)


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
    minimum_pixels_per_RNA: int | None = None,
    feature_predictor_threshold: float = 0.5,
    lowpass_sigma: tuple[float, float, float] = (3.0, 1.0, 1.0),
    magnitude_threshold: tuple[float, float] | None = None,
    skip_optimization: bool = False,
    duplicate_radius_xy: float | None = None,
    duplicate_radius_z: float | None = None,
    filter_method: str = "blank_fraction",
    target_gross_misid_rate: float = 0.05,
    lr_fdr_target: float = 0.05,
) -> None:
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    minimum_pixels_per_RNA : int, optional
        minimum pixels with same barcode ID required to call a spot.
        Defaults to 7 for 2D simulations and 28 for 3D simulations.
    feature_predictor_threshold : float, default 0.5
        threshold to accept feature_predictor prediction.
    lowpass_sigma : tuple[float, float, float], default (3.0, 1.0, 1.0)
        Gaussian lowpass sigma in z, y, x before decoding.
    magnitude_threshold: tuple[float,float], optional
        minimum magnitude across all normalized bits required to accept a spot.
        Defaults to (0.9, 10.0) for 3D simulations and a 2D lookup table keyed
        by axial sampling relative to the 0.315 um Nyquist reference:
        ~3x Nyquist -> 0.7 and ~5x Nyquist -> 0.2.
    skip_optimization : bool, default False
        if True, reuse existing normalization vectors and skip iterative decoding.
    duplicate_radius_xy : float, optional
        override XY radius, in microns, for within-tile duplicate collapse.
    duplicate_radius_z : float, optional
        override Z radius, in microns, for within-tile duplicate collapse.
    filter_method : str, default "blank_fraction"
        downstream transcript filter. Supported values are "blank_fraction",
        "blank_bit_enrichment", and "lr".
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
    if minimum_pixels_per_RNA is None:
        minimum_pixels_per_RNA = 7 if datastore.microscope_type == "2D" else 28
    if magnitude_threshold is None:
        magnitude_threshold = _default_simulation_magnitude_threshold(datastore)
    _validate_filter_arguments(
        filter_method=filter_method,
        target_gross_misid_rate=target_gross_misid_rate,
        lr_fdr_target=lr_fdr_target,
    )

    if not skip_optimization:
        decoder.optimize_normalization_by_decoding(
            n_random_tiles=1,
            n_iterations=3,
            lowpass_sigma=lowpass_sigma,
            magnitude_threshold=magnitude_threshold,
            minimum_pixels=minimum_pixels_per_RNA,
            feature_predictor_threshold=feature_predictor_threshold,
        )

    decoder.decode_all_tiles(
        assign_to_cells=False,
        lowpass_sigma=lowpass_sigma,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        feature_predictor_threshold=feature_predictor_threshold,
        duplicate_radius_xy=duplicate_radius_xy,
        duplicate_radius_z=duplicate_radius_z,
        filter_method=filter_method,
        target_gross_misid_rate=target_gross_misid_rate,
        lr_fdr_target=lr_fdr_target,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
