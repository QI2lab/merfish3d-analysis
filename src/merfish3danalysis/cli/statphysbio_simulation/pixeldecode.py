"""
Decode using qi2lab GPU decoder.

Contributions: Shepherd.
"""

from pathlib import Path

import typer

from merfish3danalysis.cli.qi2lab_microscopes.pixeldecode import (
    _nearest_nyquist_multiple,
    _validate_filter_arguments,
)
from merfish3danalysis.PixelDecoder import PixelDecoder
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import resolve_datastore_path

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
    """
    Return the sampling-aware default magnitude threshold for simulations.

    Parameters
    ----------
    datastore : qi2labDataStore
        Datastore supplying microscope type and native ZYX voxel spacing in microns.

    Returns
    -------
    tuple[float, float]
        Accepted magnitude interval selected from the simulation sampling presets.
    """
    if datastore.microscope_type != "2D":
        return SIMULATION_3D_DEFAULT_MAGNITUDE_THRESHOLD

    z_step_um = float(datastore.voxel_size_zyx_um[0])
    nyquist_multiple = z_step_um / SIMULATION_AXIAL_NYQUIST_STEP_UM
    nearest_multiple = _nearest_nyquist_multiple(
        SIMULATION_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST,
        nyquist_multiple,
    )
    lower_threshold = SIMULATION_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST[nearest_multiple]
    return (lower_threshold, 10.0)


@app.command()
def decode_pixels(
    root_path: Path,
    minimum_pixels_per_RNA: int | None = None,
    *,
    lowpass_sigma: tuple[float, float, float] = (3.0, 1.0, 1.0),
    magnitude_threshold: tuple[float, float] | None = None,
    skip_optimization: bool = False,
    duplicate_radius_xy: float | None = None,
    duplicate_radius_z: float | None = None,
    filter_method: str = "blank_fraction",
    target_gross_misid_rate: float = 0.05,
    lr_fdr_target: float = 0.05,
    normalization_iterations: int = 3,
    estimate_chromatic_affines: bool = False,
    decode_run_key: str | None = None,
) -> None:
    """Perform pixel decoding.

    Parameters
    ----------
    root_path : Path
        path to experiment
    minimum_pixels_per_RNA : int, optional
        minimum pixels with same barcode ID required to call a spot.
        Defaults to 7 for 2D simulations and 28 for 3D simulations.
    lowpass_sigma : tuple[float, float, float], default (3.0, 1.0, 1.0)
        Gaussian lowpass sigma in z, y, x before decoding.
    magnitude_threshold : tuple[float,float], optional
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
        downstream transcript filter. Supported values are "blank_fraction" and "lr".
    target_gross_misid_rate : float, default .05
        gross misidentification-rate target for blank-fraction filtering.
    lr_fdr_target : float, default .05
        false discovery rate target for LR filtering.
    normalization_iterations : int, default=3
        Number of iterative normalization rounds to run before final decoding.
    estimate_chromatic_affines : bool, default=False
        If True, estimate chromatic affine transforms during iterative
        normalization. Existing datastore calibration is still used when this
        is False, with identity fallback when no calibration is present.
    decode_run_key : str or None, default None
        Optional run name used to isolate normalization and decoded output.
    """
    # initialize datastore
    datastore_path = resolve_datastore_path(root_path)
    datastore = qi2labDataStore(datastore_path, validate=False)
    merfish_bits = datastore.num_bits

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore,
        merfish_bits=merfish_bits,
        verbose=1,
        decode_run_key=decode_run_key,
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
            n_iterations=normalization_iterations,
            lowpass_sigma=lowpass_sigma,
            magnitude_threshold=magnitude_threshold,
            minimum_pixels=minimum_pixels_per_RNA,
            estimate_chromatic_affines=estimate_chromatic_affines,
        )

    decoder.decode_all_tiles(
        assign_to_cells=False,
        lowpass_sigma=lowpass_sigma,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        duplicate_radius_xy=duplicate_radius_xy,
        duplicate_radius_z=duplicate_radius_z,
        filter_method=filter_method,
        target_gross_misid_rate=target_gross_misid_rate,
        lr_fdr_target=lr_fdr_target,
    )


def main() -> None:
    """Run the simulation pixel decoding CLI."""
    app()


if __name__ == "__main__":
    main()
