"""Estimate chromatic affine calibration from multi-channel bead images."""

from pathlib import Path

import typer

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.chromatic import (
    parse_csv_floats,
    run_chromatic_calibration,
)

app = typer.Typer()
app.pretty_exceptions_enable = False

BEAD_IMAGE_ARGUMENT = typer.Argument(
    ...,
    help="Multi-channel bead image, usually OME-TIFF.",
)
OUTPUT_OPTION = typer.Option(
    Path("chromatic_calibration.json"),
    "--output",
    "-o",
    help="Output calibration JSON path.",
)
DATASTORE_OPTION = typer.Option(
    None,
    "--datastore",
    help="Optional qi2labdatastore path to write calibration metadata.",
)


@app.command()
def calibrate(
    bead_image: Path = BEAD_IMAGE_ARGUMENT,
    output: Path = OUTPUT_OPTION,
    datastore_path: Path | None = DATASTORE_OPTION,
    channel_axis: str | None = typer.Option(
        None,
        "--axes",
        help="Optional input axis string, for example CZYX or ZCYX.",
    ),
    voxel_size_zyx: str | None = typer.Option(
        None,
        "--voxel-size-zyx",
        help="Optional voxel spacing override in microns, e.g. 0.315,0.108,0.108.",
    ),
    wavelengths_um: str | None = typer.Option(
        None,
        "--wavelengths-um",
        help="Optional channel emission wavelengths in microns, e.g. 0.520,0.580,0.670.",
    ),
    na: float = typer.Option(
        1.35,
        "--na",
        help="Objective numerical aperture for generated PSFs.",
    ),
    ri: float = typer.Option(
        1.51,
        "--ri",
        help="Immersion refractive index for generated PSFs.",
    ),
    psf_nx: int = typer.Option(
        51,
        "--psf-nx",
        help="Generated PSF lateral support in pixels.",
    ),
    gpu_id: int = typer.Option(0, "--gpu-id", help="CUDA device ID to use."),
    crop_yx: int = typer.Option(
        2048,
        "--crop-yx",
        help="Initial lateral RLGC crop size.",
    ),
    ufish_model: str | None = typer.Option(
        None,
        "--ufish-model",
        help="U-FISH model alias or path. Defaults to simfish.",
    ),
    match_radius_um: float = typer.Option(
        1.0,
        "--match-radius-um",
        help="Mutual nearest-neighbor bead match radius in microns.",
    ),
    outlier_threshold_um: float = typer.Option(
        1.0,
        "--outlier-threshold-um",
        help="Residual threshold for affine refitting in microns.",
    ),
    min_intensity_quantile: float = typer.Option(
        0.5,
        "--min-intensity-quantile",
        help="Local deconvolved intensity quantile used to keep bright beads.",
    ),
    max_beads: int | None = typer.Option(
        None,
        "--max-beads",
        help="Optional maximum number of beads per channel.",
    ),
    save_intermediates: bool = typer.Option(
        False,
        "--save-intermediates",
        help="Save deconvolved stack and per-channel bead centroid CSV files.",
    ),
) -> None:
    """
    Learn chromatic affine matrices from beads imaged in all channels.

    The lowest wavelength channel is the reference and receives an identity
    affine. Every other affine maps that channel's physical Z, Y, X coordinates
    onto the reference channel.

    Parameters
    ----------
    bead_image : Path
        Multi-channel bead image path.
    output : Path
        Output calibration JSON path.
    datastore_path : Path or None
        Optional datastore path to update.
    channel_axis : str or None
        Optional input axis string.
    voxel_size_zyx : str or None
        Optional Z, Y, X voxel spacing string in microns.
    wavelengths_um : str or None
        Optional emission wavelength string in microns.
    na : float
        Objective numerical aperture.
    ri : float
        Immersion refractive index.
    psf_nx : int
        Generated PSF lateral support in pixels.
    gpu_id : int
        CUDA device index.
    crop_yx : int
        Initial lateral crop size.
    ufish_model : str or None
        U-FISH model alias or path.
    match_radius_um : float
        Bead match radius in microns.
    outlier_threshold_um : float
        Residual threshold for affine refitting.
    min_intensity_quantile : float
        Minimum bead intensity quantile.
    max_beads : int or None
        Optional maximum beads per channel.
    save_intermediates : bool
        Whether to save intermediate outputs.
    """
    datastore = qi2labDataStore(datastore_path) if datastore_path is not None else None
    calibration = run_chromatic_calibration(
        bead_image,
        output_path=output,
        datastore=datastore,
        channel_axis=channel_axis,
        voxel_size_zyx_um=parse_csv_floats(voxel_size_zyx),
        wavelengths_um=parse_csv_floats(wavelengths_um),
        na=na,
        ri=ri,
        psf_nx=psf_nx,
        gpu_id=gpu_id,
        crop_yx=crop_yx,
        ufish_model=ufish_model,
        match_radius_um=match_radius_um,
        outlier_threshold_um=outlier_threshold_um,
        min_intensity_quantile=min_intensity_quantile,
        max_beads=max_beads,
        save_intermediates=save_intermediates,
    )

    typer.echo(f"Wrote chromatic calibration: {output}")
    if datastore_path is not None:
        typer.echo(f"Wrote datastore calibration metadata: {datastore_path}")
    for channel_name, channel in calibration["channels"].items():
        diagnostics = channel["diagnostics"]
        typer.echo(
            f"{channel_name}: status={channel.get('status', 'ok')} "
            f"matched={diagnostics.get('matched_beads')} "
            f"used={diagnostics.get('used_beads')} "
            f"median_residual_um={diagnostics.get('median_residual_um')}"
        )


def main() -> None:
    """
    Run the CLI.

    Returns
    -------
    None
        CLI command is executed.
    """
    app()


if __name__ == "__main__":
    main()
