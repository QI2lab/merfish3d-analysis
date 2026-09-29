"""Export corrected tile images and annotated local decoded spots."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import numpy as np
import pandas as pd
import tifffile
import typer

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import resolve_datastore_path
from merfish3danalysis.utils.decode_warping import (
    warp_bit_image_to_reference,
    warp_image_to_reference_frame,
)

app = typer.Typer()
app.pretty_exceptions_enable = False
SPOT_KEYS = ["tile_idx", "gene_id", "barcode_id", "z", "y", "x"]


def annotate_local_spots(
    local: pd.DataFrame, global_spots: pd.DataFrame, tile_index: int
) -> pd.DataFrame:
    """Match final global rows to local rows without changing local coordinates.

    Parameters
    ----------
    local : pandas.DataFrame
        All local decoded spots for the selected tile.
    global_spots : pandas.DataFrame
        Final filtered spots for all tiles from the same decoding output.
    tile_index : int
        Selected tile's zero-based index in the datastore.
    """
    for name, frame in (("Local", local), ("Global", global_spots)):
        missing = set(SPOT_KEYS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} spots are missing match columns: {sorted(missing)}"
            )
        if not frame.columns.is_unique:
            raise ValueError(f"{name} spots have duplicate column names.")
    selected = global_spots.loc[global_spots["tile_idx"] == tile_index].copy()
    if not local["tile_idx"].eq(tile_index).all():
        raise ValueError("Local spots contain rows from a different tile.")
    for name, frame in (("Local", local), ("Global", selected)):
        if frame[SPOT_KEYS].isna().any().any():
            raise ValueError(f"{name} spots contain missing match keys.")
        if frame.duplicated(SPOT_KEYS).any():
            raise ValueError(f"{name} spots contain ambiguous duplicate match keys.")
    local_keys = pd.MultiIndex.from_frame(local[SPOT_KEYS])
    global_keys = pd.MultiIndex.from_frame(selected[SPOT_KEYS])
    if not global_keys.isin(local_keys).all():
        raise ValueError(
            "Some global spots have no exact local match; check decoding outputs."
        )

    # Reindex instead of merging to preserve local order, index, columns and dtypes.
    result = local.copy()
    flag = "passes_global_filter"
    if flag in result.columns:
        raise ValueError(f"Local spots already contain reserved column {flag!r}.")
    result[flag] = local_keys.isin(global_keys)
    selected.index = global_keys
    matched = selected.reindex(local_keys)
    for column in selected.columns:
        if column in SPOT_KEYS:
            continue
        target = f"global_file_{column}" if column in local.columns else column
        if target in result.columns:
            raise ValueError(
                f"Global spot column conflicts with export column {target!r}."
            )
        result[target] = matched[column].to_numpy()
    return result


def _corrected_image(
    datastore: qi2labDataStore, tile: str, image_id: str, *, fiducial: bool
) -> np.ndarray:
    """Load only corrected data and retain singleton spatial dimensions.

    Parameters
    ----------
    datastore : qi2labDataStore
        Source datastore.
    tile : str
        Tile identifier.
    image_id : str
        Round or bit identifier.
    fiducial : bool
        Whether the identifier denotes a fiducial round.
    """
    selector = {"round" if fiducial else "bit": image_id}
    image = datastore.load_local_corrected_image(
        tile=tile, return_future=False, **selector
    )
    if image is None:
        raise FileNotFoundError(f"Missing corrected image: {tile}/{image_id}.")
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    if image.ndim != 3 or 0 in image.shape:
        raise ValueError(
            f"Expected nonempty ZYX image for {image_id}, got {image.shape}."
        )
    return image


def _write_tiff(path: Path, image: np.ndarray, spacing: np.ndarray) -> None:
    """Write a ZYX OME-TIFF with physical pixel sizes in microns.

    Parameters
    ----------
    path : Path
        Output TIFF path.
    image : numpy.ndarray
        ZYX image, retaining native dtype or registered float32 values.
    spacing : numpy.ndarray
        ZYX voxel sizes in microns.
    """
    metadata = {"axes": "ZYX"}
    for axis, size in zip("ZYX", spacing, strict=True):
        metadata[f"PhysicalSize{axis}"] = float(size)
        metadata[f"PhysicalSize{axis}Unit"] = "µm"
    tifffile.imwrite(
        path,
        image,
        ome=True,
        metadata=metadata,
        photometric="minisblack",
        bigtiff=image.nbytes >= 2**32 - 2**25,
    )


@app.command()
def export_tile(
    root_path: Path,
    tile_id: Annotated[str, typer.Option(help="Datastore tile ID, e.g. tile0000.")],
    output_dir: Annotated[
        Path | None, typer.Option(help="Tile export directory, outside the datastore.")
    ] = None,
    gpu_id: Annotated[int, typer.Option(min=0)] = 0,
    overwrite: bool = False,
    verbose: Annotated[int, typer.Option(min=0)] = 1,
) -> None:
    """Export a tile's corrected images and local spots annotated by global membership.

    Parameters
    ----------
    root_path : Path
        Experiment root or existing datastore directory.
    tile_id : str
        Tile identifier from the datastore.
    output_dir : Path or None
        Destination, defaulting to a sibling tile_exports/<tile_id> directory.
    gpu_id : int
        GPU used by the existing registration interpolation helpers.
    overwrite : bool
        Replace existing export files.
    verbose : int
        Progress verbosity; zero suppresses routine output.
    """
    datastore_path = resolve_datastore_path(root_path)
    datastore = qi2labDataStore(datastore_path, validate=False)
    tile_ids = list(datastore.tile_ids)
    if tile_id not in tile_ids:
        raise typer.BadParameter(f"Unknown tile ID: {tile_id}", param_hint="--tile-id")
    destination = (
        (
            output_dir
            if output_dir is not None
            else datastore_path.parent / "tile_exports" / tile_id
        )
        .expanduser()
        .resolve()
    )
    if destination.is_relative_to(datastore_path):
        raise typer.BadParameter(
            "Output must be outside the datastore.", param_hint="--output-dir"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Export directory exists: {destination}. Use --overwrite."
        )
    for name in ("native", "registered"):
        if (destination / name).is_symlink():
            raise ValueError(
                f"Export image directory cannot be a symlink: {destination / name}"
            )
    rounds, bits = list(datastore.round_ids), list(datastore.bit_ids)
    if not rounds:
        raise ValueError("Datastore has no fiducial rounds.")
    spacing = np.asarray(datastore.voxel_size_zyx_um, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Expected three positive finite ZYX voxel sizes.")
    local = datastore.load_local_decoded_spots(tile=tile_id)
    global_spots = datastore.load_global_filtered_decoded_spots()
    if local is None or global_spots is None:
        raise FileNotFoundError(
            "Both local and global decoded spot tables are required."
        )
    spots = annotate_local_spots(local, global_spots, tile_ids.index(tile_id))
    del local, global_spots
    reference = _corrected_image(datastore, tile_id, rounds[0], fiducial=True)
    reference_shape = reference.shape
    del reference

    # Finish all reads, warps and writes before publishing any output files.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{tile_id}_export_", dir=destination.parent
    ) as temp:
        staging = Path(temp)
        for name in ("native", "registered"):
            (staging / name).mkdir()
        for fiducial, identifiers in ((True, rounds), (False, bits)):
            for image_id in identifiers:
                if verbose:
                    typer.echo(f"Exporting {tile_id}/{image_id}")
                image = _corrected_image(
                    datastore, tile_id, image_id, fiducial=fiducial
                )
                filename = f"fiducial_{image_id}.tif" if fiducial else f"{image_id}.tif"
                _write_tiff(staging / "native" / filename, image, spacing)
                # Interpolate in floating point so uint16 inputs do not round
                # fractional intensities inside the GPU warp implementation.
                image = image.astype(np.float32, copy=False)
                if fiducial:
                    transform = np.eye(4, dtype=np.float32)
                    flow = None
                    if image_id != rounds[0]:
                        transform = datastore.load_local_round_transform_zyx_um(
                            tile=tile_id, round=image_id
                        )
                        if transform is None:
                            raise ValueError(
                                f"Missing local round transform: {tile_id}/{image_id}."
                            )
                        flow = datastore.load_local_sofima_flow_field(
                            tile=tile_id, round=image_id, return_future=False
                        )
                    registered = warp_image_to_reference_frame(
                        image,
                        transform_zyx_um=transform,
                        spacing_zyx_um=spacing,
                        loaded_flow_field=flow,
                        gpu_id=gpu_id,
                        reference_shape=reference_shape,
                    )
                else:
                    wavelengths = datastore.load_local_wavelengths_um(
                        tile=tile_id, bit=image_id
                    )
                    if wavelengths is None:
                        raise ValueError(
                            f"Missing wavelength metadata: {tile_id}/{image_id}."
                        )
                    registered = warp_bit_image_to_reference(
                        image,
                        datastore=datastore,
                        tile=tile_id,
                        bit_id=image_id,
                        emission_wavelength_um=float(wavelengths[1]),
                        gpu_id=gpu_id,
                        reference_shape=reference_shape,
                    )
                if registered.shape != reference_shape:
                    raise ValueError(
                        f"Registered image {image_id} does not match round-1 shape."
                    )
                _write_tiff(
                    staging / "registered" / filename,
                    registered.astype(np.float32, copy=False),
                    spacing,
                )
                del image, registered
        spots.to_csv(staging / "decoded_spots.csv.gz", index=False, compression="gzip")
        if not destination.exists():
            staging.replace(destination)
        else:
            for name in ("native", "registered"):
                (destination / name).mkdir(exist_ok=True)
                for source in (staging / name).iterdir():
                    source.replace(destination / name / source.name)
            (staging / "decoded_spots.csv.gz").replace(
                destination / "decoded_spots.csv.gz"
            )
    if verbose:
        typer.echo(
            f"Exported {len(spots)} local spots ({int(spots['passes_global_filter'].sum())} retained) to {destination}"
        )


def main() -> None:
    """Run the tile export CLI."""
    app()


if __name__ == "__main__":
    main()
