"""Fuse the first-round fiducial and all readout bits into one OME-Zarr."""

import gc
import json
import multiprocessing as mp
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
import xarray as xr
from multiview_stitcher import misc_utils, msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from tqdm import tqdm

from merfish3danalysis.DataRegistration import (
    _direct_zarr_fusion_kwargs,
    _fuse_views_to_scene,
    _read_fiducial_sim,
)
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import resolve_datastore_path
from merfish3danalysis.utils.decode_warping import load_bit_round_transform_zyx_um
from merfish3danalysis.utils.spacing import round_pixel_size_um, round_spacing_um

app = typer.Typer()
app.pretty_exceptions_enable = False


def export_ome_tiffs(
    datastore: qi2labDataStore,
    output_directory: Path,
    channel_ids: list[str],
) -> None:
    """Export each full-resolution fused channel as a tiled OME-TIFF.

    Parameters
    ----------
    datastore : qi2labDataStore
        Datastore containing the multichannel fused image.
    output_directory : pathlib.Path
        Directory in which to write the channel OME-TIFF files.
    channel_ids : list[str]
        Ordered channel names matching the OME-Zarr channel axis.

    Returns
    -------
    None
        One tiled OME-TIFF is written for each channel.
    """
    from tifffile import TiffWriter

    fused_image = datastore.load_global_fused_image("all_channels", return_future=None)
    if fused_image is None:
        raise RuntimeError("The multichannel fused image was not published.")
    array, fused_metadata = fused_image
    dims = tuple(array.domain.labels)
    axes = {dim: dims.index(dim) for dim in dims}
    shape_zyx = tuple(int(array.shape[axes[dim]]) for dim in "zyx")
    dtype = np.dtype(array.dtype.numpy_dtype)
    origin = tuple(map(float, fused_metadata["origin_zyx_um"]))
    spacing = tuple(round_spacing_um(fused_metadata["spacing_zyx_um"]))
    stage_transform_key = "stage_metadata"
    global_transform_key = "global_registered"

    def tiles(channel_index: int) -> Iterator[np.ndarray]:
        """Yield padded TIFF tiles for one channel in Z, Y, X page order.

        Parameters
        ----------
        channel_index : int
            Index of the channel to read from the fused OME-Zarr.

        Yields
        ------
        numpy.ndarray
            One 1024-by-1024 image tile. Edge tiles are zero-padded.
        """
        tile_y = tile_x = 1024
        for z_index in range(shape_zyx[0]):
            for y_start in range(0, shape_zyx[1], tile_y):
                for x_start in range(0, shape_zyx[2], tile_x):
                    y_stop = min(y_start + tile_y, shape_zyx[1])
                    x_stop = min(x_start + tile_x, shape_zyx[2])
                    selection: list[int | slice] = [0] * array.ndim
                    selection[axes["c"]] = channel_index
                    selection[axes["z"]] = z_index
                    selection[axes["y"]] = slice(y_start, y_stop)
                    selection[axes["x"]] = slice(x_start, x_stop)
                    source = array[tuple(selection)].read().result()
                    tile = np.zeros((tile_y, tile_x), dtype=dtype)
                    tile[: source.shape[0], : source.shape[1]] = source
                    yield tile

    for channel_index, name in enumerate(tqdm(channel_ids, desc="ome-tiff")):
        chain = [stage_transform_key]
        if channel_index:
            chain.append("local_round_transform_zyx_um")
        chain.append(global_transform_key)
        with TiffWriter(
            output_directory / f"{name}.ome.tiff", bigtiff=True, ome=True
        ) as tif:
            tif.write(
                data=tiles(channel_index),
                shape=shape_zyx,
                dtype=dtype,
                tile=(1024, 1024),
                resolution=(1e4 / spacing[2], 1e4 / spacing[1]),
                compression="zlib",
                compressionargs={"level": 8},
                predictor=True,
                photometric="minisblack",
                resolutionunit="CENTIMETER",
                metadata={
                    "axes": "ZYX",
                    "SignificantBits": int(dtype.itemsize * 8),
                    "PhysicalSizeX": spacing[2],
                    "PhysicalSizeXUnit": "\N{MICRO SIGN}m",
                    "PhysicalSizeY": spacing[1],
                    "PhysicalSizeYUnit": "\N{MICRO SIGN}m",
                    "PhysicalSizeZ": spacing[0],
                    "PhysicalSizeZUnit": "\N{MICRO SIGN}m",
                    "Channel": {"Name": [name]},
                    "Plane": {
                        "PositionX": [origin[2]] * shape_zyx[0],
                        "PositionXUnit": ["\N{MICRO SIGN}m"] * shape_zyx[0],
                        "PositionY": [origin[1]] * shape_zyx[0],
                        "PositionYUnit": ["\N{MICRO SIGN}m"] * shape_zyx[0],
                        "PositionZ": [
                            origin[0] + z * spacing[0] for z in range(shape_zyx[0])
                        ],
                        "PositionZUnit": ["\N{MICRO SIGN}m"] * shape_zyx[0],
                    },
                    "Description": json.dumps(
                        {
                            "affine_zyx_um": fused_metadata["affine_zyx_um"],
                            "registration_chain": chain,
                        },
                        separators=(",", ":"),
                    ),
                },
            )


def _channel_global_transforms_zyx_um(
    *,
    datastore: qi2labDataStore,
    tile_id: str,
    bit_ids: list[str],
    tile_position_zyx_um: Any,
    stage_camera_zyx_um: np.ndarray,
    global_refinement_zyx_um: np.ndarray,
) -> np.ndarray:
    """Compose native channel coordinates into the refined world frame.

    The stored local-round affine maps round-1 reference coordinates to the
    moving round, so native-to-reference uses its inverse. The known stage-camera
    affine is used as stored, followed by global refinement. Chromatic correction
    is applied only in decoding; SOFIMA fields are not loaded here.

    Parameters
    ----------
    datastore : qi2labDataStore
        Datastore containing local transform metadata.
    tile_id : str
        Tile identifier.
    bit_ids : list[str]
        Ordered readout bit identifiers.
    tile_position_zyx_um : Any
        Tile stage origin in physical Z, Y, X coordinates.
    stage_camera_zyx_um : numpy.ndarray
        Known stage-camera affine, used without inversion.
    global_refinement_zyx_um : numpy.ndarray
        Stored multiview-stitcher world refinement.

    Returns
    -------
    numpy.ndarray
        One forward 4x4 affine per channel, ordered as fiducial then bits.
    """
    tile_origin = np.eye(4, dtype=np.float32)
    tile_origin[:3, 3] = np.asarray(tile_position_zyx_um, dtype=np.float32)
    inverse_tile_origin = np.linalg.inv(tile_origin)
    global_stage = np.asarray(global_refinement_zyx_um, dtype=np.float32) @ np.asarray(
        stage_camera_zyx_um, dtype=np.float32
    )
    channel_transforms = [global_stage]

    for bit_id in bit_ids:
        _round_id, round_transform = load_bit_round_transform_zyx_um(
            datastore,
            tile=tile_id,
            bit_id=bit_id,
        )
        native_to_reference = np.linalg.inv(round_transform)
        channel_transforms.append(
            global_stage @ tile_origin @ native_to_reference @ inverse_tile_origin
        )

    return np.asarray(channel_transforms, dtype=np.float32)


def _load_tile_multichannel_msim(
    *,
    datastore: qi2labDataStore,
    tile_id: str,
    bit_ids: list[str],
    spacing_zyx_um: dict[str, float],
) -> Any:
    """
    Build a Zarr-backed multichannel tile with per-channel world affines.

    Parameters
    ----------
    datastore : qi2labDataStore
        Datastore providing image paths and registration metadata.
    tile_id : str
        Tile identifier.
    bit_ids : list[str]
        Readout bits in desired output-channel order.
    spacing_zyx_um : dict[str, float]
        Rounded physical voxel spacing keyed by Z, Y, and X.

    Returns
    -------
    Any
        MultiscaleSpatialImage whose first channel is the fiducial and whose
        remaining channels are the ordered bits.

    Raises
    ------
    FileNotFoundError
        If a required channel image is absent.
    RuntimeError
        If the tile has no stored global refinement transform.
    """
    reference_round = datastore.round_ids[0]
    tile_position, stage_camera = datastore.load_local_stage_position_zyx_um(
        tile_id,
        reference_round,
    )
    channel_ids = [datastore.fiducial_folder_name, *bit_ids]
    channel_arrays = [
        datastore.load_local_fusion_image(tile_id, round=reference_round),
        *[datastore.load_local_fusion_image(tile_id, bit=bit_id) for bit_id in bit_ids],
    ]
    from merfish3danalysis.storage.mvs import shared_mvs_chunks

    chunks = shared_mvs_chunks(channel_arrays)
    if chunks is not None:
        channel_arrays = [
            datastore.load_local_fusion_image(
                tile_id, round=reference_round, chunks=chunks
            ),
            *[
                datastore.load_local_fusion_image(tile_id, bit=bit_id, chunks=chunks)
                for bit_id in bit_ids
            ],
        ]
    translation = dict(
        zip(
            "zyx",
            np.round(tile_position, 2).astype(float),
            strict=True,
        )
    )
    channel_sims = []
    for channel_id, array in zip(channel_ids, channel_arrays, strict=True):
        channel_sim = _read_fiducial_sim(
            array=array,
            scale={
                axis: round_pixel_size_um(value)
                for axis, value in spacing_zyx_um.items()
            },
            translation=translation,
            affine_zyx_px=stage_camera,
            transform_key="stage_metadata",
        )
        channel_sims.append(channel_sim.assign_coords(c=[channel_id]))

    sim = si_utils.concat(channel_sims, dim="c")
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    stage_transform = msi_utils.get_transform_from_msim(
        msim,
        transform_key="stage_metadata",
    )
    global_refinement, _origin, _spacing = datastore.load_global_coord_xforms_um(
        tile=tile_id
    )
    if global_refinement is None:
        raise RuntimeError(f"Missing global transform for tile={tile_id}.")

    transforms = _channel_global_transforms_zyx_um(
        datastore=datastore,
        tile_id=tile_id,
        bit_ids=bit_ids,
        tile_position_zyx_um=tile_position,
        stage_camera_zyx_um=np.asarray(stage_transform).squeeze(),
        global_refinement_zyx_um=global_refinement,
    )
    transform_data = xr.DataArray(
        transforms,
        dims=("c", "x_in", "x_out"),
        coords={
            "c": channel_ids,
            "x_in": stage_transform.coords["x_in"],
            "x_out": stage_transform.coords["x_out"],
        },
    )
    msi_utils.set_affine_transform(
        msim,
        transform_data,
        transform_key="global_registered",
    )
    return msim


def _bit_number(bit_id: str) -> int:
    """Return the numeric suffix of a stored bit identifier."""
    return int(bit_id[3:])


@app.command()
def fuse_all_channels(
    root_path: Annotated[
        Path, typer.Argument(help="Experiment root containing qi2labdatastore.")
    ],
    write_ome_tiffs: Annotated[
        bool,
        typer.Option(
            "--write-ome-tiffs/--no-write-ome-tiffs",
            help="Also write one full-resolution OME-TIFF per fused channel.",
        ),
    ] = False,
    output_chunk_zyx: Annotated[
        tuple[int, int, int] | None,
        typer.Option(
            "--output-chunk-zyx",
            min=1,
            help=(
                "Fusion/output chunk shape as Z Y X. Larger chunks reduce task "
                "overhead but increase memory and downstream read amplification."
            ),
        ),
    ] = None,
    fusion_workers: Annotated[
        int | None,
        typer.Option(
            "--fusion-workers",
            min=1,
            help="Number of CPU fusion worker processes (default: available CPUs).",
        ),
    ] = None,
    compression: Annotated[
        str,
        typer.Option(
            "--compression",
            help=(
                "Lossless output compression: blosc-zstd, blosc-lz4, or zstd. "
                "Blosc codecs use bitshuffle; compression is always enabled."
            ),
        ),
    ] = "blosc-zstd",
    compression_level: Annotated[
        int,
        typer.Option(
            "--compression-level",
            min=1,
            max=9,
            help="Compression effort for every resolution level (1 is fastest).",
        ),
    ] = 1,
) -> None:
    """Fuse the first-round fiducial and all readout bits into one OME-Zarr.

    Each tile is represented as one lazy multichannel spatial image. Channel-
    dependent multiview-stitcher affines compose local round registration,
    the known stage-camera affine, stage location, and the
    stored global refinement. SOFIMA fields are intentionally ignored. One CPU
    fusion writes fiducial first, followed by numerically ordered codebook bits.

    Parameters
    ----------
    root_path : pathlib.Path
        Experiment root containing the ``qi2labdatastore`` directory.
    write_ome_tiffs : bool, default=False
        Write one full-resolution tiled OME-TIFF for each fused channel when
        ``True``.
    output_chunk_zyx : tuple[int, int, int] or None, default=None
        Optional Z, Y, X fusion and output chunk shape. When
        omitted, multiview-stitcher preserves the source Zarr chunk shape.
    fusion_workers : int or None, default=None
        Number of CPU worker processes. Defaults to the available CPU count.
    compression : str, default="blosc-zstd"
        Lossless output codec: ``blosc-zstd``, ``blosc-lz4``, or ``zstd``.
        Blosc codecs use bitshuffle. All resolution levels remain compressed.
    compression_level : int, default=1
        Compression effort from 1 through 9. Higher levels can trade encoding
        speed for a smaller output without changing pixel values.

    Returns
    -------
    None
        The multiscale image is published in the datastore Scene. Optional
        TIFFs are written to ``qi2labdatastore/fused_tiff``.
    """
    output_chunksize = (
        dict(zip("zyx", output_chunk_zyx, strict=True))
        if output_chunk_zyx is not None
        else None
    )
    try:
        fusion_options = _direct_zarr_fusion_kwargs(
            misc_utils=misc_utils,
            fusion_workers=fusion_workers,
            compression=compression,
            compression_level=compression_level,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    datastore = qi2labDataStore(resolve_datastore_path(root_path))
    stage_transform_key = "stage_metadata"
    global_transform_key = "global_registered"
    output_spacing = dict(
        zip(
            "zyx",
            round_spacing_um(datastore.voxel_size_zyx_um),
            strict=True,
        )
    )
    bit_ids = sorted(map(str, datastore.bit_ids), key=_bit_number)
    channels = [datastore.fiducial_folder_name, *bit_ids]
    tile_msims = [
        _load_tile_multichannel_msim(
            datastore=datastore,
            tile_id=tile_id,
            bit_ids=bit_ids,
            spacing_zyx_um=output_spacing,
        )
        for tile_id in tqdm(datastore.tile_ids, desc="tile")
    ]
    _fuse_views_to_scene(
        datastore,
        tile_msims,
        fusion_type="all_channels",
        output_spacing=output_spacing,
        output_chunksize=output_chunksize,
        batch_options=fusion_options["batch_options"],
        compression=compression,
        compression_level=compression_level,
        extra_attributes={
            "channel_names": channels,
            "registration_provenance": {
                "stage_transform_key": stage_transform_key,
                "local_transform": "local_round_transform_zyx_um",
                "global_transform_key": global_transform_key,
                "sofima_applied": False,
            },
        },
    )

    if write_ome_tiffs:
        output_directory = datastore.fused_tiff_path()
        output_directory.mkdir(parents=True, exist_ok=True)
        export_ome_tiffs(
            datastore,
            output_directory,
            channels,
        )

    del tile_msims
    gc.collect()


def main() -> None:
    """Run the fuse-all Typer command-line interface.

    Returns
    -------
    None
        The command-line application runs to completion.
    """
    mp.set_start_method("spawn", force=True)
    app()


if __name__ == "__main__":
    main()
