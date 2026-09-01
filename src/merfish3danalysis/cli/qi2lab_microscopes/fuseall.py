"""Fuse the first-round fiducial and all readout bits into one OME-Zarr."""

import gc
import json
import multiprocessing as mp
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import dask.array as da
import numpy as np
import typer
import xarray as xr
import zarr
from multiview_stitcher import fusion, misc_utils, msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from tqdm import tqdm

from merfish3danalysis.cli.qi2lab_microscopes._common import qi2lab_datastore_path
from merfish3danalysis.DataRegistration import (
    GlobalRegistrationConfig,
    _direct_zarr_fusion_kwargs,
)
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.decode_warping import (
    compose_decode_warp_transform_zyx_um,
    load_bit_round_transform_zyx_um,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

app = typer.Typer()
app.pretty_exceptions_enable = False


def export_ome_tiffs(
    ome_zarr_path: Path,
    output_directory: Path,
    channel_ids: list[str],
    fused_metadata: dict[str, list[Any]],
    registration_config: GlobalRegistrationConfig,
) -> None:
    """Export each full-resolution fused channel as a tiled OME-TIFF.

    Parameters
    ----------
    ome_zarr_path : pathlib.Path
        Path to the multichannel OME-Zarr v0.5 image.
    output_directory : pathlib.Path
        Directory in which to write the channel OME-TIFF files.
    channel_ids : list[str]
        Ordered channel names matching the OME-Zarr channel axis.
    fused_metadata : dict[str, list[Any]]
        Fused affine, origin, and spacing metadata in physical Z, Y, X order.
    registration_config : GlobalRegistrationConfig
        Registration transform keys used to record transform provenance.

    Returns
    -------
    None
        One tiled OME-TIFF is written for each channel.
    """
    from tifffile import TiffWriter

    group = zarr.open_group(ome_zarr_path, mode="r")
    scale0_path = group.attrs["ome"]["multiscales"][0]["datasets"][0]["path"]
    array = zarr.open_array(ome_zarr_path / scale0_path, mode="r")
    dims = tuple(str(dim) for dim in array.metadata.dimension_names)
    axes = {dim: dims.index(dim) for dim in dims}
    shape_zyx = tuple(int(array.shape[axes[dim]]) for dim in "zyx")
    origin = tuple(map(float, fused_metadata["origin_zyx_um"]))
    spacing = tuple(map(float, fused_metadata["spacing_zyx_um"]))

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
                    source = np.asarray(array[tuple(selection)])
                    tile = np.zeros((tile_y, tile_x), dtype=array.dtype)
                    tile[: source.shape[0], : source.shape[1]] = source
                    yield tile

    for channel_index, name in enumerate(tqdm(channel_ids, desc="ome-tiff")):
        chain = [registration_config.transform_key]
        if channel_index:
            chain.extend(
                [
                    "chromatic_affine_transforms_zyx_um",
                    "local_round_transform_zyx_um",
                ]
            )
        chain.append(registration_config.new_transform_key)
        with TiffWriter(
            output_directory / f"{name}.ome.tiff", bigtiff=True, ome=True
        ) as tif:
            tif.write(
                data=tiles(channel_index),
                shape=shape_zyx,
                dtype=array.dtype,
                tile=(1024, 1024),
                resolution=(1e4 / spacing[2], 1e4 / spacing[1]),
                compression="zlib",
                compressionargs={"level": 8},
                predictor=True,
                photometric="minisblack",
                resolutionunit="CENTIMETER",
                metadata={
                    "axes": "ZYX",
                    "SignificantBits": int(np.dtype(array.dtype).itemsize * 8),
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

    The stored chromatic affine maps a native readout channel to its reference
    wavelength. The stored local-round affine has the opposite direction: it
    maps round-1 reference coordinates to the moving round. Consequently, the
    forward native-to-round-1 channel transform is ``inverse(local) @
    chromatic``. The known stage-camera affine is used as stored, followed by
    the multiview-stitcher global refinement. SOFIMA fields are not loaded.

    Parameters
    ----------
    datastore : qi2labDataStore
        Datastore containing local and chromatic transform metadata.
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
        wavelengths = datastore.load_local_wavelengths_um(
            tile=tile_id,
            bit=bit_id,
        )
        if wavelengths is None:
            raise RuntimeError(f"Missing wavelengths for tile={tile_id} bit={bit_id}.")
        chromatic_transform = datastore.load_chromatic_affine_transform_zyx_um(
            wavelength_um=float(wavelengths[1])
        )
        reference_to_native = compose_decode_warp_transform_zyx_um(
            round_transform_zyx_um=round_transform,
            chromatic_transform_zyx_um=chromatic_transform,
        )
        native_to_reference = np.linalg.inv(reference_to_native)
        channel_transforms.append(
            global_stage @ tile_origin @ native_to_reference @ inverse_tile_origin
        )

    return np.asarray(channel_transforms, dtype=np.float32)


def _load_tile_multichannel_msim(
    *,
    datastore: qi2labDataStore,
    tile_id: str,
    bit_ids: list[str],
    da_module: Any = da,
    msi_utils_module: Any = msi_utils,
    si_utils_module: Any = si_utils,
) -> Any:
    """Build one lazy CZYX tile with channel-dependent world affines."""
    reference_round = datastore.round_ids[0]
    tile_position, stage_camera = datastore.load_local_stage_position_zyx_um(
        tile_id,
        reference_round,
    )
    chunks_by_dim = si_utils_module.get_default_spatial_chunksizes(3)
    spatial_chunks = tuple(chunks_by_dim[dim] for dim in "zyx")

    fiducial = datastore.load_local_fiducial_image(
        tile=tile_id,
        round=reference_round,
        return_future=None,
    )
    if fiducial is None:
        raise FileNotFoundError(
            f"Missing fusion input for tile={tile_id} channel="
            f"{datastore.fiducial_folder_name}."
        )
    channel_data = [da_module.from_array(fiducial, chunks=spatial_chunks)]

    for bit_id in bit_ids:
        readout = datastore.load_local_readout_image(
            tile=tile_id,
            bit=bit_id,
            return_future=None,
        )
        if readout is None:
            raise FileNotFoundError(
                f"Missing fusion input for tile={tile_id} channel={bit_id}."
            )
        channel_data.append(da_module.from_array(readout, chunks=spatial_chunks))

    channel_ids = [datastore.fiducial_folder_name, *bit_ids]
    sim = si_utils_module.get_sim_from_array(
        da_module.stack(channel_data, axis=0),
        dims=("c", "z", "y", "x"),
        scale=dict(zip("zyx", map(float, datastore.voxel_size_zyx_um), strict=True)),
        translation=dict(
            zip(
                "zyx",
                np.round(tile_position, 2).astype(float),
                strict=True,
            )
        ),
        affine=stage_camera,
        transform_key="stage_metadata",
        c_coords=channel_ids,
        t_coords=None,
    )
    msim = msi_utils_module.get_msim_from_sim(sim, scale_factors=[])
    stage_transform = msi_utils_module.get_transform_from_msim(
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
        transforms[:, None],
        dims=("c", "t", "x_in", "x_out"),
        coords={
            "c": channel_ids,
            "t": stage_transform.coords["t"],
            "x_in": stage_transform.coords["x_in"],
            "x_out": stage_transform.coords["x_out"],
        },
    )
    msi_utils_module.set_affine_transform(
        msim,
        transform_data,
        transform_key="global_registered",
    )
    return msim


def _read_fused_metadata(fused_msim: Any) -> dict[str, list[Any]]:
    """Return affine, origin, and spacing metadata from a fused image."""
    affine = msi_utils.get_transform_from_msim(
        fused_msim,
        transform_key="global_registered",
    ).data.squeeze()
    fused_sim = msi_utils.get_sim_from_msim(fused_msim)
    origin = si_utils.get_origin_from_sim(fused_sim, asarray=True)
    spacing = si_utils.get_spacing_from_sim(fused_sim, asarray=True)
    return {
        "affine_zyx_um": np.asarray(affine, dtype=np.float32).tolist(),
        "origin_zyx_um": np.asarray(origin, dtype=np.float32).tolist(),
        "spacing_zyx_um": np.asarray(spacing, dtype=np.float32).tolist(),
    }


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
) -> None:
    """Fuse the first-round fiducial and all readout bits into one OME-Zarr.

    Each tile is represented as one lazy multichannel spatial image. Channel-
    dependent multiview-stitcher affines compose chromatic correction, local
    round registration, the known stage-camera affine, stage location, and the
    stored global refinement. SOFIMA fields are intentionally ignored. One CPU
    fusion writes fiducial first, followed by numerically ordered codebook bits.

    Parameters
    ----------
    root_path : pathlib.Path
        Experiment root containing the ``qi2labdatastore`` directory.
    write_ome_tiffs : bool, default=False
        Write one full-resolution tiled OME-TIFF for each fused channel when
        ``True``.

    Returns
    -------
    None
        The multiscale image is written to
        ``qi2labdatastore/fused/full_dataset.ome.zarr``. Optional
        TIFFs are written beside it.
    """
    datastore = qi2labDataStore(qi2lab_datastore_path(root_path))
    registration_config = GlobalRegistrationConfig()
    bit_ids = sorted(map(str, datastore.bit_ids), key=lambda bit: int(bit[3:]))
    channels = [datastore.fiducial_folder_name, *bit_ids]
    tile_msims = [
        _load_tile_multichannel_msim(
            datastore=datastore,
            tile_id=tile_id,
            bit_ids=bit_ids,
        )
        for tile_id in tqdm(datastore.tile_ids, desc="tile")
    ]
    output_directory = datastore._fused_root_path
    output_directory.mkdir(parents=True, exist_ok=True)
    final_output = datastore._image_store_path(output_directory / "full_dataset")
    fused_msim = fusion.fuse(
        images=tile_msims,
        transform_key=registration_config.new_transform_key,
        output_zarr_url=str(final_output),
        **_direct_zarr_fusion_kwargs(misc_utils=misc_utils),
    )
    fused_metadata = _read_fused_metadata(fused_msim)
    datastore._write_extra_attributes(
        image_path=final_output,
        extra_attributes={
            **fused_metadata,
            "channel_names": channels,
            "registration_provenance": {
                "stage_transform_key": registration_config.transform_key,
                "chromatic_transform": "chromatic_affine_transforms_zyx_um",
                "local_transform": "local_round_transform_zyx_um",
                "global_transform_key": registration_config.new_transform_key,
                "sofima_applied": False,
            },
        },
        merge=True,
    )

    if write_ome_tiffs:
        export_ome_tiffs(
            final_output,
            output_directory,
            channels,
            fused_metadata,
            registration_config,
        )

    del fused_msim, tile_msims
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
