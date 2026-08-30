"""Fuse the first-round fiducial and all readout bits into one OME-Zarr."""

import gc
import json
import multiprocessing as mp
import shutil
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import dask.array as da
import numpy as np
import typer
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

    Each channel is fused sequentially from lazy datastore arrays. Readout
    channels receive their stored chromatic and local-round affine chain before
    the stored global tile transform is applied. The fiducial fusion defines
    the common output grid used by every subsequent bit.

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
        ``qi2labdatastore/fused/fused_all_channels.ome.zarr``. Optional TIFFs
        are written beside it.
    """
    datastore = qi2labDataStore(qi2lab_datastore_path(root_path))
    registration_config = GlobalRegistrationConfig()
    channels = [
        datastore.fiducial_folder_name,
        *sorted(map(str, datastore.bit_ids), key=lambda bit: int(bit[3:])),
    ]
    reference_round = datastore.round_ids[0]
    spatial_chunks = si_utils.get_default_spatial_chunksizes(3)
    spatial_chunks = tuple(spatial_chunks[dim] for dim in "zyx")
    voxel_scale = dict(zip("zyx", map(float, datastore.voxel_size_zyx_um), strict=True))

    output_directory = datastore._fused_root_path
    work_directory = output_directory / ".fuseall-work"
    if work_directory.exists():
        shutil.rmtree(work_directory)
    work_directory.mkdir()
    staged_output = datastore._image_store_path(work_directory / "fused_all_channels")
    final_output = datastore._image_store_path(output_directory / "fused_all_channels")
    output_stack_properties = None

    for channel_index, channel in enumerate(tqdm(channels, desc="channel")):
        tile_msims = []
        for tile_id in tqdm(datastore.tile_ids, desc=f"{channel} tiles", leave=False):
            tile_position, stage_affine = datastore.load_local_stage_position_zyx_um(
                tile_id, reference_round
            )
            image = (
                datastore.load_local_fiducial_image(
                    tile=tile_id,
                    round=reference_round,
                    return_future=None,
                )
                if channel == datastore.fiducial_folder_name
                else datastore.load_local_readout_image(
                    tile=tile_id,
                    bit=channel,
                    return_future=None,
                )
            )
            if image is None:
                raise FileNotFoundError(
                    f"Missing fusion input for tile={tile_id} channel={channel}."
                )

            image = da.from_array(image, chunks=spatial_chunks)
            sim = si_utils.get_sim_from_array(
                image,
                dims=("z", "y", "x"),
                scale=voxel_scale,
                translation=dict(
                    zip(
                        "zyx",
                        np.round(tile_position, 2).astype(float),
                        strict=True,
                    )
                ),
                affine=stage_affine,
                transform_key=registration_config.transform_key,
                c_coords=None,
                t_coords=None,
            )
            msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
            channel_transform, _origin, _spacing = (
                datastore.load_global_coord_xforms_um(tile=tile_id)
            )
            if channel_transform is None:
                raise RuntimeError(f"Missing global transform for tile={tile_id}.")

            if channel != datastore.fiducial_folder_name:
                _round_id, local_round = load_bit_round_transform_zyx_um(
                    datastore, tile=tile_id, bit_id=channel
                )
                wavelengths = datastore.load_local_wavelengths_um(
                    tile=tile_id, bit=channel
                )
                if wavelengths is None:
                    raise RuntimeError(
                        f"Missing wavelengths for tile={tile_id} bit={channel}."
                    )
                chromatic = datastore.load_chromatic_affine_transform_zyx_um(
                    wavelength_um=float(wavelengths[1])
                )
                reference_to_native = compose_decode_warp_transform_zyx_um(
                    round_transform_zyx_um=local_round,
                    chromatic_transform_zyx_um=chromatic,
                )
                tile_origin = np.eye(4, dtype=np.float32)
                tile_origin[:3, 3] = np.asarray(tile_position, dtype=np.float32)
                channel_transform = np.asarray(channel_transform, dtype=np.float32) @ (
                    tile_origin
                    @ np.linalg.inv(reference_to_native)
                    @ np.linalg.inv(tile_origin)
                )

            msi_utils.set_affine_transform(
                msim,
                np.asarray(channel_transform, dtype=np.float32)[None],
                transform_key=registration_config.new_transform_key,
            )
            tile_msims.append(msim)

        channel_path = datastore._image_store_path(work_directory / channel)
        fusion_kwargs = _direct_zarr_fusion_kwargs(misc_utils=misc_utils)
        if output_stack_properties is not None:
            fusion_kwargs["output_stack_properties"] = output_stack_properties
        fused_msim = fusion.fuse(
            images=tile_msims,
            transform_key=registration_config.new_transform_key,
            output_zarr_url=str(channel_path),
            **fusion_kwargs,
        )

        if channel_index == 0:
            fused_sim = msi_utils.get_sim_from_msim(fused_msim)
            spatial_dims = tuple(si_utils.get_spatial_dims_from_sim(fused_sim))
            spacing = np.asarray(
                si_utils.get_spacing_from_sim(fused_sim, asarray=True),
                dtype=np.float64,
            )
            origin = np.asarray(
                si_utils.get_origin_from_sim(fused_sim, asarray=True),
                dtype=np.float64,
            )
            output_stack_properties = {
                "spacing": dict(zip(spatial_dims, spacing.tolist(), strict=True)),
                "origin": dict(zip(spatial_dims, origin.tolist(), strict=True)),
                "shape": {dim: int(fused_sim.sizes[dim]) for dim in spatial_dims},
            }
            affine = msi_utils.get_transform_from_msim(
                fused_msim,
                transform_key=registration_config.new_transform_key,
            ).data.squeeze()
            fused_metadata = {
                "affine_zyx_um": np.asarray(affine, dtype=np.float32).tolist(),
                "origin_zyx_um": origin.astype(np.float32).tolist(),
                "spacing_zyx_um": spacing.astype(np.float32).tolist(),
            }

            shutil.copytree(channel_path, staged_output)
            output_group = zarr.open_group(staged_output, mode="r+")
            dataset_paths = [
                dataset["path"]
                for dataset in output_group.attrs["ome"]["multiscales"][0]["datasets"]
            ]
            for dataset_path in dataset_paths:
                array = zarr.open_array(staged_output / dataset_path, mode="r+")
                dims = tuple(str(dim) for dim in array.metadata.dimension_names)
                channel_axis = dims.index("c")
                shape = list(array.shape)
                shape[channel_axis] = len(channels)
                array.resize(tuple(shape))
            output_group.attrs["channel_names"] = channels
            output_group.attrs["omero"] = {
                "channels": [{"label": name} for name in channels]
            }
        else:
            source_group = zarr.open_group(channel_path, mode="r")
            source_paths = [
                dataset["path"]
                for dataset in source_group.attrs["ome"]["multiscales"][0]["datasets"]
            ]
            if source_paths != dataset_paths:
                raise ValueError("Fused channel pyramids do not match.")
            for dataset_path in dataset_paths:
                source = zarr.open_array(channel_path / dataset_path, mode="r")
                target = zarr.open_array(staged_output / dataset_path, mode="r+")
                dims = tuple(str(dim) for dim in source.metadata.dimension_names)
                channel_axis = dims.index("c")
                region = [slice(None)] * source.ndim
                region[channel_axis] = slice(channel_index, channel_index + 1)
                da.store(
                    da.from_array(source, chunks=source.chunks),
                    target,
                    regions=tuple(region),
                    lock=False,
                    compute=True,
                )

        del fused_msim, tile_msims
        gc.collect()
        shutil.rmtree(channel_path)

    datastore._write_extra_attributes(
        image_path=staged_output,
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
    if final_output.exists():
        shutil.rmtree(final_output)
    staged_output.replace(final_output)
    shutil.rmtree(work_directory)

    if write_ome_tiffs:
        export_ome_tiffs(
            final_output,
            output_directory,
            channels,
            fused_metadata,
            registration_config,
        )


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
