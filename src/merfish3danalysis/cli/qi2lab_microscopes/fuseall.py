"""
Fuse all channels into individual OME-NGFF v0.5 stores for viewing.

Shepherd 2025/03 - created script.
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import gc
import multiprocessing as mp
import re
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.decode_warping import warp_bit_image_to_reference

mp.set_start_method("spawn", force=True)


def _resolve_array(array_like: Any) -> Any:
    """Return a concrete array from datastore results or dask futures."""

    return array_like.result() if hasattr(array_like, "result") else array_like


def _load_readout_image(datastore: Any, *, tile_id: str, bit_id: str) -> np.ndarray:
    decon_image = datastore.load_local_registered_image(
        tile=tile_id,
        bit=bit_id,
        return_future=False,
    )
    return np.asarray(_resolve_array(decon_image), dtype=np.float32)


def _load_warped_readout_image_for_fusion(
    datastore: Any,
    *,
    tile_id: str,
    bit_id: str,
    gpu_id: int = 0,
) -> np.ndarray:
    """Load one readout bit and warp it into the round-0 reference frame."""

    readout = _load_readout_image(
        datastore,
        tile_id=tile_id,
        bit_id=bit_id,
    )
    _excitation_wavelength_um, emission_wavelength_um = (
        datastore.load_local_wavelengths_um(
            tile=tile_id,
            bit=bit_id,
        )
    )
    warped = warp_bit_image_to_reference(
        readout,
        datastore=datastore,
        tile=tile_id,
        bit_id=bit_id,
        emission_wavelength_um=emission_wavelength_um,
        gpu_id=gpu_id,
    )
    return np.asarray(warped).clip(0, 2**16 - 1).astype(np.uint16)


def _attach_stored_global_transform(
    msim: Any,
    *,
    datastore: Any,
    tile_id: str,
    msi_utils: Any,
    transform_key: str,
) -> None:
    affine_zyx_um, _origin_zyx_um, _spacing_zyx_um = (
        datastore.load_global_coord_xforms_um(tile=tile_id)
    )
    if affine_zyx_um is None:
        raise RuntimeError(
            "Stored global transform is required for channel fusion, "
            f"but none was found for tile {tile_id!r}."
        )
    msi_utils.set_affine_transform(
        msim,
        np.asarray(affine_zyx_um, dtype=np.float32)[None, ...],
        transform_key=transform_key,
    )


def _channel_output_name(channel_id: str) -> str:
    channel_id = str(channel_id)
    if channel_id == "fiducial":
        return "fiducial"
    match = re.fullmatch(r"bit0*(\d+)", channel_id)
    if match:
        return f"bit{int(match.group(1)):03d}"
    if channel_id.isdigit():
        return f"bit{int(channel_id):03d}"
    return channel_id


def _write_fused_ome_tiff(
    *,
    ome_zarr_path: Path,
    ome_tiff_path: Path,
    ngff_utils: Any,
    spacing_zyx_um: tuple[float, float, float],
    transform_key: str,
) -> None:
    from tifffile import TiffWriter

    sim = ngff_utils.read_sim_from_ome_zarr(
        ome_zarr_path,
        resolution_level=0,
        transform_key=transform_key,
        use_dask=False,
    )
    image = np.asarray(sim.data)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    image = np.asarray(image).clip(0, 2**16 - 1).astype(np.uint16, copy=False)

    with TiffWriter(ome_tiff_path, bigtiff=True) as tif:
        tif.write(
            image,
            photometric="minisblack",
            compression="zlib",
            compressionargs={"level": 8},
            predictor=True,
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": float(spacing_zyx_um[2]),
                "PhysicalSizeY": float(spacing_zyx_um[1]),
                "PhysicalSizeZ": float(spacing_zyx_um[0]),
            },
        )


def fuse_all_channels(
    root_path: Path,
    fused_chunk_size: int = 512,
    n_jobs: int = 8,
    use_gpu_fusion: bool = True,
    ngff_version: str = "0.5",
    gpu_id: int = 0,
) -> None:
    """Register all channels across all tiles.

    Registration is performed using the fiducial channel.

    Parameters
    ----------
    root_path: Path
        path to experiment
    fused_chunk_size : int, default=512
        Fused OME-Zarr chunk size.
    n_jobs : int, default=8
        Number of chunk fusion jobs to process per batch.
    use_gpu_fusion : bool, default=True
        Use multiview-stitcher's CuPy backend for per-chunk fusion.
    ngff_version : str, default="0.5"
        OME-NGFF version requested for direct multiview-stitcher Zarr output.
    gpu_id : int, default=0
        CUDA device ID used for local readout warping.
    """
    import dask
    import dask.array as da
    from multiview_stitcher import fusion, misc_utils, msi_utils, ngff_utils
    from multiview_stitcher import spatial_image_utils as si_utils

    # initialize datastore
    print("\nInitializing datastore...")
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    bit_ids = list(datastore.bit_ids)
    channel_ids = ["fiducial", *bit_ids]
    global_transform_key = "global_registered"
    stage_transform_key = "stage_metadata"

    first_fiducial_path = (
        datastore_path
        / Path("fiducial")
        / Path(datastore.tile_ids[0])
        / Path(datastore.round_ids[0])
        / Path("registered_decon_data.ome.zarr")
    )
    first_fiducial_sim = ngff_utils.read_sim_from_ome_zarr(
        first_fiducial_path,
        resolution_level=0,
        transform_key="stage_metadata",
        use_dask=False,
    )
    im_shape = tuple(int(first_fiducial_sim.sizes[dim]) for dim in ("z", "y", "x"))
    del first_fiducial_sim

    voxel_zyx_um = datastore.voxel_size_zyx_um
    scale = {
        "z": float(voxel_zyx_um[0]),
        "y": float(voxel_zyx_um[1]),
        "x": float(voxel_zyx_um[2]),
    }

    print("\nLazy loading and fusing full-resolution fiducial and readouts...")
    tile_ids = datastore.tile_ids
    fused_path = root_path / Path("fused")
    fused_path.mkdir(exist_ok=True)

    for ch_idx in tqdm(range(len(channel_ids)), desc="channel"):
        channel_name = _channel_output_name(channel_ids[ch_idx])
        msims_full = []
        for tile_id in tqdm(tile_ids, desc="tile"):
            tile_position_zyx_um, affine_zyx_px = (
                datastore.load_local_stage_position_zyx_um(
                    tile_id,
                    datastore.round_ids[0],
                )
            )
            tile_grid_positions = {
                "z": float(np.round(tile_position_zyx_um[0], 2)),
                "y": float(np.round(tile_position_zyx_um[1], 2)),
                "x": float(np.round(tile_position_zyx_um[2], 2)),
            }

            # temporary variable for channel data
            im_data = da.zeros(
                (1, im_shape[0], im_shape[1], im_shape[2]), dtype=np.uint16
            )

            # lazy load deconvolved fiducial
            if ch_idx == 0:
                input_path = (
                    datastore_path
                    / Path("fiducial")
                    / Path(tile_id)
                    / Path("round001")
                    / Path("registered_decon_data.ome.zarr")
                )
                im_data[0, :] = da.from_zarr(str(input_path)).astype(np.uint16)
            # lazy load readout bits warped to the first-round local reference
            else:
                bit_id = bit_ids[ch_idx - 1]
                delayed_image = dask.delayed(_load_warped_readout_image_for_fusion)(
                    datastore,
                    tile_id=tile_id,
                    bit_id=bit_id,
                    gpu_id=gpu_id,
                )
                im_data[0, :] = da.from_delayed(
                    delayed_image,
                    shape=im_shape,
                    dtype=np.uint16,
                )

            # attach stored global transforms from global registration, then fuse
            sim_full = si_utils.get_sim_from_array(
                im_data,
                dims=("c", "z", "y", "x"),
                scale=scale,
                translation=tile_grid_positions,
                affine=affine_zyx_px,
                transform_key=stage_transform_key,
                c_coords=channel_ids[ch_idx],
            )

            # convert to multiscale spatial image object and append to list for fusion
            msim_full = msi_utils.get_msim_from_sim(sim_full, scale_factors=[])
            _attach_stored_global_transform(
                msim_full,
                datastore=datastore,
                tile_id=tile_id,
                msi_utils=msi_utils,
                transform_key=global_transform_key,
            )
            msims_full.append(msim_full)
            del im_data
            gc.collect()

        # create fused image object using previously calculated registration metadata and all channels
        print("Constructing fusion...")
        ome_output_path = fused_path / Path(f"{channel_name}.ome.zarr")
        print(f"Fusing views and saving output to {ome_output_path!s}...")
        fused = fusion.fuse(
            [msi_utils.get_sim_from_msim(msim_full) for msim_full in msims_full],
            transform_key=global_transform_key,
            output_spacing=scale,
            output_chunksize=fused_chunk_size,
            overlap_in_pixels=64,
            output_zarr_url=str(ome_output_path),
            zarr_options={
                "ome_zarr": True,
                "ngff_version": ngff_version,
                "overwrite": True,
            },
            batch_options={
                "batch_func": misc_utils.process_batch_using_joblib,
                "n_batch": int(n_jobs),
                "batch_func_kwargs": {
                    "n_jobs": int(n_jobs),
                    "backend": "threading",
                },
            },
            **(
                {"backend": "cupy", "output_on_backend": False}
                if use_gpu_fusion
                else {}
            ),
        )
        if not hasattr(fused, "data"):
            fused = msi_utils.get_sim_from_msim(fused, scale="scale0")
        del fused

    print("\nWriting fused OME-TIFF outputs...")
    spacing_zyx_um = tuple(float(v) for v in voxel_zyx_um)
    for channel_id in tqdm(channel_ids, desc="ome-tiff"):
        channel_name = _channel_output_name(channel_id)
        ome_zarr_path = fused_path / Path(f"{channel_name}.ome.zarr")
        ome_tiff_path = fused_path / Path(f"{channel_name}.ome.tiff")
        print(f"Writing {ome_tiff_path!s} from {ome_zarr_path!s}...")
        _write_fused_ome_tiff(
            ome_zarr_path=ome_zarr_path,
            ome_tiff_path=ome_tiff_path,
            ngff_utils=ngff_utils,
            spacing_zyx_um=spacing_zyx_um,
            transform_key=global_transform_key,
        )


if __name__ == "__main__":
    root_path = Path(r"/mnt/data2/bioprotean/20250220_Bartelle_control_smFISH_TqIB")
    fuse_all_channels(root_path)
