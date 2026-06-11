"""
Perform registration on qi2labdatastore. By default creates a max
projection downsampled fiducial OME-TIFF for cellpose parameter optimization.

Shepherd 2025/10 - change to CLI.
Shepherd 2025/07 - rework for multiple GPU support.
Shepherd 2024/11 - rework script to accept parameters.
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

import gc
import inspect
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import typer
from joblib import Parallel, delayed
from tifffile import TiffWriter
from tqdm import tqdm

from merfish3danalysis.qi2labDataStore import qi2labDataStore

app = typer.Typer()
app.pretty_exceptions_enable = False


def batch_using_joblib(
    func: Callable[[int], None], block_ids: Sequence[int], n_jobs: int
) -> None:
    """Apply `func` to each block id in parallel using joblib.

    Parameters
    ----------
    func
        Function applied to each block id.
    block_ids
        Block IDs to process.
    n_jobs
        Number of parallel workers (joblib semantics).
    """
    Parallel(
        n_jobs=n_jobs,
        prefer="threads",
        require="sharedmem",
        pre_dispatch=n_jobs,
    )(delayed(func)(block_id) for block_id in block_ids)


def _local_registered_fiducial_path(
    datastore: qi2labDataStore,
    tile_id: str,
    round_id: str,
) -> Path:
    """
    Return the registered fiducial OME-Zarr path for one tile and round.

    Parameters
    ----------
    datastore : qi2labDataStore
        Open datastore containing registered fiducial images.
    tile_id : str
        Tile identifier, such as ``tile0000``.
    round_id : str
        Round identifier, such as ``round001``.

    Returns
    -------
    pathlib.Path
        Path to ``registered_decon_data.ome.zarr``.
    """

    return datastore._image_store_path(
        datastore._fiducial_root_path
        / Path(tile_id)
        / Path(round_id)
        / Path("registered_decon_data")
    )


def _get_batch_processing_options(
    misc_utils: Any,
    n_jobs: int,
    use_gpu_fusion: bool,
) -> dict[str, Any]:
    """
    Build multiview-stitcher batch options for direct Zarr fusion.

    Parameters
    ----------
    misc_utils : Any
        ``multiview_stitcher.misc_utils`` module when available.
    n_jobs : int
        Number of parallel fusion jobs.
    use_gpu_fusion : bool
        If True, use a thread-backed joblib batch executor so one process owns
        the CUDA context.

    Returns
    -------
    dict[str, Any]
        Batch options accepted by ``fusion.fuse``.
    """

    batch_func = getattr(misc_utils, "process_batch_using_joblib", None)
    if batch_func is None:
        batch_func = batch_using_joblib
        batch_kwargs = {"n_jobs": n_jobs}
    else:
        batch_kwargs = {"n_jobs": n_jobs}
        if use_gpu_fusion:
            batch_kwargs["backend"] = "threading"
    return {
        "batch_func": batch_func,
        "n_batch": n_jobs,
        "batch_func_kwargs": batch_kwargs,
    }


def _get_scale0_sim_from_fusion_result(
    fused: Any,
    msi_utils: Any,
) -> Any:
    """
    Return a scale0 SpatialImage from a fusion result.

    Parameters
    ----------
    fused : Any
        SpatialImage or MultiscaleSpatialImage returned by
        ``multiview_stitcher.fusion.fuse``.
    msi_utils : Any
        ``multiview_stitcher.msi_utils`` module.

    Returns
    -------
    Any
        SpatialImage at the highest written resolution.
    """

    if hasattr(fused, "data"):
        return fused
    return msi_utils.get_sim_from_msim(fused, scale="scale0")


def _get_fusion_backend_kwargs(
    fuse_func: Callable, use_gpu_fusion: bool
) -> dict[str, Any]:
    """
    Return GPU backend keyword arguments for supported multiview-stitcher versions.

    Parameters
    ----------
    fuse_func : Callable
        ``multiview_stitcher.fusion.fuse`` function.
    use_gpu_fusion : bool
        If True, request CuPy-backed per-chunk fusion.

    Returns
    -------
    dict[str, Any]
        Extra keyword arguments to pass to ``fusion.fuse``.
    """

    if not use_gpu_fusion:
        return {}

    fuse_parameters = inspect.signature(fuse_func).parameters
    if "backend" not in fuse_parameters:
        raise RuntimeError(
            "GPU fusion requires multiview-stitcher with fusion.fuse(..., "
            "backend='cupy'), expected in version 0.1.56 or newer."
        )

    try:
        import cupy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "GPU fusion requires CuPy. Install multiview-stitcher with the "
            "'gpu-cuda12' extra or install a CUDA-compatible CuPy package."
        ) from exc

    return {"backend": "cupy", "output_on_backend": False}


def _read_registered_fiducial_sim(
    input_path: Path,
    scale: dict[str, float],
    translation: dict[str, float],
    affine_zyx_px: Any,
    ngff_utils: Any,
    si_utils: Any,
) -> Any:
    """
    Read one registered fiducial OME-Zarr as a multiview-stitcher SpatialImage.

    Parameters
    ----------
    input_path : pathlib.Path
        Local registered fiducial OME-Zarr path.
    scale : dict[str, float]
        Physical pixel spacing by spatial dimension.
    translation : dict[str, float]
        Stage-derived physical origin by spatial dimension.
    affine_zyx_px : Any
        Local affine transform in pixel units.
    ngff_utils : Any
        ``multiview_stitcher.ngff_utils`` module.
    si_utils : Any
        ``multiview_stitcher.spatial_image_utils`` module.

    Returns
    -------
    Any
        SpatialImage with datastore stage metadata attached under
        ``stage_metadata``.
    """

    sim_on_disk = ngff_utils.read_sim_from_ome_zarr(
        input_path,
        resolution_level=0,
        transform_key="stage_metadata",
        use_dask=False,
    )
    return si_utils.get_sim_from_array(
        sim_on_disk.data,
        dims=sim_on_disk.dims,
        scale=scale,
        translation=translation,
        affine=affine_zyx_px,
        transform_key="stage_metadata",
        c_coords=sim_on_disk.coords["c"].values if "c" in sim_on_disk.coords else None,
        t_coords=sim_on_disk.coords["t"].values if "t" in sim_on_disk.coords else None,
    )


@app.command()
def global_register_data(
    root_path: Path,
    fused_chunk_size: int = 128,
    n_jobs: int = 16,
    swap_yx: bool = False,
    create_max_proj_tiff: bool = True,
    zstride_level: int = 0,
    use_gpu_fusion: bool = True,
    ngff_version: str = "0.5",
) -> None:
    """Register all tiles in first round in global coordinates.

    Parameters
    ----------
    root_path: Path
        path to experiment
    fused_chunk_size: int, default 128
        fused image chunk size
    n_jobs: int, default 16
        number of parallel fusion jobs to run
    swap_yx: bool, default False
        swap y and x coordinates when loading stage positions.
    create_max_proj_tiff: bool, default = True
        create max projection tiff in the segmentation/cellpose directory.
    zstride_level: int, default = 0
        look for a skip z dataset.
    use_gpu_fusion: bool, default = True
        Use multiview-stitcher's CuPy backend for per-chunk fusion. Requires
        multiview-stitcher 0.1.56 or newer and CuPy.
    ngff_version: str, default = "0.5"
        OME-NGFF version requested for direct multiview-stitcher Zarr output.
    """

    import dask.diagnostics
    from multiview_stitcher import (
        fusion,
        misc_utils,
        msi_utils,
        ngff_utils,
        registration,
    )
    from multiview_stitcher import spatial_image_utils as si_utils

    # initialize datastore
    if zstride_level == 0:
        datastore_path = root_path / Path(r"qi2labdatastore")
    else:
        datastore_path = root_path / Path(f"qi2labdatastore_zstride0{zstride_level}")
    datastore = qi2labDataStore(datastore_path)
    print(f"Using datastore at {datastore_path}")

    # convert local tiles from first round to multiscale spatial images
    msims = []
    for _, tile_id in enumerate(tqdm(datastore.tile_ids, desc="tile")):
        round_id = datastore.round_ids[0]

        voxel_zyx_um = datastore.voxel_size_zyx_um

        scale = {"z": voxel_zyx_um[0], "y": voxel_zyx_um[1], "x": voxel_zyx_um[2]}

        tile_position_zyx_um, affine_zyx_px = (
            datastore.load_local_stage_position_zyx_um(tile_id, round_id)
        )
        has_z = tile_position_zyx_um.size >= 3  # expects z,y,x
        if has_z:
            if swap_yx:
                tile_grid_positions = {
                    "z": float(np.round(tile_position_zyx_um[1], 2)),
                    "y": float(np.round(tile_position_zyx_um[0], 2)),
                    "x": float(np.round(tile_position_zyx_um[2], 2)),
                }
            else:
                tile_grid_positions = {
                    "z": float(np.round(tile_position_zyx_um[0], 2)),
                    "y": float(np.round(tile_position_zyx_um[1], 2)),
                    "x": float(np.round(tile_position_zyx_um[2], 2)),
                }
        else:
            # 2D fallback: treat input as y,x (or x,y) depending on your convention
            tile_grid_positions = {
                "z": 0.0,
                "y": float(np.round(tile_position_zyx_um[0], 2)),
                "x": float(np.round(tile_position_zyx_um[1], 2)),
            }

        input_path = _local_registered_fiducial_path(
            datastore=datastore,
            tile_id=tile_id,
            round_id=round_id,
        )
        sim = _read_registered_fiducial_sim(
            input_path=input_path,
            scale=scale,
            translation=tile_grid_positions,
            affine_zyx_px=affine_zyx_px,
            ngff_utils=ngff_utils,
            si_utils=si_utils,
        )

        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msims.append(msim)
        gc.collect()

    # perform registration in three steps, from most downsampling to least.
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        with dask.diagnostics.ProgressBar():
            _ = registration.register(
                msims,
                reg_channel_index=0,
                transform_key="stage_metadata",
                new_transform_key="affine_registered",
                registration_binning={"z": 3, "y": 6, "x": 6},
                post_registration_do_quality_filter=True,
                n_parallel_pairwise_regs=20,
            )

    # extract and save transformations into datastore
    for tile_idx, msim in enumerate(msims):
        affine = msi_utils.get_transform_from_msim(
            msim, transform_key="affine_registered"
        ).data.squeeze()
        affine = np.round(affine, 2)
        origin = si_utils.get_origin_from_sim(
            msi_utils.get_sim_from_msim(msim), asarray=True
        )
        spacing = si_utils.get_spacing_from_sim(
            msi_utils.get_sim_from_msim(msim), asarray=True
        )

        datastore.save_global_coord_xforms_um(
            affine_zyx_um=affine,
            origin_zyx_um=origin,
            spacing_zyx_um=spacing,
            tile=tile_idx,
        )

    # perform and save downsampled fusion

    output_zarr_path = datastore._image_store_path(
        datastore._fused_root_path
        / Path(f"fused_{datastore.fiducial_folder_name}_iso_zyx")
    )
    if output_zarr_path.exists():
        shutil.rmtree(output_zarr_path)

    fused_sim = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim, scale="scale0") for msim in msims],
        transform_key="affine_registered",
        output_spacing={
            "z": voxel_zyx_um[0],
            "y": voxel_zyx_um[1] * np.round(voxel_zyx_um[0] / voxel_zyx_um[1], 1),
            "x": voxel_zyx_um[2] * np.round(voxel_zyx_um[0] / voxel_zyx_um[2], 1),
        },
        output_chunksize=fused_chunk_size,
        output_zarr_url=str(output_zarr_path),
        zarr_options={
            "ome_zarr": True,
            "ngff_version": ngff_version,
            "overwrite": True,
        },
        batch_options=_get_batch_processing_options(
            misc_utils,
            n_jobs=n_jobs,
            use_gpu_fusion=use_gpu_fusion,
        ),
        **_get_fusion_backend_kwargs(
            fusion.fuse,
            use_gpu_fusion=use_gpu_fusion,
        ),
    )

    fused_sim = _get_scale0_sim_from_fusion_result(fused_sim, msi_utils=msi_utils)
    fused_msim = msi_utils.get_msim_from_sim(fused_sim, scale_factors=[])
    affine = msi_utils.get_transform_from_msim(
        fused_msim, transform_key="affine_registered"
    ).data.squeeze()
    origin = si_utils.get_origin_from_sim(
        msi_utils.get_sim_from_msim(fused_msim), asarray=True
    )
    spacing = si_utils.get_spacing_from_sim(
        msi_utils.get_sim_from_msim(fused_msim), asarray=True
    )

    del fused_msim

    qi2labDataStore._write_extra_attributes(
        image_path=output_zarr_path,
        extra_attributes={
            "affine_zyx_um": np.asarray(affine, dtype=np.float32).tolist(),
            "origin_zyx_um": np.asarray(origin, dtype=np.float32).tolist(),
            "spacing_zyx_um": np.asarray(spacing, dtype=np.float32).tolist(),
        },
        merge=True,
    )

    del fused_sim
    gc.collect()

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"GlobalRegistered": True})
    datastore_state.update({"Fused": True})
    datastore.datastore_state = datastore_state

    # write max projection OME-TIFF for cellpose GUI
    if create_max_proj_tiff:
        # load downsampled, fused fiducial image and coordinates
        fiducial_fused, _, _, spacing_zyx_um = datastore.load_global_fidicual_image(
            return_future=False
        )

        # create max projection
        fiducial_max_projection = np.max(np.squeeze(fiducial_fused), axis=0)
        del fiducial_fused

        filename = "fiducial_max_projection.ome.tiff"
        cellpose_path = (
            datastore._datastore_path / Path("segmentation") / Path("cellpose")
        )
        cellpose_path.mkdir(exist_ok=True)
        filename_path = (
            datastore._datastore_path
            / Path("segmentation")
            / Path("cellpose")
            / Path(filename)
        )
        with TiffWriter(filename_path, bigtiff=True) as tif:
            metadata = {
                "axes": "YX",
                "SignificantBits": 16,
                "PhysicalSizeX": float(spacing_zyx_um[2]),
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": float(spacing_zyx_um[1]),
                "PhysicalSizeYUnit": "µm",
            }
            options = {
                "compression": "zlib",
                "compressionargs": {"level": 8},
                "predictor": True,
                "photometric": "minisblack",
                "resolutionunit": "CENTIMETER",
            }
            tif.write(
                fiducial_max_projection,
                resolution=(
                    1e4 / float(spacing_zyx_um[2]),
                    1e4 / float(spacing_zyx_um[1]),
                ),
                **options,
                metadata=metadata,
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
