"""
Register qi2lab 3D MERFISH data using multiview-stitcher and optical flow.

This module registers fiducial rounds to the first fiducial round, applies the
saved physical-space transforms to readout data and U-FISH predictions, and can
optionally apply the legacy optical-flow deformation field after affine
registration.

History:
--------
- **2025/07**:
    - Implement anistropic downsampling for registration.
    - Implement RLGC deconvolution.
    - Implement new GPU based pixel-warping strategy using warpfield
    - Implement multi-GPU processing.
- **2024/12**: Refactor repo structure.
- **2024/08**:
    - Switched to qi2labdatastore for data access.
    - Implemented numba-accelerated downsampling.
    - Cleaned numpy usage for ryomen tiling.
- **2024/07**: Integrated pycudadecon and removed Dask usage.
- **2024/04**: Updated for U-FISH, removed SPOTS3D.
- **2024/01**: Adjusted for qi2lab MERFISH file format v0.1.
- **2023/09**: Initial commit.
"""

import multiprocessing as mp

mp.set_start_method("spawn", force=True)
import os
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*cupyx\.jit\.rawkernel is experimental.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*block stride.*last level.*"
)

import builtins
import gc
import queue
import timeit
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from merfish3danalysis.qi2labDataStore import qi2labDataStore

UFISH_MODEL_ALIASES = {
    "merfish": "finetune_models/v1.0.1-MERFISH_model.onnx",
    "seqfish": "finetune_models/v1.0.1-seqFISH_model.onnx",
    "simfish": "finetune_models/v1.0.1-simfish_model.onnx",
    "smfish": "finetune_models/v1.0.1-simfish_model.onnx",
    "deepspot": "finetune_models/v1.0.1-deepspot_model.onnx",
    "exseq": "finetune_models/v1.0.1-ExSeq_model.onnx",
}
DEFAULT_UFISH_MODEL = "simfish"


def _registration_diagnostics_enabled() -> bool:
    """
    Return whether registration diagnostics should be printed.

    Returns
    -------
    bool
        True when ``MERFISH3D_REGISTRATION_DIAGNOSTICS`` is set to a truthy
        value.
    """

    return os.environ.get("MERFISH3D_REGISTRATION_DIAGNOSTICS", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _registration_diag(message: str) -> None:
    """
    Print one timestamped registration diagnostic message when enabled.

    Parameters
    ----------
    message : str
        Diagnostic message body.

    Returns
    -------
    None
        The message is printed only when diagnostics are enabled.
    """

    if _registration_diagnostics_enabled():
        print(time_stamp(), f"[registration-diagnostics] {message}", flush=True)


def _resolve_ufish_weights_path(model: str | Path | None) -> Path | str | None:
    """
    Resolve a U-FISH model alias or path without requiring U-FISH imports.

    Parameters
    ----------
    model : str | Path | None
        U-FISH model alias, weights filename, local path, or None to use the
        default model.

    Returns
    -------
    Path | str | None
        Existing local path when one is found; otherwise a U-FISH weights file
        name accepted by ``UFish.load_weights``.
    """

    if model is None:
        model = DEFAULT_UFISH_MODEL

    model_str = str(model).strip()
    if not model_str:
        model_str = DEFAULT_UFISH_MODEL

    model_path = Path(model_str).expanduser()
    if model_path.exists():
        return model_path

    weights_file = UFISH_MODEL_ALIASES.get(model_str.lower(), model_str)
    if weights_file is None:
        return None

    local_path = Path.home() / ".ufish" / weights_file
    if local_path.exists():
        return local_path

    return weights_file


def _load_ufish_model(ufish: Any, model: str | Path | None = None) -> None:
    """
    Load configured U-FISH weights from an alias, local path, or weights file.

    Parameters
    ----------
    ufish : Any
        U-FISH instance.
    model : str | Path | None
        U-FISH model alias, weights filename, local path, or None to use the
        default model.

    Returns
    -------
    None
        The weights are loaded into ``ufish`` in place.
    """

    weights = _resolve_ufish_weights_path(model)
    if weights is None:
        raise ValueError("Resolved U-FISH weights cannot be None.")

    if isinstance(weights, Path):
        ufish.load_weights_from_path(weights)
    else:
        ufish.load_weights(weights_file=weights)


def _resolve_psf(psfs: Any, psf_idx: int) -> np.ndarray:
    """
    Fetch PSF by index from uniform or ragged channel PSF storage.

    Parameters
    ----------
    psfs : Any
        PSF container. This may be a list of arrays or a stacked numpy-like
        array whose first axis indexes channels.
    psf_idx : int
        Channel PSF index to return.

    Returns
    -------
    numpy.ndarray
        Selected PSF as a float32 array.
    """

    if isinstance(psfs, list):
        if psf_idx < 0 or psf_idx >= len(psfs):
            raise IndexError(f"PSF index {psf_idx} out of range for {len(psfs)} PSFs.")
        return np.asarray(psfs[psf_idx], dtype=np.float32)

    psf_array = np.asarray(psfs)
    if psf_array.ndim < 3:
        raise ValueError(f"Invalid PSF array shape: {psf_array.shape}")
    if psf_idx < 0 or psf_idx >= psf_array.shape[0]:
        raise IndexError(
            f"PSF index {psf_idx} out of range for shape {psf_array.shape}."
        )
    return np.asarray(psf_array[psf_idx], dtype=np.float32)


def _run_chunked_rlgc_remembering_crop(
    dr: Any,
    chunked_rlgc: Any,
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int,
    release_memory: bool = False,
) -> np.ndarray:
    """
    Run RLGC and remember the lateral chunk size that succeeds in this worker.

    Parameters
    ----------
    dr : Any
        DataRegistration-like object containing the current cached
        ``_crop_yx_decon`` value.
    chunked_rlgc : Any
        RLGC callable. It must accept ``on_successful_crop_yx``.
    image : np.ndarray
        Image to deconvolve.
    psf : np.ndarray
        PSF to use for deconvolution.
    gpu_id : int
        CUDA device index.
    release_memory : bool, default=False
        If True, ask RLGC to release its CuPy memory pools before returning.

    Returns
    -------
    numpy.ndarray
        Deconvolved image returned by ``chunked_rlgc``.
    """

    image_arr = np.asarray(image)
    if image_arr.ndim == 2:
        image_max_yx = max(image_arr.shape)
    else:
        image_max_yx = max(image_arr.shape[-2:])

    def _remember_successful_crop(successful_crop_yx: int) -> None:
        """
        Cache the successful RLGC crop size on the registration object.

        Parameters
        ----------
        successful_crop_yx : int
            Lateral crop size that completed without GPU memory fallback.

        Returns
        -------
        None
            The crop size is stored on ``dr``.
        """
        previous_crop_yx = int(dr._crop_yx_decon)
        requested_crop_yx = min(previous_crop_yx, int(image_max_yx))
        if int(successful_crop_yx) >= requested_crop_yx:
            return

        dr._crop_yx_decon = int(successful_crop_yx)
        if dr._verbose >= 1:
            print(
                time_stamp(),
                "RLGC reduced crop_yx after GPU memory fallback: "
                f"{previous_crop_yx} -> {successful_crop_yx}.",
            )

    return chunked_rlgc(
        image=image,
        psf=psf,
        gpu_id=gpu_id,
        crop_yx=dr._crop_yx_decon,
        crop_z=None,
        release_memory=release_memory,
        on_successful_crop_yx=_remember_successful_crop,
    )


def _deconvolve_fiducials_on_gpu(
    dr,  # noqa: ANN001
    round_list: list,
    gpu_id: int,
    result_queue: Any,
) -> None:
    """
    Deconvolve fiducial rounds on one GPU and send results to the parent process.

    Parameters
    ----------
    dr : DataRegistration
        Registration object pickled into the worker process. The worker uses its
        datastore, PSF list, deconvolution settings, and current tile id.
    round_list : list
        Fiducial round identifiers assigned to this worker.
    gpu_id : int
        CUDA device index visible within the worker process.
    result_queue : multiprocessing.Queue
        Queue used to send ``("result", round_id, image, crop_yx)`` messages
        back to the parent. On failure the worker sends an ``"error"`` message
        containing the formatted traceback.

    Returns
    -------
    None
        Results are returned through ``result_queue``.
    """

    import cupy as cp
    import torch

    try:
        torch.cuda.set_device(gpu_id)
        cp.cuda.Device(gpu_id).use()

        from merfish3danalysis.utils.rlgc import chunked_rlgc, clear_rlgc_caches

        for round_id in round_list:
            raw = dr._datastore.load_local_corrected_image(
                tile=dr._tile_id, round=round_id, return_future=False
            )
            _registration_diag(
                "fiducial_decon_start "
                f"tile={dr._tile_id} round={round_id} "
                f"raw_shape={tuple(int(v) for v in raw.shape)}"
            )
            start_time = timeit.default_timer()
            if dr._decon_fiducial:
                decon = _run_chunked_rlgc_remembering_crop(
                    dr=dr,
                    chunked_rlgc=chunked_rlgc,
                    image=raw,
                    psf=_resolve_psf(dr._psfs, 0),
                    gpu_id=gpu_id,
                )
                decon = decon.clip(0, 2**16 - 1).astype(np.uint16)
            else:
                decon = raw.copy().astype(np.uint16)
            _registration_diag(
                "fiducial_decon_done "
                f"tile={dr._tile_id} round={round_id} "
                f"input_shape={tuple(int(v) for v in raw.shape)} "
                f"output_shape={tuple(int(v) for v in decon.shape)} "
                f"elapsed_s={timeit.default_timer() - start_time:.2f}"
            )
            del raw
            result_queue.put(("result", round_id, decon, int(dr._crop_yx_decon)))
            del decon
            gc.collect()

        clear_rlgc_caches(clear_memory_pool=False)
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        result_queue.put(("error", None, traceback.format_exc(), None))
        raise


def _register_fiducial_transform_worker(
    dr,  # noqa: ANN001
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    round_id: str,
) -> None:
    """
    Register one fiducial round to the reference round and save its transform.

    Parameters
    ----------
    dr : DataRegistration
        Registration object pickled into the worker process. The worker uses the
        datastore to read voxel spacing and save the transform.
    fixed_image : numpy.ndarray
        Reference fiducial image in Z, Y, X order. This is normally the first
        fiducial round for the current tile.
    moving_image : numpy.ndarray
        Deconvolved fiducial image to align to ``fixed_image``.
    round_id : str
        Fiducial round identifier for ``moving_image``.

    Returns
    -------
    None
        The affine transform is saved to the datastore.
    """

    try:
        from merfish3danalysis.utils.multiview_registration import (
            register_pair_to_fixed,
        )

        spacing_zyx_um = dr._datastore.voxel_size_zyx_um
        _registration_diag(
            "fiducial_elastix_registration_start "
            f"tile={dr._tile_id} round={round_id} "
            f"fixed_shape={tuple(int(v) for v in fixed_image.shape)} "
            f"moving_shape={tuple(int(v) for v in moving_image.shape)} "
            f"spacing_zyx_um={tuple(float(v) for v in spacing_zyx_um)}"
        )
        start_time = timeit.default_timer()
        local_transform_zyx_um = register_pair_to_fixed(
            fixed_image.astype(np.float32, copy=False),
            moving_image.astype(np.float32, copy=False),
            spacing_zyx_um=spacing_zyx_um,
        )
        dr._datastore.save_local_round_transform_zyx_um(
            transform_zyx_um=local_transform_zyx_um,
            tile=dr._tile_id,
            round=round_id,
        )
        _registration_diag(
            "fiducial_elastix_registration_done "
            f"tile={dr._tile_id} round={round_id} "
            f"elapsed_s={timeit.default_timer() - start_time:.2f}"
        )
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise


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


def _read_registered_fiducial_sim(
    input_path: Path,
    scale: dict[str, float],
    translation: dict[str, float],
    affine_zyx_px: Any,
    ngff_utils: Any,
    si_utils: Any,
) -> Any:
    """
    Read one registered fiducial OME-Zarr as a SpatialImage.

    Parameters
    ----------
    input_path : pathlib.Path
        Local registered fiducial OME-Zarr path.
    scale : dict[str, float]
        Physical pixel spacing by spatial dimension.
    translation : dict[str, float]
        Stage-derived physical origin by spatial dimension.
    affine_zyx_px : Any
        Camera-to-stage affine transform loaded from datastore metadata.
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


def _apply_bits_on_gpu(dr, bit_list: list, gpu_id: int = 0) -> bool:  # noqa: ANN001
    """
    Deconvolve readout bits, run U-FISH, apply saved transforms, and save outputs.

    Parameters
    ----------
    dr : DataRegistration
        DataRegistration instance pickled into this process.
    bit_list : list
        Bit identifiers to process on this GPU.
    gpu_id : int
        CUDA device index visible within this worker process.

    Returns
    -------
    bool
        True after all assigned bits are processed.
    """

    import cupy as cp
    import torch

    torch.cuda.set_device(gpu_id)
    cp.cuda.Device(gpu_id).use()

    import os

    from warpfield.warp import warp_volume

    os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    from ufish.api import UFish

    from merfish3danalysis.utils.multiview_registration import (
        transform_points_to_reference,
        warp_array_to_reference,
    )
    from merfish3danalysis.utils.rlgc import chunked_rlgc, clear_rlgc_caches

    spacing_zyx_um = dr._datastore.voxel_size_zyx_um
    for bit_id in bit_list:
        r_idx = dr._datastore.load_local_round_linker(tile=dr._tile_id, bit=bit_id) - 1
        ex_wl, _em_wl = dr._datastore.load_local_wavelengths_um(
            tile=dr._tile_id, bit=bit_id
        )
        if ex_wl < 0.600:
            psf_idx = 1
        else:
            psf_idx = 2

        reg_on_disk = dr._has_valid_registered_image(bit_id=bit_id)
        feature_predictor_on_disk = dr._has_valid_feature_predictor_outputs(
            bit_id=bit_id
        )

        if reg_on_disk and feature_predictor_on_disk and not dr._overwrite_registered:
            continue

        if (
            (not reg_on_disk)
            or (not feature_predictor_on_disk)
            or dr._overwrite_registered
        ):
            # load data
            corrected_image = dr._datastore.load_local_corrected_image(
                tile=dr._tile_id, bit=bit_id, return_future=False
            )
            _registration_diag(
                "bit_start "
                f"tile={dr._tile_id} bit={bit_id} linked_round={dr._round_ids[r_idx]} "
                f"raw_shape={tuple(int(v) for v in corrected_image.shape)} "
                f"spacing_zyx_um={tuple(float(v) for v in spacing_zyx_um)}"
            )

            # deconvolution
            if dr._decon_readout:
                start_time = timeit.default_timer()
                decon_image = _run_chunked_rlgc_remembering_crop(
                    dr=dr,
                    chunked_rlgc=chunked_rlgc,
                    image=corrected_image,
                    psf=_resolve_psf(dr._psfs, psf_idx),
                    gpu_id=gpu_id,
                    release_memory=True,
                )
                decon_image = decon_image.clip(0, 2**16 - 1).astype(np.uint16)
                _registration_diag(
                    "bit_decon "
                    f"tile={dr._tile_id} bit={bit_id} "
                    f"input_shape={tuple(int(v) for v in corrected_image.shape)} "
                    f"output_shape={tuple(int(v) for v in decon_image.shape)} "
                    f"elapsed_s={timeit.default_timer() - start_time:.2f}"
                )
            else:
                decon_image = corrected_image.copy()

            ufish = UFish(device=f"cuda:{gpu_id}")
            _load_ufish_model(ufish, dr._ufish_model)
            start_time = timeit.default_timer()
            feature_predictor_loc, feature_predictor_data = ufish.predict(
                decon_image, axes="zyx", blend_3d=False, batch_size=1
            )
            _registration_diag(
                "bit_ufish "
                f"tile={dr._tile_id} bit={bit_id} "
                f"image_shape={tuple(int(v) for v in decon_image.shape)} "
                f"spots={len(feature_predictor_loc)} "
                f"elapsed_s={timeit.default_timer() - start_time:.2f}"
            )
            feature_predictor_loc = feature_predictor_loc.rename(
                columns={"axis-0": "z", "axis-1": "y", "axis-2": "x"}
            )
            del ufish
            torch.cuda.empty_cache()
            gc.collect()

            if r_idx > 0:
                local_transform_zyx_um = (
                    dr._datastore.load_local_round_transform_zyx_um(
                        tile=dr._tile_id, round=dr._round_ids[r_idx]
                    )
                )
                if local_transform_zyx_um is None:
                    raise RuntimeError(
                        f"Missing local round transform for tile={dr._tile_id} "
                        f"round={dr._round_ids[r_idx]}."
                    )
                start_time = timeit.default_timer()
                data_reg = warp_array_to_reference(
                    decon_image,
                    transform_zyx_um=local_transform_zyx_um,
                    spacing_zyx_um=spacing_zyx_um,
                    reference_shape=decon_image.shape,
                )
                feature_predictor_data = warp_array_to_reference(
                    feature_predictor_data,
                    transform_zyx_um=local_transform_zyx_um,
                    spacing_zyx_um=spacing_zyx_um,
                    reference_shape=decon_image.shape,
                )
                registered_points_zyx = transform_points_to_reference(
                    feature_predictor_loc[["z", "y", "x"]].to_numpy(),
                    transform_zyx_um=local_transform_zyx_um,
                    spacing_zyx_um=spacing_zyx_um,
                )
                feature_predictor_loc[["z", "y", "x"]] = registered_points_zyx
                _registration_diag(
                    "bit_transform_and_warp "
                    f"tile={dr._tile_id} bit={bit_id} "
                    f"round={dr._round_ids[r_idx]} "
                    f"image_shape={tuple(int(v) for v in decon_image.shape)} "
                    f"feature_shape={tuple(int(v) for v in feature_predictor_data.shape)} "
                    f"spots={len(feature_predictor_loc)} "
                    f"elapsed_s={timeit.default_timer() - start_time:.2f}"
                )
                del decon_image

                if dr._perform_optical_flow:
                    warp_field, block_size, block_stride = (
                        dr._datastore.load_coord_of_xform_px(
                            tile=dr._tile_id,
                            round=dr._round_ids[r_idx],
                            return_future=False,
                        )
                    )

                    block_size = cp.asarray(block_size, dtype=cp.float32)
                    block_stride = cp.asarray(block_stride, dtype=cp.float32)
                    decon_image_warped_cp = warp_volume(
                        data_reg,
                        warp_field,
                        block_stride,
                        cp.array(-block_size / block_stride / 2),
                        out=None,
                        gpu_id=gpu_id,
                    )
                    data_reg = cp.asnumpy(decon_image_warped_cp).astype(np.float32)
                    del decon_image_warped_cp
                    feature_predictor_data_cp = warp_volume(
                        feature_predictor_data,
                        warp_field,
                        block_stride,
                        cp.array(-block_size / block_stride / 2),
                        out=None,
                        gpu_id=gpu_id,
                    )
                    feature_predictor_data = cp.asnumpy(feature_predictor_data_cp)
                    del feature_predictor_data_cp
                gc.collect()
            else:
                data_reg = decon_image.copy()
                del decon_image
                gc.collect()

            # clip to uint16
            data_reg = data_reg.clip(0, 2**16 - 1).astype(np.uint16)

            # save registered readout immediately so overwrite does not depend on U-FISH
            dr._datastore.save_local_registered_image(
                data_reg,
                tile=dr._tile_id,
                deconvolution=dr._decon_readout,
                bit=bit_id,
            )

            # feature_predictor ROI sums
            roi_z, roi_y, roi_x = 7, 5, 5

            def sum_pixels_in_roi(row, image, roi_dims):  # noqa
                """
                Sum image intensities in a fixed-size ROI centered on one spot.

                Parameters
                ----------
                row : Any
                    Feature predictor spot row containing ``z``, ``y``, and
                    ``x`` coordinates.
                image : Any
                    Image from which to sample intensities.
                roi_dims : Any
                    ROI shape in Z, Y, X order.

                Returns
                -------
                float
                    Sum of pixel intensities inside the clipped ROI.
                """
                z, y, x = row["z"], row["y"], row["x"]
                rz, ry, rx = roi_dims
                zmin = max(0, z - rz // 2)
                ymin = max(0, y - ry // 2)
                xmin = max(0, x - rx // 2)
                zmax = min(image.shape[0], zmin + rz)
                ymax = min(image.shape[1], ymin + ry)
                xmax = min(image.shape[2], xmin + rx)
                roi = image[
                    int(zmin) : int(zmax), int(ymin) : int(ymax), int(xmin) : int(xmax)
                ]
                return np.sum(roi)

            feature_predictor_loc["sum_prob_pixels"] = feature_predictor_loc.apply(
                sum_pixels_in_roi,
                axis=1,
                image=feature_predictor_data,
                roi_dims=(roi_z, roi_y, roi_x),
            )
            feature_predictor_loc["sum_decon_pixels"] = feature_predictor_loc.apply(
                sum_pixels_in_roi,
                axis=1,
                image=data_reg,
                roi_dims=(roi_z, roi_y, roi_x),
            )

            feature_predictor_loc["tile_idx"] = dr._tile_ids.index(dr._tile_id)
            feature_predictor_loc["bit_idx"] = dr._bit_ids.index(bit_id) + 1
            feature_predictor_loc["tile_z_px"] = feature_predictor_loc["z"]
            feature_predictor_loc["tile_y_px"] = feature_predictor_loc["y"]
            feature_predictor_loc["tile_x_px"] = feature_predictor_loc["x"]

            # save results
            dr._datastore.save_local_feature_predictor_image(
                feature_predictor_data, tile=dr._tile_id, bit=bit_id
            )
            dr._datastore.save_local_feature_predictor_spots(
                feature_predictor_loc, tile=dr._tile_id, bit=bit_id
            )
            if dr._verbose >= 1:
                print(
                    time_stamp(),
                    f"Finished readout tile id: {dr._tile_id}; bit id: {bit_id}.",
                )

            del data_reg, feature_predictor_data, feature_predictor_loc
            gc.collect()

    try:
        clear_rlgc_caches(clear_memory_pool=False)
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    return True


class DataRegistration:
    """Register 2D or 3D MERFISH data across rounds.

    Parameters
    ----------
    datastore : qi2labDataStore
        Initialized qi2labDataStore object
    decon_fiducial: bool, default True
        Deconvolve ALL fiducial rounds. False = only deconvolve round 1 for downstream
        stitching.
    decon_readout: bool, default False
        Deconvolve readout images before registration to fiducials.
    overwrite_registered: bool, default False
        Overwrite existing registered data and registrations
    perform_optical_flow: bool, default False
        Perform optical flow registration
    save_all_fiducial_registered: bool, default True
        Save registered fiducial rounds > 1. These are not used for analysis.
    num_gpus: int, default 1
        Number of GPUs to use for registration.
    crop_yx_decon: int, default 2048
        Crop size for deconvolution applied to both y and x dimensions.
    ufish_model: str or pathlib.Path or None, default None
        U-FISH model to use for feature prediction. If omitted or ``None``, use
        the package default model, ``simfish``. Known aliases include
        ``simfish``, ``smfish``, ``merfish``, ``seqfish``, ``deepspot``, and
        ``exseq``. A local ``.onnx``/``.pth`` path or HuggingFace weights
        filename can also be supplied.
    verbose : int, default 1
        Progress verbosity. Set to 0 to suppress routine progress prints.
    """

    def __init__(
        self,
        datastore: qi2labDataStore,
        decon_fiducial: bool = True,
        decon_readout: bool = False,
        overwrite_registered: bool = False,
        perform_optical_flow: bool = True,
        save_all_fiducial_registered: bool = False,
        num_gpus: int = 1,
        crop_yx_decon: int = 2048,
        ufish_model: str | Path | None = None,
        global_registration: bool = False,
        verbose: int = 1,
    ) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        datastore : qi2labDataStore
            Datastore containing corrected images, PSFs, metadata, and output
            groups.
        decon_fiducial : bool
            If True, deconvolve fiducial images before registration.
        decon_readout : bool
            If True, deconvolve readout images before U-FISH prediction and
            warping.
        overwrite_registered : bool
            If True, regenerate registered images and feature predictor outputs
            even when they already exist.
        perform_optical_flow : bool
            If True, apply the legacy optical-flow deformation after affine
            registration.
        save_all_fiducial_registered : bool
            If True, save warped fiducial image volumes for all rounds. The
            first fiducial round is always saved.
        num_gpus : int
            Number of GPUs available for deconvolution and readout processing.
        crop_yx_decon : int
            Initial lateral RLGC crop size. GPU memory fallback can reduce this
            value and cache the successful size.
        ufish_model : str | Path | None
            U-FISH model alias, weights filename, local path, or None for the
            default model.
        global_registration : bool, default=False
            If True, run global tile registration and fused fiducial OME-Zarr
            generation after local tile preprocessing.
        verbose : int
            Verbosity level for progress messages.
        """
        self._datastore = datastore
        self._decon_fiducial = decon_fiducial
        self._tile_ids = self._datastore.tile_ids
        self._round_ids = self._datastore.round_ids
        self._bit_ids = self._datastore.bit_ids
        self._psfs = self._datastore.channel_psfs
        self._num_gpus = num_gpus
        self._crop_yx_decon = crop_yx_decon
        self._perform_optical_flow = perform_optical_flow
        self._data_raw = None
        self._has_registered_data = None
        self._overwrite_registered = overwrite_registered
        self.save_all_fiducial_registered = save_all_fiducial_registered
        self._decon_readout = decon_readout
        self._ufish_model = ufish_model
        self._global_registration = global_registration
        self._original_print = builtins.print
        self._verbose = verbose

    # -----------------------------------
    # property access for class variables
    # -----------------------------------
    @property
    def datastore(self) -> qi2labDataStore:
        """Return the qi2labDataStore object.

        Returns
        -------
        qi2labDataStore
            qi2labDataStore object
        """

        if self._dataset_path is not None:
            return self._datastore
        else:
            print("Datastore not defined.")
            return None

    @datastore.setter
    def dataset_path(self, value: qi2labDataStore) -> None:
        """Set the qi2labDataStore object.

        Parameters
        ----------
        value : qi2labDataStore
            qi2labDataStore object
        """

        del self._datastore
        self._datastore = value

    @property
    def tile_id(self) -> str:
        """Get the current tile id.

        Returns
        -------
        tile_id: Union[int,str]
            Tile id
        """

        if self._tile_id is not None:
            tile_id = self._tile_id
            return tile_id
        else:
            print("Tile coordinate not defined.")
            return None

    @tile_id.setter
    def tile_id(self, value: int | str) -> None:
        """Set the tile id.

        Parameters
        ----------
        value : Union[int,str]
            Tile id
        """

        if isinstance(value, int):
            if value < 0 or value > self._datastore.num_tiles:
                print("Set value index >=0 and <=" + str(self._datastore.num_tiles))
                return None
            else:
                self._tile_id = self._datastore.tile_ids[value]
        elif isinstance(value, str):
            if value not in self._datastore.tile_ids:
                print("set valid tile id")
                return None
            else:
                self._tile_id = value

    @property
    def perform_optical_flow(self) -> bool:
        """Get the perform_optical_flow flag.

        Returns
        -------
        perform_optical_flow: bool
            Perform optical flow registration
        """

        return self._perform_optical_flow

    @perform_optical_flow.setter
    def perform_optical_flow(self, value: bool) -> None:
        """Set the perform_optical_flow flag.

        Parameters
        ----------
        value : bool
            Perform optical flow registration
        """

        self._perform_optical_flow = value

    @property
    def overwrite_registered(self) -> bool:
        """Get the overwrite_registered flag.

        Returns
        -------
        overwrite_registered: bool
            Overwrite existing registered data and registrations
        """

        return self._overwrite_registered

    @overwrite_registered.setter
    def overwrite_registered(self, value: bool) -> None:
        """Set the overwrite_registered flag.

        Parameters
        ----------
        value : bool
            Overwrite existing registered data and registrations
        """

        self._overwrite_registered = value

    def _entity_root(
        self,
        tile_id: str,
        round_id: str | None = None,
        bit_id: str | None = None,
    ) -> Path:
        """
        Entity root.

        Parameters
        ----------
        tile_id : str
            Function argument.
        round_id : str | None
            Function argument.
        bit_id : str | None
            Function argument.

        Returns
        -------
        Path
            Function result.
        """
        if (round_id is None and bit_id is None) or (
            round_id is not None and bit_id is not None
        ):
            raise ValueError("Provide either round_id or bit_id, but not both.")

        if round_id is not None:
            return self._datastore._fiducial_root_path / Path(tile_id) / Path(round_id)
        return self._datastore._readouts_root_path / Path(tile_id) / Path(bit_id)

    def _has_valid_registered_image(
        self,
        tile_id: str | None = None,
        round_id: str | None = None,
        bit_id: str | None = None,
    ) -> bool:
        """
        Has valid registered image.

        Parameters
        ----------
        tile_id : str | None
            Function argument.
        round_id : str | None
            Function argument.
        bit_id : str | None
            Function argument.

        Returns
        -------
        bool
            Function result.
        """
        tile_id = self._tile_id if tile_id is None else tile_id
        entity_root = self._entity_root(
            tile_id=tile_id, round_id=round_id, bit_id=bit_id
        )
        corrected_shape = self._datastore._image_shape(
            entity_root / Path("corrected_data")
        )
        registered_shape = self._datastore._image_shape(
            entity_root / Path("registered_decon_data")
        )
        return corrected_shape is not None and registered_shape == corrected_shape

    def _has_valid_feature_predictor_outputs(
        self,
        tile_id: str | None = None,
        bit_id: str | None = None,
    ) -> bool:
        """
        Has valid feature predictor outputs.

        Parameters
        ----------
        tile_id : str | None
            Function argument.
        bit_id : str | None
            Function argument.

        Returns
        -------
        bool
            Function result.
        """
        tile_id = self._tile_id if tile_id is None else tile_id
        if bit_id is None:
            raise ValueError("bit_id is required for feature predictor outputs.")

        entity_root = self._entity_root(tile_id=tile_id, bit_id=bit_id)
        corrected_shape = self._datastore._image_shape(
            entity_root / Path("corrected_data")
        )
        registered_shape = self._datastore._image_shape(
            entity_root / Path("registered_decon_data")
        )
        feature_shape = self._datastore._image_shape(
            entity_root
            / Path(f"registered_{self._datastore.feature_predictor_folder_name}_data")
        )
        spots_path = (
            self._datastore._feature_predictor_localizations_root_path
            / Path(tile_id)
            / Path(bit_id + ".parquet")
        )
        return (
            corrected_shape is not None
            and registered_shape == corrected_shape
            and feature_shape == corrected_shape
            and spots_path.exists()
        )

    def _is_tile_complete(self, tile_id: str) -> bool:
        """
        Is tile complete.

        Parameters
        ----------
        tile_id : str
            Function argument.

        Returns
        -------
        bool
            Function result.
        """
        if not self._has_valid_registered_image(
            tile_id=tile_id, round_id=self._round_ids[0]
        ):
            return False

        if self.save_all_fiducial_registered:
            for round_id in self._round_ids[1:]:
                if not self._has_valid_registered_image(
                    tile_id=tile_id, round_id=round_id
                ):
                    return False

        for bit_id in self._bit_ids:
            if not self._has_valid_feature_predictor_outputs(
                tile_id=tile_id, bit_id=bit_id
            ):
                return False

        return True

    def register_all_tiles(self) -> None:
        """
        Helper function to register all tiles.

        Returns
        -------
        None
            Function result.
        """
        tile_ids = list(self._datastore.tile_ids)
        start_idx = 0
        if not self._overwrite_registered:
            for idx, tile_id in enumerate(tile_ids):
                if self._is_tile_complete(tile_id):
                    start_idx = idx + 1
                    continue
                start_idx = idx
                break
            else:
                start_idx = len(tile_ids)

            if start_idx >= len(tile_ids):
                if self._verbose >= 1:
                    print(
                        time_stamp(),
                        "All tiles already have complete registered outputs.",
                    )
                return

            if start_idx > 0:
                if self._verbose >= 1:
                    print(
                        time_stamp(),
                        f"Resuming local registration at tile id: {tile_ids[start_idx]}.",
                    )

        for tile_id in tile_ids[start_idx:]:
            self.tile_id = tile_id
            self._generate_registrations()
            self._apply_registration_to_bits()

        if self._global_registration:
            self.global_register()

    def register_one_tile(self, tile_id: int | str) -> None:
        """Helper function to register one tile.

        Parameters
        ----------
        tile_id : Union[int,str]
            Tile id
        """

        self.tile_id = tile_id
        self._generate_registrations()
        self._apply_registration_to_bits()

    def apply_registration_to_one_tile(self, tile_id: int | str) -> None:
        """Apply existing local registrations to readout bits for one tile.

        This uses the rigid and optical-flow transforms already stored in the
        datastore. It does not estimate or overwrite fiducial registrations.
        """

        self.tile_id = tile_id
        self._apply_registration_to_bits()

    def global_register(
        self,
        create_max_proj_tiff: bool = True,
    ) -> None:
        """
        Globally register first-round fiducial tiles and write fused OME-Zarr.

        The method reads the locally registered first fiducial round for every
        tile, combines stage metadata with multiview-stitcher/ITK-Elastix
        global registration, saves per-tile global transforms back to the
        datastore, and fuses the registered views directly into the datastore as
        OME-Zarr v0.5 using CuPy-backed fusion.

        Parameters
        ----------
        create_max_proj_tiff : bool, default=True
            If True, write ``segmentation/cellpose/fiducial_max_projection.ome.tiff``
            from the full-resolution fused OME-Zarr.

        Returns
        -------
        None
            Global transforms, fused fiducial OME-Zarr, datastore state, and
            optional max projection are written to the datastore.
        """

        import shutil

        from dask import config as dask_config
        from multiview_stitcher import (
            fusion,
            misc_utils,
            msi_utils,
            ngff_utils,
            registration,
        )
        from multiview_stitcher import spatial_image_utils as si_utils
        from tifffile import TiffWriter

        from merfish3danalysis.utils.multiview_registration import (
            get_batch_processing_options,
            get_gpu_fusion_backend_kwargs,
            get_scale0_sim_from_fusion_result,
        )

        if len(self._tile_ids) <= 1:
            self._datastore.save_global_coord_xforms_um(
                affine_zyx_um=np.eye(4, dtype=np.float32),
                origin_zyx_um=np.zeros(3, dtype=np.float32),
                spacing_zyx_um=np.asarray(
                    self._datastore.voxel_size_zyx_um,
                    dtype=np.float32,
                ),
                tile=self._tile_ids[0],
            )
            self._datastore.datastore_state = {"GlobalRegistered": True}
            if self._verbose >= 1:
                print(
                    time_stamp(),
                    "Skipping global registration because datastore has one tile.",
                )
            return

        voxel_zyx_um = self._datastore.voxel_size_zyx_um
        scale = {
            "z": float(voxel_zyx_um[0]),
            "y": float(voxel_zyx_um[1]),
            "x": float(voxel_zyx_um[2]),
        }
        reference_round_id = self._round_ids[0]
        msims = []

        if self._verbose >= 1:
            print(time_stamp(), "Starting global fiducial registration.")

        for tile_id in self._tile_ids:
            tile_position_zyx_um, affine_zyx_px = (
                self._datastore.load_local_stage_position_zyx_um(
                    tile_id, reference_round_id
                )
            )
            tile_grid_positions = {
                "z": float(np.round(tile_position_zyx_um[0], 2)),
                "y": float(np.round(tile_position_zyx_um[1], 2)),
                "x": float(np.round(tile_position_zyx_um[2], 2)),
            }
            input_path = _local_registered_fiducial_path(
                datastore=self._datastore,
                tile_id=tile_id,
                round_id=reference_round_id,
            )
            sim = _read_registered_fiducial_sim(
                input_path=input_path,
                scale=scale,
                translation=tile_grid_positions,
                affine_zyx_px=affine_zyx_px,
                ngff_utils=ngff_utils,
                si_utils=si_utils,
            )
            msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
            gc.collect()

        with dask_config.set(scheduler="single-threaded"):
            global_transforms = registration.register(
                msims,
                reg_channel_index=0,
                transform_key="stage_metadata",
                new_transform_key="global_registered",
                pairwise_reg_func=registration.registration_ITKElastix,
                pairwise_reg_func_kwargs={
                    "transform_types": ["translation", "rigid", "affine"],
                },
                groupwise_resolution_kwargs={
                    "reference_view": 0,
                    "transform": "affine",
                },
                n_parallel_pairwise_regs=max(1, min(4, len(msims))),
            )

        for tile_idx, (msim, transform) in enumerate(
            zip(msims, global_transforms, strict=False)
        ):
            affine = np.asarray(
                transform.data if hasattr(transform, "data") else transform
            )
            affine = np.round(np.squeeze(affine), 2)
            sim = msi_utils.get_sim_from_msim(msim)
            origin = si_utils.get_origin_from_sim(sim, asarray=True)
            spacing = si_utils.get_spacing_from_sim(sim, asarray=True)
            self._datastore.save_global_coord_xforms_um(
                affine_zyx_um=affine,
                origin_zyx_um=origin,
                spacing_zyx_um=spacing,
                tile=tile_idx,
            )

        output_zarr_path = self._datastore._image_store_path(
            self._datastore._fused_root_path
            / Path(f"fused_{self._datastore.fiducial_folder_name}_zyx")
        )
        if output_zarr_path.exists():
            shutil.rmtree(output_zarr_path)

        fused_sim = fusion.fuse(
            images=msims,
            transform_key="global_registered",
            output_spacing=scale,
            output_zarr_url=str(output_zarr_path),
            zarr_options={
                "ome_zarr": True,
                "ngff_version": "0.5",
                "overwrite": True,
            },
            batch_options=get_batch_processing_options(
                misc_utils=misc_utils,
                n_batch=20,
                n_jobs=4,
            ),
            **get_gpu_fusion_backend_kwargs(fusion.fuse),
        )

        fused_sim = get_scale0_sim_from_fusion_result(fused_sim, msi_utils=msi_utils)
        fused_msim = msi_utils.get_msim_from_sim(fused_sim, scale_factors=[])
        affine = msi_utils.get_transform_from_msim(
            fused_msim, transform_key="global_registered"
        ).data.squeeze()
        fused_scale0 = msi_utils.get_sim_from_msim(fused_msim)
        origin = si_utils.get_origin_from_sim(fused_scale0, asarray=True)
        spacing = si_utils.get_spacing_from_sim(fused_scale0, asarray=True)

        qi2labDataStore._write_extra_attributes(
            image_path=output_zarr_path,
            extra_attributes={
                "affine_zyx_um": np.asarray(affine, dtype=np.float32).tolist(),
                "origin_zyx_um": np.asarray(origin, dtype=np.float32).tolist(),
                "spacing_zyx_um": np.asarray(spacing, dtype=np.float32).tolist(),
            },
            merge=True,
        )

        del fused_msim, fused_sim
        gc.collect()

        datastore_state = self._datastore.datastore_state
        datastore_state.update({"GlobalRegistered": True, "Fused": True})
        self._datastore.datastore_state = datastore_state

        if create_max_proj_tiff:
            loaded = self._datastore.load_global_fidicual_image(return_future=False)
            if loaded is None:
                raise RuntimeError(
                    "Fused fiducial image was not readable after fusion."
                )
            fiducial_fused, _, _, spacing_zyx_um = loaded
            fiducial_max_projection = np.max(np.squeeze(fiducial_fused), axis=0)
            del fiducial_fused

            cellpose_path = (
                self._datastore._datastore_path
                / Path("segmentation")
                / Path("cellpose")
            )
            cellpose_path.mkdir(exist_ok=True)
            filename_path = cellpose_path / Path("fiducial_max_projection.ome.tiff")
            with TiffWriter(filename_path, bigtiff=True) as tif:
                tif.write(
                    fiducial_max_projection,
                    resolution=(
                        1e4 / float(spacing_zyx_um[2]),
                        1e4 / float(spacing_zyx_um[1]),
                    ),
                    compression="zlib",
                    compressionargs={"level": 8},
                    predictor=True,
                    photometric="minisblack",
                    resolutionunit="CENTIMETER",
                    metadata={
                        "axes": "YX",
                        "SignificantBits": 16,
                        "PhysicalSizeX": float(spacing_zyx_um[2]),
                        "PhysicalSizeXUnit": "µm",
                        "PhysicalSizeY": float(spacing_zyx_um[1]),
                        "PhysicalSizeYUnit": "µm",
                    },
                )

        if self._verbose >= 1:
            print(time_stamp(), "Finished global fiducial registration.")

    def _load_raw_data(self) -> None:
        """
        Load raw data across rounds for one tile.

        Returns
        -------
        None
            Function result.
        """

        self._data_raw = []
        stage_positions = []

        for round_id in self._round_ids:
            self._data_raw.append(
                self._datastore.load_local_corrected_image(
                    tile=self._tile_id,
                    round=round_id,
                )
            )

            stage_position, _ = self._datastore.load_local_stage_position_zyx_um(
                tile=self._tile_id, round=round_id
            )

            stage_positions.append(stage_position)

        self._stage_positions = np.stack(stage_positions, axis=0)
        del stage_positions
        gc.collect()

    def _generate_registrations(self) -> None:
        """
        Deconvolve fiducials, register rounds, and save local transforms.

        The first fiducial round is the local reference for the tile. All
        fiducial rounds are first deconvolved in GPU worker processes and
        returned as NumPy arrays. Moving rounds are then registered to the first
        round in parallel CPU workers using multiview-stitcher/ITK-Elastix. The
        resulting physical-space affine transforms are saved to the datastore.
        If requested, the parent process also warps and saves registered
        fiducial image volumes.

        Returns
        -------
        None
            Results are written to the datastore.
        """
        if self._num_gpus == 0:
            raise RuntimeError(
                "No GPUs detected. Cannot run _generate_registrations()."
            )

        all_rounds = list(self._round_ids)
        start_time = timeit.default_timer()
        result_queue = mp.Queue()
        num_decon_workers = min(self._num_gpus, len(all_rounds))
        chunk_size = (len(all_rounds) + num_decon_workers - 1) // num_decon_workers
        decon_processes = []

        for gpu_id in range(num_decon_workers):
            start = gpu_id * chunk_size
            end = min(start + chunk_size, len(all_rounds))
            if start >= end:
                break

            subset = all_rounds[start:end]
            old_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                process = mp.Process(
                    target=_deconvolve_fiducials_on_gpu,
                    args=(self, subset, 0, result_queue),
                )
                process.start()
                decon_processes.append(process)
            finally:
                if old_vis is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_vis

        decon_by_round = {}
        errors = []
        while len(decon_by_round) + len(errors) < len(all_rounds):
            try:
                status, round_id, payload, crop_yx = result_queue.get(timeout=5)
            except queue.Empty:
                if any(
                    process.exitcode not in (None, 0) for process in decon_processes
                ):
                    break
                if all(process.exitcode is not None for process in decon_processes):
                    break
                continue
            if status == "result":
                decon_by_round[round_id] = payload
                if crop_yx is not None:
                    requested_crop_yx = min(
                        int(self._crop_yx_decon),
                        int(max(np.asarray(payload).shape[-2:])),
                    )
                    if int(crop_yx) < requested_crop_yx:
                        self._crop_yx_decon = int(crop_yx)
            else:
                errors.append(payload)

        for process in decon_processes:
            process.join()
            if process.exitcode not in (0, None):
                errors.append(
                    f"Fiducial decon worker pid={process.pid} failed with "
                    f"exitcode={process.exitcode}."
                )

        if errors:
            raise RuntimeError("Fiducial deconvolution failed:\n" + "\n".join(errors))

        missing_rounds = [
            round_id for round_id in all_rounds if round_id not in decon_by_round
        ]
        if missing_rounds:
            raise RuntimeError(f"Missing deconvolved fiducial rounds: {missing_rounds}")

        _registration_diag(
            "fiducial_decon_all_done "
            f"tile={self._tile_id} rounds={len(decon_by_round)} "
            f"elapsed_s={timeit.default_timer() - start_time:.2f}"
        )

        reference_round_id = self._round_ids[0]
        reference_image = decon_by_round[reference_round_id]
        self._datastore.save_local_registered_image(
            reference_image,
            tile=self._tile_id,
            deconvolution=self._decon_fiducial,
            round=reference_round_id,
        )
        self._datastore.save_local_round_transform_zyx_um(
            np.eye(4, dtype=np.float32),
            tile=self._tile_id,
            round=reference_round_id,
        )
        if self._verbose >= 1:
            print(
                time_stamp(),
                f"Finished fiducial tile id: {self._tile_id}; round id: {reference_round_id}.",
            )

        registration_start_time = timeit.default_timer()
        registration_processes = []
        for round_id in self._round_ids[1:]:
            process = mp.Process(
                target=_register_fiducial_transform_worker,
                args=(self, reference_image, decon_by_round[round_id], round_id),
            )
            process.start()
            registration_processes.append(process)

        registration_errors = []
        for process in registration_processes:
            process.join()
            if process.exitcode != 0:
                registration_errors.append(
                    f"Fiducial registration worker pid={process.pid} failed with "
                    f"exitcode={process.exitcode}."
                )
        if registration_errors:
            raise RuntimeError(
                "Fiducial registration failed:\n" + "\n".join(registration_errors)
            )

        _registration_diag(
            "fiducial_registration_all_done "
            f"tile={self._tile_id} rounds={len(self._round_ids) - 1} "
            f"elapsed_s={timeit.default_timer() - registration_start_time:.2f}"
        )

        if self.save_all_fiducial_registered or self._perform_optical_flow:
            from merfish3danalysis.utils.multiview_registration import (
                warp_array_to_reference,
            )

            spacing_zyx_um = self._datastore.voxel_size_zyx_um
            for round_id in self._round_ids[1:]:
                transform_zyx_um = self._datastore.load_local_round_transform_zyx_um(
                    tile=self._tile_id,
                    round=round_id,
                )
                if transform_zyx_um is None:
                    raise RuntimeError(
                        f"Missing local round transform for tile={self._tile_id} "
                        f"round={round_id}."
                    )
                warp_start_time = timeit.default_timer()
                warped = warp_array_to_reference(
                    decon_by_round[round_id],
                    transform_zyx_um=transform_zyx_um,
                    spacing_zyx_um=spacing_zyx_um,
                    reference_shape=reference_image.shape,
                )
                _registration_diag(
                    "fiducial_parent_warp "
                    f"tile={self._tile_id} round={round_id} "
                    f"shape={tuple(int(v) for v in decon_by_round[round_id].shape)} "
                    f"elapsed_s={timeit.default_timer() - warp_start_time:.2f}"
                )

                if self._perform_optical_flow:
                    from merfish3danalysis.utils.registration import compute_warpfield

                    data_registered, warp_field, block_size, block_stride = (
                        compute_warpfield(
                            reference_image.astype(np.float32, copy=False),
                            warped.astype(np.float32, copy=False),
                            gpu_id=0,
                        )
                    )
                    self._datastore.save_coord_of_xform_px(
                        of_xform_px=warp_field,
                        tile=self._tile_id,
                        block_size=block_size,
                        block_stride=block_stride,
                        round=round_id,
                    )
                    registered_image = data_registered.clip(0, 2**16 - 1).astype(
                        np.uint16
                    )
                    del data_registered, warp_field
                else:
                    registered_image = warped.clip(0, 2**16 - 1).astype(np.uint16)

                if self.save_all_fiducial_registered:
                    self._datastore.save_local_registered_image(
                        registered_image=registered_image,
                        tile=self._tile_id,
                        deconvolution=self._decon_fiducial,
                        round=round_id,
                    )
                if self._verbose >= 1:
                    print(
                        time_stamp(),
                        f"Finished fiducial tile id: {self._tile_id}; round id: {round_id}.",
                    )
                del warped, registered_image

        del decon_by_round
        gc.collect()

    def _apply_registration_to_bits(self) -> None:
        """
        Register readout bits and save registered data plus U-FISH predictions.

        Returns
        -------
        None
            Results are written to the datastore.
        """
        # 1) How many GPUs do we have?
        if self._num_gpus == 0:
            raise RuntimeError(
                "No GPUs detected. Cannot run _apply_registration_to_bits()."
            )

        # 2) Grab all bit IDs and split into `num_gpus` chunks
        all_bits = list(self._bit_ids)
        chunk_size = (
            len(all_bits) + self._num_gpus - 1
        ) // self._num_gpus  # ceiling division

        # 3) Launch one process per GPU (only as many as needed)
        processes = []
        for gpu_id in range(self._num_gpus):
            start = gpu_id * chunk_size
            end = min(start + chunk_size, len(all_bits))
            if start >= end:
                break  # no more bits to assign

            subset = all_bits[start:end]

            old_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                p = mp.Process(target=_apply_bits_on_gpu, args=(self, subset, 0))
                p.start()
                processes.append(p)
            finally:
                if old_vis is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_vis

        # 4) Wait for all GPU-workers to finish
        for p in processes:
            p.join()


def no_op(*args: Any, **kwargs: Any) -> None:
    """Function to monkey patch print to suppress output.

    Parameters
    ----------
    args: Any
        positional arguments
    kwargs: Any
        keyword arguments
    """

    pass


def time_stamp() -> str:
    """
    Return a human-readable timestamp for progress messages.

    Returns
    -------
    str
        Current local time formatted as ``YYYY-MM-DD HH:MM:SS``.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
