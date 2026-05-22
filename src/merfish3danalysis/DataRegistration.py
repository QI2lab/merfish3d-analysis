"""
Register qi2lab 3D MERFISH data using cross-correlation and optical flow.

This module enables the registration of MERFISH datasets by utilizing
cross-correlation and optical flow techniques.

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
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

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


def _resolve_ufish_weights_path(model: str | Path | None) -> Path | str | None:
    """Resolve a U-FISH model alias or path without requiring U-FISH imports."""

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
    """Load configured U-FISH weights from an alias, local path, or weights file."""

    weights = _resolve_ufish_weights_path(model)
    if weights is None:
        raise ValueError("Resolved U-FISH weights cannot be None.")

    if isinstance(weights, Path):
        ufish.load_weights_from_path(weights)
    else:
        ufish.load_weights(weights_file=weights)


def _resolve_psf(psfs: Any, psf_idx: int) -> np.ndarray:
    """Fetch PSF by index from uniform or ragged channel PSF storage."""

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


def _apply_first_fiducial_on_gpu(dr, gpu_id: int = 0) -> bool:  # noqa: ANN001
    import cupy as cp

    cp.cuda.Device(gpu_id).use()
    from merfish3danalysis.utils.rlgc import chunked_rlgc, clear_rlgc_caches

    raw0 = dr._datastore.load_local_corrected_image(
        tile=dr._tile_id, round=0, return_future=False
    )

    ref_image_decon = chunked_rlgc(
        image=raw0,
        psf=_resolve_psf(dr._psfs, 0),
        gpu_id=0,
        crop_yx=dr._crop_yx_decon,
        release_memory=False,
    )

    dr._datastore.save_local_registered_image(
        ref_image_decon.clip(0, 2**16 - 1).astype(np.uint16),
        tile=dr._tile_id,
        deconvolution=True,
        round=dr._round_ids[0],
    )
    if dr._verbose >= 1:
        print(
            time_stamp(),
            f"Finished fiducial tile id: {dr._tile_id}; round id: round001.",
        )

    del raw0, ref_image_decon
    gc.collect()
    clear_rlgc_caches(clear_memory_pool=False)
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    return True


def _apply_fiducial_on_gpu(dr, round_list: list, gpu_id: int = 0) -> bool:  # noqa: ANN001
    """
    Run the “deconvolve→rigid+optical-flow” loop for a subset of fiducial rounds on a single GPU.

    Parameters
    ----------
    dr : Registration
        DataRegistration instance (pickled into this process)
    round_list : list
        round_ids to process on this GPU
    gpu_id : int
        physical GPU to bind in this process
    """
    import torch

    torch.cuda.set_device(gpu_id)
    import cupy as cp

    cp.cuda.Device(gpu_id).use()

    from merfish3danalysis.utils.imageprocessing import downsample_image_anisotropic
    from merfish3danalysis.utils.registration import (
        apply_transform,
        compute_rigid_transform,
        compute_warpfield,
    )
    from merfish3danalysis.utils.rlgc import chunked_rlgc, clear_rlgc_caches

    for _r_idx, round_id in enumerate(round_list):
        has_reg_decon_data = dr._has_valid_registered_image(round_id=round_id)

        if not (has_reg_decon_data) or dr._overwrite_registered:
            if dr._decon_fiducial:
                ref_image_decon_float = dr._datastore.load_local_registered_image(
                    tile=dr._tile_id, round=dr._round_ids[0], return_future=False
                ).astype(np.float32)
            else:
                ref_image = dr._datastore.load_local_corrected_image(
                    tile=dr._tile_id, round=dr._round_ids[0], return_future=False
                )
                ref_image_decon_float = ref_image.copy().astype(np.float32)
                del ref_image

            raw = dr._datastore.load_local_corrected_image(
                tile=dr._tile_id, round=round_id, return_future=False
            )

            if dr._decon_fiducial:
                fiducial_image = raw.copy()
                del raw

                if dr._datastore.microscope_type == "2D":
                    mov_image_decon = chunked_rlgc(
                        image=fiducial_image,
                        psf=_resolve_psf(dr._psfs, 0),
                        gpu_id=gpu_id,
                        crop_yx=fiducial_image.shape[-1],
                        release_memory=False,
                    )
                    mov_image_decon = mov_image_decon.clip(0, 2**16 - 1).astype(
                        np.uint16
                    )
                    del fiducial_image
                else:
                    mov_image_decon = chunked_rlgc(
                        image=fiducial_image,
                        psf=_resolve_psf(dr._psfs, 0),
                        gpu_id=gpu_id,
                        crop_yx=dr._crop_yx_decon,
                        release_memory=False,
                    )
                    mov_image_decon = mov_image_decon.clip(0, 2**16 - 1).astype(
                        np.uint16
                    )
                    del fiducial_image
            else:
                mov_image_decon = raw.copy().astype(np.uint16)
                del raw

            mov_image_decon_float = mov_image_decon.copy().astype(np.float32)
            del mov_image_decon

            if dr._datastore.microscope_type == "3D":
                downsample_factors = [3, 9, 9]
                if max(downsample_factors) > 1:
                    ref_image_decon_float_ds = downsample_image_anisotropic(
                        ref_image_decon_float, downsample_factors
                    )
                    mov_image_decon_float_ds = downsample_image_anisotropic(
                        mov_image_decon_float, downsample_factors
                    )
                else:
                    ref_image_decon_float_ds = ref_image_decon_float.copy()
                    mov_image_decon_float_ds = mov_image_decon_float.copy()
            else:
                downsample_factors = [1, 3, 3]
                if max(downsample_factors) > 1:
                    ref_image_decon_float_ds = downsample_image_anisotropic(
                        ref_image_decon_float, downsample_factors
                    )
                    mov_image_decon_float_ds = downsample_image_anisotropic(
                        mov_image_decon_float, downsample_factors
                    )
                else:
                    ref_image_decon_float_ds = ref_image_decon_float.copy()
                    mov_image_decon_float_ds = mov_image_decon_float.copy()

            _, lowres_xyz_shift = compute_rigid_transform(
                ref_image_decon_float_ds,
                mov_image_decon_float_ds,
                downsample_factors=downsample_factors,
                mask=None,
                projection=None,
                gpu_id=gpu_id,
            )

            xyz_shift = np.asarray(lowres_xyz_shift, dtype=np.float32)
            xyz_shift_float = [round(float(v), 1) for v in lowres_xyz_shift]

            initial_xyz_transform = sitk.TranslationTransform(3, xyz_shift_float)
            warped_mov_image_decon_float = apply_transform(
                ref_image_decon_float, mov_image_decon_float, initial_xyz_transform
            )
            del mov_image_decon_float
            gc.collect()

            mov_image_decon_float = warped_mov_image_decon_float.copy().astype(
                np.float32
            )
            del warped_mov_image_decon_float
            gc.collect()

            dr._datastore.save_local_rigid_xform_xyz_px(
                rigid_xform_xyz_px=xyz_shift, tile=dr._tile_id, round=round_id
            )

            if dr._perform_optical_flow:
                data_registered, warp_field, block_size, block_stride = (
                    compute_warpfield(
                        ref_image_decon_float, mov_image_decon_float, gpu_id=gpu_id
                    )
                )

                dr._datastore.save_coord_of_xform_px(
                    of_xform_px=warp_field,
                    tile=dr._tile_id,
                    block_size=block_size,
                    block_stride=block_stride,
                    round=round_id,
                )

                data_registered = data_registered.clip(0, 2**16 - 1).astype(np.uint16)

                del warp_field
                gc.collect()
            else:
                data_registered = mov_image_decon_float.clip(0, 2**16 - 1).astype(
                    np.uint16
                )

            if dr.save_all_fiducial_registered:
                dr._datastore.save_local_registered_image(
                    registered_image=data_registered.astype(np.uint16),
                    tile=dr._tile_id,
                    deconvolution=dr._decon_fiducial,
                    round=round_id,
                )
            if dr._verbose >= 1:
                print(
                    time_stamp(),
                    f"Finished fiducial tile id: {dr._tile_id}; round id: {round_id}.",
                )

            del data_registered
            gc.collect()
    try:
        clear_rlgc_caches(clear_memory_pool=False)
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        try:
            import cupyx

            cupyx.scipy.fft.clear_plan_cache()
        except Exception:
            pass
        try:
            cp.fft.config.get_plan_cache().clear()
        except Exception:
            pass
    except Exception:
        pass

    return True


def _apply_bits_on_gpu(dr, bit_list: list, gpu_id: int = 0) -> bool:  # noqa: ANN001
    """
    Run the “deconvolve→rigid+optical-flow→feature_predictor” loop for a subset of bits on a single GPU.

    Parameters
    ----------
    dr : DataRegistration
        DataRegistration instance (pickled into this process)
    bit_list : list
        bit_ids to process on this GPU
    gpu_id : int
        physical GPU to bind in this process
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

    from merfish3danalysis.utils.registration import apply_transform
    from merfish3danalysis.utils.rlgc import chunked_rlgc, clear_rlgc_caches

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

        if reg_on_disk and not dr._overwrite_registered:
            data_reg = dr._datastore.load_local_registered_image(
                tile=dr._tile_id, bit=bit_id, return_future=False
            ).astype(np.uint16, copy=False)
        else:
            # load data
            corrected_image = dr._datastore.load_local_corrected_image(
                tile=dr._tile_id, bit=bit_id, return_future=False
            )

            # deconvolution
            if dr._decon_readout:
                if dr._datastore.microscope_type == "2D":
                    decon_image = chunked_rlgc(
                        image=corrected_image,
                        psf=_resolve_psf(dr._psfs, psf_idx),
                        gpu_id=gpu_id,
                        crop_yx=corrected_image.shape[-1],
                        release_memory=False,
                    )
                    decon_image = decon_image.clip(0, 2**16 - 1).astype(np.uint16)
                else:
                    decon_image = chunked_rlgc(
                        image=corrected_image,
                        psf=_resolve_psf(dr._psfs, psf_idx),
                        gpu_id=gpu_id,
                        crop_yx=dr._crop_yx_decon,
                        release_memory=False,
                    )
                    decon_image = decon_image.clip(0, 2**16 - 1).astype(np.uint16)
            else:
                decon_image = corrected_image.copy()

            # apply rigid + (optional) optical-flow if r_idx > 0
            if r_idx > 0:
                rigid_xyz_px = dr._datastore.load_local_rigid_xform_xyz_px(
                    tile=dr._tile_id, round=dr._round_ids[r_idx]
                )
                shift_xyz = [float(v) for v in rigid_xyz_px]
                xyz_tx = sitk.TranslationTransform(3, np.asarray(shift_xyz))

                # apply rigid
                decon_image_rigid = apply_transform(decon_image, decon_image, xyz_tx)
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
                        decon_image_rigid,
                        warp_field,
                        block_stride,
                        cp.array(-block_size / block_stride / 2),
                        out=None,
                        gpu_id=gpu_id,
                    )
                    data_reg = cp.asnumpy(decon_image_warped_cp).astype(np.float32)
                    del decon_image_warped_cp
                    gc.collect()

                else:
                    data_reg = decon_image_rigid.copy()
                    del decon_image_rigid
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

        if (not feature_predictor_on_disk) or dr._overwrite_registered:
            # UFISH
            ufish = UFish(device=f"cuda:{gpu_id}")
            _load_ufish_model(ufish, dr._ufish_model)
            feature_predictor_loc, feature_predictor_data = ufish.predict(
                data_reg, axes="zyx", blend_3d=False, batch_size=1
            )

            feature_predictor_loc = feature_predictor_loc.rename(
                columns={"axis-0": "z", "axis-1": "y", "axis-2": "x"}
            )
            del ufish
            gc.collect()

            torch.cuda.empty_cache()
            gc.collect()

            # feature_predictor ROI sums
            roi_z, roi_y, roi_x = 7, 5, 5

            def sum_pixels_in_roi(row, image, roi_dims):  # noqa
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
    crop_yx_decon: int, default 1024
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
        crop_yx_decon: int = 1024,
        ufish_model: str | Path | None = None,
        verbose: int = 1,
    ) -> None:
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
        """Helper function to register all tiles."""
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

    def _load_raw_data(self) -> None:
        """Load raw data across rounds for one tile."""

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
        """Generate registered, deconvolved fiducial data and save to datastore."""
        has_reg_decon_data = self._has_valid_registered_image(
            round_id=self._round_ids[0]
        )

        if not (has_reg_decon_data) or self._overwrite_registered:
            p_first = mp.Process(target=_apply_first_fiducial_on_gpu, args=(self, 0))
            p_first.start()
            p_first.join()

        # 1) How many GPUs do we have?
        if self._num_gpus == 0:
            raise RuntimeError(
                "No GPUs detected. Cannot run _generate_registrations()."
            )

        # 2) Grab all rounds IDs after round 0 and split into `num_gpus` chunks
        all_rounds = list(self._round_ids[1:])
        chunk_size = (
            len(all_rounds) + self._num_gpus - 1
        ) // self._num_gpus  # ceiling division

        # 3) Launch one process per GPU (only as many as needed)
        processes = []
        for gpu_id in range(self._num_gpus):
            start = gpu_id * chunk_size
            end = min(start + chunk_size, len(all_rounds))
            if start >= end:
                break  # no more rounds to assign

            subset = all_rounds[start:end]

            old_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                # Inside child, logical device 0 maps to this physical GPU.
                p = mp.Process(target=_apply_fiducial_on_gpu, args=(self, subset, 0))
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

    def _apply_registration_to_bits(self) -> None:
        """Generate feature_predictor + deconvolved, registered readout data and save to datastore."""
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
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
