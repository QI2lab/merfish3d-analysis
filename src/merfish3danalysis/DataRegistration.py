"""
Register qi2lab 3D MERFISH data using cross-correlation and optical flow.

This module enables the registration of MERFISH datasets by utilizing
cross-correlation and optical flow techniques.

History:
---------
- **2025/29**:
    - Refactored to move background subtraction out of deconvolution step.
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
mp.set_start_method('spawn', force=True)
import os
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*cupyx\.jit\.rawkernel is experimental.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*block stride.*last level.*"
)

import numpy as np
from typing import Union
import gc
import SimpleITK as sitk
from merfish3danalysis.qi2labDataStore import qi2labDataStore

import builtins
from datetime import datetime
import time

def _apply_first_polyDT_on_gpu(
    dr,
    gpu_id: int =0
):
    import cupy as cp
    cp.cuda.Device(gpu_id).use()
    from merfish3danalysis.utils.rlgc import chunked_rlgc

    raw0 = dr._datastore.load_local_corrected_image(
        tile=dr._tile_id,
        round=0,
        return_future=False
    )

    if dr._bkd_subtract_polyDT:
        from merfish3danalysis.utils.imageprocessing import subtract_background_imagej
        bkd_image = subtract_background_imagej(raw0,200)
    else:
        bkd_image = raw0.copy()

    ref_image_decon = chunked_rlgc(
        image=bkd_image,
        psf=dr._psfs[0, :],
        gpu_id=0,
        crop_yx = dr._crop_yx_decon,
    )

    dr._datastore.save_local_registered_image(
        ref_image_decon.clip(0,2**16-1).astype(np.uint16),
        tile=dr._tile_id,
        deconvolution=True,
        round=dr._round_ids[0]
    )
    print(time_stamp(), f"Finished polyDT tile id: {dr._tile_id}; round id: round001.")

    del raw0, ref_image_decon
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return True

def _apply_polyDT_on_gpu(
    dr,
    round_list: list,
    gpu_id: int = 0
):
    """
    Run the “deconvolve→rigid+optical‐flow” loop for a subset of polyDT rounds on a single GPU.
    
    Parameters
    ----------
    dr : Registration
        DataRegistration instance (pickled into this process)
    bit_list : list 
        bit_ids to process on this GPU
    gpu_id : int
        physical GPU to bind in this process
    """
    import torch
    torch.cuda.set_device(gpu_id)
    import cupy as cp
    cp.cuda.Device(gpu_id).use()

    from merfish3danalysis.utils.registration import compute_warpfield
    from merfish3danalysis.utils.registration import (
        apply_transform,
        compute_rigid_transform
    )
    from merfish3danalysis.utils.imageprocessing import downsample_image_anisotropic


    for r_idx, round_id in enumerate(round_list):

        test =  dr._datastore.load_local_registered_image(
            tile=dr._tile_id,
            round=round_id,
            return_future=False
        )
        if test is None:
            has_reg_decon_data = False
        else:
            has_reg_decon_data = True

        if not (has_reg_decon_data) or dr._overwrite_registered:
            if dr._decon_polyDT:
                ref_image_decon_float = dr._datastore.load_local_registered_image(
                    tile=dr._tile_id,
                    round=dr._round_ids[0],
                    return_future=False
                ).astype(np.float32)
            else:
                ref_image = dr._datastore.load_local_corrected_image(
                    tile=dr._tile_id,
                    round=dr._round_ids[0],
                    return_future=False
                )
                if dr._bkd_subtract_polyDT:
                    from merfish3danalysis.utils.imageprocessing import subtract_background_imagej
                    ref_image_decon_float = subtract_background_imagej(ref_image,200)
                else:
                    ref_image_decon_float = ref_image.copy().astype(np.float32)
                    del ref_image

            raw = dr._datastore.load_local_corrected_image(
                tile=dr._tile_id,
                round=round_id,
                return_future=False
            )

            if dr._decon_polyDT:
                from merfish3danalysis.utils.rlgc import chunked_rlgc

                if dr._bkd_subtract_polyDT:
                    from merfish3danalysis.utils.imageprocessing import subtract_background_imagej
                    bkd_image = subtract_background_imagej(raw,200)
                else:
                    bkd_image = raw.copy()
                del raw

                if dr._datastore.microscope_type == "2D":

                    mov_image_decon = np.zeros_like(bkd_image,dtype=np.float32)
                    for z_idx in range(raw.shape[0]):
                        mov_image_decon[z_idx,:] = chunked_rlgc(
                            image = bkd_image[z_idx,:],
                            psf = dr._psfs[0,:],
                            gpu_id = gpu_id,
                            crop_yx=raw.shape[-1]
                        )
                    mov_image_decon = mov_image_decon.clip(0,2**16-1).astype(np.uint16)
                    del bkd_image
                else:
                    mov_image_decon = chunked_rlgc(
                        image=bkd_image,
                        psf=dr._psfs[0, :],
                        gpu_id=gpu_id,
                        crop_yx = dr._crop_yx_decon,
                    )
                    mov_image_decon = mov_image_decon.clip(0,2**16-1).astype(np.uint16)
                    del bkd_image
            else:
                if dr._bkd_subtract_polyDT:
                    from merfish3danalysis.utils.imageprocessing import subtract_background_imagej
                    bkd_image = subtract_background_imagej(raw,200)
                else:
                    bkd_image = raw.copy()

                mov_image_decon = bkd_image.copy().astype(np.uint16)
                del raw, bkd_image

            mov_image_decon_float = mov_image_decon.copy().astype(np.float32)
            del mov_image_decon

            if dr._datastore.microscope_type == "3D":
                downsample_factors = [3,9,9]
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
                downsample_factors = [1,3,3]
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
                mask = None,
                projection=None,
                gpu_id = gpu_id
            )

            xyz_shift = np.asarray(lowres_xyz_shift,dtype=np.float32)
            xyz_shift_float = [round(float(v),1) for v in lowres_xyz_shift]
            
            initial_xyz_transform = sitk.TranslationTransform(3, xyz_shift_float)
            warped_mov_image_decon_float = apply_transform(
                ref_image_decon_float, mov_image_decon_float, initial_xyz_transform
            )
            del mov_image_decon_float
            gc.collect()

            mov_image_decon_float = warped_mov_image_decon_float.copy().astype(np.float32)
            del warped_mov_image_decon_float
            gc.collect()

            dr._datastore.save_local_rigid_xform_xyz_px(
                rigid_xform_xyz_px=xyz_shift,
                tile=dr._tile_id,
                round=round_id
            )

            if dr._perform_optical_flow:
                
                data_registered, warp_field, block_size, block_stride = compute_warpfield(
                    ref_image_decon_float,
                    mov_image_decon_float,
                    gpu_id = gpu_id
                )

                dr._datastore.save_coord_of_xform_px(
                    of_xform_px=warp_field,
                    tile=dr._tile_id,
                    block_size=block_size,
                    block_stride=block_stride,
                    round=round_id
                )

                data_registered = data_registered.clip(0,2**16-1).astype(np.uint16)
                
                del warp_field
                gc.collect()
            else:
                data_registered = mov_image_decon_float.clip(0,2**16-1).astype(np.uint16)
                
            if dr.save_all_polyDT_registered:
                dr._datastore.save_local_registered_image(
                    registered_image=data_registered.astype(np.uint16),
                    tile=dr._tile_id,
                    deconvolution=True,
                    round=round_id
                )
            print(time_stamp(), f"Finished polyDT tile id: {dr._tile_id}; round id: {round_id}.")

            del data_registered
            gc.collect()

            try:
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

def _apply_bits_on_gpu(
    dr,
    bit_list: list, 
    gpu_id: int = 0
):
    """
    Run the “deconvolve→rigid+optical‐flow→UFish” loop for a subset of bits on a single GPU.
    
    Parameters
    ----------
    dr       : 
        DataRegistration instance (pickled into this process)
    bit_list : list 
        bit_ids to process on this GPU
    gpu_id : int
        physical GPU to bind in this process
    """

    import torch
    import cupy as cp

    torch.cuda.set_device(gpu_id)
    cp.cuda.Device(gpu_id).use()
    
    from warpfield.warp import warp_volume
    import os
    os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    from ufish.api import UFish
    from merfish3danalysis.utils.rlgc import chunked_rlgc
    from merfish3danalysis.utils.registration import apply_transform


    for bit_id in bit_list:

        r_idx = dr._datastore.load_local_round_linker(tile=dr._tile_id, bit=bit_id) - 1
        ex_wl, em_wl = dr._datastore.load_local_wavelengths_um(tile=dr._tile_id, bit=bit_id)
        psf_idx = 1 if ex_wl < 600 else 2

        test = dr._datastore.load_local_registered_image(tile=dr._tile_id, bit=bit_id)
        reg_on_disk = (test is not None)

        if (not reg_on_disk) or dr._overwrite_registered:
            # load data
            corrected_image = dr._datastore.load_local_corrected_image(
                tile=dr._tile_id, bit=bit_id, return_future=False
            )

            # deconvolution
            if dr._decon:
                if dr._datastore.microscope_type == "2D":
                    decon_image = np.zeros_like(corrected_image,dtype=np.float32)
                    for z_idx in range(corrected_image.shape[0]):
                        decon_image[z_idx,:] = chunked_rlgc(
                            image = corrected_image[z_idx,:],
                            psf = dr._psfs[psf_idx,:],
                            gpu_id = gpu_id,
                            crop_yx=corrected_image.shape[-1]
                        )
                    decon_image = decon_image.clip(0,2**16-1).astype(np.uint16)
                else:
                    decon_image = chunked_rlgc(
                        image=corrected_image,
                        psf=dr._psfs[psf_idx, :],
                        gpu_id = gpu_id,
                        crop_yx = dr._crop_yx_decon
                    )
                    decon_image = decon_image.clip(0,2**16-1).astype(np.uint16)
            else:
                decon_image = corrected_image.copy()

            # apply rigid + (optional) optical‐flow if r_idx > 0
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
                    warp_field, block_size, block_stride = dr._datastore.load_coord_of_xform_px(
                         tile=dr._tile_id, 
                         round=dr._round_ids[r_idx], 
                         return_future=False
                    )

                    block_size = cp.asarray(block_size, dtype=cp.float32)
                    block_stride = cp.asarray(block_stride, dtype=cp.float32)
                    decon_image_warped_cp = warp_volume(
                        decon_image_rigid, 
                        warp_field, 
                        block_stride, 
                        cp.array(-block_size / block_stride / 2), 
                        out=None,
                        gpu_id=gpu_id
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
            data_reg = data_reg.clip(0,2**16-1).astype(np.uint16)

            # UFISH
            ufish = UFish(device=f"cuda:{gpu_id}")
            ufish.load_weights_from_internet()
            ufish_loc, ufish_data = ufish.predict(
                data_reg, axes="zyx", blend_3d=False, batch_size=1
            )

            ufish_loc = ufish_loc.rename(
                columns={"axis-0": "z", "axis-1": "y", "axis-2": "x"}
            )
            del ufish
            gc.collect()

            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            # UFISH ROI sums
            roi_z, roi_y, roi_x = 7, 5, 5

            def sum_pixels_in_roi(row, image, roi_dims):
                z, y, x = row["z"], row["y"], row["x"]
                rz, ry, rx = roi_dims
                zmin = max(0, z - rz // 2)
                ymin = max(0, y - ry // 2)
                xmin = max(0, x - rx // 2)
                zmax = min(image.shape[0], zmin + rz)
                ymax = min(image.shape[1], ymin + ry)
                xmax = min(image.shape[2], xmin + rx)
                roi = image[int(zmin):int(zmax), int(ymin):int(ymax), int(xmin):int(xmax)]
                return np.sum(roi)

            ufish_loc["sum_prob_pixels"] = ufish_loc.apply(
                sum_pixels_in_roi, axis=1, image=ufish_data, roi_dims=(roi_z, roi_y, roi_x)
            )
            ufish_loc["sum_decon_pixels"] = ufish_loc.apply(
                sum_pixels_in_roi, axis=1, image=data_reg, roi_dims=(roi_z, roi_y, roi_x)
            )

            ufish_loc["tile_idx"] = dr._tile_ids.index(dr._tile_id)
            ufish_loc["bit_idx"] = dr._bit_ids.index(bit_id) + 1
            ufish_loc["tile_z_px"] = ufish_loc["z"]
            ufish_loc["tile_y_px"] = ufish_loc["y"]
            ufish_loc["tile_x_px"] = ufish_loc["x"]

            # save results
            dr._datastore.save_local_registered_image(
                data_reg, tile=dr._tile_id, deconvolution=True, bit=bit_id
            )
            dr._datastore.save_local_ufish_image(ufish_data, tile=dr._tile_id, bit=bit_id)
            dr._datastore.save_local_ufish_spots(ufish_loc, tile=dr._tile_id, bit=bit_id)
            print(time_stamp(), f"Finished readout tile id: {dr._tile_id}; bit id: {bit_id}.")

            del data_reg, ufish_data, ufish_loc
            gc.collect()

    return True

class DataRegistration:
    """Register 2D or 3D MERFISH data across rounds.
    
    Parameters
    ----------
    datastore : qi2labDataStore
        Initialized qi2labDataStore object
    decon_polyDT: bool, default False
        Deconvolve ALL polyDT rounds. False = only deconvolve round 1 for downstream stitching.
    bkd_subtract_polyDT: bool, default True
        Background subtraction ALL polyDT rounds.
    overwrite_registered: bool, default False
        Overwrite existing registered data and registrations
    perform_optical_flow: bool, default False
        Perform optical flow registration
    save_all_polyDT_registered: bool, default True
        Save fidicual polyDT rounds > 1. These are not used for analysis. 
    num_gpus: int, default 1
        Number of GPUs to use for registration.
    crop_yx_decon: int, default 1024
        Crop size for deconvolution applied to both y and x dimensions.
    """
        
    def __init__(
        self,
        datastore: qi2labDataStore,
        decon_polyDT: bool = False,
        bkd_subtract_polyDT: bool = True,
        overwrite_registered: bool = False,
        perform_optical_flow: bool = True,
        save_all_polyDT_registered: bool = False,
        num_gpus: int = 1,
        crop_yx_decon: int = 1024
    ):
    
        self._datastore = datastore
        self._decon_polyDT = decon_polyDT
        self._tile_ids = self._datastore.tile_ids
        self._round_ids = self._datastore.round_ids
        self._bit_ids = self._datastore.bit_ids
        self._psfs = self._datastore.channel_psfs
        self._num_gpus = num_gpus
        self._crop_yx_decon = crop_yx_decon
        self._bkd_subtract_polyDT = bkd_subtract_polyDT

        self._perform_optical_flow = perform_optical_flow
        self._data_raw = None
        self._has_registered_data = None
        self._overwrite_registered = overwrite_registered
        self.save_all_polyDT_registered = save_all_polyDT_registered
        self._decon = True
        self._original_print = builtins.print

    # -----------------------------------
    # property access for class variables
    # -----------------------------------
    @property
    def datastore(self):
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
    def dataset_path(self, value: qi2labDataStore):
        """Set the qi2labDataStore object.
        
        Parameters
        ----------
        value : qi2labDataStore
            qi2labDataStore object
        """

        del self._datastore
        self._datastore = value

    @property
    def tile_id(self):
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
    def tile_id(self, value: Union[int,str]):
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
    def perform_optical_flow(self):
        """Get the perform_optical_flow flag.
        
        Returns
        -------
        perform_optical_flow: bool
            Perform optical flow registration
        """

        return self._perform_optical_flow
    
    @perform_optical_flow.setter
    def perform_optical_flow(self, value: bool):
        """Set the perform_optical_flow flag.
        
        Parameters
        ----------
        value : bool
            Perform optical flow registration
        """

        self._perform_optical_flow = value

    @property
    def overwrite_registered(self):
        """Get the overwrite_registered flag.
        
        Returns
        -------
        overwrite_registered: bool
            Overwrite existing registered data and registrations
        """

        return self._overwrite_registered
    
    @overwrite_registered.setter
    def overwrite_registered(self, value: bool):
        """Set the overwrite_registered flag.
        
        Parameters
        ----------
        value : bool
            Overwrite existing registered data and registrations
        """

        self._overwrite_registered = value
        
    def register_all_tiles(self):
        """Helper function to register all tiles."""
        for tile_id in self._datastore.tile_ids:
            self.tile_id=tile_id
            self._generate_registrations()
            self._apply_registration_to_bits()
            
    def register_one_tile(self, tile_id: Union[int,str]):
        """Helper function to register one tile.
        
        Parameters
        ----------
        tile_id : Union[int,str]
            Tile id
        """

        self.tile_id = tile_id
        self._generate_registrations()
        self._apply_registration_to_bits()

    def _load_raw_data(self):
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
                    tile=self._tile_id,
                    round=round_id
                )
            
            stage_positions.append(stage_position)

        self._stage_positions = np.stack(stage_positions, axis=0)
        del stage_positions
        gc.collect()

    def _generate_registrations(self):
        """Generate registered, deconvolved fiducial data and save to datastore."""
        test =  self._datastore.load_local_registered_image(
            tile=self._tile_id,
            round=self._round_ids[0]
        )
        
        if test is None:
            has_reg_decon_data = False
        else:
            has_reg_decon_data = True
            
        if not (has_reg_decon_data) or self._overwrite_registered:

            p_first = mp.Process(target=_apply_first_polyDT_on_gpu, args=(self,0))
            p_first.start()
            p_first.join()

        # 1) How many GPUs do we have?
        if self._num_gpus == 0:
            raise RuntimeError("No GPUs detected. Cannot run _generate_registrations().")

        # 2) Grab all rounds IDs after round 0 and split into `num_gpus` chunks
        all_rounds = list(self._round_ids[1:])
        chunk_size = (len(all_rounds) + self._num_gpus - 1) // self._num_gpus  # ceiling division

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
                p = mp.Process(target=_apply_polyDT_on_gpu, args=(self, subset, 0))
                p.start()
                processes.append(p)
            finally:
                if old_vis is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_vis

        # 4) Wait for all GPU‐workers to finish
        for p in processes:
            p.join()

    def _apply_registration_to_bits(self):
        """Generate ufish + deconvolved, registered readout data and save to datastore."""
        # 1) How many GPUs do we have?
        if self._num_gpus == 0:
            raise RuntimeError("No GPUs detected. Cannot run _apply_registration_to_bits().")

        # 2) Grab all bit IDs and split into `num_gpus` chunks
        all_bits = list(self._bit_ids)
        chunk_size = (len(all_bits) + self._num_gpus - 1) // self._num_gpus  # ceiling division

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


        # 4) Wait for all GPU‐workers to finish
        for p in processes:
            p.join()
                
                
def _no_op(*args, **kwargs):
    """Function to monkey patch print to suppress output.
    
    Parameters
    ----------
    *args
        Variable length argument list
    **kwargs
        Arbitrary keyword arguments
    """

    pass

def time_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")