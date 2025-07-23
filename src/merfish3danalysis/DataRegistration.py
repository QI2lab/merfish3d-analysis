"""
Register qi2lab 3D MERFISH data using cross-correlation and optical flow.

This module enables the registration of MERFISH datasets by utilizing
cross-correlation and optical flow techniques.

History:
---------
- **2025/07**:
    - Implement anistropic downsampling for registration.
    - Implement RLGC deconvolution.
    - Implement new deeds-registration package.
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

import numpy as np
from typing import Union, Optional
import gc
import SimpleITK as sitk
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.registration import (
    compute_optical_flow,
    apply_transform,
    compute_rigid_transform,
)
from merfish3danalysis.utils.imageprocessing import downsample_image_anisotropic

import builtins
from tqdm import tqdm

def _apply_first_polyDT_on_gpu(
    dr,
    gpu_id: int =0
):
    import cupy as cp
    from merfish3danalysis.utils.rlgc import chunked_rlgc
    cp.cuda.Device(0).use()

    raw0 = dr._datastore.load_local_corrected_image(
        tile=dr._tile_id,
        round=0,
        return_future=False
    )

    ref_image_decon = chunked_rlgc(
        image=raw0,
        psf=dr._psfs[0, :],
        gpu_id=0,
        crop_yx = dr._crop_yx_decon
    )

    dr._datastore.save_local_registered_image(
        ref_image_decon,
        tile=dr._tile_id,
        deconvolution=True,
        round=dr._round_ids[0]
    )

    del raw0, ref_image_decon
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

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
    import cupy as cp
    from merfish3danalysis.utils.rlgc import chunked_rlgc

    torch.cuda.set_device(gpu_id)
    cp.cuda.Device(gpu_id).use()

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
            ref_image_decon_norm = dr._datastore.load_local_registered_image(
                tile=dr._tile_id,
                round=dr._round_ids[0],
                return_future=False
            ).astype(np.float32)

            min_val = ref_image_decon_norm.min()
            max_val = ref_image_decon_norm.max()

            # normalize reference to [0,1] and avoid divide‐by‐zero if image is constant
            if max_val > min_val:
                ref_image_decon_norm -= min_val
                ref_image_decon_norm /= (max_val - min_val)
            else:
                ref_image_decon_norm.fill(0)
            
            raw = dr._datastore.load_local_corrected_image(
                tile=dr._tile_id,
                round=round_id,
                return_future=False
            )     

            mov_image_decon = chunked_rlgc(
                image=raw,
                psf=dr._psfs[0, :],
                gpu_id=gpu_id,
                crop_yx = dr._crop_yx_decon
            )

            mov_image_decon_norm = mov_image_decon.copy().astype(np.float32)

            mov_min_val = mov_image_decon_norm.min()
            mov_max_val = mov_image_decon_norm.max()

            # normalize moving to [0,1] and avoid divide‐by‐zero if image is constant
            if mov_max_val > mov_min_val:
                mov_image_decon_norm -= mov_min_val
                mov_image_decon_norm /= (mov_max_val - mov_min_val)
            else:
                mov_image_decon_norm.fill(0)

            downsample_factors = (2,6,6)
            if max(downsample_factors) > 1:
                ref_image_decon_norm_ds = downsample_image_anisotropic(
                    ref_image_decon_norm, downsample_factors
                )
                mov_image_decon_norm_ds = downsample_image_anisotropic(
                    mov_image_decon_norm, downsample_factors
                )
            else:
                ref_image_decon_norm_ds = ref_image_decon_norm.copy()
                mov_image_decon_norm_ds = mov_image_decon_norm.copy()

            _, initial_xy_shift = compute_rigid_transform(
                ref_image_decon_norm_ds,
                mov_image_decon_norm_ds,
                downsample_factors=downsample_factors,
                use_mask=False,
                projection="z",
            )

            intial_xy_transform = sitk.TranslationTransform(3, initial_xy_shift)

            mov_image_decon_norm = apply_transform(
                ref_image_decon_norm, mov_image_decon_norm, intial_xy_transform
            )

            downsample_factors = (2,6,6)
            if max(downsample_factors) > 1:
                ref_image_decon_norm_ds = downsample_image_anisotropic(
                    ref_image_decon_norm, downsample_factors
                )
                mov_image_decon_norm_ds = downsample_image_anisotropic(
                    mov_image_decon_norm, downsample_factors
                )
            else:
                ref_image_decon_norm_ds = ref_image_decon_norm.copy()
                mov_image_decon_norm_ds = mov_image_decon_norm.copy()

            _, intial_z_shift = compute_rigid_transform(
                ref_image_decon_norm_ds,
                mov_image_decon_norm_ds,
                downsample_factors=downsample_factors,
                use_mask=False,
                projection="search",
            )

            intial_z_transform = sitk.TranslationTransform(3, intial_z_shift)

            mov_image_decon_norm = apply_transform(
                ref_image_decon_norm, mov_image_decon_norm, intial_z_transform
            )

            downsample_factors = (2,6,6)
            if max(downsample_factors) > 1:
                ref_image_decon_norm_ds = downsample_image_anisotropic(
                    ref_image_decon_norm, downsample_factors
                )
                mov_image_decon_norm_ds = downsample_image_anisotropic(
                    mov_image_decon_norm, downsample_factors
                )
            else:
                ref_image_decon_norm_ds = ref_image_decon_norm.copy()
                mov_image_decon_norm_ds = mov_image_decon_norm.copy()

            _, initial_xyz_shift = compute_rigid_transform(
                ref_image_decon_norm_ds,
                mov_image_decon_norm_ds,
                use_mask=False,
                downsample_factors=downsample_factors,
                projection=None,
            )
            
            final_xyz_shift = (
                np.asarray(initial_xy_shift)
                + np.asarray(intial_z_shift)
                + np.asarray(initial_xyz_shift)
            )
                        
            dr._datastore.save_local_rigid_xform_xyz_px(
                rigid_xform_xyz_px=final_xyz_shift,
                tile=dr._tile_id,
                round=round_id
            )

            final_xyz_transform = sitk.TranslationTransform(3, final_xyz_shift)
            mov_image_decon_norm = apply_transform(
                ref_image_decon_norm, mov_image_decon_norm, final_xyz_transform
            )

            if dr._perform_optical_flow:
                downsample_factors = (2,6,6)
                if max(downsample_factors) > 1:
                    ref_image_decon_norm_ds = downsample_image_anisotropic(
                        ref_image_decon_norm, downsample_factors
                    )
                    mov_image_decon_norm_ds = downsample_image_anisotropic(
                        mov_image_decon_norm, downsample_factors
                    )

                of_xform_px = compute_optical_flow(
                    ref_image_decon_norm_ds, 
                    mov_image_decon_norm_ds
                )

                dr._datastore.save_coord_of_xform_px(
                    of_xform_px=of_xform_px,
                    tile=dr._tile_id,
                    downsampling=[
                        float(downsample_factors[0]),
                        float(downsample_factors[1]),
                        float(downsample_factors[2])
                    ],
                    round=round_id
                )

                of_xform_sitk = sitk.GetImageFromArray(
                    of_xform_px.transpose(1, 2, 3, 0).astype(np.float64),
                    isVector=True,
                )

                # undo normalization from [0,1] back to full range
                mov_image_decon = mov_image_decon_norm * (mov_max_val-mov_min_val) + mov_min_val

                interpolator = sitk.sitkLinear
                identity_transform = sitk.Transform(3, sitk.sitkIdentity)
                optical_flow_sitk = sitk.Resample(
                    of_xform_sitk,
                    sitk.GetImageFromArray(mov_image_decon),
                    identity_transform,
                    interpolator,
                    0,
                    of_xform_sitk.GetPixelID(),
                )
                displacement_field = sitk.DisplacementFieldTransform(
                    optical_flow_sitk
                )
                del optical_flow_sitk, of_xform_px
                gc.collect()

                # apply optical flow
                mov_image_sitk = sitk.Resample(
                    sitk.GetImageFromArray(mov_image_decon), 
                    displacement_field
                )

                data_registered = sitk.GetArrayFromImage(
                    mov_image_sitk
                ).astype(np.float32)

                data_registered = data_registered.clip(0,2**16-1).astype(np.uint16)
                
                del mov_image_sitk, displacement_field
                gc.collect()
            else:

                data_registered = mov_image_decon.clip(0,2**16-1).astype(np.uint16)
                
            if dr.save_all_polyDT_registered:
                dr._datastore.save_local_registered_image(
                    registered_image=data_registered.astype(np.uint16),
                    tile=dr._tile_id,
                    deconvolution=True,
                    round=round_id
                )

            del data_registered
            gc.collect()


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
    from ufish.api import UFish
    from merfish3danalysis.utils.rlgc import chunked_rlgc

    torch.cuda.set_device(gpu_id)
    cp.cuda.Device(gpu_id).use()

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
                decon_image = chunked_rlgc(
                    image=corrected_image,
                    psf=dr._psfs[psf_idx, :],
                    gpu_id = gpu_id,
                    crop_yx = dr._crop_yx_decon
                )
            else:
                decon_image = corrected_image.copy()

            # apply rigid + (optional) optical‐flow if r_idx > 0
            if r_idx > 0:
                rigid_xyz_px = dr._datastore.load_local_rigid_xform_xyz_px(
                    tile=dr._tile_id, round=dr._round_ids[r_idx]
                )
                shift_xyz = [float(v) for v in rigid_xyz_px]
                xyz_tx = sitk.TranslationTransform(3, shift_xyz)

                if dr._perform_optical_flow:
                    of_xform_px, _ = dr._datastore.load_coord_of_xform_px(
                        tile=dr._tile_id, round=dr._round_ids[r_idx], return_future=False
                    )
                    of_xform_sitk = sitk.GetImageFromArray(
                        of_xform_px.transpose(1, 2, 3, 0).astype(np.float64),
                        isVector=True,
                    )
                    interp = sitk.sitkLinear
                    identity = sitk.Transform(3, sitk.sitkIdentity)
                    optical_flow_sitk = sitk.Resample(
                        of_xform_sitk,
                        sitk.GetImageFromArray(decon_image),
                        identity,
                        interp,
                        0,
                        of_xform_sitk.GetPixelID(),
                    )
                    disp_field = sitk.DisplacementFieldTransform(optical_flow_sitk)
                    del optical_flow_sitk, of_xform_px
                    gc.collect()

                # apply rigid
                decon_image_rigid = apply_transform(decon_image, decon_image, xyz_tx)
                del decon_image

                if dr._perform_optical_flow:
                    decon_bit_sitk = sitk.Resample(
                        sitk.GetImageFromArray(decon_image_rigid), disp_field
                    )
                    del disp_field
                    data_reg = sitk.GetArrayFromImage(decon_bit_sitk).astype(np.float32)
                    del decon_bit_sitk
                else:
                    data_reg = decon_image_rigid.copy()
                    del decon_image_rigid
                gc.collect()
            else:
                data_reg = decon_image.copy()
                del decon_image
                gc.collect()

            # clip to uint16
            data_reg[data_reg < 0.0] = 0.0
            data_reg = data_reg.astype(np.uint16)

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

            del data_reg, ufish_data, ufish_loc
            gc.collect()

class DataRegistration:
    """Register 2D or 3D MERFISH data across rounds.
    
    Parameters
    ----------
    datastore : qi2labDataStore
        Initialized qi2labDataStore object
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
        overwrite_registered: bool = False,
        perform_optical_flow: bool = True,
        save_all_polyDT_registered: bool = True,
        num_gpus: int = 1,
        crop_yx_decon: int = 1024
    ):
    
        self._datastore = datastore
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
        for tile_id in tqdm(self._datastore.tile_ids,desc="tiles"):
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
            p = mp.Process(target=_apply_polyDT_on_gpu, args=(self, subset, gpu_id))
            p.start()
            processes.append(p)

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
            p = mp.Process(target=_apply_bits_on_gpu, args=(self, subset, gpu_id))
            p.start()
            processes.append(p)

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
