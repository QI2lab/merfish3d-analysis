"""
Register qi2lab 3D MERFISH data using cross-correlation and optical flow.

This module enables the registration of MERFISH datasets by utilizing
cross-correlation and optical flow techniques.

History:
---------
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
mp.set_start_method('forkserver', force=True)

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
from merfish3danalysis.utils.imageprocessing import (
    chunked_cudadecon,
    downsample_image_isotropic
)
from ufish.api import UFish
import torch
import cupy as cp
import builtins
from tqdm import tqdm
import os

def _apply_bits_on_gpu(dr, bit_list, gpu_id):
    """
    Run the “deconvolve→rigid+optical‐flow→UFish” loop for a subset of bits on a single GPU.
    
    dr       : a DataRegistration instance (pickled into this process)
    bit_list : list of bit_id’s to process on this GPU
    gpu_id   : which physical GPU to bind in this process
    """
    # 1) Force this process to see only GPU=gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)      # In this subprocess, “CUDA device 0” == the real GPU gpu_id
    cp.cuda.Device(0).use()

    for bit_id in bit_list:
        # --- the exact same body you had before for processing one bit ---
        r_idx = dr._datastore.load_local_round_linker(tile=dr._tile_id, bit=bit_id) - 1
        ex_wl, em_wl = dr._datastore.load_local_wavelengths_um(tile=dr._tile_id, bit=bit_id)
        psf_idx = 1 if ex_wl < 600 else 2

        test = dr._datastore.load_local_registered_image(tile=dr._tile_id, bit=bit_id)
        reg_on_disk = (test is not None)

        if (not reg_on_disk) or dr._overwrite_registered:
            # 1) load corrected image (blocking)
            corrected_image = dr._datastore.load_local_corrected_image(
                tile=dr._tile_id, bit=bit_id, return_future=False
            )

            # 2) deconvolve or copy
            if dr._decon:
                decon_image = chunked_cudadecon(
                    image=corrected_image,
                    psf=dr._psfs[psf_idx, :],
                    image_voxel_zyx_um=dr._datastore.voxel_size_zyx_um,
                    psf_voxel_zyx_um=dr._datastore.voxel_size_zyx_um,
                    wavelength_um=em_wl,
                    na=dr._datastore.na,
                    ri=dr._datastore.ri,
                    n_iters=dr._decon_iters,
                    background=dr._decon_background,
                )
            else:
                decon_image = corrected_image.copy()

            # 3) apply rigid + (optional) optical‐flow if r_idx > 0
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

            # clamp negatives & cast to uint16
            data_reg[data_reg < 0.0] = 0.0
            data_reg = data_reg.astype(np.uint16)

            # 4) run UFish
            builtins.print = lambda *a, **k: None
            ufish = UFish(device="cuda")
            ufish.load_weights_from_internet()
            ufish_loc, ufish_data = ufish.predict(
                data_reg, axes="zyx", blend_3d=False, batch_size=1
            )
            builtins.print = dr._original_print

            ufish_loc = ufish_loc.rename(
                columns={"axis-0": "z", "axis-1": "y", "axis-2": "x"}
            )
            del ufish
            gc.collect()

            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            # 5) ROI sums
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

            # 6) save outputs
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
    decon_iters : Optional[int], default 10
        Deconvolution iterations
    decon_background: Optional[float], default 50.0
        Background to substract during deconvolution
    """
        
    def __init__(
        self,
        datastore: qi2labDataStore,
        overwrite_registered: bool = False,
        perform_optical_flow: bool = False,
        save_all_polyDT_registered: bool = True,
        decon: Optional[bool] = True,
        decon_background: Optional[float] = 0.0,
    ):
    
        self._datastore = datastore
        self._tile_ids = self._datastore.tile_ids
        self._round_ids = self._datastore.round_ids
        self._bit_ids = self._datastore.bit_ids
        self._psfs = self._datastore.channel_psfs
        self._decon = decon

        self._perform_optical_flow = perform_optical_flow
        self._data_raw = None
        self._has_registered_data = None
        self._overwrite_registered = overwrite_registered
        self.save_all_polyDT_registered = save_all_polyDT_registered
        self._decon_background = decon_background
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
            self._load_raw_data()
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
        self._load_raw_data()
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

            if self._decon:
                ref_image_decon = chunked_cudadecon(
                    image=np.asarray(self._data_raw[0].result(),dtype=np.uint16),
                    psf=self._psfs[0, :],
                    image_voxel_zyx_um=self._datastore.voxel_size_zyx_um,
                    psf_voxel_zyx_um=self._datastore.voxel_size_zyx_um,
                    wavelength_um=self._datastore.load_local_wavelengths_um(
                        tile=self._tile_id,
                        round=self._round_ids[0])[1],
                    na=self._datastore.na,
                    ri=self._datastore.ri,
                    background=self._decon_background,
                )
            else:
                ref_image_decon = np.asarray(self._data_raw[0].result(),dtype=np.uint16)

            self._datastore.save_local_registered_image(
                ref_image_decon,
                tile=self._tile_id,
                deconvolution=True,
                round=self._round_ids[0]
            )

        for r_idx, round_id in enumerate(tqdm(self._round_ids[1:],desc="rounds")):

            test =  self._datastore.load_local_registered_image(
                tile=self._tile_id,
                round=round_id
            )
            if test is None:
                has_reg_decon_data = False
            else:
                has_reg_decon_data = True

            if not (has_reg_decon_data) or self._overwrite_registered:
                try:
                    temp = ref_image_decon[0:1,0:1,0:1].astype(np.float32)
                    del temp
                    gc.collect()
                except FileNotFoundError :
                    ref_image_decon = self._datastore.load_local_registered_image(
                        tile=self._tile_id,
                        round=self._round_ids[0],
                        return_future=False
                    )
                
                if self._decon:
                    mov_image_decon = chunked_cudadecon(
                        image=np.asarray(
                            self._data_raw[r_idx].result(),dtype=np.uint16
                        ),
                        psf=self._psfs[0, :],
                        image_voxel_zyx_um=self._datastore.voxel_size_zyx_um,
                        psf_voxel_zyx_um=self._datastore.voxel_size_zyx_um,
                        wavelength_um=float(self._datastore.load_local_wavelengths_um(
                            tile=self._tile_id,
                            round=self._round_ids[0])[1]
                        ),
                        na=self._datastore.na,
                        ri=self._datastore.ri,
                        n_iters=self._decon_iters,
                        background=self._decon_background,
                    )
                else:
                    mov_image_decon = np.asarray(
                        self._data_raw[r_idx].result(),dtype=np.uint16
                    )

                downsample_factor = 2
                if downsample_factor > 1:
                    ref_image_decon_ds = downsample_image_isotropic(
                        ref_image_decon, downsample_factor
                    )
                    mov_image_decon_ds = downsample_image_isotropic(
                        mov_image_decon, downsample_factor
                    )
                else:
                    ref_image_decon_ds = ref_image_decon.copy()
                    mov_image_decon_ds = mov_image_decon.copy()

                _, initial_xy_shift = compute_rigid_transform(
                    ref_image_decon_ds,
                    mov_image_decon_ds,
                    use_mask=True,
                    downsample_factor=downsample_factor,
                    projection="z",
                )

                intial_xy_transform = sitk.TranslationTransform(3, initial_xy_shift)

                mov_image_decon = apply_transform(
                    ref_image_decon, mov_image_decon, intial_xy_transform
                )

                downsample_factor = 2
                if downsample_factor > 1:
                    ref_image_decon_ds = downsample_image_isotropic(
                        ref_image_decon, downsample_factor
                    )
                    mov_image_decon_ds = downsample_image_isotropic(
                        mov_image_decon, downsample_factor
                    )
                else:
                    ref_image_decon_ds = ref_image_decon.copy()
                    mov_image_decon_ds = mov_image_decon.copy()

                _, intial_z_shift = compute_rigid_transform(
                    ref_image_decon_ds,
                    mov_image_decon_ds,
                    use_mask=False,
                    downsample_factor=downsample_factor,
                    projection="search",
                )

                intial_z_transform = sitk.TranslationTransform(3, intial_z_shift)

                mov_image_decon = apply_transform(
                    ref_image_decon, mov_image_decon, intial_z_transform
                )

                downsample_factor = 4
                if downsample_factor > 1:
                    ref_image_decon_ds = downsample_image_isotropic(
                        ref_image_decon, downsample_factor
                    )
                    mov_image_decon_ds = downsample_image_isotropic(
                        mov_image_decon, downsample_factor
                    )
                else:
                    ref_image_decon_ds = ref_image_decon.copy()
                    mov_image_decon_ds = mov_image_decon.copy()

                _, xyz_shift_4x = compute_rigid_transform(
                    ref_image_decon_ds,
                    mov_image_decon_ds,
                    use_mask=True,
                    downsample_factor=downsample_factor,
                    projection=None,
                )
                
                final_xyz_shift = (
                    np.asarray(initial_xy_shift)
                    + np.asarray(intial_z_shift)
                    + np.asarray(xyz_shift_4x)
                )
                # final_xyz_shift = np.asarray(xyz_shift_4x)
                self._datastore.save_local_rigid_xform_xyz_px(
                    rigid_xform_xyz_px=final_xyz_shift,
                    tile=self._tile_id,
                    round=round_id
                )

                xyz_transform_4x = sitk.TranslationTransform(3, xyz_shift_4x)
                mov_image_decon = apply_transform(
                    ref_image_decon, mov_image_decon, xyz_transform_4x
                )
                
                if self._perform_optical_flow:
                    downsample_factor = 3
                    if downsample_factor > 1:
                        ref_image_decon_ds = downsample_image_isotropic(
                            ref_image_decon, downsample_factor
                        )
                        mov_image_decon_ds = downsample_image_isotropic(
                            mov_image_decon, downsample_factor
                        )

                    of_xform_px = compute_optical_flow(
                        ref_image_decon_ds, 
                        mov_image_decon_ds
                    )

                    self._datastore.save_coord_of_xform_px(
                        of_xform_px=of_xform_px,
                        tile=self._tile_id,
                        downsampling=[
                            float(downsample_factor),
                            float(downsample_factor),
                            float(downsample_factor)],
                        round=round_id
                    )

                    of_xform_sitk = sitk.GetImageFromArray(
                        of_xform_px.transpose(1, 2, 3, 0).astype(np.float64),
                        isVector=True,
                    )
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
                    data_registered[data_registered < 0.0] = 0
                    data_registered = data_registered.astype(np.uint16)
                    
                    del mov_image_sitk, displacement_field
                    gc.collect()
                else:
                    mov_image_decon[mov_image_decon < 0.0] = 0
                    data_registered = mov_image_decon.astype(np.uint16)
                    
                if self.save_all_polyDT_registered:
                    self._datastore.save_local_registered_image(
                        registered_image=data_registered.astype(np.uint16),
                        tile=self._tile_id,
                        deconvolution=True,
                        round=round_id
                    )

                del data_registered
                gc.collect()

    def _apply_registration_to_bits(self):
        """
        Split self._bit_ids across all available GPUs (N = torch.cuda.device_count()).
        Then spawn up to N child processes, each binding to GPU i and processing its subset.
        Works even if N=1.
        """
        # 1) How many GPUs do we have?
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs detected. Cannot run _apply_registration_to_bits().")

        # 2) Grab all bit IDs and split into `num_gpus` chunks
        all_bits = list(self._bit_ids)
        chunk_size = (len(all_bits) + num_gpus - 1) // num_gpus  # ceiling division

        # 3) Launch one process per GPU (only as many as needed)
        processes = []
        for gpu_id in range(num_gpus):
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

    # def _apply_registration_to_bits(self):
    #     """Generate ufish + deconvolved, registered readout data and save to datastore."""
        
    #     for bit_idx, bit_id in enumerate(tqdm(self._bit_ids,desc='bits')):

    #         r_idx = self._datastore.load_local_round_linker(
    #             tile=self._tile_id,
    #             bit=bit_id
    #         )
    #         r_idx = r_idx - 1
    #         ex_wavelength_um, em_wavelength_um = self._datastore.load_local_wavelengths_um(
    #             tile=self._tile_id,
    #             bit=bit_id
    #         )
            
    #         # TO DO: hacky fix. Need to come up with a better way.
    #         if ex_wavelength_um < 600:
    #             psf_idx = 1
    #         else:
    #             psf_idx = 2

    #         test = self._datastore.load_local_registered_image(
    #             tile=self._tile_id,
    #             bit=bit_id
    #         )
            
    #         if test is None:
    #             reg_decon_data_on_disk = False
    #         else:
    #             reg_decon_data_on_disk = True


    #         if (not (reg_decon_data_on_disk) or self._overwrite_registered):
                
    #             corrected_image = self._datastore.load_local_corrected_image(
    #                 tile=self._tile_id,
    #                 bit=bit_id,
    #                 return_future=False,
    #             )

    #             if self._decon:
    #                 decon_image = chunked_cudadecon(
    #                     image=corrected_image,
    #                     psf=self._psfs[psf_idx, :],
    #                     image_voxel_zyx_um=self._datastore.voxel_size_zyx_um,
    #                     psf_voxel_zyx_um=self._datastore.voxel_size_zyx_um,
    #                     wavelength_um=em_wavelength_um,
    #                     na=self._datastore.na,
    #                     ri=self._datastore.ri,
    #                     n_iters=self._decon_iters,
    #                     background=self._decon_background,
    #                 )
    #             else:
    #                 decon_image = corrected_image.copy()

    #             if r_idx > 0:
    #                 rigid_xform_xyz_um = self._datastore.load_local_rigid_xform_xyz_px(
    #                     tile=self._tile_id,
    #                     round=self._round_ids[r_idx],
    #                 )
    #                 shift_xyz = [float(i) for i in rigid_xform_xyz_um]
    #                 xyz_transform = sitk.TranslationTransform(3, shift_xyz)

    #                 if self._perform_optical_flow:
                        
    #                     of_xform_px, _ = self._datastore.load_coord_of_xform_px(
    #                         tile=self._tile_id,
    #                         round=self._round_ids[r_idx],
    #                         return_future=False
    #                     )

    #                     of_xform_sitk = sitk.GetImageFromArray(
    #                         of_xform_px.transpose(1, 2, 3, 0).astype(np.float64),
    #                         isVector=True,
    #                     )

    #                     interpolator = sitk.sitkLinear
    #                     identity_transform = sitk.Transform(3, sitk.sitkIdentity)
                        
    #                     optical_flow_sitk = sitk.Resample(
    #                         of_xform_sitk,
    #                         sitk.GetImageFromArray(decon_image),
    #                         identity_transform,
    #                         interpolator,
    #                         0,
    #                         of_xform_sitk.GetPixelID(),
    #                     )
    #                     displacement_field = sitk.DisplacementFieldTransform(
    #                         optical_flow_sitk
    #                     )
    #                     del optical_flow_sitk, of_xform_px
    #                     gc.collect()

    #                 decon_image_rigid = apply_transform(
    #                     decon_image, 
    #                     decon_image, 
    #                     xyz_transform
    #                 )
    #                 del decon_image

    #                 if self._perform_optical_flow:
    #                     decon_bit_image_sitk = sitk.Resample(
    #                         sitk.GetImageFromArray(decon_image_rigid), 
    #                         displacement_field
    #                     )
    #                     del displacement_field

    #                     data_decon_registered = sitk.GetArrayFromImage(
    #                         decon_bit_image_sitk
    #                     ).astype(np.float32)
    #                     del decon_bit_image_sitk
    #                 else:
    #                     data_decon_registered = decon_image_rigid.copy()
    #                     del decon_image_rigid
    #                 gc.collect()

    #             else:
    #                 data_decon_registered = decon_image.copy()
    #                 del decon_image
    #                 gc.collect()
                    
    #             data_decon_registered[data_decon_registered<0.]=0.0

    #             builtins.print = _no_op
    #             ufish = UFish(device="cuda")
    #             ufish.load_weights_from_internet()

    #             ufish_localization, ufish_data = ufish.predict(
    #                 data_decon_registered, axes="zyx", blend_3d=False, batch_size=1
    #             )
    #             builtins.print = self._original_print

    #             ufish_localization = ufish_localization.rename(columns={"axis-0": "z"})
    #             ufish_localization = ufish_localization.rename(columns={"axis-1": "y"})
    #             ufish_localization = ufish_localization.rename(columns={"axis-2": "x"})

    #             del ufish
    #             gc.collect()

    #             torch.cuda.empty_cache()
    #             cp.get_default_memory_pool().free_all_blocks()
    #             gc.collect()

    #             roi_z, roi_y, roi_x = 7, 5, 5

    #             def sum_pixels_in_roi(row, image, roi_dims):
    #                 z, y, x = row["z"], row["y"], row["x"]
    #                 roi_z, roi_y, roi_x = roi_dims
    #                 z_min, y_min, x_min = (
    #                     max(0, z - roi_z // 2),
    #                     max(0, y - roi_y // 2),
    #                     max(0, x - roi_x // 2),
    #                 )
    #                 z_max, y_max, x_max = (
    #                     min(image.shape[0], z_min + roi_z),
    #                     min(image.shape[1], y_min + roi_y),
    #                     min(image.shape[2], x_min + roi_x),
    #                 )
    #                 roi = image[
    #                     int(z_min) : int(z_max),
    #                     int(y_min) : int(y_max),
    #                     int(x_min) : int(x_max),
    #                 ]
    #                 return np.sum(roi)

    #             ufish_localization["sum_prob_pixels"] = ufish_localization.apply(
    #                 sum_pixels_in_roi,
    #                 axis=1,
    #                 image=ufish_data,
    #                 roi_dims=(roi_z, roi_y, roi_x),
    #             )
    #             ufish_localization["sum_decon_pixels"] = ufish_localization.apply(
    #                 sum_pixels_in_roi,
    #                 axis=1,
    #                 image=data_decon_registered,
    #                 roi_dims=(roi_z, roi_y, roi_x),
    #             )

    #             ufish_localization["tile_idx"] = self._tile_ids.index(self._tile_id)
    #             ufish_localization["bit_idx"] = bit_idx + 1
    #             ufish_localization["tile_z_px"] = ufish_localization["z"]
    #             ufish_localization["tile_y_px"] = ufish_localization["y"]
    #             ufish_localization["tile_x_px"] = ufish_localization["x"]

    #             self._datastore.save_local_registered_image(
    #                 data_decon_registered.astype(np.uint16),
    #                 tile=self._tile_id,
    #                 deconvolution=True,
    #                 bit=bit_id
    #             )
    #             self._datastore.save_local_ufish_image(
    #                 ufish_data,
    #                 tile=self._tile_id,
    #                 bit=bit_id
    #             )
    #             self._datastore.save_local_ufish_spots(
    #                 ufish_localization,
    #                 tile=self._tile_id,
    #                 bit=bit_id
    #             )
                
    #             del (
    #                 data_decon_registered,
    #                 ufish_data,
    #                 ufish_localization,
    #             )
    #             gc.collect()
                
                
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