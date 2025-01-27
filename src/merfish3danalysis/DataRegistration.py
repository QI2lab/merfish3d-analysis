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
    downsample_image_isotropic,
)
from ufish.api import UFish
import torch
import cupy as cp
import builtins
from tqdm import tqdm

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
        decon_iters: Optional[int] = 10,
        decon_background: Optional[float] = 50.0,
    ):
    
        self._datastore = datastore
        self._tile_ids = self._datastore.tile_ids
        self._round_ids = self._datastore.round_ids
        self._bit_ids = self._datastore.bit_ids
        self._psfs = self._datastore.channel_psfs

        self._perform_optical_flow = perform_optical_flow
        self._data_raw = None
        self._has_registered_data = None
        self._overwrite_registered = overwrite_registered
        self.save_all_polyDT_registered = save_all_polyDT_registered
        self._decon_iters = decon_iters
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
            
            stage_positions.append(
                self._datastore.load_local_stage_position_zyx_um(
                    tile=self._tile_id,
                    round=round_id
                )
            )

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
                n_iters=self._decon_iters,
                background=self._decon_background,
            )

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
        """Generate ufish + deconvolved, registered readout data and save to datastore."""
        
        for bit_idx, bit_id in enumerate(tqdm(self._bit_ids,desc='bits')):

            r_idx = self._datastore.load_local_round_linker(
                tile=self._tile_id,
                bit=bit_id
            )
            r_idx = r_idx - 1
            ex_wavelength_um, em_wavelength_um = self._datastore.load_local_wavelengths_um(
                tile=self._tile_id,
                bit=bit_id
            )
            
            # TO DO: hacky fix. Need to come up with a better way.
            if ex_wavelength_um < 600:
                psf_idx = 1
            else:
                psf_idx = 2

            test = self._datastore.load_local_registered_image(
                tile=self._tile_id,
                bit=bit_id
            )
            
            if test is None:
                reg_decon_data_on_disk = False
            else:
                reg_decon_data_on_disk = True


            if (not (reg_decon_data_on_disk) or self._overwrite_registered):
                
                corrected_image = self._datastore.load_local_corrected_image(
                    tile=self._tile_id,
                    bit=bit_id,
                    return_future=False,
                )

                decon_image = chunked_cudadecon(
                    image=corrected_image,
                    psf=self._psfs[psf_idx, :],
                    image_voxel_zyx_um=self._datastore.voxel_size_zyx_um,
                    psf_voxel_zyx_um=self._datastore.voxel_size_zyx_um,
                    wavelength_um=em_wavelength_um,
                    na=self._datastore.na,
                    ri=self._datastore.ri,
                    n_iters=self._decon_iters,
                    background=self._decon_background,
                )


                if r_idx > 0:
                    rigid_xform_xyz_um = self._datastore.load_local_rigid_xform_xyz_px(
                        tile=self._tile_id,
                        round=self._round_ids[r_idx],
                    )
                    shift_xyz = [float(i) for i in rigid_xform_xyz_um]
                    xyz_transform = sitk.TranslationTransform(3, shift_xyz)

                    if self._perform_optical_flow:
                        
                        of_xform_px, _ = self._datastore.load_coord_of_xform_px(
                            tile=self._tile_id,
                            round=self._round_ids[r_idx],
                            return_future=False
                        )

                        of_xform_sitk = sitk.GetImageFromArray(
                            of_xform_px.transpose(1, 2, 3, 0).astype(np.float64),
                            isVector=True,
                        )

                        interpolator = sitk.sitkLinear
                        identity_transform = sitk.Transform(3, sitk.sitkIdentity)
                        
                        optical_flow_sitk = sitk.Resample(
                            of_xform_sitk,
                            sitk.GetImageFromArray(decon_image),
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

                    decon_image_rigid = apply_transform(
                        decon_image, 
                        decon_image, 
                        xyz_transform
                    )
                    del decon_image

                    if self._perform_optical_flow:
                        decon_bit_image_sitk = sitk.Resample(
                            sitk.GetImageFromArray(decon_image_rigid), 
                            displacement_field
                        )
                        del displacement_field

                        data_decon_registered = sitk.GetArrayFromImage(
                            decon_bit_image_sitk
                        ).astype(np.float32)
                        del decon_bit_image_sitk
                    else:
                        data_decon_registered = decon_image_rigid.copy()
                        del decon_image_rigid
                    gc.collect()

                else:
                    data_decon_registered = decon_image.copy()
                    del decon_image
                    gc.collect()
                    
                data_decon_registered[data_decon_registered<0.]=0.0
                
                print(np.mean(data_decon_registered))

                builtins.print = _no_op
                ufish = UFish(device="cuda")
                ufish.load_weights_from_internet()

                ufish_localization, ufish_data = ufish.predict(
                    data_decon_registered, axes="zyx", blend_3d=False, batch_size=1
                )
                builtins.print = self._original_print

                ufish_localization = ufish_localization.rename(columns={"axis-0": "z"})
                ufish_localization = ufish_localization.rename(columns={"axis-1": "y"})
                ufish_localization = ufish_localization.rename(columns={"axis-2": "x"})

                del ufish
                gc.collect()

                torch.cuda.empty_cache()
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

                roi_z, roi_y, roi_x = 7, 5, 5

                def sum_pixels_in_roi(row, image, roi_dims):
                    z, y, x = row["z"], row["y"], row["x"]
                    roi_z, roi_y, roi_x = roi_dims
                    z_min, y_min, x_min = (
                        max(0, z - roi_z // 2),
                        max(0, y - roi_y // 2),
                        max(0, x - roi_x // 2),
                    )
                    z_max, y_max, x_max = (
                        min(image.shape[0], z_min + roi_z),
                        min(image.shape[1], y_min + roi_y),
                        min(image.shape[2], x_min + roi_x),
                    )
                    roi = image[
                        int(z_min) : int(z_max),
                        int(y_min) : int(y_max),
                        int(x_min) : int(x_max),
                    ]
                    return np.sum(roi)

                ufish_localization["sum_prob_pixels"] = ufish_localization.apply(
                    sum_pixels_in_roi,
                    axis=1,
                    image=ufish_data,
                    roi_dims=(roi_z, roi_y, roi_x),
                )
                ufish_localization["sum_decon_pixels"] = ufish_localization.apply(
                    sum_pixels_in_roi,
                    axis=1,
                    image=data_decon_registered,
                    roi_dims=(roi_z, roi_y, roi_x),
                )

                ufish_localization["tile_idx"] = self._tile_ids.index(self._tile_id)
                ufish_localization["bit_idx"] = bit_idx + 1
                ufish_localization["tile_z_px"] = ufish_localization["z"]
                ufish_localization["tile_y_px"] = ufish_localization["y"]
                ufish_localization["tile_x_px"] = ufish_localization["x"]

                self._datastore.save_local_registered_image(
                    data_decon_registered.astype(np.uint16),
                    tile=self._tile_id,
                    deconvolution=True,
                    bit=bit_id
                )
                self._datastore.save_local_ufish_image(
                    ufish_data,
                    tile=self._tile_id,
                    bit=bit_id
                )
                self._datastore.save_local_ufish_spots(
                    ufish_localization,
                    tile=self._tile_id,
                    bit=bit_id
                )
                
                del (
                    data_decon_registered,
                    ufish_data,
                    ufish_localization,
                )
                gc.collect()
                
                
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