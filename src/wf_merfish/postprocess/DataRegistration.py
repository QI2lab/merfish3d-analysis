"""
DataRegistration: Register qi2lab widefield MERFISH data using cross-correlation and optical flow

Shepherd 2024/04 - updates to use U-FISH and remove SPOTS3D
Shepherd 2024/01 - updates for qi2lab MERFISH file format v1.0
Shepherd 2023/09 - initial commit
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional
from numpy.typing import NDArray
import zarr
import gc
from cmap import Colormap
import warnings
import SimpleITK as sitk
from numcodecs import blosc
from wf_merfish.postprocess._registration import compute_optical_flow, apply_transform, downsample_image, compute_rigid_transform
from clij2fft.richardson_lucy import richardson_lucy_nc, getlib
from clij2fft.richardson_lucy_dask import richardson_lucy_dask
import pandas as pd
from ufish.api import UFish
import dask.array as da
import torch
import cupy as cp

class DataRegistration:
    def __init__(self,
                 dataset_path: Union[str, Path],
                 overwrite_registered: bool = False,
                 perform_optical_flow: bool = False,
                 tile_idx: Optional[int] = None):
        """
        Retrieve and pre-process one tile from qi2lab 3D widefield zarr structure.
        Apply rigid and optical flow registration transformations if available.

        Parameters
        ----------
        dataset_path : Union[str, Path]
            Path to Zarr dataset
        tile_idx : int
            tile index to retrieve
        """

        self._dataset_path = dataset_path
        self._polyDT_dir_path = dataset_path / Path("polyDT")

        if tile_idx is None:
            tile_idx = 0
        self._tile_idx = tile_idx        
        self._parse_dataset()
        
        self._perform_optical_flow = perform_optical_flow
        self._data_raw = None
        self._rigid_xforms = None
        self._of_xforms = None
        self._has_registered_data = None
        self._has_rigid_registrations = False
        self._has_of_registrations = False
        self._compressor = blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.Blosc.BITSHUFFLE)
        blosc.set_nthreads(20)
        self._overwrite_registered = overwrite_registered
        self._lib = getlib()
        self._RL_mem_limit = 16

    # -----------------------------------
    # property access for class variables
    # -----------------------------------
    @property
    def dataset_path(self):
        if self._dataset_path is not None:
            return self._dataset_path
        else:
            warnings.warn('Dataset path not defined.',UserWarning)
            return None

    @dataset_path.setter
    def dataset_path(self,new_dataset_path: Union[str, Path]):
        del self.dataset_path
        self._dataset_path = new_dataset_path
        self._dataset = zarr.open_group(self._dataset_path)

    @dataset_path.deleter
    def dataset_path(self):
        if self._dataset_path is not None:
            del self._dataset_path
            del self._dataset
            self._dataset_path = None
            self._dataset = None

    @property
    def tile_idx(self):
        if self._tile_idx is not None:
            tile_idx = self._tile_idx
            return tile_idx
        else:
            warnings.warn('Tile coordinates not defined.',UserWarning)
            return None

    @tile_idx.setter
    def tile_idx(self,new_tile_idx: int):
        self._has_registrations = False
        self._tile_idx = new_tile_idx
        self._tile_id = 'tile'+str(self._tile_idx).zfill(4)

    @tile_idx.deleter
    def tile_idx(self):
        if self._tile_idx is not None:
            del self.data_raw
            del self.data_registered
            del self.rigid_xforms
            del self.of_xforms
            del self._tile_idx
            self._has_registrations = False
            self._tile_idx = None
            self._tile_id = None

    @property
    def data_raw(self):
        if self._data_raw is not None:
            return self._data_raw
        else:
            warnings.warn('Data not loaded.',UserWarning)
            return None
    
    @data_raw.deleter
    def data_raw(self):
        if self._data_raw is not None:
            del self._data_raw
            gc.collect()
            self._data_raw = None

    @property
    def data_registered(self):
        return self._data_registered
    
    @data_registered.deleter
    def data_registered(self):
        del self._data_registered
        gc.collect()
        self._data_registered= None

    @property
    def rigid_xforms(self):
        if self._total_xform is not None:
            return self._total_xform
        else:
            warnings.warn('Rigid transforms not loaded.',UserWarning)
            return None
    
    @rigid_xforms.deleter
    def rigid_xforms(self):
        if self._total_xform is not None:
            del self._total_xform
            gc.collect()
            self._total_xform = None

    @property
    def perform_optical_flow(self):
        return self._perform_optical_flow
    
    @perform_optical_flow.setter
    def perform_optical_flow(self, new_perform_optical_flow: bool):
        self._perform_optical_flow = new_perform_optical_flow

    @property
    def of_xforms(self):
        if self._of_xforms is not None:
            return self._of_xforms
        else:
            warnings.warn('Optical flow fields not loaded.',UserWarning)
            return None
    
    @of_xforms.deleter
    def of_xforms(self):
        if self._of_xforms is not None:
            del self._of_xforms
            gc.collect()
            self._of_xforms = None

    def _parse_dataset(self):
        """
        Parse dataset to discover number of tiles, number of rounds, and voxel size.
        """

        self._tile_ids = sorted([entry.name for entry in self._polyDT_dir_path.iterdir() if entry.is_dir()],
                                key=lambda x: int(x.split('tile')[1].split('.zarr')[0]))
        self._num_tiles = len(self._tile_ids)
        self._tile_id = self._tile_ids[self._tile_idx]

        current_tile_dir_path = self._polyDT_dir_path / Path(self._tile_id)
        self._round_ids = sorted([entry.name.split('.')[0]  for entry in current_tile_dir_path.iterdir() if entry.is_dir()],
                                 key=lambda x: int(x.split('round')[1].split('.zarr')[0]))
        round_id = self._round_ids[0]
        current_round_zarr_path = current_tile_dir_path / Path(round_id + ".zarr")
        current_round = zarr.open(current_round_zarr_path,mode='r')

        self._voxel_size = np.asarray(current_round.attrs['voxel_zyx_um'], dtype=np.float32)
        
        calibrations_dir_path = self._dataset_path / Path('calibrations.zarr')
        calibrations_zarr = zarr.open(calibrations_dir_path,mode='r')
        self._psfs = np.asarray(calibrations_zarr['psf_data'],dtype=np.uint16)

        del current_round, calibrations_zarr
        gc.collect()

    def load_raw_data(self):
        """
        Load raw data across rounds.
        """

        if self._tile_idx is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None

        data_raw = []
        stage_positions = []
        
        for round_id in self._round_ids:
            current_round_zarr_path = self._polyDT_dir_path / Path(self._tile_id) / Path(round_id + ".zarr")
            current_round = zarr.open(current_round_zarr_path,mode='r')
            data_raw.append(da.from_array(current_round['corrected_data']))
            stage_positions.append(np.asarray(current_round.attrs['stage_zyx_um'], dtype=np.float32))

        self._data_raw = da.stack(data_raw, axis=0)
        self._stage_positions = np.stack(stage_positions,axis=0)
        del data_raw, stage_positions, current_round
        gc.collect()

    def load_registered_data(self,
                             readouts: Optional[bool]=False,
                             data_to_read: Optional[str]=True):
        """
        If available, load registered data across rounds.
        """
        if self._tile_idx is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None
        
        data_registered = []
        
        if not(readouts):
            try:
                for round_id in self._round_ids:
                    current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(round_id + ".zarr")
                    current_round = zarr.open(current_round_path,mode='r')
                    data_registered.append(np.asarray(current_round["registered_decon_data"], dtype=np.uint16))
                            
                self._data_registered = np.stack(data_registered,axis=0)
                del data_registered, current_round
                gc.collect()
                self._has_registered_data = True

            except Exception:
                warnings.warn('Generate registered data first.',UserWarning)
                return None
        else:
            try:
                readout_dir_path = self._dataset_path / Path('readouts')
                tile_ids = sorted([entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()],
                                  key=lambda x: int(x.split('tile')[1].split('.zarr')[0]))
                tile_dir_path = readout_dir_path / Path(tile_ids[self._tile_idx])
                self._bit_ids = sorted([entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()],
                                       key=lambda x: int(x.split('bit')[1].split('.zarr')[0]))
                
                current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(self._round_ids[0] + ".zarr")
                current_round = zarr.open(current_round_path,mode='r')
                if data_to_read == 'ufish' or data_to_read == 'both':
                    data_registered.append(np.asarray(current_round["registered_decon_data"], dtype=np.float32))
                else:
                    data_registered.append(np.asarray(current_round["registered_decon_data"], dtype=np.uint16))
                
                for bit_id in self._bit_ids:
                    tile_dir_path = readout_dir_path / Path(tile_ids[self._tile_idx])
                    bit_dir_path = tile_dir_path / Path(bit_id)
                    current_bit = zarr.open(bit_dir_path,mode='r')
                    if data_to_read == 'ufish':
                        data_registered.append(np.asarray(current_bit["registered_ufish_data"], dtype=np.float32))
                    elif data_to_read == 'decon':
                        data_registered.append(np.asarray(current_bit["registered_decon_data"], dtype=np.uint16))
                    elif data_to_read == 'both':
                        data_registered.append(np.asarray(current_bit["registered_ufish_data"], dtype=np.float32))
                        data_registered.append(np.asarray(current_bit["registered_decon_data"], dtype=np.float32))
                    
                self._data_registered = np.stack(data_registered,axis=0)
                del data_registered, current_round
                gc.collect()
                self._has_registered_data = True
            except Exception:
                warnings.warn('Generate registered data first.',UserWarning)
                return None
 
    def generate_registrations(self):
        """
        Generate registration transforms using reference channel.
        Use cross-correlation translation in stages followed by optional optical flow refinement.
        """

        if self._tile_idx is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None
        
        if self._data_raw is None:
            self.load_raw_data()
        
        current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(self._round_ids[0] + ".zarr")
        current_round = zarr.open(current_round_path,mode='a')
        
        try:
            data_reg_zarr = current_round['registered_decon_data']
            ref_image_decon = np.asarray(data_reg_zarr,dtype=np.uint16)
            if np.mean(ref_image_decon) == 0.0:
                has_reg_decon_data = False
            else:
                has_reg_decon_data = True
        except:
            has_reg_decon_data = False
            
        if not(has_reg_decon_data) or self._overwrite_registered:
            ref_image_decon = richardson_lucy_dask(np.asarray(self._data_raw[0,:].compute().astype(np.uint16)),
                                                psf=self._psfs[0,:],
                                                numiterations=500,
                                                regularizationfactor=1e-4,
                                                mem_to_use=self._RL_mem_limit)
                
            try:
                data_reg_zarr = current_round.zeros('registered_decon_data',
                                                shape=ref_image_decon.shape,
                                                chunks=(1,ref_image_decon.shape[1],ref_image_decon.shape[2]),
                                                compressor=self._compressor,
                                                dtype=np.uint16)
                data_reg_zarr[:] = ref_image_decon
            except Exception:
                data_reg_zarr = current_round['registered_decon_data']
                data_reg_zarr[:] = ref_image_decon
                        
        ref_image_sitk = sitk.GetImageFromArray(ref_image_decon.astype(np.float32))
        
        del current_round_path, current_round, data_reg_zarr
        gc.collect()

        for r_idx, round_id in enumerate(self._round_ids[1:]):
            r_idx = r_idx + 1
            current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(round_id + ".zarr")
            current_round = zarr.open(current_round_path,mode='a')
            psf_idx = current_round.attrs["psf_idx"]
            
            try:
                data_reg_zarr = current_round['registered_decon_data']
                mov_image_decon = np.asarray(data_reg_zarr,dtype=np.uint16)
                test = mov_image_decon[0:1,0:1,0:1]
                has_reg_decon_data = True
            except:
                has_reg_decon_data = False
            
            if not(has_reg_decon_data) or self._overwrite_registered:
                mov_image_decon = richardson_lucy_dask(self._data_raw[r_idx,:].compute().astype(np.uint16),
                                                        psf=self._psfs[psf_idx,:],
                                                        numiterations=500,
                                                        regularizationfactor=1e-5,
                                                mem_to_use=self._RL_mem_limit)

                mov_image_sitk = sitk.GetImageFromArray(mov_image_decon.astype(np.float32))
                                    
                downsample_factor = 2
                if downsample_factor > 1:
                    ref_ds_image_sitk = downsample_image(ref_image_sitk, downsample_factor)
                    mov_ds_image_sitk = downsample_image(mov_image_sitk, downsample_factor)
                else:
                    ref_ds_image_sitk = ref_image_sitk
                    mov_ds_image_sitk = mov_image_sitk
                    
                _, initial_xy_shift = compute_rigid_transform(ref_ds_image_sitk, 
                                                                mov_ds_image_sitk,
                                                                use_mask=True,
                                                                downsample_factor=downsample_factor,
                                                                projection='z')
                
                intial_xy_transform = sitk.TranslationTransform(3, initial_xy_shift)

                mov_image_sitk = apply_transform(ref_image_sitk,
                                                    mov_image_sitk,
                                                    intial_xy_transform)
                
                del ref_ds_image_sitk
                gc.collect()
                
                downsample_factor = 2
                if downsample_factor > 1:
                    ref_ds_image_sitk = downsample_image(ref_image_sitk, downsample_factor)
                    mov_ds_image_sitk = downsample_image(mov_image_sitk, downsample_factor)
                else:
                    ref_ds_image_sitk = ref_image_sitk
                    mov_ds_image_sitk = mov_image_sitk
                    
                _, intial_z_shift = compute_rigid_transform(ref_ds_image_sitk, 
                                                            mov_ds_image_sitk,
                                                            use_mask=True,
                                                            downsample_factor=downsample_factor,
                                                            projection='search')
                
                intial_z_transform = sitk.TranslationTransform(3, intial_z_shift)

                mov_image_sitk = apply_transform(ref_image_sitk,
                                                mov_image_sitk,
                                                intial_z_transform)
                
                del ref_ds_image_sitk
                gc.collect()
                
                downsample_factor = 4
                if downsample_factor > 1:
                    ref_ds_image_sitk = downsample_image(ref_image_sitk, downsample_factor)
                    mov_ds_image_sitk = downsample_image(mov_image_sitk, downsample_factor)
                else:
                    ref_ds_image_sitk = ref_image_sitk
                    mov_ds_image_sitk = mov_image_sitk
                                    
                _, xyz_shift_4x = compute_rigid_transform(ref_ds_image_sitk, 
                                                            mov_ds_image_sitk,
                                                            use_mask=True,
                                                            downsample_factor=downsample_factor,
                                                            projection=None)
        
                
                final_xyz_shift = np.asarray(initial_xy_shift) + np.asarray(intial_z_shift) + np.asarray(xyz_shift_4x)                        
                # final_xyz_shift = np.asarray(xyz_shift_4x)
                current_round.attrs["rigid_xform_xyz_px"] = final_xyz_shift.tolist()
                
                xyz_transform_4x = sitk.TranslationTransform(3, xyz_shift_4x)
                mov_image_sitk = apply_transform(ref_image_sitk,
                                                        mov_image_sitk,
                                                        xyz_transform_4x)
                del ref_ds_image_sitk
                gc.collect()
                
                if self._perform_optical_flow:
                        
                    downsample_factor = 3
                    if downsample_factor > 1:
                        ref_ds_image_sitk = downsample_image(ref_image_sitk, downsample_factor)
                        mov_ds_image_sitk = downsample_image(mov_image_sitk, downsample_factor)
                    else:
                        mov_ds_image_sitk = mov_image_sitk

                    ref_ds_image = sitk.GetArrayFromImage(ref_ds_image_sitk).astype(np.float32)
                    mov_ds_image = sitk.GetArrayFromImage(mov_ds_image_sitk).astype(np.float32)
                    del ref_ds_image_sitk, mov_ds_image_sitk
                    gc.collect()
                    
                    of_xform_3x_px = compute_optical_flow(ref_ds_image,mov_ds_image)
                    del ref_ds_image, mov_ds_image
                    gc.collect()

                    try:
                        of_xform_zarr = current_round.zeros('of_xform_3x_px',
                                                        shape=of_xform_3x_px.shape,
                                                        chunks=(1,1,of_xform_3x_px.shape[2],of_xform_3x_px.shape[3]),
                                                        compressor=self._compressor,
                                                        dtype=np.float32)
                    except Exception:
                        of_xform_zarr = current_round['of_xform_3x_px']
                    
                    of_xform_zarr[:] = of_xform_3x_px
                    
                    of_3x_sitk = sitk.GetImageFromArray(of_xform_3x_px.transpose(1, 2, 3, 0).astype(np.float64),
                                                            isVector = True)
                    interpolator = sitk.sitkLinear
                    identity_transform = sitk.Transform(3, sitk.sitkIdentity)
                    optical_flow_sitk = sitk.Resample(of_3x_sitk, mov_image_sitk, identity_transform, interpolator,
                                                0, of_3x_sitk.GetPixelID())
                    displacement_field = sitk.DisplacementFieldTransform(optical_flow_sitk)
                    del of_3x_sitk, of_xform_3x_px
                    gc.collect()
                    
                    # apply optical flow 
                    mov_image_sitk = sitk.Resample(mov_image_sitk,displacement_field)
                    data_registered = sitk.GetArrayFromImage(mov_image_sitk).astype(np.uint16)

                    del optical_flow_sitk, displacement_field
                    gc.collect()
                else:
                    data_registered = sitk.GetArrayFromImage(mov_image_sitk).astype(np.uint16)
                    
                try:
                    data_reg_zarr = current_round.zeros('registered_decon_data',
                                                    shape=data_registered.shape,
                                                    chunks=(1,data_registered.shape[1],data_registered.shape[2]),
                                                    compressor=self._compressor,
                                                    dtype=np.uint16)
                except Exception:
                    data_reg_zarr = current_round['registered_decon_data']
                
                data_reg_zarr[:] = data_registered
                
                del data_registered, mov_image_sitk
                gc.collect()
                
        del ref_image_sitk
        gc.collect()
                        
    def apply_registration_to_bits(self):
        """
        Generate registered data and save to zarr.
        """
       
        if self._tile_idx is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None
               
        if not(self._has_rigid_registrations):
            self.load_rigid_registrations()
            if not(self._has_rigid_registrations):
                raise Exception("Create rigid registrations first.")

        if self._perform_optical_flow and not(self._has_of_registrations):
            self.load_opticalflow_registrations()
            if not(self._has_of_registrations):
                raise Exception("Create vector field registrations first.")
            

        readout_dir_path = self._dataset_path / Path('readouts')
        tile_dir_path = readout_dir_path / Path(self._tile_id)
        bit_ids = sorted([entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()],
                         key=lambda x: int(x.split('bit')[1].split('.zarr')[0]))
                
        tile_dir_path = readout_dir_path / Path(self._tile_id)
        
        localization_output_dir_path = self._dataset_path / Path('ufish_localizations')
        localization_output_dir_path.mkdir(parents=True, exist_ok=True)
        
        round_list = []
        for bit_id in bit_ids:
            bit_dir_path = tile_dir_path / Path(bit_id)
            current_bit_channel = zarr.open(bit_dir_path,mode='a')      
            round_list.append(int(current_bit_channel.attrs['round']))
            
        round_list = np.array(round_list)
        ref_idx = int(np.argwhere(round_list==0)[0])
        
        ref_bit_id = bit_ids[ref_idx]
        bit_dir_path = tile_dir_path / Path(ref_bit_id)
        reference_bit_channel = zarr.open(bit_dir_path,mode='a')      
    
        ref_bit_sitk = sitk.GetImageFromArray(reference_bit_channel['raw_data'].astype(np.float32))
            
        del reference_bit_channel
        gc.collect()
        
        for bit_idx, bit_id in enumerate(bit_ids):
            bit_dir_path = tile_dir_path / Path(bit_id)
            current_bit_channel = zarr.open(bit_dir_path,mode='a')      
            r_idx = int(current_bit_channel.attrs['round'])
            psf_idx = int(current_bit_channel.attrs['psf_idx'])
            gain = float(current_bit_channel.attrs['gain'])
            em_wvl = float(current_bit_channel.attrs['emission_um'])
            
            try:
                data_decon_zarr = current_bit_channel['registered_decon_data']
                decon_image = np.asarray(data_decon_zarr,dtype=np.uint16)   
                reg_decon_data_on_disk = True 
            except:
                reg_decon_data_on_disk = False
                                
        
            if not(reg_decon_data_on_disk) or self._overwrite_registered:
                decon_image = richardson_lucy_dask(np.asarray(current_bit_channel['raw_data']),
                                                    psf=self._psfs[psf_idx,:],
                                                    numiterations=40,
                                                    regularizationfactor=.001,
                                                mem_to_use=self._RL_mem_limit)

                if r_idx > 0:
                    polyDT_tile_round_path = self._dataset_path / Path('polyDT') / Path(self._tile_id) / Path('round'+str(r_idx).zfill(3)+'.zarr')
                    current_polyDT_channel = zarr.open(polyDT_tile_round_path,mode='r')
                    
                    rigid_xform_xyz_um = np.asarray(current_polyDT_channel.attrs['rigid_xform_xyz_px'],dtype=np.float32)
                    shift_xyz = [float(i) for i in rigid_xform_xyz_um]
                    xyx_transform = sitk.TranslationTransform(3, shift_xyz)           
                    
                    if self._perform_optical_flow:
                        of_xform_3x_px_xyz = np.asarray(current_polyDT_channel['of_xform_3x_px'],dtype=np.float32)
                        of_3x_sitk = sitk.GetImageFromArray(of_xform_3x_px_xyz.transpose(1, 2, 3, 0).astype(np.float64),
                                                                    isVector = True)
                        interpolator = sitk.sitkLinear
                        identity_transform = sitk.Transform(3, sitk.sitkIdentity)
                        optical_flow_sitk = sitk.Resample(of_3x_sitk, ref_bit_sitk, identity_transform, interpolator,
                                                        0, of_3x_sitk.GetPixelID())
                        displacement_field = sitk.DisplacementFieldTransform(optical_flow_sitk)
                        del rigid_xform_xyz_um, shift_xyz, of_xform_3x_px_xyz, of_3x_sitk, optical_flow_sitk
                        gc.collect()
                        
                    decon_bit_image_sitk = apply_transform(ref_bit_sitk,
                                                        sitk.GetImageFromArray(decon_image),
                                                        xyx_transform)
                    
                    if self._perform_optical_flow:

                        decon_bit_image_sitk = sitk.Resample(decon_bit_image_sitk,displacement_field)                        
                        del displacement_field
                    
                    data_decon_registered = sitk.GetArrayFromImage(decon_bit_image_sitk).astype(np.float32)
                    del decon_bit_image_sitk
                    gc.collect()
                    
                else:
                    data_decon_registered = decon_image.copy()
                    gc.collect()
                
                del decon_image
                
                ufish = UFish(device='cuda')
                ufish.load_weights_from_internet()
            
                ufish_localization, ufish_data = ufish.predict(data_decon_registered,
                                                                axes='zyx',
                                                                blend_3d=False,
                                                                batch_size=1)
                
                ufish_localization = ufish_localization.rename(columns={'axis-0': 'z'})
                ufish_localization = ufish_localization.rename(columns={'axis-1': 'y'})
                ufish_localization = ufish_localization.rename(columns={'axis-2': 'x'})
                
                del ufish
                gc.collect()
                
                torch.cuda.empty_cache()
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                

                roi_z, roi_y, roi_x = 7, 5, 5 

                def sum_pixels_in_roi(row, image, roi_dims):
                    z, y, x = row['z'], row['y'], row['x']
                    roi_z, roi_y, roi_x = roi_dims
                    z_min, y_min, x_min = max(0, z - roi_z // 2), max(0, y - roi_y // 2), max(0, x - roi_x // 2)
                    z_max, y_max, x_max = min(image.shape[0], z_min + roi_z), min(image.shape[1], y_min + roi_y), min(image.shape[2], x_min + roi_x)
                    roi = image[int(z_min):int(z_max), int(y_min):int(y_max), int(x_min):int(x_max)]
                    return np.sum(roi)
                
                ufish_localization['sum_prob_pixels'] = ufish_localization.apply(sum_pixels_in_roi, axis=1, image=ufish_data, roi_dims=(roi_z, roi_y, roi_x))
                ufish_localization['sum_decon_pixels'] = ufish_localization.apply(sum_pixels_in_roi, axis=1, image=data_decon_registered, roi_dims=(roi_z, roi_y, roi_x))
                ufish_localization['tile_idx'] = self._tile_idx
                ufish_localization['bit_idx'] = bit_idx + 1
                ufish_localization['tile_z_px'] = ufish_localization['z'] 
                ufish_localization['tile_y_px'] = ufish_localization['y'] 
                ufish_localization['tile_x_px'] = ufish_localization['x'] 
                
                localization_parquet_dir_path = localization_output_dir_path / Path(self._tile_id) 
                localization_parquet_dir_path.mkdir(parents=True, exist_ok=True)
                localization_parquet_path = localization_parquet_dir_path / Path(Path(bit_id).stem+".parquet")
                ufish_localization.to_parquet(localization_parquet_path)
                
                try:
                    data_decon_reg_zarr = current_bit_channel.zeros('registered_decon_data',
                                                            shape=data_decon_registered.shape,
                                                            chunks=(1,data_decon_registered.shape[1],data_decon_registered.shape[2]),
                                                            compressor=self._compressor,
                                                            dtype=np.uint16)
                    ufish_reg_zarr = current_bit_channel.zeros('registered_ufish_data',
                                                            shape=ufish_data.shape,
                                                            chunks=(1,ufish_data.shape[1],ufish_data.shape[2]),
                                                            compressor=self._compressor,
                                                            dtype=np.float32)
                    
                except Exception:
                    data_decon_reg_zarr = current_bit_channel['registered_decon_data']
                    ufish_reg_zarr = current_bit_channel['registered_ufish_data']

                data_decon_reg_zarr[:] = data_decon_registered.astype(np.uint16)
                ufish_reg_zarr[:] = ufish_data.astype(np.float32)
                del data_decon_registered, ufish_data, ufish_localization
                gc.collect()

    def load_rigid_registrations(self):
        """
        Load rigid registrations.
        """

        if self._tile_idx is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None

        rigid_xform = []
        self._has_rigid_registrations = False

        try:
            for round_id in self._round_ids[1:]:
                current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(round_id + ".zarr")
                current_round = zarr.open(current_round_path,mode='r')
                
                rigid_xform.append(np.asarray(current_round.attrs['rigid_xform_xyz_px'],dtype=np.float32))
            self._rigid_xforms = np.stack(rigid_xform,axis=0)
            self._has_rigid_registrations = True
            del rigid_xform
            gc.collect()
        except Exception:
            self._rigid_xforms = None
            self._has_rigid_registrations = False

    def load_opticalflow_registrations(self):
        """"
        Load optical flow registrations.
        """

        of_xform = []
        self._has_of_registrations = False

        try:
            for round_id in self._round_ids[1:]:
                current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(round_id + ".zarr")
                current_round = zarr.open(current_round_path,mode='r')
                
                of_xform.append(np.asarray(current_round['of_xform_3x_px'],dtype=np.float32))
                self._of_xforms = np.stack(of_xform,axis=0)
            self._has_of_registrations = True
            del of_xform
            gc.collect()
        except Exception:
            self._of_xforms = None
            self._has_of_registrations = False

    def create_figure(self,
                      readouts: Optional[bool] = False,
                      data_to_display: Optional[str] = 'ufish'):
        """
        Generate napari figure for debugging
        """
        import napari
        from qtpy.QtWidgets import QApplication

        def on_close_callback():
            viewer.layers.clear()
            gc.collect()
        
        viewer = napari.Viewer()
        app = QApplication.instance()

        app.lastWindowClosed.connect(on_close_callback)
        if not(readouts):
            viewer.window._qt_window.setWindowTitle(self._tile_ids[self._tile_idx] + ' alignment aross rounds')
        else:
            viewer.window._qt_window.setWindowTitle(self._tile_ids[self._tile_idx] + ' alignment aross bits')
        
        colormaps = [Colormap('cmap:white'),
                     Colormap('cmap:cyan'),
                     Colormap('cmap:yellow'),
                     Colormap('cmap:red'),
                     Colormap('cmap:green'),
                     Colormap('chrisluts:OPF_Fresh'),
                     Colormap('chrisluts:OPF_Orange'),
                     Colormap('chrisluts:OPF_Purple'),
                     Colormap('chrisluts:BOP_Blue'),
                     Colormap('chrisluts:BOP_Orange'),
                     Colormap('chrisluts:BOP_Purple'),
                     Colormap('cmap:cyan'),
                     Colormap('cmap:yellow'),
                     Colormap('cmap:red'),
                     Colormap('cmap:green'),
                     Colormap('chrisluts:OPF_Fresh'),
                     Colormap('chrisluts:OPF_Orange'),
                     Colormap('cmap:magenta')]
        
        if not(readouts):
            for idx in range(len(self._round_ids)):
                viewer.add_image(data=self._data_registered[idx],
                                name=self._round_ids[idx],
                                scale=self._voxel_size,
                                blending='additive',
                                colormap=colormaps[idx].to_napari())
        else:
            viewer.add_image(data=self._data_registered[0],
                name='polyDT',
                scale=self._voxel_size,
                blending='additive')
            if data_to_display == 'ufish' or data_to_display == 'decon':
                for idx in range(len(self._bit_ids)):
                    viewer.add_image(data=self._data_registered[idx+1],
                                    name=self._bit_ids[idx]+'_'+data_to_display,
                                    scale=self._voxel_size,
                                    blending='additive',
                                    colormap=colormaps[idx].to_napari())
            else:
                data_idx = 1
                for idx in range(len(self._bit_ids)):
                    ufish_prediction = np.where(self._data_registered[data_idx]>.01,1,0)
                    data_idx = data_idx + 1
                    viewer.add_image(data=ufish_prediction * self._data_registered[data_idx],
                                    name=self._bit_ids[idx]+'_predict',
                                    scale=self._voxel_size,
                                    blending='additive',
                                    colormap=colormaps[idx].to_napari())
                    data_idx = data_idx + 1

        napari.run()