"""
DataRegistration: Register qi2lab widefield MERFISH data using cross-correlation and optical flow

Shepherd 2024/01 - updates for qi2lab MERFISH file format v1.0
Shepherd 2023/09 - initial commit
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional
import zarr
import gc
from cmap import Colormap
import warnings
import SimpleITK as sitk
from numcodecs import blosc
from wf_merfish.postprocess._registration import compute_optical_flow, apply_transform, downsample_image, compute_rigid_transform

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
            tile ID to retrieve
        """

        self._dataset_path = dataset_path
        self._dataset = zarr.open_group(self._dataset_path)

        if tile_idx is None:
            tile_idx = 0
        self._tile_idx = tile_idx
        self._tile_id = 'tile'+str(tile_idx).zfill(4)
        
        self._parse_dataset()
        
        self._perform_optical_flow = perform_optical_flow
        self._data_raw = None
        self._rigid_xforms = None
        self._of_xforms = None
        self._has_registered_data = None
        self._compressor = blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.Blosc.BITSHUFFLE)
        blosc.set_nthreads(6)
        self._overwrite_registered = overwrite_registered

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

        tile_ids = list(self._dataset.group_keys())
        self._num_tiles = len(tile_ids)

        current_tile = zarr.open_group(self._dataset_path,mode='r',path=self._tile_idx)
        self._round_ids = list(current_tile.group_keys())
        round_id = self._round_ids[0]
        current_round = zarr.open_group(self._dataset_path,mode='r',
                                        path=self._tile_idx + "/" + round_id)

        self._voxel_size = np.asarray(current_tile['voxel_zyx_um'], dtype=np.float32)

        del current_round, current_tile
        gc.collect()

    def load_raw_data(self):
        """
        Load raw data across rounds.
        """

        if self._tile_coordinates is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None

        data_raw = []
        stage_positions = []
        
        for round_id in self._round_ids:
            current_round = zarr.open_group(self._dataset_path,mode='r',path=self._tile_idx+'/'+round_id)
            data_raw.append(np.asarray(current_round['raw_data'], dtype=np.uint16))
            stage_positions.append(np.asarray(current_round.attrs['stage_position'], dtype=np.float32))

        self._data_raw = np.stack(data_raw, axis=0)
        self._stage_positions = np.stack(stage_positions,axis=0)
        del data_raw, stage_positions
        gc.collect()

    def load_registered_data(self):
        """
        If available, load registered data across rounds.
        """
        if self._tile_coordinates is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None
        
        data_registered = []
        try:
            for round_id in self._round_ids:
                current_round = zarr.open_group(self._dataset_path,mode='r',path=self._tile_idx+'/'+round_id)
                data_registered.append(np.asarray(current_round["registered_data"], dtype=np.uint16))
                        
            self._data_registered = np.stack(data_registered,axis=0)
            del data_registered
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
        
        ref_image_sitk = sitk.GetImageFromArray(self._data_raw[0,:].astype(np.float32))

        for round_id in self._round_ids[1:]:
            current_round = zarr.open_group(self._dataset_path,mode='a',
                                            path=self._tile_id+'/'+round_id)
            try:
                test_reg = np.asarray(current_round['registered_data'], dtype=np.uint16)
            except Exception:
                has_registered = False
            else:
                has_registered = True
            del test_reg
            gc.collect()
                
            if (self._overwrite_registered and has_registered) or not(has_registered):
                mov_image_sitk = sitk.GetImageFromArray(self._data_raw[round_id,:].astype(np.float32))
                                
                downsample_factor = 4
                if downsample_factor > 1:
                    ref_ds_image_sitk = downsample_image(ref_image_sitk, downsample_factor)
                    mov_ds_image_sitk = downsample_image(mov_image_sitk, downsample_factor)
                else:
                    mov_ds_image_sitk = mov_image_sitk
                    
                _, xyz_shift = compute_rigid_transform(ref_ds_image_sitk, 
                                                       mov_ds_image_sitk,
                                                       use_mask=False,
                                                       downsample_factor=downsample_factor,
                                                       projection=None)
                                
                current_round.attrs["rigid_xform_xyz_um"] = xyz_shift.tolist()

                final_transform = sitk.TranslationTransform(3, xyz_shift)
                mov_translation_sitk = apply_transform(ref_image_sitk,
                                                       mov_image_sitk,
                                                       final_transform)
                del ref_image_sitk, mov_image_sitk
                gc.collect()
                

                if self._perform_optical_flow:
                    downsample_factor = 4
                    if downsample_factor > 1:
                        mov_ds_image_sitk = downsample_image(mov_translation_sitk, downsample_factor)
                    else:
                        mov_ds_image_sitk = mov_translation_sitk

                    ref_ds_image = sitk.GetArrayFromImage(ref_ds_image_sitk).astype(np.float32)
                    mov_ds_image = sitk.GetArrayFromImage(mov_ds_image_sitk).astype(np.float32)
                    del ref_ds_image_sitk, mov_ds_image_sitk, mov_translation_sitk
                    gc.collect()
                    
                    of_xform_4x = compute_optical_flow(ref_ds_image,mov_ds_image)
                    del ref_ds_image, mov_ds_image
                    gc.collect()

                    try:
                        of_xform_zarr = current_round.zeros('of_xform_4x',
                                                        shape=(of_xform_4x.shape[0],
                                                            of_xform_4x.shape[1],
                                                            of_xform_4x.shape[2]),
                                                        chunks=(1,of_xform_4x.shape[1],of_xform_4x.shape[2]),
                                                        compressor=self._compressor,
                                                        dtype=np.float32)
                    except Exception:
                        of_xform_zarr = current_round['of_xform_4x']
                    
                    of_xform_zarr[:] = of_xform_4x

        self._has_rigid_registrations=True
        if self.perform_optical_flow:
            self._has_of_registrations=True
                
    def apply_registrations(self):
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
                self.generate_registrations()

        if self._perform_optical_flow and not(self._has_of_registrations):
            self.load_opticalflow_registrations()
            if not(self._has_of_registrations):
                self.generate_registrations()
  
        for r_idx, round_id in enumerate(self._round_ids):
            current_round = zarr.open_group(self._dataset_path,mode='a',
                                            path=self._tile_id+'/'+round_id)
            if round_id == self._round_ids[0]:
                data_registered = self._raw_data[r_idx,:]
                ref_image_sitk = sitk.GetImageFromArray(self._data_raw[0,:].astype(np.float32))
            else:
                mov_image_sitk = sitk.GetImageFromArray(self._data_raw[r_idx,:].astype(np.float32))
                rigid_xform = np.asarray(current_round.attrs["rigid_xform_xyz_um"],
                                                     dtype=np.float32)
                total_shift_xyz = [float(i) for i in rigid_xform]
                final_transform = sitk.TranslationTransform(3, total_shift_xyz)

                mov_translation_sitk = apply_transform(ref_image_sitk,
                                                       mov_image_sitk,
                                                       final_transform)
                
                del final_transform
                gc.collect()

                if self.perform_optical_flow:
                    of_xform_4x = self._of_xform_4x[0,:]
                    # prepare 4x downsample optical flow for sitk, upsample, and create transform
                    of_4x_sitk = sitk.GetImageFromArray(of_xform_4x.transpose(1, 2, 3, 0).astype(np.float64),
                                                        isVector = True)
                    final_shape = mov_translation_sitk.GetSize()
                    optical_flow_sitk = sitk.Resample(of_4x_sitk,final_shape)
                    displacement_field = sitk.DisplacementFieldTransform(optical_flow_sitk)
                    del of_4x_sitk, of_xform_4x
                    gc.collect()
                    
                    # apply optical flow 
                    mov_translation_sitk = sitk.Resample(mov_translation_sitk,displacement_field)

                    del optical_flow_sitk, displacement_field
                    gc.collect()

                data_registered = sitk.GetArrayFromImage(mov_translation_sitk).astype(np.uint16)
                del mov_translation_sitk

            try:
                data_reg_zarr = current_round.zeros('registered_data',
                                                shape=(data_registered.shape[0],
                                                    data_registered.shape[1],
                                                    data_registered.shape[2]),
                                                chunks=(1,data_registered.shape[1],data_registered.shape[2]),
                                                compressor=self._compressor,
                                                dtype=np.uint16)
            except Exception:
                data_reg_zarr = current_round['registered_data']
            
            data_reg_zarr[:] = data_registered
                
    def load_rigid_registrations(self):
        """
        Load rigid registrations.
        """

        if self._tile_idx is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None

        total_xform = []
        self._has_rigid_registrations = False

        try:
            for round_id in range(self._num_r):
                current_round = zarr.open_group(self._dataset_path,mode='a',
                                            path=self._tile_id+'/'+round_id)
                
                total_xform.append(np.asarray(current_round.attrs['rigid_xform_xyz_um'],dtype=np.float32))
        except Exception:
            self._total_xform = None
            self._has_rigid_registrations = False
        else:
            self._total_xform = np.stack(total_xform,axis=0)
            self._has_rigid_registrations = True
        del total_xform
        gc.collect()

    def load_opticalflow_registrations(self):
        """"
        Load optical flow registrations.
        """

        of_xform = []
        self._has_of_registrations = False

        try:
            for round_id in range(self._num_r):
                current_round = zarr.open_group(self._dataset_path,mode='a',
                                            path=self._tile_id+'/'+round_id)
                
                of_xform.append(np.asarray(current_round['of_xform_xyz_um'],dtype=np.float32))
        except Exception:
            self._of_xform = None
            self._has_of_registrations = False
        else:
            self._of_xform = np.stack(of_xform,axis=0)
            self._has_of_registrations = True
        del of_xform
        gc.collect()

    # def export_tiled_tiffs(self,
    #                        tile_size: int = 425,
    #                        z_downsample: int = 2):
    #     """
    #     Export all RNA rounds, in codebook order, as small tiffs
    #     """
        
    #     import tifffile
    #     from skimage.transform import downscale_local_mean
    #     from functools import partial
    #     import json

    #     def write_tiled_tiffs(data_tile,
    #                           z_downsample: int,
    #                           stage_position: Sequence[float],
    #                           output_path: Union[str,Path],
    #                           block_info: dict = None):
    #         """
    #         Function to process and write tiff tiles and associated metadata.

    #         Parameters
    #         ----------
    #         data_tile: np.ndarray
    #             tile from map_overlap
    #         z_dowsample: int
    #             z downsampling factor
    #         stage_position: Sequene[float]
    #             zyx stage positions
    #         output_path: Union[str,Path]
    #             where to place the image
    #         block_info: dict
    #             block metadata from map_overlap

    #         Returns
    #         -------
    #         temp: np.ndarray
    #             fake image to make sure somethign is
    #             returned to map_overlap.
    #             TO DO: improve handling this so the code doesn't error
    #         """
            
    #         # helpful diagnostic code for checking tile metadata
    #         # print("========================")
    #         # print(block_info[0]['array-location'])
    #         # print(block_info[0]['chunk-location'])
    #         # print("========================")

    #         x_min = block_info[0]['array-location'][3][0] 
    #         x_max = block_info[0]['array-location'][3][1]
    #         y_min = block_info[0]['array-location'][2][0]
    #         y_max = block_info[0]['array-location'][2][1]
    #         z_min = block_info[0]['array-location'][1][0]
    #         z_max = block_info[0]['array-location'][1][1]

    #         x = block_info[0]['chunk-location'][3]
    #         y = block_info[0]['chunk-location'][2]
    #         z = block_info[0]['chunk-location'][1]

    #         stage_x = np.round(float(stage_position[0] + .115*(x_max-x_min)/2),2)
    #         stage_y = np.round(float(stage_position[1] + .115*(y_max-y_min)/2),2)
    #         stage_z = np.round(float(stage_position[2] + .230),2)

    #         downsample = downscale_local_mean(data_tile,(1,z_downsample,1,1),cval=0.0).astype(np.uint16)
    #         file_path = output_path / Path('subtile_x'+str(x).zfill(4)+'_y'+str(y).zfill(4)+'.tiff')
    #         metadata_path_json = output_path / Path('subtile_x'+str(x).zfill(4)+'_y'+str(y).zfill(4)+'.json')
    #         metadata_path_txt = output_path / Path('subtile_x'+str(x).zfill(4)+'_y'+str(y).zfill(4)+'.txt')
    #         tile_metadata = {'subtile_id': 'subtile_x'+str(x).zfill(4)+'_y'+str(y).zfill(4)+'.tiff',
    #                          'start_x_pixel': x_min,
    #                          'end_x_pixel': x_max,
    #                          'start_y_pixel': y_min,
    #                          'end_y_pixel': y_max,
    #                          'start_z_pixel': z_min+1,
    #                          'end_z_pixel': z_max,
    #                         'stage_x': stage_x,
    #                         'stage_y': stage_y,
    #                         'stage_z': stage_z}
            
    #         tifffile.imwrite(output_path/file_path,
    #                          downsample[:,1:,:,:])
            
    #         with metadata_path_json.open('w') as f:
    #             json.dump(tile_metadata,f)

    #         df = pd.DataFrame(tile_metadata, index=[0]).T
    #         df.to_csv(metadata_path_txt, sep='\t', header=False)

    #         return np.array([1,1,1,1],dtype=np.uint8)
        
    #     # setup paths, metadata, codebook
    #     path_parts = self._dataset_path.parts
    #     tile_path = Path('x'+str(self._tile_coordinates[0]).zfill(3)\
    #                      +'_y'+str(self._tile_coordinates[1]).zfill(3)\
    #                      +'_z'+str(self._tile_coordinates[2]).zfill(3))
    #     output_path = Path(path_parts[0]) / Path(path_parts[1]) / Path(path_parts[2]) / Path("tiled_tiffs") / tile_path
    #     output_path.mkdir(parents=True, exist_ok=True)
    #     data_path = output_path / Path("data")
    #     data_path.mkdir(parents=True, exist_ok=True)
    #     metadata_path = output_path
    #     codebook_path_json = metadata_path / Path('codebook.json')
    #     codebook_path_txt = metadata_path / Path('codebook.txt')
    #     metadata_path_json = metadata_path / Path('exp_metadata.json')
    #     metadata_path_txt = metadata_path / Path('exp_metadata.txt')

    #     # generate registered data
    #     self.generate_registered_data()
                     
    #     # determine number of blocks in scan direction for tile size
    #     y_crop_index_max = (self.data_registered.shape[2]-1) // 425 * 425

    #     # reorder registered data in codebook order
    #     ordered_data = da.zeros((16,
    #                              self.data_registered.shape[1],
    #                              y_crop_index_max,
    #                              self.data_registered.shape[3]),
    #                             dtype=self.data_registered.dtype)
    #     idx = 0
    #     for key in self.exp_order:
    #         ordered_data[key-1,:] = self.data_registered[idx,:,0:y_crop_index_max,:]
    #         idx += 1
            
    #     ordered_data = da.rechunk(ordered_data,chunks=(-1,-1,tile_size,tile_size))

    #     exp_metadata = {'exp_name': str(output_path.parts[1]),
    #                     'tile_position_x': int(self.tile_coordinates['x']),
    #                     'tile_position_y': int(self.tile_coordinates['y']),
    #                     'tile_position_z': int(self.tile_coordinates['z']),
    #                     'tile_size_xy_pixel': int(tile_size),
    #                     'y_crop_begin_pixel': 0,
    #                     'y_crop_end_pixel':  int(y_crop_index_max),
    #                     'codebook_name': str(codebook_path.name),
    #                     'na': float(1.35),
    #                     'pixel_size_xy_um': float(0.115),
    #                     'pixel_size_z_um': float(0.115*z_downsample),
    #                     'slab_height_above_coverslip_um' : float(30.0*self.tile_coordinates['z']),
    #                     'deskewed': True,
    #                     'registered': True,
    #                     'deconvolved': False,
    #                     'DoG_filter': False}

    #     # write codebook and metadata
    #     with codebook_path_json.open('w') as f:
    #         json.dump(self.codebook,f)

    #     with metadata_path_json.open('w') as f:
    #         json.dump(exp_metadata,f)

    #     tile_loader._codebook.to_csv(codebook_path_txt,sep='\t',header=False,index=False)

    #     df = pd.DataFrame(exp_metadata, index=[0]).T
    #     df.to_csv(metadata_path_txt, sep='\t', header=False)

    #     # call map_blocks (or map_overlap) with requested tile_size, z downsampling, and tile overlap
    #     tile_writer_da = partial(write_tiled_tiffs,
    #                              z_downsample=z_downsample,
    #                              stage_position=self._stage_positions[0],
    #                              output_path=data_path)
    #     dask_writer = da.map_blocks(tile_writer_da,
    #                                 ordered_data,
    #                                 meta=np.array((), dtype=np.uint8))
        
    #     # use try to catch map_block dimension issue
    #     print('Generating tiled tiffs')
    #     print('----------------------')
    #     try:
    #         with TqdmCallback(desc="Subtiles"):
    #             dask_writer.compute(num_workers=8)
    #     except:
    #         pass

    #     del ordered_data, dask_writer, self._data_registered
    #     gc.collect()

    def _create_figure(self):
        """
        Generate napari figure for debugging
        """
        import napari
        
        viewer = napari.Viewer()
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
        
        for idx in range(self._num_channels):
            middle_slice = self._data_registered[idx].shape[0]//2
            viewer.add_image(data=self._data_registered[idx],
                             name=self._round_ids[idx],
                             scale=self._voxel_size,
                             blending='additive',
                             colormap=colormaps[idx].to_napari(),
                             contrast_limits=[100,np.percentile(self._data_registered[idx][middle_slice,:].ravel(),99.98)])

        napari.run()