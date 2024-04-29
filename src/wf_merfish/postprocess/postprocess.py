#!/usr/bin/env python
'''
qi2lab WF MERFISH / smFISH processing
Microscope post-processing v0.1.2

qi2lab LED scope MERFISH post-processing
- Rewrite raw data in compressed Zarr in qi2lab MERFISH format
- Correct hot pixels or corrects illumination shading
- Calculate registration across rounds for each tile for decoding
- Calculate registration across tiles for global coordinates
- Mask all potential RNA spots using U-FISH package
- Segment polyDT using Cellpose for guess at cell boundaries
- Decode MERFISH spots on GPU 
    - TO DO: help user determine optimal normalization and decoding settings
    - TO DO: Process smFISH bits if requested
- Assign RNA to Cellpose cell outlines in global coordinate system
- Refine cell boundaries using Baysor in global coordinate system
- Create single-cell expression matrix (mtx format)
    - TO DO: figure out how to write a SpatialData object in global coordinate system
- TO DO: write all run parameters to .json with date/timestamp for tracking
- TO DO: docstrings and comments

Change log:
Shepherd 05/24 - added cellpose and baysor for segmentation and assignment.
                 rework function for explicit flags and parameter dictionaries.
Shepherd 04/24 - fully automated processing and decoding.
Shepherd 01/24 - updates for qi2lab MERFISH file format v1.0.
Shepherd 09/23 - new LED widefield scope post-processing.
'''

# imports
import numpy as np
from pathlib import Path
from pycromanager import Dataset
import gc
from itertools import compress
from wf_merfish.utils._dataio import return_data_dask, read_metadatafile
import zarr
from numcodecs import blosc
import time
from psfmodels import make_psf
from tifffile import imread
import pandas as pd
from typing import Generator, Optional
from tifffile import TiffWriter
from tqdm import tqdm

# parse experimental directory, load data, and process
def postprocess(dataset_path: Path, 
                codebook_path: Path, 
                bit_order_path: Path,
                write_raw_camera_data: bool = False,
                run_hotpixel_correction: bool = True,
                run_shading_correction: bool = False,
                run_tile_registration: bool = True,
                write_polyDT_tiff: bool = False,
                run_global_registration: bool =  True,
                global_registration_parameters: dict = {'parallel_fusion': False},
                write_fused_zarr: bool = False,
                run_cellpose: bool = True,
                cellpose_parameters: dict = {'diam_mean_pixels': 30,
                                              'flow_threshold': 0.0,
                                              'normalization': [10,90]},
                run_tile_decoding: bool =  True,
                tile_decoding_parameters: dict = {'normalization': [.1,80],
                                                   'calculate_normalization': True,
                                                   'exp_type': '3D',
                                                   'merfish_bits': 16,
                                                   'lowpass_sigma': (3,1,1),
                                                   'distance_threshold': 0.8,
                                                   'magnitude_threshold': 0.3,
                                                   'minimum_pixels': 27,
                                                   'fdr_target': .05},
                run_baysor: bool = True,
                baysor_parameters: dict = {'cell_size_microns': 10,
                                            'min_molecules_per_cell': 20,
                                            'cellpose_prior_confidence': 0.5},
                baysor_ignore_genes: bool = False,
                baysor_genes_to_exclude: Optional[str] = None,
                baysor_filtering_parameters: dict = {'cell_area_microns' : 7.5,
                                                    'confidence' : 0.7,
                                                    'lifespan' : 100},
                run_mtx_creation: bool = True,
                mtx_creation_parameters: dict = {'confidence_cutoff' : 0.7},
                noise_map_path: Optional[Path] = None,
                darkfield_image_path: Optional[Path] = None,
                shading_images_path: Optional[Path] = None) -> Generator[dict[str, int], None, None]:
    
    
    compressor = blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.Blosc.BITSHUFFLE)
                    
    # read metadata for this experiment
    df_metadata = read_metadatafile(dataset_path / Path('scan_metadata.csv'))
    root_name = df_metadata['root_name']
    pixel_size = 2.4 / (60 * (165/180)) #TO DO: fix to load from file.
    axial_step = .310 #TO DO: fix to load from file.
    tile_overlap = 0.2 #TO DO: fix to load from file.
    binning = 2
    pixel_size = np.round(pixel_size*binning,3)
    gain = 27
    num_r = df_metadata['num_r']
    num_tiles = df_metadata['num_xyz']
    chan_dpc1_active = df_metadata['dpc1_active']
    chan_dpc2_active = df_metadata['dpc2_active']
    chan_dpc3_active = df_metadata['dpc3_active']
    chan_dpc4_active = df_metadata['dpc4_active']
    chan_blue_active = df_metadata['blue_active']
    chan_yellow_active = df_metadata['yellow_active']
    chan_red_active = df_metadata['red_active']
    exposure_dpc1_ms = df_metadata['dpc1_exposure']
    exposure_dpc2_ms = df_metadata['dpc2_exposure']
    exposure_dpc3_ms = df_metadata['dpc3_exposure']
    exposure_dpc4_ms = df_metadata['dpc4_exposure']
    exposure_blue_ms = df_metadata['blue_exposure']
    exposure_yellow_ms = df_metadata['yellow_exposure']
    exposure_red_ms = df_metadata['red_exposure']
    active_channels = [chan_dpc1_active,
                       chan_dpc2_active,
                       chan_dpc3_active,
                       chan_dpc4_active,
                       chan_blue_active,
                       chan_yellow_active,
                       chan_red_active]
    exposures_ms = [exposure_dpc1_ms,
                    exposure_dpc2_ms,
                    exposure_dpc3_ms,
                    exposure_dpc4_ms,
                    exposure_blue_ms,
                    exposure_yellow_ms,
                    exposure_red_ms]
    channel_idxs = [0,1,2,3,4,5,6]
    channels_in_data = list(compress(channel_idxs, active_channels))
    # n_active_channels = len(channels_in_data)
    

    # create output directory
    output_dir_path_base = dataset_path
    output_dir_path = output_dir_path_base / 'processed_v2'
    
    qi2lab_exists = True
    # check if qi2lab zarr structure exists
    if output_dir_path.exists():
        polyDT_output_dir_path = output_dir_path / Path('polyDT')
        if not(polyDT_output_dir_path.exists()):
            qi2lab_exists = False
    
        readout_output_dir_path = output_dir_path / Path('readouts')
        if not(readout_output_dir_path.exists()):
            qi2lab_exists = False
        calibrations_output_dir_path = output_dir_path / Path('calibrations.zarr')
        if not(calibrations_output_dir_path.exists()):
            qi2lab_exists = False
            
    else:
        qi2lab_exists = False
        
    progress_updates = {
        "Round": 0,
        "Tile": 0,
        "Channel": 0,
        "Register/Process": 0,
        "Decode": 0,
    }
        
    
    if not(qi2lab_exists):
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # create directory for data type      
        polyDT_output_dir_path = output_dir_path / Path('polyDT')
        polyDT_output_dir_path.mkdir(parents=True, exist_ok=True)
        
        readout_output_dir_path = output_dir_path / Path('readouts')
        readout_output_dir_path.mkdir(parents=True, exist_ok=True)

        calibrations_output_dir_path = output_dir_path / Path('calibrations.zarr')
        calibrations_output_dir_path.mkdir(parents=True, exist_ok=True)
        calibrations_zarr = zarr.open(str(calibrations_output_dir_path), mode="a")
        
        blosc.set_nthreads(20)
                        
        # import image processing functions
        if run_hotpixel_correction and not(run_shading_correction):
            from wf_merfish.utils._imageprocessing import replace_hot_pixels
        elif run_shading_correction:
            from wf_merfish.utils._imageprocessing import correct_shading

        # initialize tile counter and channel information
        ex_wavelengths=[.520,.520,.520,.520,.488,.561,.635]
        em_wavelengths=[.520,.520,.520,.520,.520,.580,.670]

        tile_idx=0

        # TO DO : Load from WF setup data
        channel_ids = ['T-P1',
                    'T-P2',
                    'T-P3',
                    'T-P4',
                    'F-Blue',
                    'F-Yellow',
                    'F-Red']

        # load codebook and experimental order from disk
        codebook_df = pd.read_csv(codebook_path)
        bit_order_df = pd.read_csv(bit_order_path)
        bit_order = bit_order_df.to_numpy()
        
        # deal with calibrations first
        # save noise map
        if noise_map_path is not None and (run_hotpixel_correction or run_shading_correction):
            noise_map = imread(noise_map_path)
            noise_map_zarr = calibrations_zarr.zeros('noise_map',
                                                shape=noise_map.shape,
                                                chunks=(noise_map.shape[0],noise_map.shape[1]),
                                                compressor=compressor,
                                                dtype=np.uint16)
            noise_map_zarr[:] = noise_map
        
        # save darkfield image
        if darkfield_image_path is not None and (run_hotpixel_correction or run_shading_correction):
            darkfield_image = imread(darkfield_image_path)
            darkfield_image_zarr = calibrations_zarr.zeros('darkfield_image',
                                                            shape=darkfield_image.shape,
                                                            chunks=(darkfield_image.shape[0],darkfield_image.shape[1]),
                                                            compressor=compressor,
                                                            dtype=np.uint16)
            darkfield_image_zarr[:] = darkfield_image

        # save shading image
        if shading_images_path is not None and (run_hotpixel_correction or run_shading_correction):
            shading_images = imread(shading_images_path)
            shading_images_zarr = calibrations_zarr.zeros('shading_images',
                                                            shape=shading_images.shape,
                                                            chunks=(1,shading_images.shape[0],shading_images.shape[1]),
                                                            compressor=compressor,
                                                            dtype=np.uint16)
            shading_images_zarr[:] = shading_images

        # save codebook
        calibrations_zarr.attrs['codebook'] = codebook_df.values.tolist()

        # save experimental oder
        calibrations_zarr.attrs['experiment_order'] = bit_order_df.values.tolist()

        # helpful metadata needed by registration and decoding classes so they don't have to traverse nested zarr groups   
        calibrations_zarr.attrs["num_rounds"] = int(num_r)
        calibrations_zarr.attrs["num_tiles"] = int(num_tiles)
        calibrations_zarr.attrs["channels_in_data"] = channels_in_data
        calibrations_zarr.attrs["tile_overlap"] = float(tile_overlap)
        calibrations_zarr.attrs["binning"] = int(binning)
        calibrations_zarr.attrs["gain"] = float(gain)
        calibrations_zarr.attrs["na"] = float(1.35)
        calibrations_zarr.attrs["ri"] = float(1.51)
        
        # generate and save PSFs
        channel_psfs = []
        for channel_id in channels_in_data:
            channel_psfs.append(make_psf(z=33,
                                nx=33,
                                dxy=pixel_size,
                                dz=axial_step,
                                NA=1.35,
                                wvl=em_wavelengths[channel_id],
                                ns=1.33,
                                ni=1.51,
                                ni0=1.51,
                                model='vectorial'))
        channel_psfs = np.array(channel_psfs)

        psf_data = calibrations_zarr.zeros('psf_data',
                                            shape=(channel_psfs.shape[0],channel_psfs.shape[1],channel_psfs.shape[2],channel_psfs.shape[3]),
                                            chunks=(1,1,channel_psfs.shape[2],channel_psfs.shape[3]),
                                            compressor=compressor,
                                            dtype=np.uint16)
        psf_data[:] = channel_psfs
    
        # loop over all tiles.
        for (tile_idx) in range(num_tiles):

            tile_name = 'tile'+str(tile_idx).zfill(4)

            for r_idx in range(num_r):

                round_name = 'round'+str(r_idx).zfill(3)

                # open stage positions file
                stage_position_filename = Path(root_name+'_r'+str(r_idx+1).zfill(4)+'_tile'+str(tile_idx).zfill(4)+'_stage_positions.csv')
                stage_position_path = dataset_path / stage_position_filename

                read_metadata = False
                skip_tile = False
                # check if this is the first tile. If so, wait longer for fluidics
                while(not(read_metadata)):
                    try:
                        df_stage_positions = read_metadatafile(stage_position_path)
                        read_metadata = True
                    except Exception:
                        read_metadata = False
                        time.sleep(60*1)                    

                if not(skip_tile):

                    # grab recorded stage positions
                    stage_x = np.round(float(df_stage_positions['stage_x']),2)
                    stage_y = np.round(float(df_stage_positions['stage_y']),2)
                    stage_z = np.round(float(df_stage_positions['stage_z']),2)

                    # grab channels in this tile
                    chan_dpc1_active_tile = df_stage_positions['dpc1_active']
                    chan_dpc2_active_tile = df_stage_positions['dpc2_active']
                    chan_dpc3_active_tile = df_stage_positions['dpc3_active']
                    chan_dpc4_active_tile = df_stage_positions['dpc4_active']
                    chan_blue_active_tile = df_stage_positions['blue_active']
                    chan_yellow_active_tile = df_stage_positions['yellow_active']
                    chan_red_active_tile = df_stage_positions['red_active']

                    active_channels_tile = [chan_dpc1_active_tile,
                                            chan_dpc2_active_tile,
                                            chan_dpc3_active_tile,
                                            chan_dpc4_active_tile,
                                            chan_blue_active_tile,
                                            chan_yellow_active_tile,
                                            chan_red_active_tile]
                    channels_idxs_in_data_tile = list(compress(channel_idxs, active_channels_tile))
                    channels_ids_in_data_tile = list(compress(channel_ids, active_channels_tile))

                    # construct directory name
                    current_tile_dir_path = Path(root_name+'_r'+str(r_idx+1).zfill(4)+'_tile'+str(tile_idx).zfill(4)+'_1')
                    tile_dir_path_to_load = dataset_path / current_tile_dir_path

                    file_load = False
                    # open dataset directory using NDTIFF
                    try:
                        dataset = Dataset(str(tile_dir_path_to_load))
                        file_load = True
                    except Exception:
                        file_load = False
                        print('Dataset loading error. Skipping this stage position.')
    

                    # polyDT zarr store
                    polyDT_tile_dir_path = polyDT_output_dir_path / Path(tile_name)
                    polyDT_tile_dir_path.mkdir(parents=True, exist_ok=True)
                    polydT_round_dir_path = polyDT_tile_dir_path / Path(round_name + '.zarr')
                    polydT_round_zarr = zarr.open(str(polydT_round_dir_path), mode="a")
                    

                    # # yellow readout zarr store
                    yellow_readout_round_idx = bit_order[r_idx,1]
                    yellow_bit_name = "bit"+str(yellow_readout_round_idx).zfill(2)
                    yellow_tile_dir_path = readout_output_dir_path / Path(tile_name)
                    yellow_tile_dir_path.mkdir(parents=True, exist_ok=True)
                    yellow_bit_dir_path = yellow_tile_dir_path / Path(yellow_bit_name + '.zarr')
                    yellow_bit_zarr = zarr.open(str(yellow_bit_dir_path), mode="a")
                    

                    # red readout zarr store
                    red_readout_round_idx = bit_order[r_idx,2]
                    red_bit_name = "bit"+str(red_readout_round_idx).zfill(2)
                    red_tile_dir_path = readout_output_dir_path / Path(tile_name)
                    red_tile_dir_path.mkdir(parents=True, exist_ok=True)
                    red_bit_dir_path = red_tile_dir_path / Path(red_bit_name + '.zarr')
                    red_bit_zarr = zarr.open(str(red_bit_dir_path), mode="a")

                    if file_load:
                        # loop over all channels in this round/tile
                        for channel_id, ch_idx in zip(channels_ids_in_data_tile,channels_idxs_in_data_tile):
                            # load raw data into memory via Dask
                            raw_data_success = False
                            try:
                                raw_data = return_data_dask(dataset,channel_id)
                                raw_data_success = True
                            except Exception:
                                raw_data_success = False
                                print('Internal NDTIFF error. Skipping this tile/channel combination.')
                                                    
                            if raw_data_success:
                            
                                # load psf into memory
                                ex_wvl = ex_wavelengths[ch_idx]
                                em_wvl = em_wavelengths[ch_idx]
                                exposure_ms = exposures_ms[ch_idx]
                                
                                if channel_id == 'F-Blue':
                                    current_channel = polydT_round_zarr
                                    current_channel.attrs['bits'] = bit_order[r_idx,:].tolist()
                                    current_channel.attrs["tile_overlap"] = tile_overlap
                                    current_channel.attrs["psf_idx"] = int(0)
                                    tiff_file_path = polyDT_tile_dir_path / Path(tile_name + "_polyDT.ome.tiff")
                                elif channel_id == 'F-Yellow':
                                    current_channel = yellow_bit_zarr
                                    current_channel.attrs['round'] = int(r_idx)
                                    current_channel.attrs["psf_idx"] = int(1)
                                elif channel_id == 'F-Red':
                                    current_channel = red_bit_zarr
                                    current_channel.attrs['round'] = int(r_idx)
                                    current_channel.attrs["psf_idx"] = int(2)
                                    
                                if write_raw_camera_data:
                                    camera_data = raw_data.copy()
                                
                                if run_shading_correction:
                                    raw_data = correct_shading(noise_map,darkfield_image,shading_images[ch_idx],raw_data)
                                elif not(run_shading_correction) and (run_hotpixel_correction):
                                    raw_data = replace_hot_pixels(noise_map,raw_data)
                                    

                                if write_raw_camera_data:
                                    current_camera_data = current_channel.zeros('camera_data',
                                                                                shape=(camera_data.shape[0],camera_data.shape[1],camera_data.shape[2]),
                                                                                chunks=(1,camera_data.shape[1],camera_data.shape[2]),
                                                                                compressor=compressor,
                                                                                dtype=np.uint16)
                                    current_camera_data[:] = camera_data

                                current_raw_data = current_channel.zeros('raw_data',
                                                                        shape=(raw_data.shape[0],raw_data.shape[1],raw_data.shape[2]),
                                                                        chunks=(1,raw_data.shape[1],raw_data.shape[2]),
                                                                        compressor=compressor,
                                                                        dtype=np.uint16)
                                
                                
                                current_channel.attrs['stage_zyx_um'] = np.array([stage_z,stage_y,stage_x]).tolist()
                                current_channel.attrs['voxel_zyx_um'] = np.array([float(axial_step),float(pixel_size),float(pixel_size)]).tolist()
                                current_channel.attrs['excitation_um'] = float(ex_wvl)
                                current_channel.attrs['gain'] = float(gain)
                                current_channel.attrs['emission_um'] = float(em_wvl)
                                current_channel.attrs['exposure_ms'] = float(exposure_ms)
                                current_channel.attrs['hotpixel'] = bool(run_hotpixel_correction)
                                current_channel.attrs['shading'] = bool(run_shading_correction)

                                current_raw_data[:] = raw_data
                                
                                if channel_id == 'F-Blue' and write_polyDT_tiff and r_idx == 0:
                                    with TiffWriter(tiff_file_path, bigtiff=False) as tif:
                                        metadata = {'Name' : str(tile_name),
                                                    'axes': 'ZYX',
                                                    'PhysicalSizeX': pixel_size,
                                                    'PhysicalSizeXUnit': 'µm',
                                                    'PhysicalSizeY': pixel_size,
                                                    'PhysicalSizeYUnit': 'µm',
                                                    'PhysicalSizeZ': axial_step,
                                                    'PhysicalSizeZUnit': 'µm',
                                                    'Channel': {'Name': 'polyDT'},
                                                    }
                                        tif.write(raw_data,
                                                    resolution=(1e4 / pixel_size, 1e4 / pixel_size),
                                                    metadata=metadata,
                                                    photometric='minisblack',
                                                    resolutionunit='CENTIMETER')
                                        
                                progress_updates['Channel'] = ((ch_idx-channels_idxs_in_data_tile[0]+1) / len(channels_idxs_in_data_tile)) * 100
                                yield progress_updates

                        dataset.close()
                        del dataset
                        gc.collect()

                progress_updates['Channel'] = 0
                progress_updates['Round'] = ((r_idx+1) / num_r) * 100
                yield progress_updates

            progress_updates['Channel'] = 0
            progress_updates['Round'] = 0
            progress_updates['Tile'] = ((tile_idx+1) / num_tiles) * 100
            yield progress_updates
            
    progress_updates['Channel'] = 100
    progress_updates['Round'] = 100
    progress_updates['Tile'] = 100
    yield progress_updates
                    
    if run_tile_registration:
        run_optical_flow = True
        from wf_merfish.postprocess.DataRegistration import DataRegistration

        for tile_idx in range(num_tiles):
            data_register_factory = DataRegistration(dataset_path=output_dir_path,
                                                    overwrite_registered=True,
                                                    perform_optical_flow=run_optical_flow,
                                                    tile_idx=tile_idx)
            data_register_factory.generate_registrations()
            data_register_factory.apply_registration_to_bits()
            
            progress_updates['Register/Process'] = ((tile_idx+1) / num_tiles) * 100
            yield progress_updates

            del data_register_factory
            gc.collect()
            
    progress_updates['Register/Process'] = 100
    yield progress_updates
    
    if run_global_registration:
        fuse_readouts = False        
        
        from multiview_stitcher import spatial_image_utils as si_utils
        from multiview_stitcher import msi_utils, fusion, registration
        import dask.diagnostics
        import dask.array as da
        
        fused_dir_path = output_dir_path / Path('fused')
        fused_dir_path.mkdir(parents=True, exist_ok=True)
        
        polyDT_dir_path = output_dir_path / Path('polyDT')
        readout_dir_path = output_dir_path / Path('readouts')

        tile_ids = sorted([entry.name for entry in polyDT_dir_path.iterdir() if entry.is_dir()],
                            key=lambda x: int(x.split('tile')[1].split('.zarr')[0]))

        msims = []
        for tile_idx, tile_id in enumerate(tqdm(tile_ids,desc='tile')):
        
            polyDT_current_path = polyDT_dir_path / Path(tile_id) / Path("round000.zarr")
            polyDT_current_tile = zarr.open(polyDT_current_path,mode='r')

            voxel_zyx_um = np.asarray(polyDT_current_tile.attrs['voxel_zyx_um'],
                                                dtype=np.float32)

            scale = {'z': voxel_zyx_um[0],
                    'y': voxel_zyx_um[1],
                    'x': voxel_zyx_um[2]}
                
            tile_position_zyx_um = np.asarray(polyDT_current_tile.attrs['stage_zyx_um'],
                                                dtype=np.float32)
            
            tile_grid_positions = {
                'z': tile_position_zyx_um[0],
                'y': tile_position_zyx_um[1],
                'x': tile_position_zyx_um[2],
            }

            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                im_data = []
                im_data.append(da.from_array(polyDT_current_tile['registered_decon_data']))
                
                readout_current_tile_path = readout_dir_path / Path(tile_id)
                bit_ids = sorted([entry.name for entry in readout_current_tile_path.iterdir() if entry.is_dir()],
                                        key=lambda x: int(x.split('bit')[1].split('.zarr')[0]))
                
                for bit_id in bit_ids:
                    bit_current_tile = readout_current_tile_path / Path(bit_id)
                    bit_current_tile = zarr.open(bit_current_tile,mode='r')
                    im_data.append(da.from_array((bit_current_tile['registered_decon_data'])))

                im_data = da.stack(im_data)
                
            if ~fuse_readouts:
                im_data = im_data[0,:]
                sim = si_utils.get_sim_from_array(da.expand_dims(im_data,axis=0),
                                                dims=('c','z', 'y', 'x'),
                                                scale=scale,
                                                translation=tile_grid_positions,
                                                transform_key='stage_metadata')
            
            else:
                sim = si_utils.get_sim_from_array(im_data,
                                                dims=('c', 'z', 'y', 'x'),
                                                scale=scale,
                                                translation=tile_grid_positions,
                                                transform_key='stage_metadata')
                
            msim = msi_utils.get_msim_from_sim(sim,scale_factors=[])
            msims.append(msim)
            del im_data
            gc.collect()
                
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            with dask.diagnostics.ProgressBar():
                params = registration.register(
                    msims,
                    reg_channel_index=0,
                    transform_key='stage_metadata',
                    new_transform_key='translation_registered',
                    registration_binning={'z': 4, 'y': 4, 'x': 4},
                    pre_registration_pruning_method='otsu_threshold_on_overlap',
                    plot_summary=False,
                )
        
        for tile_idx, msim in enumerate(msims):
            polyDT_current_path = polyDT_dir_path / Path(tile_ids[tile_idx]) / Path("round000.zarr")
            polyDT_current_tile = zarr.open(polyDT_current_path,mode='a')
            
            affine = msi_utils.get_transform_from_msim(msim, transform_key='translation_registered').data.squeeze()
            origin = si_utils.get_origin_from_sim(msi_utils.get_sim_from_msim(msim), asarray=True)
            spacing = si_utils.get_spacing_from_sim(msi_utils.get_sim_from_msim(msim), asarray=True)
                    
            polyDT_current_tile.attrs['affine_zyx_um'] = affine.tolist()
            polyDT_current_tile.attrs['origin_zyx_um'] = origin.tolist()
            polyDT_current_tile.attrs['spacing_zyx_um'] = spacing.tolist()
            del polyDT_current_tile, affine, origin, spacing
            
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):

            fused_sim = fusion.fuse(
                [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in msims],
                transform_key='translation_registered',
                output_spacing={'z': voxel_zyx_um[0], 'y': voxel_zyx_um[1]*3.5, 'x': voxel_zyx_um[2]*3.5},
                output_chunksize=512,
                overlap_in_pixels=256,
            )
            
        fused_msim = msi_utils.get_msim_from_sim(fused_sim, scale_factors=[])
        affine = msi_utils.get_transform_from_msim(fused_msim, transform_key='translation_registered').data.squeeze()
        origin = si_utils.get_origin_from_sim(msi_utils.get_sim_from_msim(fused_msim), asarray=True)
        spacing = si_utils.get_spacing_from_sim(msi_utils.get_sim_from_msim(fused_msim), asarray=True)
        
        with dask.diagnostics.ProgressBar():
            if global_registration_parameters['parallel_fusion']:
                fused = fused_sim.compute()
            else:
                fused = fused_sim.compute(scheduler='single-threaded')
        
        if write_fused_zarr or run_cellpose:
            fused_output_path = fused_dir_path / Path("fused.zarr")
            fused_zarr = zarr.open(str(fused_output_path), mode="a")
                
        if write_fused_zarr:
            fused_image = np.squeeze(fused.data)
            
            if fuse_readouts:
                try:
                    fused_data = fused_zarr.zeros('fused_iso_zyx',
                                                shape=(fused_image.shape[0],fused_image.shape[1],fused_image.shape[2],fused_image.shape[3]),
                                                chunks=(1,1,256,256),
                                                compressor=compressor,
                                                dtype=np.uint16)
                except:
                    fused_data = fused_zarr['fused_iso_zyx']
            else:
                try:
                    fused_data = fused_zarr.zeros('fused_iso_zyx',
                                            shape=(fused_image.shape[0],fused_image.shape[1],fused_image.shape[2]),
                                            chunks=(1,256,256),
                                            compressor=compressor,
                                            dtype=np.uint16)
                except:
                    fused_data = fused_zarr['fused_iso_zyx']
            
            fused_data[:] = np.squeeze(fused_image)
                
            fused_data.attrs['affine_zyx_um'] = affine.tolist()
            fused_data.attrs['origin_zyx_um'] = origin.tolist()
            fused_data.attrs['spacing_zyx_um'] = spacing.tolist()
            
            del fused_sim, fused_image
            gc.collect()
        
    if run_cellpose:
        from cellpose import models
        from cellpose.utils import outlines_list
        from shapely.geometry import Polygon
        import geopandas as gpd
        
        def warp_pixels(pixel_space_point: np.ndarray,
                    spacing: np.ndarray,
                    origin: np.ndarray,
                    affine: np.ndarray) -> np.ndarray:

            physical_space_point = pixel_space_point * spacing + origin
            registered_space_point = (np.array(affine) @ np.array(list(physical_space_point) + [1]))[:-1]
            
            return registered_space_point
        
        channels = [[0,0]]
        
        model = models.Cellpose(gpu=True,model_type='cyto3')
        model.diam_mean = diam_mean=cellpose_parameters['diam_mean_pixels']
                                
        if fuse_readouts:
            data_to_segment = np.max(np.squeeze(fused[0,:].data),axis=0)
        else:
            data_to_segment = np.max(np.squeeze(fused.data),axis=0)

        masks, _, _, _ = model.eval(data_to_segment,
                                    channels=channels,
                                    flow_threshold=cellpose_parameters['flow_threshold'],
                                    normalize = {'normalize': True,
                                                 'percentile': cellpose_parameters['normalization']})
        
        segmentation_dir_path = output_dir_path / Path('segmentation')
        segmentation_dir_path.mkdir(parents=True, exist_ok=True)
        cellpose_dir_path = segmentation_dir_path / Path('cellpose')
        cellpose_dir_path.mkdir(parents=True, exist_ok=True)
        
        masks_output_path = cellpose_dir_path / Path("cellpose.zarr")
        masks_zarr = zarr.open(str(masks_output_path), mode="a")
        
        try:
            masks_data = masks_zarr.zeros('masks_iso_zyx',
                                    shape=(masks.shape[0],masks.shape[1]),
                                    chunks=(256,256),
                                    compressor=compressor,
                                    dtype=np.uint16)
        except:
            masks_data = masks_zarr['masks_iso_zyx']

        masks_data[:] = masks
        masks_data.attrs['affine_zyx_um'] = affine.tolist()
        masks_data.attrs['origin_zyx_um'] = origin.tolist()
        masks_data.attrs['spacing_zyx_um'] = spacing.tolist()
        
        outlines = outlines_list(masks)
        n_outlines = len(outlines)
        for outline_idx in range(n_outlines):
            n_pts = len(outlines[outline_idx])
            for pt_idx in range(n_pts):
                pt_fakez= [0,
                           outlines[outline_idx][pt_idx][0],
                           outlines[outline_idx][pt_idx][1]]
                warp_pt_fakez = warp_pixels(pt_fakez,spacing,origin,affine)
                outlines[outline_idx][pt_idx][0]=warp_pt_fakez[1]
                outlines[outline_idx][pt_idx][1]=warp_pt_fakez[2]

        polygons = [Polygon(outline.reshape(-1, 2)) for outline in outlines if len(outline) >= 3]
        masks_gdf = gpd.GeoDataFrame(geometry=polygons)
        masks_json_path = cellpose_dir_path / Path('cell_outlines.geojson')
        masks_gdf.to_file(masks_json_path, driver='GeoJSON')
        
        del masks, outlines, masks_gdf
        del affine, origin, spacing
            
    if run_tile_decoding:
        
        from wf_merfish.postprocess.PixelDecoder import PixelDecoder
        import cupy as cp
        
        global_normalization_limits = tile_decoding_parameters['normalization']
        calculate_normalization = tile_decoding_parameters['calculate_normalization']
        exp_type = tile_decoding_parameters['exp_type']
        merfish_bits = tile_decoding_parameters['merfish_bits']
        lowpass_sigma = tile_decoding_parameters['lowpass_sigma']
        distance_threshold = tile_decoding_parameters['distance_threshold']
        magnitude_threshold = tile_decoding_parameters['magnitude_threshold']
        minimum_pixels = tile_decoding_parameters['minimum_pixels']
        fdr_target = tile_decoding_parameters['fdr_target']
            
        decode_factory = PixelDecoder(dataset_path=output_dir_path,
                                      global_normalization_limits=global_normalization_limits,
                                      overwrite_normalization=calculate_normalization,
                                      exp_type=exp_type,
                                      merfish_bits=merfish_bits)

        tile_ids = decode_factory._tile_ids
        
        del decode_factory
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        for tile_idx, tile_id in enumerate(tqdm(tile_ids,desc='tile',leave=True)):
    
            decode_factory = PixelDecoder(dataset_path=output_dir_path,
                                        global_normalization_limits=global_normalization_limits,
                                        overwrite_normalization=False,
                                        tile_idx=tile_idx,
                                        exp_type=exp_type,
                                        merfish_bits=merfish_bits)
            decode_factory.run_decoding(lowpass_sigma=lowpass_sigma,
                                        distance_threshold=distance_threshold,
                                        magnitude_threshold=magnitude_threshold,
                                        minimum_pixels=minimum_pixels,
                                        skip_extraction=False)
            decode_factory.save_barcodes()
            decode_factory.cleanup()
            
            del decode_factory
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            
            progress_updates['Decode'] = ((tile_idx+1) / num_tiles) * 100
            yield progress_updates
                        
        decode_factory = PixelDecoder(dataset_path=output_dir_path,
                                    global_normalization_limits=global_normalization_limits,
                                    overwrite_normalization=False,
                                    exp_type=exp_type,
                                    merfish_bits=merfish_bits)
    
        decode_factory.load_all_barcodes()
        decode_factory.filter_all_barcodes(fdr_target=fdr_target)
        decode_factory.assign_cells()
        decode_factory.save_barcodes(format='parquet')
        decode_factory.save_all_barcodes_for_baysor()

        del decode_factory
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        progress_updates['Decode'] = 100
        yield progress_updates
        
    if run_baysor:
        import subprocess
        
        baysor_path = Path(baysor_parameters['baysor_path'])
        baysor_num_threads = baysor_parameters['num_threads']
        baysor_cell_size = baysor_parameters['cell_size_microns']
        baysor_min_molecules = baysor_parameters['min_molecules_per_cell']
        baysor_cellpose_prior = baysor_parameters['cellpose_prior_confidence']
                
        # construct baysor command
        julia_threading = "JULIA_NUM_THREADS="+str(baysor_num_threads)+ " "
        baysor_options = r" run -p -x global_x -y global_y -g gene_id -s "+str(baysor_cell_size) +\
            r" --config.segmentation.iters 2000 --config.data.force_2d=true"+\
            r" --count-matrix-format 'tsv' --save-polygons 'GeoJSON' --min-molecules-per-cell "+str(baysor_min_molecules)+\
            r" --prior-segmentation-confidence "+str(baysor_cellpose_prior)+r" "
        if baysor_ignore_genes:
            baysor_genes_to_ignore = r"--config.data.exclude_genes='" + baysor_genes_to_exclude + "' "
            baysor_options = baysor_options + baysor_genes_to_ignore
        baysor_genes_path = output_dir_path / Path("decoded") / Path('baysor_formatted_genes.csv')
        baysor_output_path = output_dir_path / Path("segmentation") / Path('baysor')
        baysor_output_path.mkdir(exist_ok=True, parents=True)
        
        command = julia_threading + str(baysor_path) + baysor_options + "-o " +\
            str(baysor_output_path) + " " + str(baysor_genes_path) + " :cell_id"
                    
        try:
            result = subprocess.run(command, shell=True, check=True)
            print("Command finished with return code:", result.returncode)
            baysor_success = True
        except subprocess.CalledProcessError as e:
            print("Command failed with:", e)
            baysor_success = False
        
        if baysor_success:
            import geopandas as gpd
            from shapely.geometry import Polygon
            import json
            
            if baysor_ignore_genes:
                baysor_output_genes_path = baysor_genes_path
            else:
                baysor_output_genes_path = baysor_output_path / Path("segmentation.csv")
            
            baysor_outlines_path = baysor_output_path / Path("segmentation_polygons.json")
            baysor_stats_path = baysor_output_path / Path("segmentation_cell_stats.csv")
            baysor_genes_df = pd.read_csv(baysor_output_genes_path)
            baysor_cell_stats_df = pd.read_csv(baysor_stats_path)

            with open(baysor_outlines_path, 'r') as f:
                geojson_data = json.load(f)

            polygons = []
            cells = []
            for geom in geojson_data['geometries']:
                coords = geom['coordinates'][0]
                if coords[0] != coords[-1]:
                    coords.append(coords[0].copy()) 
                if len(coords) >= 4:
                    polygon = Polygon(coords)
                    polygons.append(polygon)
                    cells.append(geom['cell']) 

            corrected_geojson = gpd.GeoDataFrame({
                'geometry': polygons,
                'cell': cells
            })

            def extract_number(cell_value):
                if pd.isna(cell_value):
                    return -1
                elif isinstance(cell_value, str):
                    try:
                        return int(cell_value.split('-')[-1])
                    except (ValueError, IndexError):
                        return -1
                else:
                    return -1

            baysor_cell_stats_df['cell_number'] = baysor_cell_stats_df['cell'].apply(extract_number)
            filtered_cell_df = baysor_cell_stats_df[(baysor_cell_stats_df['area'] > baysor_filtering_parameters['cell_area_microns']) &\
                                                    (baysor_cell_stats_df['avg_confidence'] > baysor_filtering_parameters['confidence']) &\
                                                    (baysor_cell_stats_df['lifespan'] > baysor_filtering_parameters['lifespan'])]
            filtered_outlines_gdf = corrected_geojson[corrected_geojson['cell'].isin(filtered_cell_df['cell_number'])]

            if baysor_ignore_genes:
                points_gdf = gpd.GeoDataFrame(
                    baysor_genes_df,
                    geometry=gpd.points_from_xy(baysor_genes_df.global_x, baysor_genes_df.global_y)
                )
            else:
                points_gdf = gpd.GeoDataFrame(
                    baysor_genes_df,
                    geometry=gpd.points_from_xy(baysor_genes_df.x, baysor_genes_df.y)
                )
                
            if points_gdf.crs is None and filtered_outlines_gdf.crs is not None:
                points_gdf.set_crs(filtered_outlines_gdf.crs, inplace=True)
            elif points_gdf.crs != filtered_outlines_gdf.crs:
                points_gdf = points_gdf.to_crs(filtered_outlines_gdf.crs)

            joined_data = gpd.sjoin(points_gdf, filtered_outlines_gdf, how="left", predicate='within')
            joined_data.reset_index(inplace=True, drop=True)
            joined_data['index_right'] = joined_data['index_right'].fillna(-1).astype(int) + 1
            baysor_genes_df['baysor_cell_id'] = joined_data['index_right']
            
            baysor_filtered_genes_path = output_dir_path / Path("decoded") / Path("baysor_decoded.parquet")
            baysor_genes_df.to_parquet(baysor_filtered_genes_path)
            
    if run_mtx_creation:
        from src.wf_merfish.utils._dataio import create_mtx
        
        create_mtx(baysor_filtered_genes_path,
                   output_dir_path / Path('mtx_output'),
                   mtx_creation_parameters['confidence_cutoff'])
                   
    return True

if __name__ == '__main__':
    
    
    # example run setup for human olfactory bulb 
    dataset_path = Path('/mnt/opm3/20240317_OB_MERFISH_7/')
    codebook_path = dataset_path / ('codebook.csv')
    bit_order_path = dataset_path / ('bit_order.csv')
    noise_map_path = Path('/home/qi2lab/Documents/github/wf-merfish/hot_pixel_image.tif')
    baysor_genes_to_exclude = "OR10C1, OR10G2, OR10H1, OR10H5, OR10Q1, OR10S1, OR10W1, OR11A1,\
                            OR12D1, OR13A1, OR13J1, OR1F1, OR1I1, OR1M1, OR2A1, OR2A14,\
                            OR2A20P, OR2A4, OR2A42, OR2A9P, OR2AT4, OR2B11, OR2C1, OR2C3,\
                            OR2F1, OR2H1, OR2H2, OR2L13, OR2S2, OR2T2, OR2T27, OR2T35,\
                            OR2T5, OR2T7, OR2Z1, OR3A2, OR3A3, OR3A4P, OR51D1, OR51E1,\
                            OR51E2, OR51G1, OR52I1, OR52I2, OR52K2, OR52L1, OR52W1,\
                            OR56B1, OR56B4, OR5AU1, OR5C1, OR6A2, OR6J1, OR6W1P, OR7A5,\
                            OR8A1, OR9Q1, Blank*"

    func = postprocess(dataset_path = dataset_path, 
                       codebook_path = codebook_path,
                       bit_order_path = bit_order_path,
                       write_raw_camera_data = False,
                       run_hotpixel_correction = True,
                       run_shading_correction = False,
                       run_tile_registration = False,
                       write_polyDT_tiff = False,
                       run_global_registration =  True,
                       global_registration_parameters = {'parallel_fusion': False}, # for qi2lab network drive, must be false due to Dask issue
                       write_fused_zarr = True,
                       run_cellpose = True,
                       cellpose_parameters = {'diam_mean_pixels': 30,
                                              'flow_threshold': 0.0,
                                              'normalization': [10,90]},
                       run_tile_decoding =  True,
                       tile_decoding_parameters = {'normalization': [.1,80],
                                                   'calculate_normalization': True,
                                                   'exp_type': '3D',
                                                   'merfish_bits': 16,
                                                   'lowpass_sigma': (3,1,1),
                                                   'distance_threshold': 0.8,
                                                   'magnitude_threshold': 0.3,
                                                   'minimum_pixels': 27,
                                                   'fdr_target': .05},
                       # smfish_parameters = {'bits': [17,18],
                       #                      'threshold': -1}
                       run_baysor= True,
                       baysor_parameters = {'baysor_path' : r"~/Documents/github/Baysor/bin/baysor/bin/./baysor",
                                            'num_threads': 24,
                                            'cell_size_microns': 10,
                                            'min_molecules_per_cell': 20,
                                            'cellpose_prior_confidence': 0.5},
                       baysor_ignore_genes = True,
                       baysor_genes_to_exclude = baysor_genes_to_exclude,
                       baysor_filtering_parameters = {'cell_area_microns' : 7.5,
                                                      'confidence_cutoff' : 0.7,
                                                      'lifespan' : 100},
                       run_mtx_creation = True,
                       mtx_creation_parameters = {'confidence_cutoff' : 0.7},
                       noise_map_path = noise_map_path)
    
    for val in func:
        temp_val = val