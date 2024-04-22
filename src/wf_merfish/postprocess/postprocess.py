#!/usr/bin/env python
'''
qi2lab WF MERFISH / FISH processing
Microscope post-processing v1.1

qi2lab LED scope MERFISH post-processing
- Rewrites raw data in compressed Zarr in qi2lab MERFISH format
- Corrects hot pixels or corrects illumination shading
- Calculates registration across rounds for each tile for decoding
- Calculates registration across tiles for global coordinates
- Calls RNA spots using U-FISH package
- Decode spots on GPU 

Change log:
Shepherd 04/24 - fully automated processing and decoding
Shepherd 01/24 - updates for qi2lab MERFISH file format v1.0
Shepherd 09/23 - new LED widefield scope post-processing 
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
from typing import Dict, Generator, Optional
from tifffile import TiffWriter
from tqdm import tqdm

# parse experimental directory, load data, and process
def postprocess(correction_option: str, 
                processing_options: Dict[str,bool], 
                dataset_path: Path, 
                codebook_path: Path, 
                bit_order_path: Path,
                noise_map_path: Optional[Path] = None,
                darkfield_image_path: Optional[Path] = None,
                shading_images_path: Optional[Path] = None) -> Generator[Dict[str, int], None, None]:
    
    compressor = blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.Blosc.BITSHUFFLE)
    write_camera_data = False
    hotpixel_flag = False
    shading_flag = False
    round_registration_flag = False
    decode_tiles = False
    write_polyDT_tiff = False
    if correction_option == "Hotpixel correct" and (noise_map_path is not None):
        hotpixel_flag = True
    elif correction_option == "Flatfield correct" and (noise_map_path is not None) and (darkfield_image_path is not None) and (shading_images_path is not None):
        shading_flag = True
    
    for option_name, is_selected in processing_options.items():
        if is_selected:
            if option_name == "Register and process tiles":
                round_registration_flag = True
            elif option_name == "Decode tiles":
                decode_tiles = True
            elif option_name == "Write fused, downsampled polyDT tiff":
                write_fused_zarr = True
            elif option_name == "Write polyDT tiffs":
                write_polyDT_tiff = True
            
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
        if hotpixel_flag and not(shading_flag):
            from wf_merfish.utils._imageprocessing import replace_hot_pixels
        elif shading_flag:
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
        if noise_map_path is not None and (hotpixel_flag or shading_flag):
            noise_map = imread(noise_map_path)
            noise_map_zarr = calibrations_zarr.zeros('noise_map',
                                                shape=noise_map.shape,
                                                chunks=(noise_map.shape[0],noise_map.shape[1]),
                                                compressor=compressor,
                                                dtype=np.uint16)
            noise_map_zarr[:] = noise_map
        
        # save darkfield image
        if darkfield_image_path is not None and (hotpixel_flag or shading_flag):
            darkfield_image = imread(darkfield_image_path)
            darkfield_image_zarr = calibrations_zarr.zeros('darkfield_image',
                                                            shape=darkfield_image.shape,
                                                            chunks=(darkfield_image.shape[0],darkfield_image.shape[1]),
                                                            compressor=compressor,
                                                            dtype=np.uint16)
            darkfield_image_zarr[:] = darkfield_image

        # save shading image
        if shading_images_path is not None and (hotpixel_flag or shading_flag):
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
                                    
                                if write_camera_data:
                                    camera_data = raw_data.copy()
                                
                                if shading_flag:
                                    raw_data = correct_shading(noise_map,darkfield_image,shading_images[ch_idx],raw_data)
                                elif not(shading_flag) and (hotpixel_flag):
                                    raw_data = replace_hot_pixels(noise_map,raw_data)
                                    

                                if write_camera_data:
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
                                current_channel.attrs['hotpixel'] = bool(hotpixel_flag)
                                current_channel.attrs['shading'] = bool(shading_flag)

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
                    
    if round_registration_flag:
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
    
    fuse_readouts = False
    
    from multiview_stitcher import spatial_image_utils as si_utils
    from multiview_stitcher import msi_utils, vis_utils, fusion, registration
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
    
    for tile_idx, param in enumerate(params):
        polyDT_current_path = polyDT_dir_path / Path(tile_ids[tile_idx]) / Path("round000.zarr")
        polyDT_current_tile = zarr.open(polyDT_current_path,mode='a')
        translation_zyx_um = np.round(np.array(param).squeeze()[:-1, -1],3).tolist()
        
        polyDT_current_tile.attrs['translation_zyx_um'] = translation_zyx_um
        del polyDT_current_tile
        
    if write_fused_zarr:

        fused_output_path = fused_dir_path / Path("fused.zarr")

        with dask.config.set(**{'array.slicing.split_large_chunks': False}):

            fused_sim = fusion.fuse(
                [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in msims],
                transform_key='translation_registered',
                output_spacing={'z': voxel_zyx_um[0], 'y': voxel_zyx_um[1]*3.5, 'x': voxel_zyx_um[2]*3.5},
                output_chunksize=512,
                overlap_in_pixels=256,
            )
            
        fused_zarr = zarr.open(str(fused_output_path), mode="a")
        
        da_fused_data = da.squeeze(fused_sim.data)
        
        if fuse_readouts:
            try:
                fused_data = fused_zarr.zeros('fused_iso_zyx',
                                            shape=(da_fused_data.shape[0],da_fused_data.shape[1],da_fused_data.shape[2],da_fused_data.shape[3]),
                                            chunks=(1,1,256,256),
                                            compressor=compressor,
                                            dtype=np.uint16)
            except:
                fused_data = fused_zarr['fused_iso_zyx']
        else:
            try:
                fused_data = fused_zarr.zeros('fused_iso_zyx',
                                        shape=(da_fused_data.shape[0],da_fused_data.shape[1],da_fused_data.shape[2]),
                                        chunks=(1,256,256),
                                        compressor=compressor,
                                        dtype=np.uint16)
            except:
                fused_data = fused_zarr['fused_iso_zyx']
        
        with dask.diagnostics.ProgressBar():
            fused_data[:] = da_fused_data.compute(scheduler='single-threaded')
            
        fused_data.attrs['voxel_zyx_um'] = np.array([voxel_zyx_um[0], 
                                                    voxel_zyx_um[1]*3.5, 
                                                    voxel_zyx_um[2]*3.5]).tolist()
        
        del fused_sim
        gc.collect()
    
    if decode_tiles:
        
        from wf_merfish.postprocess.PixelDecoder import PixelDecoder
        import cupy as cp
        
        decode_factory = PixelDecoder(dataset_path=output_dir_path,
                                      global_normalization_limits=[.1,80.0],
                                      overwrite_normalization=True,
                                      exp_type='3D',
                                      merfish_bits=16)

        tile_ids = decode_factory._tile_ids
        
        del decode_factory
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        for tile_idx, tile_id in enumerate(tqdm(tile_ids,desc='tile',leave=True)):
    
            decode_factory = PixelDecoder(dataset_path=output_dir_path,
                                        global_normalization_limits=[.1,80.0],
                                        overwrite_normalization=False,
                                        tile_idx=tile_idx,
                                        exp_type='3D',
                                        merfish_bits=16)
            decode_factory.run_decoding(lowpass_sigma=(3,1,1),
                                        distance_threshold=0.8,
                                        magnitude_threshold=.3,
                                        minimum_pixels=27,
                                        skip_extraction=False)
            decode_factory.save_barcodes()
            decode_factory.cleanup()
            
            del decode_factory
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            
            progress_updates['Decode'] = ((tile_idx+1) / num_tiles) * 100
            yield progress_updates
            
        decode_factory = PixelDecoder(dataset_path=output_dir_path,
                                    global_normalization_limits=[.1,80.0],
                                    overwrite_normalization=False,
                                    exp_type='3D',
                                    merfish_bits=16,
                                    verbose=2)
    
        decode_factory.load_all_barcodes()
        decode_factory.filter_all_barcodes(fdr_target=.05)
        decode_factory.save_barcodes(format='parquet')

        del decode_factory
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        progress_updates['Decode'] = 100
        yield progress_updates
    
    return True

if __name__ == '__main__':
    
    data_path = Path('/mnt/opm3/20240317_OB_MERFISH_7/')
    exp_order_path = data_path / ('bit_order.csv')
    codebook_path = data_path / ('codebook.csv')
    noise_map_path = Path('/home/qi2lab/Documents/github/wf-merfish/hot_pixel_image.tif')

    func = postprocess(correction_option='Hotpixel correct',
                        processing_options={'Register and process tiles' : True,
                                            'Write fused, downsampled polyDT tiff': True,
                                            'Decode tiles': True,
                                            'Write polyDT tiffs': False},
                        dataset_path=data_path,
                        noise_map_path=noise_map_path,
                        codebook_path=codebook_path,
                        bit_order_path=exp_order_path)
    
    for val in func:
        print(val)