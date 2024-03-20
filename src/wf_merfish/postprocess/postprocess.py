#!/usr/bin/env python
'''
qi2lab WF MERFISH / FISH processing
Microscope post-processing v1.1

qi2lab LED scope MERFISH post-processing
- Rewrites raw data in compressed Zarr in qi2lab MERFISH format
- Corrects hot pixels or corrects illumination shading
- Calculates registration across rounds for each tile
- Calculates registartion for first round across all tiles

Change log:
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
from superqt.utils import thread_worker
from tifffile import TiffWriter


# parse experimental directory, load data, and process
@thread_worker
def postprocess(correction_option: str, 
                stitching_options: Dict[str,bool], 
                dataset_path: Path, 
                codebook_path: Path, 
                bit_order_path: Path,
                noise_map_path: Optional[Path] = None,
                darkfield_image_path: Optional[Path] = None,
                shading_images_path: Optional[Path] = None) -> Generator[Dict[str, int], None, None]:
    
    hotpixel_flag = False
    shading_flag = False
    round_registration_flag = False
    tile_registration_flag = False
    write_polyDT_tiff = False
    if correction_option == "Hotpixel correct" and (noise_map_path is not None):
        hotpixel_flag = True
    elif correction_option == "Flatfield correct" and (noise_map_path is not None) and (darkfield_image_path is not None) and (shading_images_path is not None):
        shading_flag = True
    
    for option_name, is_selected in stitching_options.items():
        if is_selected:
            if option_name == "Register polyDT/readouts within tile":
                round_registration_flag = True
            elif option_name == "Register polyDT across tiles":
                tile_registration_flag = True
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
        "PolyDT Tile": 0,
        "PolyDT Round": 0,
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
        compressor = blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.Blosc.BITSHUFFLE)
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
                    

                    # yellow readout zarr store
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
                                    
                                if shading_flag:
                                    raw_data = correct_shading(noise_map,darkfield_image,shading_images[ch_idx],raw_data)
                                elif not(shading_flag) and (hotpixel_flag):
                                    raw_data = replace_hot_pixels(noise_map,raw_data)

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
        run_optical_flow = False
        from wf_merfish.postprocess.DataRegistration import DataRegistration

        for tile_idx in range(num_tiles):
            data_register_factory = DataRegistration(dataset_path=output_dir_path,
                                                    overwrite_registered=False,
                                                    perform_optical_flow=run_optical_flow,
                                                    tile_idx=tile_idx)
            data_register_factory.generate_registrations()
            data_register_factory.apply_registration_to_bits()

            progress_updates['PolyDT Tile'] = ((tile_idx+1) / num_tiles) * 100
            yield progress_updates

            del data_register_factory
            
    progress_updates['PolyDT Tile'] = 100
    yield progress_updates

    if tile_registration_flag:
        
        multiview_stitcher = True
        
        try:
        
            import dask.diagnostics
            import ngff_zarr
            from multiview_stitcher import (
                fusion,
                io,
                msi_utils,
                ngff_utils,
                registration,
                vis_utils
            )
            import matplotlib.pyplot as plt
            
        except:
            multiview_stitcher = False


        if multiview_stitcher:
            polyDT_output_dir_path = output_dir_path / Path('polyDT')

            msims = []

            stitched_dir_path = output_dir_path / Path('stitched')
            stitched_dir_path.mkdir(parents=True, exist_ok=True)

            tile_ids = sorted([entry.name for entry in polyDT_output_dir_path.iterdir() if entry.is_dir()],
                              key=lambda x: int(x.split('tile')[1].split('.zarr')[0]))

            for tile_idx, tile_id in enumerate(tile_ids):
                
                polyDT_current_path = polyDT_output_dir_path / Path(tile_id) / Path("round000.zarr")
                if not(polyDT_current_path.exists()):
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
                    
                    overlap = np.asarray(polyDT_current_tile.attrs['tile_overlap'],
                                                        dtype=np.float32)

                    im_data = np.asarray(polyDT_current_tile['registered_data'],dtype=np.uint16)
                    
                    shape = {dim: im_data.shape[idim] for idim, dim in enumerate(scale.keys())}
                    translation = {dim: np.round((tile_grid_positions[dim]),2) for dim in scale}

                    ngff_im = ngff_zarr.NgffImage(
                            im_data,
                            dims=('z', 'y', 'x'),
                            scale=scale,
                            translation=translation,
                            )
                    
                    # These translations match the recorded stage position converted to pixels
                    
                    ngff_multiscales = ngff_zarr.to_multiscales(ngff_im)

                    zarr_path = stitched_dir_path / Path(tile_id + ".zarr")

                    ngff_zarr.to_ngff_zarr(zarr_path, ngff_multiscales)
                    
                    del polyDT_current_tile
                else:
                    zarr_path = stitched_dir_path / Path(tile_id + ".zarr")

                msim = ngff_utils.ngff_multiscales_to_msim(
                            ngff_zarr.from_ngff_zarr(zarr_path),
                            transform_key=io.METADATA_TRANSFORM_KEY)

                msims.append(msim)
                
                
                progress_updates["PolyDT Round"] = ((tile_idx+1) / num_tiles) * 100
                    
            fig, ax = vis_utils.plot_positions(
                msims,
                use_positional_colors=True, # set to False for faster execution in case of more than 20 tiles/views
                transform_key='affine_metadata'
                )

            plt.show()
                
            params = registration.register(
                msims,
                registration_binning={'z': 2, 'y': 4, 'x': 4},
                reg_channel_index=0,
                transform_key='affine_metadata',
            )
                
            for msim, param in zip(msims, params):
                msi_utils.set_affine_transform(msim, param, transform_key='affine_registered', base_transform_key='affine_metadata')

            for imsim, msim in enumerate(msims):
                affine = np.array(msi_utils.get_transform_from_msim(msim, transform_key='affine_registered')[0])
                polyDT_current_path = polyDT_output_dir_path / Path(tile_ids[imsim]) / Path("round000.zarr")
                polyDT_current_tile = zarr.open(polyDT_current_path,mode='a')
                polyDT_current_tile.attrs['affine_zyx_um'] = affine.tolist()
                del polyDT_current_tile

            sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

            stitched_output_path = stitched_dir_path / Path("round000_fused.zarr")

            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                with dask.diagnostics.ProgressBar():
                    fused = fusion.fuse(
                        sims[:],
                        transform_key='affine_registered',
                        output_chunksize=512
                        )
                
                    fused.compute()
                
            stitched_output_zarr = zarr.open(stitched_output_path,mode='a')
            current_stitched_data = stitched_output_zarr.zeros('fused_data',
                                                        shape=fused.shape,
                                                        chunks=(1,1,1,512,512),
                                                        compressor=compressor,
                                                        dtype=np.uint16)

            current_stitched_data[:] = np.asarray(fused,dtype=np.uint16)
            
            del fused
            gc.collect()
            current_stitched_data.attrs['voxel_zyx_um'] = np.array([float(axial_step),float(pixel_size),float(pixel_size)]).tolist()
    
if __name__ == '__main__':
    
    data_path = Path('/mnt/opm3/')
    codebook_path = Path('/home/qi2lab/Documents/github/wf-merfish/exp_order.csv')
    exp_order_path = Path('/home/qi2lab/Documents/github/wf-merfish/exp_order.csv')

    test = postprocess(selected_options={'Hotpixel correct': False,
                                        'Flatfield correct': False,
                                        'Register polyDT each tile across rounds': True,
                                        'Register polyDT all tiles first round': True},
                        dataset_path=data_path,
                        codebook_path=codebook_path,
                        bit_order_path=exp_order_path)
    
    test.start()