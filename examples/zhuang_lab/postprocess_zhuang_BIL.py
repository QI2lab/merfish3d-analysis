#!/usr/bin/env python
'''
qi2lab WF MERFISH / FISH processing

Microscope post-processing v1.1

Convert BIL Zhuang lab data to qi2lab merfish-zarr
- Rewrites Zhuang BIL data into compressed Zarr in qi2lab MERFISH format

Change log:
Shepherd 04/24 - initial work on Zhuang lab converter
'''

# imports
import numpy as np
from pathlib import Path
import gc
from itertools import compress
import zarr
from numcodecs import blosc
from tifffile import imread
import pandas as pd
from typing import Dict, Generator, Optional
import json
import re
from ufish.api import UFish
from tqdm import tqdm

def load_codebook(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Rename the first column to "gene"
    df.columns = ['gene'] + df.columns[1:].tolist()
    
    # Identify "RS" columns and rename them to "bitXX"
    rs_columns = [col for col in df.columns if col.startswith('RS')]
    bit_mapping = {rs_columns[i]: f'bit{str(i+1).zfill(2)}' for i in range(len(rs_columns))}
    df.rename(columns=bit_mapping, inplace=True)
    
    # Extract a numpy matrix for the "bit" columns
    bit_columns = [col for col in df.columns if col.startswith('bit')]
    bit_matrix = df[bit_columns].to_numpy()
    
    # Extract a list from the "gene" column
    gene_list = df['gene'].tolist()
    
    return df, bit_matrix, gene_list

def load_metadata(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Extract the required columns
    channel_name_col = [col for col in df.columns if col.startswith('channelName')][0]
    color_col = [col for col in df.columns if col.startswith('color')][0]
    zpos_col = [col for col in df.columns if col.startswith('zPos')][0]

    # Extract lists for ChannelName and color columns
    channel_name_list = df[channel_name_col].tolist()
    color_list = df[color_col].tolist()
    
    # Parse the zPos column to extract arrays and calculate the absolute difference
    zpos_list = [np.fromstring(x.strip('[]'), sep=' ') for x in df[zpos_col].astype(str)]
    zpos_array = zpos_list[0]  # Assuming all rows have the same zPos value
    abs_diff_zpos = abs(zpos_array[1] - zpos_array[0])
    
    return channel_name_list, color_list, zpos_array, abs_diff_zpos

def extract_pixel_size(file_path):
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    return float(data['microns_per_pixel'])

# parse experimental directory, load data, and process
def postprocess_zhuang_BIL(dataset_path: Path):
           
    # read metadata for this experiment
    data_files_path = dataset_path / Path("mouse1_sample1_raw")
    codebook_path = dataset_path / Path("additional_files") / Path('codebook.csv')
    bit_info_path = dataset_path / Path("additional_files") / Path('data_organization_raw.csv')
    microscope_param_path = dataset_path / Path("additional_files") / Path('microscope.json')
    stage_positions_path = dataset_path / Path("additional_files") / Path("fov_positions") / Path("mouse1_sample1.txt")
    
    ufish = UFish()
    ufish.load_weights_from_internet()
    
    codebook_df, codebook_matrix, gene_list = load_codebook(codebook_path)
    channel_name_list, em_wavelengths, zpos_array, abs_diff_zpos = load_metadata(bit_info_path)
    tile_file_ids = sorted(list(data_files_path.glob('*.tif')))
    stage_positions = np.loadtxt(stage_positions_path, delimiter=',')
      
    axial_step = abs_diff_zpos
    tile_overlap = 0.2
    binning = 1
    pixel_size = extract_pixel_size(microscope_param_path)
    gain = 1
    num_r = 19
    num_bits = len(em_wavelengths)
    num_merfish_bits = num_bits - 2
    polyDT_bit = num_bits - 2
    dapi_bit = num_bits - 1
    num_tiles = len(tile_file_ids)
    chan_purple_active = True
    chan_blue_active = True
    chan_red_active = True
    chan_nir_active = True
    exposure_purple_ms = 100.
    exposure_blue_ms = 100.
    exposure_red_ms = 100.
    exposure_nir_ms = 100.
    active_channels = [chan_purple_active,
                       chan_blue_active,
                       chan_red_active,
                       chan_nir_active]
    exposures_ms = [exposure_purple_ms,
                    exposure_blue_ms,
                    exposure_red_ms,
                    exposure_nir_ms]
    channel_idxs = [0,1,2,3]
    channels_in_data = list(compress(channel_idxs, active_channels))

    # create output directory
    output_dir_path_base = dataset_path
    output_dir_path = output_dir_path_base / 'qi2lab_merfish'
    
    qi2lab_exists = True
    # check if qi2lab zarr structure exists
    if output_dir_path.exists():
        dapi_output_dir_path = output_dir_path / Path('dapi')
        if not(dapi_output_dir_path.exists()):
            qi2lab_exists = False
        
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
    
    qi2lab_exists = False
            
    if not(qi2lab_exists):
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # create directory for data type
        dapi_output_dir_path = output_dir_path / Path('dapi')
        dapi_output_dir_path.mkdir(parents=True, exist_ok=True)
              
        polyDT_output_dir_path = output_dir_path / Path('polyDT')
        polyDT_output_dir_path.mkdir(parents=True, exist_ok=True)
        
        readout_output_dir_path = output_dir_path / Path('readouts')
        readout_output_dir_path.mkdir(parents=True, exist_ok=True)

        calibrations_output_dir_path = output_dir_path / Path('calibrations.zarr')
        calibrations_output_dir_path.mkdir(parents=True, exist_ok=True)
        compressor = blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.Blosc.BITSHUFFLE)
        calibrations_zarr = zarr.open(str(calibrations_output_dir_path), mode="a")
        
        blosc.set_nthreads(20)
                        
        tile_idx=0

        # load codebook and experimental order from disk
        codebook_df = pd.read_csv(codebook_path)
        
        # save codebook
        calibrations_zarr.attrs['codebook'] = codebook_df.values.tolist()

        # save experimental oder
        calibrations_zarr.attrs['experiment_order'] = em_wavelengths

        # helpful metadata needed by registration and decoding classes so they don't have to traverse nested zarr groups   
        calibrations_zarr.attrs["num_rounds"] = int(num_r)
        calibrations_zarr.attrs["num_tiles"] = int(num_tiles)
        calibrations_zarr.attrs["channels_in_data"] = channels_in_data
        calibrations_zarr.attrs["tile_overlap"] = float(tile_overlap)
        calibrations_zarr.attrs["binning"] = int(binning)
        calibrations_zarr.attrs["gain"] = float(gain)
        calibrations_zarr.attrs["na"] = float(1.4)
        calibrations_zarr.attrs["ri"] = float(1.51)
            
        # loop over all tiles.
        for tile_file_id in tqdm(tile_file_ids,desc='tile',leave=True):
            
            pattern = r"aligned_images(\d+)\.tif"
            match = re.search(pattern, str(tile_file_id.name))
            if match:
                # Extract the number as an integer and add it to the list
                tile_idx_file = int(match.group(1))

            data_loaded = True
            try:
                tile_data = imread(tile_file_id)
                stage_x = stage_positions[tile_idx_file,0]
                stage_y = stage_positions[tile_idx_file,1]
            except:
                data_loaded = False
            
            if data_loaded:
                        
                for bit_idx in tqdm(range(num_bits),desc='bit',leave=False):
                    fish_data = False
                    if bit_idx < num_merfish_bits:
                        # bit zarr store
                        bit_name = "bit"+str(bit_idx+1).zfill(2)
                        tile_dir_path = readout_output_dir_path / Path('tile'+str(tile_idx_file).zfill(4))
                        tile_dir_path.mkdir(parents=True, exist_ok=True)
                        bit_dir_path = tile_dir_path / Path(bit_name + '.zarr')
                        current_channel = zarr.open(str(bit_dir_path), mode="a")
                        fish_data = True
                    elif bit_idx == polyDT_bit:
                        # polyDT zarr store
                        polyDT_tile_dir_path = polyDT_output_dir_path / Path('tile'+str(tile_idx_file).zfill(4))
                        polyDT_tile_dir_path.mkdir(parents=True, exist_ok=True)
                        polydT_round_dir_path = polyDT_tile_dir_path / Path('round000.zarr')
                        current_channel = zarr.open(str(polydT_round_dir_path), mode="a")
                        fish_data = False
                    elif bit_idx == dapi_bit:
                        # polyDT zarr store
                        dapi_tile_dir_path = dapi_output_dir_path / Path('tile'+str(tile_idx_file).zfill(4))
                        dapi_tile_dir_path.mkdir(parents=True, exist_ok=True)
                        dapi_round_dir_path = dapi_tile_dir_path / Path('round000.zarr')
                        current_channel = zarr.open(str(dapi_round_dir_path), mode="a")
                        fish_data = False
                        
                    raw_data = np.squeeze(tile_data[bit_idx,:])

                    ex_wvl = np.round(em_wavelengths[bit_idx]/100 - 30.0)
                    em_wvl = np.round(em_wavelengths[bit_idx],3)
                    exposure_ms = 100.
                                
                    current_raw_data = current_channel.zeros('registered_data',
                                                            shape=(raw_data.shape[0],raw_data.shape[1],raw_data.shape[2]),
                                                            chunks=(1,raw_data.shape[1],raw_data.shape[2]),
                                                            compressor=compressor,
                                                            dtype=np.uint16)
                    
                    current_channel.attrs['stage_yx_um'] = np.array([stage_y,stage_x]).tolist()
                    current_channel.attrs['voxel_zyx_um'] = np.array([float(axial_step),float(pixel_size),float(pixel_size)]).tolist()
                    current_channel.attrs['excitation_um'] = float(ex_wvl)
                    current_channel.attrs['gain'] = float(gain)
                    current_channel.attrs['emission_um'] = float(em_wvl)
                    current_channel.attrs['exposure_ms'] = float(exposure_ms)
                    current_channel.attrs['bit_name'] = channel_name_list[bit_idx]
                                    
                    current_raw_data[:] = raw_data
                    
                    if fish_data:
                        ufish_data = np.zeros((raw_data.shape[0],raw_data.shape[1],raw_data.shape[2]),
                            dtype=np.float32)
                        for z_idx in tqdm(range(raw_data.shape[0]),desc='ufish z',leave=False):
                            _, ufish_data[z_idx,:] = ufish.predict(raw_data[z_idx,:],
                                                                axes='yx',
                                                                batch_size=1)
                        current_ufish_data = current_channel.zeros('registered_ufish_data',
                                                            shape=(ufish_data.shape[0],ufish_data.shape[1],ufish_data.shape[2]),
                                                            chunks=(1,ufish_data.shape[1],ufish_data.shape[2]),
                                                            compressor=compressor,
                                                            dtype=np.float32)
                        
                        current_ufish_data[:] = ufish_data

    return True

if __name__ == '__main__':
    
    data_path = Path('/mnt/opm3/20240329_zhuang_MOP_data/')
    test = postprocess_zhuang_BIL(data_path)