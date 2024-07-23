from pathlib import Path
import zarr
from merfish3danalysis.utils._opmtools import chunked_orthogonal_deskew
from numcodecs import blosc
from itertools import compress
import pandas as pd
import numpy as np

input_path = Path('/mnt/data/qi2lab/20230602_humanOB/raw_zarr/humanOB.zarr')
codebook_path = Path('/mnt/data/qi2lab/20230602_humanOB/raw_zarr/codebook.csv')
bit_order_path = Path('/mnt/data/qi2lab/20230602_humanOB/raw_zarr/bit_order.csv')
fidicual_ch_id = 'ch488'

# parse for rounds, tile_ids, and 
round_ids = sorted([d.name for d in input_path.iterdir() if d.is_dir()])
test_round = input_path / round_ids[0]
tile_ids = sorted([d.name for d in test_round.iterdir() if d.is_dir()])
test_tile = test_round / tile_ids[0]

input_zarr = zarr.open(input_path,mode='r')

scan_step = 0.4
axial_step = .230
tile_overlap = 0.2
binning = 1
pixel_size = .115
e_per_ADU = .24
num_r = len(round_ids)
num_bits = 18
num_merfish_bits = num_bits - 2
num_tiles = len(tile_ids)
chan_purple_active = True
chan_blue_active = True
chan_yellow_active = True
chan_red_active = True
exposure_purple_ms = 60.
exposure_blue_ms = 60.
exposure_yellow_ms = 60.
exposure_red_ms = 60.
active_channels = [chan_purple_active,
                    chan_blue_active,
                    chan_yellow_active,
                    chan_red_active]
exposures_ms = [exposure_purple_ms,
                exposure_blue_ms,
                exposure_yellow_ms,
                exposure_red_ms]
channel_idxs = [0,1,2,3]
channels_in_data = list(compress(channel_idxs, active_channels))

# create output directory
output_dir_path_base = Path('/mnt/data/qi2lab/20230602_humanOB/')
output_dir_path = output_dir_path_base / 'qi2lab_merfish_v2'

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
    
    # initialize tile counter and channel information
    ex_wavelengths=[.405,.488,.561,.635]
    em_wavelengths=[.480,.520,.580,.670]
                    
    tile_idx=0

    channel_ids = ['ch405','ch488','ch561','ch635']

    # load codebook and experimental order from disk
    codebook_df = pd.read_csv(codebook_path)
    bit_order_df = pd.read_csv(bit_order_path)
    bit_order = bit_order_df.to_numpy()
    
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
    calibrations_zarr.attrs["e_per_ADU"] = float(e_per_ADU)
    calibrations_zarr.attrs["na"] = float(1.35)
    calibrations_zarr.attrs["ri"] = float(1.4)
    
    # loop over all tiles.
    for tile_idx in range(num_tiles):

        tile_name = tile_ids[tile_idx]
        store_tile_name = 'tile'+str(tile_idx).zfill(4)

        for r_idx in range(num_r):

            round_name = round_ids[r_idx]
            
            # open stage positions file
            stage_position_zarr_grp = Path(round_name) / Path(tile_name) / Path('ch488') / Path('stage_position')
            stage_positions = np.array(input_zarr[stage_position_zarr_grp])

            # grab recorded stage positions
            stage_x = np.round(stage_positions[2],2)
            stage_y = np.round(stage_positions[1],2)
            stage_z = np.round(stage_positions[0],2)

            # grab channels in this tile
            active_round_tile_path = input_path / Path(round_name) / Path(tile_name)
            active_ch_ids = sorted([d.name for d in active_round_tile_path.iterdir() if d.is_dir()])
            
            if 'ch405' in active_ch_ids:
                chan_purple_active_tile = True
            else:
                chan_purple_active_tile = False
            if 'ch488' in active_ch_ids:
                chan_blue_active_tile = True
            else:
                chan_blue_active_tile = False
            if 'ch561' in active_ch_ids:
                chan_yellow_active_tile = True
            else:
                chan_yellow_active_tile = False
            if 'ch635' in active_ch_ids:
                chan_red_active_tile = True
            else:
                chan_red_active_tile = False

            active_channels_tile = [chan_purple_active_tile,
                                    chan_blue_active_tile,
                                    chan_yellow_active_tile,
                                    chan_red_active_tile]
            channels_idxs_in_data_tile = list(compress(channel_idxs, active_channels_tile))
            channels_ids_in_data_tile = list(compress(channel_ids, active_channels_tile))
            
            if r_idx == (num_r-1):
                # dapi zarr store
                dapi_tile_dir_path = dapi_output_dir_path / Path(store_tile_name)
                dapi_tile_dir_path.mkdir(parents=True, exist_ok=True)
                dapi_round_dir_path = dapi_tile_dir_path / Path(round_name + '.zarr')
                dapi_round_zarr = zarr.open(str(dapi_round_dir_path), mode="a")

            # polyDT zarr store
            polyDT_tile_dir_path = polyDT_output_dir_path / Path(store_tile_name)
            polyDT_tile_dir_path.mkdir(parents=True, exist_ok=True)
            polydT_round_dir_path = polyDT_tile_dir_path / Path(round_name + '.zarr')
            polydT_round_zarr = zarr.open(str(polydT_round_dir_path), mode="a")
            
            if r_idx < (num_r-1):
                # yellow readout zarr store
                yellow_readout_round_idx = bit_order[r_idx,1]
                yellow_bit_name = "bit"+str(yellow_readout_round_idx).zfill(2)
                yellow_tile_dir_path = readout_output_dir_path / Path(store_tile_name)
                yellow_tile_dir_path.mkdir(parents=True, exist_ok=True)
                yellow_bit_dir_path = yellow_tile_dir_path / Path(yellow_bit_name + '.zarr')
                yellow_bit_zarr = zarr.open(str(yellow_bit_dir_path), mode="a")
                
                # red readout zarr store
                red_readout_round_idx = bit_order[r_idx,2]
                red_bit_name = "bit"+str(red_readout_round_idx).zfill(2)
                red_tile_dir_path = readout_output_dir_path / Path(store_tile_name)
                red_tile_dir_path.mkdir(parents=True, exist_ok=True)
                red_bit_dir_path = red_tile_dir_path / Path(red_bit_name + '.zarr')
                red_bit_zarr = zarr.open(str(red_bit_dir_path), mode="a")
            
            # loop over all channels in this round/tile
            for channel_id, ch_idx in zip(channels_ids_in_data_tile,channels_idxs_in_data_tile):
                            
                if channel_id == 'ch405':
                    current_channel = dapi_round_zarr
                    current_channel.attrs["tile_overlap"] = tile_overlap
                    current_channel.attrs["psf_idx"] = int(0)
                elif channel_id == 'ch488':
                    current_channel = polydT_round_zarr
                    if r_idx < (num_r-1):
                        current_channel.attrs['bits'] = bit_order[r_idx,:].tolist()
                    current_channel.attrs["tile_overlap"] = tile_overlap
                    current_channel.attrs["psf_idx"] = int(1)
                elif channel_id == 'ch561':
                    current_channel = yellow_bit_zarr
                    current_channel.attrs['round'] = int(r_idx)
                    current_channel.attrs["psf_idx"] = int(2)
                elif channel_id == 'ch635':
                    current_channel = red_bit_zarr
                    current_channel.attrs['round'] = int(r_idx)
                    current_channel.attrs["psf_idx"] = int(3)
                
                converted_data_exists = False
                try:
                    test = np.array(current_channel['corrected_data'][0:1,0:1,0:1])
                    converted_data_exists = True
                except:
                    converted_data_exists = False
                
                if not(converted_data_exists):
                    active_round_tile_channel_path = Path(round_name) / Path(tile_name) / Path(channel_id)
                    oblique_data_grp = active_round_tile_channel_path / Path('raw_data')
                    oblique_psf_grp = active_round_tile_channel_path / Path('raw_psf')
                    oblique_theta_grp = active_round_tile_channel_path / Path('theta')
                    
                    oblique_data = np.array(input_zarr[str(oblique_data_grp)])
                    oblique_psf = np.array(input_zarr[str(oblique_psf_grp)])
                    oblique_theta = np.array(input_zarr[str(oblique_theta_grp)])
                    ex_wvl = ex_wavelengths[ch_idx]
                    em_wvl = em_wavelengths[ch_idx]
                    exposure_ms = exposures_ms[ch_idx]
                        
                    corrected_data = chunked_orthogonal_deskew(oblique_image=oblique_data,
                                                            psf_data=oblique_psf)
                            
                    current_corrected_data = current_channel.zeros('corrected_data',
                                                                    shape=(corrected_data.shape[0],corrected_data.shape[1],corrected_data.shape[2]),
                                                                    chunks=(1,corrected_data.shape[1],corrected_data.shape[2]),
                                                                    compressor=compressor,
                                                                    dtype=np.uint16)
                        
                    current_channel.attrs['stage_zyx_um'] = np.array([stage_z,stage_y,stage_x]).tolist()
                    current_channel.attrs['voxel_zyx_um'] = np.array([float(axial_step),float(pixel_size),float(pixel_size)]).tolist()
                    current_channel.attrs['excitation_um'] = float(ex_wvl)
                    current_channel.attrs['e_per_ADU'] = float(e_per_ADU)
                    current_channel.attrs['emission_um'] = float(em_wvl)
                    current_channel.attrs['exposure_ms'] = float(exposure_ms)
                    current_channel.attrs['hotpixel'] = False
                    current_channel.attrs['shading'] = False

                    current_corrected_data[:] = corrected_data