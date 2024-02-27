from pathlib import Path
import zarr
from numcodecs import blosc
import time
import numpy as np
from psfmodels import make_psf
from tifffile import imread
import pandas as pd
from clij2fft.richardson_lucy import richardson_lucy_nc, getlib
import gc
import cupy as cp

data_dir_path = Path('/home/qi2lab/Documents/github/wf-merfish/examples/simulated_images/cylinder/images/jitter-0_shift_amp-0_prop_fn-0_prop_fp-0')

codebook_path = data_dir_path / Path('codebook.csv')
bit_order_path = data_dir_path / Path('bit_order.csv')

tiff_files = data_dir_path.glob('*.tiff')
for tiff_file in tiff_files:
    data = imread(tiff_file)
data = np.swapaxes(data,0,1)

readout_dir_path = data_dir_path / Path('processed') / Path('readouts')
readout_dir_path.mkdir(parents=True, exist_ok=True)
calibration_dir_path = data_dir_path / Path('processed') / Path('calibrations.zarr')
calibration_dir_path.mkdir(parents=True, exist_ok=True)
compressor = blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.Blosc.BITSHUFFLE)
calibrations_zarr = zarr.open(str(calibration_dir_path), mode="a")

na = 1.35
pixel_size = .088
axial_step = .310
ex_wvl = .561
em_wvl = .580
num_r = 16
num_tiles = 1
channels_in_data = ['Yellow']
tile_overlap = 0.2
binning = 2
gain = 0.5
tile_name = 'tile0000'
stage_x = 100.0
stage_y = 200.0
stage_z = 300.0
exposure_ms = 100.
hotpixel_flag = False
shading_flag = False

psf = make_psf(z=33,
               nx=33,
               dxy=pixel_size,
               dz=axial_step,
               wvl=em_wvl,
               ns=1.4,
               ni=1.515)

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
calibrations_zarr.attrs["gain"] = float(gain)


channel_psfs = np.array(psf)

psf_data = calibrations_zarr.zeros('psf_data',
                                    shape=(channel_psfs.shape[0],channel_psfs.shape[1],channel_psfs.shape[2]),
                                    chunks=(1,channel_psfs.shape[1],channel_psfs.shape[2]),
                                    compressor=compressor,
                                    dtype=np.uint16)
psf_data[:] = channel_psfs


for r_idx in range(num_r):
    readout_round_idx = bit_order[r_idx,1] - 1
    yellow_bit_name = "bit"+str(readout_round_idx).zfill(2)
    yellow_tile_dir_path = readout_dir_path / Path(tile_name)
    yellow_tile_dir_path.mkdir(parents=True, exist_ok=True)
    yellow_bit_dir_path = yellow_tile_dir_path / Path(yellow_bit_name + '.zarr')
    yellow_bit_zarr = zarr.open(str(yellow_bit_dir_path), mode="a")
    current_channel = yellow_bit_zarr
    
    raw_data = data[readout_round_idx,:]
    
    current_channel.attrs['round'] = int(r_idx)
    current_channel.attrs["psf_idx"] = int(0)
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

    lib = getlib()
            
    image_decon = richardson_lucy_nc(raw_data,
                                    psf=psf,
                                    numiterations=40,
                                    regularizationfactor=.00001,
                                    lib=lib)
    
    data_reg_zarr = current_channel.zeros('registered_data',
                                        shape=image_decon.shape,
                                        chunks=(1,image_decon.shape[1],image_decon.shape[2]),
                                        compressor=compressor,
                                        dtype=np.uint16)
    
    data_reg_zarr[:] = image_decon
