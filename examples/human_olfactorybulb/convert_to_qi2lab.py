"""
Convert raw qi2lab WF MERFISH OB data to qi2labdatastore.

Change path on line 287

Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
import numpy as np
import pandas as pd
from psfmodels import make_psf
from tifffile import imread
from tqdm import tqdm
from merfish3danalysis.utils._dataio import read_metadatafile
from merfish3danalysis.utils._imageprocessing import replace_hot_pixels
from itertools import compress

def convert_data(root_path: Path):
    """Convert qi2lab microscope data to qi2lab datastore.
    
    Parameters
    ----------
    root_path: Path
        path to dataset
    """

    # load codebook
    # --------------
    codebook_path = root_path / Path("codebook.csv")
    codebook = pd.read_csv(codebook_path)

    # load experimental order
    # -----------------------
    experiment_order_path = root_path / Path("bit_order.csv")
    df_experiment_order = pd.read_csv(experiment_order_path)
    experiment_order = df_experiment_order.values

    # load experiment metadata
    # ------------------------
    metadata_path = root_path / Path("scan_metadata.csv")
    metadata = read_metadatafile(metadata_path)
    root_name = metadata["root_name"]
    num_rounds = metadata["num_r"]
    num_tiles = metadata["num_xyz"]
    num_ch = metadata["num_ch"]
    camera = metadata["camera"]
    channels_active = [
        metadata["blue_active"],
        metadata["yellow_active"],
        metadata["red_active"],
    ]
    channels_exposure_ms = [
        metadata["blue_exposure"],
        metadata["yellow_exposure"],
        metadata["red_exposure"],
    ]
    channel_order = metadata["channels_reversed"]
    voxel_size_zyx_um = [0.310, 0.098, 0.098] # in acq. metadata, will fix
    na = 1.35 # in acq. metadata, will fix
    ri = 1.51 # in acq. metadata, will fix
    ex_wavelengths_um = [0.488, 0.561, 0.635]
    em_wavelengths_um = [0.520, 0.580, 0.670]
    channel_idxs = [0, 1, 2]
    channels_in_data = list(compress(channel_idxs, channels_active))
    # these are also in acquisition metadata, left over hard-coding here. 
    stage_flipped_x = True
    stage_flipped_y = True
    image_rotated = True
    image_flipped_y = True
    image_flipped_x = False

    if camera == 'bsi':
        # camera gain and offset
        # from Photometrics manual
        # ----------------------
        e_per_ADU = 1.0
        offset = 100.0
        noise_map = None
    elif camera == "flir":
        # camera gain and offset
        # from flir calibration
        # ----------------------
        e_per_ADU = .03
        offset = 0.0
        noise_map = imread(root_path / Path(r"flir_hot_pixel_image.tif"))
    elif camera == "orcav3":
        # camera gain and offset
        # from hamamatsu calibration
        # ----------------------
        e_per_ADU = .46
        offset = 100.0
        noise_map = offset * np.ones((2048,2048),dtype=np.uint16)

    # generate PSFs
    # --------------
    channel_psfs = []
    for channel_id in channels_in_data:
        psf = make_psf(
            z=51,
            nx=51,
            dxy=voxel_size_zyx_um[1],
            dz=voxel_size_zyx_um[0],
            NA=na,
            wvl=em_wavelengths_um[channel_id],
            ns=1.47,
            ni=ri,
            ni0=ri,
            model="vectorial",
        ).astype(np.float32)
        psf = psf / np.sum(psf, axis=(0, 1, 2))
        channel_psfs.append(psf)
    channel_psfs = np.asarray(channel_psfs, dtype=np.float32)

    # # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    datastore.num_rounds = num_rounds
    datastore.codebook = codebook
    datastore.channels_in_data = ["alexa488", "atto565", "alexa647"]
    datastore.experiment_order = experiment_order
    datastore.num_tiles = num_tiles
    datastore.microscope_type = "3D" # this could be automatically determined from z steps
    datastore.camera_model = camera
    datastore.tile_overlap = 0.2 # in acq. metadata, will fix.
    datastore.e_per_ADU = e_per_ADU
    datastore.na = na
    datastore.ri = ri
    datastore.binning = 1 # in acq. metadata, will fix.
    datastore.noise_map = noise_map
    datastore._shading_maps = np.ones((3, 2048, 2048), dtype=np.float32)
    datastore.channel_psfs = channel_psfs
    datastore.voxel_size_zyx_um = voxel_size_zyx_um
    datastore.baysor_path = Path(r"/home/qi2lab/Documents/github/Baysor/bin/baysor/bin/./baysor")
    datastore.baysor_options = Path(r"/home/qi2lab/Documents/github/merfish3d-analysis/qi2lab.toml")
    datastore.julia_threads = 20

    # Update datastore state to note that calibrations are doen
    datastore_state = datastore.datastore_state
    datastore_state.update({"Calibrations": True})
    datastore.datastore_state = datastore_state
    
    round_idx = 0
    if stage_flipped_x or stage_flipped_y:
        for tile_idx in range(num_tiles):
            stage_position_path = root_path / Path(
                root_name
                + "_r"
                + str(round_idx + 1).zfill(4)
                + "_tile"
                + str(tile_idx).zfill(4)
                + "_stage_positions.csv"
            )
            stage_positions = read_metadatafile(stage_position_path)
            stage_x = np.round(float(stage_positions["stage_x"]), 2)
            stage_y = np.round(float(stage_positions["stage_y"]), 2)
            if tile_idx == 0:
                max_y = stage_y
                max_x = stage_x
            else:
                if max_y < stage_y:
                    max_y = stage_y
                if max_x < stage_x:
                    max_x = stage_x

    for round_idx in tqdm(range(num_rounds), desc="rounds"):
        for tile_idx in tqdm(range(num_tiles), desc="tile",leave=False):
            # initialize datastore tile
            # this creates the directory structure and links fiducial rounds <-> readout bits
            if round_idx == 0:
                datastore.initialize_tile(tile_idx)

            # load raw image
            image_path = (
                root_path
                / Path(root_name + "_r"+str(round_idx+1).zfill(4)+"_tile"+str(tile_idx).zfill(4)+"_1")
                / Path(root_name + "_r"+str(round_idx+1).zfill(4)+"_tile"+str(tile_idx).zfill(4)+"_NDTiffStack.tif")
            )

            raw_image = imread(image_path)
            raw_image = np.swapaxes(raw_image,0,1)
            if tile_idx == 0 and round_idx == 0:
                correct_shape = raw_image.shape
            if raw_image is None or raw_image.shape != correct_shape:
                    print('\nround='+str(round_idx+1)+'; tile='+str(tile_idx+1))
                    print('Found shape: '+str(raw_image.shape))
                    print('Correct shape: '+str(correct_shape))
                    print('Replacing data with zeros.\n')
                    raw_image = np.zeros(correct_shape, dtype=np.uint16)
                    
            if channel_order == "reversed":
                raw_image = np.flip(raw_image,axis=0)
                
            if image_rotated:
                raw_image = np.rot90(raw_image, k=-1, axes=(3, 2))
                
            if image_flipped_y:
                raw_image = np.flip(raw_image,axis=2)
                
            if image_flipped_x:
                raw_image = np.flip(raw_image,axis=3)
                
            # Correct for gain and offset
            if camera == "flir":
                raw_image = replace_hot_pixels(noise_map,raw_image)
                raw_image = replace_hot_pixels(
                    np.max(raw_image,axis=0),
                    raw_image,
                    threshold=100)
            
            raw_image = ((raw_image.astype(np.float32) - offset) * e_per_ADU)
            raw_image[raw_image < 0.0] = 0.0
            raw_image = raw_image.astype(np.uint16)
            
            # load stage position
            stage_position_path = (
                root_path
                / Path(root_name + "_r"+str(round_idx+1).zfill(4)+"_tile"+str(tile_idx).zfill(4)+"_stage_positions.csv")
            )
            df_stage_positions = read_metadatafile(stage_position_path)
            stage_x = np.round(float(df_stage_positions['stage_x']),2)
            stage_y = np.round(float(df_stage_positions['stage_y']),2)
            stage_z = np.round(float(df_stage_positions['stage_z']),2)
            
            if stage_flipped_x or stage_flipped_y:
                if stage_flipped_y:
                    corrected_y =  max_y - stage_y
                else:
                    corrected_y = stage_y
                if stage_flipped_x:
                    corrected_x = max_x - stage_x
                else:
                    corrected_x = stage_x
                
            stage_pos_zyx_um = np.asarray([stage_z,corrected_y,corrected_x],dtype=np.float32)

            # write fidicual data and metadata
            datastore.save_local_corrected_image(
                np.squeeze(raw_image[0, :]).astype(np.uint16),
                tile=tile_idx,
                psf_idx=0,
                gain_correction=True,
                hotpixel_correction=True,
                shading_correction=False,
                round=round_idx,
            )
            
            datastore.save_local_stage_position_zyx_um(
                stage_pos_zyx_um, tile=tile_idx, round=round_idx
            )
            
            datastore.save_local_wavelengths_um(
                (ex_wavelengths_um[0], em_wavelengths_um[0]),
                tile=tile_idx,
                round=round_idx,
            )

            # write first readout channel and metadata
            datastore.save_local_corrected_image(
                np.squeeze(raw_image[1, :]).astype(np.uint16),
                tile=tile_idx,
                psf_idx=1,
                gain_correction=True,
                hotpixel_correction=True,
                shading_correction=False,
                bit=int(experiment_order[round_idx,1])-1,
            )
            datastore.save_local_wavelengths_um(
                (ex_wavelengths_um[1], em_wavelengths_um[1]),
                tile=tile_idx,
                bit=int(experiment_order[round_idx,1])-1,
            )
            
            # write second readout channel and metadata
            datastore.save_local_corrected_image(
                np.squeeze(raw_image[2,:]).astype(np.uint16),
                tile=tile_idx,
                psf_idx=2,
                gain_correction=True,
                hotpixel_correction=True,
                shading_correction=False,
                bit=int(experiment_order[round_idx,2])-1,
            )
            datastore.save_local_wavelengths_um(
                (ex_wavelengths_um[2], em_wavelengths_um[2]),
                tile=tile_idx,
                bit=int(experiment_order[round_idx,2])-1,
            )

    datastore_state = datastore.datastore_state
    datastore_state.update({"Corrected": True})
    datastore.datastore_state = datastore_state
    
if __name__ == "__main__":
    root_path = Path(r"/mnt/data/qi2lab/20241030_OB_22bit_MERFISH")
    convert_data(root_path)