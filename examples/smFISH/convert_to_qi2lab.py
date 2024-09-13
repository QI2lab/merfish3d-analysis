"""
Convert raw WF iterative RNA-FISH data to qi2labdatastore.

Data found here: TO BE DEPOSISTED

Download the "rawdata" folder.

Change the path on line 23 to where the data is downloaded.

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
from itertools import compress


def convert_data():
    # root data folder
    root_path = Path(r"/mnt/data/qi2lab/20240823_OB_22bit_2")

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
    channel_order = "reversed"
    voxel_size_zyx_um = [0.310, 0.098, 0.098]
    na = 1.35
    ri = 1.51
    ex_wavelengths_um = [0.488, 0.561, 0.635]
    em_wavelengths_um = [0.520, 0.580, 0.670]
    channel_idxs = [0, 1, 2]
    channels_in_data = list(compress(channel_idxs, channels_active))

    # camera gain and offset
    # from Photometrics manual
    # ----------------------
    e_per_ADU = 1.0
    offset = 100.0
    noise_map = None

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
    datastore.channels_in_data = ["alexa488", "atto565", "alxa647"]
    datastore.experiment_order = experiment_order
    datastore.num_tiles = num_tiles
    datastore.microscope_type = "3D"
    datastore.camera_model = "bsi"
    datastore.tile_overlap = 0.2
    datastore.e_per_ADU = e_per_ADU
    datastore.na = na
    datastore.ri = ri
    datastore.binning = 1
    datastore.noise_map = noise_map
    datastore._shading_maps = np.ones((3, 2048, 2048), dtype=np.float32)
    datastore.channel_psfs = channel_psfs
    datastore.voxel_size_zyx_um = voxel_size_zyx_um

    # Update datastore state to note that calibrations are doen
    datastore_state = datastore.datastore_state
    datastore_state.update({"Calibrations": True})
    datastore.datastore_state = datastore_state

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
            if channel_order == "reversed":
                raw_image = np.flip(raw_image,axis=0)

            # Correct for gain and offset
            raw_image = (raw_image).astype(np.float32) - offset
            raw_image[raw_image < 0.0] = 0.0
            raw_image = (raw_image * e_per_ADU).astype(np.uint16)
            
            # load stage position
            stage_position_path = (
                root_path
                / Path(root_name + "_r"+str(round_idx+1).zfill(4)+"_tile"+str(tile_idx).zfill(4)+"_stage_positions.csv")
            )
            df_stage_positions = read_metadatafile(stage_position_path)
            stage_x = np.round(float(df_stage_positions['stage_x']),2)
            stage_y = np.round(float(df_stage_positions['stage_y']),2)
            stage_z = np.round(float(df_stage_positions['stage_z']),2)
            stage_pos_zyx_um = np.asarray([stage_z,stage_y,stage_x],dtype=np.float32)

            # write fidicual data and metadata
            datastore.save_local_corrected_image(
                np.squeeze(raw_image[0, :]).astype(np.uint16),
                tile=tile_idx,
                psf_idx=0,
                gain_correction=True,
                hotpixel_correction=False,
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
                hotpixel_correction=False,
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
                np.squeeze(raw_image[2 :]),
                tile=tile_idx,
                psf_idx=2,
                gain_correction=True,
                hotpixel_correction=False,
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
    convert_data()