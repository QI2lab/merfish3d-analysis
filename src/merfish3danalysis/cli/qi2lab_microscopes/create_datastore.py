"""
Convert raw qi2lab WF MERFISH data to qi2labdatastore.

This is an example on how to convert a qi2lab experiment to the datastore
object that the qi2lab "merfish3d-analysis" package uses. Most of the
parameters are automatically extracted from the metadata written by qi2lab
microscopes. For another microscope, you will need to write new code on how to
extract the correct parameters.

Shepherd 2025/10 - change to CLI.
Shepherd 2024/12 - added more NDTIFF metadata extraction for camera and binning.
Shepherd 2024/12 - refactor
Shepherd 2024/11 - rework script to accept parameters.
Shepherd 2024/08 - rework script to utilize qi2labdatastore object.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

# ensure JAXLIB uses local CUDA
import os
prefix = os.environ["CONDA_PREFIX"]
os.environ["XLA_PTXAS_PATH"] = f"{prefix}/bin/ptxas"
os.environ["XLA_NVLINK_PATH"] = f"{prefix}/bin/nvlink"
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={prefix}"

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
import numpy as np
import pandas as pd
from psfmodels import make_psf
from tifffile import imread
from tqdm import tqdm
from merfish3danalysis.utils.dataio import read_metadatafile
from merfish3danalysis.utils.imageprocessing import replace_hot_pixels, estimate_shading, no_op
from itertools import compress
import gc
import builtins

import typer

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def convert_data(
    root_path: Path,
    baysor_binary_path: Path = None,
    baysor_options_path: Path = None,
    julia_threads: int = 1,
    channel_names: list[str] = ["alexa488", "atto565", "alexa647"],
    hot_pixel_image_path: Path = None,
    output_path: Path = None,
    codebook_path: Path = None,
    bit_order_path: Path = None,
):
    """Convert qi2lab microscope data to qi2lab datastore.

    Parameters
    ----------
    root_path: Path
        path to dataset
    baysor_binary_path: Path
        path to baysor binary
    baysor_options_path: Path
        path to baysor options toml
    julia_threads: int
        number of threads to use for Julia
    channel_names: list[str], default ["alexa488", "atto565", "alexa647"]
        name of dye molecules used in ascending order of wavelength
    hot_pixel_image_path: Optional[Path], default None
        path to hot pixel map. Default of `None` will set it to all zeros.
    output_path: Optional[Path], default None
        path to output directory. Default of `None` and will be created
        within the root_path
    codebook_path: Optional[Path], default None
        path to codebook. Default of `None` assumes the file is in
        the root_path.
    bit_order_path: Optional[Path], default None
        path to bit order file. This file defines what bits are present in each
        imaging round, in channel order. Default of `None` assumes
        the file is in the root_path.
    """

    # load codebook
    # --------------
    if codebook_path is None:
        codebook = pd.read_csv(root_path / Path("codebook.csv"))
    else:
        codebook = pd.read_csv(codebook_path)

    # load experimental order
    # -----------------------
    if bit_order_path is None:
        df_experiment_order = pd.read_csv(root_path / Path("bit_order.csv"))
        experiment_order = df_experiment_order.values
    else:
        df_experiment_order = pd.read_csv(bit_order_path)
        experiment_order = df_experiment_order.values

    # load experiment metadata
    # ------------------------
    metadata_path = root_path / Path("scan_metadata.csv")
    metadata = read_metadatafile(metadata_path)
    root_name = metadata["root_name"]
    num_rounds = metadata["num_r"]
    num_tiles = metadata["num_xyz"]
    num_ch = metadata["num_ch"]

    from ndstorage import Dataset

    # load first tile to get experimental metadata
    dataset_path = root_path / Path(
        root_name + "_r" + str(1).zfill(4) + "_tile" + str(0).zfill(4) + "_1"
    )
    original_print = builtins.print
    builtins.print = no_op
    dataset = Dataset(str(dataset_path))
    builtins.print = original_print
    channel_to_test = dataset.get_image_coordinates_list()[0]["channel"]
    ndtiff_metadata = dataset.read_metadata(channel=channel_to_test, z=0)
    try:
        camera_id = ndtiff_metadata["Camera-CameraName"]
        camera_id_alt = None
    except KeyError:
        camera_id = None
        camera_id_alt = ndtiff_metadata["Core-Camera"]
    if camera_id == "C13440-20CU" or camera_id_alt == "C13440-20CU":
        camera = "orcav3"
        e_per_ADU = float(ndtiff_metadata["Camera-CONVERSION FACTOR COEFF"])
        offset = float(ndtiff_metadata["Camera-CONVERSION FACTOR OFFSET"])
    elif camera_id == "Blackfly S BFS-U3-200S6M" or camera_id_alt == "Blackfly S BFS-U3-200S6M":
        camera = "flir"
        e_per_ADU = 0.03  # this comes from separate calibration
        offset = 0.0  # this comes from separate calibration
    try:
        binning = metadata["binning"]
    except Exception:
        if camera == "orcav3":
            binning_str = ndtiff_metadata["Camera-Binning"]
            if binning_str == "1x1":
                binning = 1
            elif binning_str == "2x2":
                binning = 2
        elif camera == "flir":
            binning_str = ndtiff_metadata["Binning"]
            if binning_str == "1":
                binning = 1
            elif binning_str == "2":
                binning = 2
    channels_active = [
        metadata["blue_active"],
        metadata["yellow_active"],
        metadata["red_active"],
    ]
    # this entry was not contained in pre-v8 microscope csv, it was instead stored
    # in the imaging data itself. We added it to > v8 qi2lab-scope metadata csv to make the
    # access pattern easier.
    try:
        channel_order_bool = metadata["channels_reversed"]
        if channel_order_bool:
            channel_order = "reversed"
        else:
            channel_order = "forward"
    except KeyError:
        if (dataset.get_image_coordinates_list()[0]["channel"]) == "F-Blue":
            channel_order = "forward"
        else:
            channel_order = "reversed"

    # this entry was not contained in pre-v8 microscope csv, it was instead stored
    # in the imaging data itself. We added it to > v8 qi2lab-scope metadata csv to make the
    # access pattern easier.
    try:
        voxel_size_zyx_um = [metadata["z_step_um"], metadata["yx_pixel_um"]]
    except Exception:
        yx_pixel_um = np.round(float(ndtiff_metadata["PixelSizeUm"]), 3)
        next_ndtiff_metadata = dataset.read_metadata(channel=channel_to_test, z=1)
        z_pixel_um = np.round(
            np.abs(
                float(next_ndtiff_metadata["ZPosition_um_Intended"])
                - float(ndtiff_metadata["ZPosition_um_Intended"])
            ),
            3,
        )
        voxel_size_zyx_um = [z_pixel_um, yx_pixel_um, yx_pixel_um]

        del next_ndtiff_metadata

    # this entry was not contained in pre-v8 metadata csv, it was instead stored
    # in the imaging data itself. We added it to > v8 qi2lab-scope metadata csv to make the
    # access pattern easier.
    try:
        na = metadata["na"]
    except Exception:
        na = 1.35

    # this entry was not contained in pre-v8 microscope csv, it was instead stored
    # in the imaging data itself. We added it to > v8 qi2lab-scope metadata csv to make the
    # access pattern easier.
    try:
        ri = metadata["ri"]
    except Exception:
        ri = 1.51

    ex_wavelengths_um = [0.488, 0.561, 0.635]  # selected by channel IDs
    em_wavelengths_um = [0.520, 0.580, 0.670]  # selected by channel IDs
    channel_idxs = list(range(num_ch))
    channels_in_data = list(compress(channel_idxs, channels_active))

    # load camera specific stage vs camera vs computer orientation
    # parameters.
    #
    # these entries were not contained in pre-v8 microscope csv. There were
    # instead stored in the imaging data itself.
    #
    # We added it to > v8 qi2lab-scope metadata csv to make the access pattern easier.
    # The defaults are the "known" defaults for this camera configuration.

    if camera == "flir":
        if hot_pixel_image_path is None:
            noise_map = offset * np.ones((2048, 2048), dtype=np.uint16)
        else:
            noise_map = imread(hot_pixel_image_path)
    elif camera == "orcav3":
        if hot_pixel_image_path is None:
            noise_map = offset * np.ones((2048, 2048), dtype=np.uint16)
        else:
            noise_map = imread(hot_pixel_image_path)
            
    stage_affine_str = ndtiff_metadata["PixelSizeAffine"]
    stage_affine_values = np.asarray(list(map(float, stage_affine_str.split(';'))),dtype=np.float32)
    stage_affine_values = np.round(stage_affine_values / float(ndtiff_metadata["PixelSizeUm"]),2)
    affine_zyx_px = np.array([
        [1,0,0,0],
        [0,stage_affine_values[4],stage_affine_values[3],0],
        [0,stage_affine_values[1],stage_affine_values[0],0],
        [0,0,0,1]
    ],dtype=np.float32) 

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
        #psf = psf / np.sum(psf, axis=(0, 1, 2))
        channel_psfs.append(psf)
    channel_psfs = np.asarray(channel_psfs, dtype=np.float32)

    # initialize datastore
    if output_path is None:
        datastore_path = root_path / Path(r"qi2labdatastore")
        datastore = qi2labDataStore(datastore_path)
    else:
        datastore = qi2labDataStore(output_path)

    # required user parameters
    datastore.channels_in_data = channel_names
    datastore.baysor_path = baysor_binary_path
    datastore.baysor_options = baysor_options_path
    datastore.julia_threads = julia_threads

    # parameters from qi2lab microscope metadata
    datastore.num_rounds = num_rounds
    datastore.codebook = codebook
    datastore.experiment_order = experiment_order
    datastore.num_tiles = num_tiles
    try:
        datastore.microscope_type = metadata["experiment_type"]
    except Exception:
        if z_pixel_um < 0.5:
            datastore.microscope_type = "3D"
        else:
            datastore.microscope_type = "2D"
    datastore.camera_model = camera
    try:
        datastore.tile_overlap = metadata["tile_overlap"]
    except Exception:
        datastore.tile_overlap = 0.2
    datastore.e_per_ADU = e_per_ADU
    datastore.na = na
    datastore.ri = ri
    datastore.binning = binning
    datastore.noise_map = noise_map
    datastore._shading_maps = np.ones((3, 2048, 2048), dtype=np.float32)  # not used yet
    datastore.channel_psfs = channel_psfs
    datastore.voxel_size_zyx_um = voxel_size_zyx_um

    # Update datastore state to note that calibrations are done
    datastore_state = datastore.datastore_state
    datastore_state.update({"Calibrations": True})
    datastore.datastore_state = datastore_state

    # Loop over data and create datastore.
    for round_idx in tqdm(range(num_rounds), desc="rounds"):
        # Get all stage positions for this round
        position_list = []
        for tile_idx in range(num_tiles):
            dataset_path = root_path / Path(
                root_name + "_r" + str(round_idx+1).zfill(4) + "_tile" + str(tile_idx).zfill(4) + "_1"
            )
            builtins.print = no_op
            dataset = Dataset(str(dataset_path))
            builtins.print = original_print
            x_pos_um = np.round(float(dataset.read_metadata(channel=channel_to_test, z=0)["XPosition_um_Intended"]),2)
            y_pos_um = np.round(float(dataset.read_metadata(channel=channel_to_test, z=0)["YPosition_um_Intended"]),2)
            z_pos_um = np.round(float(dataset.read_metadata(channel=channel_to_test, z=0)["ZPosition_um_Intended"]),2)
            temp = [z_pos_um,y_pos_um,x_pos_um]
            position_list.append(np.asarray(temp))
            del dataset
        position_list = np.asarray(position_list)
        
        for tile_idx in tqdm(range(num_tiles), desc="tile", leave=False):
            # initialize datastore tile
            # this creates the directory structure and links fiducial rounds <-> readout bits
            if round_idx == 0:
                datastore.initialize_tile(tile_idx)

            # load raw image
            image_path = (
                root_path
                / Path(
                    root_name
                    + "_r"
                    + str(round_idx + 1).zfill(4)
                    + "_tile"
                    + str(tile_idx).zfill(4)
                    + "_1"
                )
                / Path(
                    root_name
                    + "_r"
                    + str(round_idx + 1).zfill(4)
                    + "_tile"
                    + str(tile_idx).zfill(4)
                    + "_NDTiffStack.tif"
                )
            )

            # load raw data and make sure it is the right shape. If not, write
            # zeros for this round/stage position.
            raw_image = imread(image_path)
            if camera == "orcav3":
                raw_image = np.swapaxes(raw_image, 0, 1)
                if tile_idx == 0 and round_idx == 0:
                    correct_shape = raw_image.shape
            elif camera == "flir":
                if tile_idx == 0 and round_idx == 0:
                    correct_shape = raw_image.shape
            if raw_image is None or raw_image.shape != correct_shape:
                if raw_image.shape[0] < correct_shape[0]:
                    print("\nround=" + str(round_idx + 1) + "; tile=" + str(tile_idx + 1))
                    print("Found shape: " + str(raw_image.shape))
                    print("Correct shape: " + str(correct_shape))
                    print("Replacing data with zeros.\n")
                    raw_image = np.zeros(correct_shape, dtype=np.uint16)
                else:                    
                    size_to_trim = raw_image.shape[1] - correct_shape[1]
                    raw_image = raw_image[:,size_to_trim:,:].copy()

            # Correct if channels were acquired in reverse order (red->purple)
            if channel_order == "reversed":
                raw_image = np.flip(raw_image, axis=0)

            # Correct for known camera gain and offset
            raw_image = (raw_image.astype(np.float32) - offset) * e_per_ADU
            raw_image[raw_image < 0.0] = 0.0
            raw_image = raw_image.astype(np.uint16)
            gain_corrected = True

            # Correct for known hot pixel map
            if camera == "flir":
                raw_image = replace_hot_pixels(noise_map, raw_image)
                raw_image = replace_hot_pixels(
                    np.max(raw_image, axis=0), raw_image, threshold=100
                )
                hot_pixel_corrected = True
            else:
                hot_pixel_corrected = False

            # load stage position
            if int(ndtiff_metadata["XYStage-TransposeMirrorX"]) == 1:
                corrected_y = np.max(position_list[:,2]) - position_list[tile_idx,2]
                corrected_x = np.max(position_list[:,1]) - position_list[tile_idx,1]
            elif int(ndtiff_metadata["XYStage-TransposeMirrorY"]) == 1:
                corrected_y = np.max(position_list[:,2]) - position_list[tile_idx,2]
                corrected_x = np.max(position_list[:,1]) - position_list[tile_idx,1]
            else:
                corrected_y = position_list[tile_idx,1]
                corrected_x = position_list[tile_idx,2]
            
            corrected_x = np.round(corrected_x,2)
            corrected_y = np.round(corrected_y,2)
            stage_z = np.round(position_list[tile_idx,0],2)
            
            stage_pos_zyx_um = np.asarray(
                [stage_z, corrected_y, corrected_x], dtype=np.float32
            )

            # write fidicual data (ch_idx = 0) and metadata
            datastore.save_local_corrected_image(
                np.squeeze(raw_image[0, :]).astype(np.uint16),
                tile=tile_idx,
                psf_idx=0,
                gain_correction=gain_corrected,
                hotpixel_correction=hot_pixel_corrected,
                shading_correction=False,
                round=round_idx,
            )

            datastore.save_local_stage_position_zyx_um(
                stage_pos_zyx_um, 
                affine_zyx_px,
                tile=tile_idx, 
                round=round_idx
            )

            datastore.save_local_wavelengths_um(
                (ex_wavelengths_um[0], em_wavelengths_um[0]),
                tile=tile_idx,
                round=round_idx,
            )

            # write first readout channel (ch_idx = 1) and metadata
            datastore.save_local_corrected_image(
                np.squeeze(raw_image[1, :]).astype(np.uint16),
                tile=tile_idx,
                psf_idx=1,
                gain_correction=gain_corrected,
                hotpixel_correction=hot_pixel_corrected,
                shading_correction=False,
                bit=int(experiment_order[round_idx, 1]) - 1,
            )
            datastore.save_local_wavelengths_um(
                (ex_wavelengths_um[1], em_wavelengths_um[1]),
                tile=tile_idx,
                bit=int(experiment_order[round_idx, 1]) - 1,
            )

            # write second readout channel (ch_idx = 2) and metadata
            datastore.save_local_corrected_image(
                np.squeeze(raw_image[2, :]).astype(np.uint16),
                tile=tile_idx,
                psf_idx=2,
                gain_correction=gain_corrected,
                hotpixel_correction=hot_pixel_corrected,
                shading_correction=False,
                bit=int(experiment_order[round_idx, 2]) - 1,
            )
            datastore.save_local_wavelengths_um(
                (ex_wavelengths_um[2], em_wavelengths_um[2]),
                tile=tile_idx,
                bit=int(experiment_order[round_idx, 2]) - 1,
            )
    
    datastore_state = datastore.datastore_state
    datastore_state.update({"Corrected": True})
    datastore.datastore_state = datastore_state
    del datastore
    gc.collect()
    
    # Calculate and apply flatfield corrections
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    if datastore.num_tiles > 100:
        n_flatfield_images = 100
    else:
        n_flatfield_images = datastore.num_tiles
    sample_indices = np.asarray(np.random.choice(datastore.num_tiles, size=n_flatfield_images, replace=False))
    data_camera_corrected = []

    # calculate fiducial correction
    for rand_tile_idx in tqdm(sample_indices,desc='flatfield data',leave=False):
        data_camera_corrected.append(
            datastore.load_local_corrected_image(
                tile=int(rand_tile_idx),
                round=0,
            )
        )    
    fidicual_illumination = estimate_shading(data_camera_corrected)
    del data_camera_corrected
    gc.collect()
    
    for round_idx in tqdm(range(datastore.num_rounds), desc="rounds"):     
        for tile_idx in tqdm(range(datastore.num_tiles), desc="tile", leave=False):
            data_camera_corrected = datastore.load_local_corrected_image(
                tile=tile_idx,
                round=round_idx,
                return_future=False)
            data_camera_corrected = (data_camera_corrected.astype(np.float32) / fidicual_illumination).clip(0,2**16-1).astype(np.uint16)
            datastore.save_local_corrected_image(
                data_camera_corrected,
                tile=tile_idx,
                psf_idx=0,
                gain_correction=True,
                hotpixel_correction=False,
                shading_correction=True,
                round=round_idx,
            )
    
    for bit_id in tqdm(datastore.bit_ids, desc="bit", leave=True):
        data_camera_corrected = []

        # calculate fiducial correction
        for rand_tile_idx in tqdm(sample_indices,desc='flatfield data',leave=False):
            data_camera_corrected.append(
                datastore.load_local_corrected_image(
                    tile=int(rand_tile_idx),
                    bit=bit_id,
                )
            )
        readout_illumimation = estimate_shading(data_camera_corrected)
        del data_camera_corrected
        gc.collect()
        for tile_idx in tqdm(range(datastore.num_tiles), desc="tile", leave=False):
            data_camera_corrected = datastore.load_local_corrected_image(
                tile=tile_idx,
                bit=bit_id,
                return_future=False)
            data_camera_corrected = (data_camera_corrected.astype(np.float32) / readout_illumimation).clip(0,2**16-1).astype(np.uint16)

            ex_wavelength_um, em_wavelength_um = datastore.load_local_wavelengths_um(
                tile=tile_idx,
                bit=bit_id
            )
            
            # TO DO: hacky fix. Need to come up with a better way.
            if ex_wavelength_um < 600:
                psf_idx = 1
            else:
                psf_idx = 2

            datastore.save_local_corrected_image(
                data_camera_corrected.astype(np.uint16),
                tile=tile_idx,
                psf_idx=psf_idx,
                gain_correction=True,
                hotpixel_correction=False,
                shading_correction=True,
                bit=bit_id
            )


    datastore_state = datastore.datastore_state
    datastore_state.update({"Corrected": True})
    datastore.datastore_state = datastore_state

def main():
    app()

if __name__ == "__main__":
    main()