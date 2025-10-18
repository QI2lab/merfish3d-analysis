"""
Convert synthetic statphysbio MERFISH data to qi2labdatastore.

This is an example on how to convert a synthetic experiment to the datastore
object that the qi2lab "merfish3d-analysis" package uses.

Required user parameters for system dependent variables are at end of script.

Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2025/08 - update for v0.7
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

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
import numpy as np
import pandas as pd
from psfmodels import make_psf
from tifffile import imread
from tqdm.auto import tqdm
from merfish3danalysis.utils.dataio import read_metadatafile
from itertools import compress
import typer

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def convert_data(
    root_path: Path,
    baysor_binary_path: str = r"/path/to/baysor",
    baysor_options_path: str = r"/path/to/baysor_options.toml",
    julia_threads: int = 20,
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
    num_z  = metadata["num_z"]

    camera = "synthetic"
    e_per_ADU = metadata["gain"]
    offset = metadata["offset"]
    binning = 1
    channels_active = [
        metadata["blue_active"],
        metadata["yellow_active"],
        metadata["red_active"],
    ]
 
    channel_order_bool = metadata["channels_reversed"]
    if channel_order_bool:
        channel_order = "reversed"
    else:
        channel_order = "forward"

    voxel_size_zyx_um = [metadata["z_step_um"], metadata["yx_pixel_um"], metadata["yx_pixel_um"]]
    na = metadata["na"]
    ri = metadata["ri"]
    
    ex_wavelengths_um = [0.488, 0.561, 0.635]  # selected by channel IDs
    em_wavelengths_um = [0.520, 0.580, 0.670]  # selected by channel IDs
    channel_idxs = list(range(num_ch))
    channels_in_data = list(compress(channel_idxs, channels_active))

    if hot_pixel_image_path is None:
        noise_map = None
    else:
        noise_map = imread(hot_pixel_image_path)
    
    affine_zyx_px = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ],dtype=np.float32)

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
        if voxel_size_zyx_um[0] < 0.5:
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
    datastore.voxel_size_zyx_um = voxel_size_zyx_um



    # generate PSFs
    # --------------
    channel_psfs = []
    for channel_id in channels_in_data:
        if datastore.microscope_type == "3D":
            psf = make_psf(
                z=num_z,
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
        else:
            psf = make_psf(
                z=1,
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
    datastore.channel_psfs = channel_psfs

    # Update datastore state to note that calibrations are done
    datastore_state = datastore.datastore_state
    datastore_state.update({"Calibrations": True})
    datastore.datastore_state = datastore_state

    # Loop over data and create datastore.
    for round_idx in tqdm(range(num_rounds), desc="rounds"):
        # Get all stage positions for this round
        position_list = []
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
            stage_z = np.round(float(stage_positions["stage_y"]), 2)
            temp = [stage_z,stage_y,stage_x]
            position_list.append(np.asarray(temp))
        position_list = np.asarray(position_list)
        
        for tile_idx in tqdm(range(num_tiles), desc="tile", leave=False):
            # initialize datastore tile
            # this creates the directory structure and links fiducial rounds <-> readout bits
            if round_idx == 0:
                datastore.initialize_tile(tile_idx)

            # load raw image
            image_path = (
                root_path
                / Path(root_name+ "_r"+ str(round_idx + 1).zfill(4)+ "_tile"+ str(tile_idx).zfill(4)+ "_1")
                / Path("data_r"+str(round_idx+1).zfill(4)+"_tile"+str(tile_idx).zfill(4)+".tif")
            )

            # load raw data and make sure it is the right shape. If not, write
            # zeros for this round/stage position.
            raw_image = imread(image_path)
            if tile_idx == 0 and round_idx == 0:
                correct_shape = raw_image.shape
            else:
                if raw_image is None or raw_image.shape != correct_shape:
                    if raw_image.shape[0] < correct_shape[0]:
                        print("\nround=" + str(round_idx + 1) + "; tile=" + str(tile_idx + 1))
                        print("Found shape: " + str(raw_image.shape))
                        print("Correct shape: " + str(correct_shape))
                        print("Replacing data with zeros.\n")
                        raw_image = np.zeros(correct_shape, dtype=np.uint16)
                    else:                    
                        # print("\nround=" + str(round_idx + 1) + "; tile=" + str(tile_idx + 1))
                        # print("Found shape: " + str(raw_image.shape))
                        size_to_trim = raw_image.shape[1] - correct_shape[1]
                        raw_image = raw_image[:,size_to_trim:,:].copy()
                        # print("Correct shape: " + str(correct_shape))
                        # print("Corrected to shape: " + str(raw_image.shape) + "\n")

            # Correct if channels were acquired in reverse order (red->purple)
            if channel_order == "reversed":
                raw_image = np.flip(raw_image, axis=0)

            # Correct for known camera gain and offset
            raw_image = (raw_image.astype(np.float32) - offset) * e_per_ADU
            raw_image[raw_image < 0.0] = 0.0
            raw_image = raw_image.astype(np.uint16)
            gain_corrected = True
            hot_pixel_corrected = False

            # load stage position
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
  
    datastore.noise_map = np.zeros((3, correct_shape[1], correct_shape[2]), dtype=np.float32)  
    datastore._shading_maps = np.ones((3, correct_shape[1], correct_shape[2]), dtype=np.float32)
    datastore_state = datastore.datastore_state
    datastore_state.update({"Corrected": True})
    datastore.datastore_state = datastore_state

def main():
    app()

if __name__ == "__main__":
    main()