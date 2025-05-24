"""
Convert raw qi2lab OPM MERFISH data to qi2labdatastore.

This is an example on how to convert a qi2lab OPM experiment to the datastore
object that the qi2lab "merfish3d-analysis" package uses. Most of the
parameters are automatically extracted from the metadata written by qi2lab
microscopes.

Required user parameters for system dependent variables are at end of script.

Shepherd 2024/05 - Adapt for qi2lab OPM data
"""

import multiprocessing as mp
mp.set_start_method('forkserver', force=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

from merfish3danalysis.qi2labDataStore import qi2labDataStore
import tensorstore as ts
from opm_processing.imageprocessing.opmpsf import generate_skewed_psf
from opm_processing.imageprocessing.rlgc import chunked_rlgc
from opm_processing.imageprocessing.opmtools import orthogonal_deskew, deskew_shape_estimator
import json
from pathlib import Path
import numpy as np
import pandas as pd
from psfmodels import make_psf
from tifffile import imread
from tqdm import tqdm
from merfish3danalysis.utils.dataio import read_metadatafile
from merfish3danalysis.utils.imageprocessing import replace_hot_pixels
from itertools import compress
from typing import Optional

def convert_data(
    root_path: Path,
    baysor_binary_path: Path,
    baysor_options_path: Path,
    julia_threads: int,
    channel_names: Optional[list[str]] = ["alexa488", "atto565", "alexa647"],
    hot_pixel_image_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    codebook_path: Optional[Path] = None,
    bit_order_path: Optional[Path] = None,
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

    # Deal with camera vs stage orientation for stage positions.
    # This is required because we want all of the data in global world
    # coordinates, but the camera and software may not match the stage's
    # orientation or motion direction.
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

    # Precalculate number of tiles based on OPM scan size
    padded_y_shape = pad_to_deskewed_x(deskewed_shape)
    fake_deskewed = np.empty(
        (
            deskewed_shape.shape[0], 
            padded_y_shape,
            deskewed_shape.shape[2]
        ),
        dtype=np.uint16
    )
    tile_size = (deskewed_shape.shape[0], deskewed_shape.shape[2],deskewed_shape.shape[2])
    overlap = (0,tile_shape[2] * overlap, 0)
    

    # Loop over data and create datastore.
    for round_idx in tqdm(range(num_rounds), desc="rounds"):
        for pos_idx in tqdm(range(num_pos), desc="position", leave=False):
            # initialize datastore tile
            # this creates the directory structure and links fiducial rounds <-> readout bits
            if round_idx == 0:
                datastore.initialize_tile(tile_idx)

            # load raw image


            # load raw data and make sure it is the right shape. If not, write
            # zeros for this round/stage position.

            # Correct for known camera gain and offset

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

            # correct for stage direction reversed wrt to global coordinates

            
            stage_pos_zyx_um = np.asarray(
                [stage_z, corrected_y, corrected_x], dtype=np.float32
            )

            # process fidicual data (ch_idx = 0)
            # deconvolve oblique fidicual data
            # ------------------------


            # deskew oblique fidicual data
            # ------------------------

            # write deconvolved and deskewed fidicual data and metadata
            # ------------------------
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
                stage_pos_zyx_um, tile=tile_idx, round=round_idx
            )

            datastore.save_local_wavelengths_um(
                (ex_wavelengths_um[0], em_wavelengths_um[0]),
                tile=tile_idx,
                round=round_idx,
            )

            # process fidicual data (ch_idx = 0)
            # deconvolve oblique fidicual data
            # ------------------------


            # deskew oblique fidicual data
            # ------------------------

            # write deconvolved and deskewed fidicual data and metadata
            # ------------------------
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

if __name__ == "__main__":
    root_path = Path(r"/mnt/data2/bioprotean/20241206_Bartelle24hrcryo_sample2")
    baysor_binary_path = Path(
        r"/home/qi2lab/Documents/github/Baysor/bin/baysor/bin/./baysor"
    )
    baysor_options_path = Path(
        r"/home/qi2lab/Documents/github/merfish3d-analysis/examples/human_olfactorybulb/qi2lab_humanOB.toml"
    )
    julia_threads = 20

    hot_pixel_image_path = None
    #hot_pixel_image_path = Path(r"/mnt/data/qi2lab/20240317_OB_MERFISH_7/flir_hot_pixel_image.tif")

    convert_data(
        root_path=root_path,
        baysor_binary_path=baysor_binary_path,
        baysor_options_path=baysor_options_path,
        julia_threads=julia_threads,
        hot_pixel_image_path=hot_pixel_image_path
    )