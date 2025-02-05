"""
Convert raw qi2lab WF MERFISH data to qi2labdatastore.

This is an example on how to convert a qi2lab experiment to the datastore
object that the qi2lab "merfish3d-analysis" package uses. Most of the
parameters are automatically extracted from the metadata written by qi2lab
microscopes. For another microscope, you will need to write new code on how to
extract the correct parameters.

For statphysbio simulated data, we converted simulation into a one tile qi2lab
acquistion to re-use existing conversion code.

Required user parameters for system dependent variables are at end of script.

Shepherd 2024/12 - create for simulated data
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
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
    channel_names: list[str]
        name of dye molecules used in ascending order of wavelength
    hot_pixel_image_path: Optional[Path]
        path to hot pixel map. Default is None
    output_path: Optional[Path]
        path to output directory. Default is None and will be created
        within the root_path
    codebook_path: Optional[Path]
        path to codebook. Default is None and it assumed the file is in
        the root_path.
    bit_order_path: Optional[Path]
        path to bit order file. This file defines what bits are present in each
        imaging round, in channel order. Default is None and it assumed
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
    
    try:
        camera = metadata["camera"]
    except Exception:
        camera = None
        
    if not(camera == "simulated"):
        from ndstorage import Dataset

        # load first tile to get experimental metadata
        dataset_path = root_path / Path(
            root_name + "_r" + str(1).zfill(4) + "_tile" + str(0).zfill(4) + "_1"
        )
        dataset = Dataset(str(dataset_path))
        channel_to_test = dataset.get_image_coordinates_list()[0]["channel"]
        ndtiff_metadata = dataset.read_metadata(channel=channel_to_test, z=0)
        camera_id = ndtiff_metadata["Camera-CameraName"]
        if camera_id == "C13440-20CU":
            camera = "orcav3"
            e_per_ADU = float(ndtiff_metadata["Camera-CONVERSION FACTOR COEFF"])
            offset = float(ndtiff_metadata["Camera-CONVERSION FACTOR OFFSET"])
        else:
            camera = "flir"
            e_per_ADU = 0.03  # this comes from separate calibration
            offset = 0.0  # this comes from separate calibration
        try:
            binning = metadata["binning"]
        except Exception:
            binning_str = ndtiff_metadata["Camera-Binning"]
            if binning_str == "1x1":
                binning = 1
            elif binning_str == "2x2":
                binning = 2
    channels_active = [
        metadata["blue_active"],
        metadata["yellow_active"],
        metadata["red_active"],
    ]
    try:
        channel_order_bool = metadata["channels_reversed"]
        if channel_order_bool:
            channel_order = "reversed"
        else:
            channel_order = "forward"
    except KeyError:
        if not(camera=="simulated"):
            if (dataset.get_image_coordinates_list()[0]["channel"]) == "F-Blue":
                channel_order = "forward"
            else:
                channel_order = "reversed"

    # this entry was not contained in pre-v8 microscope csv, it was instead stored
    # in the imaging data itself. We added it to > v8 metadata csv to make the
    # access pattern easier.
    try:
        voxel_size_zyx_um = [metadata["z_step_um"], metadata["yx_pixel_um"]]
        z_pixel_um = voxel_size_zyx_um[0]
        yx_pixel_um = voxel_size_zyx_um[1]
    except Exception:
        if not(camera=="simulated"):
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

            del ndtiff_metadata, next_ndtiff_metadata, dataset

    # this entry was not contained in pre-v8 metadata csv, it was instead stored
    # in the imaging data itself. We added it to > v8 metadata csv to make the
    # access pattern easier.
    try:
        na = metadata["na"]
    except Exception:
        na = 1.35

    # this entry was not contained in pre-v8 microscope csv, it was instead stored
    # in the imaging data itself. We added it to > v8 metadata csv to make the
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
    # We added it to > v8 metadata csv to make the access pattern easier.
    # The defaults are the "known" defaults for this camera configuration.

    if camera == "flir":
        if hot_pixel_image_path is None:
            noise_map = offset * np.ones((2048, 2048), dtype=np.uint16)
        else:
            noise_map = imread(hot_pixel_image_path)
        try:
            stage_flipped_x = metadata["stage_flipped_x"]
        except Exception:
            stage_flipped_x = False
        try:
            stage_flipped_y = metadata["stage_flipped_y"]
        except Exception:
            stage_flipped_y = False
        try:
            image_rotated = metadata["image_rotated"]
        except Exception:
            image_rotated = False
        try:
            image_flipped_y = metadata["image_flipped_y"]
        except Exception:
            image_flipped_y = False
        try:
            image_flipped_x = metadata["image_flipped_x"]
        except Exception:
            image_flipped_x = False
    elif camera == "orcav3":
        if hot_pixel_image_path is None:
            noise_map = offset * np.ones((2048, 2048), dtype=np.uint16)
        else:
            noise_map = imread(hot_pixel_image_path)

        try:
            stage_flipped_x = metadata["stage_flipped_x"]
        except Exception:
            stage_flipped_x = True
        try:
            stage_flipped_y = metadata["stage_flipped_y"]
        except Exception:
            stage_flipped_y = True
        try:
            image_rotated = metadata["image_rotated"]
        except Exception:
            image_rotated = True
        try:
            image_flipped_y = metadata["image_flipped_y"]
        except Exception:
            image_flipped_y = True
        try:
            image_flipped_x = metadata["image_flipped_x"]
        except Exception:
            image_flipped_x = False
    elif camera == "simulated":
        camera = "simulated"
        e_per_ADU = 1. / float(metadata["gain"])
        offset = float(metadata["offset"])
        stage_flipped_x = bool(metadata["stage_flipped_x"])
        stage_flipped_y = bool(metadata["stage_flipped_y"])
        image_rotated = bool(metadata["image_rotated"])
        image_flipped_y = bool(metadata["image_flipped_y"])
        image_flipped_x = bool(metadata["image_flipped_x"])
        binning = int(metadata["binning"])
        noise_map = offset * np.ones((256, 256), dtype=np.uint16)

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

    # Loop over data and create datastore.
    for round_idx in tqdm(range(num_rounds), desc="rounds"):
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
            try:
                raw_image = imread(image_path)
            except Exception:
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
                        + ".tif"
                    )
                )
                raw_image = imread(image_path)
            raw_image = np.swapaxes(raw_image, 0, 1)
            if tile_idx == 0 and round_idx == 0:
                correct_shape = raw_image.shape
            if raw_image is None or raw_image.shape != correct_shape:
                print("\nround=" + str(round_idx + 1) + "; tile=" + str(tile_idx + 1))
                print("Found shape: " + str(raw_image.shape))
                print("Correct shape: " + str(correct_shape))
                print("Replacing data with zeros.\n")
                raw_image = np.zeros(correct_shape, dtype=np.uint16)

            # Correct if channels were acquired in reverse order (red->purple)
            if channel_order == "reversed":
                raw_image = np.flip(raw_image, axis=0)

            # Correct if camera is rotated wrt to stage
            if image_rotated:
                raw_image = np.rot90(raw_image, k=-1, axes=(3, 2))

            # Correct if camera is flipped in y wrt to stage
            if image_flipped_y:
                raw_image = np.flip(raw_image, axis=2)

            # Correct if camera is flipped in x wrt to stage
            if image_flipped_x:
                raw_image = np.flip(raw_image, axis=3)

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
            stage_position_path = root_path / Path(
                root_name
                + "_r"
                + str(round_idx + 1).zfill(4)
                + "_tile"
                + str(tile_idx).zfill(4)
                + "_stage_positions.csv"
            )
            df_stage_positions = read_metadatafile(stage_position_path)
            stage_x = np.round(float(df_stage_positions["stage_x"]), 2)
            stage_y = np.round(float(df_stage_positions["stage_y"]), 2)
            stage_z = np.round(float(df_stage_positions["stage_z"]), 2)

            # correct for stage direction reversed wrt to global coordinates
            if stage_flipped_x or stage_flipped_y:
                if stage_flipped_y:
                    corrected_y = max_y - stage_y
                else:
                    corrected_y = stage_y
                if stage_flipped_x:
                    corrected_x = max_x - stage_x
                else:
                    corrected_x = stage_x
            else:
                corrected_x = stage_x
                corrected_y = stage_y
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
                stage_pos_zyx_um, tile=tile_idx, round=round_idx
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

if __name__ == "__main__":
    root_path = Path(r"/mnt/opm3/20241218_statphysbio/sim_acquisition")
    baysor_binary_path = None
    baysor_options_path = None
    julia_threads = 20

    convert_data(
        root_path=root_path,
        baysor_binary_path=baysor_binary_path,
        baysor_options_path=baysor_options_path,
        julia_threads=julia_threads,
    )