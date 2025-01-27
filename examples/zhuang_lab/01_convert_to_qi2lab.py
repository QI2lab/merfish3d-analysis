"""Convert Zhuang MERFISH MOP data to qi2labdatastore.

Data found here: https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/

Download the "additional_files", "mouse1_sample1_raw", and 
"dataset_metadata.xslx" folders.

Shepherd 2025/01 - rework script to accept parameters
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
import numpy as np
import pandas as pd
from psfmodels import make_psf
from tifffile import imread
from tqdm import tqdm
from natsort import natsorted
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
    
    # codebook
    codebook = pd.read_csv(codebook_path)
    codebook.drop(columns=["id"], inplace=True)
    codebook.rename(columns={"name": "gene_id"}, inplace=True)
    # Identify "RS" columns and rename them to "bitXX"
    rs_columns = [col for col in codebook.columns if col.startswith("RS")]
    bit_mapping = {rs_columns[i]: f"bit{str(i+1).zfill(2)}" for i in range(len(rs_columns))}
    codebook.rename(columns=bit_mapping, inplace=True)

    # experimental order. 19 rounds with two readouts per round. The 20th round is polyDT and DAPI.
    # The actual experiment is more complicated, but the BIL dataset has already parsed the data.
    experiment_order = np.zeros((19, 3))
    for i in range(19):
        experiment_order[i, :] = [(i + 1), ((i + 1) * 2) - 1, (i + 1) * 2]

    # wavelengths from metadata and paper
    # https://www.nature.com/articles/s41586-021-03705-x
    wavelengths_um = np.zeros((3, 2))
    wavelengths_um[0, :] = [0.488, 0.520]
    wavelengths_um[1, :] = [0.650, 0.690]
    wavelengths_um[2, :] = [0.750, 0.790]

    # voxel size from metadata and paper
    # https://www.nature.com/articles/s41586-021-03705-x
    voxel_zyx_um = [1.5, 0.108, 0.108]

    # NA and RI from metadata and paper
    # https://www.nature.com/articles/s41586-021-03705-x
    na = 1.45
    ri = 1.51

    # gain and offset based on camera model (orca flash v3)
    # https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/static/sys/en/manual/C13440-20CU_IM_En.pdf
    e_per_ADU = .46 # from hamamatsu manual
    offset = 100. # from hamamats manual

    # stage positions from metadata
    stage_position_path = (
        root_path
        / Path("additional_files")
        / Path("fov_positions")
        / Path("mouse1_sample1.txt")
    )
    stage_position_df = pd.read_csv(stage_position_path,header=None)
    stage_positions = stage_position_df.values

    # num tiles back on number of stage positions
    num_tiles = stage_positions.shape[0]

    # generate 2D PSFs for each channel from metadata
    psfs = []
    for psf_idx in range(3):
        psf = make_psf(z=1, nx=51, dxy=voxel_zyx_um[1], NA=na, ni=ri)
        psf = psf / np.sum(psf, axis=(0, 1))
        psfs.append(psf)
    psfs = np.asarray(psfs, dtype=np.float32)

    # initialize datastore
    datastore_path = root_path.parents[1] / Path("qi2labdatastore")

    # setup global datastore properties
    datastore = qi2labDataStore(datastore_path)
    datastore.num_rounds = 19
    datastore.codebook = codebook
    datastore.channels_in_data = channel_names
    datastore.experiment_order = experiment_order
    datastore.num_tiles = num_tiles
    datastore.microscope_type = "2D"
    datastore.camera_model = "zhuang_orcav3"
    datastore.tile_overlap = 0.2
    datastore.e_per_ADU = e_per_ADU 
    datastore.na = na
    datastore.ri = ri
    datastore.binning = 1
    datastore.noise_map = offset * (
        np.ones((2048, 2048), dtype=np.float32)
    )
    datastore._shading_maps = np.ones(
        (3, 2048, 2048), dtype=np.float32
    )  # unknown flatfield. set shading value to one.
    datastore.channel_psfs = psfs
    datastore.voxel_size_zyx_um = voxel_zyx_um
    datastore.baysor_path = baysor_binary_path
    datastore.baysor_options = baysor_options_path
    datastore.julia_threads = julia_threads

    # Update datastore state to note that calibrations are doen
    datastore_state = datastore.datastore_state
    datastore_state.update({"Calibrations": True})
    datastore.datastore_state = datastore_state

    # generate natural sorted list of raw data files
    raw_images_files_path = root_path / Path("mouse1_sample1_raw")
    raw_image_files = natsorted(list(raw_images_files_path.glob("*.tif")))

    for tile_idx, raw_image_file in enumerate(tqdm(raw_image_files,desc="tile")):
        # initialize datastore tile
        # this creates the directory structure and links fiducial rounds <-> readout bits
        datastore.initialize_tile(tile_idx)
        
        # load raw image
        # some Zhuang tif files appear to be corrupted when downloaded with wget, 
        # so we catch those errors and notify user.
        try:
            raw_image = imread(raw_image_file).astype(np.uint16)
            good_shape = raw_image.shape
        except Exception:
            print("Error reading: " + raw_image_file  + "; Please re-download")
            raw_image = np.zeros((good_shape),dtype=np.uint16)
                
        # Correct for gain and offset
        raw_image = (raw_image).astype(np.float32) - offset
        raw_image[raw_image<0.] = 0.
        raw_image = (raw_image * e_per_ADU).astype(np.uint16)
        
        # write fidicual data first.
        # Write the same polyDT for each round, as the data is already locally registered.
        # The metadata tells us polyDT is the 39th entry
        # The Zhuang data is both transposed and flipped, which we fix when writing the data
        psf_idx = 0
        for round_idx, round_id in enumerate(tqdm(datastore.round_ids,desc="round",leave=False)):
            datastore.save_local_corrected_image(
                np.squeeze(np.swapaxes(raw_image[38,:],1,2)),
                tile=tile_idx,
                psf_idx=psf_idx,
                gain_correction=True,
                hotpixel_correction=False,
                shading_correction=False,
                round=round_id,
            )
            datastore.save_local_stage_position_zyx_um(
                stage_positions[tile_idx, :], tile=tile_idx, round=round_id
            )
            datastore.save_local_wavelengths_um(
                (wavelengths_um[psf_idx, 0],wavelengths_um[psf_idx, 1]), 
                tile=tile_idx, 
                round=round_id
            )

        # write all readouts
        # The bits go in order of the codebook
        # The Zhuang data is both transposed and flipped, which we fix when writing the data
        psf_idx = 1
        for bit_idx, bit_id in enumerate(tqdm(datastore.bit_ids,desc="bit",leave=False)):
            datastore.save_local_corrected_image(
                np.squeeze(np.swapaxes(raw_image[bit_idx, :],1,2)),
                tile=tile_idx,
                psf_idx=psf_idx,
                gain_correction=True,
                hotpixel_correction=False,
                shading_correction=False,
                bit=bit_id,
            )
            datastore.save_local_wavelengths_um(
                (wavelengths_um[psf_idx, 0], wavelengths_um[psf_idx, 1]),
                tile=tile_idx,
                bit=bit_idx,
            )
            if psf_idx == 2:
                psf_idx = 1
            else:
                psf_idx = 2

    # update datastore state that "corrected_data" is complete
    datastore_state = datastore.datastore_state
    datastore_state.update({"Corrected": True})
    datastore.datastore_state = datastore_state

if __name__ == "__main__":
    root_path = Path(r"/mnt/data/zhuang/mop/mouse_sample1_raw/")
    baysor_binary_path = Path(
        r"/home/qi2lab/Documents/github/Baysor/bin/baysor/bin/./baysor"
    )
    baysor_options_path = Path(
        r"/home/qi2lab/Documents/github/merfish3d-analysis/examples/human_olfactorybulb/qi2lab_humanOB.toml"
    )
    julia_threads = 20

    hot_pixel_image_path = None

    convert_data(
        root_path=root_path,
        baysor_binary_path=baysor_binary_path,
        baysor_options_path=baysor_options_path,
        julia_threads=julia_threads,
        channel_names=["alexa488", "cy5", "alexa750"],
        hot_pixel_image_path=hot_pixel_image_path,
        codebook_path = root_path / Path("additional_files") / Path("codebook.csv")
    )