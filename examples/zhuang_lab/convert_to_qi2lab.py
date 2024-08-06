"""Convert Zhuang MERFISH MOP data to qi2labdatastore.

Data found here: https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/

Download the "additional_files", "mouse1_sample1_raw", and 
"dataset_metadata.xslx" folders.

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
from natsort import natsorted

# root data folder
root_path = Path(r"/mnt/data/zhuang/mop/mouse_sample1_raw/")

# codebook
codebook_path = root_path / Path("additional_files") / Path("codebook.csv")
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
wavelengths_um = np.zeros((3, 2))
wavelengths_um[0, :] = [0.488, 0.520]
wavelengths_um[1, :] = [0.650, 0.690]
wavelengths_um[2, :] = [0.750, 0.790]

# voxel size from metadata and paper
voxel_zyx_um = [1.5, 0.108, 0.108]

# NA and RI from metadata and paper
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
datastore_path = Path(r"/mnt/data/zhuang/mop/mouse_sample1_raw/processed_v3")

# setup global datastore properties
datastore = qi2labDataStore(datastore_path)
datastore.num_rounds = 19
datastore.codebook = codebook
datastore.channels_in_data = ["alexa488", "cy5", "alexa750"]
datastore.experiment_order = experiment_order
datastore.num_tiles = num_tiles
datastore.microscope_type = "2D"
datastore.camera_model = "zhuang"
datastore.tile_overlap = 0.2
datastore.e_per_ADU = e_per_ADU  # unknown camera. don't convert to electrons
datastore.na = na
datastore.ri = ri
datastore.binning = 1
datastore.noise_map = (
    np.zeros((2048, 2048), dtype=np.float32)
)  # unknown camera. set noise / offset to zero.
datastore._shading_maps = np.ones(
    (3, 2048, 2048), dtype=np.float32
)  # unknown flatfield. set shading value to one.
datastore.channel_psfs = psfs
datastore.voxel_size_zyx_um = voxel_zyx_um

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
    psf_idx = 0
    for round_idx, round_id in enumerate(tqdm(datastore.round_ids,desc="round",leave=False)):
        datastore.save_local_corrected_image(
            np.squeeze(raw_image[38,:]),
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
    psf_idx = 1
    for bit_idx, bit_id in enumerate(tqdm(datastore.bit_ids,desc="bit",leave=False)):
        datastore.save_local_corrected_image(
            np.squeeze(raw_image[bit_idx, :]),
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
