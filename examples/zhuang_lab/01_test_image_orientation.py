from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imread
from tqdm import tqdm
from natsort import natsorted
import napari

viewer = napari.Viewer()

root_path = Path(r"/mnt/data/zhuang/mop/mouse_sample1_raw/")
offset = 100.
e_per_ADU = .46
voxel_zyx_um = [1.5, 0.108, 0.108]

# stage positions from metadata
stage_position_path = (
    root_path
    / Path("additional_files")
    / Path("fov_positions")
    / Path("mouse1_sample1.txt")
)
stage_position_df = pd.read_csv(stage_position_path,header=None)

# Because the data is transposed and flipped, we need to process the stage
# positions to match data
# stage_position_df[[1]] = stage_position_df[[1]].max() - stage_position_df[[1]]
stage_positions = stage_position_df.values

# num tiles back on number of stage positions
num_tiles = stage_positions.shape[0]

# generate natural sorted list of raw data files
raw_images_files_path = root_path / Path("mouse1_sample1_raw")
raw_image_files = natsorted(list(raw_images_files_path.glob("*.tif")))

for tile_idx, raw_image_file in enumerate(tqdm(raw_image_files,desc="tile")):
    
    # polyDT is slice 38
    # For determining the image orientation, we only need the max Z proejction
    raw_image = np.squeeze(np.max(imread(raw_image_file).astype(np.uint16)[38,:],axis=0))    
    
    # based on the metadata, it appears the x & y axes need to be swapped
    image = np.swapaxes(raw_image,0,1)
    stage_yx_um = stage_positions[tile_idx, :]
    
    viewer.add_image(image,scale=voxel_zyx_um[1:],translate=stage_yx_um)

napari.run()