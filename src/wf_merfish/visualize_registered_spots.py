import numpy as np
from pathlib import Path
import pandas as pd
import napari
import re
from tifffile import imread

data_dir_path = Path('/mnt/opm3/20240124_OB_Full_MERFISH_UA_3_allrds/processed_v2')
polyDT_dir_path = data_dir_path / Path('polyDT')
readout_dir_path = data_dir_path / Path('readouts')
localization_dir_path = data_dir_path / Path('localizations')
tile_ids = [entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()]
tile_dir_path = readout_dir_path / Path(tile_ids[0])
bit_ids = [entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()]

try:
    mask_path = polyDT_dir_path / Path(tile_ids[0]) / Path("round000_mask.tiff")
    binary_mask = imread(mask_path)
    binary_mask[binary_mask>0]=1
except:
    binary_mask = None
        
color_names = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'purple',
    'cyan',
    'magenta',
    'lime',
    'pink',
    'teal',
    'lavender',
    'brown',
    'beige',
    'maroon',
    'navy'
]


file_id = 'candidates_registered_tile_coords.parquet'
scale = [1,1,1]
viewer = napari.Viewer()

physical_spacing=[.31,.088,.088]

for bit_id in bit_ids[2:]:
    registered_localization_tile_bit_path = localization_dir_path / Path(tile_ids[0]) / Path(bit_id).stem / Path(file_id)
    df_localization = pd.read_parquet(registered_localization_tile_bit_path)
    
    s = str(Path(bit_id).stem)
    numbers = re.findall(r'\d+', s)
    if numbers:  # Check if the list is not empty
        extracted_bit = int(numbers[0])
        
    # Convert physical coordinates to pixel coordinates
    df_localization['pixel_z'] = df_localization['z'] / physical_spacing[0]
    df_localization['pixel_y'] = df_localization['y'] / physical_spacing[1]
    df_localization['pixel_x'] = df_localization['x'] / physical_spacing[2]

    # Since pixel indices must be integers, round and convert to int
    df_localization['pixel_z'] = df_localization['pixel_z'].round().astype(int)
    df_localization['pixel_y'] = df_localization['pixel_y'].round().astype(int)
    df_localization['pixel_x'] = df_localization['pixel_x'].round().astype(int)
 
    if binary_mask is not None:
    # Ensure pixel coordinates are integers and within the mask bounds
        df_localization['pixel_z'] = df_localization['pixel_z'].astype(int).clip(0, binary_mask.shape[0] - 1)
        df_localization['pixel_y'] = df_localization['pixel_y'].astype(int).clip(0, binary_mask.shape[1] - 1)
        df_localization['pixel_x'] = df_localization['pixel_x'].astype(int).clip(0, binary_mask.shape[2] - 1)

        # Use vectorized indexing to check mask values for all points
        mask_values = binary_mask[df_localization['pixel_z'], df_localization['pixel_y'], df_localization['pixel_x']]

        # Filter DataFrame based on mask values
        df_within_mask = df_localization[mask_values == 1]
    else:
        df_within_mask = df_localization.copy()
        
    del df_localization

   
    # Assuming df is your DataFrame
    points = df_within_mask[['z', 'y', 'x']].to_numpy()
    
    # Add the points layer
    viewer.add_points(points, 
                      face_color=color_names[extracted_bit-1], 
                      size=.5, 
                      scale=scale)
    
    del df_localization, points

# Start the Napari GUI event loop
napari.run()