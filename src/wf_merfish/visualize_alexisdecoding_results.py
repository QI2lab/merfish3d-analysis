import numpy as np
from pathlib import Path
import pandas as pd
import napari
import re
from tifffile import imread
import zarr
from magicgui import magicgui

data_dir_path = Path('/mnt/opm3/20240214_MouseBrain_UA_NewRO_RK/processed_v2')
polyDT_dir_path = data_dir_path / Path('polyDT')
readout_dir_path = data_dir_path / Path('readouts')
calibration_dir_path = data_dir_path / Path('calibrations.zarr')
calibration_zarr = zarr.open(calibration_dir_path)
df_codebook = pd.DataFrame(calibration_zarr.attrs['codebook'])
print(df_codebook.head())
localization_dir_path = data_dir_path / Path('localizations')
tile_ids = sorted([entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()],
                  key=lambda x: int(x.split('tile')[1].split('.zarr')[0]))
tile_dir_path = readout_dir_path / Path(tile_ids[0])
bit_ids = sorted([entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()],
                 key=lambda x: int(x.split('bit')[1].split('.zarr')[0]))

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


file_id = 'localization_candidates_localization_tile_coords.parquet'
scale = [1,1,1]


physical_spacing=[.31,.088,.088]
df_candidates = []

for bit_id in bit_ids:
    registered_localization_tile_bit_path = localization_dir_path / Path(tile_ids[0]) / Path(bit_id).stem / Path(file_id)
    df_localization = pd.read_parquet(registered_localization_tile_bit_path)
    
    s = str(Path(bit_id).stem)
    numbers = re.findall(r'\d+', s)
    if numbers:  # Check if the list is not empty
        extracted_bit = int(numbers[0])
        
    
    df_localization['bit'] = extracted_bit - 1
    
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
    
    df_candidates.append(df_within_mask)

df_all_candidates= pd.concat(df_candidates, ignore_index=True)

decoded_tile_path = localization_dir_path / Path(tile_ids[0]) / Path('decoded_nomask.parquet')
df_decoded = pd.read_parquet(decoded_tile_path)
df_decoded['pixel_z'] = df_decoded['z'] / physical_spacing[0]
df_decoded['pixel_y'] = df_decoded['y'] / physical_spacing[1]
df_decoded['pixel_x'] = df_decoded['x'] / physical_spacing[2]

# Since pixel indices must be integers, round and convert to int
df_decoded['pixel_z'] = df_decoded['pixel_z'].round().astype(int)
df_decoded['pixel_y'] = df_decoded['pixel_y'].round().astype(int)
df_decoded['pixel_x'] = df_decoded['pixel_x'].round().astype(int)

df_decoded_no_blanks = df_decoded[~df_decoded['species'].str.startswith('Blank')].copy()

points = df_decoded_no_blanks[['pixel_z', 'pixel_y', 'pixel_x']].to_numpy()

@magicgui(call_button="Display Points", species={"choices": df_codebook[0].tolist()})
def display_points(species: str):
    viewer.layers.clear()  # Clear existing layers
    
    # Display 'decoded' points
    decoded_points = df_decoded[df_decoded['species'] == species][['pixel_z', 'pixel_y', 'pixel_x']].values
    viewer.add_points(decoded_points, name="decoded", size=5, face_color='yellow', edge_color='yellow', symbol='ring')
    
    # Find bits to display based on df_codebook entry
    bits = df_codebook.loc[df_codebook[0] == species, range(1, 17)].values.flatten()
    
    bits_indices = [index for index, value in enumerate(bits) if value == 1]
    print(bits_indices)
    
    colors = ['red', 'blue', 'green', 'magenta']
    
    # Display points for each bit
    for i, bit in enumerate(bits_indices):
        bit_points = df_all_candidates[df_all_candidates['bit'] == bit][['pixel_z', 'pixel_y', 'pixel_x']].values
        viewer.add_points(bit_points, name=f"Bit {bit}", size=5, face_color=colors[i % len(colors)], edge_color=colors[i % len(colors)])
        
viewer = napari.Viewer()
viewer.window.add_dock_widget(display_points, area='right')

# Start the Napari GUI event loop
napari.run()