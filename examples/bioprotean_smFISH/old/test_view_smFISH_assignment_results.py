import numpy as np
import pandas as pd
from pathlib import Path
import json
import napari

root_path = Path('/mnt/data/bartelle/20240423_ECL_24CryoA_2_PL025_restart_practice')
dataset_path = root_path / Path('processed_v2')

def load_microjson(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        outlines = {}
        
        for feature in data['features']:
            cell_id = feature['properties']['cell_id']
            coordinates = feature['geometry']['coordinates'][0]
            outlines[cell_id] = np.array(coordinates)
        return outlines
    
outlines_path = dataset_path / Path("segmentation") / Path("cellpose") / Path("cell_outlines.json")
cell_outlines = load_microjson(outlines_path)

decoded_spots_path = dataset_path / Path('decoded')

ufish_path = dataset_path / Path('ufish_localizations')
tile_ids = sorted([entry.name for entry in ufish_path.iterdir() if entry.is_dir()])
first_tile_path = ufish_path / tile_ids[0]
file_info = []
for bit_file in sorted(first_tile_path.glob('bit*.parquet')):
    full_file_name = bit_file.name
    part_before_dot = bit_file.stem
    file_info.append((full_file_name, part_before_dot))

viewer = napari.Viewer()

file_info = []
for bit_file in sorted(decoded_spots_path.glob('bit*.csv')):
   df_current_bit = pd.read_csv(decoded_spots_path / bit_file)
   df_selected_spots = df_current_bit[df_current_bit['cell_id']>=0].copy()
   pts = df_selected_spots[['global_y', 'global_x']].to_numpy()
   gene_name = df_current_bit['gene_id'][0]
   viewer.add_points(pts,name=gene_name,size=1)

viewer.add_shapes(list(cell_outlines.values()), shape_type='path', edge_color='coral')
napari.run()