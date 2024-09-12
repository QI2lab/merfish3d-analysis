import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from pathlib import Path
import zarr
from shapely.geometry import Polygon, Point
import json
from concurrent.futures import ProcessPoolExecutor
import rtree

# modify this line
root_path = Path('/mnt/data/bartelle/20240423_ECL_24CryoA_2_PL025_restart_practice')


# don't modify 
dataset_path = root_path / Path('processed_v2')
calibration_dir_path = dataset_path / Path("calibrations.zarr")
calibration_zarr = zarr.open(calibration_dir_path, mode='a')
df_codebook = pd.DataFrame(calibration_zarr.attrs['codebook'])
df_codebook.fillna(0, inplace=True)
codebook_matrix = df_codebook.iloc[:, 1:].to_numpy().astype(int)
gene_ids = df_codebook.iloc[:, 0].tolist()

ufish_path = dataset_path / Path('ufish_localizations')
tile_ids = sorted([entry.name for entry in ufish_path.iterdir() if entry.is_dir()])
first_tile_path = ufish_path / tile_ids[0]
file_info = []
for bit_file in sorted(first_tile_path.glob('bit*.parquet')):
    full_file_name = bit_file.name
    part_before_dot = bit_file.stem
    file_info.append((full_file_name, part_before_dot))

decoded_spots_path = dataset_path / Path('decoded')
decoded_spots_path.mkdir(exist_ok=True)

def warp_pixel(pixel_space_point: np.ndarray, spacing: np.ndarray, origin: np.ndarray, affine: np.ndarray) -> np.ndarray:
    physical_space_point = pixel_space_point * spacing + origin
    registered_space_point = (np.array(affine) @ np.array(list(physical_space_point) + [1]))[:-1]
    return registered_space_point

def load_microjson(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    outlines = {feature['properties']['cell_id']: np.array(feature['geometry']['coordinates'][0]) for feature in data['features']}
    return outlines

def assign_cells(df_spots):
    try:
        outlines_path = dataset_path / Path("segmentation") / Path("cellpose") / Path("cell_outlines.json")
        cell_outlines = load_microjson(outlines_path)
        has_cell_outlines = True
    except:
        has_cell_outlines = False
    
    if has_cell_outlines:
        outline_polygons = {cell_id: Polygon(outline) for cell_id, outline in cell_outlines.items()}
        rtree_index = rtree.index.Index()

        for cell_id, polygon in outline_polygons.items():
            rtree_index.insert(cell_id, polygon.bounds)

        def check_point(row):
            point = Point(row['global_y'], row['global_x'])
            possible_cells = [cell_id for cell_id in rtree_index.intersection(point.bounds) if outline_polygons[cell_id].contains(point)]
            return possible_cells[0] if possible_cells else -1
        
        df_spots['cell_id'] = df_spots.apply(check_point, axis=1)
        
    return df_spots

def process_bit_file(bit_file_info):
    bit_file_path, bit_id = bit_file_info
    temp_df = []
    for tile_id in tile_ids:
        df_tile = pd.read_parquet(ufish_path / tile_id / Path(bit_file_path))
        bit_idx = df_tile['bit_idx'][0]
        
        df_selected_spots = df_tile[df_tile['use_spot'] == 1]
        if df_selected_spots.empty:
            continue

        polyDT_dir_path = dataset_path / Path('polyDT')
        polyDT_current_path = polyDT_dir_path / Path(tile_id) / Path("round000.zarr")
        polyDT_current_tile = zarr.open(polyDT_current_path, mode='r')
        try:
            affine = np.asarray(polyDT_current_tile.attrs['affine_zyx_um'], dtype=np.float32)
            origin = np.asarray(polyDT_current_tile.attrs['origin_zyx_um'], dtype=np.float32)
            spacing = np.asarray(polyDT_current_tile.attrs['spacing_zyx_um'], dtype=np.float32)
            voxel_size_zyx_um = np.asarray(polyDT_current_tile.attrs['voxel_zyx_um'], dtype=np.float32)
        except:
            affine = np.eye(4)
            origin = np.asarray(polyDT_current_tile.attrs['stage_zyx_um'], dtype=np.float32)
            spacing = np.asarray(polyDT_current_tile.attrs['voxel_zyx_um'], dtype=np.float32)
            
        df_selected_spots['gene_id'] = df_selected_spots.apply(lambda x: gene_ids[bit_idx-1], axis=1)

        pts = df_selected_spots[['tile_z_px', 'tile_y_px', 'tile_x_px']].to_numpy()
        pts = np.apply_along_axis(warp_pixel, 1, pts, spacing, origin, affine)
        
        df_selected_spots['global_z'] = np.round(pts[:, 0], 3)
        df_selected_spots['global_y'] = np.round(pts[:, 1], 3)
        df_selected_spots['global_x'] = np.round(pts[:, 2], 3)

        temp_df.append(df_selected_spots)
        
        if not temp_df:
            return None
    
    df_all_tiles = pd.concat(temp_df, ignore_index=True)
    df_all_tiles_with_cells = assign_cells(df_all_tiles)
    
    data_path = decoded_spots_path / Path(bit_id + '.csv')
    df_all_tiles_with_cells.to_csv(data_path, index=False)

    return df_all_tiles_with_cells

# Process all bit files in parallel
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_bit_file, file_info))
    
# TO DO:
# 1. double check outlines vs masks
# 2. sane defaults for thresholds. Re-run to not lose spots
# 3. add mtx creation at end of smFISH pipeline
# 4. rerun cellpose without RNA masks (add flag)
# 5. normalization (?) for histogram display