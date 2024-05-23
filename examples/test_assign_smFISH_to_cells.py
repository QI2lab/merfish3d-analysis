import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from pathlib import Path
import zarr
from shapely import Polygon, Point
import json

root_path = Path('/mnt/data/bartelle/20240423_ECL_24CryoA_2_PL025_restart')
dataset_path = root_path / Path('processed_v2')

calibration_dir_path = dataset_path / Path("calibrations.zarr")
calibration_zarr = zarr.open(calibration_dir_path,mode='a')
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

def warp_pixel(pixel_space_point: np.ndarray,
                spacing: np.ndarray,
                origin: np.ndarray,
                affine: np.ndarray) -> np.ndarray:

    physical_space_point = pixel_space_point * spacing + origin
    registered_space_point = (np.array(affine) @ np.array(list(physical_space_point) + [1]))[:-1]
    
    return registered_space_point

def load_microjson(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        outlines = {}
        
        for feature in data['features']:
            cell_id = feature['properties']['cell_id']
            coordinates = feature['geometry']['coordinates'][0]
            outlines[cell_id] = np.array(coordinates)
        return outlines
                       
def assign_cells(df_barcodes):
    try:
        outlines_path = dataset_path / Path("segmentation") / Path("cellpose") / Path("cell_outlines.json")
        cell_outlines = load_microjson(outlines_path)
        has_cell_outlines = True
    except:
        has_cell_outlines = False
    
    if has_cell_outlines:
        outline_polygons = {cell_id: Polygon(outline) for cell_id, outline in cell_outlines.items()}
        def check_point(row):
            point = Point(row['global_y'], row['global_x'])
            for cell_id, polygon in outline_polygons.items():
                if polygon.contains(point):
                    return cell_id
            return -1
        
        df_barcodes['cell_id'] = df_barcodes.apply(check_point, axis=1)
        
    return df_barcodes

for bit_file_path, bit_id in file_info:
    df_barcode = []
    for tile_idx, tile_id in enumerate(tile_ids):
        df_tile = pd.read_parquet(ufish_path / tile_id / Path(bit_file_path))
        bit_idx = df_tile['bit_idx'][0]
        
        df_selected_spots = df_tile[df_tile['use_spot']==1]

        polyDT_dir_path = dataset_path / Path('polyDT')
        polyDT_current_path = polyDT_dir_path / Path(tile_id) / Path("round000.zarr")
        polyDT_current_tile = zarr.open(polyDT_current_path,mode='r')
        try:
            affine = np.asarray(polyDT_current_tile.attrs['affine_zyx_um'],
                                        dtype=np.float32)
            origin = np.asarray(polyDT_current_tile.attrs['origin_zyx_um'],
                                        dtype=np.float32)
            spacing = np.asarray(polyDT_current_tile.attrs['spacing_zyx_um'],
                                        dtype=np.float32)
            voxel_size_zyx_um = np.asarray(polyDT_current_tile.attrs['voxel_zyx_um'],dtype=np.float32)
            pixel_size = voxel_size_zyx_um[1]
            axial_step = voxel_size_zyx_um[0]        
        except:
            affine = np.eye(4)
            origin = np.asarray(polyDT_current_tile.attrs['stage_zyx_um'],
                                        dtype=np.float32)
            spacing = np.asarray(polyDT_current_tile.attrs['voxel_zyx_um'],
                                        dtype=np.float32)
            
        df_selected_spots['gene_id'] = df_barcode.apply(lambda x: gene_ids[bit_idx-1], axis=1) 

        pts = df_barcode[['tile_z_px', 'tile_y_px', 'tile_x_px']].to_numpy()
        for pt_idx,pt in enumerate(pts):
            pts[pt_idx,:] = warp_pixel(pts[pt_idx,:].copy(),
                                       spacing,
                                       origin,
                                       affine)
            
        df_selected_spots['global_z'] = np.round(pts[:,0],2)
        df_selected_spots['global_y'] = np.round(pts[:,1],2)
        df_selected_spots['global_x'] = np.round(pts[:,2],2)
        
        if tile_idx == 0:
            df_all_tiles = df_selected_spots.copy()
        else:
            df_all_tiles = pd.concat([df_all_tiles,df_selected_spots])
            df_all_tiles.reset_index(drop=True, inplace=True)
            
    df_all_tiles_with_cells = assign_cells(df_all_tiles)
    
    data_path = decoded_spots_path / Path(bit_id+'.csv')
    df_all_tiles_with_cells.to_csv(data_path)
    
    del df_all_tiles, df_selected_spots, df_all_tiles_with_cells