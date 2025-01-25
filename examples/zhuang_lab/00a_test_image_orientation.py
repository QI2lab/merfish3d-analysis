import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tifffile import imread
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import zarr

def create_overview_image(root_path: Path, n_tiles: int = 2):
    """Load and create blended tile overview image.
    
    This function loads 3D tiles, performs a maximum z-projection, and provides 
    blended global images for both (y,x) and (x,y) in the stage positions file.
    It is also possible to test image flips and stage reversals with this 
    approach with minor modifications.
    
    It is currently hard-coded to work with the Zhuang lab example as part of
    the merfish3d-analysis documentation.
    
    Parameters
    ----------
    root_path: Path
        path to root data directory
    n_tiles: int, default 2
        number of tiles to load
    """
    
    # Load n_tiles
    tiles = []
    for tile_idx in tqdm(range(n_tiles), desc="loading tiles"):
        tile_path = root_path / Path(f"mop/mouse_sample1_raw/mouse1_sample1_raw/aligned_images{tile_idx}.tif")
        store = imread(tile_path, mode='r', aszarr=True)
        z = zarr.open(store, mode='r')
        tiles.append(np.max(z[38, :].astype(np.uint16), axis=0))
    tiles = np.asarray(tiles)

    # Load the spatial calibration (microns per pixel)
    microscope_json_path = root_path / Path(r"mop/mouse_sample1_raw/additional_files/microscope.json")
    with open(microscope_json_path, 'r') as file:
        data = json.load(file)
    yx_pixel_size_um = float(data["microns_per_pixel"])

    # Load the global stage translations for each tile in microns
    stage_translation_path = root_path / Path(
        r"mop/mouse_sample1_raw/additional_files/fov_positions/mouse1_sample1.txt"
    )
    stage_position_df = pd.read_csv(stage_translation_path, header=None)
    stage_positions = stage_position_df.values
    tile_translations = []
    for tile_idx in range(n_tiles):
        tile_translations.append(stage_positions[tile_idx, :])
    tile_translations = np.asarray(tile_translations)

    # Calculate extents in the global coordinate system
    tile_extents_yx_order = []
    tile_extents_xy_order = []
    for tile_idx in range(n_tiles):
        tile_extents_yx_order.append(
            [
                tile_translations[tile_idx, 1] / yx_pixel_size_um,  # x_min
                tile_translations[tile_idx, 1] / yx_pixel_size_um + tiles[tile_idx, :].shape[1],  # x_max
                tile_translations[tile_idx, 0] / yx_pixel_size_um,  # y_min
                tile_translations[tile_idx, 0] / yx_pixel_size_um + tiles[tile_idx, :].shape[0],  # y_max
            ]
        )
        tile_extents_xy_order.append(
            [
                tile_translations[tile_idx, 0] / yx_pixel_size_um,  # x_min
                tile_translations[tile_idx, 0] / yx_pixel_size_um + tiles[tile_idx, :].shape[0],  # x_max
                tile_translations[tile_idx, 1] / yx_pixel_size_um,  # y_min
                tile_translations[tile_idx, 1] / yx_pixel_size_um + tiles[tile_idx, :].shape[1],  # y_max
            ]
        )
    tile_extents_yx_order = np.asarray(tile_extents_yx_order)
    tile_extents_xy_order = np.asarray(tile_extents_xy_order)

    # Determine the overall image extents for both (y,x) and (x,y) order
    x_min_yx_order = np.min(tile_extents_yx_order[:, 0])
    x_max_yx_order = np.max(tile_extents_yx_order[:, 1])
    y_min_yx_order = np.min(tile_extents_yx_order[:, 2])
    y_max_yx_order = np.max(tile_extents_yx_order[:, 3])

    x_min_xy_order = np.min(tile_extents_xy_order[:, 0])
    x_max_xy_order = np.max(tile_extents_xy_order[:, 1])
    y_min_xy_order = np.min(tile_extents_xy_order[:, 2])
    y_max_xy_order = np.max(tile_extents_xy_order[:, 3])

    # Create blending canvases
    canvas_yx = np.zeros((int(np.ceil(y_max_yx_order - y_min_yx_order)), 
                        int(np.ceil(x_max_yx_order - x_min_yx_order)), 3), dtype=np.float32)
    weights_yx = np.zeros(canvas_yx.shape[:2], dtype=np.float32)

    canvas_xy = np.zeros((int(np.ceil(y_max_xy_order - y_min_xy_order)), 
                        int(np.ceil(x_max_xy_order - x_min_xy_order)), 3), dtype=np.float32)
    weights_xy = np.zeros(canvas_xy.shape[:2], dtype=np.float32)

    # Define colormaps for alternating colors
    colormaps = [cm.Blues_r, cm.Oranges_r]

    # Set gamma value
    gamma = 0.25

    # Blend tiles for (y,x) order
    for tile_idx in range(n_tiles):
        cmap = colormaps[tile_idx % len(colormaps)]  # Alternate colors
        tile_normalized = (tiles[tile_idx] / np.max(tiles[tile_idx])) ** gamma
        tile_rgb = cmap(tile_normalized)[:, :, :3]

        # Calculate tile position on the canvas
        x_start = int(np.round(tile_extents_yx_order[tile_idx, 0] - x_min_yx_order))
        x_end = x_start + tiles[tile_idx].shape[1]
        y_start = int(np.round(tile_extents_yx_order[tile_idx, 2] - y_min_yx_order))
        y_end = y_start + tiles[tile_idx].shape[0]

        # Add tile to canvas
        canvas_yx[y_start:y_end, x_start:x_end, :] += tile_rgb
        weights_yx[y_start:y_end, x_start:x_end] += 1

    # Blend tiles for (x,y) order
    for tile_idx in range(n_tiles):
        cmap = colormaps[tile_idx % len(colormaps)]  # Alternate colors
        tile_normalized = (tiles[tile_idx] / np.max(tiles[tile_idx])) ** gamma
        tile_rgb = cmap(tile_normalized)[:, :, :3]

        # Calculate tile position on the canvas
        x_start = int(np.round(tile_extents_xy_order[tile_idx, 0] - x_min_xy_order))
        x_end = x_start + tiles[tile_idx].shape[1]
        y_start = int(np.round(tile_extents_xy_order[tile_idx, 2] - y_min_xy_order))
        y_end = y_start + tiles[tile_idx].shape[0]

        # Add tile to canvas
        canvas_xy[y_start:y_end, x_start:x_end, :] += tile_rgb
        weights_xy[y_start:y_end, x_start:x_end] += 1

    # Normalize the canvases
    weights_yx = np.maximum(weights_yx, 1)
    canvas_yx /= weights_yx[:, :, np.newaxis]

    weights_xy = np.maximum(weights_xy, 1)
    canvas_xy /= weights_xy[:, :, np.newaxis]

    # Plot the z max projections of the tiles assuming the stage file is in (y,x) order
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas_yx, origin='lower')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('Tiles assuming (y,x) order')
    plt.grid(False)
    
    plt.show()

    # Plot the z max projections of the tiles assuming the stage file is in (x,y) order
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas_xy, origin='lower')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('Tiles assuming (x,y) order')
    plt.grid(False)

    plt.show()


if __name__ == "__main__":
    root_path = Path(r"/mnt/data/zhuang/")
    n_tiles = 2
    create_overview_image(root_path,n_tiles)