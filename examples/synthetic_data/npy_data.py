import numpy as np
import tifffile as tf
import os
import zarr

base_input_dir = '/home/julian/Documents/julian/training_data/numpy_data'
base_output_dir = "/home/julian/Documents/julian/training_data/readouts"

num_rounds = 6
num_tiles = 256

for i in range(num_tiles):
    file = f"{base_input_dir}/training{i+1}/compatibility/raw_data_signal.npy"
    I = np.load(file)
    
    for j in range(num_rounds):
        # Create directory path with proper zero-padding
        tile_dir = f"tile{i:03d}"
        round_dir = f"round{j:03d}"
        full_dir = os.path.join(base_output_dir, tile_dir, round_dir)
        
        # Create all necessary directories
        os.makedirs(full_dir, exist_ok=True)
        
        # Save TIFF file
        tiff_file = os.path.join(full_dir, f"{j:03d}.tiff")
        tf.imwrite(tiff_file, I[:,:,:,j])

        bit_dir = f"bit{j:01}.zarr"
        full_dir = os.path.join(full_dir, bit_dir)
        os.makedirs(full_dir, exist_ok=True)
    
    # Save codebook and bit order files in the tile directory
    tile_dir = os.path.join(base_output_dir, f"tile{i:03d}")
    
    cb = np.load(f"{base_input_dir}/training{i+1}/compatibility/raw_data_code.npy")
    np.savetxt(os.path.join(tile_dir, "codebook.csv"), cb, delimiter=",")
    
    bit_order = np.load(f"{base_input_dir}/training{i+1}/compatibility/gt_info_RNA.npy")
    np.savetxt(os.path.join(tile_dir, "bit_order.csv"), bit_order, delimiter=",")