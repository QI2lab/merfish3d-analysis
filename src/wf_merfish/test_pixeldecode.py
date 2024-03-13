from wf_merfish.postprocess.PixelDecoder import PixelDecoder
from pathlib import Path
import napari
from tqdm import tqdm
import gc
import cupy as cp

#data_dir_path = Path('/home/qi2lab/Documents/github/wf-merfish/examples/simulated_images/cylinder/images/jitter-0_shift_amp-0_prop_fn-0_prop_fp-0/processed')
dataset_path = Path('/mnt/opm3/20240214_MouseBrain_UA_NewRO_RK/processed_v2')


decode_factory = PixelDecoder(dataset_path=dataset_path,
                            global_normalization_limits = (80,99.95),
                            overwrite_normalization=False)
# decode_factory.load_all_barcodes()
# decode_factory.filter_all_barcodes()
#decode_factory.save_barcodes()


tile_ids = decode_factory._tile_ids

del decode_factory
gc.collect()
cp.get_default_memory_pool().free_all_blocks()

for tile_idx, tile_id in enumerate(tqdm(tile_ids,desc='tile',leave=True)):
    

    decode_factory = PixelDecoder(dataset_path=dataset_path,
                                tile_idx=tile_idx)
    decode_factory.run_decoding(lowpass_sigma=(3,2,2),
                                distance_threshold=0.78,
                                magnitude_threshold=0.4,
                                skip_extraction=False)
    decode_factory.save_barcodes()
    decode_factory.cleanup()
    
    del decode_factory
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    if False:
        viewer = napari.Viewer()
        
        viewer.add_image(decode_factory._scaled_pixel_images,
                        scale=[.31,.088,.088],
                            name='pixels')
        
        viewer.add_image(decode_factory._decoded_image,
                            scale=[.31,.088,.088],
                            name='decoded')

        viewer.add_image(decode_factory._magnitude_image,
                            scale=[.31,.088,.088],
                            name='magnitude')

        viewer.add_image(decode_factory._distance_image,
                            scale=[.31,.088,.088],
                            name='distance')

        napari.run()


# if False:
#     viewer = napari.Viewer()
    
#     viewer.add_image(decode_factory._scaled_pixel_images,
#                      scale=[.31,.088,.088],
#                         name='pixels')
    
#     viewer.add_image(decode_factory._decoded_image,
#                         scale=[.31,.088,.088],
#                         name='decoded')

#     viewer.add_image(decode_factory._magnitude_image,
#                         scale=[.31,.088,.088],
#                         name='magnitude')

#     viewer.add_image(decode_factory._distance_image,
#                         scale=[.31,.088,.088],
#                         name='distance')

#     napari.run()

# if False:
#     import pandas as pd
#     import numpy as np
#     from scipy.spatial import cKDTree


#     true_localization_path = Path('/home/qi2lab/Documents/github/wf-merfish/examples/simulated_images/cylinder/coordinates/jitter-0_shift_amp-0_prop_fn-0_prop_fp-0/true_coords_n_species-121_n_spots-5_repeat_id-0.csv')
#     df_true = pd.read_csv(true_localization_path)
#     true_coordinates = df_true[['z', 'y', 'x']].to_numpy()
#     true_coordinates[:,0] = true_coordinates[:,0]/.31
#     true_coordinates[:,1] = true_coordinates[:,1]/.088
#     true_coordinates[:,2] = true_coordinates[:,2]/.088
#     true_id = df_true['species'].str.extract('(\d+)$').astype(int).to_numpy().ravel()

#     df_exp = pd.read_csv(decode_factory._barcode_path)
#     exp_coordinates = df_exp[['z', 'y', 'x']].to_numpy()
#     exp_id = df_exp['barcode_id'].to_numpy()

#     allowed_error = 1.5  # Define your allowed error

#     # Create a cKDTree for the ground truth
#     tree_ground_truth = cKDTree(true_coordinates)

#     # Initialize a list to store indices of coordinates with matches in ground truth
#     matching_indices = []

#     # Iterate through each point in coordinates and query the KD-tree
#     for i, coord in enumerate(exp_coordinates):
#         # Query the KD-tree for matches within the allowed error
#         matches = tree_ground_truth.query_ball_point(coord, r=allowed_error)
        
#         # If there are matches, add the index to the list
#         if matches:
#             matching_indices.append(i)

#     # Example values, replace these with your actual counts
#     tp = len(matching_indices)  # Number of true positives
#     total_ground_truth = len(true_coordinates)  # Total number of ground truth points
#     total_predicted = len(exp_coordinates)  # Total number of points you attempted to match

#     # Assuming all coordinates were attempted to match and matches were checked against ground truth
#     fp = total_predicted - tp  # False positives
#     fn = total_ground_truth - tp  # False negatives

#     # Calculate precision and recall
#     precision = tp / (tp + fp) if tp + fp > 0 else 0
#     recall = tp / (tp + fn) if tp + fp > 0 else 0

#     # Calculate F1 score
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#     print(f"F1 Score: {np.round(f1,2)}")