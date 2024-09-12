"""
Run Cellpose on fused polyDT within qi2labdatastore.

Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils._outlinesprocessing import extract_outlines, create_microjson, calculate_centroids
from cellpose import models
from pathlib import Path
import numpy as np

def segment_fused_image():
    # root data folder
    root_path = Path(r"/mnt/data/qi2lab/20240807_OB_22bit_PL028_2")

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    
    fused_image, affine, origin, spacing = datastore.load_global_fidicual_image(return_future=False)
    
    channels = [[0,0]]
    
    model = models.Cellpose(gpu=True,model_type='cyto3')
    model.diam_mean = 30.0

    masks, _, _, _ = model.eval(np.squeeze(np.max(np.squeeze(fused_image),axis=0)),
                                channels=channels,
                                flow_threshold=0.0,
                                normalize = {'normalize': True,
                                            'percentile': [10,90]})
    
    import napari
    viewer = napari.Viewer()
    viewer.add_labels(masks,scale=spacing[1:])
    viewer.add_image(
        np.squeeze(np.max(np.squeeze(fused_image),axis=0)),
        scale=spacing[1:], 
    )
    napari.run()
       
    # cell_outlines_px = extract_outlines(masks)
    # cell_outlines_microjson = create_microjson(cell_outlines_px,
    #                                             spacing,
    #                                             origin,
    #                                             affine)
    # cell_centroids = calculate_centroids(cell_outlines_px,
    #                                         spacing,
    #                                         origin,
    #                                         affine)
    
    # masks_json_path = cellpose_dir_path / Path('cell_outlines.json')
    # with open(masks_json_path, "w") as f:
    #     json.dump(cell_outlines_microjson, f, indent=2)
        
    # mask_centroids_path = cellpose_dir_path / Path('cell_centroids.parquet')
    # cell_centroids.to_parquet(mask_centroids_path)
        
    # del masks, data_to_segment
    # del affine, origin, spacing
    
if __name__ == "__main__":
    segment_fused_image()