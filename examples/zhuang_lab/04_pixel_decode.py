"""Decode using qi2lab GPU decoder

Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.postprocess.PixelDecoder import PixelDecoder
from pathlib import Path

def decode_pixels():
    datastore_path = Path(r"/mnt/data/zhuang/mop/mouse_sample1_raw/processed_v3")
    datastore = qi2labDataStore(datastore_path=datastore_path)
    
    decoder = PixelDecoder(
        datastore=datastore,
        merfish_bits=22,
    )
    
    decoder.decode_one_tile(tile_idx=0,
                            lowpass_sigma=(1,1,1),
                            minimum_pixels=2,
                            ufish_threshold=0.6)
    # Max: this threshold is applied to the ufish output before multiplying the data
    # decoder.optimize_normalization_by_decoding(
    #     n_iterations=4,
    #     n_random_tiles=50,
    #     minimum_pixels=2.0,
    #     ufish_threshold=0.7
    # )
    
    # decoder.decode_all_tiles(
    #     assign_to_cells=False,
    #     prep_for_baysor=False,
    #     minimum_pixels=2.0,
    #     fdr_target=.05,
    #     ufish_threshold=0.6
    # )
    
if __name__ == "__main__":
    decode_pixels()