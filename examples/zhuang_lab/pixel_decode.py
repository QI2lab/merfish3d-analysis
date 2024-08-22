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
        use_mask=False,
        merfish_bits=22,
        verbose=1
    )
    
    decoder.optimize_normalization_by_decoding(n_iterations=2,
                                               n_random_tiles=50,
                                               minimum_pixels=2.0)
    decoder.decode_all_tiles(assign_to_cells=False,
                             prep_for_baysor=False,
                             minimum_pixels=2.0,
                             fdr_target=.05)
    
if __name__ == "__main__":
    decode_pixels()