"""
Decode using qi2lab GPU decoder

Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.postprocess.PixelDecoder import PixelDecoder
from pathlib import Path

def decode_pixels():
    # root data folder
    root_path = Path(r"/mnt/data/qi2lab/20241012_OB_22bit_MERFISH")

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    
    decoder = PixelDecoder(
        datastore=datastore,
        use_mask=False,
        merfish_bits=22,
        verbose=1
    )
    
    decoder.optimize_normalization_by_decoding(n_random_tiles=10, 
                                               n_iterations=20,
                                               minimum_pixels=9,
                                               ufish_threshold=0.6)
    
    decoder.decode_all_tiles(assign_to_cells=False,
                             prep_for_baysor=True,
                             minimum_pixels=9,
                             fdr_target=.2,
                             ufish_threshold=0.6)
    
    datastore.run_baysor()
    datastore.save_mtx()
    
    
if __name__ == "__main__":
    decode_pixels()