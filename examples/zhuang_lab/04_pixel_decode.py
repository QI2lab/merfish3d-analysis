"""
Decode using qi2lab GPU decoder and (re)-segment cells based on decoded RNA.

Baysor re-segmentation is performed without OR genes to avoid biasing results.

Shepherd 2024/01 - modified script to accept parameters with sensible defaults.
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.PixelDecoder import PixelDecoder
from pathlib import Path

def pixeldecode_and_baysor(
    root_path: Path,
    minimum_pixels_per_RNA: int = 2,
    ufish_threshold: float = 0.5,
    fdr_target: float = .05,
    run_baysor: bool = True,
):
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    merfish_bits : int
        number of bits in codebook
    minimum_pixels_per_RNA : int, default = 9
        minimum pixels with same barcode ID required to call a spot.
    ufish_threshold : float, default = 0.5
        threshold to accept ufish prediction. 
    fdr_target : float, default = .05
        false discovery rate (FDR) target. 
    run_baysor : bool, default True
        flag to run Baysor segmentation.
    """

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    merfish_bits = 22

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore, 
        use_mask=False, 
        merfish_bits=merfish_bits, 
        verbose=1
    )

    # optimize normalization weights through iterative decoding and update
    decoder.optimize_normalization_by_decoding(
        n_random_tiles=50,
        n_iterations=10,
        minimum_pixels=minimum_pixels_per_RNA,
        ufish_threshold=ufish_threshold,
    )

    # decode all tiles using iterative normalization weights
    decoder.decode_all_tiles(
        assign_to_cells=True,
        prep_for_baysor=True,
        minimum_pixels=minimum_pixels_per_RNA,
        fdr_target=fdr_target,
        ufish_threshold=ufish_threshold,
    )

    # # resegment data using baysor and cellpose prior assignments
    if run_baysor:
        datastore.run_baysor()
        datastore.save_mtx(spots_source="baysor")

if __name__ == "__main__":
    root_path = Path(r"/mnt/data/zhuang/")
    pixeldecode_and_baysor(root_path=root_path,run_baysor=True,fdr_target=.05)