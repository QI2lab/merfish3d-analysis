"""
Decode using qi2lab GPU decoder and (re)-segment cells based on decoded RNA.

Shepherd 2024/1 - modified script to accept parameters with sensible defaults.
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.postprocess.PixelDecoder import PixelDecoder
from pathlib import Path

def decode_pixels(
    root_path: Path,
    minimum_pixels_per_RNA: int = 9,
    ufish_threshold: float = 0.5,
    fdr_target: float = 0.2,
    run_baysor: bool = True,
):
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    merfish_bits : int
        number of bits in codebook
    minimum_pixels_per_RNA : int
        minimum pixels with same barcode ID required to call a spot. Default = 9.
    ufish_threshold : float
        threshold to accept ufish prediction. Default = 0.5
    fdr_target : float
        false discovery rate (FDR) target. Default = .2
        NOTE: This is higher than usual, but we are finding that .05 is too 
        aggressive for nyquist-sampled 3D data  and MLP filtering strategy we 
        have implemented. Ongoing effort to fully understand this issue using 
        synthetic data.
    run_baysor : bool
        flag to run Baysor segmentation. Default = True
    """

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    merfish_bits = datastore.num_bits

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore, 
        use_mask=False, 
        merfish_bits=merfish_bits, 
        verbose=1
    )

    # optimize normalization weights through iterative decoding and update
    decoder.optimize_normalization_by_decoding(
        n_random_tiles=10,
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

    # resegment data using baysor and cellpose prior assignments
    if run_baysor:
        datastore.run_baysor()
        datastore.save_mtx()

if __name__ == "__main__":
    root_path = Path(r"/mnt/data/bartelle/20241108_Bartelle_MouseMERFISH_LC")
    decode_pixels(root_path=root_path)