"""
Decode using qi2lab GPU decoder and (re)-segment cells based on decoded RNA.

Shepherd 2025/07 - refactor for multiple GPU support.
Shepherd 2024/12 - refactor
Shepherd 2024/11 - modified script to accept parameters with sensible defaults.
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from pathlib import Path

from merfish3danalysis.PixelDecoder import PixelDecoder
from merfish3danalysis.qi2labDataStore import qi2labDataStore


def decode_pixels(
    root_path: Path,
    minimum_pixels_per_RNA: int = 2,
    ufish_threshold: float = 0.01,
    magnitude_threshold: float = (0.1, 10.0),
    fdr_target: float = 0.3,
    run_baysor: bool = False,
) -> None:
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
        threshold to accept ufish prediction. Default = 0.1
    magnitude_threshold: tuple[float,float], default = (1.,5.)
        lower and upper magnitude threshold to accept a spot. We allow for >2 on upper because
        spots are normalized to median spot value, not maximum.
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
    merfish_bits = 22

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        num_gpus=1,
        verbose=1,
    )

    # optimize normalization weights through iterative decoding and update
    decoder.optimize_normalization_by_decoding(
        n_random_tiles=10,
        n_iterations=5,
        minimum_pixels=minimum_pixels_per_RNA,
        ufish_threshold=ufish_threshold,
        magnitude_threshold=magnitude_threshold,
    )

    # decode all tiles using iterative normalization weights
    decoder.decode_all_tiles(
        assign_to_cells=True,
        prep_for_baysor=True,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        ufish_threshold=ufish_threshold,
        fdr_target=fdr_target,
    )

    # resegment data using baysor and cellpose prior assignments
    if run_baysor:
        datastore.run_baysor()
        datastore.save_mtx()


if __name__ == "__main__":
    root_path = Path(r"/media/dps/data/zhuang")
    decode_pixels(root_path=root_path, run_baysor=False)
