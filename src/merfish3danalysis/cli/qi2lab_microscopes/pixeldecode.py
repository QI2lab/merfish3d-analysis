"""
Decode using qi2lab GPU decoder and (re)-segment cells based on decoded RNA.

Shepherd 2025/07 - refactor for multiple GPU support.
Shepherd 2024/12 - refactor
Shepherd 2024/11 - modified script to accept parameters with sensible defaults.
Shepherd 2024/08 - rework script to utilized qi2labdatastore object.
"""

from pathlib import Path

import typer

from merfish3danalysis.PixelDecoder import PixelDecoder
from merfish3danalysis.qi2labDataStore import qi2labDataStore

app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def decode_pixels(
    root_path: Path,
    num_gpus: int = 1,
    minimum_pixels_per_RNA: int = 3,
    ufish_threshold: float = 0.25,
    magnitude_threshold: tuple[float, float] = [0.9, 10.0],
    fdr_target: float = 0.05,
    run_baysor: bool = False,
    merfish_bits: int | None = None,
    smFISH: bool = False,
    skip_optimization: bool = False,
    reprocess_existing: bool = False,
) -> None:
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    num_gpus : int
        number of gpus to use. Default = 1.
    minimum_pixels_per_RNA : int
        minimum pixels with same barcode ID required to call a spot. Default = 3.
    ufish_threshold : float
        threshold to accept ufish prediction. Default = 0.25
    magnitude_threshold : tuple[float,float]. Default = [0.9,10.0]
        list of two floats [min, max] magnitude thresholds to accept a decoded pixel.
    fdr_target : float
        false discovery rate (FDR) target. Default = .05
    run_baysor : bool
        flag to run Baysor segmentation. Default = False
    merfish_bits : int. default = None
        number of bits in codebook. By default uses all bits in codebook.
    smFISH: bool, default = False
        run in smFISH processing mode.
    skip_optimization: bool, default = False
        skip running iterative optimization.
    reprocess_existing : bool, default = False
        flag to reprocess existing decoded data.
    """

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    if merfish_bits is None:
        merfish_bits = datastore.num_bits

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        num_gpus=num_gpus,
        verbose=1,
        smFISH=smFISH,
    )

    if smFISH:
        decoder._distance_threshold = 1.0

    if not (reprocess_existing):
        if not skip_optimization:
            # optimize normalization weights through iterative decoding and update
            decoder.optimize_normalization_by_decoding(
                n_random_tiles=20,
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
    else:
        decoder.optimize_filtering(
            assign_to_cells=True, prep_for_baysor=True, fdr_target=fdr_target
        )

    # resegment data using baysor and cellpose prior assignments
    if run_baysor:
        datastore.run_baysor()
        datastore.save_mtx()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
