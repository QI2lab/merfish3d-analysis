"""
Decode using qi2lab GPU decoder.

Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2024/12 - create script to run on simulation.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.PixelDecoder import PixelDecoder
from pathlib import Path
import typer

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def decode_pixels(
    root_path: Path,
    minimum_pixels_per_RNA: int = 2,
    distance_threshold: float = 1.0,
    ufish_threshold: float = 0.1,
    magnitude_threshold: tuple[float,float]  =[0.9, 10.0],
    fdr_target: float = .05,
    smFISH: bool = False,
):
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    minimum_pixels_per_RNA : int, default 2
        minimum pixels with same barcode ID required to call a spot.
    ufish_threshold : float, default 0.1
        threshold to accept ufish prediction.
    magnitude_threshold: Sequence[float], default (1.1,10.0)
        minimum magnitude across all normalized bits required to accept a spot.
    fdr_target : float, default .05
        false discovery rate (FDR) target.
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
        verbose=1,
        smFISH=smFISH
    )
    if smFISH:
        decoder._distance_threshold = distance_threshold

    decoder.optimize_normalization_by_decoding(
        n_random_tiles=1,
        n_iterations=1,
        distance_threshold=distance_threshold,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        ufish_threshold=ufish_threshold,
    )
    
    decoder.decode_all_tiles(
        assign_to_cells=False,
        prep_for_baysor=False,
        distance_threshold=distance_threshold,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        ufish_threshold=ufish_threshold,
        fdr_target=fdr_target
    )
    
def main():
    app()

if __name__ == "__main__":
    main()