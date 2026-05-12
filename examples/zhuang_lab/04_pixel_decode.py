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

QI2LAB_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST = {
    3.0: 0.7,
    5.0: 0.2,
}
QI2LAB_AXIAL_NYQUIST_STEP_UM = 0.315


def _default_minimum_pixels(datastore: qi2labDataStore) -> int:
    """Return the current qi2lab default minimum-pixel threshold."""

    return 7 if datastore.microscope_type == "2D" else 28


def _default_magnitude_threshold(
    datastore: qi2labDataStore,
) -> tuple[float, float]:
    """Return the current qi2lab default magnitude threshold."""

    if datastore.microscope_type != "2D":
        return (0.9, 10.0)

    z_step_um = float(datastore.voxel_size_zyx_um[0])
    nyquist_multiple = z_step_um / QI2LAB_AXIAL_NYQUIST_STEP_UM
    nearest_multiple = min(
        QI2LAB_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST,
        key=lambda value: abs(value - nyquist_multiple),
    )
    return (QI2LAB_2D_MAGNITUDE_THRESHOLD_BY_NYQUIST[nearest_multiple], 10.0)


def decode_pixels(
    root_path: Path,
    minimum_pixels_per_RNA: int | None = None,
    feature_predictor_threshold: float = 0.5,
    magnitude_threshold: tuple[float, float] | None = None,
    target_gross_misid_rate: float = 0.05,
) -> None:
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    minimum_pixels_per_RNA : int, optional
        minimum pixels with same barcode ID required to call a spot.
        Defaults to the current qi2lab policy: 7 for 2D and 28 for 3D.
    feature_predictor_threshold : float
        threshold to accept feature_predictor prediction. Default = 0.5
    magnitude_threshold: tuple[float,float], optional
        lower and upper magnitude threshold to accept a spot.
        Defaults to the current qi2lab policy:
        3D -> (0.9, 10.0), 2D -> lookup by axial sampling.
    target_gross_misid_rate : float
        gross barcode misidentification-rate target for blank-fraction filtering.
        Default = .05
    """

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path, validate=False)
    merfish_bits = 22
    if minimum_pixels_per_RNA is None:
        minimum_pixels_per_RNA = _default_minimum_pixels(datastore)
    if magnitude_threshold is None:
        magnitude_threshold = _default_magnitude_threshold(datastore)

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
        feature_predictor_threshold=feature_predictor_threshold,
        magnitude_threshold=magnitude_threshold,
    )

    # decode all tiles using iterative normalization weights
    decoder.decode_all_tiles(
        assign_to_cells=True,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        feature_predictor_threshold=feature_predictor_threshold,
        target_gross_misid_rate=target_gross_misid_rate,
    )


if __name__ == "__main__":
    root_path = Path(r"/media/dps/data/zhuang")
    decode_pixels(root_path=root_path)
