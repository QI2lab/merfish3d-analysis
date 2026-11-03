"""
Sweep through decoding parameters and calculate F1-score using known ground truth.

Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2024/12 - create script to run on simulation.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import typer

from merfish3danalysis.cli.statphysbio_simulation.calculate_F1 import (
    calculate_F1 as _calculate_simulation_f1,
)
from merfish3danalysis.PixelDecoder import PixelDecoder, time_stamp
from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import resolve_datastore_path
from merfish3danalysis.utils.metrics import (
    calculate_F1_with_radius as _match_transcripts,
)


def calculate_F1_with_radius(
    qi2lab_coords: np.ndarray,
    qi2lab_gene_ids: np.ndarray,
    gt_coords: np.ndarray,
    gt_gene_ids: np.ndarray,
    radius: float,
) -> dict:
    """Return spatial matching metrics using the shared transcript matcher.

    Parameters
    ----------
    qi2lab_coords : np.ndarray
        Decoded transcript coordinates.
    qi2lab_gene_ids : np.ndarray
        Decoded transcript gene identifiers.
    gt_coords : np.ndarray
        Ground-truth transcript coordinates.
    gt_gene_ids : np.ndarray
        Ground-truth gene identifiers.
    radius : float
        Maximum distance for a match.

    Returns
    -------
    dict
        Same-gene, one-to-one matching metrics.
    """
    results, _, _, _ = _match_transcripts(
        qi2lab_coords, qi2lab_gene_ids, gt_coords, gt_gene_ids, radius
    )
    return results


def calculate_F1(root_path: Path, gt_path: Path, search_radius: float) -> dict:
    """Calculate F1 using ground truth.

    Parameters
    ----------
    root_path: Path
        path to experiment
    gt_path: Path
        path to ground truth file
    search_radius: float
        search radius for a sphere in microns. Should be 2-3x the z step,
        depending on the amount of low-pass blur applied.

    Returns
    -------
    results: dict
        dictionary of results for F1 score calculation
    """
    return _calculate_simulation_f1(
        root_path=root_path, gt_path=gt_path, search_radius=search_radius
    )


def decode_pixels(
    root_path: Path,
    mag_threshold: Sequence[float],
    feature_predictor_threshold: float,
    minimum_pixels: float,
) -> None:
    """Run pixel decoding with the given parameters.

    Parameters
    ----------
    root_path : Path
        The root path of the experiment.
    mag_threshold : Sequence[float,float]
        The magnitude thresholds
    feature_predictor_threshold : float
        The feature_predictor threshold.
    minimum_pixels: float
        The minimum pixel threshold.
    """
    datastore_path = resolve_datastore_path(root_path / "sim_acquisition")
    datastore = qi2labDataStore(datastore_path)
    merfish_bits = datastore.num_bits

    decoder = PixelDecoder(
        datastore=datastore, use_mask=False, merfish_bits=merfish_bits, verbose=0
    )

    # decoder._global_normalization_vectors()
    decoder.optimize_normalization_by_decoding(
        n_random_tiles=1,
        n_iterations=10,
        minimum_pixels=minimum_pixels,
        feature_predictor_threshold=feature_predictor_threshold,
        magnitude_threshold=mag_threshold,
    )

    decoder.decode_all_tiles(
        assign_to_cells=False,
        minimum_pixels=minimum_pixels,
        magnitude_threshold=mag_threshold,
        feature_predictor_threshold=feature_predictor_threshold,
        filter_method="blank_fraction",
    )


app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def sweep_decode_params(
    root_path: Path,
    feature_predictor_threshold_range: tuple[float] = (0.05, 0.4),
    feature_predictor_threshold_step: float = 0.1,
    mag_threshold_range: tuple[float] = (1.0, 2.0),
    mag_threshold_step: float = 0.1,
    minimum_pixels_range: tuple[float] = (3.0, 11.0),
    minimum_pixels_step: float = 2.0,
) -> None:
    """Sweep through decoding parameters and calculate F1 scores.

    Parameters
    ----------
    root_path : Path
        The root path of the experiment.
    feature_predictor_threshold_range : tuple, default (0.05, 0.4)
        The range of feature_predictor thresholds to sweep through.
    feature_predictor_threshold_step : float, default 0.1
        The step size for the feature_predictor threshold sweep
    mag_threshold_range : tuple, default [1.0,2.0]
        The range of minimum magnitude threshold to sweep through.
    mag_threshold_step : float, default 0.1
        The step size for the magnitude threshold.
    minimum_pixels_range: tuple[float], default = (3.0, 11.0)
        Range of minimum pixels to sweep through.
    minimum_pixels_step: float, default = 2.0
        The step size for the minimum pixel threshold.
    """
    mag_values = np.arange(
        mag_threshold_range[0],
        mag_threshold_range[1],
        mag_threshold_step,
        dtype=np.float32,
    ).tolist()

    feature_predictor_values = np.arange(
        feature_predictor_threshold_range[0],
        feature_predictor_threshold_range[1],
        feature_predictor_threshold_step,
        dtype=np.float32,
    ).tolist()

    pixels_values = np.arange(
        minimum_pixels_range[0],
        minimum_pixels_range[1],
        minimum_pixels_step,
        dtype=np.float32,
    ).tolist()

    results = {}
    save_path = root_path / "decode_params_results.json"
    gt_path = root_path / "GT_spots.csv"

    for pixels in pixels_values:
        for feature_predictor in feature_predictor_values:
            for mag in mag_values:
                params = {
                    "lr_fdr_target": 0.05,
                    "min_pixels": round(pixels, 2),
                    "mag_lower_thresh": round(mag, 2),
                    "mag_upper_thresh": 2.0,
                    "feature_predictor_threshold": round(feature_predictor, 2),
                }

                try:
                    print(
                        time_stamp(),
                        f"min pixels: {round(pixels, 2)}; feature_predictor threshold: {round(feature_predictor, 2)}; magnitude threshold: {round(mag, 2)}",
                    )
                    decode_pixels(
                        root_path=root_path,
                        mag_threshold=(round(mag, 2), 2.0),
                        feature_predictor_threshold=round(feature_predictor, 2),
                        minimum_pixels=round(pixels, 2),
                    )

                    result = calculate_F1(
                        root_path=root_path, gt_path=gt_path, search_radius=1.0
                    )
                except Exception as e:
                    result = {"error": str(e)}

                results[str(params)] = result

                with save_path.open(mode="w", encoding="utf-8") as file:
                    json.dump(results, file, indent=2)


def main() -> None:
    """Run the F1 sweep CLI."""
    app()


if __name__ == "__main__":
    main()
