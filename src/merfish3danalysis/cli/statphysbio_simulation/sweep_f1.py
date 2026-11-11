"""
Sweep through decoding parameters and calculate F1-score using known ground truth.

Contributions: Shepherd.
"""

import json
from pathlib import Path

import numpy as np
import typer

from merfish3danalysis.cli.statphysbio_simulation.calculate_F1 import calculate_F1
from merfish3danalysis.cli.statphysbio_simulation.pixeldecode import decode_pixels
from merfish3danalysis.utils.dataio import time_stamp

app = typer.Typer()
app.pretty_exceptions_enable = False


@app.command()
def sweep_decode_params(
    root_path: Path,
    *,
    mag_threshold_range: tuple[float, float] = (1.0, 2.0),
    mag_threshold_step: float = 0.1,
    minimum_pixels_range: tuple[float, float] = (3.0, 11.0),
    minimum_pixels_step: float = 2.0,
) -> None:
    """Sweep through decoding parameters and calculate F1 scores.

    Parameters
    ----------
    root_path : Path
        The root path of the experiment.
    mag_threshold_range : tuple, default [1.0,2.0]
        The range of minimum magnitude threshold to sweep through.
    mag_threshold_step : float, default 0.1
        The step size for the magnitude threshold.
    minimum_pixels_range : tuple[float, float], default = (3.0, 11.0)
        Range of minimum pixels to sweep through.
    minimum_pixels_step : float, default = 2.0
        The step size for the minimum pixel threshold.
    """
    mag_values = np.arange(
        mag_threshold_range[0],
        mag_threshold_range[1],
        mag_threshold_step,
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
        for mag in mag_values:
            params = {
                "target_gross_misid_rate": 0.05,
                "min_pixels": round(pixels, 2),
                "mag_lower_thresh": round(mag, 2),
                "mag_upper_thresh": 2.0,
            }

            try:
                print(
                    time_stamp(),
                    f"min pixels: {round(pixels, 2)}; magnitude threshold: {round(mag, 2)}",
                )
                decode_pixels(
                    root_path=root_path / "sim_acquisition",
                    magnitude_threshold=(round(mag, 2), 2.0),
                    minimum_pixels_per_RNA=round(pixels, 2),
                    normalization_iterations=10,
                    filter_method="blank_fraction",
                    target_gross_misid_rate=params["target_gross_misid_rate"],
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
