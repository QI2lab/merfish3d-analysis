"""Command-line parsing and dispatch for the standalone example workflows."""

import runpy
from pathlib import Path

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from merfish3danalysis.cli.qi2lab_microscopes import pixeldecode as qi2lab_decode
from merfish3danalysis.cli.statphysbio_simulation import (
    pixeldecode as simulation_decode,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/zhuang_lab"
runner = CliRunner()


@pytest.mark.unit
@pytest.mark.parametrize("measured, expected", [(2.8, 3), (4, 3), (5.1, 5)])
def test_nearest_sampling_preset_preserves_first_key_on_ties(measured, expected):
    assert (
        qi2lab_decode._nearest_nyquist_multiple({3.0: 0.7, 5.0: 0.2}, measured)
        == expected
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "method, gross, fdr",
    [("lr", 0.1, 0.05), ("blank_fraction", 0.05, 0.1), ("unknown", 0.05, 0.05)],
)
def test_filter_control_must_match_selected_method(method, gross, fdr):
    with pytest.raises(typer.BadParameter):
        qi2lab_decode._validate_filter_arguments(method, gross, fdr)


@pytest.mark.unit
@pytest.mark.parametrize(
    "select_threshold",
    [
        qi2lab_decode._default_qi2lab_feature_predictor_threshold,
        simulation_decode._default_simulation_feature_predictor_threshold,
    ],
)
@pytest.mark.parametrize("deconvolved", [False, True])
def test_readout_deconvolution_sets_default_threshold(
    select_threshold, deconvolved, image_datastore
):
    image_datastore.microscope_type = "2D"
    image_datastore.voxel_size_zyx_um = [1.5, 0.2, 0.4]
    image = np.ones((4, 8, 8), dtype=np.uint16)
    image_datastore.save_local_corrected_image(image, tile=0, bit=0)
    if deconvolved:
        image_datastore.save_local_deconvolved_readout_image(image, tile=0, bit=0)
    assert select_threshold(image_datastore) == (0.2 if deconvolved else 0.5)


@pytest.mark.unit
def test_registration_cli_rejects_conflicting_stages_before_processing():
    example = runpy.run_path(str(EXAMPLES / "02_register_and_deconvolve.py"))
    result = runner.invoke(example["app"], ["data", "--local-only", "--global-only"])

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


@pytest.mark.unit
def test_orientation_cli_rejects_noninteger_tile_count():
    example = runpy.run_path(str(EXAMPLES / "00a_test_image_orientation.py"))
    result = runner.invoke(example["app"], ["data", "--n-tiles", "invalid"])
    assert result.exit_code == 2
