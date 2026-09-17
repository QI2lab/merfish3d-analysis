"""Command-line parsing and dispatch for the standalone example workflows."""

import runpy
from pathlib import Path
from unittest.mock import Mock, call

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
    zhuang = runpy.run_path(str(EXAMPLES / "04_pixel_decode.py"))
    for select in (
        qi2lab_decode._nearest_nyquist_multiple,
        simulation_decode._nearest_nyquist_multiple,
        zhuang["_nearest_nyquist_multiple"],
    ):
        assert select({3.0: 0.7, 5.0: 0.2}, measured) == expected


@pytest.mark.unit
@pytest.mark.parametrize("command", [qi2lab_decode, simulation_decode])
@pytest.mark.parametrize(
    "method, gross, fdr",
    [("lr", 0.1, 0.05), ("blank_fraction", 0.05, 0.1), ("unknown", 0.05, 0.05)],
)
def test_filter_control_must_match_selected_method(command, method, gross, fdr):
    with pytest.raises(typer.BadParameter):
        command._validate_filter_arguments(method, gross, fdr)


@pytest.mark.unit
@pytest.mark.parametrize("command", [qi2lab_decode, simulation_decode])
@pytest.mark.parametrize(
    "attributes, expected",
    [({"deconvolution": True}, True), ({"deconvolution": False}, False), ({}, False)],
)
def test_readout_deconvolution_uses_stored_flag(command, attributes, expected):
    datastore = Mock(tile_ids=["tile0000"], bit_ids=["bit001"])
    datastore.load_local_image_metadata.return_value = attributes
    assert command._readouts_are_deconvolved(datastore) is expected
    datastore.load_local_image_metadata.assert_called_once_with(
        "tile0000", bit="bit001", image_names=("decon_data",)
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "filename, operation, arguments, expected",
    [
        ("00a_test_image_orientation.py", "create_overview_image", [], 2),
        (
            "00a_test_image_orientation.py",
            "create_overview_image",
            ["--n-tiles", "5"],
            5,
        ),
        ("01_convert_to_qi2lab.py", "convert_data", [], None),
        ("03_cellpose_segmentation.py", "run_cellpose", [], None),
        ("04_pixel_decode.py", "decode_pixels", [], None),
        ("05_one_tile_F1.py", "decode_pixels", [], None),
        ("05_calculate_f1_score.py", "calculate_F1", [], None),
    ],
)
def test_zhuang_cli_preserves_arguments_and_defaults(
    tmp_path, monkeypatch, filename, operation, arguments, expected
):
    example = runpy.run_path(str(EXAMPLES / filename))
    process = Mock()
    monkeypatch.setitem(example["main"].__globals__, operation, process)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(example["app"], ["input data", *arguments])

    assert result.exit_code == 0, result.output
    root = tmp_path / "input data"
    if operation == "create_overview_image":
        process.assert_called_once_with(root, expected)
    elif operation == "convert_data":
        process.assert_called_once_with(
            root_path=root,
            channel_names=["alexa488", "cy5", "alexa750"],
            codebook_path=root / "additional_files/codebook.csv",
        )
    elif operation == "run_cellpose":
        process.assert_called_once_with(
            root,
            {
                "normalization": [0.5, 99.5],
                "flow_threshold": 0.4,
                "cellprob_threshold": 1.0,
                "diameter": 15,
            },
        )
    elif operation == "calculate_F1":
        process.assert_called_once_with(
            root,
            root
            / "mop/mouse_sample1_raw/zhuang_decoded_codewords/spots_mouse1sample1.csv",
            search_radius=3.0,
        )
    else:
        process.assert_called_once_with(root_path=root)


@pytest.mark.unit
@pytest.mark.parametrize("flags", [[], ["--local-only"], ["--global-only"]])
def test_registration_cli_selects_stages_in_order(tmp_path, monkeypatch, flags):
    example = runpy.run_path(str(EXAMPLES / "02_register_and_deconvolve.py"))
    process = Mock()
    monkeypatch.setitem(
        example["main"].__globals__, "local_register_data", process.local
    )
    monkeypatch.setitem(
        example["main"].__globals__, "global_register_data", process.global_register
    )

    result = runner.invoke(example["app"], [str(tmp_path), *flags])

    assert result.exit_code == 0, result.output
    expected = []
    if "--global-only" not in flags:
        expected.append(call.local(tmp_path))
    if "--local-only" not in flags:
        expected.append(call.global_register(tmp_path, create_max_proj_tiff=True))
    assert process.mock_calls == expected


@pytest.mark.unit
def test_registration_cli_rejects_conflicting_stages_before_processing(monkeypatch):
    example = runpy.run_path(str(EXAMPLES / "02_register_and_deconvolve.py"))
    local = Mock()
    global_register = Mock()
    monkeypatch.setitem(example["main"].__globals__, "local_register_data", local)
    monkeypatch.setitem(
        example["main"].__globals__, "global_register_data", global_register
    )

    result = runner.invoke(example["app"], ["data", "--local-only", "--global-only"])

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output
    local.assert_not_called()
    global_register.assert_not_called()


@pytest.mark.unit
def test_orientation_cli_rejects_noninteger_tile_count(monkeypatch):
    example = runpy.run_path(str(EXAMPLES / "00a_test_image_orientation.py"))
    process = Mock()
    monkeypatch.setitem(example["main"].__globals__, "create_overview_image", process)
    result = runner.invoke(example["app"], ["data", "--n-tiles", "invalid"])
    assert result.exit_code == 2
    process.assert_not_called()


@pytest.mark.unit
def test_fused_viewer_cli_passes_a_path(tmp_path, monkeypatch):
    from merfish3danalysis.viewer import fused

    process = Mock()

    def view_fused(root_path: Path) -> None:
        process(root_path)

    monkeypatch.setattr(fused, "view_fused_channels", view_fused)
    example = runpy.run_path(
        str(ROOT / "src/merfish3danalysis/cli/qi2lab_microscopes/05_view_fused.py")
    )
    result = runner.invoke(example["app"], [str(tmp_path)])
    assert result.exit_code == 0, result.output
    process.assert_called_once_with(tmp_path)
