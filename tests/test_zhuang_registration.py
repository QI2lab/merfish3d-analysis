import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import merfish3danalysis.DataRegistration as registration_module

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/zhuang_lab/02_register_and_deconvolve.py"
)


@pytest.mark.unit
@pytest.mark.parametrize("datastore_argument", [False, True])
def test_local_registration_preserves_existing_settings_and_state(
    tmp_path, monkeypatch, datastore_argument
):
    datastore_path = tmp_path / "qi2labdatastore"
    example = runpy.run_path(str(EXAMPLE_PATH))
    monkeypatch.setitem(
        example["local_register_data"].__globals__,
        "resolve_datastore_path",
        Mock(return_value=datastore_path),
    )
    datastore = SimpleNamespace(datastore_state={"Calibrations": True})
    factory = Mock()
    constructor = Mock(return_value=factory)
    loader = Mock(return_value=datastore)
    monkeypatch.setitem(
        example["local_register_data"].__globals__, "qi2labDataStore", loader
    )
    monkeypatch.setattr(registration_module, "DataRegistration", constructor)

    example["local_register_data"](datastore_path if datastore_argument else tmp_path)

    loader.assert_called_once_with(datastore_path)
    constructor.assert_called_once_with(
        datastore=datastore,
        decon_fiducial=False,
        decon_readout=False,
        perform_deformable_registration=False,
        overwrite_outputs=True,
        crop_yx_decon=2048,
    )
    factory.register_all_tiles.assert_called_once_with()
    assert datastore.datastore_state == {
        "Calibrations": True,
        "LocalRegistered": True,
    }


@pytest.mark.unit
@pytest.mark.parametrize("datastore_argument", [False, True])
@pytest.mark.parametrize("create_projection", [False, True])
def test_global_registration_uses_shared_workflow(
    tmp_path, monkeypatch, datastore_argument, create_projection
):
    datastore_path = tmp_path / "qi2labdatastore"
    example = runpy.run_path(str(EXAMPLE_PATH))
    monkeypatch.setitem(
        example["global_register_data"].__globals__,
        "resolve_datastore_path",
        Mock(return_value=datastore_path),
    )
    datastore = Mock()
    factory = Mock()
    constructor = Mock(return_value=factory)
    loader = Mock(return_value=datastore)
    monkeypatch.setitem(
        example["global_register_data"].__globals__, "qi2labDataStore", loader
    )
    monkeypatch.setattr(registration_module, "DataRegistration", constructor)

    example["global_register_data"](
        datastore_path if datastore_argument else tmp_path,
        create_max_proj_tiff=create_projection,
    )

    loader.assert_called_once_with(datastore_path, validate=False)
    constructor.assert_called_once_with(
        datastore=datastore,
        perform_deformable_registration=False,
        global_registration=True,
        global_registration_config=registration_module.GlobalRegistrationConfig(
            registration_binning_zyx=(1, 3, 3),
        ),
    )
    factory.global_register.assert_called_once_with(
        create_max_proj_tiff=create_projection
    )


@pytest.mark.unit
@pytest.mark.parametrize("function", ["local_register_data", "global_register_data"])
def test_registration_rejects_missing_datastore_before_creating_one(
    tmp_path, monkeypatch, function
):
    example = runpy.run_path(str(EXAMPLE_PATH))
    monkeypatch.setattr(Path, "is_file", Mock(return_value=False))
    loader = Mock()
    monkeypatch.setitem(example[function].__globals__, "qi2labDataStore", loader)
    with pytest.raises(FileNotFoundError, match="No qi2lab datastore"):
        example[function](tmp_path)
    loader.assert_not_called()
