import runpy
from pathlib import Path

import pytest

import merfish3danalysis.DataRegistration as registration_module
from merfish3danalysis.qi2labDataStore import qi2labDataStore

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/zhuang_lab/02_register_and_deconvolve.py"
)


@pytest.mark.unit
@pytest.mark.parametrize("datastore_argument", [False, True])
def test_local_registration_preserves_existing_settings_and_state(
    image_datastore, monkeypatch, datastore_argument
):
    datastore_path = image_datastore.datastore_path
    example = runpy.run_path(str(EXAMPLE_PATH))
    registrations = []
    monkeypatch.setattr(
        registration_module.DataRegistration,
        "register_all_tiles",
        lambda registration: registrations.append(registration),
    )

    example["local_register_data"](
        datastore_path if datastore_argument else datastore_path.parent
    )

    assert len(registrations) == 1
    registration = registrations[0]
    assert registration.datastore.datastore_path == datastore_path
    assert registration._decon_fiducial is False
    assert registration._decon_readout is False
    assert registration._perform_deformable_registration is False
    assert registration._overwrite_outputs is True
    assert registration._crop_yx_decon == 2048
    reopened = qi2labDataStore(datastore_path, validate=False)
    assert reopened.datastore_state["Calibrations"]
    assert reopened.datastore_state["LocalRegistered"]


@pytest.mark.unit
@pytest.mark.parametrize("datastore_argument", [False, True])
@pytest.mark.parametrize("create_projection", [False, True])
def test_global_registration_uses_shared_workflow(
    image_datastore, monkeypatch, datastore_argument, create_projection
):
    datastore_path = image_datastore.datastore_path
    example = runpy.run_path(str(EXAMPLE_PATH))
    calls = []
    monkeypatch.setattr(
        registration_module.DataRegistration,
        "global_register",
        lambda registration, **kwargs: calls.append((registration, kwargs)),
    )

    example["global_register_data"](
        datastore_path if datastore_argument else datastore_path.parent,
        create_max_proj_tiff=create_projection,
    )

    assert len(calls) == 1
    registration, options = calls[0]
    assert registration.datastore.datastore_path == datastore_path
    assert registration._perform_deformable_registration is False
    assert registration._global_registration is True
    assert registration._global_registration_config == (
        registration_module.GlobalRegistrationConfig(registration_binning_zyx=(1, 3, 3))
    )
    assert options == {"create_max_proj_tiff": create_projection}


@pytest.mark.unit
@pytest.mark.parametrize("function", ["local_register_data", "global_register_data"])
def test_registration_rejects_missing_datastore_before_creating_one(tmp_path, function):
    example = runpy.run_path(str(EXAMPLE_PATH))
    with pytest.raises(FileNotFoundError, match="No qi2lab datastore"):
        example[function](tmp_path)
    assert not (tmp_path / "qi2labdatastore").exists()
