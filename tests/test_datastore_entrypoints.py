"""Processing entry points resolve existing stores before constructing a datastore."""

import importlib
from unittest.mock import Mock

import pytest

ENTRY_POINTS = [
    ("cli.qi2lab_microscopes.preprocess", "local_register_data"),
    ("cli.qi2lab_microscopes.segment_fiducial", "run_cellpose"),
    ("cli.qi2lab_microscopes.pixeldecode", "decode_pixels"),
    ("cli.statphysbio_simulation.register_and_deconvolve", "local_register_data"),
    ("cli.statphysbio_simulation.register_and_deconvolve", "global_register_data"),
    ("cli.statphysbio_simulation.pixeldecode", "decode_pixels"),
    ("viewer.datastore", "load_datastore_for_viewer"),
]


class DatastoreSelected(Exception):
    """Stop at construction before invoking scientific processing or the GUI."""


@pytest.mark.integration
@pytest.mark.parametrize("module_name, function_name", ENTRY_POINTS)
@pytest.mark.parametrize("argument", ["experiment", "direct", "custom", "missing"])
def test_entry_point_selects_existing_store_without_creating_nested_output(
    tmp_path, monkeypatch, module_name, function_name, argument
):
    module = importlib.import_module("merfish3danalysis." + module_name)
    constructor = Mock(side_effect=DatastoreSelected)
    monkeypatch.setattr(module, "qi2labDataStore", constructor)
    store_path = tmp_path / (
        "custom_store" if argument == "custom" else "qi2labdatastore"
    )
    if argument != "missing":
        store_path.mkdir()
        (store_path / "datastore_state.json").write_text("{}")
    root_path = tmp_path if argument in ["experiment", "missing"] else store_path
    function = getattr(module, function_name)
    if argument == "missing":
        with pytest.raises(FileNotFoundError, match="datastore"):
            function(root_path)
        constructor.assert_not_called()
        assert list(tmp_path.iterdir()) == []
    else:
        with pytest.raises(DatastoreSelected):
            function(root_path)
        assert constructor.call_args.args[0] == store_path
        assert not (store_path / "qi2labdatastore").exists()
