"""Existing datastore discovery is independent of directory names."""

import pytest

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import resolve_datastore_path


@pytest.mark.integration
def test_existing_datastore_accepts_experiment_or_direct_path(image_datastore):
    datastore_path = image_datastore.datastore_path
    assert resolve_datastore_path(datastore_path.parent) == datastore_path
    assert resolve_datastore_path(datastore_path) == datastore_path
    assert not (datastore_path / "qi2labdatastore").exists()


@pytest.mark.integration
def test_custom_datastore_name_is_identified_by_metadata(tmp_path):
    datastore = qi2labDataStore(tmp_path / "custom_datastore")
    assert resolve_datastore_path(datastore.datastore_path) == datastore.datastore_path


@pytest.mark.integration
def test_missing_datastore_is_rejected_without_creating_directories(tmp_path):
    with pytest.raises(FileNotFoundError, match="datastore"):
        resolve_datastore_path(tmp_path / "missing")
    assert list(tmp_path.iterdir()) == []
