"""Viewer entry point opens acquisition metadata and images from real stores."""

import numpy as np
import pytest

from merfish3danalysis.viewer.datastore import load_datastore_for_viewer


@pytest.mark.integration
@pytest.mark.parametrize("argument", ["experiment", "direct", "custom"])
def test_viewer_opens_store_and_populates_options(image_datastore, tmp_path, argument):
    datastore = image_datastore
    image = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
    datastore.save_local_corrected_image(image, tile=0, round=0)
    store_path = datastore.datastore_path
    if argument == "custom":
        custom_path = tmp_path / "custom_store"
        store_path.rename(custom_path)
        store_path = custom_path
    root_path = tmp_path if argument == "experiment" else store_path

    result = load_datastore_for_viewer(root_path)

    assert result.datastore_path == store_path
    assert result.options.tile_ids == ("tile0000", "tile0001")
    assert result.options.round_ids == ("round001", "round002")
    assert result.options.bit_ids == ("bit001", "bit002", "bit003", "bit004")
    assert result.options.transcript_gene_to_bits == {
        "gene": ["bit001", "bit002", "bit003", "bit004"]
    }
    assert result.options.components["Calibrations"]
    np.testing.assert_array_equal(
        result.datastore.load_local_corrected_image(
            tile=0, round=0, return_future=False
        ),
        image,
    )
    assert not (store_path / "qi2labdatastore").exists()


@pytest.mark.unit
def test_viewer_rejects_missing_store_without_creating_directories(tmp_path):
    with pytest.raises(FileNotFoundError, match="datastore"):
        load_datastore_for_viewer(tmp_path / "missing")
    assert list(tmp_path.iterdir()) == []
