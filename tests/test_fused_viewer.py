"""The fused viewer retains all saved channels and their grids without eager reads."""

import dask.array as da
import numpy as np
import pytest

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.viewer.fused import load_fused_channels


@pytest.mark.integration
def test_fused_channels_remain_lazy_and_distinct_grids_are_preserved(tmp_path):
    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    inputs = [
        ("a", np.ones((2, 3, 4), dtype=np.uint16), [1.5, 0.2, 0.2]),
        ("b", np.full((2, 3, 4), 2, dtype=np.uint16), [1.5, 0.2, 0.2]),
        ("c", np.full((1, 2, 2, 3, 4), 3, dtype=np.uint16), [1.5, 0.4, 0.4]),
    ]
    for name, pixels, spacing in inputs:
        qi2labDataStore._save_to_zarr_array(
            pixels,
            datastore.fused_image_path(name),
            ome_scale=spacing,
            ome_translation=[12, 100, 200],
        )
    stacks = load_fused_channels(tmp_path)
    assert len(stacks) == 2
    assert all(isinstance(stack.data, da.Array) for stack in stacks)
    assert list(stacks[0].c.values) == ["a: 0", "b: 0"]
    np.testing.assert_array_equal(
        stacks[0].data.compute(), np.stack([inputs[0][1], inputs[1][1]])
    )
    np.testing.assert_array_equal(stacks[1].data.compute(), inputs[2][1][0])
    np.testing.assert_allclose(stacks[0].x, [200, 200.2, 200.4, 200.6])
    np.testing.assert_allclose(stacks[1].x, [200, 200.4, 200.8, 201.2])
    assert stacks[1].dims == ("c", "z", "y", "x")


@pytest.mark.integration
def test_fused_viewer_displays_distinct_channel_pixels_and_physical_coordinates(
    tmp_path, monkeypatch
):
    import ndv
    from qtpy import QtWidgets

    from merfish3danalysis.viewer.fused import view_fused_channels

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    pixels = np.arange(120, dtype=np.uint16).reshape(1, 3, 2, 4, 5)
    qi2labDataStore._save_to_zarr_array(
        pixels,
        datastore.fused_image_path("full_dataset"),
        ome_scale=[1.5, 0.2, 0.4],
        ome_translation=[12, 100, 200],
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    created = []
    constructor = ndv.ArrayViewer

    def record(*args, **kwargs):
        viewer = constructor(*args, **kwargs)
        created.append(viewer)
        return viewer

    def inspect():
        app.processEvents()
        assert len(created) == 1
        data = created[0].data
        assert data.shape == (3, 2, 4, 5)
        np.testing.assert_array_equal(data, pixels[0])
        np.testing.assert_allclose(data.z, [12, 13.5])
        np.testing.assert_allclose(data.y, [100, 100.2, 100.4, 100.6])
        np.testing.assert_allclose(data.x, [200, 200.4, 200.8, 201.2, 201.6])
        assert list(data.c.values) == [
            "full_dataset: 0",
            "full_dataset: 1",
            "full_dataset: 2",
        ]
        assert created[0].display_model.channel_mode.value == "composite"

    monkeypatch.setattr(ndv, "ArrayViewer", record)
    monkeypatch.setattr(app, "exec", inspect)
    view_fused_channels(datastore.datastore_path)
