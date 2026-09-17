"""Diagnostic image channels retain their pixels, labels, and coordinate scale."""

import numpy as np
import pytest

from merfish3danalysis.viewer.diagnostics import diagnostic_channel_data


@pytest.mark.unit
def test_diagnostic_channels_keep_bit_axis_and_physical_spacing():
    bits = np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    labels = np.ones((3, 4, 5))
    data = diagnostic_channel_data(
        {"bits": bits, "decoded": labels}, [1.5, 0.108, 0.108]
    )
    assert data.dims == ("c", "z", "y", "x")
    assert list(data.c.values) == ["bits 0", "bits 1", "decoded"]
    np.testing.assert_array_equal(data[:2], bits)
    np.testing.assert_array_equal(data[2], labels)
    np.testing.assert_allclose(data.z, [0, 1.5, 3])
    np.testing.assert_allclose(data.x, np.arange(5) * 0.108)


@pytest.mark.unit
def test_diagnostic_images_reject_incompatible_spatial_shapes():
    with pytest.raises(ValueError, match="ZYX shape"):
        diagnostic_channel_data(
            {"image": np.zeros((2, 3, 4)), "labels": np.zeros((3, 3, 4))}, [1, 1, 1]
        )


@pytest.mark.integration
def test_ndv_diagnostics_displays_channels_and_point_overlays(monkeypatch):
    import ndv
    from qtpy import QtWidgets

    from merfish3danalysis.viewer.diagnostics import show_diagnostic_images

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    created = []
    viewer_class = ndv.ArrayViewer

    def record_viewer(*args, **kwargs):
        viewer = viewer_class(*args, **kwargs)
        created.append(viewer)
        return viewer

    def inspect_viewer():
        app.processEvents()
        viewer = created[0]
        assert viewer.data.shape == (2, 2, 3, 4)
        view = viewer._canvas._view
        from vispy.scene.visuals import Markers

        markers = [child for child in view.scene.children if isinstance(child, Markers)]
        assert len(markers) == 1
        np.testing.assert_allclose(markers[0]._data["a_position"][0, :2], [0.8, 0.2])
        viewer.display_model.current_index["z"] = 1
        app.processEvents()
        np.testing.assert_allclose(markers[0]._data["a_position"][0, :2], [1.2, 0.4])

    monkeypatch.setattr(ndv, "ArrayViewer", record_viewer)
    monkeypatch.setattr(app, "exec", inspect_viewer)
    show_diagnostic_images(
        {"image": np.zeros((2, 3, 4)), "labels": np.ones((2, 3, 4))},
        [1.5, 0.2, 0.4],
        points=(("spots", np.array([[0, 1, 2], [1, 2, 3]]), "cyan", "o"),),
    )
